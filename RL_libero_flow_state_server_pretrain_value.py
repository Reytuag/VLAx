"""
RL training script that ONLY updates the value network on state-based trajectories from LIBERO,
using a VLM for conditioning. The flow model is frozen (used for action sampling only, never updated).
It collects trajectories of observations, actions, rewards, and KV caches for analysis and value training.
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import gc
import sys
import json
import time
import functools
import numpy as np
import jax
import jax.numpy as jnp
from datetime import datetime
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
try:
    from gemma import gm
    from gemma import peft
    from sampler import Sampler
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False
    print("Warning: gemma module not available. Will use dummy cache.")

from networks.flow_network_state_rope import TransformerFlow
from libero_misc.libero_comms import (
    start_libero_worker, stop_libero_worker,
    libero_reset, libero_step, libero_step_batch,
)
import libero_misc.libero_comms as libero_comms
from utils import create_cache_mask, extract_cache_from_layers, linear_lr_schedule, cosine_lr_schedule
import wandb
from networks.value_network_state_rope import ValueNetworkStateRope

gemma_path="/home/reytuag/VLA/gemma-3-flax-gemma3-4b-it-v1"


# ============================================================================
# Flow Model Utilities
# ============================================================================

def load_flow_model(checkpoint_path, action_shape=7, action_horizon=16, num_layers=3):
    model = TransformerFlow(
        num_layers=num_layers, num_heads=4, qkv_features=1024, out_features=512,
        input_size=action_shape, gating=True, gating_bias=2.,
        norm_type="rmsnorm", post_attention_norm=True, post_mlp_norm=True,use_state=True
    )
    params = jnp.load(checkpoint_path, allow_pickle=True).item()
    apply_fn = jax.jit(model.apply)
    return model, params, apply_fn


@functools.partial(jax.jit, static_argnums=(1, 6, 7, 8, 10))
def sample_actions(params, apply_fn, cache_k, cache_v, cache_mask, robot_state,
                   action_shape=7, action_horizon=16, num_steps=10, rng_key=None,
                   zero_sampling=False, noise_scale=1.0):
    """Sample actions from the flow model using iterative refinement.
    
    Args:
        zero_sampling: If True, initialize from epsilon=0 instead of random noise
                       (see FPO++ paper Section III-D). Recommended for evaluation
                       and deployment for improved performance.
        noise_scale: Multiplicative factor on the initial noise. Controls exploration:
                     1.0 = standard, <1.0 = less exploration, >1.0 = more exploration.
                     Has no effect when zero_sampling=True.
    """
    batch_size = cache_k.shape[0]
    dt = 1.0 / num_steps
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    x = jnp.where(
        zero_sampling,
        jnp.zeros((batch_size, action_horizon, action_shape)),
        noise_scale * jax.random.normal(rng_key, (batch_size, action_horizon, action_shape)),
    )

    def iter_fn(carry, _):
        params, x, cache_mask, cache_k, cache_v, t, dt, state = carry
        output = apply_fn(params, x, cache_mask, cache_k, cache_v, t, state)
        x = x + output * dt
        return (params, x, cache_mask, cache_k, cache_v, t - dt, dt, state), None

    time_steps = jnp.ones((batch_size,))
    (_, x_final, _, _, _, _, _, _), _ = jax.lax.scan(
        iter_fn,
        (params, x, cache_mask, cache_k, cache_v, time_steps, dt, robot_state),
        jnp.arange(num_steps), length=num_steps
    )
    return x_final


# ============================================================================
# Closed-Loop Rollout
# ============================================================================

def run_rollout(params_flow, apply_fn, params_value, value_fn, sampler, config, rng,
                action_shape=7, action_horizon=16,
                num_diffusion_steps=10, num_episodes=4, max_nb_replan=6,
                steps_per_replan=16, bin_centers=None):
    """Run a closed-loop rollout over multiple episodes. Returns a trajectory dictionary.
    
    Args:
        num_episodes: Number of episodes to run.
        max_nb_replan: Max replans per episode before forcing a reset.
        steps_per_replan: Number of action steps executed per replan.
    """
    task_instruction = libero_comms.libero_task_description
    agentview, eye_in_hand, robot_state = libero_reset()

    trajectory = {
        'agentview_observations': [],
        'eye_in_hand_observations': [],
        'robot_states': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'mask': [],  # True = real step, False = dummy (post-done padding)
        'infos': [],
        'replans': [],
        'values': [],
        'kv_caches': [],  # Store KV cache (k and v) per replan
    }

    total_steps = 0
    total_replans = 0
    episode_successes = []  # Track per-episode success (1.0 = success, 0.0 = failure)
    episode_lengths = []  # Track per-episode step counts

    for episode_idx in range(num_episodes):
        print(f"\n{'='*40} Episode {episode_idx + 1}/{num_episodes} {'='*40}")
        cumulative_episode_reward = 0.0
        episode_success = False
        episode_step_count = 0

        for replan_in_ep in range(max_nb_replan):
            total_replans += 1

            # --- Get KV cache from VLM ---
            if sampler is not None:
                rng, _rng = jax.random.split(rng)
                images_stacked = jnp.array(np.stack([agentview, eye_in_hand], axis=0)[None, ...])
                task_prompt = (f'You are a robotic arm. Task: {task_instruction} '
                              f'Agent view: <start_of_image> Eye in hand view: <start_of_image>. '
                              f'Give detailed subtasks to complete the task.')

                if config["USE_SAMPLE_WITH_STATE"]:
                    out = sampler.sample(task_prompt, images=images_stacked,
                                         max_new_tokens=100, rng=_rng, return_state=True)
                    cache = out.state.cache
                    print(f"    VLM: {(out.text[0] if isinstance(out.text, (list, tuple)) else str(out.text))[:80]}...")
                else:
                    cache = sampler.get_cache_prompt(task_prompt, images=images_stacked, rng=_rng)

                cache_k = extract_cache_from_layers(cache, config["CACHE_LAYERS"], "k")
                cache_v = extract_cache_from_layers(cache, config["CACHE_LAYERS"], "v")
                first_layer = config["CACHE_LAYERS"][0]
                cache_mask = create_cache_mask(
                    cache[first_layer]["end_index"],
                    cache_k.shape[2], action_horizon + 1, cache_k.shape[3]
                )
            else:
                cache_k = jnp.ones((1, len(config["CACHE_LAYERS"]), 512, 8, 64))
                cache_v = jnp.ones((1, len(config["CACHE_LAYERS"]), 512, 8, 64))
                cache_mask = jnp.ones((1, 8, action_horizon + 1, 512))

            # --- Sample actions (flow model is frozen, just used for inference) ---
            rng, _rng = jax.random.split(rng)
            robot_state_jax = jnp.array(robot_state[None, ...])
            action_sequence = sample_actions(
                params_flow, apply_fn, cache_k, cache_v, cache_mask, robot_state_jax,
                action_shape=action_shape, action_horizon=action_horizon,
                num_steps=num_diffusion_steps, rng_key=_rng,
                zero_sampling=config.get("ZERO_SAMPLING_ROLLOUT", False),
                noise_scale=config.get("NOISE_SCALE", 1.0)
            )
            cache_mask_value = create_cache_mask(
                    cache[first_layer]["end_index"],
                    cache_k.shape[2], 1, cache_k.shape[3]
                )
            value_logits = value_fn(params_value, cache_mask_value, cache_k, cache_v, robot_state_jax[:,None])
            value_probs = jax.nn.softmax(value_logits, axis=-1)  # (1, 1, num_bins)
            value = jnp.sum(value_probs * bin_centers, axis=-1)  # (1, 1) weighted average
            trajectory['values'].append(value[0,0])
            trajectory['kv_caches'].append({
                'cache_k': np.array(cache_k[0]),
                'cache_v': np.array(cache_v[0]),
                'cache_mask': np.array(cache_mask[0]),
                'cache_mask_value': np.array(cache_mask_value[0]),
            })

            # Save images and states at the start of each replan (before actions)
            trajectory['agentview_observations'].append(agentview)
            trajectory['eye_in_hand_observations'].append(eye_in_hand)
            trajectory['robot_states'].append(robot_state)

            # --- Execute actions (batched IPC) ---
            t_env_start = time.time()
            # Build the list of actions to send
            num_actions_available = min(steps_per_replan, action_sequence.shape[1])
            actions_to_send = np.clip(np.array(action_sequence[0, :num_actions_available, :]), -1, 1)
            # Pad with zeros if steps_per_replan > action_horizon
            if num_actions_available < steps_per_replan:
                pad = np.zeros((steps_per_replan - num_actions_available, action_shape))
                actions_to_send = np.concatenate([actions_to_send, pad], axis=0)

            agentview, eye_in_hand, robot_state, rewards_batch, dones_batch, info, steps_executed = \
                libero_step_batch(actions_to_send)

            a = max_nb_replan * 16.
            done = False
            for step_i in range(steps_executed):
                raw_reward = rewards_batch[step_i]
                step_done = dones_batch[step_i]
                if step_done and raw_reward > 0:
                    episode_success = True
                reward = ((raw_reward > 0.) * a ) / a
                trajectory['actions'].append(actions_to_send[step_i].tolist())
                trajectory['rewards'].append(reward)
                trajectory['dones'].append(step_done)
                trajectory['mask'].append(True)
                trajectory['infos'].append({})
                total_steps += 1
                episode_step_count += 1
                if step_done:
                    done = True
                    action_idx = step_i
                    break
            else:
                action_idx = steps_executed - 1

            if done:
                print(f"[info] Episode {episode_idx + 1} finished at step {total_steps} reward={reward:.4f}")
                remaining = steps_per_replan - steps_executed
                for _ in range(remaining):
                    trajectory['actions'].append([0.] * action_shape)
                    trajectory['rewards'].append(0.0)
                    trajectory['dones'].append(True)
                    trajectory['mask'].append(False)
                    trajectory['infos'].append({})
                    total_steps += 1
                print(f"    Filled {remaining} remaining steps with dummy values")

            t_env_end = time.time()
            t_env_elapsed = t_env_end - t_env_start

            # Compute reward for this replan (sum of rewards over steps in this replan)
            replan_rewards = trajectory['rewards'][-steps_per_replan:]
            replan_reward = sum(replan_rewards)
            replan_mask = trajectory['mask'][-steps_per_replan:]
            cumulative_episode_reward += replan_reward

            # Clipped action sequence for this replan
            clipped_action_sequence = np.clip(np.array(action_sequence[0]), -1, 1)
            if(replan_in_ep==max_nb_replan-1):
                done=True
            
            trajectory['replans'].append({
                'replan_idx': total_replans - 1,
                'episode_idx': episode_idx,
                'replan_in_episode': replan_in_ep,
                'steps_executed': action_idx + 1,
                'total_steps': total_steps,
                'replan_reward': replan_reward,
                'cumulative_episode_reward': cumulative_episode_reward,
                'action_sequence': clipped_action_sequence,
                'replan_mask': list(replan_mask),
                "done": done,
            })

            if done:
                break

        # Reset environment for next episode
        episode_successes.append(float(episode_success))
        episode_lengths.append(episode_step_count)
        if episode_idx < num_episodes - 1:
            agentview, eye_in_hand, robot_state = libero_reset()
            print(f"    Environment reset for next episode")

    rewards = np.array(trajectory['rewards'])
    print(f"\n  Total steps: {total_steps}, Total replans: {total_replans}, "
          f"Total reward: {rewards.sum():.4f}, Mean reward: {rewards.mean():.4f}, "
          f"Success rate: {np.mean(episode_successes):.3f} ({int(sum(episode_successes))}/{num_episodes})")
    trajectory['episode_successes'] = episode_successes
    trajectory['episode_lengths'] = episode_lengths
    return trajectory


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("LIBERO State-Based Flow Model — VALUE-ONLY Training")
    print("=" * 60)

    # --- Parse args ---
    checkpoint_dir = None
    task_id = 1
    cli_overrides = {}
    rng = jax.random.PRNGKey(0)
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--task-id" and i + 1 < len(sys.argv):
            task_id = int(sys.argv[i + 1]); i += 2
        elif arg == "--diffusion-steps" and i + 1 < len(sys.argv):
            cli_overrides["num_diffusion_steps"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--lr-value" and i + 1 < len(sys.argv):
            cli_overrides["LR_VALUE"] = float(sys.argv[i + 1]); i += 2
        elif arg == "--epochs" and i + 1 < len(sys.argv):
            cli_overrides["NUM_EPOCHS"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--batch-size" and i + 1 < len(sys.argv):
            cli_overrides["BATCH_SIZE"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--grad-accum" and i + 1 < len(sys.argv):
            cli_overrides["GRAD_ACCUM_STEPS"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--gamma" and i + 1 < len(sys.argv):
            cli_overrides["GAMMA"] = float(sys.argv[i + 1]); i += 2
        elif arg == "--gae-lambda" and i + 1 < len(sys.argv):
            cli_overrides["GAE_LAMBDA"] = float(sys.argv[i + 1]); i += 2
        elif arg == "--advantage-method" and i + 1 < len(sys.argv):
            cli_overrides["ADVANTAGE_METHOD"] = sys.argv[i + 1]; i += 2
        elif arg == "--num-episodes" and i + 1 < len(sys.argv):
            cli_overrides["NUM_EPISODES"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--max-replan" and i + 1 < len(sys.argv):
            cli_overrides["MAX_NB_REPLAN"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--train-iters" and i + 1 < len(sys.argv):
            cli_overrides["TOTAL_TRAIN_ITERS"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--anneal-lr-value":
            cli_overrides["ANNEAL_LR_VALUE"] = True; i += 1
        elif arg == "--noise-scale" and i + 1 < len(sys.argv):
            cli_overrides["NOISE_SCALE"] = float(sys.argv[i + 1]); i += 2
        elif not arg.startswith("--"):
            checkpoint_dir = arg; i += 1
        else:
            i += 1

    # --- Config ---
    loaded_config = None
    if checkpoint_dir:
        config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                loaded_config = json.load(f)
            print(f"[info] Loaded config from {config_path}")

    default_config = {
        # --- Environment / model architecture ---
        "action_shape": 7,
        "action_horizon": 16,
        "num_diffusion_steps": 20,
        "CACHE_LAYERS": ["layer_9", "layer_26", "layer_31"],
        "USE_SAMPLE_WITH_STATE": True,
        "task_suite_name": "libero_spatial",

        # --- Rollout ---
        "NUM_EPISODES": 40,
        "MAX_NB_REPLAN": 16,
        "STEPS_PER_REPLAN": 16,

        # --- Training loop ---
        "TOTAL_TRAIN_ITERS": 2,
        "NUM_EPOCHS": 20,
        "BATCH_SIZE": 8,
        "GRAD_ACCUM_STEPS": 8,

        # --- Optimiser (value network) ---
        "LR_VALUE": 6e-5,
        "ANNEAL_LR_VALUE": True,
        "LR_SCHEDULE_TYPE_VALUE": "cosine",   # "cosine" | "linear"
        "MAX_GRAD_NORM_VALUE": 0.5,

        # --- Advantage ---
        "GAMMA": 0.87,
        "GAE_LAMBDA": 0.95,
        "ADVANTAGE_METHOD": "mc",     # "mc" (Monte Carlo returns) | "gae" (Generalized Advantage Estimation)

        # --- Zero-sampling (paper Section III-D) ---
        "ZERO_SAMPLING_ROLLOUT": False,
        "NOISE_SCALE": 0.05,

        # --- Distributional value (categorical) ---
        "NUM_BINS": 41,
        "V_MIN": 0.0,
        "V_MAX": 1.0,

        # --- Misc ---
        "TOTAL_TRAIN_STEPS": 4,  # for LR schedule denominator
    }

    # Only pick specific keys from the loaded checkpoint config
    _KEYS_FROM_CHECKPOINT = {"CACHE_LAYERS", "action_horizon", "action_shape", "USE_SAMPLE_WITH_STATE"}
    config = dict(default_config)
    if loaded_config:
        for k in _KEYS_FROM_CHECKPOINT:
            if k in loaded_config:
                config[k] = loaded_config[k]

    # Apply CLI overrides on top of config
    config.update(cli_overrides)

    action_shape = config["action_shape"]
    action_horizon = config["action_horizon"]
    num_diffusion_steps = config["num_diffusion_steps"]
    num_cache_layers = len(config["CACHE_LAYERS"])
    
    wandb.init(
        entity="gautier-hamon",
        project='VLA_LIBERO_RL',
        config=config,
        tags=["value-only"],
    )

    # --- Start LIBERO worker ---
    try:
        start_libero_worker(task_id=task_id,
                            python_bin="/home/reytuag/miniconda3/envs/libero_env/bin/python")
        msg = libero_comms._read_json_from_worker()
        libero_comms.libero_task_name = msg.get("task_name", "Unknown")
        libero_comms.libero_task_description = msg.get("task_description", "")
        print(f"[info] Task: {libero_comms.libero_task_name} — {libero_comms.libero_task_description}")

        # Verify reset
        libero_comms.libero_process.stdin.write(json.dumps({"cmd": "reset"}) + "\n")
        libero_comms.libero_process.stdin.flush()
        obs = libero_comms._read_json_from_worker()
        print(f"[info] Observation keys: {list(obs.keys())}")
    except Exception as e:
        print(f"[error] Failed to start worker: {e}")
        return

    # --- Load flow model (orbax checkpoint preferred, .npy fallback) ---
    # Flow model is FROZEN — loaded once, never updated.
    model_flow = TransformerFlow(
        num_layers=num_cache_layers, num_heads=4, qkv_features=1024, out_features=512,
        input_size=action_shape, gating=True, gating_bias=2.,
        norm_type="rmsnorm", post_attention_norm=True, post_mlp_norm=True
    )
    apply_fn = jax.jit(model_flow.apply)

    params_flow = None

    # 1) Try orbax checkpoint inside checkpoint_dir/checkpoints
    orbax_ckpt_dir = None
    if checkpoint_dir:
        candidate_dir = os.path.join(checkpoint_dir, "checkpoints")
        if os.path.isdir(candidate_dir):
            orbax_ckpt_dir = os.path.abspath(candidate_dir)

    if orbax_ckpt_dir is not None:
        print(f"[info] Found orbax checkpoint directory: {orbax_ckpt_dir}")
        try:
            _sl_tx = optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(learning_rate=1e-5, eps=1e-5),
            )
            _dummy_x = jnp.ones((1, action_horizon, action_shape))
            _dummy_cache_k = jnp.ones((1, num_cache_layers, 512, 8, 64))
            _dummy_cache_v = jnp.ones((1, num_cache_layers, 512, 8, 64))
            _dummy_mask = jnp.ones((1, 8, action_horizon + 1, 512))
            _dummy_t = jnp.ones((1,))
            _dummy_state = jnp.ones((1, 9))
            _dummy_params = model_flow.init(
                jax.random.PRNGKey(0), _dummy_x, _dummy_mask, _dummy_cache_k, _dummy_cache_v, _dummy_t, _dummy_state
            )
            _sl_template = TrainState.create(
                apply_fn=model_flow.apply, params=_dummy_params, tx=_sl_tx,
            )

            _ckpt_mgr = ocp.CheckpointManager(
                directory=orbax_ckpt_dir,
                options=ocp.CheckpointManagerOptions(max_to_keep=None, save_interval_steps=1),
                item_names=('train_state', 'metadata'),
                item_handlers={
                    'train_state': ocp.StandardCheckpointHandler(),
                    'metadata': ocp.JsonCheckpointHandler(),
                },
            )
            _latest = _ckpt_mgr.latest_step()
            if _latest is not None:
                print(f"[info] Restoring orbax checkpoint at step {_latest} ...")
                _restored = _ckpt_mgr.restore(
                    _latest,
                    args=ocp.args.Composite(
                        train_state=ocp.args.StandardRestore(_sl_template),
                        metadata=ocp.args.JsonRestore(),
                    ),
                )
                params_flow = _restored.train_state.params
                _meta = _restored.metadata
                print(f"[info] Orbax checkpoint restored — flow params frozen "
                      f"(SL step={_meta.get('global_step', '?')}, epoch={_meta.get('epoch', '?')})")
                del _restored, _sl_template, _ckpt_mgr
            else:
                print("[info] Orbax checkpoint dir exists but contains no checkpoints — falling back to .npy")
        except Exception as e:
            print(f"[warning] Failed to restore orbax checkpoint: {e} — falling back to .npy")

    # 2) Fall back to .npy params file if orbax didn't work
    if params_flow is None:
        flow_checkpoint = None
        if checkpoint_dir:
            candidate = os.path.join(checkpoint_dir, "flow_model_kvcache_batchedaa_shifted_final.npy")
            if os.path.exists(candidate):
                flow_checkpoint = candidate
            else:
                for i in range(607, 0, -1):
                    candidate = os.path.join(checkpoint_dir, f"flow_model_kvcache_batched_shifted_full_{i}.npy")
                    if os.path.exists(candidate):
                        flow_checkpoint = candidate
                        break

        if not flow_checkpoint:
            flow_checkpoint = "flow_model_f_9.npy"
            if not os.path.exists(flow_checkpoint):
                for i in range(400, 0, -1):
                    candidate = f"LIBERO/flow_model_state_{i}.npy"
                    if os.path.exists(candidate):
                        flow_checkpoint = candidate
                        break

        if not os.path.exists(flow_checkpoint):
            print("[error] No flow model checkpoint found!")
            stop_libero_worker()
            return

        params_flow = jnp.load(flow_checkpoint, allow_pickle=True).item()
        print(f"[info] Flow model loaded from .npy: {flow_checkpoint}")

    num_params = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params_flow))
    print(f"[info] Flow model params (frozen): {num_params:,}")

    # --- Load vision model ---
    sampler = None
    if GEMMA_AVAILABLE:
        try:
            model_vision = gm.nn.IntWrapper(model=gm.nn.Gemma3_4B(), dtype=jnp.int4)
            original_params = gm.ckpts.load_params(
                os.path.abspath(gemma_path+"/gemma3-4b-it")
            )
            params_vision = peft.quantize(original_params, method='INT4', checkpoint_kernel_key='w')
            del original_params; gc.collect()

            tokenizer = gm.text.Gemma3Tokenizer(os.path.abspath(gemma_path+"/tokenizer.model"))
            sampler = Sampler(model=model_vision, params=params_vision,
                              tokenizer=tokenizer, cache_length=256, max_out_length=100)
            print("[info] Vision model loaded")
        except Exception as e:
            print(f"[warning] Vision model failed to load: {e}")
    
    cache = sampler.get_cache_prompt("Hello", images=jnp.zeros((1, 1, 128, 128, 3), dtype=jnp.uint8), rng=jax.random.PRNGKey(0))

    # Create value function model
    num_bins = config["NUM_BINS"]
    v_min = config["V_MIN"]
    v_max = config["V_MAX"]
    bin_centers = jnp.linspace(v_min, v_max, num_bins)
    value_model = ValueNetworkStateRope(
        num_layers=num_cache_layers, num_heads=4, qkv_features=1024, out_features=512,
        input_size=action_shape, gating=True, gating_bias=.1,
        norm_type="rmsnorm", post_attention_norm=True, post_mlp_norm=True,
        num_bins=num_bins
    )
    cache_k = extract_cache_from_layers(cache, config["CACHE_LAYERS"], "k")
    cache_v = extract_cache_from_layers(cache, config["CACHE_LAYERS"], "v")
    first_layer = config["CACHE_LAYERS"][0]
    state = jnp.zeros((1, 1, 9))
    cache_mask = create_cache_mask(cache[first_layer]["end_index"], cache_k.shape[2], state.shape[1], cache_k.shape[3])

    params_value = value_model.init(jax.random.PRNGKey(0), cache_mask, cache_k, cache_v, state)
    value_fn = jax.jit(value_model.apply)

    # --- Try to load previously saved value network params ---
    if checkpoint_dir:
        value_ckpt_dir = os.path.join(checkpoint_dir, "value_checkpoints")
        value_latest_path = os.path.join(value_ckpt_dir, "value_network_latest.npy")
        if os.path.exists(value_latest_path):
            try:
                loaded_value_params = jnp.load(value_latest_path, allow_pickle=True).item()
                # Verify structure matches by checking leaf count and shapes
                loaded_leaves = jax.tree_util.tree_leaves(loaded_value_params)
                init_leaves = jax.tree_util.tree_leaves(params_value)
                if len(loaded_leaves) == len(init_leaves) and all(
                    l.shape == i.shape for l, i in zip(loaded_leaves, init_leaves)
                ):
                    params_value = loaded_value_params
                    num_value_params = sum(np.prod(p.shape) for p in loaded_leaves)
                    print(f"[info] Value network params loaded from {value_latest_path} ({num_value_params:,} params)")
                else:
                    print(f"[warning] Value checkpoint shape mismatch — using random init")
            except Exception as e:
                print(f"[warning] Failed to load value checkpoint: {e} — using random init")
        else:
            print(f"[info] No value checkpoint found at {value_latest_path} — using random init")

    if config["ANNEAL_LR_VALUE"]:
        if config["LR_SCHEDULE_TYPE_VALUE"].lower() == "cosine":
            lr_schedule_value = cosine_lr_schedule(config["LR_VALUE"], config["TOTAL_TRAIN_STEPS"])
        else:
            lr_schedule_value = linear_lr_schedule(config["LR_VALUE"], config["TOTAL_TRAIN_STEPS"])
        tx_value = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM_VALUE"]),
            optax.adam(learning_rate=lr_schedule_value, eps=1e-5),
        )
    else:
        tx_value = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM_VALUE"]), optax.adam(config["LR_VALUE"], eps=1e-5))

    train_state_value = TrainState.create(
        apply_fn=value_fn,
        params=params_value,
        tx=tx_value,
    )

    # --- Save resolved config ---
    print(f"\n[info] Resolved config:")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"[info] Config saved to {os.path.join(checkpoint_dir, 'config.json')}")

    # --- Run rollout + value-only training loop ---
    trajectory = None

    for iter in range(config["TOTAL_TRAIN_ITERS"]):

        rng, rollout_rng = jax.random.split(rng)
        # GATHER A TRAJECTORY (flow model frozen, used only for action sampling)
        trajectory = run_rollout(
            params_flow, apply_fn, train_state_value.params, value_fn, sampler, config, rollout_rng,
            action_shape=action_shape, action_horizon=action_horizon,
            num_diffusion_steps=num_diffusion_steps,
            num_episodes=config["NUM_EPISODES"],
            max_nb_replan=config["MAX_NB_REPLAN"],
            steps_per_replan=config["STEPS_PER_REPLAN"],
            bin_centers=bin_centers,
        )
        # --- Print summary ---
        if trajectory is not None:
            rewards = np.array(trajectory['rewards'])
            print(f"\n{'=' * 60}")
            print(f"Task: {libero_comms.libero_task_name}")
            print(f"Steps: {len(trajectory['actions'])}, Replans: {len(trajectory['replans'])}")
            print(f"Total reward: {rewards.sum():.4f}, Mean reward: {rewards.mean():.4f}")
            print(f"KV caches saved: {len(trajectory['kv_caches'])} "
                f"(shape per replan: k={trajectory['kv_caches'][0]['cache_k'].shape})")
            print(f"{'=' * 60}")

        # Compute discounted rewards for value targets
        value = jnp.array(trajectory['values'])
        
        gamma = config["GAMMA"]
        advantage_method = config.get("ADVANTAGE_METHOD", "mc")
        n = len(trajectory['replans'])

        if advantage_method == "gae":
            gae_lambda = config.get("GAE_LAMBDA", 0.95)
            advantages_list = [0.0] * n
            gae = 0.0
            for i in reversed(range(n)):
                replan = trajectory['replans'][i]
                reward_i = replan['replan_reward']
                
                if replan['done'] or i == n - 1:
                    next_value = 0.0
                else:
                    next_value = float(value[i + 1])
                
                if replan['done']:
                    gae = 0.0
                
                delta = reward_i + gamma * next_value - float(value[i])
                gae = delta + gamma * gae_lambda * gae
                advantages_list[i] = gae
            
            advantages = jnp.array(advantages_list)
            discounted_rewards = advantages + value
            raw_advantages = advantages

        else:
            # --- Monte Carlo returns ---
            discounted_rewards_list = []
            cumulative = 0.0
            for i, replan in enumerate(reversed(trajectory['replans'])):
                if replan['done']:
                    cumulative = 0.0
                cumulative = cumulative * gamma + replan['replan_reward']
                discounted_rewards_list.append(cumulative)
            discounted_rewards_list.reverse()
            discounted_rewards = jnp.array(discounted_rewards_list)
            advantages = discounted_rewards - value
            raw_advantages = advantages

        # --- Value-only training ---
        num_epochs = config["NUM_EPOCHS"]
        grad_accum_steps = config["GRAD_ACCUM_STEPS"]

        iter_loss_value_total = 0.0
        iter_loss_updates = 0
        for ep in range(num_epochs):
            print(f"\nEpoch {ep+1}/{num_epochs}")

            n = len(trajectory['replans'])
            batch_size = config["BATCH_SIZE"]
            indices = np.arange(n)
            np.random.shuffle(indices)
            num_minibatches = n // batch_size

            accumulated_grads_value = None
            num_accumulated = 0
            loss_mean_value = 0.0

            for j in range(num_minibatches):
                start = j * batch_size
                end = start + batch_size
                batch_indices = indices[start:end]
                batch_cache_k = jnp.array([trajectory['kv_caches'][idx]['cache_k'] for idx in batch_indices])
                batch_cache_v = jnp.array([trajectory['kv_caches'][idx]['cache_v'] for idx in batch_indices])
                batch_mask_value = jnp.array([trajectory['kv_caches'][idx]['cache_mask_value'] for idx in batch_indices])
                batch_discounted_rewards = discounted_rewards[batch_indices]

                state = jnp.array([trajectory['robot_states'][idx] for idx in batch_indices])

                # Compute the grad for value
                grad_value, (loss_value, _) = _compute_gradients_value(
                    value_fn, train_state_value.params, batch_discounted_rewards,
                    batch_cache_k, batch_cache_v, batch_mask_value, state[:, None], bin_centers
                )
                    
                # Accumulate gradients with averaging
                if accumulated_grads_value is None:
                    accumulated_grads_value = jax.tree_util.tree_map(
                        lambda g: g / grad_accum_steps, grad_value
                    )
                else:
                    accumulated_grads_value = jax.tree_util.tree_map(
                        lambda a, g: a + g / grad_accum_steps,
                        accumulated_grads_value, grad_value
                    )
                num_accumulated += 1
                loss_mean_value = loss_mean_value + loss_value

                # Apply accumulated gradients when we reach the accumulation steps
                if num_accumulated >= grad_accum_steps:
                    train_state_value = _apply_accumulated_gradients(train_state_value, accumulated_grads_value)
                    print(f"  Applied accumulated grads (accum={grad_accum_steps}), "
                          f"avg loss value={loss_mean_value / num_accumulated:.6f}")
                    iter_loss_value_total += loss_mean_value / num_accumulated
                    iter_loss_updates += 1
                    accumulated_grads_value = None
                    num_accumulated = 0
                    loss_mean_value = 0.0

            # Apply any remaining accumulated gradients at end of epoch
            if accumulated_grads_value is not None and num_accumulated > 0:
                scale = grad_accum_steps / num_accumulated
                accumulated_grads_value = jax.tree_util.tree_map(lambda g: g * scale, accumulated_grads_value)
                train_state_value = _apply_accumulated_gradients(train_state_value, accumulated_grads_value)
                print(f"  Applied remaining grads (accum={num_accumulated}/{grad_accum_steps}), "
                      f"avg loss value={loss_mean_value / num_accumulated:.6f}")
                iter_loss_value_total += loss_mean_value / num_accumulated
                iter_loss_updates += 1

        # --- Wandb logging ---
        rewards_arr = np.array(trajectory['rewards'])
        mean_reward = float(rewards_arr.mean())
        avg_success = float(np.mean(trajectory['episode_successes']))
        avg_episode_length = float(np.mean(trajectory['episode_lengths']))
        mean_value_loss = iter_loss_value_total / max(iter_loss_updates, 1)
        log_dict = {
            "rollout/avg_success": avg_success,
            "rollout/mean_reward": mean_reward,
            "rollout/avg_episode_length": avg_episode_length,
            "train/mean_value_loss": mean_value_loss,
            "rollout/raw_advantages": wandb.Histogram(np.array(raw_advantages)),
        }
        wandb.log(log_dict, step=iter)
        print(f"\n[wandb] iter={iter} | avg_success={avg_success:.3f} | mean_reward={mean_reward:.4f} "
              f"| avg_ep_len={avg_episode_length:.1f} "
              f"| mean_value_loss={mean_value_loss:.6f}")

        # --- Save value network checkpoint ---
        if checkpoint_dir:
            value_ckpt_dir = os.path.join(checkpoint_dir, "value_checkpoints")
            os.makedirs(value_ckpt_dir, exist_ok=True)
            # Save current iteration (overwrite-friendly numbered files)
            value_params_np = jax.device_get(train_state_value.params)
            save_path = os.path.join(value_ckpt_dir, f"value_network_{iter}.npy")
            np.save(save_path, value_params_np, allow_pickle=True)
            # Also keep a "latest" symlink / copy for easy loading
            latest_path = os.path.join(value_ckpt_dir, "value_network_latest.npy")
            np.save(latest_path, value_params_np, allow_pickle=True)
            print(f"[info] Value network saved to {save_path}")

    print("\n✓ Done.")
    stop_libero_worker()


@functools.partial(jax.jit, static_argnums=(0,))
def _compute_gradients_value(value_fn, params_value, values, cache_k, cache_v, cache_mask, state, bin_centers):
    """
    Compute gradients for the distributional (categorical) value network.
    
    The value network outputs logits over bins. We binarize the discounted
    return into a two-hot target distribution and train with cross-entropy.
    """
    def _loss_fn(params_value, values, cache_mask, cache_k, cache_v, state, bin_centers):
        logits = value_fn(params_value, cache_mask, cache_k, cache_v, state)  # (batch, 1, num_bins)
        logits = logits[:, 0, :]  # (batch, num_bins)
        
        num_bins = bin_centers.shape[0]
        v_min = bin_centers[0]
        v_max = bin_centers[-1]
        
        # Clip targets to [v_min, v_max]
        targets = jnp.clip(values, v_min, v_max)
        
        # Hard binarization: snap to nearest bin
        bin_width = (v_max - v_min) / (num_bins - 1)
        bin_idx = jnp.round((targets - v_min) / bin_width).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
        
        # Build one-hot target: (batch, num_bins)
        target_dist = jax.nn.one_hot(bin_idx, num_bins)
        
        # Cross-entropy loss: -sum(target * log_softmax(logits))
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.sum(target_dist * log_probs, axis=-1))
        
        return loss, None

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    loss, grads = grad_fn(params_value, values, cache_mask, cache_k, cache_v, state, bin_centers)
    return grads, loss


def _apply_accumulated_gradients(train_state, accumulated_grads):
    """Apply accumulated gradients to the training state."""
    train_state = train_state.apply_gradients(grads=accumulated_grads)
    return train_state


if __name__ == "__main__":
    main()
