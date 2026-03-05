"""
RL training script that finetune a flow model on state-based trajectories from LIBERO, 
using a VLM for conditioning. The script runs closed-loop rollouts in the environment, periodically replanning with the flow model and executing actions until done.
 It collects trajectories of observations, actions, rewards, and KV caches for analysis and potential training.
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

def run_rollout(params_flow, apply_fn,params_value,value_fn, sampler, config,rng,
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
            #print(f"\n  [Episode {episode_idx + 1}, Replan {replan_in_ep + 1}/{max_nb_replan}]")

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

            # Save KV cache for this replan
            

            # --- Sample actions ---
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
            value_logits=value_fn(params_value, cache_mask_value, cache_k, cache_v, robot_state_jax[:,None])
            value_probs = jax.nn.softmax(value_logits, axis=-1)  # (1, 1, num_bins)
            value = jnp.sum(value_probs * bin_centers, axis=-1)  # (1, 1) weighted average
            trajectory['values'].append(value[0,0])
            trajectory['kv_caches'].append({
                'cache_k': np.array(cache_k[0]),
                'cache_v': np.array(cache_v[0]),
                'cache_mask': np.array(cache_mask[0]),
                'cache_mask_value': np.array(cache_mask_value[0]),
            })
            #print(f"    Value function output: {value} shape {value.shape}")

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

            # --- Timing summary for this replan ---
            # print(f"  [TIMING] Replan {replan_in_ep+1}: "
            #       f"Env steps={t_env_elapsed:.3f}s ({steps_executed} steps, "
            #       f"{t_env_elapsed/max(steps_executed,1)*1000:.1f}ms/step)")

            # Compute reward for this replan (sum of rewards over steps in this replan)
            replan_rewards = trajectory['rewards'][-steps_per_replan:]
            replan_reward = sum(replan_rewards)
            replan_mask = trajectory['mask'][-steps_per_replan:]
            cumulative_episode_reward += replan_reward
            # print(f"    Replan {replan_in_ep + 1}/{max_nb_replan}: "
            #       f"reward={replan_reward:.4f}, "
            #       f"cumulative_episode_reward={cumulative_episode_reward:.4f}")

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
    print("LIBERO State-Based Flow Model Test (no visualization)")
    print("=" * 60)

    # --- Parse args ---
    checkpoint_dir = None
    task_id = 1
    cli_overrides = {}
    rng=jax.random.PRNGKey(0)
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--task-id" and i + 1 < len(sys.argv):
            task_id = int(sys.argv[i + 1]); i += 2
        elif arg == "--diffusion-steps" and i + 1 < len(sys.argv):
            cli_overrides["num_diffusion_steps"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--lr-flow" and i + 1 < len(sys.argv):
            cli_overrides["LR_FLOW"] = float(sys.argv[i + 1]); i += 2
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
        elif arg == "--ppo-clip" and i + 1 < len(sys.argv):
            cli_overrides["PPO_CLIP_EPS"] = float(sys.argv[i + 1]); i += 2
        elif arg == "--num-episodes" and i + 1 < len(sys.argv):
            cli_overrides["NUM_EPISODES"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--max-replan" and i + 1 < len(sys.argv):
            cli_overrides["MAX_NB_REPLAN"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--train-iters" and i + 1 < len(sys.argv):
            cli_overrides["TOTAL_TRAIN_ITERS"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--n-flow-samples" and i + 1 < len(sys.argv):
            cli_overrides["N_FLOW_SAMPLES"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--warmup-iters" and i + 1 < len(sys.argv):
            cli_overrides["WARMUP_ITERS"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--warmup-epochs" and i + 1 < len(sys.argv):
            cli_overrides["WARMUP_EPOCHS"] = int(sys.argv[i + 1]); i += 2
        elif arg == "--anneal-lr-flow" :
            cli_overrides["ANNEAL_LR_FLOW"] = True; i += 1
        elif arg == "--anneal-lr-value":
            cli_overrides["ANNEAL_LR_VALUE"] = True; i += 1
        elif arg == "--noise-scale" and i + 1 < len(sys.argv):
            cli_overrides["NOISE_SCALE"] = float(sys.argv[i + 1]); i += 2
        elif arg == "--leaky-relu-slope" and i + 1 < len(sys.argv):
            cli_overrides["LEAKY_RELU_SLOPE"] = float(sys.argv[i + 1]); i += 2
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
        "NUM_EPISODES": 5,
        "MAX_NB_REPLAN": 16,
        "STEPS_PER_REPLAN": 16,

        # --- Training loop ---
        "TOTAL_TRAIN_ITERS": 30,
        "NUM_EPOCHS": 6,
        "BATCH_SIZE": 8,
        "GRAD_ACCUM_STEPS": 8,

        # --- Optimiser (flow network) ---
        "LR_FLOW": 1e-6,
        "ANNEAL_LR_FLOW": True,
        "LR_SCHEDULE_TYPE_FLOW": "cosine",   # "cosine" | "linear"
        "MAX_GRAD_NORM_FLOW": 0.5,

        # --- Optimiser (value network) ---
        "LR_VALUE": 6e-5,
        "ANNEAL_LR_VALUE": True,
        "LR_SCHEDULE_TYPE_VALUE": "cosine",   # "cosine" | "linear"
        "MAX_GRAD_NORM_VALUE": 0.5,

        # --- PPO / advantage ---
        "GAMMA": 0.87,
        "GAE_LAMBDA": 0.95,            # λ for GAE (only used when ADVANTAGE_METHOD="gae")
        "ADVANTAGE_METHOD": "mc",     # "mc" (Monte Carlo returns) | "gae" (Generalized Advantage Estimation)
        "PPO_CLIP_EPS": 0.08,          # Paper sweeps over {0.04, 0.05, 0.06} for FPO++
        "N_FLOW_SAMPLES": 10,

        # --- CFM loss clamping (paper Appendix C.23) ---
        "CFM_LOSS_CLAMP": 1.,      # clamp individual CFM losses before differencing
        "CFM_DIFF_CLAMP": 1.,       # clamp (loss_old - loss) before exp()

        # --- Zero-sampling (paper Section III-D) ---
        "ZERO_SAMPLING_ROLLOUT": False,  # Use ε=0 during evaluation rollouts (True for eval/deploy)
        "NOISE_SCALE": 0.1,             # Scale factor on initial noise ε for exploration (1.0=standard)

        # --- Leaky ReLU slope for advantage clipping ---
        "LEAKY_RELU_SLOPE": 0.00001,     # Negative slope for leaky ReLU applied to advantages

        # --- Distributional value (categorical) ---
        "NUM_BINS": 41,       # number of atoms for categorical value distribution
        "V_MIN": 0.0,        # minimum value support
        "V_MAX": 1.0,         # maximum value support

        # --- Warmup (value-only) ---
        "WARMUP_ITERS": 0,           # number of initial iters where only value is updated
        "WARMUP_EPOCHS": 10,         # number of epochs per warmup iter (more than normal)

        # --- Misc ---
        "TOTAL_TRAIN_STEPS": 10000,  # for LR schedule denominator
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
    project='VLA_LIBERO_RL_bis',
    config=config
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
    model_flow = TransformerFlow(
        num_layers=num_cache_layers, num_heads=4, qkv_features=1024, out_features=512,
        input_size=action_shape, gating=True, gating_bias=2.,
        norm_type="rmsnorm", post_attention_norm=True, post_mlp_norm=True
    )
    apply_fn = jax.jit(model_flow.apply)

    orbax_restored_state = None  # Will be set if we successfully restore from orbax

    # 1) Try orbax checkpoint inside checkpoint_dir/checkpoints
    orbax_ckpt_dir = None
    if checkpoint_dir:
        candidate_dir = os.path.join(checkpoint_dir, "checkpoints")
        if os.path.isdir(candidate_dir):
            orbax_ckpt_dir = os.path.abspath(candidate_dir)

    if orbax_ckpt_dir is not None:
        print(f"[info] Found orbax checkpoint directory: {orbax_ckpt_dir}")
        try:
            # Use PyTreeRestore (shape-agnostic) instead of StandardRestore with a
            # dummy template.  Gemma-3-4B uses GQA so different cache layers have
            # different KV head counts / head dims (e.g. 8×64 vs 4×256).  Building
            # a single dummy with uniform shapes causes shape-mismatch errors when
            # Orbax tries to reconstruct the parameter arrays.
            _ckpt_mgr = ocp.CheckpointManager(
                directory=orbax_ckpt_dir,
                options=ocp.CheckpointManagerOptions(max_to_keep=None, save_interval_steps=1),
                item_names=('train_state', 'metadata'),
            )
            _latest = _ckpt_mgr.latest_step()
            if _latest is not None:
                print(f"[info] Restoring orbax checkpoint at step {_latest} ...")
                _restored = _ckpt_mgr.restore(
                    _latest,
                    args=ocp.args.Composite(
                        train_state=ocp.args.PyTreeRestore(),
                        metadata=ocp.args.JsonRestore(),
                    ),
                )
                # PyTreeRestore returns a raw pytree (dict); extract params.
                _ts = _restored.train_state
                if isinstance(_ts, dict) and 'params' in _ts:
                    params_flow = jax.tree_util.tree_map(jnp.asarray, _ts['params'])
                else:
                    params_flow = jax.tree_util.tree_map(jnp.asarray, getattr(_ts, 'params', _ts))
                orbax_restored_state = _ts  # mark as successfully restored
                _meta = _restored.metadata
                print(f"[info] Orbax checkpoint restored — params "
                      f"(SL step={_meta.get('global_step', '?')}, epoch={_meta.get('epoch', '?')})")
                del _restored, _ckpt_mgr
            else:
                print("[info] Orbax checkpoint dir exists but contains no checkpoints — falling back to .npy")
        except Exception as e:
            print(f"[warning] Failed to restore orbax checkpoint: {e} — falling back to .npy")
            orbax_restored_state = None

    # 2) Fall back to .npy params file if orbax didn't work
    if orbax_restored_state is None:
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
    print(f"[info] Flow model params: {num_params:,}")

    if config["ANNEAL_LR_FLOW"]:
            if config["LR_SCHEDULE_TYPE_FLOW"].lower() == "cosine":
                lr_schedule_flow = cosine_lr_schedule(config["LR_FLOW"], config["TOTAL_TRAIN_STEPS"])
            else:
                lr_schedule_flow = linear_lr_schedule(config["LR_FLOW"], config["TOTAL_TRAIN_STEPS"])
            tx_flow = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM_FLOW"]),
                optax.adam(learning_rate=lr_schedule_flow, eps=1e-5),
            )
    else:
        tx_flow = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM_FLOW"]), optax.adam(config["LR_FLOW"], eps=1e-5))

    train_state_flow = TrainState.create(
                apply_fn=jax.jit(model_flow.apply),
                params=params_flow,
                tx=tx_flow,
            )

    # If we restored from orbax, transplant the Adam mu/nu into the fresh state
    # (keeping step=0 so the RL LR schedule starts fresh).
    if orbax_restored_state is not None:
        # PyTreeRestore returns everything as raw dicts/lists (not namedtuples),
        # so we extract mu/nu via dict keys and convert to jax arrays.
        if isinstance(orbax_restored_state, dict):
            restored_opt_raw = orbax_restored_state['opt_state']
        else:
            restored_opt_raw = orbax_restored_state.opt_state

        # Navigate the optimizer pytree:
        #   opt_state is a tuple: (ClipByGlobalNormState, (ScaleByAdamState, ScaleState))
        # PyTreeRestore may serialize tuples as lists and namedtuples as dicts.
        if isinstance(restored_opt_raw, (list, tuple)):
            restored_adam_raw = restored_opt_raw[1][0]
        else:
            # Fallback: try dict keys that orbax may use
            restored_adam_raw = restored_opt_raw

        # Extract mu and nu — could be dict keys or attributes
        if isinstance(restored_adam_raw, dict):
            restored_mu = jax.tree_util.tree_map(jnp.asarray, restored_adam_raw['mu'])
            restored_nu = jax.tree_util.tree_map(jnp.asarray, restored_adam_raw['nu'])
        else:
            restored_mu = jax.tree_util.tree_map(jnp.asarray, restored_adam_raw.mu)
            restored_nu = jax.tree_util.tree_map(jnp.asarray, restored_adam_raw.nu)

        fresh_opt = train_state_flow.opt_state
        fresh_adam = fresh_opt[1][0]
        new_adam = fresh_adam._replace(mu=restored_mu, nu=restored_nu)
        new_opt_state = (
            fresh_opt[0],                    # clip_by_global_norm (unchanged)
            (new_adam, fresh_opt[1][1]),      # (adam with restored mu/nu, scale state)
        )
        train_state_flow = train_state_flow.replace(opt_state=new_opt_state)
        print(f"[info] Adam mu/nu transplanted from orbax checkpoint (step reset to 0)")
        del orbax_restored_state
    
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
    
    cache = sampler.get_cache_prompt("Hello",images=jnp.zeros((1, 1, 128, 128, 3),dtype=jnp.uint8), rng=jax.random.PRNGKey(0))
    #Create value function model
    num_bins = config["NUM_BINS"]
    v_min = config["V_MIN"]
    v_max = config["V_MAX"]
    bin_centers = jnp.linspace(v_min, v_max, num_bins)
    value_model= ValueNetworkStateRope(
        num_layers=num_cache_layers, num_heads=4, qkv_features=1024, out_features=512,
        input_size=action_shape, gating=True, gating_bias=.1,
        norm_type="rmsnorm", post_attention_norm=True, post_mlp_norm=True,
        num_bins=num_bins
    )
    cache_k = extract_cache_from_layers(cache, config["CACHE_LAYERS"], "k")
    cache_v = extract_cache_from_layers(cache, config["CACHE_LAYERS"], "v")
    # Use first layer from config to get end_index (all layers should have same end_index)
    first_layer = config["CACHE_LAYERS"][0]
    state=jnp.zeros((1,1,9))
    cache_mask=create_cache_mask(cache[first_layer]["end_index"], cache_k.shape[2], state.shape[1], cache_k.shape[3])
    #cache_mask=jnp.ones((4,8,16,32))

    params_value = value_model.init(jax.random.PRNGKey(0), cache_mask, cache_k, cache_v, state)
    value_fn=jax.jit(value_model.apply)

    # --- Try to load previously saved value network params ---
    if checkpoint_dir:
        value_ckpt_dir = os.path.join(checkpoint_dir, "value_checkpoints")
        value_latest_path = os.path.join(value_ckpt_dir, "value_network_latest.npy")
        if os.path.exists(value_latest_path):
            try:
                loaded_value_params = jnp.load(value_latest_path, allow_pickle=True).item()
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

    # --- Run rollout ---
    trajectory = None

    for iter in range(config["TOTAL_TRAIN_ITERS"]):

        rng, rollout_rng = jax.random.split(rng)
        # GATHER A TRAJECTORY
        trajectory = run_rollout(
            train_state_flow.params, apply_fn,train_state_value.params,value_fn, sampler, config, rollout_rng,
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

        #compute advantages over replans over trajectory
        value=jnp.array(trajectory['values'])
        
        gamma = config["GAMMA"]
        advantage_method = config.get("ADVANTAGE_METHOD", "mc")
        n = len(trajectory['replans'])

        if advantage_method == "gae":
            # --- Generalized Advantage Estimation (GAE-λ) ---
            # Paper uses GAE with λ=0.99 (Section C.3).
            # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            # A_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}
            gae_lambda = config.get("GAE_LAMBDA", 0.95)
            advantages_list = [0.0] * n
            gae = 0.0
            for i in reversed(range(n)):
                replan = trajectory['replans'][i]
                reward_i = replan['replan_reward']
                
                if replan['done'] or i == n - 1:
                    # Terminal or last step: no next value
                    next_value = 0.0
                else:
                    next_value = float(value[i + 1])
                
                # If the *previous* replan was terminal (i.e. new episode started
                # at step i), reset the running GAE accumulator.
                if replan['done']:
                    gae = 0.0
                
                delta = reward_i + gamma * next_value - float(value[i])
                gae = delta + gamma * gae_lambda * gae
                advantages_list[i] = gae
            
            advantages = jnp.array(advantages_list)
            # Recompute returns from GAE: R_t = A_t + V(s_t)
            discounted_rewards = advantages + value
            raw_advantages = advantages  # before leaky relu
            advantages=jax.nn.leaky_relu(advantages, negative_slope=config["LEAKY_RELU_SLOPE"])

        else:
            # --- Monte Carlo returns (original implementation) ---
            discounted_rewards_list = []
            cumulative = 0.0
            for i, replan in enumerate(reversed(trajectory['replans'])):
                if replan['done']:
                    cumulative = 0.0
                cumulative = cumulative * gamma + replan['replan_reward']
                discounted_rewards_list.append(cumulative)
            discounted_rewards_list.reverse()
            discounted_rewards = jnp.array(discounted_rewards_list)
            #advantages=jnp.where(discounted_rewards >0.1, 1.0, -1.0)
            advantages = discounted_rewards - value
            # advantages = (adv antages - advantages.mean()) / (advantages.std() + 1e-8)
            #advantages = (advantages - advantages.mean())
            #advantages=jnp.where(advantages > 0., advantages+0.1, advantages * 0.00001)
            raw_advantages = advantages  # before leaky relu
            advantages=jax.nn.leaky_relu(advantages, negative_slope=config["LEAKY_RELU_SLOPE"])
            # advantages=(discounted_rewards>0.1)
            #advantages=advantages+0.05*(discounted_rewards>0.1)
            # advantages= advantages**2 * jnp.sign(advantages)
        # for i, replan in enumerate(trajectory['replans']):
        #     print(f"Replan {i}: reward={replan['replan_reward']:.4f}, "
        #           f"discounted_reward={discounted_rewards[i]}, "
        #           f"value={value[i]}, "
        #           f"advantage={advantages[i]}")
            

        # Determine if we are in warmup phase (value-only, more epochs)
        is_warmup = iter < config["WARMUP_ITERS"]
        if is_warmup:
            num_epochs = config["WARMUP_EPOCHS"]
            print(f"\n*** WARMUP iter {iter+1}/{config['WARMUP_ITERS']} — "
                  f"value-only training, {num_epochs} epochs ***")
        else:
            num_epochs = config["NUM_EPOCHS"]
        grad_accum_steps = config["GRAD_ACCUM_STEPS"]

        params_flow_before = train_state_flow.params
        iter_loss_value_total = 0.0
        iter_loss_flow_total = 0.0
        iter_loss_updates = 0
        diag_list = []  # Will hold (loss_before_raw, loss_raw, cfm_diff, has_pos_adv) from last epoch
        for i in range(num_epochs):
            print(f"\nEpoch {i+1}/{num_epochs}")
            # Here you would typically update your flow model using the collected trajectory,
            # e.g. by computing loss based on the advantages and discounted rewards, and performing back

            # create mini-batches from replans and perform updates
            n=len(trajectory['replans'])
            batch_size = config["BATCH_SIZE"]
            indices=np.arange(n)
            np.random.shuffle(indices)
            num_minibatches=n//batch_size

            accumulated_grads_value = None
            accumulated_grads_flow = None
            num_accumulated = 0
            loss_mean_value=0.0
            loss_mean_flow=0.0
            # Reset per-minibatch diagnostics for this epoch (overwritten each epoch, keep last)
            diag_list = []

            for j in range(num_minibatches):
                start=j*batch_size
                end = start + batch_size
                batch_indices = indices[start:end]
                batch_cache_k = jnp.array([trajectory['kv_caches'][idx]['cache_k'] for idx in batch_indices])
                batch_cache_v = jnp.array([trajectory['kv_caches'][idx]['cache_v'] for idx in batch_indices])
                batch_mask_value = jnp.array([trajectory['kv_caches'][idx]['cache_mask_value'] for idx in batch_indices])
                batch_mask_flow=jnp.array([trajectory['kv_caches'][idx]['cache_mask'] for idx in batch_indices])
                batch_advantages = advantages[batch_indices]
                batch_discounted_rewards = discounted_rewards[batch_indices]

                action_mask = jnp.array([trajectory['replans'][idx]['replan_mask'] for idx in batch_indices])

                actions=jnp.array([trajectory['replans'][idx]['action_sequence'] for idx in batch_indices])
                state = jnp.array([trajectory['robot_states'][idx] for idx in batch_indices])
                # Here you would compute the loss using the batch data and perform an optimization step
                
                # compute the grad for value here
                grad_value,(loss_value,_)=_compute_gradients_value(value_fn,train_state_value.params,batch_discounted_rewards, batch_cache_k, batch_cache_v, batch_mask_value, state[:,None], bin_centers)
                    
                # Accumulate gradients with averaging
                if accumulated_grads_value is None:
                    accumulated_grads_value = jax.tree_util.tree_map(
                        lambda g: g / grad_accum_steps,
                        grad_value
                    )
                else:
                    accumulated_grads_value = jax.tree_util.tree_map(
                        lambda a, g: a + g / grad_accum_steps,
                        accumulated_grads_value,
                        grad_value
                    )
                num_accumulated += 1

                loss_mean_value = loss_mean_value + loss_value

                if not is_warmup:
                    rng, _rng = jax.random.split(rng)
                    grad_flow,(loss_flow,(batch_lb, batch_lr, batch_cd))= _compute_gradient_flow(flow_fn=apply_fn,params=train_state_flow.params,params_before=params_flow_before, cache_k=batch_cache_k, cache_v=batch_cache_v, cache_mask=batch_mask_flow, actions=actions, state=state, action_mask=action_mask, rng=_rng, advantages=batch_advantages, clip_eps=config["PPO_CLIP_EPS"], n_samples=config["N_FLOW_SAMPLES"], cfm_loss_clamp=config["CFM_LOSS_CLAMP"], cfm_diff_clamp=config["CFM_DIFF_CLAMP"])
                    
                    # Store per-minibatch diagnostics (scalar values averaged over n_samples)
                    has_pos_adv = bool(jnp.any(batch_advantages > 0))
                    diag_list.append((float(batch_lb), float(batch_lr), float(batch_cd), has_pos_adv))
                    
                    # Accumulate gradients with averaging
                    if accumulated_grads_flow is None:
                        accumulated_grads_flow = jax.tree_util.tree_map(
                            lambda g: g / grad_accum_steps,
                            grad_flow
                        )
                    else:
                        accumulated_grads_flow = jax.tree_util.tree_map(
                            lambda a, g: a + g / grad_accum_steps,
                            accumulated_grads_flow,
                            grad_flow
                        )
                    loss_mean_flow = loss_mean_flow + loss_flow

                # Apply accumulated gradients when we reach the accumulation steps
                if num_accumulated >= grad_accum_steps:
                    train_state_value = _apply_accumulated_gradients(train_state_value, accumulated_grads_value)
                    if not is_warmup and accumulated_grads_flow is not None:
                        train_state_flow = _apply_accumulated_gradients(train_state_flow, accumulated_grads_flow)
                    print(f"  Applied accumulated grads (accum={grad_accum_steps}), avg loss value={loss_mean_value / num_accumulated:.6f}, avg loss flow={loss_mean_flow / num_accumulated:.6f}{' [warmup: value-only]' if is_warmup else ''}")
                    iter_loss_value_total += loss_mean_value / num_accumulated
                    iter_loss_flow_total += loss_mean_flow / num_accumulated
                    iter_loss_updates += 1
                    accumulated_grads_value = None
                    accumulated_grads_flow = None
                    num_accumulated = 0
                    loss_mean_value = 0.0
                    loss_mean_flow = 0.0

            # Apply any remaining accumulated gradients at end of epoch
            if accumulated_grads_value is not None and num_accumulated > 0:
                # Re-scale: grads were divided by grad_accum_steps, but we only accumulated num_accumulated
                scale = grad_accum_steps / num_accumulated
                accumulated_grads_value = jax.tree_util.tree_map(lambda g: g * scale, accumulated_grads_value)
                train_state_value = _apply_accumulated_gradients(train_state_value, accumulated_grads_value)
                print(f"  Applied remaining grads (accum={num_accumulated}/{grad_accum_steps}), avg loss value={loss_mean_value / num_accumulated:.6f}")
                iter_loss_value_total += loss_mean_value / num_accumulated
                iter_loss_updates += 1
            if not is_warmup and accumulated_grads_flow is not None and num_accumulated > 0:
                # Re-scale: grads were divided by grad_accum_steps, but we only accumulated num_accumulated
                scale = grad_accum_steps / num_accumulated
                accumulated_grads_flow = jax.tree_util.tree_map(lambda g: g * scale, accumulated_grads_flow)
                train_state_flow = _apply_accumulated_gradients(train_state_flow, accumulated_grads_flow)
                print(f"  Applied remaining grads (accum={num_accumulated}/{grad_accum_steps}), avg loss flow={loss_mean_flow / num_accumulated:.6f}")
                iter_loss_flow_total += loss_mean_flow / num_accumulated
  
        # --- Compute flow diagnostics for positive-advantage replans ---
        wandb_extras = {}
        if not is_warmup and diag_list:
            # Filter for minibatches that contained positive advantages
            pos_diags = [(lb, lr, cd) for lb, lr, cd, has_pos in diag_list if has_pos]
            if pos_diags:
                diag_lb_arr = np.array([d[0] for d in pos_diags])
                diag_lr_arr = np.array([d[1] for d in pos_diags])
                diag_cd_arr = np.array([d[2] for d in pos_diags])
                wandb_extras["train/loss_before_raw_pos_adv"] = wandb.Histogram(diag_lb_arr)
                wandb_extras["train/loss_raw_pos_adv"] = wandb.Histogram(diag_lr_arr)
                wandb_extras["train/cfm_diff_pos_adv"] = wandb.Histogram(diag_cd_arr)

        # --- Wandb logging ---
        rewards_arr = np.array(trajectory['rewards'])
        mean_reward = float(rewards_arr.mean())
        avg_success = float(np.mean(trajectory['episode_successes']))
        avg_episode_length = float(np.mean(trajectory['episode_lengths']))
        mean_value_loss = iter_loss_value_total / max(iter_loss_updates, 1)
        mean_flow_loss = iter_loss_flow_total / max(iter_loss_updates, 1)
        log_dict = {
            "rollout/avg_success": avg_success,
            "rollout/mean_reward": mean_reward,
            "rollout/avg_episode_length": avg_episode_length,
            "train/mean_value_loss": mean_value_loss,
            "train/mean_flow_loss": mean_flow_loss,
            "train/is_warmup": float(is_warmup),
            "rollout/raw_advantages": wandb.Histogram(np.array(raw_advantages)),
        }
        log_dict.update(wandb_extras)
        wandb.log(log_dict, step=iter)
        print(f"\n[wandb] iter={iter} | avg_success={avg_success:.3f} | mean_reward={mean_reward:.4f} "
              f"| avg_ep_len={avg_episode_length:.1f} "
              f"| mean_value_loss={mean_value_loss:.6f} | mean_flow_loss={mean_flow_loss:.6f}"
              f"{' | WARMUP' if is_warmup else ''}")

        # --- Save RL-finetuned models ---
        if checkpoint_dir:
            rl_save_dir = os.path.join(checkpoint_dir, "rl_finetune")
            os.makedirs(rl_save_dir, exist_ok=True)

            # Save flow model params (latest + periodic)
            flow_params_np = jax.device_get(train_state_flow.params)
            flow_latest_path = os.path.join(rl_save_dir, "flow_model_latest.npy")
            np.save(flow_latest_path, flow_params_np, allow_pickle=True)

            flow_iter_path = os.path.join(rl_save_dir, f"flow_model_iter_{iter}.npy")
            np.save(flow_iter_path, flow_params_np, allow_pickle=True)

            # Save value model params (latest + periodic)
            value_params_np = jax.device_get(train_state_value.params)
            value_latest_path = os.path.join(rl_save_dir, "value_network_latest.npy")
            np.save(value_latest_path, value_params_np, allow_pickle=True)

            value_iter_path = os.path.join(rl_save_dir, f"value_network_iter_{iter}.npy")
            np.save(value_iter_path, value_params_np, allow_pickle=True)

            # Save current config alongside
            config_save_path = os.path.join(rl_save_dir, "config.json")
            with open(config_save_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"[info] RL-finetuned models saved to {rl_save_dir} (iter {iter})")

    print("\n✓ Done.")
    stop_libero_worker()


@functools.partial(jax.jit, static_argnums=(0,))
def _compute_gradients_value(value_fn,params_value,values, cache_k, cache_v, cache_mask, state, bin_centers):
    """
    Compute gradients for the distributional (categorical) value network.
    
    The value network outputs logits over bins. We binarize the discounted
    return into a two-hot target distribution and train with cross-entropy.
    
    Args:
        value_fn: value network apply function
        params_value: Model parameters
        values: Discounted returns (batch,) - scalar targets
        cache_k: Key cache from Gemma
        cache_v: Value cache from Gemma
        cache_mask: Attention mask for cache
        state: Robot state, shape (batch, 1, 9)
        bin_centers: (num_bins,) array of bin center values
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
        bin_idx = jnp.round((targets - v_min) / bin_width).astype(jnp.int32)  # (batch,)
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

def loss_flow(flow_fn,params, actions, cache_mask, cache_k, cache_v, state, action_mask, t,x0):
        
        # Linear interpolation
        xt = (t[:, None, None]) * x0 + (1 - t[:, None, None]) * actions

        # Target velocity
        target_v = actions - x0
        
        pred_v = flow_fn(
            params, xt, cache_mask, cache_k, cache_v, t, state
        )
        
        # Compute squared error per timestep: (batch, action_horizon, action_shape)
        squared_error = (pred_v - target_v) ** 2

        # Mean over action dimensions: (batch, action_horizon)
        squared_error_per_timestep = jnp.mean(squared_error, axis=-1)

        # Apply action mask to zero out loss from padded actions
        masked_squared_error = squared_error_per_timestep * action_mask

        # Compute mean loss only over valid (non-padded) actions
        # num_valid_actions = jnp.sum(action_mask) + 1e-8  # Avoid division by zero
        # total_loss = jnp.sum(masked_squared_error) / num_valid_actions
        
        #acording to the FPO++ paper, they use the sum of squared error over the horizon (not mean), so we will do the same here
        total_loss = jnp.mean(jnp.sum(masked_squared_error,axis=-1)) # shape (batch,)
        return total_loss


@functools.partial(jax.jit, static_argnums=(0, 12))
def _compute_gradient_flow(flow_fn,params,params_before, cache_k, cache_v, cache_mask, actions, state, action_mask, rng,advantages, clip_eps=0.2, n_samples=20, cfm_loss_clamp=20.0, cfm_diff_clamp=5.0):
    """
    Compute gradients averaged over multiple samples of x0 and t using jax.lax.scan.
    Returns averaged gradients and mean loss.
    
    Args:
        params: Model parameters
        cache_k: Key cache from Gemma
        cache_v: Value cache from Gemma
        cache_mask: Attention mask for cache
        actions: Target actions, shape (batch, action_horizon, action_shape)
        state: Robot state, shape (batch, 9)
        action_mask: Mask for valid actions (1=real, 0=padded), shape (batch, action_horizon)
        rng: Random key
        clip_eps: PPO clipping epsilon (default 0.2 → ratio clipped to [1-eps, 1+eps])
        n_samples: Number of (x0, t) samples to average over
        cfm_loss_clamp: Clamp individual CFM losses to [0, cfm_loss_clamp] before differencing
        cfm_diff_clamp: Clamp (loss_old - loss) to [-cfm_diff_clamp, cfm_diff_clamp] before exp()
    """
    
    def _loss_fn(params, actions, cache_mask, cache_k, cache_v, state, action_mask, t, x0):
        loss_before_raw = jax.lax.stop_gradient(loss_flow(flow_fn, params_before, actions, cache_mask, cache_k, cache_v, state, action_mask, t, x0))
        loss_raw = loss_flow(flow_fn, params, actions, cache_mask, cache_k, cache_v, state, action_mask, t, x0)

        # CFM loss clamping (paper Appendix C.23):
        # Step (i): clamp individual CFM losses before taking differences
        loss_before = jnp.clip(loss_before_raw, 0.0, cfm_loss_clamp)
        loss = jnp.clip(loss_raw, 0.0, cfm_loss_clamp)

        # Step (ii): clamp the difference before exponentiation
        cfm_diff = jnp.clip(loss_before - loss, -cfm_diff_clamp, cfm_diff_clamp)
        
        exp_diff = jnp.exp(cfm_diff)
        ppo_loss = jnp.minimum(exp_diff * advantages, jnp.clip(exp_diff, 1.0 - 0.2*clip_eps, 1.0 + clip_eps) * advantages)
        
        # ppo_loss=advantages*cfm_diff
        #ppo_loss=jnp.where(advantages > 0, jnp.minimum(exp_diff * advantages, jnp.clip(exp_diff, 1.0 - clip_eps, 1.0 + clip_eps) * advantages), advantages*exp_diff-jnp.abs(advantages)/(2*clip_eps)*((exp_diff-1)**2))
        ppo_loss=-jnp.mean(ppo_loss)

        # Per-sample diagnostics (stop_gradient, no effect on grads)
        diag = jax.lax.stop_gradient((loss_before_raw, loss_raw, cfm_diff))
        return ppo_loss, diag

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

    # Initialize accumulated grads as zeros with same structure as params
    acc_grads = jax.tree.map(jnp.zeros_like, params)

    def scan_body(carry, rng_i):
        acc_grads, acc_loss, acc_lb, acc_lr, acc_cd = carry
        rng_i, rng_x0, rng_t = jax.random.split(rng_i, 3)
        x0 = jax.random.normal(rng_x0, actions.shape)*0.05
        t = jax.random.beta(rng_t, 2.0, 2.0, (actions.shape[0],))
        t = jnp.clip(t, 0, 1)

        (loss_val, (lb, lr, cd)), grads = grad_fn(params, actions, cache_mask, cache_k, cache_v, state, action_mask, t, x0)

        # Accumulate averaged gradients and loss directly in the carry
        acc_grads = jax.tree.map(lambda a, g: a + g / n_samples, acc_grads, grads)
        acc_loss = acc_loss + loss_val / n_samples
        acc_lb = acc_lb + lb / n_samples
        acc_lr = acc_lr + lr / n_samples
        acc_cd = acc_cd + cd / n_samples
        return (acc_grads, acc_loss, acc_lb, acc_lr, acc_cd), None

    # Generate n_samples different rng keys
    rng_keys = jax.random.split(rng, n_samples)

    (avg_grads, avg_loss, avg_lb, avg_lr, avg_cd), _ = jax.lax.scan(
        scan_body, (acc_grads, 0.0, 0.0, 0.0, 0.0), rng_keys
    )

    return avg_grads, (avg_loss, (avg_lb, avg_lr, avg_cd))


def _apply_accumulated_gradients(train_state, accumulated_grads):
    """
    Apply accumulated gradients to the training state.
    Gradients are already averaged during accumulation.
    """
    train_state = train_state.apply_gradients(grads=accumulated_grads)
    return train_state


if __name__ == "__main__":
    #python RL_libero_flow_state_server.py run_20260218_011505  --diffusion-steps 10 --task-id 1
    main()
