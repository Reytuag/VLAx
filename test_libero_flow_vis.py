"""
Test script for the state-based flow model on LIBERO spatial manipulation tasks.
Loads a trained flow model and uses it to generate actions conditioned on vision from Gemma3.
Includes rendering and visualization capabilities with dual camera views (agentview + eye-in-hand).

Key differences from test_libero_flow_server_vis.py:
- Uses TransformerFlow with 3 layers (from flow_network_state.py)
- Inputs robot state (9 dimensions) in addition to visual observations
- Uses 3 layers of cache (layer_9, layer_26, layer_31)
- Displays both agentview and eye-in-hand camera views
- Action horizon: 16 (instead of 32)
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import gc
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, Tuple, Optional
import h5py
import subprocess
import json
from datetime import datetime

try:
    from gemma import gm
    from gemma import peft
    from sampler import Sampler
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False
    print("Warning: gemma module not available. Will use dummy cache for testing.")

gemma_path="/home/reytuag/VLA/gemma-3-flax-gemma3-4b-it-v1"
from networks.flow_network_state_rope_attention import TransformerFlow
from visualization.visualization_dual import RealtimeAgentVisualizerDual, render_attention_video_frame
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
import optax


# ============================================================================
# LIBERO Worker Process Communication
# ============================================================================

# Global subprocess for LIBERO environment communication
libero_process = None
libero_task_name = None  # Task name from LIBERO worker
libero_task_description = None  # Task description from LIBERO worker


def _read_json_from_worker():
    """
    Read lines from the worker until we get valid JSON.
    Ignores log spam and prints it as worker logs.
    """
    global libero_process

    while True:
        line = libero_process.stdout.readline()
        if line == "":
            raise RuntimeError("LIBERO worker died (EOF)")

        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Fast JSON heuristic
        if line[0] not in "{[":
            print(f"[worker] {line}")
            continue

        try:
            return json.loads(line)

        except json.JSONDecodeError:
            # It looked like JSON but wasn't valid
            print(f"[worker malformed] {line}")
            continue


def start_libero_worker(task_id: int = 1):
    """
    Start the LIBERO worker subprocess.
    
    Args:
        task_id: Index of the task to use (default: 1)
        
    Returns:
        Popen object for the worker process
    """
    global libero_process
    
    print(f"[info] Starting LIBERO worker process with task_id={task_id}...")
    try:
        libero_process = subprocess.Popen(
            ["/home/reytuag/miniconda3/envs/libero_env/bin/python", "-u", "libero_misc/libero_worker.py", str(task_id)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Important: use text mode
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        print(f"[info] LIBERO worker process started successfully with task_id={task_id}")
        return libero_process
    except Exception as e:
        print(f"[error] Failed to start LIBERO worker: {e}")
        return None


def stop_libero_worker():
    """
    Stop the LIBERO worker subprocess.
    """
    global libero_process
    if libero_process is not None:
        try:
            libero_process.terminate()
            libero_process.wait(timeout=5)
            print("[info] LIBERO worker process terminated")
        except subprocess.TimeoutExpired:
            libero_process.kill()
            print("[info] LIBERO worker process killed")
        libero_process = None


def libero_reset():
    """
    Send reset command to LIBERO worker and get initial observation.
    
    Returns:
        Tuple of (agentview_image, eye_in_hand_image, robot_state) as numpy arrays
    """
    global libero_process
    
    if libero_process is None:
        raise RuntimeError("LIBERO worker not started. Call start_libero_worker() first.")
    
    try:
        # Send reset command
        libero_process.stdin.write(json.dumps({"cmd": "reset"}) + "\n")
        libero_process.stdin.flush()
        
        # Read response
        obs_dict = _read_json_from_worker()
        
        # Extract observations
        if isinstance(obs_dict, dict):
            # Agentview image
            if 'agentview_image' in obs_dict:
                agentview = np.array(obs_dict['agentview_image'], dtype=np.uint8)[::-1] # flip vertically to correct orientation
                if len(agentview.shape) == 1:
                    agentview = agentview.reshape(128, 128, 3)
            else:
                raise ValueError(f"agentview_image not found in observation")
            
            # Eye-in-hand image
            if 'robot0_eye_in_hand_image' in obs_dict:
                eye_in_hand = np.array(obs_dict['robot0_eye_in_hand_image'], dtype=np.uint8)
                if len(eye_in_hand.shape) == 1:
                    eye_in_hand = eye_in_hand.reshape(128, 128, 3)
            else:
                raise ValueError(f"eye_in_hand_image not found in observation")
            
            # Robot state (9D: gripper_qpos(2) + eef_pos(3) + eef_quat(4))
            # This matches the HDF5 robot_states format used in training
            if 'robot0_gripper_qpos' in obs_dict and 'robot0_eef_pos' in obs_dict and 'robot0_eef_quat' in obs_dict:
                gripper_qpos = np.array(obs_dict['robot0_gripper_qpos'], dtype=np.float32)
                eef_pos = np.array(obs_dict['robot0_eef_pos'], dtype=np.float32)
                eef_quat = np.array(obs_dict['robot0_eef_quat'], dtype=np.float32)
                # Concatenate to get 9D state: 2 gripper + 3 eef_pos + 4 eef_quat
                robot_state = np.concatenate([gripper_qpos, eef_pos, eef_quat])
            elif 'robot0_gripper_qpos' in obs_dict and 'robot0_eef_pos' in obs_dict:
                gripper_qpos = np.array(obs_dict['robot0_gripper_qpos'], dtype=np.float32)
                eef_pos = np.array(obs_dict['robot0_eef_pos'], dtype=np.float32)
                # Pad with identity quaternion if eef_quat not available
                eef_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                robot_state = np.concatenate([gripper_qpos, eef_pos, eef_quat])
            else:
                # Fallback to dummy state
                print(f"[warning] Required robot state keys not found, using dummy state")
                print(f"[warning] Available keys: {list(obs_dict.keys())}")
                robot_state = np.zeros(9, dtype=np.float32)
            
            return agentview, eye_in_hand, robot_state
        else:
            raise ValueError(f"Invalid observation format: expected dict, got {type(obs_dict)}")
    except Exception as e:
        print(f"[error] Error in libero_reset: {e}")
        raise


def libero_step(action):
    """
    Send step command to LIBERO worker with action and get result.
    
    Args:
        action: Action to execute (numpy array or list)
        
    Returns:
        Tuple of (agentview_image, eye_in_hand_image, robot_state, reward, done, info)
    """
    global libero_process
    
    if libero_process is None:
        raise RuntimeError("LIBERO worker not started. Call start_libero_worker() first.")
    
    try:
        # Convert action to list if numpy array
        action_list = action.tolist() if isinstance(action, np.ndarray) else action
        
        # Send step command
        libero_process.stdin.write(json.dumps({
            "cmd": "step",
            "action": action_list
        }) + "\n")
        libero_process.stdin.flush()
        
        # Read response
        result = _read_json_from_worker()
        
        # Extract observation
        obs_dict = result.get("obs", {})
        if isinstance(obs_dict, dict):
            # Agentview image
            if 'agentview_image' in obs_dict:
                agentview = np.array(obs_dict['agentview_image'], dtype=np.uint8)[::-1] # flip vertically to correct orientation
                if len(agentview.shape) == 1:
                    agentview = agentview.reshape(128, 128, 3)
            else:
                print(f"[warning] agentview_image not found, creating dummy")
                agentview = np.zeros((128, 128, 3), dtype=np.uint8)
            
            # Eye-in-hand image
            if 'robot0_eye_in_hand_image' in obs_dict:
                eye_in_hand = np.array(obs_dict['robot0_eye_in_hand_image'], dtype=np.uint8)
                if len(eye_in_hand.shape) == 1:
                    eye_in_hand = eye_in_hand.reshape(128, 128, 3)
            else:
                print(f"[warning] eye_in_hand_image not found, creating dummy")
                eye_in_hand = np.zeros((128, 128, 3), dtype=np.uint8)
            
            # Robot state (9D: gripper_qpos(2) + eef_pos(3) + eef_quat(4))
            # This matches the HDF5 robot_states format used in training
            if 'robot0_gripper_qpos' in obs_dict and 'robot0_eef_pos' in obs_dict and 'robot0_eef_quat' in obs_dict:
                gripper_qpos = np.array(obs_dict['robot0_gripper_qpos'], dtype=np.float32)
                eef_pos = np.array(obs_dict['robot0_eef_pos'], dtype=np.float32)
                eef_quat = np.array(obs_dict['robot0_eef_quat'], dtype=np.float32)
                # Concatenate to get 9D state: 2 gripper + 3 eef_pos + 4 eef_quat
                robot_state = np.concatenate([gripper_qpos, eef_pos, eef_quat])
            elif 'robot0_gripper_qpos' in obs_dict and 'robot0_eef_pos' in obs_dict:
                gripper_qpos = np.array(obs_dict['robot0_gripper_qpos'], dtype=np.float32)
                eef_pos = np.array(obs_dict['robot0_eef_pos'], dtype=np.float32)
                # Pad with identity quaternion if eef_quat not available
                eef_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                robot_state = np.concatenate([gripper_qpos, eef_pos, eef_quat])
            else:
                robot_state = np.zeros(9, dtype=np.float32)
        else:
            # Fallback
            agentview = np.zeros((128, 128, 3), dtype=np.uint8)
            eye_in_hand = np.zeros((128, 128, 3), dtype=np.uint8)
            robot_state = np.zeros(9, dtype=np.float32)
        
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        info = result.get("info", {})
        
        return agentview, eye_in_hand, robot_state, reward, done, info
    except Exception as e:
        print(f"[error] Error in libero_step: {e}")
        raise


def load_config_from_checkpoint_dir(checkpoint_dir: str) -> Optional[Dict]:
    """
    Load configuration from a checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the directory containing config.json
        
    Returns:
        Dictionary with configuration, or None if config.json not found
    """
    import json
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[info] Loaded configuration from: {config_path}")
        return config
    else:
        print(f"[warning] Configuration file not found at: {config_path}")
        return None


def load_flow_model(
    checkpoint_path: str,
    action_shape: int = 7,
    action_horizon: int = 16,
    num_layers: int = 3
) -> Tuple:
    """
    Load a trained flow model from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved flow model parameters (.npy file)
        action_shape: Dimension of action space (default: 7 for LIBERO)
        action_horizon: Length of action sequence (default: 16 for state-based model)
        num_layers: Number of transformer layers (default: 3)
        
    Returns:
        Tuple of (model, params, apply_fn)
    """
    # Initialize flow model architecture with configurable number of layers
    model_flow = TransformerFlow(
        num_layers=num_layers,
        num_heads=4,
        qkv_features=1024,
        out_features=512,
        input_size=action_shape,
        gating=True,
        gating_bias=2.,
        norm_type="rmsnorm",  # layernorm or rmsnorm
        post_attention_norm=True,
        post_mlp_norm=True
    )
    
    
    # Load parameters
    params = jnp.load(checkpoint_path, allow_pickle=True).item()
    
    # JIT compile the apply function
    apply_fn = jax.jit(model_flow.apply)
    
    return model_flow, params, apply_fn


def try_restore_params_from_orbax(checkpoint_dir: str):
    """
    Try to restore train_state from an Orbax checkpoint directory and
    extract the model parameters. Returns (params, metadata) or (None, None).
    """
    ckpt_root = os.path.abspath(checkpoint_dir)
    # If user passed the run dir, look into its checkpoints/ subdir
    if not os.path.basename(ckpt_root).startswith("checkpoints"):
        ckpt_root = os.path.join(ckpt_root, "checkpoints")

    if not os.path.isdir(ckpt_root):
        print(f"[info] Orbax checkpoint dir not found at: {ckpt_root}")
        return None, None

    try:
        options = ocp.CheckpointManagerOptions(max_to_keep=3)
        manager = ocp.CheckpointManager(directory=ckpt_root, options=options)
        latest = manager.latest_step()
        if latest is None:
            print(f"[info] No orbax checkpoints found in {ckpt_root}")
            return None, None

        print(f"[info] Restoring Orbax checkpoint at step: {latest}")

        restored = manager.restore(
            latest,
            args=ocp.args.Composite(
                train_state=ocp.args.PyTreeRestore(),
                metadata=ocp.args.JsonRestore(),
            ),
        )

        # The restored.train_state may be a nested dict. Try common locations for params.
        ts = restored.train_state
        metadata = getattr(restored, 'metadata', None)

        # params could be under ts['params'] or ts.params
        params = None
        if isinstance(ts, dict):
            if 'params' in ts:
                params = ts['params']
            elif 'train_state' in ts and isinstance(ts['train_state'], dict) and 'params' in ts['train_state']:
                params = ts['train_state']['params']
        else:
            # try attribute access
            params = getattr(ts, 'params', None)

        if params is None:
            print(f"[warning] Could not find 'params' in restored train_state. Keys: {list(ts.keys()) if isinstance(ts, dict) else dir(ts)}")
            return None, metadata

        # Convert numpy arrays to jax arrays if needed
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)
        print(f"[info] Restored params from Orbax checkpoint (step={latest})")
        return params, metadata

    except Exception as e:
        print(f"[error] Failed to restore Orbax checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_cache_mask(indices, key_length, query_length, num_heads):
    """
    Create attention mask for the cache.
    
    Args:
        indices: End indices for each sequence in batch
        key_length: Length of key cache dimension
        query_length: Length of query sequence
        num_heads: Number of attention heads
        
    Returns:
        Attention mask tensor
    """
    mask = jnp.arange(key_length)[None, :] < indices[:, None]
    mask = mask[:, None, None, :].repeat(num_heads, axis=1).repeat(query_length, axis=2)
    return mask


def extract_cache_from_layers(cache_dict, layer_names, cache_type="k"):
    """
    Extract and concatenate cache tensors from specified layers.
    
    Args:
        cache_dict: Dictionary containing cache for all layers
        layer_names: List of layer names to extract (e.g., ["layer_9", "layer_26", "layer_31"])
        cache_type: Either "k" or "v" for key or value cache
        
    Returns:
        Concatenated cache tensor with shape (batch, num_layers, seq_len, num_heads, head_dim)
    """
    cache_tensors = []
    for layer_name in layer_names:
        cache_tensors.append(cache_dict[layer_name][cache_type][:, None, ...])
    return jnp.concatenate(cache_tensors, axis=1)


def create_video_frame(agentview_image: np.ndarray, eye_in_hand_image: np.ndarray, 
                       task_description: str, step: int) -> np.ndarray:
    """
    Create a video frame with concatenated dual-view images and task descriptor on top.
    
    Args:
        agentview_image: RGB image from agentview camera (H, W, 3)
        eye_in_hand_image: RGB image from eye-in-hand camera (H, W, 3)
        task_description: Task description text to display
        step: Current step number
        
    Returns:
        Combined frame as numpy array (H_total, W, 3) with uint8 dtype
    """
    # Ensure images are uint8
    agentview = agentview_image.astype(np.uint8)
    eye_in_hand = eye_in_hand_image.astype(np.uint8)
    
    # Get dimensions
    h, w, c = agentview.shape
    
    # Create text header using PIL for better text rendering
    text_height = 80
    header_width = w * 2  # Two images side by side
    
    # Create PIL Image for header
    header_img = Image.new('RGB', (header_width, text_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(header_img)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font_task = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_step = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_task = ImageFont.load_default()
        font_step = ImageFont.load_default()
    
    # Wrap task description to fit in 2 lines
    task_text = f"Task: {task_description}"
    max_chars_per_line = header_width // 7  # Approximate char width for font size 11
    if len(task_text) > max_chars_per_line:
        # Find a good split point near the middle
        split_at = task_text.rfind(' ', 0, max_chars_per_line)
        if split_at == -1:
            split_at = max_chars_per_line
        line1 = task_text[:split_at]
        line2 = task_text[split_at:].lstrip()
    else:
        line1 = task_text
        line2 = ""
    
    step_text = f"Step: {step}"
    
    draw.text((10, 5), line1, fill=(255, 255, 255), font=font_task)
    if line2:
        draw.text((10, 22), line2, fill=(255, 255, 255), font=font_task)
    draw.text((10, 50), step_text, fill=(200, 200, 200), font=font_step)
    
    # Convert header to numpy
    header_array = np.array(header_img)
    
    # Concatenate images horizontally
    dual_view = np.concatenate([agentview, eye_in_hand], axis=1)
    
    # Stack header on top of dual view
    final_frame = np.concatenate([header_array, dual_view], axis=0)
    
    return final_frame


def save_trajectory_video(trajectory: Dict, task_description: str, output_path: str, fps: int = 10):
    """
    Save trajectory as a video with concatenated dual-view images and task descriptor.
    
    Args:
        trajectory: Dictionary containing 'agentview_observations' and 'eye_in_hand_observations'
        task_description: Task description to display in video
        output_path: Path to save the video file
        fps: Frames per second for the video
    """
    print(f"\n[info] Saving trajectory video to {output_path}...")
    
    # Import cv2 here to avoid import errors when not saving video
    try:
        import cv2
    except ImportError as e:
        print(f"[error] Failed to import cv2: {e}")
        print("  Please install opencv-python: pip install opencv-python")
        return
    
    agentview_obs = trajectory['agentview_observations']
    eye_in_hand_obs = trajectory['eye_in_hand_observations']
    
    if len(agentview_obs) == 0 or len(eye_in_hand_obs) == 0:
        print("[warning] No observations to save")
        return
    
    # Create first frame to get dimensions
    first_frame = create_video_frame(agentview_obs[0], eye_in_hand_obs[0], task_description, 0)
    height, width, _ = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write all frames
    num_frames = min(len(agentview_obs), len(eye_in_hand_obs))
    for i in range(num_frames):
        frame = create_video_frame(agentview_obs[i], eye_in_hand_obs[i], task_description, i)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"  ✓ Video saved: {output_path} ({num_frames} frames at {fps} fps)")


def save_trajectory_video_with_attention(
    trajectory: Dict,
    task_description: str,
    output_path: str,
    cache_layers: list = None,
    fps: int = 10
):
    """
    Save trajectory as a video with dual camera views AND attention heatmaps,
    similar to the real-time visualization layout.
    
    Args:
        trajectory: Dictionary containing 'agentview_observations', 'eye_in_hand_observations',
                    'robot_states', 'rewards', and 'attention_weights' (list of per-step attention dicts)
        task_description: Task description to display in video
        output_path: Path to save the video file
        cache_layers: List of cache layer names for attention display
        fps: Frames per second for the video
    """
    import matplotlib
    matplotlib.use('Agg')
    
    print(f"\n[info] Saving attention visualization video to {output_path}...")
    
    try:
        import cv2
    except ImportError as e:
        print(f"[error] Failed to import cv2: {e}")
        print("  Please install opencv-python: pip install opencv-python")
        return
    
    agentview_obs = trajectory['agentview_observations']
    eye_in_hand_obs = trajectory['eye_in_hand_observations']
    robot_states = trajectory.get('robot_states', [None] * len(agentview_obs))
    rewards = trajectory.get('rewards', [0.0] * len(agentview_obs))
    attention_list = trajectory.get('attention_weights', [None] * len(agentview_obs))
    
    if len(agentview_obs) == 0:
        print("[warning] No observations to save")
        return
    
    if cache_layers is None:
        cache_layers = ["layer_9", "layer_26", "layer_31"]
    
    num_frames = min(len(agentview_obs), len(eye_in_hand_obs))
    fig_ref = {}  # Reused across frames for performance
    out = None
    
    for i in range(num_frames):
        attn = attention_list[i] if i < len(attention_list) else None
        rstate = robot_states[i] if i < len(robot_states) else None
        rwd = rewards[i - 1] if i > 0 and i - 1 < len(rewards) else 0.0
        
        frame, fig_ref = render_attention_video_frame(
            agentview_image=agentview_obs[i],
            eye_in_hand_image=eye_in_hand_obs[i],
            task_description=task_description,
            step=i,
            reward=float(rwd),
            robot_state=rstate,
            attention_weights=attn,
            cache_layers=cache_layers,
            vlm_cache_length=256,
            fig_ref=fig_ref,
        )
        
        # Initialize video writer on first frame (to get correct dimensions)
        if out is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    if out is not None:
        out.release()
    
    # Close the offscreen figure
    if 'fig' in fig_ref:
        import matplotlib.pyplot as plt
        plt.close(fig_ref['fig'])
    
    print(f"  ✓ Attention video saved: {output_path} ({num_frames} frames at {fps} fps)")


def sample_actions_from_flow(
    params,
    apply_fn,
    cache_k,
    cache_v,
    cache_mask,
    robot_state,
    action_shape: int = 7,
    action_horizon: int = 16,
    num_steps: int = 10,
    dt: float = None,  # Now optional - will be computed from num_steps if not provided
    rng_key=None
) -> jnp.ndarray:
    """
    Sample actions from the flow model using iterative refinement.
    
    Args:
        params: Flow model parameters
        apply_fn: JIT-compiled apply function
        cache_k: KV cache keys from vision model
        cache_v: KV cache values from vision model
        cache_mask: Attention mask for cache
        robot_state: Current robot state (9D)
        action_shape: Dimension of action space
        action_horizon: Length of action sequence
        num_steps: Number of diffusion steps
        dt: Time step for diffusion (optional - computed as 1.0/num_steps if not provided)
        rng_key: Random key for initialization
        
    Returns:
        Tuple of (sampled_actions, attention_weights_dict)
        - sampled_actions: shape (batch_size, action_horizon, action_shape)
        - attention_weights_dict: dict mapping step indices (0, 4, 9) to attention weights
    """
    batch_size = cache_k.shape[0]
    
    # Compute dt from num_steps if not provided (this matches training behavior)
    if dt is None:
        dt = 1.0 / num_steps
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Initialize with random noise
    x = jax.random.normal(rng_key, (batch_size, action_horizon, action_shape))
    
    # Define iteration function for diffusion process
    def iter_fn(carry, step_idx):
        params, x, cache_mask, cache_k, cache_v, time_step, dt, state = carry
        # Get velocity prediction and attention weights from flow model
        output, attn_weights = apply_fn(params, x, cache_mask, cache_k, cache_v, time_step, state)
        # Update x by moving along the velocity (refinement step)
        x = x + output * dt
        # Update time step
        carry = (params, x, cache_mask, cache_k, cache_v, time_step - dt, dt, state)
        return carry, (output, attn_weights)  # Return output and attention for debugging/visualization
    
    # Run diffusion loop
    time_steps = jnp.ones((batch_size,))
    (params, x_final, _, _, _, _, _, _), (outputs, all_attn_weights) = jax.lax.scan(
        iter_fn,
        (params, x, cache_mask, cache_k, cache_v, time_steps, dt, robot_state),
        jnp.arange(num_steps),
        length=num_steps
    )
    
    # outputs is now shape (num_steps, batch, action_horizon, action_shape)
    # all_attn_weights is shape (num_steps, batch, num_layers, num_heads, query_len, key_len)
    # For visualization, extract attention from steps: 0 (1st), 4 (5th), 9 (10th)
    steps_to_visualize = [0, 4, 9]  # 1st, 5th, 10th steps
    attention_dict = {}
    for step_idx in steps_to_visualize:
        if step_idx < num_steps:
            attention_dict[step_idx] = all_attn_weights[step_idx]
    
    return x_final, attention_dict


def extract_libero_action(action_array: np.ndarray, action_idx: int = 0) -> Dict:
    """
    Convert action array to interpretable LIBERO action format.
    
    Args:
        action_array: Action from flow model (shape: [batch, horizon, 7])
                      Format: [ee_x, ee_y, ee_z, qx, qy, qz, gripper]
        action_idx: Which action in the sequence to use
        
    Returns:
        Dictionary with LIBERO action components
    """
    action = action_array[action_idx]
    
    # Parse action components
    ee_x = float(action[0])
    ee_y = float(action[1])
    ee_z = float(action[2])
    qx = float(action[3])
    qy = float(action[4])
    qz = float(action[5])
    gripper = float(action[6])
    
    action_dict = {
        'ee_pos': np.array([ee_x, ee_y, ee_z]),
        'ee_ori_partial': np.array([qx, qy, qz]),  # Partial quaternion (w is implicit)
        'gripper': 1.0 if gripper > 0.5 else 0.0,  # Binary: 0=open, 1=closed
    }
    
    return action_dict


def test_policy_in_environment_closed_loop(
    model_flow,
    params_flow,
    apply_fn,
    action_shape: int = 7,
    action_horizon: int = 16,
    num_diffusion_steps: int = 10,
    sampler=None,
    task_instruction: str = "",
    num_replans: int = 3,
    steps_per_replan: int = 16,
    visualizer: Optional[RealtimeAgentVisualizerDual] = None,
    enable_visualization: bool = False,
    config: Dict = None
) -> Dict:
    """
    Test the policy in the LIBERO environment with closed-loop control.
    
    Re-evaluates the observation through the VLM at each replan step to generate new actions.
    This creates a feedback loop where the policy adapts to the current state.
    
    Uses subprocess communication with libero_worker.py for environment interaction.
    
    Args:
        model_flow: Trained flow model
        params_flow: Flow model parameters
        apply_fn: JIT-compiled apply function
        action_shape: Dimension of action space
        action_horizon: Length of action sequence
        num_diffusion_steps: Number of diffusion steps for sampling
        dt: Time step for diffusion
        sampler: Vision model sampler (optional)
        task_instruction: Task instruction text
        num_replans: Number of times to replan during the episode
        steps_per_replan: Number of steps to execute per replan cycle
        visualizer: Optional RealtimeAgentVisualizerDual instance
        enable_visualization: Whether to create and show visualization
        
    Returns:
        Dictionary with trajectory information and observations
    """
    print(f"\n[info] Testing policy with closed-loop control ({num_replans} replans, {steps_per_replan} steps per replan)")
    
    # Use task information from LIBERO worker if not provided explicitly
    global libero_task_name, libero_task_description
    effective_task_instruction = libero_task_description
    
    # Initialize visualization if requested
    if enable_visualization and visualizer is None:
        print("[info] Creating real-time visualizer with dual camera views...")
        # Get number of cache layers from config
        num_cache_layers = len(config["CACHE_LAYERS"]) if config else 3
        visualizer = RealtimeAgentVisualizerDual(
            image_height=128,
            image_width=128,
            show_action_history=True,
            max_history=10,
            show_attention=True,  # Enable attention visualization
            num_layers=num_cache_layers,  # Use config's number of layers
            cache_layers=config["CACHE_LAYERS"] if config else None  # Pass layer names
        )
        visualizer.show(block=False)
    
    # Get initial observation from LIBERO worker
    agentview, eye_in_hand, robot_state = libero_reset()
    
    # Show initial observation
    if visualizer is not None:
        visualizer.update(
            agentview_image=agentview,
            eye_in_hand_image=eye_in_hand,
            robot_state=robot_state,
            task_info=effective_task_instruction,
            step=0,
            diffusion_step_attentions=None
        )
        visualizer.refresh(pause_time=0.01)  # Force immediate display
    
    trajectory = {
        'agentview_observations': [agentview],
        'eye_in_hand_observations': [eye_in_hand],
        'robot_states': [robot_state],
        'actions': [],
        'rewards': [],
        'dones': [],
        'infos': [],
        'replans': [],
        'attention_weights': [None],  # per-step attention dicts (None for initial obs)
    }
    
    total_steps = 0
    rng = jax.random.PRNGKey(0)
    # Execute multiple replan cycles
    for replan_idx in range(num_replans):
        print(f"\n  [Replan {replan_idx + 1}/{num_replans}]")
        
        # Get current observation and feed to VLM
        current_agentview = agentview
        current_eye_in_hand = eye_in_hand
        current_robot_state = robot_state
        
        # Initialize variables for prompt and response
        vlm_prompt = None
        vlm_response = None
        

        print(f"    Processing current observations through VLM...")
        rng,_rng = jax.random.split(rng)
        
        # Stack both camera views: (batch, 2, H, W, 3)
        # First camera: agentview, Second camera: eye_in_hand
        agentview_arr = current_agentview
        eye_in_hand_arr = current_eye_in_hand
        
        # Ensure images are in correct format
        if len(agentview_arr.shape) == 2:
            agentview_arr = np.stack([agentview_arr] * 3, axis=-1)
        if len(eye_in_hand_arr.shape) == 2:
            eye_in_hand_arr = np.stack([eye_in_hand_arr] * 3, axis=-1)
        
        # Stack: (1, 2, 128, 128, 3)
        images_stacked = np.stack([agentview_arr, eye_in_hand_arr], axis=0)[None, ...]
        images_stacked = jnp.array(images_stacked)
        
        # Generate prompt with task instruction from worker (matching libero_flow_state_grad_accum_multifile.py)
        task_prompt = f'You are a robotic arm. Task: {effective_task_instruction} Agent view: <start_of_image> Eye in hand view: <start_of_image>. Give detailed subtasks to complete the task.'
        
        # Store prompt without image tokens for display
        vlm_prompt = f'You are a robotic arm. Task: {effective_task_instruction}. Give detailed subtasks to complete the task.'
        
        # Sample from vision model based on config
        if config["USE_SAMPLE_WITH_STATE"]:
            # Use sampler.sample() with return_state=True to get full output state
            out = sampler.sample(
                task_prompt,
                images=images_stacked,
                max_new_tokens=100,
                rng=_rng,
                return_state=True,
            )
            cache = out.state.cache
            vlm_response = out.text[0] if isinstance(out.text, (list, tuple)) else str(out.text)
        else:
            # Use sampler.get_cache_prompt() for direct cache retrieval (if available)
            cache = sampler.get_cache_prompt(
                task_prompt,
                images=images_stacked,
                rng=_rng,
            )
            vlm_response = ""  # No text output when using get_cache_prompt
        
        print(f"    VLM output: {vlm_response[:80] if vlm_response else '(using get_cache_prompt)'}...")
        
        # Extract KV cache from vision model using configured layers
        cache_k = extract_cache_from_layers(cache, config["CACHE_LAYERS"], "k")
        cache_v = extract_cache_from_layers(cache, config["CACHE_LAYERS"], "v")
        
        # Create cache mask using first configured layer
        first_layer = config["CACHE_LAYERS"][0]
        cache_mask = create_cache_mask(
            cache[first_layer]["end_index"],
            cache_k.shape[2],
            action_horizon + 1,  # +1 for state token
            cache_k.shape[3]
        )
                
         
        # Convert robot state to JAX array
        robot_state_jax = jnp.array(current_robot_state[None, ...])  # Add batch dimension
        
        # Sample new actions from flow model
        print(f"    Sampling new action sequence from flow model...")
        try:
            rng,_rng = jax.random.split(rng)
            action_sequence, attention_weights_dict = sample_actions_from_flow(
                params_flow,
                apply_fn,
                cache_k,
                cache_v,
                cache_mask,
                robot_state_jax,
                action_shape=action_shape,
                action_horizon=action_horizon,
                num_steps=num_diffusion_steps,
                rng_key=_rng
            )
        except Exception as e:
            print(f"    Error sampling actions: {e}")
            import traceback
            traceback.print_exc()
            action_sequence = jnp.zeros((1, action_horizon, action_shape))
            attention_weights_dict = None

        # Execute the sampled action sequence
        for action_idx in range(steps_per_replan):
            if action_idx < action_sequence.shape[1]:
                action = action_sequence[0, action_idx, :]
                # Clip actions to [-1, 1] range (same as training data normalization)
                action = np.clip(action, -1, 1)
                action = action.tolist()
            else:
                action = [0.] * action_shape
            
            # Execute action in environment via LIBERO worker
            agentview, eye_in_hand, robot_state, reward, done, info = libero_step(action)
            
            trajectory['agentview_observations'].append(agentview)
            trajectory['eye_in_hand_observations'].append(eye_in_hand)
            trajectory['robot_states'].append(robot_state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            trajectory['infos'].append(info)
            # Store attention weights for this step (convert JAX arrays to numpy for video)
            if attention_weights_dict is not None:
                attn_np = {k: np.array(v) for k, v in attention_weights_dict.items()}
            else:
                attn_np = None
            trajectory['attention_weights'].append(attn_np)
            
            total_steps += 1
            print(attention_weights_dict.keys() if attention_weights_dict is not None else "No attention weights")
            # Update visualization
            if visualizer is not None:
                visualizer.update(
                    agentview_image=agentview,
                    eye_in_hand_image=eye_in_hand,
                    robot_state=robot_state,
                    action=np.array(action),
                    reward=reward,
                    done=done,
                    task_info=effective_task_instruction,
                    step=total_steps,
                    diffusion_step_attentions=attention_weights_dict if 'attention_weights_dict' in locals() else None,
                    vlm_prompt=vlm_prompt if 'vlm_prompt' in locals() else None,
                    vlm_response=vlm_response if 'vlm_response' in locals() else None
                )
                visualizer.refresh(pause_time=0.01)  # Force immediate display update
            
            print(f"    Step {total_steps}: action={[f'{a:.3f}' for a in action[:3]]}, reward={reward:.4f}, done={done}")
            
            if done:
                print(f"[info] Episode finished at step {total_steps}")
                break
        
        trajectory['replans'].append({
            'replan_idx': replan_idx,
            'steps_executed': action_idx + 1,
            'total_steps': total_steps
        })
        
        if done:
            break
    
    return trajectory


def main():
    """Main test loop."""
    
    print("=" * 80)
    print("LIBERO Spatial Task - State-Based Flow Model Test with Dual Camera Views")
    print("=" * 80)
    
    # Parse command-line arguments
    import sys
    checkpoint_dir = None
    task_id = 1  # Default task ID
    num_diffusion_steps_override = None  # Will override config if provided
    
    # Parse arguments: can be checkpoint_dir and/or task_id and/or diffusion_steps
    # Usage: python script.py [checkpoint_dir] [--task-id <id>] [--diffusion-steps <n>]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--task-id" and i + 1 < len(sys.argv):
            try:
                task_id = int(sys.argv[i + 1])
                i += 2
                print(f"[info] Task ID set to: {task_id}")
            except ValueError:
                print(f"[warning] Invalid task ID '{sys.argv[i + 1]}', using default task_id=1")
                i += 2
        elif arg == "--diffusion-steps" and i + 1 < len(sys.argv):
            try:
                num_diffusion_steps_override = int(sys.argv[i + 1])
                i += 2
                print(f"[info] Number of diffusion steps set to: {num_diffusion_steps_override}")
            except ValueError:
                print(f"[warning] Invalid diffusion steps '{sys.argv[i + 1]}', will use config default")
                i += 2
        elif not arg.startswith("--"):
            # Assume it's the checkpoint directory
            checkpoint_dir = arg
            print(f"[info] Loading configuration from checkpoint directory: {checkpoint_dir}")
            i += 1
        else:
            i += 1
    
    # Try to load config from checkpoint directory if provided
    loaded_config = None
    if checkpoint_dir:
        loaded_config = load_config_from_checkpoint_dir(checkpoint_dir)
    
    # Configuration - use loaded config if available, otherwise use defaults
    if loaded_config:
        config = loaded_config
        print("[info] Using configuration from checkpoint directory")
    else:
        config = {
            "action_shape": 7,  # LIBERO has 7D actions
            "action_horizon": 16,  # State-based model uses 16 (not 32)
            "num_diffusion_steps": 20,
            "CACHE_LAYERS": ["layer_9", "layer_26", "layer_31"],  # Which Gemma layers to extract for cache
            "USE_SAMPLE_WITH_STATE": True,  # If True, use sampler.sample() with return_state=True; if False, use sampler.get_cache_prompt()
            "task_suite_name": "libero_spatial"
        }
        print("[info] Using default configuration")
    
    action_shape = config["action_shape"]
    action_horizon = config["action_horizon"]
    num_diffusion_steps = config["num_diffusion_steps"]
    
    # Override num_diffusion_steps if provided via command-line
    if num_diffusion_steps_override is not None:
        num_diffusion_steps = num_diffusion_steps_override
        print(f"[info] Overriding config: using {num_diffusion_steps} diffusion steps from command-line")
    
    task_suite_name = config["task_suite_name"]
    num_cache_layers = len(config["CACHE_LAYERS"])
    
    # Step 1: Start LIBERO worker subprocess
    print("\n[1/6] Starting LIBERO worker subprocess...")
    
    libero_worker_started = False
    try:
        start_libero_worker(task_id=task_id)
        libero_worker_started = True
        print(f"  ✓ LIBERO worker started successfully with task_id={task_id}")
        print("[info] Waiting for worker ready signal...")
        msg = _read_json_from_worker()
        print("[info] Worker ready:", msg)
        
        # Extract task information from worker
        global libero_task_name, libero_task_description
        libero_task_name = msg.get("task_name", "Unknown Task")
        libero_task_description = msg.get("task_description", "")
        print(f"[info] Task name: {libero_task_name}")
        print(f"[info] Task description: {libero_task_description}")
        
        # Test reset to verify dual camera observations
        libero_process.stdin.write(json.dumps({"cmd": "reset"}) + "\n")
        libero_process.stdin.flush()
        obs = _read_json_from_worker()
        print(f"[info] Observation keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")
        
    except Exception as e:
        print(f"  Error starting LIBERO worker: {e}")
        import traceback
        traceback.print_exc()
        print("  Cannot proceed without LIBERO worker")
        return
    
    # Step 2: Load flow model
    print("\n[2/6] Loading trained state-based flow model...")
    
    # Determine checkpoint location
    flow_checkpoint = None
    restored_params = None
    restored_meta = None
    if checkpoint_dir:
        restored_params, restored_meta = try_restore_params_from_orbax(checkpoint_dir)
        if restored_params is not None:
            # Initialize model architecture and set params
            model_flow = TransformerFlow(
                num_layers=num_cache_layers,
                num_heads=4,
                qkv_features=1024,
                out_features=512,
                input_size=action_shape,
                gating=True,
                gating_bias=2.,
                norm_type="rmsnorm",
                post_attention_norm=True,
                post_mlp_norm=True
            )
            apply_fn = jax.jit(model_flow.apply)
            params_flow = restored_params
            print(f"  ✓ Restored flow model params from Orbax checkpoint")
        else:
            # Look for flow_model_final.npy or flow_model_f_*.npy in checkpoint directory
            final_model_in_dir = os.path.join(checkpoint_dir, "flow_model_kvcache_batchedaa_shifted_final.npy")
            if os.path.exists(final_model_in_dir):
                flow_checkpoint = final_model_in_dir
                print(f"  Found final model in checkpoint directory: {flow_checkpoint}")
            else:
                # Look for any flow_model_f_*.npy in the directory
                for i in range(400, 0, -1):
                    alt_in_dir = os.path.join(checkpoint_dir, f"flow_model_kvcache_batched_shifted_{i}.npy")
                    if os.path.exists(alt_in_dir):
                        flow_checkpoint = alt_in_dir
                        print(f"  Found model in checkpoint directory: {flow_checkpoint}")
                        break
    
    # Fall back to default locations if not found in checkpoint directory
    if restored_params is None and not flow_checkpoint:
        flow_checkpoint = "flow_model_f_9.npy"
        
        # Try alternative checkpoints if final doesn't exist
        if not os.path.exists(flow_checkpoint):
            for i in range(400, 0, -1):
                alt_checkpoint = f"LIBERO/flow_model_state_{i}.npy"
                if os.path.exists(alt_checkpoint):
                    flow_checkpoint = alt_checkpoint
                    print(f"  Final model not found, using {flow_checkpoint} instead")
                    break
    
    if restored_params is None and (not flow_checkpoint or not os.path.exists(flow_checkpoint)):
        print(f"Error: No state-based flow model checkpoint found!")
        print(f"  Looked in: {checkpoint_dir if checkpoint_dir else 'current directory'}")
        return
    
    if restored_params is None:
        try:
            model_flow, params_flow, apply_fn = load_flow_model(
                flow_checkpoint,
                action_shape=action_shape,
                action_horizon=action_horizon,
                num_layers=num_cache_layers
            )
            print(f"  ✓ Loaded flow model from {flow_checkpoint}")
            
            # Count parameters
            def count_params(params):
                return sum([np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)])
            
            num_params = count_params(params_flow)
            print(f"  Model parameters: {num_params:,}")
            print(f"  Model cache layers: {num_cache_layers}, action_horizon={action_horizon}")
            
        except Exception as e:
            print(f"Error loading flow model: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Step 3: Load vision model (Gemma3)
    print("\n[3/6] Loading vision model (Gemma3)...")
    sampler = None
    
    if GEMMA_AVAILABLE:
        try:
            model_vision = gm.nn.IntWrapper(model=gm.nn.Gemma3_4B(), dtype=jnp.int4)
            original_params = gm.ckpts.load_params(
                os.path.abspath(gemma_path + "/gemma3-4b-it")
            )
            params_vision = peft.quantize(original_params, method='INT4', checkpoint_kernel_key='w')
            
            del original_params
            gc.collect()
            jax.clear_caches()
            
            # Load tokenizer
            print("  Loading tokenizer...")
            tokenizer = gm.text.Gemma3Tokenizer(
                os.path.abspath(gemma_path + "/tokenizer.model")
            )
            
            # Initialize sampler
            sampler = Sampler(
                model=model_vision,
                params=params_vision,
                tokenizer=tokenizer,
                cache_length=256,
                max_out_length=100,
            )
            print("  ✓ Vision model loaded and sampler initialized")
            
        except Exception as e:
            print(f"Error loading vision model: {e}")
            print("  Proceeding with dummy cache (for testing model architecture only)...")
            sampler = None
    else:
        print("  Gemma module not available. Using dummy cache for testing...")
    
    # Step 4: Test policy in environment with LIBERO worker
    print("\n[4/6] Testing policy in LIBERO environment with closed-loop control...")
    
    trajectory = None
    if libero_worker_started:
        try:
            trajectory = test_policy_in_environment_closed_loop(
                model_flow,
                params_flow,
                apply_fn,
                action_shape=action_shape,
                action_horizon=action_horizon,
                num_diffusion_steps=num_diffusion_steps,
                sampler=sampler,
                task_instruction=libero_task_description,
                num_replans=24,
                steps_per_replan=16,
                enable_visualization=True,  # Enable real-time visualization with dual cameras
                config=config  # Pass config to the test function
            )
            print(f"  ✓ Successfully executed {len(trajectory['agentview_observations'])} steps in environment")
            
            # Calculate and print trajectory statistics
            rewards = np.array(trajectory['rewards'])
            print(f"  Episode reward: {rewards.sum():.4f}")
            print(f"  Average reward per step: {rewards.mean():.4f}")
            
            # Step 5: Save trajectory videos
            print("\n[5/6] Saving trajectory videos...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create trajectories/ directory
            trajectories_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")
            os.makedirs(trajectories_dir, exist_ok=True)
            
            # Save simple dual-view video
            video_filename = f"trajectory_task{task_id}_{timestamp}.mp4"
            video_path = os.path.join(trajectories_dir, video_filename)
            try:
                save_trajectory_video(
                    trajectory=trajectory,
                    task_description=libero_task_description,
                    output_path=video_path,
                    fps=10
                )
            except Exception as video_error:
                print(f"  Error saving dual-view video: {video_error}")
                import traceback
                traceback.print_exc()
            
            # Save attention visualization video
            attn_video_filename = f"trajectory_attention_task{task_id}_{timestamp}.mp4"
            attn_video_path = os.path.join(trajectories_dir, attn_video_filename)
            try:
                save_trajectory_video_with_attention(
                    trajectory=trajectory,
                    task_description=libero_task_description,
                    output_path=attn_video_path,
                    cache_layers=config.get("CACHE_LAYERS", None),
                    fps=10
                )
            except Exception as video_error:
                print(f"  Error saving attention video: {video_error}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"Error testing policy in environment: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Stop LIBERO worker
            if libero_worker_started:
                stop_libero_worker()
                print("  LIBERO worker closed")
    else:
        print("  Skipping environment testing (LIBERO worker not available)")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Task suite: {task_suite_name}")
    print(f"Task: {libero_task_name}")
    print(f"Generated action horizon: {action_horizon} steps")
    print(f"Model architecture: 3 layers (state-based)")
    
    if trajectory is not None:
        print(f"Environment steps executed: {len(trajectory['agentview_observations'])}")
        print(f"Number of replans: {len(trajectory['replans'])}")
        if trajectory['replans']:
            for replan in trajectory['replans']:
                print(f"  Replan {replan['replan_idx'] + 1}: {replan['steps_executed']} steps (total: {replan['total_steps']})")
        
        rewards = np.array(trajectory['rewards'])
        print(f"Total episode reward: {rewards.sum():.4f}")
        print(f"Average reward per step: {rewards.mean():.4f}")
    else:
        print(f"Environment steps executed: 0")
    
    print(f"Diffusion steps per action sequence: {num_diffusion_steps}")
    
    print("\n✓ Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Usage:
    #   python test_libero_flow_state_server_vis.py                                                # Use default config, task_id=1
    #   python test_libero_flow_state_server_vis.py --task-id 3                                    # Use task_id=3 with default config
    #   python test_libero_flow_state_server_vis.py --diffusion-steps 50                           # Use 50 diffusion steps
    #   python test_libero_flow_state_server_vis.py --task-id 3 --diffusion-steps 50               # Combine both options
    #   python test_libero_flow_state_server_vis.py run_20260212_120000                            # Load config from checkpoint, task_id=1
    #   python test_libero_flow_state_server_vis.py run_20260212_120000 --task-id 5                # Load config and use task_id=5
    #   python test_libero_flow_state_server_vis.py run_20260212_120000 --diffusion-steps 30       # Override config's diffusion steps
    #   python test_libero_flow_state_server_vis.py run_20260212_120000 --task-id 5 --diffusion-steps 30  # All options combined
    
    main()
