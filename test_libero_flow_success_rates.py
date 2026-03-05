"""
Test script for calculating success rate of the state-based flow model across ALL task suites.
Iterates over each task suite one after the other, running all tasks within each suite.
Computes per-task, per-suite, and overall average success rates.
All videos and results are saved in a single output folder.

Task suites iterated (by default):
- libero_spatial (10 tasks)
- libero_object (10 tasks)
- libero_goal (10 tasks)

Usage:
  # Run all 3 default suites (spatial, object, goal), 10 trials per task
  python test_libero_flow_state_server_success_rate_all_suites.py

  # Run all suites with a specific checkpoint directory
  python test_libero_flow_state_server_success_rate_all_suites.py run_20260212_120000

  # Override which suites to run (comma-separated)
  python test_libero_flow_state_server_success_rate_all_suites.py --suites libero_spatial,libero_object

  # Change number of trials per task
  python test_libero_flow_state_server_success_rate_all_suites.py --num-trials 20

  # Change diffusion steps
  python test_libero_flow_state_server_success_rate_all_suites.py --diffusion-steps 50

  # Full example
  python test_libero_flow_state_server_success_rate_all_suites.py run_20260212_120000 --suites libero_spatial,libero_object,libero_goal --num-trials 10 --diffusion-steps 20

Key differences from test_libero_flow_state_server_success_rate.py:
- Iterates over multiple task suites automatically
- Runs ALL tasks (0-9) within each suite
- Restarts the LIBERO worker for each task
- Computes per-suite average success rate
- Computes overall average success rate across all suites
- Saves everything (videos, summaries) in a single output folder
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import gc
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, Tuple, Optional, List
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


from networks.flow_network_state_rope_attention import TransformerFlow
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

gemma_path="/home/reytuag/VLA/gemma-3-flax-gemma3-4b-it-v1"
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


def start_libero_worker(task_id: int = 1, task_suite_name: str = "libero_spatial"):
    """
    Start the LIBERO worker subprocess.
    
    Args:
        task_id: Index of the task to use (default: 1)
        task_suite_name: Name of the task suite, e.g. "libero_spatial" or "libero_object" (default: "libero_spatial")
        
    Returns:
        Popen object for the worker process
    """
    global libero_process
    
    print(f"[info] Starting LIBERO worker process with task_id={task_id}, task_suite='{task_suite_name}'...")
    try:
        libero_process = subprocess.Popen(
            ["/home/reytuag/miniconda3/envs/libero_env/bin/python", "-u", "libero_misc/libero_worker.py", str(task_id), task_suite_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Important: use text mode
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        print(f"[info] LIBERO worker process started successfully with task_id={task_id}, task_suite='{task_suite_name}'")
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


def libero_step_batch(actions):
    """
    Send a batch of actions to the LIBERO worker in a single IPC call.

    The worker executes all actions sequentially and returns only the final
    observation together with per-step rewards / dones.  This avoids one
    JSON round-trip per timestep and dramatically reduces IPC overhead.

    Args:
        actions: List of actions (each a list/numpy array of length action_dim)

    Returns:
        Tuple of (agentview, eye_in_hand, robot_state, rewards, dones, info, steps_executed)
        - rewards: list of float, one per executed step
        - dones: list of bool, one per executed step
        - steps_executed: int, may be < len(actions) if done early
    """
    global libero_process

    if libero_process is None:
        raise RuntimeError("LIBERO worker not started. Call start_libero_worker() first.")

    # Convert numpy arrays to plain lists for JSON serialisation
    action_lists = [
        a.tolist() if isinstance(a, np.ndarray) else list(a) for a in actions
    ]

    try:
        libero_process.stdin.write(
            json.dumps({"cmd": "step_batch", "actions": action_lists}) + "\n"
        )
        libero_process.stdin.flush()

        result = _read_json_from_worker()

        # ---- parse final observation (same logic as libero_step) ----
        obs_dict = result.get("obs", {})
        if isinstance(obs_dict, dict):
            if "agentview_image" in obs_dict:
                agentview = np.array(obs_dict["agentview_image"], dtype=np.uint8)[::-1]
                if len(agentview.shape) == 1:
                    agentview = agentview.reshape(128, 128, 3)
            else:
                agentview = np.zeros((128, 128, 3), dtype=np.uint8)

            if "robot0_eye_in_hand_image" in obs_dict:
                eye_in_hand = np.array(obs_dict["robot0_eye_in_hand_image"], dtype=np.uint8)
                if len(eye_in_hand.shape) == 1:
                    eye_in_hand = eye_in_hand.reshape(128, 128, 3)
            else:
                eye_in_hand = np.zeros((128, 128, 3), dtype=np.uint8)

            if (
                "robot0_gripper_qpos" in obs_dict
                and "robot0_eef_pos" in obs_dict
                and "robot0_eef_quat" in obs_dict
            ):
                robot_state = np.concatenate([
                    np.array(obs_dict["robot0_gripper_qpos"], dtype=np.float32),
                    np.array(obs_dict["robot0_eef_pos"], dtype=np.float32),
                    np.array(obs_dict["robot0_eef_quat"], dtype=np.float32),
                ])
            elif "robot0_gripper_qpos" in obs_dict and "robot0_eef_pos" in obs_dict:
                robot_state = np.concatenate([
                    np.array(obs_dict["robot0_gripper_qpos"], dtype=np.float32),
                    np.array(obs_dict["robot0_eef_pos"], dtype=np.float32),
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                ])
            else:
                robot_state = np.zeros(9, dtype=np.float32)
        else:
            agentview = np.zeros((128, 128, 3), dtype=np.uint8)
            eye_in_hand = np.zeros((128, 128, 3), dtype=np.uint8)
            robot_state = np.zeros(9, dtype=np.float32)

        rewards = result.get("rewards", [])
        dones = result.get("dones", [])
        info = result.get("info", {})
        steps_executed = int(result.get("steps_executed", len(rewards)))

        return agentview, eye_in_hand, robot_state, rewards, dones, info, steps_executed

    except Exception as e:
        print(f"[error] Error in libero_step_batch: {e}")
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
                       task_description: str, step: int, trial_num: int = None,
                       suite_name: str = None, task_id: int = None) -> np.ndarray:
    """
    Create a video frame with concatenated dual-view images and task descriptor on top.
    
    Args:
        agentview_image: RGB image from agentview camera (H, W, 3)
        eye_in_hand_image: RGB image from eye-in-hand camera (H, W, 3)
        task_description: Task description text to display
        step: Current step number
        trial_num: Trial number (optional, for multi-trial testing)
        suite_name: Task suite name (optional)
        task_id: Task ID (optional)
        
    Returns:
        Combined frame as numpy array (H_total, W, 3) with uint8 dtype
    """
    # Ensure images are uint8
    agentview = agentview_image.astype(np.uint8)
    eye_in_hand = eye_in_hand_image.astype(np.uint8)
    
    # Get dimensions
    h, w, c = agentview.shape
    
    # Create text header using PIL for better text rendering
    text_height = 60
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
    
    # Draw task description and step number
    task_text = f"Task: {task_description}"
    parts = []
    if suite_name is not None:
        parts.append(f"Suite: {suite_name}")
    if task_id is not None:
        parts.append(f"Task {task_id}")
    if trial_num is not None:
        parts.append(f"Trial {trial_num}")
    parts.append(f"Step: {step}")
    step_text = " | ".join(parts)
    
    draw.text((10, 10), task_text, fill=(255, 255, 255), font=font_task)
    draw.text((10, 35), step_text, fill=(200, 200, 200), font=font_step)
    
    # Convert header to numpy
    header_array = np.array(header_img)
    
    # Concatenate images horizontally
    dual_view = np.concatenate([agentview, eye_in_hand], axis=1)
    
    # Stack header on top of dual view
    final_frame = np.concatenate([header_array, dual_view], axis=0)
    
    return final_frame


def save_trajectory_video(trajectory: Dict, task_description: str, output_path: str, fps: int = 10,
                          trial_num: int = None, suite_name: str = None, task_id: int = None):
    """
    Save trajectory as a video with concatenated dual-view images and task descriptor.
    
    Args:
        trajectory: Dictionary containing 'agentview_observations' and 'eye_in_hand_observations'
        task_description: Task description to display in video
        output_path: Path to save the video file
        fps: Frames per second for the video
        trial_num: Trial number (optional, for multi-trial testing)
        suite_name: Task suite name (optional)
        task_id: Task ID (optional)
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
    first_frame = create_video_frame(agentview_obs[0], eye_in_hand_obs[0], task_description, 0,
                                     trial_num, suite_name, task_id)
    height, width, _ = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write all frames
    num_frames = min(len(agentview_obs), len(eye_in_hand_obs))
    for i in range(num_frames):
        frame = create_video_frame(agentview_obs[i], eye_in_hand_obs[i], task_description, i,
                                   trial_num, suite_name, task_id)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"  ✓ Video saved: {output_path} ({num_frames} frames at {fps} fps)")


import functools


@functools.partial(jax.jit, static_argnums=(1, 6, 7, 8))
def _sample_actions_jit(
    params,
    apply_fn,
    cache_k,
    cache_v,
    cache_mask,
    robot_state,
    action_shape: int,
    action_horizon: int,
    num_steps: int,
    rng_key,
):
    """
    JIT-compiled core of action sampling from the flow model.

    Args:
        params: Flow model parameters
        apply_fn: Model apply function (static)
        cache_k, cache_v, cache_mask: KV cache from VLM
        robot_state: Current robot state
        action_shape, action_horizon, num_steps: Static ints
        rng_key: JAX PRNG key

    Returns:
        Sampled action sequence (batch_size, action_horizon, action_shape)
    """
    batch_size = cache_k.shape[0]
    dt = 1.0 / num_steps

    # Initialize with random noise
    x = jax.random.normal(rng_key, (batch_size, action_horizon, action_shape))*0.01

    # Iteration function for the diffusion process
    def iter_fn(carry, step_idx):
        params, x, cache_mask, cache_k, cache_v, time_step, dt, state = carry
        output, _attn_weights = apply_fn(params, x, cache_mask, cache_k, cache_v, time_step, state)
        x = x + output * dt
        carry = (params, x, cache_mask, cache_k, cache_v, time_step - dt, dt, state)
        return carry, None

    time_steps = jnp.ones((batch_size,))
    (_, x_final, _, _, _, _, _, _), _ = jax.lax.scan(
        iter_fn,
        (params, x, cache_mask, cache_k, cache_v, time_steps, dt, robot_state),
        jnp.arange(num_steps),
        length=num_steps,
    )

    return x_final


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
    dt: float = None,  # Kept for API compat; ignored (computed inside JIT)
    rng_key=None
) -> jnp.ndarray:
    """
    Sample actions from the flow model using iterative refinement (JIT-compiled).
    
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
        dt: (unused, kept for API compatibility)
        rng_key: Random key for initialization
        
    Returns:
        Sampled actions with shape (batch_size, action_horizon, action_shape)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    x_final = _sample_actions_jit(
        params,
        apply_fn,
        cache_k,
        cache_v,
        cache_mask,
        robot_state,
        action_shape,
        action_horizon,
        num_steps,
        rng_key,
    )

    return x_final


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
    config: Dict = None,
    trial_num: int = None,
    trial_rng: Optional[jax.random.PRNGKey] = None
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
        sampler: Vision model sampler (optional)
        task_instruction: Task instruction text
        num_replans: Number of times to replan during the episode
        steps_per_replan: Number of steps to execute per replan cycle
        trial_num: Trial number (optional, for multi-trial testing)
        trial_rng: Random key for this trial
        
    Returns:
        Dictionary with trajectory information and observations
    """
    trial_str = f" (Trial {trial_num})" if trial_num is not None else ""
    print(f"\n[info] Testing policy with closed-loop control{trial_str} ({num_replans} replans, {steps_per_replan} steps per replan)")
    
    # Use task information from LIBERO worker if not provided explicitly
    global libero_task_name, libero_task_description
    effective_task_instruction = libero_task_description
    
    # Get initial observation from LIBERO worker
    agentview, eye_in_hand, robot_state = libero_reset()
    
    trajectory = {
        'agentview_observations': [agentview],
        'eye_in_hand_observations': [eye_in_hand],
        'robot_states': [robot_state],
        'actions': [],
        'rewards': [],
        'dones': [],
        'infos': [],
        'replans': []
    }
    
    total_steps = 0
    rng = trial_rng
    done = False
    # Execute multiple replan cycles
    for replan_idx in range(num_replans):
        #print(f"\n  [Replan {replan_idx + 1}/{num_replans}]")
        
        # Get current observation and feed to VLM
        current_agentview = agentview
        current_eye_in_hand = eye_in_hand
        current_robot_state = robot_state
        
        # Initialize variables for prompt and response
        vlm_prompt = None
        vlm_response = None
        
        if sampler is not None:
            try:
                #print(f"    Processing current observations through VLM...")
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
                
                #print(f"    VLM output: {vlm_response[:80] if vlm_response else '(using get_cache_prompt)'}...")
                
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
                
                # Free the full VLM cache and intermediate objects to reclaim VRAM
                del cache
                if 'out' in dir():
                    del out
                gc.collect()
                
            except Exception as e:
                print(f"    Error processing with VLM: {e}")
                print(f"    Creating dummy cache for this replan...")
                batch_size = 1
                num_cache_layers = len(config["CACHE_LAYERS"])
                cache_k = jnp.ones((batch_size, num_cache_layers, 512, 8, 64))
                cache_v = jnp.ones((batch_size, num_cache_layers, 512, 8, 64))
                cache_mask = jnp.ones((batch_size, 8, action_horizon + 1, 512))
        else:
            # Create dummy cache if no sampler
            batch_size = 1
            cache_k = jnp.ones((batch_size, 3, 512, 8, 64))  # 3 layers
            cache_v = jnp.ones((batch_size, 3, 512, 8, 64))
            cache_mask = jnp.ones((batch_size, 8, action_horizon + 1, 512))
        
        # Convert robot state to JAX array
        robot_state_jax = jnp.array(current_robot_state[None, ...])  # Add batch dimension
        
        # Sample new actions from flow model
        #print(f"    Sampling new action sequence from flow model...")
        try:
            rng,_rng = jax.random.split(rng)
            action_sequence = sample_actions_from_flow(
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

        # Build the list of actions to send in one batch
        actions_to_send = []
        for action_idx in range(steps_per_replan):
            if action_idx < action_sequence.shape[1]:
                action = action_sequence[0, action_idx, :]
                # Clip actions to [-1, 1] range (same as training data normalization)
                action = np.clip(np.asarray(action), -1, 1)
                actions_to_send.append(action)
            else:
                actions_to_send.append(np.zeros(action_shape))

        # Execute the full action chunk in a single IPC call
        agentview, eye_in_hand, robot_state, rewards, dones, info, steps_executed = \
            libero_step_batch(actions_to_send)

        # Record per-step data from the batch result
        for si in range(steps_executed):
            trajectory['actions'].append(actions_to_send[si].tolist())
            trajectory['rewards'].append(rewards[si])
            trajectory['dones'].append(dones[si])
            trajectory['infos'].append(info if si == steps_executed - 1 else {})

        # Only the final observation is returned; append it once
        trajectory['agentview_observations'].append(agentview)
        trajectory['eye_in_hand_observations'].append(eye_in_hand)
        trajectory['robot_states'].append(robot_state)

        total_steps += steps_executed
        done = any(dones)

        if done:
            print(f"[info] Episode finished at step {total_steps}")
        
        trajectory['replans'].append({
            'replan_idx': replan_idx,
            'steps_executed': steps_executed,
            'total_steps': total_steps
        })
        
        if done:
            break
    
    return trajectory


def save_all_suites_summary(all_suite_results: Dict[str, Dict], output_path: str, config: Dict,
                            num_trials: int, timestamp: str):
    """
    Save a comprehensive summary of success rate testing across all suites and tasks.
    
    Args:
        all_suite_results: Dict mapping suite_name -> {task_id -> list of trial results}
        output_path: Path to save the summary file
        config: Configuration dictionary
        num_trials: Number of trials per task
        timestamp: Timestamp string for this run
    """
    print(f"\n[info] Saving comprehensive summary to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("LIBERO Flow Model - ALL SUITES Success Rate Testing Summary\n")
        f.write(f"Date: {timestamp}\n")
        f.write("=" * 100 + "\n\n")
        
        # Configuration
        f.write("Configuration:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Action Shape: {config['action_shape']}\n")
        f.write(f"Action Horizon: {config['action_horizon']}\n")
        f.write(f"Diffusion Steps: {config['num_diffusion_steps']}\n")
        f.write(f"Cache Layers: {config['CACHE_LAYERS']}\n")
        f.write(f"Number of Trials per Task: {num_trials}\n")
        f.write(f"Task Suites Tested: {list(all_suite_results.keys())}\n\n")
        
        # ============================================================
        # Overall summary across all suites
        # ============================================================
        f.write("=" * 100 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        overall_successes = 0
        overall_trials = 0
        suite_success_rates = {}
        
        for suite_name, task_results in all_suite_results.items():
            suite_successes = 0
            suite_trials = 0
            for task_id, results in task_results.items():
                for r in results:
                    suite_trials += 1
                    overall_trials += 1
                    if r['success']:
                        suite_successes += 1
                        overall_successes += 1
            
            suite_rate = (suite_successes / suite_trials * 100) if suite_trials > 0 else 0.0
            suite_success_rates[suite_name] = suite_rate
            f.write(f"  {suite_name:20s}: {suite_rate:6.1f}%  ({suite_successes}/{suite_trials})\n")
        
        overall_rate = (overall_successes / overall_trials * 100) if overall_trials > 0 else 0.0
        f.write(f"\n  {'OVERALL':20s}: {overall_rate:6.1f}%  ({overall_successes}/{overall_trials})\n")
        
        # Average of suite averages
        if suite_success_rates:
            avg_suite_rate = np.mean(list(suite_success_rates.values()))
            f.write(f"  {'Avg across suites':20s}: {avg_suite_rate:6.1f}%\n")
        
        # ============================================================
        # Per-suite detailed results
        # ============================================================
        for suite_name, task_results in all_suite_results.items():
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"SUITE: {suite_name}\n")
            f.write("=" * 100 + "\n\n")
            
            suite_successes = 0
            suite_trials = 0
            suite_rewards = []
            suite_lengths = []
            
            for task_id in sorted(task_results.keys()):
                results = task_results[task_id]
                task_successes = sum(1 for r in results if r['success'])
                task_trials = len(results)
                task_rate = (task_successes / task_trials * 100) if task_trials > 0 else 0.0
                
                suite_successes += task_successes
                suite_trials += task_trials
                
                task_rewards = [r['total_reward'] for r in results]
                task_lengths = [r['episode_length'] for r in results]
                suite_rewards.extend(task_rewards)
                suite_lengths.extend(task_lengths)
                
                task_name = results[0].get('task_name', 'Unknown') if results else 'Unknown'
                task_desc = results[0].get('task_description', '') if results else ''
                
                f.write(f"  Task {task_id}: {task_desc}\n")
                f.write(f"    Name: {task_name}\n")
                f.write(f"    Success Rate: {task_rate:6.1f}%  ({task_successes}/{task_trials})\n")
                f.write(f"    Avg Reward: {np.mean(task_rewards):.4f} ± {np.std(task_rewards):.4f}\n")
                f.write(f"    Avg Episode Length: {np.mean(task_lengths):.1f} ± {np.std(task_lengths):.1f}\n")
                
                # Per-trial details
                for r in results:
                    status = '✓' if r['success'] else '✗'
                    f.write(f"      Trial {r['trial']:2d}: {status}  reward={r['total_reward']:.4f}  steps={r['episode_length']}")
                    if r.get('video_path'):
                        f.write(f"  video={os.path.basename(r['video_path'])}")
                    if r.get('error'):
                        f.write(f"  ERROR: {r['error']}")
                    f.write("\n")
                f.write("\n")
            
            suite_rate = (suite_successes / suite_trials * 100) if suite_trials > 0 else 0.0
            f.write(f"  Suite Summary: {suite_rate:.1f}% ({suite_successes}/{suite_trials})\n")
            if suite_rewards:
                f.write(f"  Average Reward: {np.mean(suite_rewards):.4f} ± {np.std(suite_rewards):.4f}\n")
                f.write(f"  Average Episode Length: {np.mean(suite_lengths):.1f} ± {np.std(suite_lengths):.1f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("End of Summary\n")
        f.write("=" * 100 + "\n")
    
    print(f"  ✓ Summary saved to: {output_path}")


def main():
    """Main test loop for success rate evaluation across all task suites."""
    
    print("=" * 100)
    print("LIBERO Flow Model - ALL SUITES Success Rate Testing")
    print("=" * 100)
    
    # Parse command-line arguments
    import sys
    checkpoint_dir = None
    num_diffusion_steps_override = None
    num_trials = 10  # Default number of trials per task
    num_tasks_per_suite = 10  # LIBERO suites have 10 tasks each (0-9)
    suites_override = None
    
    # Default suites to test
    DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal"]
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--suites" and i + 1 < len(sys.argv):
            valid_suites = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}
            suite_args = sys.argv[i + 1].split(",")
            suites_override = []
            for s in suite_args:
                s = s.strip()
                if s in valid_suites:
                    suites_override.append(s)
                else:
                    print(f"[warning] Unknown task suite '{s}'. Valid options: {sorted(valid_suites)}. Skipping.")
            if not suites_override:
                print(f"[warning] No valid suites provided, using defaults: {DEFAULT_SUITES}")
                suites_override = None
            else:
                print(f"[info] Task suites set to: {suites_override}")
            i += 2
        elif arg == "--diffusion-steps" and i + 1 < len(sys.argv):
            try:
                num_diffusion_steps_override = int(sys.argv[i + 1])
                i += 2
                print(f"[info] Number of diffusion steps set to: {num_diffusion_steps_override}")
            except ValueError:
                print(f"[warning] Invalid diffusion steps '{sys.argv[i + 1]}', will use config default")
                i += 2
        elif arg == "--num-trials" and i + 1 < len(sys.argv):
            try:
                num_trials = int(sys.argv[i + 1])
                i += 2
                print(f"[info] Number of trials per task set to: {num_trials}")
            except ValueError:
                print(f"[warning] Invalid num trials '{sys.argv[i + 1]}', using default num_trials=10")
                i += 2
        elif arg == "--num-tasks" and i + 1 < len(sys.argv):
            try:
                num_tasks_per_suite = int(sys.argv[i + 1])
                i += 2
                print(f"[info] Number of tasks per suite set to: {num_tasks_per_suite}")
            except ValueError:
                print(f"[warning] Invalid num tasks '{sys.argv[i + 1]}', using default=10")
                i += 2
        elif not arg.startswith("--"):
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
            "action_shape": 7,
            "action_horizon": 16,
            "num_diffusion_steps": 20,
            "CACHE_LAYERS": ["layer_9", "layer_26", "layer_31"],
            "USE_SAMPLE_WITH_STATE": True,
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
    
    num_cache_layers = len(config["CACHE_LAYERS"])
    
    # Determine suites to test
    suites_to_test = suites_override if suites_override else DEFAULT_SUITES
    
    # =========================================================================
    # Step 1: Load flow model (only once, shared across all suites/tasks)
    # =========================================================================
    print("\n[1/4] Loading trained state-based flow model...")
    
    flow_checkpoint = None
    # First, try to restore params from Orbax checkpoint if provided
    restored_params = None
    restored_meta = None
    if checkpoint_dir:
        restored_params, restored_meta = try_restore_params_from_orbax(checkpoint_dir)
        if restored_params is not None:
            # Initialize flow model architecture
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
            # Fallback to looking for legacy .npy param files in checkpoint dir
            #print(checkpoint_dir)
            final_model_in_dir = os.path.join(checkpoint_dir, "flow_model_kvcache_batchedaa_shifted_final.npy")
            if os.path.exists(final_model_in_dir):
                flow_checkpoint = final_model_in_dir
                print(f"  Found final model in checkpoint directory: {flow_checkpoint}")
            else:
                for i in range(665, 0, -1):
                    alt_in_dir = os.path.join(checkpoint_dir, f"flow_model_kvcache_batched_shifted_full_{i}.npy")
                    if os.path.exists(alt_in_dir):
                        flow_checkpoint = alt_in_dir
                        print(f"  Found model in checkpoint directory: {flow_checkpoint}")
                        break
    
    # Fall back to default locations if not found in checkpoint directory
    if restored_params is None and not flow_checkpoint:
        flow_checkpoint = "flow_model_f_9.npy"
        if not os.path.exists(flow_checkpoint):
            for i in range(180, 0, -1):
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
    
    # =========================================================================
    # Step 2: Load vision model (only once, shared across all suites/tasks)
    # =========================================================================
    print("\n[2/4] Loading vision model (Gemma3)...")
    sampler = None
    
    if GEMMA_AVAILABLE:
        try:
            model_vision = gm.nn.IntWrapper(model=gm.nn.Gemma3_4B(), dtype=jnp.int4)
            original_params = gm.ckpts.load_params(
                os.path.abspath(gemma_path+"/gemma3-4b-it")
            )
            params_vision = peft.quantize(original_params, method='INT4', checkpoint_kernel_key='w')
            
            del original_params
            gc.collect()
            jax.clear_caches()
            
            print("  Loading tokenizer...")
            tokenizer = gm.text.Gemma3Tokenizer(
                os.path.abspath(gemma_path+"/tokenizer.model")
            )
            
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
    
    # =========================================================================
    # Step 3: Create output directory (single folder for everything)
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = "results"
    output_dir = os.path.join(results_base, f"all_suites_success_rate_diffsteps{num_diffusion_steps}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[3/4] All results will be saved to: {output_dir}/")
    
    # =========================================================================
    # Step 4: Iterate over all suites and all tasks
    # =========================================================================
    print(f"\n[4/4] Running evaluation across {len(suites_to_test)} suites, "
          f"{num_tasks_per_suite} tasks each, {num_trials} trials per task...")
    print(f"       Total trials: {len(suites_to_test) * num_tasks_per_suite * num_trials}")
    
    # Master RNG
    rng = jax.random.PRNGKey(42)
    
    # Store all results: suite_name -> {task_id -> [trial_results]}
    all_suite_results = {}
    
    for suite_idx, suite_name in enumerate(suites_to_test):
        print(f"\n{'#' * 100}")
        print(f"# SUITE {suite_idx + 1}/{len(suites_to_test)}: {suite_name}")
        print(f"{'#' * 100}")
        
        suite_results = {}  # task_id -> [trial_results]
        
        for task_id in range(num_tasks_per_suite):
            print(f"\n{'=' * 80}")
            print(f"  Suite: {suite_name} | Task {task_id}/{num_tasks_per_suite - 1}")
            print(f"{'=' * 80}")
            
            # Start a new LIBERO worker for this task
            libero_worker_started = False
            try:
                start_libero_worker(task_id=task_id, task_suite_name=suite_name)
                libero_worker_started = True
                print(f"  ✓ LIBERO worker started for {suite_name} task {task_id}")
                print("[info] Waiting for worker ready signal...")
                msg = _read_json_from_worker()
                print("[info] Worker ready:", msg)
                
                global libero_task_name, libero_task_description
                libero_task_name = msg.get("task_name", "Unknown Task")
                libero_task_description = msg.get("task_description", "")
                print(f"[info] Task name: {libero_task_name}")
                print(f"[info] Task description: {libero_task_description}")
                
                # Test reset to verify observations
                libero_process.stdin.write(json.dumps({"cmd": "reset"}) + "\n")
                libero_process.stdin.flush()
                obs = _read_json_from_worker()
                print(f"[info] Observation keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")
                
            except Exception as e:
                print(f"  Error starting LIBERO worker for {suite_name} task {task_id}: {e}")
                import traceback
                traceback.print_exc()
                # Record all trials as failed for this task
                suite_results[task_id] = [{
                    'trial': t + 1,
                    'success': False,
                    'total_reward': 0.0,
                    'episode_length': 0,
                    'num_replans': 0,
                    'video_path': None,
                    'task_name': 'Unknown',
                    'task_description': '',
                    'error': str(e)
                } for t in range(num_trials)]
                continue
            
            # Run trials for this task
            task_trial_results = []
            
            for trial_num in range(1, num_trials + 1):
                print(f"\n  --- {suite_name} | Task {task_id} | Trial {trial_num}/{num_trials} ---")
                
                rng, trial_rng = jax.random.split(rng)
                
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
                        num_replans=16,
                        steps_per_replan=16,
                        config=config,
                        trial_num=trial_num,
                        trial_rng=trial_rng
                    )
                    
                    total_reward = sum(trajectory['rewards'])
                    episode_length = len(trajectory['agentview_observations'])
                    num_replans_used = len(trajectory['replans'])
                    success = (total_reward > 0.5)
                    
                    result = {
                        'trial': trial_num,
                        'success': success,
                        'total_reward': total_reward,
                        'episode_length': episode_length,
                        'num_replans': num_replans_used,
                        'video_path': None,
                        'task_name': libero_task_name,
                        'task_description': libero_task_description
                    }
                    
                    print(f"\n  Trial {trial_num} Results:")
                    print(f"    Success: {'✓' if success else '✗'}")
                    print(f"    Total Reward: {total_reward:.4f}")
                    print(f"    Episode Length: {episode_length} steps")
                    print(f"    Replans Used: {num_replans_used}")
                    
                    # Save trajectory video
                    video_filename = f"{suite_name}_task{task_id}_trial{trial_num:02d}.mp4"
                    video_path = os.path.join(output_dir, video_filename)
                    try:
                        save_trajectory_video(
                            trajectory=trajectory,
                            task_description=libero_task_description,
                            output_path=video_path,
                            fps=10,
                            trial_num=trial_num,
                            suite_name=suite_name,
                            task_id=task_id
                        )
                        result['video_path'] = video_path
                    except Exception as video_error:
                        print(f"    Error saving video: {video_error}")
                    
                    task_trial_results.append(result)
                    
                    # Free trajectory data (images) to reclaim RAM
                    del trajectory
                    gc.collect()
                    
                except Exception as e:
                    print(f"  Error in {suite_name} task {task_id} trial {trial_num}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    task_trial_results.append({
                        'trial': trial_num,
                        'success': False,
                        'total_reward': 0.0,
                        'episode_length': 0,
                        'num_replans': 0,
                        'video_path': None,
                        'task_name': libero_task_name,
                        'task_description': libero_task_description,
                        'error': str(e)
                    })
            
            suite_results[task_id] = task_trial_results
            
            # Print per-task summary
            task_successes = sum(1 for r in task_trial_results if r['success'])
            task_rate = (task_successes / len(task_trial_results) * 100) if task_trial_results else 0.0
            print(f"\n  >> Task {task_id} ({libero_task_name}): {task_rate:.1f}% ({task_successes}/{len(task_trial_results)})")
            
            # Stop worker for this task before moving to next
            if libero_worker_started:
                stop_libero_worker()
                print(f"  ✓ LIBERO worker stopped for {suite_name} task {task_id}")
        
        all_suite_results[suite_name] = suite_results
        
        # Print per-suite summary
        suite_successes = sum(1 for tid, results in suite_results.items() for r in results if r['success'])
        suite_total = sum(len(results) for results in suite_results.values())
        suite_rate = (suite_successes / suite_total * 100) if suite_total > 0 else 0.0
        
        print(f"\n{'=' * 80}")
        print(f"  SUITE {suite_name} COMPLETE: {suite_rate:.1f}% ({suite_successes}/{suite_total})")
        print(f"{'=' * 80}")
        
        # Clear JAX caches once per suite to reclaim VRAM without excessive recompilation
        jax.clear_caches()
        gc.collect()
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n\n{'#' * 100}")
    print(f"# FINAL RESULTS - ALL SUITES")
    print(f"{'#' * 100}\n")
    
    overall_successes = 0
    overall_trials = 0
    
    for suite_name, task_results in all_suite_results.items():
        suite_successes = sum(1 for tid, results in task_results.items() for r in results if r['success'])
        suite_total = sum(len(results) for results in task_results.values())
        suite_rate = (suite_successes / suite_total * 100) if suite_total > 0 else 0.0
        overall_successes += suite_successes
        overall_trials += suite_total
        print(f"  {suite_name:20s}: {suite_rate:6.1f}%  ({suite_successes}/{suite_total})")
    
    overall_rate = (overall_successes / overall_trials * 100) if overall_trials > 0 else 0.0
    print(f"\n  {'OVERALL':20s}: {overall_rate:6.1f}%  ({overall_successes}/{overall_trials})")
    
    if all_suite_results:
        suite_rates = []
        for suite_name, task_results in all_suite_results.items():
            s = sum(1 for tid, results in task_results.items() for r in results if r['success'])
            t = sum(len(results) for results in task_results.values())
            suite_rates.append((s / t * 100) if t > 0 else 0.0)
        print(f"  {'Avg across suites':20s}: {np.mean(suite_rates):6.1f}%")
    
    # Save comprehensive summary
    summary_path = os.path.join(output_dir, "all_suites_summary.txt")
    save_all_suites_summary(all_suite_results, summary_path, config, num_trials, timestamp)
    
    # Also save as JSON for programmatic access
    json_path = os.path.join(output_dir, "all_suites_results.json")
    json_results = {}
    for suite_name, task_results in all_suite_results.items():
        json_results[suite_name] = {}
        for task_id, results in task_results.items():
            json_results[suite_name][str(task_id)] = []
            for r in results:
                json_r = {k: v for k, v in r.items()}
                # Convert any non-serializable types
                for k, v in json_r.items():
                    if isinstance(v, np.floating):
                        json_r[k] = float(v)
                    elif isinstance(v, np.integer):
                        json_r[k] = int(v)
                json_results[suite_name][str(task_id)].append(json_r)
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  ✓ JSON results saved to: {json_path}")
    
    # Print final info
    print("\n" + "=" * 100)
    print("All Suites Success Rate Testing Complete")
    print("=" * 100)
    print(f"Suites tested: {suites_to_test}")
    print(f"Tasks per suite: {num_tasks_per_suite}")
    print(f"Trials per task: {num_trials}")
    print(f"Total trials: {overall_trials}")
    if overall_trials > 0:
        print(f"Overall Success Rate: {overall_rate:.1f}% ({overall_successes}/{overall_trials})")
    print(f"Diffusion steps per action sequence: {num_diffusion_steps}")
    print(f"Flow model parameters: {count_params(params_flow):,}")
    print(f"All results saved to: {output_dir}/")
    
    print("\n✓ All suites testing completed!")
    print("=" * 100)


if __name__ == "__main__":
    # Usage:
    #   python test_libero_flow_success_rates.py                                                          # All 3 default suites, 10 tasks each, 10 trials per task
    #   python test_libero_flow_success_rates.py run_20260212_120000                                      # Load config from checkpoint
    #   python test_libero_flow_success_rates.py --suites libero_spatial,libero_object                    # Only test 2 suites
    #   python test_libero_flow_success_rates.py --suites libero_spatial,libero_object,libero_goal,libero_10  # Test 4 suites
    #   python test_libero_flow_success_rates.py --num-trials 20                                          # 20 trials per task
    #   python test_libero_flow_success_rates.py --num-tasks 5                                            # Only first 5 tasks per suite
    #   python test_libero_flow_success_rates.py --diffusion-steps 50                                     # 50 diffusion steps
    #   python test_libero_flow_success_rates.py run_20260212_120000 --suites libero_spatial --num-trials 5 --diffusion-steps 20
    
    main()
