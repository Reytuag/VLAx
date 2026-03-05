import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional


def create_cache_mask(indices, key_length, query_length, num_heads):
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


def linear_lr_schedule(init_lr, total_steps):
    """Linear decay learning rate schedule: decreases linearly from init_lr to 0."""
    def schedule(step):
        progress = step / total_steps
        return init_lr * (1.0 - progress)
    return schedule


def cosine_lr_schedule(init_lr, total_steps, min_lr_ratio=0.0):
    """
    Cosine annealing learning rate schedule.
    Decreases learning rate following a cosine curve from init_lr to min_lr.
    
    Args:
        init_lr: Initial learning rate
        total_steps: Total number of training steps
        min_lr_ratio: Minimum LR as a ratio of init_lr (default 0.0 for decay to near 0)
    """
    def schedule(step):
        progress = step / total_steps
        # Cosine annealing: starts at 1, ends at min_lr_ratio
        cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decayed = (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio
        return init_lr * decayed
    return schedule


# ============================================================================
# Action helpers
# ============================================================================

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


# ============================================================================
# Video / frame helpers
# ============================================================================

def create_video_frame(agentview_image: np.ndarray,
                       eye_in_hand_image: np.ndarray,
                       task_description: str,
                       step: int,
                       trial_num: Optional[int] = None,
                       suite_name: Optional[str] = None,
                       task_id: Optional[int] = None) -> np.ndarray:
    """
    Create a video frame with concatenated dual-view images and task descriptor on top.

    Args:
        agentview_image: RGB image from agentview camera (H, W, 3)
        eye_in_hand_image: RGB image from eye-in-hand camera (H, W, 3)
        task_description: Task description text to display
        step: Current step number
        trial_num: Trial number (optional)
        suite_name: Task suite name (optional)
        task_id: Task ID (optional)

    Returns:
        Combined frame as numpy array (H_total, W, 3) with uint8 dtype
    """
    from PIL import Image, ImageDraw, ImageFont

    agentview = agentview_image.astype(np.uint8)
    eye_in_hand = eye_in_hand_image.astype(np.uint8)

    h, w, c = agentview.shape

    text_height = 60
    header_width = w * 2  # Two images side by side

    header_img = Image.new('RGB', (header_width, text_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(header_img)

    try:
        font_task = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_step = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font_task = ImageFont.load_default()
        font_step = ImageFont.load_default()

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

    header_array = np.array(header_img)

    dual_view = np.concatenate([agentview, eye_in_hand], axis=1)
    final_frame = np.concatenate([header_array, dual_view], axis=0)

    return final_frame


def save_trajectory_video(trajectory: Dict,
                          task_description: str,
                          output_path: str,
                          fps: int = 10,
                          trial_num: Optional[int] = None,
                          suite_name: Optional[str] = None,
                          task_id: Optional[int] = None):
    """
    Save trajectory as a video with concatenated dual-view images and task descriptor.

    Args:
        trajectory: Dictionary containing 'agentview_observations' and 'eye_in_hand_observations'
        task_description: Task description to display in video
        output_path: Path to save the video file
        fps: Frames per second for the video
        trial_num: Trial number (optional)
        suite_name: Task suite name (optional)
        task_id: Task ID (optional)
    """
    print(f"\n[info] Saving trajectory video to {output_path}...")

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

    first_frame = create_video_frame(agentview_obs[0], eye_in_hand_obs[0], task_description, 0,
                                     trial_num, suite_name, task_id)
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    num_frames = min(len(agentview_obs), len(eye_in_hand_obs))
    for i in range(num_frames):
        frame = create_video_frame(agentview_obs[i], eye_in_hand_obs[i], task_description, i,
                                   trial_num, suite_name, task_id)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"  ✓ Video saved: {output_path} ({num_frames} frames at {fps} fps)")
