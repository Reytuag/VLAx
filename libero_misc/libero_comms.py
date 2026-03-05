"""
LIBERO Worker Communication Module

Shared functions for communicating with the LIBERO environment via a subprocess
worker (libero_worker.py). Used by RL training, testing, and visualization scripts.
"""

import json
import subprocess
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# Module-level state
# ============================================================================

libero_process = None
libero_task_name: Optional[str] = None
libero_task_description: Optional[str] = None


# ============================================================================
# Internal helpers
# ============================================================================

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


def _parse_obs(obs_dict: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse observation dict into (agentview, eye_in_hand, robot_state).

    Handles both flat-list and shaped-array formats from the worker.

    Returns:
        agentview: uint8 array (128, 128, 3)
        eye_in_hand: uint8 array (128, 128, 3)
        robot_state: float32 array (9,) — [gripper_qpos(2), eef_pos(3), eef_quat(4)]
    """
    # --- agentview image ---
    if 'agentview_image' in obs_dict:
        agentview = np.array(obs_dict['agentview_image'], dtype=np.uint8)[::-1]
        if len(agentview.shape) == 1:
            agentview = agentview.reshape(128, 128, 3)
    else:
        print("[warning] agentview_image not found, creating dummy")
        agentview = np.zeros((128, 128, 3), dtype=np.uint8)

    # --- eye-in-hand image ---
    if 'robot0_eye_in_hand_image' in obs_dict:
        eye_in_hand = np.array(obs_dict['robot0_eye_in_hand_image'], dtype=np.uint8)
        if len(eye_in_hand.shape) == 1:
            eye_in_hand = eye_in_hand.reshape(128, 128, 3)
    else:
        print("[warning] eye_in_hand_image not found, creating dummy")
        eye_in_hand = np.zeros((128, 128, 3), dtype=np.uint8)

    # --- robot state (9D) ---
    if ('robot0_gripper_qpos' in obs_dict
            and 'robot0_eef_pos' in obs_dict
            and 'robot0_eef_quat' in obs_dict):
        gripper_qpos = np.array(obs_dict['robot0_gripper_qpos'], dtype=np.float32)
        eef_pos = np.array(obs_dict['robot0_eef_pos'], dtype=np.float32)
        eef_quat = np.array(obs_dict['robot0_eef_quat'], dtype=np.float32)
        robot_state = np.concatenate([gripper_qpos, eef_pos, eef_quat])
    elif 'robot0_gripper_qpos' in obs_dict and 'robot0_eef_pos' in obs_dict:
        gripper_qpos = np.array(obs_dict['robot0_gripper_qpos'], dtype=np.float32)
        eef_pos = np.array(obs_dict['robot0_eef_pos'], dtype=np.float32)
        eef_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        robot_state = np.concatenate([gripper_qpos, eef_pos, eef_quat])
    else:
        print("[warning] Required robot state keys not found, using dummy state")
        robot_state = np.zeros(9, dtype=np.float32)

    return agentview, eye_in_hand, robot_state


# ============================================================================
# Public API
# ============================================================================

def start_libero_worker(task_id: int = 1,
                        task_suite_name: str = "libero_spatial",
                        python_bin: str = "/home/reytuag/miniconda3/envs/libero_env/bin/python",
                        worker_script: str = "libero_misc/libero_worker.py"):
    """
    Start the LIBERO worker subprocess.

    Args:
        task_id: Index of the task to use (default: 1)
        task_suite_name: Name of the task suite (default: "libero_spatial")
        python_bin: Path to Python executable for the worker
        worker_script: Path to the worker script

    Returns:
        The Popen object for the worker process
    """
    global libero_process

    print(f"[info] Starting LIBERO worker (task_id={task_id}, task_suite='{task_suite_name}')...")
    try:
        libero_process = subprocess.Popen(
            [python_bin, "-u", worker_script, str(task_id), task_suite_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        print(f"[info] LIBERO worker started successfully (task_id={task_id}, task_suite='{task_suite_name}')")
        return libero_process
    except Exception as e:
        print(f"[error] Failed to start LIBERO worker: {e}")
        return None


def stop_libero_worker():
    """Stop the LIBERO worker subprocess."""
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


def libero_reset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Send reset command to LIBERO worker and get initial observation.

    Returns:
        (agentview, eye_in_hand, robot_state)
    """
    global libero_process

    if libero_process is None:
        raise RuntimeError("LIBERO worker not started. Call start_libero_worker() first.")

    libero_process.stdin.write(json.dumps({"cmd": "reset"}) + "\n")
    libero_process.stdin.flush()
    obs_dict = _read_json_from_worker()

    if not isinstance(obs_dict, dict):
        raise ValueError(f"Invalid observation format: expected dict, got {type(obs_dict)}")

    return _parse_obs(obs_dict)


def libero_step(action):
    """
    Send step command to LIBERO worker with action and get result.

    Args:
        action: Action to execute (numpy array or list)

    Returns:
        (agentview, eye_in_hand, robot_state, reward, done, info)
    """
    global libero_process

    if libero_process is None:
        raise RuntimeError("LIBERO worker not started. Call start_libero_worker() first.")

    action_list = action.tolist() if isinstance(action, np.ndarray) else action

    libero_process.stdin.write(json.dumps({"cmd": "step", "action": action_list}) + "\n")
    libero_process.stdin.flush()

    result = _read_json_from_worker()

    obs_dict = result.get("obs", {})
    if isinstance(obs_dict, dict):
        agentview, eye_in_hand, robot_state = _parse_obs(obs_dict)
    else:
        agentview = np.zeros((128, 128, 3), dtype=np.uint8)
        eye_in_hand = np.zeros((128, 128, 3), dtype=np.uint8)
        robot_state = np.zeros(9, dtype=np.float32)

    reward = float(result.get("reward", 0.0))
    done = bool(result.get("done", False))
    info = result.get("info", {})

    return agentview, eye_in_hand, robot_state, reward, done, info


def libero_step_batch(actions):
    """
    Send a batch of actions to the LIBERO worker in a single IPC call.

    The worker executes all actions sequentially and returns only the final
    observation together with per-step rewards / dones.

    Args:
        actions: list of action lists (or 2-D numpy array), shape (N, action_dim)

    Returns:
        (agentview, eye_in_hand, robot_state, rewards, dones, info, steps_executed)
    """
    global libero_process

    if libero_process is None:
        raise RuntimeError("LIBERO worker not started. Call start_libero_worker() first.")

    # Convert numpy arrays to plain lists for JSON serialisation
    if isinstance(actions, np.ndarray):
        action_lists = actions.tolist()
    else:
        action_lists = [a.tolist() if isinstance(a, np.ndarray) else list(a) for a in actions]

    libero_process.stdin.write(
        json.dumps({"cmd": "step_batch", "actions": action_lists}) + "\n"
    )
    libero_process.stdin.flush()

    result = _read_json_from_worker()

    obs_dict = result.get("obs", {})
    if isinstance(obs_dict, dict):
        agentview, eye_in_hand, robot_state = _parse_obs(obs_dict)
    else:
        agentview = np.zeros((128, 128, 3), dtype=np.uint8)
        eye_in_hand = np.zeros((128, 128, 3), dtype=np.uint8)
        robot_state = np.zeros(9, dtype=np.float32)

    rewards = result.get("rewards", [])
    dones = result.get("dones", [])
    info = result.get("info", {})
    steps_executed = int(result.get("steps_executed", len(rewards)))

    return agentview, eye_in_hand, robot_state, rewards, dones, info, steps_executed
