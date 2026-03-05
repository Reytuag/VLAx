import sys
import json
import numpy as np
from typing import Dict, Tuple, Optional
import os
import matplotlib.pyplot as plt

try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path
    LIBERO_ENV_AVAILABLE = True
except ImportError:
    LIBERO_ENV_AVAILABLE = False
    sys.stderr.write("Warning: LIBERO environment not available.\n")
    sys.exit(1)

from .libero_utils import (
    extract_task_instruction,
    load_libero_demo_with_instruction,
    load_all_libero_tasks,
)


def convert_to_serializable(obj):
    """Recursively convert numpy arrays and other non-serializable objects to JSON-compatible types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def setup_libero_environment(task_suite_name: str = "libero_spatial", task_id: int = 1) -> Tuple:
    """
    Set up a LIBERO environment for testing.
    
    Args:
        task_suite_name: Name of the benchmark suite (e.g., "libero_10", "libero_spatial")
        task_id: Index of the task to use
        
    Returns:
        Tuple of (env, task, task_name, task_description)
    """
    if not LIBERO_ENV_AVAILABLE:
        sys.stderr.write("Error: LIBERO environment is not available!\n")
        sys.exit(1)
    
    sys.stderr.write(f"[info] Setting up LIBERO environment with task suite '{task_suite_name}'\n")
    
    # Get benchmark and task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    
    # Retrieve a specific task
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    
    sys.stderr.write(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
          f"language instruction is {task_description}, and the bddl file is {task_bddl_file}\n")
    
    # Create environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    for i in range(16):
        env.step([0, 0, 0, 0, 0, 0,-1])  # Take a few no-op steps to stabilize the environment
    return env, task, task_name, task_description


# Initialize environment
# Accept task_id as first argument, task_suite_name as second argument
task_id = 1  # Default task ID
task_suite_name = "libero_spatial"  # Default task suite

VALID_TASK_SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}

if len(sys.argv) > 1:
    try:
        task_id = int(sys.argv[1])
        sys.stderr.write(f"[info] Using task_id {task_id} from command-line argument\n")
    except ValueError:
        sys.stderr.write(f"[warning] Invalid task_id argument '{sys.argv[1]}', using default task_id=1\n")

if len(sys.argv) > 2:
    suite_arg = sys.argv[2]
    if suite_arg in VALID_TASK_SUITES:
        task_suite_name = suite_arg
        sys.stderr.write(f"[info] Using task_suite_name '{task_suite_name}' from command-line argument\n")
    else:
        sys.stderr.write(f"[warning] Unknown task_suite_name '{suite_arg}', valid options: {sorted(VALID_TASK_SUITES)}. Using default '{task_suite_name}'\n")

env, task, task_name, task_description = setup_libero_environment(task_suite_name=task_suite_name, task_id=task_id)

# Signal ready to parent process with task information
sys.stdout.write(json.dumps({
    "status": "ready",
    "task_name": task_name,
    "task_description": task_description
}) + "\n")
sys.stdout.flush()


# Main command loop
while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break
        
        line = line.strip()
        if not line:
            continue
        
        msg = json.loads(line)
        
        if msg["cmd"] == "reset":
            obs = env.reset()
            # plt.imshow(obs["agentview_image"])
            # plt.savefig("reset_obs.png")
            # plt.close()
            for i in range(16):
                obs,_,_,_=env.step([0, 0, 0, 0, 0, 0,-1])

            # Convert observation dict to serializable format
            obs_serializable = convert_to_serializable(obs)
            sys.stdout.write(json.dumps(obs_serializable) + "\n")
            sys.stdout.flush()
        
        elif msg["cmd"] == "step":
            action = np.array(msg["action"])
            obs, reward, done, info = env.step(action)
            
            # Convert all components to serializable format
            result = {
                "obs": convert_to_serializable(obs),
                "reward": float(reward),
                "done": bool(done),
                "info": convert_to_serializable(info)
            }
            
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
        
        elif msg["cmd"] == "step_batch":
            # Execute a batch of actions in one IPC call.
            # Returns per-step rewards/dones and only the final observation.
            actions = msg["actions"]  # list of action lists
            rewards = []
            dones = []
            last_obs = None
            last_info = {}
            for act in actions:
                action = np.array(act)
                obs, reward, done, info = env.step(action)
                rewards.append(float(reward))
                dones.append(bool(done))
                last_obs = obs
                last_info = info
                if done:
                    break
            
            result = {
                "obs": convert_to_serializable(last_obs),
                "rewards": rewards,
                "dones": dones,
                "info": convert_to_serializable(last_info),
                "steps_executed": len(rewards),
            }
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()

        else:
            sys.stderr.write(f"Unknown command: {msg.get('cmd')}\n")
    
    except Exception as e:
        sys.stderr.write(f"Error in worker loop: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Send error response
        sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
        sys.stdout.flush()