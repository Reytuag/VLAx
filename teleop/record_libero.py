import json
import logging
import os
import subprocess
import time
from typing import Any, Dict

import shutil
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lerobot", "src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Initialize pygame for controller input (optional, can be adapted to keyboard)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame


def _read_json_from_worker(worker: subprocess.Popen):
    """
    Read lines from the worker until we get valid JSON.
    Ignores log spam and prints it as worker logs.
    """
    while True:
        line = worker.stdout.readline()
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
    """Starts the libero worker subprocess."""
    worker = subprocess.Popen(
        ["/home/reytuag/miniconda3/envs/libero_env/bin/python", "-u", "libero_worker.py", str(task_id), task_suite_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    # Wait for ready signal
    msg = _read_json_from_worker(worker)
    if msg.get("status") == "ready":
        print(f"Worker ready: {msg.get('task_name')} - {msg.get('task_description')}")
        return worker, msg.get("task_description")
    
    raise RuntimeError(f"Failed to start libero worker. Got: {msg}")


def reset_env(worker: subprocess.Popen) -> Dict[str, Any]:
    worker.stdin.write(json.dumps({"cmd": "reset"}) + "\n")
    worker.stdin.flush()
    return _read_json_from_worker(worker)


def step_env(worker: subprocess.Popen, action: np.ndarray) -> Dict[str, Any]:
    worker.stdin.write(json.dumps({"cmd": "step", "action": action.tolist()}) + "\n")
    worker.stdin.flush()
    return _read_json_from_worker(worker)


def get_controller_action(joystick: pygame.joystick.Joystick) -> np.ndarray:
    """
    Reads action from a connected gamepad/controller.
    Maps joystick axes to 7D action space (x, y, z, roll, pitch, yaw, gripper).
    """
    pygame.event.pump()
    action = np.zeros(7, dtype=np.float32)
    
    # Left stick for XYZ
    action[0] = joystick.get_axis(1)  # X
    action[1] = -joystick.get_axis(0) # Y
    
    # Right stick for Pitch/Yaw (or Z)
    action[2] = -joystick.get_axis(4)/2 # Z (using right stick Y)
    action[3] = joystick.get_axis(3)/2  # Yaw (using right stick X)
    
    # Triggers for gripper (Axis 5 is usually right trigger on Xbox/PlayStation controllers)
    # The value usually goes from -1 (unpressed) to 1 (fully pressed) on Linux
    trigger_val = joystick.get_axis(5)
    # Map from [-1, 1] analog to discrete open/close (-1 or 1). 
    # Let's say pulling trigger more than halfway closes it.
    gripper = 1.0 if trigger_val > 0.0 else -1.0
    action[6] = gripper
    
    # Apply small deadzone
    action[np.abs(action) < 0.3] = 0.0
    
    return action


def main():
    repo_id = "local/libero_teleop"
    fps = 5
    num_episodes = 5
    
    # Initialize Pygame and Joystick
    pygame.init()
    pygame.joystick.init()
    
    # Initialize a display for the camera feeds (scaling up for better visibility)
    scale_factor = 3
    base_width, base_height = 128, 128
    display_width = (base_width * 2) * scale_factor
    display_height = base_height * scale_factor
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption('Libero Teleoperation')
    
    if pygame.joystick.get_count() == 0:
        print("No controller found. Please connect a gamepad/joystick.")
        return
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Initialized controller: {joystick.get_name()}")

    # Define dataset features specifically following LeRobot structure
    features = {
        "observation.images.agentview_image": {
            "dtype": "video",
            "shape": (3, 128, 128), # Expected LIBERO image shape
            "names": ["channels", "height", "width"],
        },
        "observation.images.robot0_eye_in_hand_image": {
            "dtype": "video",
            "shape": (3, 128, 128), # Expected LIBERO image shape
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,), # Replace with actual state shape from LIBERO
            "names": ["dim"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dim"],
        },
    }

    print("Creating dataset...")
    # Clear out existing dataset folder to prevent FileExistsError
    default_root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    if default_root.exists():
        print(f"Removing existing dataset folder at {default_root}")
        shutil.rmtree(default_root)

    # Using LeRobot's dataset datastructure
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
        vcodec="h264",
    )

    try:
        worker, task_description = start_libero_worker()
        
        for ep_idx in range(num_episodes):
            print(f"Starting episode {ep_idx + 1}/{num_episodes}")
            obs = reset_env(worker)
            done = False
            
            while not done:
                start_t = time.perf_counter()
                
                # Get action from controller
                action = get_controller_action(joystick)
                print(action)
                # Step environment
                step_result = step_env(worker, action)
                next_obs = step_result["obs"]
                done = step_result["done"]
                
                # Process observation images
                # IMPORTANT: Convert from Libero (which might be (H,W,C) or different) 
                # to Torch tensor (C, H, W) as expected by LeRobot
                agent_img_np = np.array(obs["agentview_image"], dtype=np.uint8)
                eye_in_hand_np = np.array(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
                
                # Render to pygame screen
                # Note: pygame expects (H,W,C) or (W,H,C) depending on surf array but np from libero is (H,W,C)
                # To display correctly we need to swap axes for pygame (W,H,C)
                disp_agent = np.swapaxes(agent_img_np, 0, 1)
                disp_eye = np.swapaxes(eye_in_hand_np, 0, 1)
                
                agent_surface = pygame.surfarray.make_surface(disp_agent)
                agent_surface = pygame.transform.scale(agent_surface, (base_width * scale_factor, base_height * scale_factor))
                screen.blit(agent_surface, (0, 0))
                
                # Need to offset the eye in hand image to the right
                eye_surface = pygame.surfarray.make_surface(disp_eye)
                eye_surface = pygame.transform.scale(eye_surface, (base_width * scale_factor, base_height * scale_factor))
                screen.blit(eye_surface, (base_width * scale_factor, 0))
                pygame.display.flip()

                # Process images for dataset (must be C, H, W)
                if agent_img_np.shape[-1] != 3: # In case it arrives as CHW somehow
                    agent_img_np = agent_img_np.transpose(1, 2, 0)
                agent_img_tensor = torch.from_numpy(agent_img_np).permute(2, 0, 1) # C, H, W
                
                if eye_in_hand_np.shape[-1] != 3:
                    eye_in_hand_np = eye_in_hand_np.transpose(1, 2, 0)
                eye_in_hand_tensor = torch.from_numpy(eye_in_hand_np).permute(2, 0, 1) # C, H, W
                
                # Process state (e.g., robot joint states)
                state = np.array(obs["robot0_joint_pos"] + [obs["robot0_gripper_qpos"][0]], dtype=np.float32)
                state_tensor = torch.from_numpy(state)
                
                action_tensor = torch.from_numpy(action)
                
                # Format to LeRobot datastructure standards
                frame = {
                    "observation.images.agentview_image": agent_img_tensor,
                    "observation.images.robot0_eye_in_hand_image": eye_in_hand_tensor,
                    "observation.state": state_tensor,
                    "action": action_tensor,
                    "task": task_description
                }
                
                # Add to dataset buffer
                dataset.add_frame(frame)
                obs = next_obs
                
                # Maintain FPS
                elapsed = time.perf_counter() - start_t
                time.sleep(max(0, (1.0 / fps) - elapsed))
                
            print(f"Episode {ep_idx + 1} completed. Saving...")
            dataset.save_episode()
            
    finally:
        print("Finalizing dataset...")
        dataset.finalize()
        pygame.quit()

if __name__ == "__main__":
    main()
