"""
LIBERO Dataset Utilities - Language Instruction Extraction and Integration

This module provides utilities for:
1. Extracting task instructions from LIBERO filenames
2. Loading demonstrations with task context
3. Creating task-aware prompts for the sampler
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import h5py
import jax.numpy as jnp
import numpy as np


def extract_task_instruction(file_path: str) -> str:
    """
    Extract task instruction from LIBERO dataset filename.
    
    The instruction is encoded in the filename by replacing spaces with underscores
    and appending '_demo.hdf5' suffix.
    
    Args:
        file_path: Path to HDF5 file or just the filename
                   e.g., "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5"
    
    Returns:
        Task instruction string
        e.g., "pick up the black bowl from table center and place it on the plate"
    
    Example:
        >>> path = "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5"
        >>> extract_task_instruction(path)
        'pick up the black bowl from table center and place it on the plate'
    """
    filename = os.path.basename(file_path)
    # Remove '_demo.hdf5' suffix and replace underscores with spaces
    task_instruction = filename.replace('_demo.hdf5', '').replace('_', ' ')
    return task_instruction


def load_libero_demo_with_instruction(
    file_path: str,
    demo_idx: int = 0,
    return_all_timesteps: bool = False
) -> Dict[str, Any]:
    """
    Load a single demonstration from LIBERO dataset with its task instruction.
    
    Args:
        file_path: Path to LIBERO HDF5 file
        demo_idx: Index of the demonstration to load (0-49, each file has 50 demos)
        return_all_timesteps: If True, return full trajectories; if False, return only first timestep
    
    Returns:
        Dictionary containing:
            - 'task_instruction': Task description (str)
            - 'demo_idx': Demonstration index (int)
            - 'agentview_rgb': Agent's viewpoint image (np.ndarray)
            - 'eye_in_hand_rgb': Gripper camera image (np.ndarray)
            - 'actions': Action sequence (np.ndarray, shape: T×7)
            - 'ee_pos': End-effector position trajectory (np.ndarray, shape: T×3)
            - 'ee_ori': End-effector orientation trajectory (np.ndarray, shape: T×3)
            - 'gripper_states': Gripper state trajectory (np.ndarray, shape: T×2)
            - 'joint_states': Joint state trajectory (np.ndarray, shape: T×7)
            - 'trajectory_length': Number of timesteps (int)
            - 'prompt': Task-aware prompt for sampler (str)
    
    Example:
        >>> file_path = "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5"
        >>> demo = load_libero_demo_with_instruction(file_path, demo_idx=0)
        >>> print(demo['task_instruction'])
        'pick up the black bowl from table center and place it on the plate'
        >>> print(demo['agentview_rgb'].shape)
        (128, 128, 3)
    """
    task_instruction = extract_task_instruction(file_path)
    
    with h5py.File(file_path, 'r') as f:
        demo_key = f'demo_{demo_idx}'
        demo = f['data'][demo_key]
        
        # Load observation from first timestep
        agentview_rgb = np.array(demo['obs']['agentview_rgb'][0])
        eye_in_hand_rgb = np.array(demo['obs']['eye_in_hand_rgb'][0])
        
        # Load full trajectories
        actions = np.array(demo['actions'][()])
        ee_pos = np.array(demo['obs']['ee_pos'][()])
        ee_ori = np.array(demo['obs']['ee_ori'][()])
        gripper_states = np.array(demo['obs']['gripper_states'][()])
        joint_states = np.array(demo['obs']['joint_states'][()])
        
        trajectory_length = actions.shape[0]
        
        # Create task-aware prompt
        prompt = f'Task: {task_instruction} Current image: <start_of_image>.'
        
        result = {
            'task_instruction': task_instruction,
            'demo_idx': demo_idx,
            'agentview_rgb': agentview_rgb,
            'eye_in_hand_rgb': eye_in_hand_rgb,
            'actions': actions,
            'ee_pos': ee_pos,
            'ee_ori': ee_ori,
            'gripper_states': gripper_states,
            'joint_states': joint_states,
            'trajectory_length': trajectory_length,
            'prompt': prompt,
        }
        
        return result


def create_task_prompt(
    task_instruction: str,
    prompt_format: str = 'minimal'
) -> str:
    """
    Create a task-aware prompt for the sampler.
    
    Args:
        task_instruction: The task instruction text
        prompt_format: One of 'minimal', 'detailed', or 'system'
            - 'minimal': "Task: [instruction] Current image: <start_of_image>."
            - 'detailed': "You are a robotic arm. Task: [instruction] Describe the current image: <start_of_image>."
            - 'system': "System prompt-style instruction"
    
    Returns:
        Formatted prompt string
    
    Example:
        >>> task = "pick up the black bowl from table center and place it on the plate"
        >>> create_task_prompt(task, 'minimal')
        'Task: pick up the black bowl from table center and place it on the plate Current image: <start_of_image>.'
    """
    if prompt_format == 'minimal':
        return f'Task: {task_instruction} Current image: <start_of_image>.'
    elif prompt_format == 'detailed':
        return f'You are a robotic arm. Task: {task_instruction} Describe the current image: <start_of_image>.'
    elif prompt_format == 'system':
        return (
            f'[SYSTEM] You are an advanced robotic manipulation system.\n'
            f'[TASK] {task_instruction}\n'
            f'[INSTRUCTION] Analyze the current image and determine the next action:\n'
            f'<start_of_image>.'
        )
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}. Choose from 'minimal', 'detailed', or 'system'")


def load_all_libero_tasks(libero_dir: str = 'libero_spatial/') -> Dict[str, str]:
    """
    Load all task instructions from LIBERO dataset.
    
    Args:
        libero_dir: Directory containing LIBERO HDF5 files
    
    Returns:
        Dictionary mapping filenames to task instructions
    
    Example:
        >>> tasks = load_all_libero_tasks()
        >>> for filename, instruction in tasks.items():
        ...     print(f"{filename}: {instruction}")
    """
    libero_dir = Path(libero_dir)
    tasks = {}
    
    for hdf5_file in sorted(libero_dir.glob('*_demo.hdf5')):
        task_instruction = extract_task_instruction(str(hdf5_file))
        tasks[hdf5_file.name] = task_instruction
    
    return tasks


def batch_load_libero_demos(
    file_path: str,
    demo_indices: Optional[list] = None,
    max_demos: Optional[int] = None
) -> list:
    """
    Load multiple demonstrations from a single LIBERO task file.
    
    Args:
        file_path: Path to LIBERO HDF5 file
        demo_indices: Specific demo indices to load. If None, loads sequentially.
        max_demos: Maximum number of demos to load (default: load all 50)
    
    Returns:
        List of demonstration dictionaries
    
    Example:
        >>> file_path = "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5"
        >>> demos = batch_load_libero_demos(file_path, max_demos=5)
        >>> print(f"Loaded {len(demos)} demonstrations")
    """
    with h5py.File(file_path, 'r') as f:
        available_demos = list(f['data'].keys())
        num_demos = len(available_demos)
    
    if demo_indices is None:
        max_demos = max_demos or num_demos
        demo_indices = list(range(min(max_demos, num_demos)))
    
    demos = []
    for idx in demo_indices:
        if idx < num_demos:
            demo = load_libero_demo_with_instruction(file_path, demo_idx=idx)
            demos.append(demo)
    
    return demos


def get_libero_action_space_info() -> Dict[str, Any]:
    """
    Get information about LIBERO action space.
    
    Returns:
        Dictionary with action space details
    """
    return {
        'action_dim': 7,
        'action_names': [
            'ee_pos_x', 'ee_pos_y', 'ee_pos_z',
            'ee_ori_qx', 'ee_ori_qy', 'ee_ori_qz',
            'gripper'
        ],
        'description': 'End-effector position (3D) + orientation (quaternion components, 3D) + gripper state (1D)',
        'gripper_values': {0: 'open', 1: 'closed'},
    }


def get_libero_observation_info() -> Dict[str, Any]:
    """
    Get information about LIBERO observation spaces.
    
    Returns:
        Dictionary with observation space details
    """
    return {
        'agentview_rgb': {
            'shape': (128, 128, 3),
            'dtype': 'uint8',
            'description': 'Third-person view of the scene'
        },
        'eye_in_hand_rgb': {
            'shape': (128, 128, 3),
            'dtype': 'uint8',
            'description': 'First-person view from the gripper'
        },
        'ee_pos': {
            'shape': (3,),
            'dtype': 'float64',
            'description': 'End-effector position (x, y, z)'
        },
        'ee_ori': {
            'shape': (3,),
            'dtype': 'float64',
            'description': 'End-effector orientation (rotation angle/axis representation)'
        },
        'gripper_states': {
            'shape': (2,),
            'dtype': 'float64',
            'description': 'Gripper joint states'
        },
        'joint_states': {
            'shape': (7,),
            'dtype': 'float64',
            'description': 'Robot joint states (7-DOF arm)'
        }
    }


if __name__ == '__main__':
    # Example usage
    print("=== LIBERO Dataset Utilities Demo ===\n")
    
    # 1. Extract instructions from all tasks
    print("1. Available LIBERO Tasks:")
    tasks = load_all_libero_tasks()
    for filename, instruction in tasks.items():
        print(f"   - {instruction}")
    
    print("\n2. Load single demonstration:")
    if tasks:
        first_file = list(tasks.keys())[0]
        file_path = f'libero_spatial/{first_file}'
        try:
            demo = load_libero_demo_with_instruction(file_path, demo_idx=0)
            print(f"   Task: {demo['task_instruction']}")
            print(f"   Image shape: {demo['agentview_rgb'].shape}")
            print(f"   Trajectory length: {demo['trajectory_length']}")
            print(f"   Prompt: {demo['prompt']}")
        except Exception as e:
            print(f"   (Could not load: {e})")
    
    print("\n3. Action space info:")
    action_info = get_libero_action_space_info()
    print(f"   Dimensions: {action_info['action_dim']}")
    print(f"   Action names: {action_info['action_names']}")
    
    print("\n4. Create prompts with different formats:")
    task = "pick up the black bowl from table center and place it on the plate"
    for fmt in ['minimal', 'detailed', 'system']:
        prompt = create_task_prompt(task, fmt)
        print(f"   [{fmt}]: {prompt[:60]}...")
