"""
Data loading utilities for LIBERO HDF5 datasets.

Functions for loading demonstrations, extracting actions, and creating training samples
for the flow matching training pipeline.
"""

import os
import traceback

import h5py
import jax
import jax.numpy as jnp
import numpy as _np


def load_hdf5_demonstrations(hdf5_path: str) -> list:
    """
    Load all demonstrations from an HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        
    Returns:
        List of demonstrations, each containing 'images', 'actions', 'states'
    """
    try:
        demonstrations = []
        with h5py.File(hdf5_path, 'r') as f:
            # Iterate through all demo_X groups
            demo_idx = 0
            while f'data/demo_{demo_idx}' in f:
                demo_group = f[f'data/demo_{demo_idx}']
                
                # Load both camera views (shape: (T, 128, 128, 3) each)
                agentview_rgb = demo_group['obs/agentview_rgb'][:][:,::-1,...]  # Flip vertically if needed
                eye_in_hand_rgb = demo_group['obs/eye_in_hand_rgb'][:]
                

                # Stack along new axis: (T, 2, 128, 128, 3)
                # axis 0: agentview_rgb, axis 1: eye_in_hand_rgb
                images = _np.stack([agentview_rgb, eye_in_hand_rgb], axis=1)
                
                # Load actions (shape: (T, 7) for LIBERO)
                actions = demo_group['actions'][:]
                
                # Load robot states (shape: (T, 9) for LIBERO)
                states = demo_group['robot_states'][:]
                
                demonstrations.append({
                    'images': images,
                    'actions': actions,
                    'states': states
                })
                
                demo_idx += 1
        
        if not demonstrations:
            print(f"Warning: No demonstrations found in {hdf5_path}")
            return None
        
        return demonstrations
    except Exception as e:
        print(f"Error loading HDF5 file {hdf5_path}: {e}")
        return None


def get_image_for_action_seq_numpy(images: _np.ndarray, seq_idx: int, action_horizon: int, shift: int = 0) -> _np.ndarray:
    """
    Extract one image per action sequence from the image array.
    Uses the first frame of the sequence as the representative image.
    Keeps as NumPy array on CPU.
    
    Args:
        images: Numpy array of all frames from demonstration (shape: (T, 2, 128, 128, 3))
                where axis 1 contains [agentview_rgb, eye_in_hand_rgb]
        seq_idx: Index of the action sequence (0-based)
        action_horizon: Length of each action sequence
        shift: Timestep offset (0 for original sequences, >0 for shifted sequences)
        
    Returns:
        Single image as NumPy array of shape (1, 2, height, width, 3)
    """
    # Calculate which frame corresponds to this action sequence with shift
    # Use the first frame of the corresponding sequence
    frame_idx = int(shift + seq_idx * action_horizon)
    frame_idx = min(frame_idx, len(images) - 1)
    
    # Return single frame with batch dimension: (1, 2, H, W, 3)
    # Keep as NumPy array (CPU)
    image = images[frame_idx:frame_idx+1].astype(_np.uint8)
    return image


def extract_actions_from_libero(actions_np, action_shape):
    """
    Extract and normalize LIBERO robot actions.
    
    LIBERO actions have 7 dimensions:
    - 3 for end-effector position (xyz)
    - 3 for end-effector orientation (xyz representation)
    - 1 for gripper action
    
    Args:
        actions_np: Numpy array of shape (num_timesteps, 7) containing raw actions
        action_shape: Expected action dimension (7 for LIBERO)
        
    Returns:
        Numpy array of shape (num_timesteps, action_shape) with normalized actions
    """
    if actions_np.shape[1] != action_shape:
        raise ValueError(f"Expected action_shape {action_shape}, got {actions_np.shape[1]}")
    
    # For LIBERO, actions are already in a reasonable range
    # We can optionally normalize them or clip to a reasonable range
    # Position: typically in [-1, 1] or similar
    # Orientation: typically in [-pi, pi] or normalized
    # Gripper: typically binary or in [0, 1]
    
    # Clip and normalize to [-1, 1] range for consistency
    actions_normalized = _np.clip(actions_np, -1, 1)
    
    return actions_normalized


def create_training_samples_from_file(hdf5_file, action_shape, action_horizon, shuffle=True, rng_key=None):
    """
    Create all training samples from a single HDF5 file.
    Keep data as NumPy arrays on CPU.
    
    NEW: For each demonstration, we create TWO sets of sequences:
    1. Original sequences starting at timestep 0, action_horizon, 2*action_horizon, ...
    2. Shifted sequences starting at random shift (1 to action_horizon-1), 
       then shift+action_horizon, shift+2*action_horizon, ...
    
    This effectively extracts more training data from each demonstration.
    
    Args:
        hdf5_file: Path to HDF5 file
        action_shape: Dimension of action space
        action_horizon: Length of action sequences
        shuffle: Whether to randomize sample order
        rng_key: JAX PRNGKey for random operations (shift generation and shuffling)
    
    Returns:
        List of tuples: (images_np, actions_np, initial_state_np)
        All arrays are kept as NumPy arrays on CPU
    """
    samples = []
    
    try:
        filename = os.path.basename(hdf5_file)
        
        # Load all demonstrations from the HDF5 file
        demonstrations = load_hdf5_demonstrations(hdf5_file)
        if demonstrations is None:
            return samples
        
        # Process each demonstration
        for demo_idx, demo in enumerate(demonstrations):
            images = demo['images']
            actions_np = demo['actions']
            states_np = demo['states']  # robot states with shape (T, 9)
            
            # Extract and normalize actions
            actions_np = extract_actions_from_libero(actions_np, action_shape)
            
            # Check if we have enough data points for the action horizon
            if len(actions_np) < action_horizon:
                continue
            

            # Process BOTH original (shift=0) and shifted sequences
            for current_shift in [0]:
                # Start from the shift position
                shifted_actions = actions_np[current_shift:]
                shifted_states = states_np[current_shift:]
                shifted_images = images  # We'll handle image indexing in get_image_for_action_seq_numpy
                
                # Check if we still have enough data after shifting
                if len(shifted_actions) < action_horizon:
                    continue
                
                # Reshape actions into sequences of action_horizon length with zero-padding
                num_sequences = (len(shifted_actions) + action_horizon - 1) // action_horizon  # Ceiling division
                # Pad actions to have num_sequences * action_horizon timesteps
                padded_length = num_sequences * action_horizon
                actions_padded = _np.zeros((padded_length, action_shape), dtype=shifted_actions.dtype)
                actions_padded[:len(shifted_actions)] = shifted_actions
                actions_sequences = actions_padded.reshape(num_sequences, action_horizon, action_shape)
                
                # Create action mask: 1 for real actions, 0 for padded actions
                action_mask = _np.zeros((padded_length,), dtype=_np.float32)
                action_mask[:len(shifted_actions)] = 1.0
                action_mask_sequences = action_mask.reshape(num_sequences, action_horizon)
                
                # Pad states similarly to match action sequences
                states_padded = _np.zeros((padded_length, shifted_states.shape[1]), dtype=shifted_states.dtype)
                states_padded[:len(shifted_states)] = shifted_states
                states_sequences = states_padded.reshape(num_sequences, action_horizon, shifted_states.shape[1])
                
                # Create samples for each sequence
                for seq_idx in range(num_sequences):
                    # Get one image per action sequence (keep as numpy), accounting for shift
                    images_stacked = get_image_for_action_seq_numpy(
                        shifted_images, seq_idx, action_horizon, shift=current_shift
                    )
                    if images_stacked is None:
                        continue
                    
                    # Get single sequence (keep as numpy)
                    action_seq = actions_sequences[seq_idx:seq_idx+1]  # NumPy array
                    state_seq = states_sequences[seq_idx:seq_idx+1]  # Shape: (1, action_horizon, 9)
                    # Take the first state for each sequence (initial state)
                    initial_state = state_seq[:, 0, :]  # Shape: (1, 9), NumPy array
                    # Get action mask for this sequence (1 for real actions, 0 for padded)
                    action_mask_seq = action_mask_sequences[seq_idx:seq_idx+1]  # Shape: (1, action_horizon)
                    
                    samples.append((images_stacked, action_seq, initial_state, action_mask_seq))
        
        # Randomize sample order within this file
        if shuffle and len(samples) > 0:
            shuffle_subkey = jax.random.fold_in(rng_key, len(samples))
            perm = jax.random.permutation(shuffle_subkey, len(samples))
            samples = [samples[int(i)] for i in perm]

        
    except Exception as e:
        print(f"  Error processing {hdf5_file}: {e}")
        traceback.print_exc()
    
    return samples
