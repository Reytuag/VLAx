import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

"""
Multi-file Gradient Accumulation Training Script with BATCHED KV Cache Precomputation
WITH SHIFTED TIMESTEP AUGMENTATION

FULL DATASET VERSION: Includes both libero_spatial AND libero_object datasets.

Key Features:
1. BATCHED KV CACHE PRECOMPUTATION: Processes multiple samples from the same HDF5 file
   in a single batch through sampler.get_cache_prompt(). This is more efficient than
   processing samples one-by-one.

2. SAME TASK PROMPT PER BATCH: Each batch contains samples from the same HDF5 file,
   ensuring all samples in a batch share the same task prompt.

3. MEMORY-EFFICIENT CACHING: Stores batches of samples with their precomputed KV caches
   in memory, allowing for efficient reuse during multiple minibatch passes.

4. MULTI-PASS TRAINING: Each cached sample can be processed multiple times with different
   random noise patterns during gradient accumulation, improving training efficiency.

5. CPU-GPU MEMORY OPTIMIZATION: Keeps all data as NumPy arrays on CPU until needed,
   only converting to JAX arrays on GPU during processing.

6. EFFICIENT BATCH PROCESSING: Groups samples by HDF5 file to maximize batching
   efficiency when computing KV caches.

7. SHIFTED TIMESTEP AUGMENTATION: For each demonstration, in addition to sequences starting
   at timestep 0, action_horizon, 2*action_horizon, etc., we also create sequences starting
   at a random shift (between 1 and action_horizon-1). This means we get sequences at:
   - shift, shift+action_horizon, shift+2*action_horizon, etc.
   This effectively doubles the amount of training data extracted from each demonstration.

8. FULL DATASET: Combines libero_spatial and libero_object datasets for comprehensive training.
"""

# Standard library
import gc
import glob
import json
import time
from datetime import datetime
from pathlib import Path

# Third-party
import dataclasses
import gym
import numpy as _np
import optax
import wandb
from flax.training.train_state import TrainState
from PIL import Image

# JAX
import jax
import jax.numpy as jnp

# Gemma / project
from gemma import gm
from gemma import peft
from sampler import Sampler

# Local
from libero_misc.libero_utils import extract_task_instruction
from libero_misc.data_loading import (
    load_hdf5_demonstrations,
    get_image_for_action_seq_numpy,
    extract_actions_from_libero,
    create_training_samples_from_file,
)
from networks.flow_network_state_rope import TransformerFlow
from networks.checkpoint_utils import (
    create_dated_save_directory,
    setup_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
)
from utils import (
    create_cache_mask,
    extract_cache_from_layers,
    linear_lr_schedule,
    cosine_lr_schedule,
)


gemma_path="/home/reytuag/VLA/gemma-3-flax-gemma3-4b-it-v1"
rng = jax.random.PRNGKey(0)


# LIBERO actions have 7 dimensions (end-effector position xyz, orientation wxyz, gripper)
action_shape = 7
action_horizon = 16

# Configuration
config = {
    "action_shape": action_shape,
    "action_horizon": action_horizon,
    "num_diffusion_steps": 10,
    "task_suite_name": "libero_spatial_object",
    "num_epochs": 35,
    "LR": 4e-4,
    "ANNEAL_LR": True,
    "LR_SCHEDULE_TYPE": "cosine",  # Options: "linear", "cosine"
    "MAX_GRAD_NORM": 1.0,
    "TOTAL_TRAIN_STEPS": 19000,
    "GRAD_ACCUM_STEPS": 128 // 8,
    "CACHE_POOL_SIZE": 128,
    "CACHE_BATCH_SIZE": 8,
    "CACHE_REUSE_PASSES": 10,
    "CACHE_LAYERS": [
        "layer_8", "layer_10", 
        "layer_12", "layer_16",
        "layer_18", "layer_24", 
        "layer_28","layer_30",
    ],
    "USE_SAMPLE_WITH_STATE": False,
    "CHECKPOINT_EVERY_STEPS": 4000,
    "CHECKPOINT_MAX_TO_KEEP": 10,
    "RESUME_FROM": None, #"run_20260223_132829/checkpoints"
}

# Path to HDF5 files - FULL DATASET: libero_spatial + libero_object
hdf5_file_path_spatial = "/home/reytuag/VLA/robotics/LIBERO/libero_spatial/*.hdf5"
hdf5_file_path_object = "/home/reytuag/VLA/robotics/LIBERO/libero_object/*.hdf5"

wandb.init(
    entity="gautier-hamon",
    project='VLA_LIBERO_test',
    config=config
)


# ============================================
# Model initialization
# ============================================


image= jnp.zeros((256,256,3),dtype=jnp.uint8)

model = gm.nn.IntWrapper(model=gm.nn.Gemma3_4B(), dtype=jnp.int4)
original = gm.ckpts.load_params(os.path.abspath(os.path.join(gemma_path, "gemma3-4b-it")))
params = peft.quantize(original, method='INT4', checkpoint_kernel_key='w')

del original
gc.collect()
jax.clear_caches()
tokenizer = gm.text.Gemma3Tokenizer(os.path.join(gemma_path, "tokenizer.model"))

sampler = Sampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
    cache_length=256,
    max_out_length=100,
)

# Example task-aware prompt based on LIBERO filename
libero_task = "pick up the black bowl from table center and place it on the plate"
task_prompt = f'Task: {libero_task} Current image: <start_of_image>.'

out = sampler.sample(
            task_prompt,
            images=image[None, ...],
            max_new_tokens=100,
            rng=rng,
            return_state=True,
        )
print(out.text)
print(out.state.cache["layer_9"]["v"].shape)

def precompute_kv_cache_batched(task_prompt, images_batch_np, sampler, config, rng_key):
    """
    Precompute KV cache for a batch of samples using sampler.get_cache_prompt.
    All samples in the batch share the same task prompt.
    
    Args:
        task_prompt: Text prompt for the task (same for all samples in batch)
        images_batch_np: NumPy array of images with shape (batch_size, 2, H, W, 3) (CPU)
        sampler: Sampler instance
        config: Configuration dictionary
        rng_key: JAX random key
        
    Returns:
        Tuple of (cache_k, cache_v, cache_mask) as JAX arrays on GPU
        Each has batch dimension equal to input batch size
    """
    # Convert images to JAX array on GPU
    images_jax = jnp.asarray(images_batch_np, dtype=jnp.uint8)
    
    # Process batch through sampler to get KV cache
    cache = sampler.get_cache_prompt(
        task_prompt,
        images=images_jax,
        rng=rng_key,
    )
    
    # Extract cache from sampler output using config
    cache_k = extract_cache_from_layers(cache, config["CACHE_LAYERS"], "k")
    cache_v = extract_cache_from_layers(cache, config["CACHE_LAYERS"], "v")
    first_layer = config["CACHE_LAYERS"][0]
    cache_mask = create_cache_mask(
        cache[first_layer]["end_index"],
        cache_k.shape[2],
        config["action_horizon"] + 1,
        cache_k.shape[3]
    )
    
    return cache_k, cache_v, cache_mask


# Initialize flow model with num_layers derived from cache layers config
num_flow_layers = len(config["CACHE_LAYERS"])
model_flow = TransformerFlow(
    num_layers=num_flow_layers,
    num_heads=4,
    qkv_features=1024,
    out_features=512,
    input_size=action_shape,
    gating=True,
    gating_bias=2.,
    norm_type="rmsnorm",
    post_attention_norm=True,
    post_mlp_norm=True,
)

x = jnp.ones((1, action_horizon, action_shape))
# Extract cache using config
cache_k = extract_cache_from_layers(out.state.cache, config["CACHE_LAYERS"], "k")
cache_v = extract_cache_from_layers(out.state.cache, config["CACHE_LAYERS"], "v")
# Use first layer from config to get end_index (all layers should have same end_index)
first_layer = config["CACHE_LAYERS"][0]
cache_mask = create_cache_mask(
    out.state.cache[first_layer]["end_index"], cache_k.shape[2], x.shape[1] + 1, cache_k.shape[3]
)

state = jnp.ones((x.shape[0], 9))  # robot state with 9 dimensions (from LIBERO dataset)
params_flow = model_flow.init(
    jax.random.PRNGKey(0), x, cache_mask, cache_k, cache_v, 1.0 * jnp.ones((x.shape[0],)), state
)
flow_fn = jax.jit(model_flow.apply)

#params_flow=jnp.load("run_20260218_011505/flow_model_kvcache_batched_shifted_full_600.npy", allow_pickle=True).item()

# get the number of parameters in the model
def count_params(params):
    return sum([_np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)])

print("Number of parameters in flow model:", count_params(params_flow))


def iter_fn(carry, _):
    params, x, cache_mask, cache_k, cache_v, t, dt, state = carry
    output = flow_fn(params, x, cache_mask, cache_k, cache_v, t, state)
    x = x - output * dt
    carry = (params, x, cache_mask, cache_k, cache_v, t - dt, dt, state)
    return carry, None


# --- Directory & config setup ---

if config["RESUME_FROM"] is not None:
    # When resuming, reuse the existing run directory (parent of the checkpoints dir)
    resume_ckpt_dir = os.path.abspath(config["RESUME_FROM"])
    save_dir = os.path.dirname(resume_ckpt_dir) if resume_ckpt_dir.endswith("checkpoints") else resume_ckpt_dir
    # The checkpoint directory is always <save_dir>/checkpoints
    if not resume_ckpt_dir.endswith("checkpoints"):
        resume_ckpt_dir = os.path.join(resume_ckpt_dir, "checkpoints")
    print(f"Resuming from: {resume_ckpt_dir}")
else:
    save_dir = create_dated_save_directory()
    resume_ckpt_dir = None

# Save configuration at the beginning for reproducibility
# Convert config to JSON-serializable format (convert numpy types, etc.)
config_to_save = {}
for key, value in config.items():
    if isinstance(value, (bool, int, float, str)):
        config_to_save[key] = value
    elif isinstance(value, (list, tuple)):
        config_to_save[key] = list(value)
    else:
        config_to_save[key] = str(value)

config_path = os.path.join(save_dir, "config.json")
with open(config_path, 'w') as f:
    json.dump(config_to_save, f, indent=2)
print(f"Configuration saved to: {config_path}")



if config["ANNEAL_LR"]:
    if config["LR_SCHEDULE_TYPE"].lower() == "cosine":
        lr_schedule = cosine_lr_schedule(config["LR"], config["TOTAL_TRAIN_STEPS"])
    else:
        lr_schedule = linear_lr_schedule(config["LR"], config["TOTAL_TRAIN_STEPS"])
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=lr_schedule, eps=1e-5),
    )
else:
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )

train_state = TrainState.create(
    apply_fn=model_flow.apply,
    params=params_flow,
    tx=tx,
)

# --- Set up Orbax CheckpointManager ---
checkpoint_dir = os.path.abspath(os.path.join(save_dir, "checkpoints"))
checkpoint_manager = setup_checkpoint_manager(
    checkpoint_dir,
    max_to_keep=config["CHECKPOINT_MAX_TO_KEEP"],
    save_interval_steps=config["CHECKPOINT_EVERY_STEPS"],
)
print(f"Checkpoint directory: {checkpoint_dir}")

# --- Restore from checkpoint if resuming ---
start_epoch = 0
start_step = 0

if config["RESUME_FROM"] is not None:
    restored_state, restored_meta = restore_checkpoint(checkpoint_manager, train_state)
    if restored_state is not None:
        train_state = restored_state
        start_epoch = restored_meta['epoch']
        start_step = restored_meta['global_step']
        rng_key = jnp.array(restored_meta['rng_key'], dtype=jnp.uint32)
        print(f"  Resuming training from epoch {start_epoch}, step {start_step}")
    else:
        print("  No checkpoint found at the resume path — starting from scratch.")


@jax.jit
def _compute_gradients(params, cache_k, cache_v, cache_mask, actions, state, action_mask, rng):
    """
    Compute gradients without applying them.
    Returns gradients and loss.
    
    Args:
        params: Model parameters
        cache_k: Key cache from Gemma
        cache_v: Value cache from Gemma
        cache_mask: Attention mask for cache
        actions: Target actions, shape (batch, action_horizon, action_shape)
        state: Robot state, shape (batch, 9)
        action_mask: Mask for valid actions (1=real, 0=padded), shape (batch, action_horizon)
        rng: Random key
    """
    def _loss_fn(params, actions, cache_mask, cache_k, cache_v, state, action_mask, rng):
        rng, _rng = jax.random.split(rng)
        x0 = jax.random.normal(_rng, actions.shape)
        rng, _rng = jax.random.split(rng)
        t = jax.random.uniform(_rng, (actions.shape[0],), minval=0.0, maxval=1.0)
        t = jax.random.beta(_rng, 2.0, 2.0, (actions.shape[0],))
        t = jnp.clip(t, 0, 1)
        # Linear interpolation
        xt = (t[:, None, None]) * x0 + (1 - t[:, None, None]) * actions

        # Target velocity
        target_v = actions - x0
        
        pred_v = model_flow.apply(
            params, xt, cache_mask, cache_k, cache_v, t, state
        )
        
        # Compute squared error per timestep: (batch, action_horizon, action_shape)
        squared_error = (pred_v - target_v) ** 2

        # Mean over action dimensions: (batch, action_horizon)
        squared_error_per_timestep = jnp.mean(squared_error, axis=-1)

        # Apply action mask to zero out loss from padded actions
        masked_squared_error = squared_error_per_timestep * action_mask

        # Compute mean loss only over valid (non-padded) actions
        num_valid_actions = jnp.sum(action_mask) + 1e-8  # Avoid division by zero
        total_loss = jnp.sum(masked_squared_error) / num_valid_actions

        return total_loss, None

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    loss, grads = grad_fn(params, actions, cache_mask, cache_k, cache_v, state, action_mask, rng)
    return grads, loss


def _apply_accumulated_gradients(train_state, accumulated_grads):
    """
    Apply accumulated gradients to the training state.
    Gradients are already averaged during accumulation.
    """
    # Apply the accumulated gradients directly
    train_state = train_state.apply_gradients(grads=accumulated_grads)
    return train_state


# ============================================
# Training loop loading HDF5 files one by one
# WITH MULTI-FILE GRADIENT ACCUMULATION
# AND BATCHED KV CACHE PRECOMPUTATION
# AND SHIFTED TIMESTEP AUGMENTATION
# FULL DATASET: libero_spatial + libero_object
# ============================================




# Combine both datasets
hdf5_files_spatial = sorted(glob.glob(hdf5_file_path_spatial))
hdf5_files_object = sorted(glob.glob(hdf5_file_path_object))
hdf5_files = hdf5_files_spatial + hdf5_files_object

print(f"Found {len(hdf5_files_spatial)} HDF5 files from libero_spatial")
print(f"Found {len(hdf5_files_object)} HDF5 files from libero_object")
print(f"Total: {len(hdf5_files)} HDF5 files to train on")

# Training configuration
if config["RESUME_FROM"] is None:
    rng_key = jax.random.PRNGKey(1)
step = start_step
grad_accum_steps = config["GRAD_ACCUM_STEPS"]
cache_pool_size = config["CACHE_POOL_SIZE"]
cache_batch_size = config["CACHE_BATCH_SIZE"]
cache_reuse_passes = config["CACHE_REUSE_PASSES"]
accumulated_grads = None
num_accumulated = 0
jax.clear_caches()
# Training loop with epochs
for epoch in range(start_epoch, config["num_epochs"]):
    # Shuffle HDF5 files at the start of each epoch
    shuffled_files = hdf5_files.copy()
    rng_key, shuffle_key = jax.random.split(rng_key)
    shuffle_indices = jax.random.permutation(shuffle_key, len(shuffled_files))
    shuffled_files = [shuffled_files[int(i)] for i in shuffle_indices]
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{config['num_epochs']}")
    print(f"{'='*60}")
    
    loss_mean = 0.0
    start_time = time.time()
    
    # First, load all samples from all files (kept on CPU as NumPy arrays)
    print("\n--- Loading samples from all files (WITH SHIFTED AUGMENTATION) ---")
    all_file_data = []
    for file_idx, hdf5_file in enumerate(shuffled_files):
        filename = os.path.basename(hdf5_file)
        
        # Extract task instruction from filename
        task_instruction = extract_task_instruction(hdf5_file)
        task_prompt = f'You are a robotic arm. Task: {task_instruction} Agent view: <start_of_image> Eye in hand view: <start_of_image>. Give detailed subtasks to complete the task.'
        
        # Create samples from this file (kept on CPU as NumPy arrays)
        # This now includes both original and shifted sequences
        file_rng_key = jax.random.fold_in(shuffle_key, file_idx)
        samples = create_training_samples_from_file(hdf5_file, action_shape, action_horizon, shuffle=True, rng_key=file_rng_key)
        
        if not samples or len(samples) == 0:
            print(f"  Skipping {filename} - no valid samples")
            continue
        
        print(f"  Loaded {len(samples)} samples from {filename} (includes shifted augmentation)")
        
        all_file_data.append({
            'filename': filename,
            'task_prompt': task_prompt,
            'task_instruction': task_instruction,
            'samples': samples,
        })
    
    # Calculate total samples across all files
    total_samples = sum(len(file_data['samples']) for file_data in all_file_data)
    print(f"\nTotal samples across all files: {total_samples} (approximately 2x original due to shifting)")
    
    # Create random shuffle indices for each file's samples
    rng_key, shuffle_samples_key = jax.random.split(rng_key)
    file_shuffle_indices = []
    for file_idx, file_data in enumerate(all_file_data):
        file_perm_key = jax.random.fold_in(shuffle_samples_key, file_idx)
        indices = jax.random.permutation(file_perm_key, len(file_data['samples']))
        file_shuffle_indices.append(indices)
    
    # Process ALL samples from ALL files in this epoch
    file_indices = [0] * len(all_file_data)  # Track current index in each file
    samples_processed_in_epoch = 0
    
    # Continue until all samples from all files are processed
    while True:
        # Check if all files are exhausted
        all_files_exhausted = all(file_indices[i] >= len(all_file_data[i]['samples']) 
                                   for i in range(len(all_file_data)))
        if all_files_exhausted:
            print(f"\n--- All samples processed in epoch: {samples_processed_in_epoch} samples ---")
            break
        
        # Create diverse cache batches by round-robin selection from different files
        print(f"\n--- Creating diverse cache batch (pool size: {cache_pool_size}) ---")
        cached_samples = []
        
        
        # Collect samples until we reach cache_pool_size or exhaust all files
        while len(cached_samples) < cache_pool_size and not all_files_exhausted:
            # Group samples by task for batched processing
            batches_to_process = []
            
            # Randomize file order for each collection round
            
            rng_key, file_order_key = jax.random.split(rng_key)
            file_order = jax.random.permutation(file_order_key, len(all_file_data))
            
            for file_idx in file_order:
                file_idx = int(file_idx)
                file_data = all_file_data[file_idx]
                if file_indices[file_idx] >= len(file_data['samples']):
                    continue  # This file is exhausted
                
                # Collect a batch from this file using shuffled indices
                start_idx = file_indices[file_idx]
                end_idx = min(start_idx + cache_batch_size, len(file_data['samples']))
                
                # Get shuffled sample indices for this batch
                shuffled_batch_indices = file_shuffle_indices[file_idx][start_idx:end_idx]
                batch_samples = [file_data['samples'][int(i)] for i in shuffled_batch_indices]
                
                file_indices[file_idx] = end_idx
                
                if batch_samples:
                    batches_to_process.append({
                        'filename': file_data['filename'],
                        'task_prompt': file_data['task_prompt'],
                        'samples': batch_samples,
                    })
                
                # Stop if we have enough cached samples
                if len(cached_samples) >= cache_pool_size:
                    break
            
            # Update exhaustion check
            all_files_exhausted = all(file_indices[i] >= len(all_file_data[i]['samples']) 
                                      for i in range(len(all_file_data)))
            
            if not batches_to_process:
                print(f"  All files exhausted for this cache batch. Cached {len(cached_samples)} samples.")
                break
            
            # Process each batch
            for batch_data in batches_to_process:
                batch_samples = batch_data['samples']
                filename = batch_data['filename']
                task_prompt = batch_data['task_prompt']
                current_batch_size = len(batch_samples)
                
                # Gather images for this batch
                images_list = []
                actions_list = []
                states_list = []
                action_masks_list = []
                
                for images_np, action_seq_np, initial_state_np, action_mask_np in batch_samples:
                    images_list.append(images_np)
                    actions_list.append(action_seq_np)
                    states_list.append(initial_state_np)
                    action_masks_list.append(action_mask_np)
                
                # Stack into batch arrays (still on CPU as NumPy)
                images_batch_np = _np.concatenate(images_list, axis=0)  # (batch_size, 2, H, W, 3)
                actions_batch_np = _np.concatenate(actions_list, axis=0)  # (batch_size, action_horizon, action_shape)
                states_batch_np = _np.concatenate(states_list, axis=0)  # (batch_size, 9)
                action_masks_batch_np = _np.concatenate(action_masks_list, axis=0)  # (batch_size, action_horizon)
                
                # Pad to cache_batch_size if needed to ensure consistent batch size for JIT compilation
                if current_batch_size < cache_batch_size:
                    pad_size = cache_batch_size - current_batch_size
                    # Pad images: (batch_size, 2, H, W, 3) -> (cache_batch_size, 2, H, W, 3)
                    images_pad = _np.zeros((pad_size,) + images_batch_np.shape[1:], dtype=images_batch_np.dtype)
                    images_batch_np = _np.concatenate([images_batch_np, images_pad], axis=0)
                    # Pad actions: (batch_size, action_horizon, action_shape) -> (cache_batch_size, action_horizon, action_shape)
                    actions_pad = _np.zeros((pad_size,) + actions_batch_np.shape[1:], dtype=actions_batch_np.dtype)
                    actions_batch_np = _np.concatenate([actions_batch_np, actions_pad], axis=0)
                    # Pad states: (batch_size, 9) -> (cache_batch_size, 9)
                    states_pad = _np.zeros((pad_size,) + states_batch_np.shape[1:], dtype=states_batch_np.dtype)
                    states_batch_np = _np.concatenate([states_batch_np, states_pad], axis=0)
                    # Pad action masks: (batch_size, action_horizon) -> (cache_batch_size, action_horizon)
                    # Pad with zeros (masked out) for the padding samples
                    action_masks_pad = _np.zeros((pad_size,) + action_masks_batch_np.shape[1:], dtype=action_masks_batch_np.dtype)
                    action_masks_batch_np = _np.concatenate([action_masks_batch_np, action_masks_pad], axis=0)
                
                # Precompute KV cache for the entire batch (this moves to GPU)
                # Batch size is now always cache_batch_size for consistent JIT compilation
                rng_key, cache_key = jax.random.split(rng_key)
                cache_k, cache_v, cache_mask = precompute_kv_cache_batched(
                    task_prompt, images_batch_np, sampler, config, cache_key
                )
                
                # Convert action and state arrays to JAX (move to GPU)
                actions_batch = jnp.asarray(actions_batch_np)
                states_batch = jnp.asarray(states_batch_np)
                action_masks_batch = jnp.asarray(action_masks_batch_np)
                
                # Store each sample individually with its cache slice
                for i in range(current_batch_size):
                    if len(cached_samples) >= cache_pool_size:
                        break
                    cached_samples.append({
                        'cache_k': cache_k[i:i+1],  # Keep batch dimension
                        'cache_v': cache_v[i:i+1],
                        'cache_mask': cache_mask[i:i+1],
                        'action_seq': actions_batch[i:i+1],
                        'initial_state': states_batch[i:i+1],
                        'action_mask': action_masks_batch[i:i+1],  # Shape: (1, action_horizon)
                        'filename': filename,
                    })
                    samples_processed_in_epoch += 1
                
                if len(cached_samples) % 32 == 0 or len(cached_samples) >= cache_pool_size:
                    print(f"  Cached {len(cached_samples)}/{cache_pool_size} samples (last batch: {filename}, size: {current_batch_size})")
                
                if len(cached_samples) >= cache_pool_size:
                    break
        
        print(f"\n--- KV cache precomputation complete: {len(cached_samples)} diverse samples cached ---")
        
        # Count samples per file for verification
        file_counts = {}
        for sample in cached_samples:
            filename = sample['filename']
            file_counts[filename] = file_counts.get(filename, 0) + 1
        
        print("  Sample distribution across files:")
        for filename, count in sorted(file_counts.items()):
            print(f"    {filename}: {count} samples")
        
        print(f"\n--- Now processing cached samples {cache_reuse_passes} times ---")
        
        # Define minibatch size for processing cached samples
        minibatch_size = 8  # Process 8 samples at a time
        
        # Now process the cached samples multiple times
        for pass_idx in range(cache_reuse_passes):
            # Shuffle the cached samples for each pass
            rng_key, pass_shuffle_key = jax.random.split(rng_key)
            shuffle_indices = jax.random.permutation(pass_shuffle_key, len(cached_samples))
            
            # Process in minibatches
            for minibatch_start in range(0, len(shuffle_indices), minibatch_size):
                minibatch_end = min(minibatch_start + minibatch_size, len(shuffle_indices))
                minibatch_indices = shuffle_indices[minibatch_start:minibatch_end]
                
                # Gather data for this minibatch
                batch_cache_k = []
                batch_cache_v = []
                batch_cache_mask = []
                batch_action_seq = []
                batch_initial_state = []
                batch_action_mask = []
                batch_filenames = []
                
                for cached_idx in minibatch_indices:
                    cached_sample = cached_samples[int(cached_idx)]
                    batch_cache_k.append(cached_sample['cache_k'])
                    batch_cache_v.append(cached_sample['cache_v'])
                    batch_cache_mask.append(cached_sample['cache_mask'])
                    batch_action_seq.append(cached_sample['action_seq'])
                    batch_initial_state.append(cached_sample['initial_state'])
                    batch_action_mask.append(cached_sample['action_mask'])
                    batch_filenames.append(cached_sample['filename'])
                
                # Concatenate along batch dimension
                batch_cache_k = jnp.concatenate(batch_cache_k, axis=0)
                batch_cache_v = jnp.concatenate(batch_cache_v, axis=0)
                batch_cache_mask = jnp.concatenate(batch_cache_mask, axis=0)
                batch_action_seq = jnp.concatenate(batch_action_seq, axis=0)
                batch_initial_state = jnp.concatenate(batch_initial_state, axis=0)
                batch_action_mask = jnp.concatenate(batch_action_mask, axis=0)
                
                # Count unique files in this minibatch for diversity metric
                unique_files = len(set(batch_filenames))
                
                # Compute gradients for the entire minibatch
                rng_key, grad_key = jax.random.split(rng_key)
                grads, loss = _compute_gradients(
                    train_state.params, 
                    batch_cache_k, 
                    batch_cache_v, 
                    batch_cache_mask, 
                    batch_action_seq, 
                    batch_initial_state, 
                    batch_action_mask,
                    grad_key
                )
                
                # Accumulate gradients with averaging
                if accumulated_grads is None:
                    # First gradient: divide by total accumulation steps
                    accumulated_grads = jax.tree_util.tree_map(
                        lambda g: g / grad_accum_steps,
                        grads
                    )
                else:
                    # Add subsequent gradients (also divided by total accumulation steps)
                    accumulated_grads = jax.tree_util.tree_map(
                        lambda a, g: a + g / grad_accum_steps,
                        accumulated_grads,
                        grads
                    )
                num_accumulated += 1
                
                loss_mean = loss_mean + loss[0] * 0.05
                step += 1
                
                # Apply accumulated gradients when we reach the accumulation steps
                if num_accumulated >= grad_accum_steps:
                    train_state = _apply_accumulated_gradients(train_state, accumulated_grads)
                    accumulated_grads = None
                    num_accumulated = 0
                
                if step % 20 == 0:
                    end_time = time.time()
                    print(f"      Step {step}, Loss = {loss_mean:.6f}, Pass = {pass_idx+1}/{cache_reuse_passes}, Minibatch = {len(minibatch_indices)}, Unique files = {unique_files}/{len(minibatch_indices)}, Time = {end_time-start_time:.4f}s, Accum = {num_accumulated}/{grad_accum_steps}")
                    

                    # Get current learning rate
                    if config["ANNEAL_LR"]:
                        current_lr = lr_schedule(train_state.step)
                    else:
                        current_lr = config["LR"]
                    
                    wandb.log({
                        "loss": loss_mean.item(), 
                        "learning_rate": current_lr,
                    })
            
                    loss_mean = 0.0
                    start_time = time.time()
                
                if step % config["CHECKPOINT_EVERY_STEPS"] == 0:
                    # Save full checkpoint (train_state + metadata) via Orbax
                    save_checkpoint(checkpoint_manager, train_state, step, epoch, rng_key)
                    # Also keep a legacy .npy snapshot for quick param inspection
                    model_path = os.path.join(save_dir, "flow_model_kvcache_batched_shifted_full_" + str(step//500) + ".npy")
                    jnp.save(model_path, train_state.params)
                    
        
    
    # Clear file data to free CPU memory before next epoch
    del all_file_data
    gc.collect()

    # Apply any remaining accumulated gradients at the end of epoch
    if accumulated_grads is not None and num_accumulated > 0:
        train_state = _apply_accumulated_gradients(train_state, accumulated_grads)
        accumulated_grads = None
        num_accumulated = 0
    
    print(f"\nEpoch {epoch + 1} completed! Steps so far: {step}")
    # Save a checkpoint at the end of each epoch
    save_checkpoint(checkpoint_manager, train_state, step, epoch + 1, rng_key)

print(f"\n{'='*60}")
print(f"Training completed! Total epochs: {config['num_epochs']}, Total steps: {step}")
print(f"{'='*60}")
# Final checkpoint
save_checkpoint(checkpoint_manager, train_state, step, config['num_epochs'], rng_key)
# Legacy .npy for quick param inspection
final_model_path = os.path.join(save_dir, "flow_model_kvcache_batched_shifted_full_final.npy")
jnp.save(final_model_path, train_state.params)
print(f"Final model saved to: {final_model_path}")
# Close the checkpoint manager
checkpoint_manager.close()
