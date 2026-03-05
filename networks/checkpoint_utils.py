"""Checkpoint utility functions for saving and restoring training state."""

import os
from datetime import datetime

import jax
import orbax.checkpoint as ocp


def create_dated_save_directory():
    """
    Create a directory with the current date and time for saving models.
    Returns the path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"run_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created save directory: {save_dir}")
    return save_dir


def setup_checkpoint_manager(checkpoint_dir, max_to_keep=3, save_interval_steps=4000):
    """
    Set up an Orbax CheckpointManager that saves train_state + metadata.
    
    The checkpoint stores:
      - 'train_state': the full Flax TrainState (params + opt_state + step)
      - 'metadata': a dict with epoch, global_step, rng_key serialised as list
    
    Args:
        checkpoint_dir: Absolute path to the checkpoint directory
        max_to_keep: Maximum number of checkpoints to keep on disk
        save_interval_steps: (informational only, actual save is triggered manually)
    
    Returns:
        ocp.CheckpointManager instance
    """
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=1,  # We control saving ourselves; set to 1 so every .save() is accepted
    )
    
    checkpoint_manager = ocp.CheckpointManager(
        directory=checkpoint_dir,
        options=options,
        item_names=('train_state', 'metadata'),
        item_handlers={
            'train_state': ocp.StandardCheckpointHandler(),
            'metadata': ocp.JsonCheckpointHandler(),
        },
    )
    return checkpoint_manager


def save_checkpoint(checkpoint_manager, train_state, step, epoch, rng_key):
    """Save a checkpoint containing the full training state + metadata."""
    metadata = {
        'epoch': int(epoch),
        'global_step': int(step),
        'rng_key': jax.device_get(rng_key).tolist(),
    }
    checkpoint_manager.save(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardSave(train_state),
            metadata=ocp.args.JsonSave(metadata),
        ),
    )
    checkpoint_manager.wait_until_finished()
    print(f"  ✓ Checkpoint saved at step {step} (epoch {epoch})")


def restore_checkpoint(checkpoint_manager, train_state_template):
    """
    Restore the latest checkpoint.
    
    Args:
        checkpoint_manager: The ocp.CheckpointManager
        train_state_template: An initialised TrainState used as the pytree 
                              structure for restoration (abstract_train_state)
    
    Returns:
        (train_state, metadata_dict) or (None, None) if no checkpoint exists
    """
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        print("  No checkpoint found — starting from scratch.")
        return None, None
    
    print(f"  Restoring checkpoint from step {latest_step} ...")
    restored = checkpoint_manager.restore(
        latest_step,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardRestore(train_state_template),
            metadata=ocp.args.JsonRestore(),
        ),
    )
    train_state = restored.train_state
    metadata = restored.metadata
    print(f"  ✓ Restored: step={metadata['global_step']}, epoch={metadata['epoch']}")
    return train_state, metadata


def load_config_from_checkpoint_dir(checkpoint_dir: str):
    """
    Load configuration from a checkpoint directory's config.json.

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


def try_restore_params_from_orbax(checkpoint_dir: str):
    """
    Try to restore train_state from an Orbax checkpoint directory and
    extract the model parameters.

    Returns:
        (params, metadata) or (None, None) if restoration fails.
    """
    import jax.numpy as jnp
    import traceback

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

        # The restored.train_state may be a nested dict.
        ts = restored.train_state
        metadata = getattr(restored, 'metadata', None)

        # params could be under ts['params'] or ts.params
        params = None
        if isinstance(ts, dict):
            if 'params' in ts:
                params = ts['params']
            elif ('train_state' in ts
                  and isinstance(ts['train_state'], dict)
                  and 'params' in ts['train_state']):
                params = ts['train_state']['params']
        else:
            params = getattr(ts, 'params', None)

        if params is None:
            keys_info = list(ts.keys()) if isinstance(ts, dict) else dir(ts)
            print(f"[warning] Could not find 'params' in restored train_state. Keys: {keys_info}")
            return None, metadata

        # Convert numpy arrays to jax arrays if needed
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)
        print(f"[info] Restored params from Orbax checkpoint (step={latest})")
        return params, metadata

    except Exception as e:
        print(f"[error] Failed to restore Orbax checkpoint: {e}")
        traceback.print_exc()
        return None, None
