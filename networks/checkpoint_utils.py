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
