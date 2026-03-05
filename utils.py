import jax
import jax.numpy as jnp


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
