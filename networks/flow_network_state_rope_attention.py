import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant
from typing import Optional, Tuple

import functools
from typing import (Any, Callable, Optional, Tuple)
from flax.linen.dtypes import promote_dtype
from flax.linen import dot_product_attention
from flax.linen import initializers
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.linear import DotGeneralT
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
from jax import lax
from gemma.gm.math import _positional_embeddings


from attention_with_vis import dot_product_attention_with_weights

import einops

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

class PositionalEmbedding(nn.Module):
    dim_emb: int
    def setup(self):
        self.inv_freq = 1 / (10000 ** (jnp.arange(0.0, self.dim_emb, 2.0) / self.dim_emb))

    def __call__(self, pos_seq):
        sinusoid_inp = jnp.outer(pos_seq, self.inv_freq)
        pos_emb = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
        return pos_emb
def posemb_sincos(
    pos: jnp.array, embedding_dim: int, min_period: float, max_period: float
) -> jnp.array:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)

class Gating(nn.Module):
    #code taken from https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    d_input:int
    bg:float=0.
    @nn.compact
    def __call__(self,x,y):
        r = jax.nn.sigmoid(nn.Dense(self.d_input,use_bias=False)(y) + nn.Dense(self.d_input,use_bias=False)(x))
        z = jax.nn.sigmoid(nn.Dense(self.d_input,use_bias=False)(y) + nn.Dense(self.d_input,use_bias=False)(x) - self.param('gating_bias',constant(self.bg),(self.d_input,)))
        h = jnp.tanh(nn.Dense(self.d_input,use_bias=False)(y) + nn.Dense(self.d_input,use_bias=False)(r*x))
        g = (1 - z)* x + (z*  h)
        return g

class MultiHeadDotProductAttention(Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation
        (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
  """
  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
  use_bias: bool = True
  attention_fn: Callable[..., Array] = dot_product_attention_with_weights
  decode: bool = False
  qkv_dot_general: DotGeneralT = lax.dot_general
  out_dot_general: DotGeneralT = lax.dot_general

  @compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               cache_k: Optional[Array] = None,
               cache_v: Optional[Array] = None,
               cache_mask: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        DenseGeneral,
        axis=-1,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
        dot_general=self.qkv_dot_general,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(name='query')(inputs_q),
                         dense(name='key')(inputs_kv),
                         dense(name='value')(inputs_kv))

    key = jnp.concat([cache_k, key], axis=-3)
    value = jnp.concat([cache_v, value], axis=-3)
    n = query.shape[-3]
    causal_mask_actions=jnp.tril(jnp.ones((n, n), dtype=bool), k=0)[None,None,:,:]
    mask_actions=jnp.broadcast_to(causal_mask_actions, (query.shape[0], self.num_heads, query.shape[-3], query.shape[-3]))
    mask=jnp.concat([cache_mask, mask_actions], axis=-1)
    
    # mask=jnp.concat([cache_mask, jnp.ones((cache_mask.shape[:-2])+(query.shape[-3],)+(query.shape[-3],))], axis=-1)



    key=_positional_embeddings.apply_rope(key,positions=mask[:,0,0].cumsum(axis=-1)-1, base_frequency=10000)
    
    offset=cache_mask[:,0,0].sum(axis=-1, keepdims=True)

    query= _positional_embeddings.apply_rope(query,positions=offset+jnp.arange(query.shape[-3])[None,:].repeat(query.shape[0], axis=0), base_frequency=10000)


    # apply attention with weights output
    x, attn_weights = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    out = DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        dot_general=self.out_dot_general,
        name='out', # type: ignore[call-arg]
    )(x)
    return out, attn_weights
  


  class Gating(nn.Module):
    #code taken from https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    d_input:int
    bg:float=0.
    @nn.compact
    def __call__(self,x,y):
        r = jax.nn.sigmoid(nn.Dense(self.d_input,use_bias=False)(y) + nn.Dense(self.d_input,use_bias=False)(x))
        z = jax.nn.sigmoid(nn.Dense(self.d_input,use_bias=False)(y) + nn.Dense(self.d_input,use_bias=False)(x) - self.param('gating_bias',constant(self.bg),(self.d_input,)))
        h = jnp.tanh(nn.Dense(self.d_input,use_bias=False)(y) + nn.Dense(self.d_input,use_bias=False)(r*x))
        g = (1 - z)* x + (z*  h)
        return g


class transformer_layer(nn.Module):
    num_heads: int
    out_features: int
    qkv_features: int
    gating:bool =False
    gating_bias:float =0.
    norm_type: str = "rmsnorm"  # "rmsnorm" or "layernorm"
    post_attention_norm: bool = False  # Apply norm after attention residual
    post_mlp_norm: bool = False  # Apply norm after MLP residual (after dense2)

    def setup(self):
        self.attention1 = MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.qkv_features,
                                           out_features=self.out_features)

        # Choose normalization type based on norm_type parameter
        def make_norm():
            if self.norm_type == "layernorm":
                return nn.LayerNorm()
            else:  # default to rmsnorm
                return nn.RMSNorm()
        
        self.ln1 = make_norm()
        self.ln2 = make_norm()
        
        # Optional post-attention and post-MLP norms
        if self.post_attention_norm:
            self.ln_post_attn = make_norm()
        if self.post_mlp_norm:
            self.ln_post_mlp = make_norm()

        self.dense1 = nn.Dense(self.out_features)

        self.dense2 = nn.Dense(self.out_features)
        if(self.gating):
            self.gate1=Gating(self.out_features,self.gating_bias)
            self.gate2=Gating(self.out_features,self.gating_bias)



    def __call__(self, values_keys: jnp.ndarray, queries: jnp.ndarray, cache_mask: jnp.ndarray, cache_k: Optional[jnp.ndarray] = None, cache_v: Optional[jnp.ndarray] = None):

        ### Post norm

        #out_attention = queries+ self.attention1(inputs_kv=keys,inputs_q=queries,mask=mask)
        #out_attention = self.ln1(out_attention)

        #out = self.dense1(out_attention)
        #out = nn.activation.relu(out)
        #out = self.dense2(out_attention)

        #out = out + out_attention

        #out = self.ln2(out)

        #pre norm
        values_keys = self.ln1(values_keys)
        queries_n = self.ln1(queries)

        # forward cache to attention so keys/values can be concatenated inside MultiHeadDotProductAttention
        attention, attn_weights = self.attention1(inputs_kv=values_keys, inputs_q=queries_n, cache_mask=cache_mask, cache_k=cache_k, cache_v=cache_v)
        

        #Optional post-attention norm
        if self.post_attention_norm:
            attention = self.ln_post_attn(attention)

        if (self.gating):
            out_attention = self.gate1(queries, jax.nn.relu(attention))
        else:
            out_attention = queries + attention

        out_attention_n = self.ln2(out_attention)
        out = self.dense1(out_attention_n)
        out = nn.activation.gelu(out)
        #out = nn.activation.relu(out)
        out = self.dense2(out)

        # Optional post-MLP norm
        if self.post_mlp_norm:
            out = self.ln_post_mlp(out)

        if (self.gating):
            #out = self.gate2(out, jax.nn.relu(out_attention))
            out = self.gate2(out_attention, jax.nn.relu(out))
        else:
            out = out + out_attention

        

        return out, attn_weights


class TransformerFlow(nn.Module):
    """Transformer network that consumes a cache and a noise vector x and
    returns a flow of the same shape as x.

    Contract:
    - inputs:
        - cache: dict with optional keys 'key', 'value', 'mask' used by attention layers
        - x: jnp.ndarray, shape (batch, seq_len, dim) or (batch, dim)
        - mask: optional attention mask
    - output: jnp.ndarray same shape as x

    This module composes `num_layers` of `transformer_layer`.
    """

    num_layers: int = 2
    num_heads: int = 8
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    input_size: int = 512
    gating: bool = False
    gating_bias: float = 0.0
    norm_type: str = "rmsnorm"  # "rmsnorm" or "layernorm"
    post_attention_norm: bool = False  # Apply norm after attention residual
    post_mlp_norm: bool = False  # Apply norm after MLP residual (after dense2)
    def setup(self):


        self.embedding = nn.Dense(self.out_features, name='embedding')
        self.out_proj = nn.Dense(self.input_size, name='out_proj')
        self.pos_embed = PositionalEmbedding(self.out_features)

        self.action_time_mlp_in=nn.Dense(self.out_features)
        self.action_time_mlp_out=nn.Dense(self.out_features)

        self.embedding_state = nn.Dense(self.out_features, name='embedding_state')
        # create transformer layers and register them as attributes
        for i in range(self.num_layers):
            setattr(self, f"layer_{i}", transformer_layer(
                num_heads=self.num_heads,
                qkv_features=self.qkv_features ,
                out_features=self.out_features,
                gating=self.gating,
                gating_bias=self.gating_bias,
                norm_type=self.norm_type,
                post_attention_norm=self.post_attention_norm,
                post_mlp_norm=self.post_mlp_norm,
            ))


    def __call__(self,x,cache_mask, cache_k, cache_v,time,state ) -> jnp.ndarray:
        orig_shape = x.shape


        seq = self.embedding(x)

        state_emb = self.embedding_state(state)

        #time=jnp.broadcast_to(time, x.shape[0])
        time_emb = posemb_sincos(time, self.out_features, min_period=4e-3, max_period=4.0)

        #pos_emb = self.pos_embed(jnp.arange(x.shape[1]+1))


        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=x.shape[1])
        action_time_tokens = jnp.concatenate([seq, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = jax.nn.swish(action_time_tokens)
        seq = self.action_time_mlp_out(action_time_tokens)

        seq=jnp.concatenate([state_emb[:, None, :], seq], axis=1)

        seq = seq 
        # run through transformer layers created in setup
        all_attn_weights = []
        for i in range(self.num_layers):
            layer = getattr(self, f"layer_{i}")
            seq, attn_weights = layer(values_keys=seq, queries=seq, cache_mask=cache_mask, cache_k=cache_k[:,i], cache_v=cache_v[:,i])
            all_attn_weights.append(attn_weights)

        seq = self.out_proj(seq)    
        # ensure output shape matches input
        #assert seq.shape == orig_shape, f"TransformerFlow output shape {seq.shape} != input shape {orig_shape}"
        
        # Stack all attention weights from all layers into a single array
        # Shape: (batch, num_layers, num_heads, query_len, key_len)
        all_attn_weights = jnp.stack(all_attn_weights, axis=1)
        
        return seq[:,1:,:], all_attn_weights  # remove state token before returning output





# action_shape=4
# model=TransformerFlow(num_layers=2, num_heads=8, qkv_features=512, out_features=256, input_size=action_shape, gating=True, gating_bias=2.)


# x= jnp.ones((4,16,action_shape))
# cache_k=jnp.ones((4,2,32,8,64))
# cache_v=jnp.ones((4,2,32,8,64))
# cache_mask=jnp.ones((4,8,16,32))

# params=model.init(jax.random.PRNGKey(0),x, cache_mask, cache_k, cache_v,1.0)
# flow_fn=jax.jit(model.apply)
# output=flow_fn(params, x, cache_mask, cache_k, cache_v,1.0)
# print(output.shape)

# import time
# start=time.time()
# output=flow_fn(params, x, cache_mask, cache_k, cache_v,1.0)
# print(output)
# output=flow_fn(params, x, cache_mask, cache_k, cache_v,0.5)
# print(output)


# end=time.time()
# print("Time per forward pass:", (end-start)/2)

# def iter_fn(carry, _):
#     params, x, cache_mask, cache_k, cache_v,time,dt=carry
#     output=flow_fn(params, x, cache_mask, cache_k, cache_v,time)
#     x=x-output*dt
#     carry=(params, x, cache_mask, cache_k, cache_v,time-dt,dt)
#     return carry,None

# start=time.time()
# output,_=jax.lax.scan(iter_fn, (params, x, cache_mask, cache_k, cache_v,1.0,1/10), None, length=10)
# end=time.time()
# print("Time per forward pass:", (end-start)/10)
