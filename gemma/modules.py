# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Transformer sub-modules."""

import enum
from dataclasses import dataclass

from flax import linen as nn
from gemma import layers
from gemma import positional_embeddings
import jax
import jax.numpy as jnp
from flax import nnx

K_MASK = -2.3819763e38  # Set to a large negative number.
LayerCache = dict[str, jax.Array]


def _create_sliding_mask(
    segment_pos: jnp.ndarray,
    end_index: int,
    cache_len: int,
    sliding_window_size: int,
):
  """Creates mask for sliding window attention."""
  total_tokens = end_index + segment_pos.shape[1]  # cached + processing tokens

  def _reconstruct_rotated_cache_positions():
    cache_positions = jnp.arange(cache_len) + total_tokens - cache_len
    cache_positions = (
        jnp.zeros_like(cache_positions)
        # kv were placed at index (position_id % cache_len) in the cache.
        .at[cache_positions % cache_len].set(cache_positions)
    )
    return cache_positions

  # Reconstruct position_ids for cached kv.
  cache_positions = jax.lax.cond(
      total_tokens <= cache_len,
      lambda: jnp.arange(cache_len),
      _reconstruct_rotated_cache_positions,
  )

  cache_positions = cache_positions[None, None, :]  # [1, 1, cache_len]
  segment_pos = segment_pos[:, :, None]  # [B, seq_len, 1]
  sliding_mask = (cache_positions > segment_pos - sliding_window_size)
  sliding_mask *= (cache_positions < segment_pos + sliding_window_size)
  return sliding_mask


class AttentionType(enum.Enum):
  GLOBAL = 1
  LOCAL_SLIDING = 2


class Embedder(nnx.Module):
  """Embedder module."""
  def __init__(self, vocab_size: int, embed_dim: int, rngs: nnx.Rngs):
    self.vocab_size, self.embed_dim = vocab_size, embed_dim
    self.input_embedding = nnx.Param(nn.initializers.normal()(
      rngs(), (self.vocab_size, self.embed_dim)), names=("vocab", "features"))

  def encode(self, x: jax.Array) -> jax.Array:
    x = self.input_embedding.value[(x,)]
    x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
    return x

  def decode(self, x: jax.Array) -> jax.Array:
    return jnp.dot(x, self.input_embedding.value.T)


@dataclass
class Attention(nnx.Module):
  """Attention module."""

  num_heads: int
  num_kv_heads: int
  features: int
  head_dim: int
  attn_type: AttentionType
  query_pre_attn_scalar: float
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  rngs: nnx.Rngs | None = None
  
  def __post_init__(self):
    self.attn_vec_einsum = layers.Einsum(
        shape=(self.num_heads, self.head_dim, self.features),
        axis_names=("kv_heads", "head_dim", "features"),
        rngs=self.rngs,
    )

    if self.use_qkv_einsum:
      self.qkv_einsum = layers.Einsum(
          shape=(3, self.num_heads, self.features, self.head_dim),
          axis_names=(None, "kv_heads", "features", "head_dim"),
          rngs=self.rngs,
      )
    else:
      self.q_einsum = layers.Einsum(
          shape=(self.num_heads, self.features, self.head_dim),
          axis_names=("q_heads", "features", "head_dim"),
          rngs=self.rngs,
      )
      self.kv_einsum = layers.Einsum(
          shape=(2, self.num_kv_heads, self.features, self.head_dim),
          axis_names=(None, "kv_heads", "features", "head_dim"),
          rngs=self.rngs,
      )
    self.rngs = None

  @property
  def use_qkv_einsum(self):
    return self.num_kv_heads == self.num_heads

  @property
  def use_gqa(self):
    return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array]:
    seq_len = x.shape[1]

    if self.use_qkv_einsum:
      query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x)
    else:
      query_proj = self.q_einsum('BTD,NDH->BTNH', x)
      key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

    query_proj = positional_embeddings.apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
    )
    query_scaled = query_proj * self.query_pre_attn_scalar
    key_proj = positional_embeddings.apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
    )

    # Cache is left aligned.
    if cache is not None:
      def update_cache(cache_k, cache_v, key_proj, value_proj, end_index):
        slice_indices = (end_index % cache['v'].value.shape[1], 0, 0)
        value_proj = jax.lax.dynamic_update_slice(
            cache_v, value_proj, slice_indices,
        )
        key_proj = jax.lax.dynamic_update_slice(
            cache_k, key_proj, slice_indices
        )
        return key_proj, value_proj
      
      key_proj, value_proj = jax.vmap(update_cache)(
        cache['k'].value, cache['v'].value, key_proj, value_proj, 
        cache["end_index"].value)
        
      key_proj = nn.with_logical_constraint(key_proj, cache['k'].names)
      value_proj = nn.with_logical_constraint(value_proj, cache['v'].names)
        

      #end_index = cache['end_index'].value[0]
      #slice_indices = (0, end_index % cache['v'].value.shape[1], 0, 0)
      #value_proj = jax.lax.dynamic_update_slice(
      #    cache['v'].value,
      #    value_proj,
      #    slice_indices,
      #)
      #key_proj = jax.lax.dynamic_update_slice(
      #    cache['k'].value, key_proj, slice_indices
      #)

    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = query_scaled.shape
      query_scaled = query_scaled.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_scaled, key_proj)
      b, t, k, g, s = logits.shape
      logits = logits.reshape((b, t, k * g, s))
    else:
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError(
            'Sliding_window_size must be set if Local Sliding attention type'
        )
      sliding_mask = _create_sliding_mask(
          segment_pos,
          end_index=cache['end_index'].value[0] if cache is not None else 0,
          # Derive cache length from attn_mask shape in case cache is None
          cache_len=attn_mask.shape[-1],
          sliding_window_size=self.sliding_window_size,
      )
      attn_mask *= sliding_mask

    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)
    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = probs.shape
      probs = probs.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, value_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)

    if cache is not None:
      new_cache = {
          'v': nnx.Variable(value_proj, names=cache['v'].names),
          'k': nnx.Variable(key_proj, names=cache['k'].names),
          'end_index': nnx.Variable(cache['end_index'].value + seq_len, 
                                    names=cache['end_index'].names),
      }
    else:
      new_cache = None

    return new_cache, attn_output

  @classmethod
  def init_cache(
      cls,
      cache_size: int,
      num_heads: int,
      head_dim: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> LayerCache:
    del cls  # not used
    v = nnx.Variable(jnp.zeros(
      (batch_size, cache_size, num_heads, head_dim), dtype=dtype), 
      names=("batch", "sequence", "kv_heads", "head_dim"))
    k = nnx.Variable(jnp.zeros(
      (batch_size, cache_size, num_heads, head_dim), dtype=dtype), 
      names=("batch", "sequence", "kv_heads", "head_dim"))
    end_index = nnx.Variable(
      jnp.zeros((batch_size,), dtype=jnp.int32), names=("batch",))
    return {'v': v, 'k': k, 'end_index': end_index}


class FeedForward(nnx.Module):
  """Feed forward module."""
  def __init__(self, features: int, hidden_dim: int, 
               transpose_gating_einsum: bool, rngs: nnx.Rngs):
    self.features = features
    self.hidden_dim = hidden_dim
    self.transpose_gating_einsum = transpose_gating_einsum
    if self.transpose_gating_einsum:
      self.gating_einsum = nnx.Param(
        nnx.initializers.normal()(rngs(), (2, self.hidden_dim, self.features)), 
        names=(None, "features", "ffw"))
    else:
      self.gating_einsum = nnx.Param(
        nnx.initializers.normal()(rngs(), (2, self.features, self.hidden_dim)), 
        names=(None, "ffw", "features"))
    self.linear = nnx.Param(
      jnp.ones((self.hidden_dim, self.features)), names=("ffw", "features"))
    
  def __call__(self, x):
    # Some versions use an alternate parameter ordering that
    # transposes hidden_dim and features.
    gating_einsum_w = self.gating_einsum.value
    if self.transpose_gating_einsum:
      gating_einsum_w = gating_einsum_w.transpose((0, 2, 1))
    ff_gate = jnp.dot(x, gating_einsum_w[0])
    gate_value = nn.gelu(ff_gate)

    # Up projection
    ff1 = jnp.dot(x, gating_einsum_w[1])
    activations = gate_value * ff1

    # Down projection
    outputs = jnp.dot(activations, self.linear.value)

    return outputs


@dataclass
class Block(nnx.Module):
  """Transformer block."""

  num_heads: int
  num_kv_heads: int
  embed_dim: int
  head_dim: int
  hidden_dim: int
  use_post_attn_norm: bool
  use_post_ffw_norm: bool
  attn_type: AttentionType
  query_pre_attn_scalar: float
  transpose_gating_einsum: bool
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  rngs: nnx.Rngs | None = None
  
  def __post_init__(self):
    self.pre_attention_norm = layers.RMSNorm(self.embed_dim)
    self.attn = Attention(
        num_heads=self.num_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
        num_kv_heads=self.num_kv_heads,
        attn_type=self.attn_type,
        query_pre_attn_scalar=self.query_pre_attn_scalar,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        rngs=self.rngs,
    )
    self.post_attention_norm = None
    if self.use_post_attn_norm:
      self.post_attention_norm = layers.RMSNorm(self.embed_dim)

    self.pre_ffw_norm = layers.RMSNorm(self.embed_dim)
    self.mlp = FeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
        transpose_gating_einsum=self.transpose_gating_einsum,
        rngs=self.rngs,
    )
    self.post_ffw_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw_norm = layers.RMSNorm(self.embed_dim)
    self.rngs = None

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array]:
    inputs_normalized = self.pre_attention_norm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )
    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)
    attn_output += x
    outputs = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(outputs)
    if self.post_ffw_norm is not None:
      outputs = self.post_ffw_norm(outputs)
    outputs += attn_output
    return cache, outputs
