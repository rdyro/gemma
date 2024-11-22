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
"""Base layers."""

from flax import linen as nn
import jax
import jax.numpy as jnp

def quantize_int8(w: jax.Array) -> jax.Array:
  reduction_axes = (-1, -2) if w.ndim >= 4 else (-1,)
  amax = jnp.max(jnp.abs(w), axis=reduction_axes, keepdims=True)
  scale = (amax / 127.0 + jnp.finfo(jnp.float16).tiny).astype(jnp.float16)
  w_quant = jnp.round(w / scale).astype(jnp.int8)
  return w_quant, scale

def dequantize_int8(w: jax.Array, scale: jax.Array):
  return (w * scale).astype(scale.dtype)

class Einsum(nn.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""
  shape: tuple[int, ...]
  axis_names: list[str]
  quantized: bool = False

  def setup(self):
    init_fn = nn.initializers.normal()

    if self.quantized:
      if len(self.shape) >= 4:
        s_shape = self.shape[:-2] + (1, 1)
        s_axis_names = self.axis_names[:-2] + (None, None)
      else:
        s_shape = self.shape[:-1] + (1,)
        s_axis_names = self.axis_names[:-1] + (None,)
      scale_init_fn = nn.with_logical_partitioning(nn.initializers.ones_init(),
                                                   s_axis_names)
      self.s = self.param('s', scale_init_fn, s_shape)

      # modify the weight init function to output int8
      base_init_fn = init_fn
      init_fn = lambda *args, **kw: quantize_int8(base_init_fn(*args, **kw))[0]

    weight_init_fn = nn.with_logical_partitioning(init_fn,
                                                  self.axis_names)
    self.w = self.param('w', weight_init_fn, self.shape)

  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    if self.quantized:
      return jnp.einsum(eqn, x, dequantize_int8(self.w, self.s))
    else:
      return jnp.einsum(eqn, x, self.w)

  def get_weight(self):
    if self.quantized:
      return dequantize_int8(self.w, self.s)
    else:
      return self.w


class RMSNorm(nn.Module):
  """RMSNorm layer."""

  @nn.compact
  def __call__(self, x):
    scale_init = nn.with_logical_partitioning(nn.initializers.zeros_init(),
                                              ("features",))
    scale = self.param('scale', scale_init, (x.shape[-1]))
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

    # Jax.lax.rsqrt is used because it returns different floats than
    # jnp.reciprocal(jnp.sqrt(var + 1e-06))
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

    # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
    # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
    # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
    scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
    normed_inputs = normed_inputs * (1 + scale)
    return normed_inputs
