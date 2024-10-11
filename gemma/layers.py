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
from flax import nnx
import jax
import jax.numpy as jnp


class Einsum(nnx.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""
  def __init__(self, shape: tuple[int, ...], axis_names: list[str] | None = None, 
               rngs: nnx.Rngs | None = None):
    assert rngs is not None
    self.w = nnx.Param(nnx.initializers.normal()(rngs(), shape), 
                       names=axis_names)

  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    return jnp.einsum(eqn, x, self.w.value)


class RMSNorm(nnx.Module):
  """RMSNorm layer."""
  def __init__(self, features: int):
    self.scale = nnx.Param(jnp.zeros((features,)), names=("features",))

  def __call__(self, x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

    # Jax.lax.rsqrt is used because it returns different floats than
    # jnp.reciprocal(jnp.sqrt(var + 1e-06))
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

    # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
    # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
    # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
    scale = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
    normed_inputs = normed_inputs * (1 + scale)
    return normed_inputs
