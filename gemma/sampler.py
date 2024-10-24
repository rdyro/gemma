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
"""Sampler for Gemma transformer.

An example of a sampling class for a Gemma model.
"""
from collections.abc import Sequence
import dataclasses
import math
import functools
import logging
import time

import chex
from gemma import modules
from gemma import transformer as transformer_lib
from flax import nnx
from flax.linen import logical_axis_rules, logical_to_mesh_sharding
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import sentencepiece as spm

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

# we'll pad static dimensions to the nearest power of two to limit recompilation
next_power_of_two = lambda i: 2 ** math.ceil(math.log2(i))

@chex.dataclass
class SamplingState:
  """Internal sampling state."""
  # padding id of the tokenizer
  pad_id: jnp.int32
  # end of sequence id
  eos_id: jnp.int32

  # Decoding step.
  decoding_step: jnp.int32

  # Number of tokens in the prompt.
  num_input_tokens: jnp.ndarray  # [B]

  # Fixed-size buffer for accumulating the output tokens.
  token_buffer: jnp.ndarray  # [B, L]

  # Position indices, based on ignoring pad tokens.
  positions: jnp.ndarray  # [B, L]

  # Model state for conditioning the model on autoregressively.
  cache: dict[str, modules.LayerCache]

  # Is decoding done on the given sequence?
  done: jnp.ndarray  # [B]

  # Total sampling steps (including the prompt).
  total_sampling_steps: int

  # Fixed-size buffer for accumulating the output logits.
  logits_buffer: jnp.ndarray | None = None  # [B, L, V]

  # List of tokens that are forbidden to be generated.
  forbidden_token_ids: Sequence[int] | None = None


@dataclasses.dataclass
class SamplerOutput:

  # Decoded samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: list[list[float]]

  # Tokens corresponding to the generated samples.
  tokens: list[list[int]]
  
def _compute_attention_masks(
    time_step: jax.Array, seq_len: int, input_mask: jax.Array
) -> jax.Array:
  """Computes causal attention mask."""
  bsz = input_mask.shape[0]
  batch_time_step = jnp.full((bsz, 1), time_step, dtype=jnp.uint32)
  causal_mask = jnp.less_equal(
      jnp.expand_dims(jnp.arange(seq_len), 0), batch_time_step
  )
  max_seq_len = min(input_mask.shape[-1], seq_len)
  input_mask = jax.lax.dynamic_slice(
      input_mask,
      (0, jnp.maximum(time_step - seq_len + 1, 0)),
      (bsz, max_seq_len),
  )
  input_mask = (
      jnp.ones((bsz, seq_len), dtype=jnp.bool_)
      .at[:, :max_seq_len]
      .set(input_mask)
  )

  causal_mask = jnp.logical_and(causal_mask, input_mask)
  attention_mask = causal_mask[:, jnp.newaxis, :].astype(jnp.bool_)

  return attention_mask

@functools.partial(jax.jit, static_argnums=(0, 1))
def casual_attention_mask(seq_len: int, max_seq_len: int) -> jax.Array:
  return jnp.arange(seq_len)[..., None] >= jnp.arange(max_seq_len)[None, ...]

@functools.partial(jax.jit, static_argnums=(1,))
def construct_positions_and_attn_mask(input: jax.Array, max_len: int,
                                      pad_id: int
                                      ) -> tuple[jax.Array, jax.Array]:
  """Given a (left and/or right) padded input, create positions and attn_mask"""
  assert input.ndim == 2 and input.shape[-1] <= max_len
  input_len = input.shape[-1]
  input = input.astype(jnp.int32)
  input_mask = input != pad_id
  # positions are zero-indexed, cumsum gives one-indexed values
  positions = ((jnp.cumsum(input_mask, axis=-1, dtype=jnp.int32) - 1)
               * input_mask)
  attention_mask = casual_attention_mask(input_len, max_len)[None, ...]
  pad_len = max(0, max_len - input_len)
  padded_input_mask = jnp.pad(input_mask, [(0, 0), (0, pad_len)])
  attention_mask = attention_mask * (input_mask[..., None]
                                     * padded_input_mask[..., None, :])
  return positions, attention_mask

@functools.partial(nnx.jit, static_argnums=(3,))
def prefill(transformer: transformer_lib.Transformer, inputs: jax.Array,
            pad_id: int, max_len: int):
  """Ingest context, prefilling the cache for auto-regressive decoding."""
  dtype = jax.tree.leaves(nnx.state(transformer))[0].dtype
  config = dataclasses.replace(transformer.config, max_cache_length=max_len)
  transformer.config = config
  cache = config.init_cache(inputs.shape[0], dtype=dtype)
  positions, attention_mask = construct_positions_and_attn_mask(
    inputs, config.max_cache_length, pad_id)
  logits, cache = transformer(inputs, positions, cache, attention_mask)
  return logits, nnx.state(cache)

def sample_step(transformer: transformer_lib.Transformer,
                sampler_state: SamplingState) -> SamplingState:
  """Perform a single sampling step."""
  static_max_len = sampler_state.token_buffer.shape[-1]

  decoding_step = sampler_state.decoding_step
  last_token = sampler_state.token_buffer[:, decoding_step - 1][..., None]
  input_mask = (sampler_state.token_buffer != sampler_state.pad_id)[:, None, :]
  causal_mask = (jnp.arange(static_max_len) < decoding_step)[None, None, :]
  attention_mask = input_mask & causal_mask
  positions = sampler_state.positions[:, decoding_step][..., None]

  logits, cache = transformer(last_token, positions, sampler_state.cache,
                              attention_mask)
  if sampler_state.forbidden_token_ids:
    logits = logits.at[:, :, sampler_state.forbidden_token_ids].set(-jnp.inf)

  next_token_candidate = jnp.argmax(logits, axis=-1)  # [B, 1]
  next_token_candidate = next_token_candidate[:, 0]  # [B,]
  token_buffer = sampler_state.token_buffer.at[:, decoding_step].set(
      next_token_candidate
  )

  if sampler_state.logits_buffer is not None:
    next_logits = logits[:, 0, ...]
    logits_buffer = sampler_state.logits_buffer.at[:, decoding_step].set(
        next_logits
    )
  else:
    logits_buffer = sampler_state.logits_buffer

  done = sampler_state.done | (
    token_buffer[:, decoding_step] == sampler_state.eos_id)

  return dataclasses.replace(sampler_state,
                             decoding_step=sampler_state.decoding_step + 1,
                             token_buffer=token_buffer,
                             logits_buffer=logits_buffer,
                             done=done,
                             cache=nnx.state(cache))

@functools.partial(nnx.jit, donate_argnums=(1,))
def sample_fn(transformer: transformer_lib.Transformer,
              initial_sampling_state: SamplingState) -> SamplingState:
  """Auto-regressive sampling until all sequences reach eos token."""

  def sample_with_params(sampler_state: SamplingState):
    return sample_step(transformer, sampler_state)

  def cond_fn(sampler_state: SamplingState):
    step = sampler_state.decoding_step
    total_steps = sampler_state.total_sampling_steps
    return (step < total_steps) & jnp.any(jnp.logical_not(sampler_state.done))

  return jax.lax.while_loop(
      cond_fn, sample_with_params, initial_sampling_state
  )

@jax.jit
def mask_tokens_after_eos_ids(token_buffer, pad_id: int, eos_id: int):
  """Mask token IDs after the EOS token with the padding ID."""
  eos_exists = jnp.any(jnp.equal(token_buffer, eos_id), axis=-1)
  eos_indices = jnp.where(
      eos_exists,
      jnp.argmax(jnp.equal(token_buffer, eos_id), axis=-1),
      token_buffer.shape[-1],
  )
  mask = jnp.less_equal(
      jnp.arange(token_buffer.shape[-1]), eos_indices[:, None]
  )
  return token_buffer * mask + pad_id * (1 - mask)

is_param = lambda x: isinstance(x, nnx.VariableState)

class Sampler:
  """Sampler for gemma transformer."""

  def __init__(
      self,
      transformer: transformer_lib.Transformer,
      vocab: spm.SentencePieceProcessor,
      mesh: Mesh | None = None,
      rules: list[tuple[str, list[str] | str]] | None = None,
  ):
    """Initializes a sampler for a Gemma model.

    Args:
      transformer: an instance of the Gemma transformer.
      vocab: vocabulary of the given model.
      params: weights of the model.
    """
    self.transformer = transformer
    self.transformer_graphdef, self.transformer_params = nnx.split(transformer)
    self.vocab = vocab
    self.mesh, self.rules = mesh, rules

    if (self.mesh is None) ^ (self.rules is None):
      raise ValueError("When specifying a mesh, you must also logical"
                       "sharding rules.")

    if self.mesh is None or self.rules is None:
      # create a transformer-params prefill function
      def _prefill_fn(transformer_params: nnx.State, inputs: jax.Array,
                      max_len: int):
        transformer = nnx.merge(self.transformer_graphdef, transformer_params)
        return prefill(transformer, inputs, self.vocab.pad_id(), max_len)
    else:
      # construct shardings for cache and output logits
      example_cache = nnx.state(self.transformer.config.init_cache(0))
      cache_specs = jax.tree.map(lambda x: P(*x.names), example_cache,
                                 is_leaf=is_param)
      self.cache_shardings = logical_to_mesh_sharding(cache_specs, mesh, rules)
      logit_specs = P("batch", "sequence", "vocab")
      self.logits_sharding = logical_to_mesh_sharding(logit_specs, mesh, rules)
      out_shardings = (self.logits_sharding, self.cache_shardings)

      # create a sharding consraint enforcing prefill function
      @functools.partial(jax.jit, static_argnums=(2,),
                         out_shardings=out_shardings)
      def _prefill_fn(transformer_params: nnx.State, inputs: jax.Array,
                   max_len: int):
        transformer = nnx.merge(self.transformer_graphdef, transformer_params)
        with self.mesh, logical_axis_rules(self.rules):
          return prefill(transformer, inputs, self.vocab.pad_id(), max_len)

    self.sample_fn, self.prefill_fn = sample_fn, _prefill_fn


  @property
  def dtype(self) -> jnp.dtype:
    return jax.tree_util.tree_leaves(nnx.state(self.transformer))[0].dtype

  def init_sample_state(
      self,
      all_input_ids: list[list[int]],
      total_sampling_steps: int,
      include_logits: bool = False,
      forbidden_token_ids: Sequence[int] | None = None,
  ) -> SamplingState:
    """Initializes the sampling state given input prompts."""

    bsz = len(all_input_ids)
    static_max_len = next_power_of_two(total_sampling_steps)
    num_input_tokens = [len(input_ids) for input_ids in all_input_ids]
    max_input_len = max(num_input_tokens)
    buffer_size = total_sampling_steps + 1


    # right align input sequences (left-pad)
    right_aligned_tokens = jnp.array([
      [self.vocab.pad_id()] * (max_input_len - num_tokens) + input
      for (input, num_tokens) in zip(all_input_ids, num_input_tokens)],
      dtype=jnp.int32)
    token_buffer = self.vocab.pad_id() * jnp.ones(
      (bsz, static_max_len), dtype=jnp.int32)
    token_buffer = token_buffer.at[:, :max_input_len].set(right_aligned_tokens)

    # pad inputs further to a power of 2 length to reduce prefill recompilations
    pad_after = min(buffer_size, next_power_of_two(max_input_len)
                    ) - right_aligned_tokens.shape[-1]
    padded_tokens = jnp.pad(right_aligned_tokens, ((0, 0), (0, pad_after)))

    # fill in the cache during prefill stage
    t = time.time()
    logits, cache = self.prefill_fn(self.transformer_params, padded_tokens,
                                    static_max_len)
    logits.block_until_ready()
    t = time.time() - t
    logger.debug(f"Prefill took {t:.4e} s")

    # set the cache end_index to right aligned size, not the power of 2 padded
    for lid in cache.keys():
      cache[lid]["end_index"].value = (
        cache[lid]["end_index"].value.at[:].set(max_input_len))

    if include_logits:
      logits_buffer = jnp.zeros((bsz, static_max_len, logits.shape[-1]),
                                dtype=self.dtype)
      logits_buffer = logits_buffer.at[:, :padded_tokens.shape[-1], :].set(
        logits)
    else:
      logits_buffer = None

    # compute token positions, skipping padded tokens
    pad_after = static_max_len - right_aligned_tokens.shape[-1]
    positions = jnp.concatenate([
      (right_aligned_tokens != self.vocab.pad_id()).astype(jnp.int32),
      jnp.ones((bsz, pad_after), dtype=jnp.int32)], -1)
    positions = jnp.cumsum(positions, -1) - 1

    return SamplingState(
        pad_id=self.vocab.pad_id(),
        eos_id=self.vocab.eos_id(),
        decoding_step=max_input_len,
        num_input_tokens=jnp.array(num_input_tokens, dtype=jnp.int32),
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        cache=cache,
        done=jnp.zeros((bsz,), dtype=jnp.bool),
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
    )

  def tokenize(self, input_string: str) -> list:
    """Tokenizes the input string."""
    input_ids = self.vocab.EncodeAsIds(input_string)
    input_ids = [self.vocab.bos_id()] + input_ids
    return input_ids

  def __call__(
      self,
      input_strings: Sequence[str],
      total_generation_steps: int,
      apply_chat_template: bool,
      echo: bool = False,
      return_logits: bool = True,
      forbidden_tokens: Sequence[str] | None = None,
  ) -> SamplerOutput:
    """Samples a completion of the input string.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      total_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      echo: whether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      forbidden_tokens: list of tokens that are forbidden to be generated. Each
        token must map to a single token id in the vocab.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """

    chat_template = ("<start_of_turn>user\n{}\n<end_of_turn>\n"
                     "<start_of_turn>model\n")
    if apply_chat_template:
      input_strings = [chat_template.format(x) for x in input_strings]

    forbidden_token_ids = None
    if forbidden_tokens is not None:
      forbidden_token_ids = []
      for token in forbidden_tokens:
        token_id = self.vocab.EncodeAsIds(token)
        if len(token_id) != 1:
          raise ValueError("Forbidden tokens must map to single token ids in"
                           " the vocab.")
        forbidden_token_ids.extend(token_id)
      forbidden_token_ids = tuple(forbidden_token_ids)
    all_input_ids = [self.tokenize(x) for x in input_strings]
    max_input_length = max(len(input_ids) for input_ids in all_input_ids)
    total_sampling_steps = max_input_length + total_generation_steps

    # initialization including prefill
    t_init = time.time()
    initial_sampling_state = self.init_sample_state(
        all_input_ids,
        include_logits=return_logits,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
    )
    inital_size = initial_sampling_state.decoding_step
    t_init = time.time() - t_init
    logger.debug(f"Initialization took {t_init:.4e} s")

    # autoregressive sampling
    t_sampling = time.time()
    sampling_state = self.sample_fn(self.transformer, initial_sampling_state)
    t_sampling = time.time() - t_sampling
    logger.debug(f"AR Sampling took {t_sampling:.4e} s")

    # compute throughput
    sampled_steps = int(sampling_state.decoding_step - inital_size)
    logger.debug(f"Throughput: tok / sec: {sampled_steps / t_sampling:.3f} for"
                 f" {sampled_steps = }")

    masked_token_buffer = mask_tokens_after_eos_ids(
      sampling_state.token_buffer, sampling_state.pad_id, sampling_state.eos_id)

    num_longest = jnp.max(sampling_state.num_input_tokens)
    out_tokens, out_logits = [], []
    for i, (token_buffer, num_tokens) in enumerate(
        zip(masked_token_buffer, sampling_state.num_input_tokens)
    ):
      # start idx is the last 0 position
      start_idx = num_longest - num_tokens + (0 if echo else num_tokens)
      out_tokens.append(token_buffer[start_idx:total_sampling_steps])
      if return_logits:
        logits_buffer = sampling_state.logits_buffer[i]
        out_logits.append(logits_buffer[start_idx:total_sampling_steps])

    decoded_outputs = [self.vocab.DecodeIds(tokens.tolist())
                       for tokens in out_tokens]

    result = SamplerOutput(text=decoded_outputs, logits=out_logits,
                           tokens=out_tokens)
    return result
