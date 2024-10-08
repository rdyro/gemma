import os
import functools
from pprint import pprint
from pathlib import Path
import math
from typing import Any

# to observe the actual memory getting somewhat conservatively allocated
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import linen as nn

from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib
import sentencepiece as spm
import kagglehub
# kagglehub.login() # you might need to log in

cpu_device, devices = jax.devices("cpu")[0], jax.devices("cuda")
pprint([f"{k} = {v}" for k, v in os.environ.items() if k.startswith("XLA")])

jax.config.update("jax_compilation_cache_dir", 
                  str(Path("~/.cache/jax_compilation_cache").expanduser()))

# variants version v1 have gemma/Flax
# variant = '2b-it' # @param ['2b', '2b-it', '7b', '7b-it'] {type:"string"}
# weights_dir = kagglehub.model_download(f'google/gemma/Flax/{variant}')

variant = "gemma2-2b-it"
weights_dir = kagglehub.model_download(f"google/gemma-2/flax/{variant}")
print(weights_dir)
ckpt_path = os.path.join(weights_dir, variant)
vocab_path = os.path.join(weights_dir, 'tokenizer.model')
vocab = spm.SentencePieceProcessor()
vocab.Load(vocab_path)
PAD_ID = vocab.pad_id()


@functools.partial(jax.jit, static_argnums=(0, 1))
def casual_attention_mask(seq_len: int, max_seq_len: int) -> jax.Array:
  return jnp.arange(seq_len)[..., None] >= jnp.arange(max_seq_len)[None, ...]

@functools.partial(jax.jit, static_argnums=(1, 2))
def construct_positions_and_attn_mask(input: jax.Array, max_len: int, 
                                      pad_id: int = PAD_ID
                                      ) -> tuple[jax.Array, jax.Array]:
  assert input.ndim == 2 and input.shape[-1] <= max_len
  input_len = input.shape[-1]
  input = input.astype(jnp.int32)
  input_mask = input != pad_id
  # positions are zero-indexed, cumsum gives one-indexed values
  positions = ((jnp.cumsum(input_mask, axis=-1, dtype=jnp.int32) - 1) 
               * input_mask)
  attention_mask = casual_attention_mask(input_len, max_len)[
    None, ...]
  pad_len = max(0, max_len - input_len)
  padded_input_mask= jnp.pad(input_mask, [(0, 0), (0, pad_len)])
  attention_mask = attention_mask * (input_mask[..., None] 
                                     * padded_input_mask[..., None, :])
  return positions, attention_mask


# alternatively: 
# with jax.default_device(cpu_device):
#   params_host = params_lib.load_and_format_params(ckpt_path)
# config = transformer_lib.TransformerConfig.from_params(
#     params_host,
#     cache_size=128  # Number of time steps in the transformer's cache
# )
config = transformer_lib.TransformerConfig.gemma2_2b(cache_size=128)
transformer = transformer_lib.Transformer(config)

is_param = lambda x: isinstance(x, nn.LogicallyPartitioned)

def init_params(batch_size: int, dtype=jnp.bfloat16):
  input_len = 1 # or 1 or 7, this dimension doesn't matter in initialization
  random_key = random.key(0)
  input_sequence = jnp.zeros((batch_size, input_len), dtype=jnp.int32)
  positions, attn_mask = construct_positions_and_attn_mask(
    input_sequence, config.max_cache_length)
  cache = config.init_cache(batch_size, jnp.float32, logically_partitioned=True)
  cache_value = jax.tree.map(lambda x: x.value if is_param(x) else x, cache, 
                             is_leaf=is_param)
  params = transformer.init(random_key, input_sequence, positions, 
                            cache_value, attn_mask)
  return (params, cache)
  
# we use jax.eval_shape to get just the shape of the parameters for sharding
BATCH_SIZE = 1
params_struct, cache_struct = jax.eval_shape(lambda: init_params(BATCH_SIZE))

axis_names = jax.tree.reduce(lambda x, y: x | set(y.names),
        (params_struct, cache_struct), initializer=set(), is_leaf=is_param)
print(f"logical axis names = {axis_names}")
model_parallel_rules = {
  None: None, 
  "batch": None,
  "sequence": None,
  "vocab": None, 
  "features": "x",  # or 'd_model'
  "q_heads": None, 
  "kv_heads": None, 
  "head_dim": None, 
  "ffw": None
}
assert all(k in model_parallel_rules for k in axis_names)

mesh = Mesh(devices, ("x",))
rules = model_parallel_rules
params_sharding, cache_sharding = jax.tree.map(
  lambda x: NamedSharding(mesh, P(*[rules[name] for name in x.names])),
  (params_struct, cache_struct), is_leaf=is_param)

@functools.partial(jax.jit, static_argnums=(0, 1), 
                   out_shardings=(params_sharding, cache_sharding))
def unpack_params(batch_size: int, dtype=jnp.bfloat16
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
  to_dtype = (lambda x: x.astype(dtype) 
              if jnp.issubdtype(x.dtype, jnp.floating) else x)
  # the model is initialized in float32, we don't have much of a choice
  # we could generate our own random weights from the model weight shapes
  # but we want to use the initializers that the model authors used
  params_cache = jax.tree.map(to_dtype, init_params(batch_size))
  # unpack the parameters from the nn.LogicallyPartitioned wrapper
  return jax.tree.map(lambda x: x.value, params_cache, is_leaf=is_param)


LOAD_PRETRAINED_PARAMETERS = True
DTYPE = jnp.bfloat16

if LOAD_PRETRAINED_PARAMETERS:
  with jax.default_device(cpu_device):
    params_host = params_lib.load_and_format_params(ckpt_path)
    params = jax.tree.map(lambda x, y: jax.device_put(x.astype(DTYPE), y), 
                          {"params": params_host["transformer"]}, params_sharding)
else:                
  params, cache = unpack_params(BATCH_SIZE)

def get_size(arr: jax.Array):
  shard_shapes = [shard.data.shape for shard in arr.addressable_shards]
  global_shape = arr.shape
  per_host_local_size = sum(math.prod(local_shape) * arr.dtype.itemsize 
                            for local_shape in shard_shapes)
  per_device_local_size = math.prod(shard_shapes[0]) * arr.dtype.itemsize
  global_size = math.prod(global_shape) * arr.dtype.itemsize
  return (global_size, per_host_local_size, per_device_local_size)
  
def reduce_size(size1, size2):
  return tuple(x1 + x2 for (x1, x2) in zip(size1, size2))

global_size, per_host_size, per_device_size = jax.tree.reduce(
  reduce_size, jax.tree.map(get_size, params), initializer=(0, 0, 0), 
  is_leaf=lambda a: isinstance(a, tuple))
print(f"Global model size in bytes:     {global_size / (1024 ** 3):.4f} GB")
print(f"Per-host model size in bytes:   {per_host_size / (1024 ** 3):.4f} GB")
print(f"Per-device model size in bytes: {per_device_size / (1024 ** 3):.4f} GB")

# visualize the sharding on an example layer
jax.debug.visualize_array_sharding(params["params"]["layer_0"]["mlp"]["linear"])


@functools.partial(jax.jit, static_argnames=("config",), 
                   in_shardings=(params_sharding,  None))
def prefill(params: dict[str, Any], input: jax.Array, 
            config: transformer_lib.TransformerConfig):
  assert input.ndim == 2
  batch_size, input_len = input.shape
  max_len: int = config.max_cache_length
  positions, attention_mask = construct_positions_and_attn_mask(input, max_len)
  dtype = jax.tree.flatten(params)[0][0].dtype
  cache = jax.lax.with_sharding_constraint(
    config.init_cache(batch_size, dtype=dtype), cache_sharding)
  logits, cache = transformer.apply(params, input, positions, cache, 
                                    attention_mask)
  return logits, cache


def id_from_str(text: str):
  tokens = vocab.Encode(text)
  assert len(tokens) == 1 and isinstance(tokens[0], int)
  return tokens[0]

sot_id, eot_id = id_from_str("<start_of_turn>"), id_from_str("<end_of_turn>")
user_id, model_id = id_from_str("user"), id_from_str("model")
newline_id = id_from_str("\n")

def right_align_sequences(inputs: list[str]) -> jax.Array:
  chat_template = ("<start_of_turn>user\n{}\n<end_of_turn>\n" 
                   "<start_of_turn>model\n")
  encoded = [vocab.Encode(chat_template.format(input)) for input in inputs]
  max_len = max([len(x) for x in encoded])
  
  # NEED TO add bos at the beginning or the model will give very bad results
  return jnp.array([[PAD_ID] * max(0, max_len - len(x)) + [vocab.bos_id()] + x 
                    for x in encoded])
  
batch_input = right_align_sequences(["hello, how are you?", 
                                     "The weather today is"])

# let's explore how right-aligned sequences have their mask generated
input = jnp.asarray(batch_input, dtype=jnp.int32)
positions, attn_mask = construct_positions_and_attn_mask(input, input.shape[-1])

print(f"example batch_input =\n{batch_input}")
print(f"example positions =\n{positions}")
print(f"example attn_mask =\n{attn_mask * 1}")

@functools.partial(jax.jit, static_argnames=("config", "max_len"), 
                   in_shardings=(params_sharding, cache_sharding, None))
def decode(params: dict[str, Any], cache: dict[str, Any], input: jax.Array, 
            config: transformer_lib.TransformerConfig, max_len: int = -1):
  if max_len < 0:
    max_len = config.max_cache_length
  assert max_len <= config.max_cache_length

  idx = input.shape[-1]
  tokens = jnp.ones(input.shape[:-1] + (max_len,), 
                    dtype=jnp.int32) * (PAD_ID + 1)
  tokens = tokens.at[..., :idx].set(input)

  # construct a position and attention mask for autoregressive decoding
  positions, attn_mask = construct_positions_and_attn_mask(
    tokens, max_len=config.max_cache_length)

  # a decode step, carry = (current_tokens, all_tokens, autoregressive_cache)
  def _decode_step(carry, i):
    tokens, cache = carry
    last_tokens = tokens[..., i-1][..., None]
    curr_positions = positions[..., i-1][..., None]
    curr_attn_mask = attn_mask[..., i-1, :][..., None, :]
    logits, cache = transformer.apply(params, last_tokens, curr_positions, 
                                      cache, curr_attn_mask)                                     
    next_tokens = jnp.argmax(logits, -1)[..., 0]
    tokens = tokens.at[..., i].set(next_tokens)
    carry = (tokens, cache)
    return carry, 0
  
  autoreg_idxs = jnp.arange(idx, max_len)
  # pseudocode for scan:
  # carry_last, stacked_partial_results = scan(
  #       lambda carry, i: function(carry, i), carry_init, 
  #       scanned_array = range(x, y))
  (new_tokens, _), _ = jax.lax.scan(_decode_step, (tokens, cache), autoreg_idxs)
  return new_tokens


logits, prefilled_cache = prefill(params, batch_input, config)
new_tokens = decode(params, prefilled_cache, batch_input, config, 128)

for i in range(batch_input.shape[0]):
  print(f"Prompt {i}: `{vocab.Decode(batch_input[i, :].tolist())}`")
print("#" * 80)
for i in range(new_tokens.shape[0]):
  print(f"Full Response {i}:\n```\n"
        f"{vocab.Decode(new_tokens[i, :].tolist())}\n```")
  print("-" * 80)