import os
import functools
import time
from typing import Callable, Any
import dataclasses
from pathlib import Path
import logging
import random as pyrandom

# to observe the actual memory getting somewhat conservatively allocated
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P
from flax import linen as nn
import numpy as np
from tqdm import tqdm

from gemma import params as params_lib
from gemma import transformer as transformer_lib
from gemma import sampler as sampler_lib
from gemma.layers import quantize_int8
import sentencepiece as spm
import kagglehub

from datasets import load_dataset
from grain import python as grain
#kagglehub.login() # you might need to log in

cpu_device, compute_devices = jax.devices("cpu")[0], jax.devices("cuda")
jax.config.update("jax_compilation_cache_dir", 
                  str(Path("~/.cache/jax_compilation_cache").expanduser()))
sampler_logger = logging.getLogger(sampler_lib.__name__)


VARIANT = "gemma2-27b-it"
QUANTIZE_FFW = True
PROFILE = False

weights_dir = Path("~/storage").expanduser()
ckpt_path = os.environ["MODEL_CHECKPOINT_PATH"]
vocab_path = os.environ["TOKENIZER_PATH"]
(vocab := spm.SentencePieceProcessor()).Load(vocab_path)

BATCH_SIZE = 4
MAX_LENGTH = 2048
prompt = """
The following are multiple choice questions (with answers) about {}.

{}
(A) {}
(B) {}
(C) {}
(D) {}
"""

CHAT_TEMPLATE = ("<start_of_turn>user\n{}\n<end_of_turn>\n"
                 "<start_of_turn>model\n")
ANSWER_TOKENS = jnp.array([vocab.tokenize(z) for z in ["A", "B", "C", "D"]]
                          ).reshape(-1)

def format_question(example, tokenize: bool = False, 
                    chat_template: str | None = None, 
                    append_answer_template: bool = False):
  text = prompt.format(example["subject"], example["question"], 
                       *example["choices"]).strip() + " "
  if chat_template is not None:
    text = chat_template.format(text) 
    if append_answer_template:
      text += "Answer: "
  if tokenize:
    tokenized_text = [vocab.bos_id()] + vocab.tokenize(text)
    assert len(tokenized_text) <= MAX_LENGTH
    num_tokens = len(tokenized_text)
    tokenized_text = np.array(tokenized_text 
                              + [vocab.pad_id()] * (2048 - len(tokenized_text)), 
                              dtype=np.int32)
    return tokenized_text, num_tokens, example["answer"]
  else:
    return text, example["answer"]


@dataclasses.dataclass
class MapFn(grain.MapTransform):
  map_function: Callable
  
  def map(self, el):
    return self.map_function(el)

def get_dataloader(ds_key: str = "test"):
  ds = load_dataset("cais/mmlu", "all")
  operations = []
  operations.append(MapFn(lambda x: format_question(
    x, tokenize=True, chat_template=CHAT_TEMPLATE)))
  operations.append(grain.Batch(batch_size=BATCH_SIZE))
  sampler = grain.IndexSampler(
    len(ds[ds_key]), shard_options=grain.NoSharding(), shuffle=True, 
    seed=time.time_ns() % 2 ** 31)
  dl = grain.DataLoader(data_source=ds[ds_key], sampler=sampler, 
                          worker_count=0, operations=operations)
  num_batches = len(ds[ds_key]) // BATCH_SIZE
  return dl, num_batches


is_param = lambda x: isinstance(x, nn.LogicallyPartitioned)

def eval_model_abstract(config):
  transformer = transformer_lib.Transformer(config)
  tokens = jnp.ones((BATCH_SIZE, 1), dtype=jnp.int32)
  postions, attention_mask = tokens, tokens
  cache_abstract = config.init_cache(BATCH_SIZE, logically_partitioned=True)
  cache = jax.tree.map(lambda x: x.value, cache_abstract, 
                       is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned))
  state_abstract = transformer.init(random.key(0), tokens, postions, cache,
                                   attention_mask=attention_mask)
  return state_abstract, cache_abstract

@functools.partial(jax.jit, static_argnames=("config", "decode_steps"))
def transformer_generate(params, batch, config, decode_steps: int = 10):
  transformer = transformer_lib.Transformer(config)
  with jax.named_scope("prefill"):
    input, num_tokens, _ = batch
    positions = jnp.broadcast_to(jnp.arange(input.shape[-1])[None, :], 
                                input.shape)
    causal_mask = positions[..., None, :] <= positions[..., :, None]
    attention_mask = (((input != vocab.pad_id())[..., None] 
                      & (input != vocab.pad_id())[..., None, :]) 
                      & causal_mask)
    positions = nn.with_logical_constraint(positions, ("batch", None))
    causal_mask = nn.with_logical_constraint(causal_mask, ("batch", None, None))
    attention_mask = nn.with_logical_constraint(attention_mask, 
                                                ("batch", None, None))
      
    cache = dataclasses.replace(config, max_cache_length=MAX_LENGTH).init_cache(
      BATCH_SIZE, dtype=jnp.bfloat16, logically_partitioned=True)
    cache = jax.tree.map(lambda x: nn.with_logical_constraint(x.value, x.names),
                         cache, is_leaf=is_param)
    logits, cache = transformer.apply(params, input, positions, cache=cache, 
                                      attention_mask=attention_mask)
    for l in cache.keys():
      assert cache[l]["end_index"].shape == num_tokens.shape
      cache[l]["end_index"] = num_tokens - 1
    
  last_tokens = jax.vmap(lambda x, i: jnp.argmax(x[i, :], -1))(
    logits, num_tokens - 1)
    
  def body_fn(carry, _):
    cache, last_tokens, i = carry
    positions_ = jax.vmap(lambda x, idx: x[idx][None])(positions, i - 1)
    attn_mask_ = jax.vmap(lambda x, idx: x[idx, :][None, :])(causal_mask, i - 1) 

    with jax.named_scope("single_decode_step"):
      next_logits, cache = transformer.apply(params, last_tokens[:, None], 
                                             positions_, cache=cache, 
                                             attention_mask=attn_mask_)

    next_tokens = jnp.argmax(next_logits, -1)[..., 0]
    return (cache, next_tokens, i + 1), next_tokens
  
  with jax.named_scope("decoding"):
    _, tokens = jax.lax.scan(body_fn, (cache, last_tokens, num_tokens), 
                             length=decode_steps)
  tokens = jnp.concatenate([last_tokens[:, None], tokens.swapaxes(-1, -2)], -1)
  return tokens

def extract_answers(tokens: jax.Array):
  lines = [vocab.decode(line.tolist()) for line in tokens]
  possible_answers = ["(A)", "(B)", "(C)", "(D)"]
  answers = []
  for line in lines:
    ans = None
    for i, possible_answer in enumerate(possible_answers):
      if possible_answer in line:
        ans = i
        break
    if ans is None:
      ans = pyrandom.randint(0, 3)
    answers.append(ans)
  return np.array(answers)

def quantize_int8_params(params: dict[str, Any]):
  assert "layer_0" in params, 'Provide params["params"]'
  # a deepcopy without copying leaves
  params_quant = jax.tree.unflatten(jax.tree.structure(params), 
                                    jax.tree.leaves(params))
  for layer_name, layer in [kv for kv in params.items() if "layer_" in kv[0]]:
    for name, weight in layer["mlp"].items():
      params_quant[layer_name]["mlp"][name] = dict(
        zip(("w", "s"), quantize_int8(weight)))
  return params_quant

def main():
  if "2b" in VARIANT:
    config = transformer_lib.TransformerConfig.gemma2_2b(cache_size=MAX_LENGTH)
  else:
    config = transformer_lib.TransformerConfig.gemma2_27b(cache_size=MAX_LENGTH)
  if QUANTIZE_FFW:
    config = dataclasses.replace(config, quantize_ffw=True)
  
  state_abstract, _ = jax.eval_shape(lambda: eval_model_abstract(config))

  mesh = jax.sharding.Mesh(compute_devices, ("x",))
  model_parallel_rules = {
    None: None, 
    "batch": "x",
    "sequence": None,
    "vocab": None,
    "features": "x",
    "q_heads": "x", 
    "kv_heads": "x", 
    "head_dim": None, 
    "ffw": None,
    "act_batch": None,
    "act_sequence": None,
    "act_heads": None,
    "act_kv_heads": None,
    "act_head_dim": None,
  }
  rules = list(model_parallel_rules.items())
  state_shardings = jax.tree.map(lambda x: nn.logical_to_mesh_sharding(
    P(*x.names), mesh, rules), state_abstract, is_leaf=is_param)

  with jax.default_device(jax.devices("cpu")[0]):
    params = params_lib.load_and_format_params(ckpt_path)
    params = params["transformer"]
    if QUANTIZE_FFW:
      params = quantize_int8_params(params)  # on CPU
    shardings_flat = jax.tree.leaves(state_shardings)
    params_flat = jax.jit(lambda x: jax.tree.leaves(x), 
                          out_shardings=shardings_flat)(params)

  transformer_params = jax.tree.unflatten(jax.tree.structure(state_abstract), 
                                          params_flat)

  # MMLU eval loop #######################
  dl, num_batches = get_dataloader("test")
  
  if PROFILE:
    batch = next(iter(dl))
    with mesh, nn.logical_axis_rules(rules):
      gen_tokens = transformer_generate(transformer_params, batch, config, 8)
    with jax.profiler.trace("mmlu-profiles"):
      with mesh, nn.logical_axis_rules(rules):
        gen_tokens = transformer_generate(transformer_params, batch, config, 8)

  pbar = tqdm(dl, total=num_batches)
  total, correct = 0, 0
  for i, batch in enumerate(pbar):
    if i >= num_batches:
      break
    with mesh, nn.logical_axis_rules(rules):
      gen_tokens = transformer_generate(transformer_params, batch, config, 16)
    pred_answers = extract_answers(gen_tokens)
    answers = batch[2]
    correct += np.sum(pred_answers == np.array(answers))
    total += len(answers)
    pbar.set_description(f"Correct fraction: {correct / total:.2%}")

if __name__ == "__main__":
  main()