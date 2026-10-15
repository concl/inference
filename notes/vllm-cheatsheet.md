# vLLM serving cheat sheet

Reference for Qwen3.8-27B on 5x RTX 3090 (24 GB each, PCIe only, no NVLink).

---

## The one problem every flag solves

Fitting into 24 GB per card.

```
One RTX 3090 (24 GB)
├── 1. WEIGHTS      fixed size — you shard this, you never tune it
├── 2. KV CACHE     grows with context length x concurrent requests
└── 3. OVERHEAD     activations, CUDA graphs, compile scratch (~1-2 GB)
```

---

## The four knobs that matter

| Flag | Plain meaning | Turn it up | Turn it down |
|---|---|---|---|
| `--tensor-parallel-size` | How many GPUs share the weights | Bigger model fits | — |
| `--gpu-memory-utilization` | % of each card vLLM may claim | More KV cache | Safer against OOM |
| `--max-model-len` | Longest **single** request (input + output) | Longer conversations | More requests fit |
| `--max-num-seqs` | Requests allowed in flight **at once** | More concurrency | Safer, less KV |

### max-num-seqs is a permission, not a reservation

This is the most common misunderstanding. `--max-num-seqs 4` does **not** reserve
4 full-length contexts. It just says "up to 4 may be active." All active requests
draw from **one shared KV pool**. If the pool only holds one full-length context,
you get one full-length request and the other three wait — no error.

So `max-model-len 262144` + `max-num-seqs 4` is a safe *combination to set*.
It never allocates 4 x 262144. The real limit is whatever the pool prints at boot.

---

## Tiered build-up

Grow the command one tier at a time instead of swallowing it whole. Add a tier,
restart, record what changed — that is the experiment.

```bash
# Tier 1 — will it even boot?
vllm serve lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --tensor-parallel-size 4 --dtype bfloat16 --gpu-memory-utilization 0.92

# Tier 2 — memory control
  --max-model-len 262144 --max-num-seqs 4 --kv-cache-dtype fp8_e4m3

# Tier 3 — wire up the model (MTP, GDN, parsers)
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice

# Tier 4 — this machine's quirks
  --disable-custom-all-reduce --no-enable-prefix-caching
  # + export NCCL_P2P_DISABLE=1
```

---

## Three kinds of flag (they are not all "settings")

**1. Tuning** — pick a value based on your workload:
`--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`,
`--max-num-seqs`, `--kv-cache-dtype`, `--max-num-batched-tokens`

**2. Wiring** — no value to choose; omitting means broken or silently degraded:

| Flag | Why it is required |
|---|---|
| `--mamba-cache-mode align` | Qwen3.8 has 48 GDN linear-attention layers. Documented as required. |
| `--speculative-config {"method":"mtp",...}` | The `-MTP` checkpoint has a built-in draft model. Omit this and it sits unused — you lose the speedup. |
| `--reasoning-parser qwen3` | Otherwise the raw `</think>` tag leaks into `content`. |
| `--tool-call-parser` | Otherwise tool calls arrive as plain text. |
| `--trust-remote-code` | Custom architecture code. |

**3. Local workarounds** — about this box, not the model:

| Flag | Why |
|---|---|
| `NCCL_P2P_DISABLE=1`, `--disable-custom-all-reduce` | 3090s here have no NVLink (`topo -m` shows `PHB`/`NODE`). Without these NCCL can stall forever. |
| `--max-num-seqs 2` (card default) | 2+ concurrent requests intermittently crash the engine on this architecture (async `cudaErrorIllegalAddress`); upstream fix not merged. |
| `--no-enable-prefix-caching` | Prefix caching can corrupt recurrent state here, and it fails **silently** — wrong output, no error. |

---

## Symptom -> which knob

| What you see | Reach for |
|---|---|
| OOM while loading weights | Raise `--tensor-parallel-size` |
| OOM during KV cache setup | Lower `--max-model-len` or `--max-num-seqs` |
| OOM on startup, nothing else | Lower `--gpu-memory-utilization` |
| Boots, but decode is slow | Check MTP is actually enabled |
| First request hangs forever | The NCCL/P2P workarounds |
| Worker dies: `FileNotFoundError: 'ninja'` | `.venv/bin` not on PATH — see Troubleshooting |

---

## Troubleshooting

### `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`

**Not** a missing package, and nothing to do with parsers. Two different
resolution mechanisms are at play:

| Call | Resolved by | Works when |
|---|---|---|
| `import ninja` | Python module search path | package installed in venv |
| `subprocess.run(["ninja"])` | shell `PATH` lookup | `.venv/bin` on `PATH` |

So `import ninja` can succeed while the subprocess launch fails. FlashInfer's JIT
(`flashinfer/jit/cpp_ext.py:370 run_ninja`) shells out to the bare command `ninja`
when compiling prefill kernels at startup.

Triggered by running `.venv/bin/vllm` **directly**: that works for Python imports
but does not put `.venv/bin` on `PATH` the way `source .venv/bin/activate` does.

Fix, in the launcher:

```bash
export PATH="$PWD/.venv/bin:$PATH"
export VIRTUAL_ENV="$PWD/.venv"
```

Other JIT toolchain requirements, worth checking together so you don't trade one
missing-tool error for another: `nvcc` (CUDA JIT), `g++` (host compiler).

This pattern applies to any tool that shells out to a build binary — `ninja` for
Triton/Inductor/FlashInfer, `nvcc`, `git` for some build backends. Note that
`VLLM_USE_FLASHINFER_SAMPLER=0` disables the FlashInfer *sampler*, not FlashInfer
*attention* — kernel compilation is still required.

---

## KV cache math for this model

Read from `config.json` -> `text_config`:

| Field | Value |
|---|---|
| `num_hidden_layers` | 64 |
| `layer_types` | 48 `linear_attention` (GDN) + 16 `full_attention` |
| `full_attention_interval` | 4 |
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 4 (GQA) |
| `head_dim` | 256 |
| `max_position_embeddings` | 262144 |

### Only 16 of 64 layers have a token-proportional cache

- **16 `full_attention` layers** -> standard paged KV cache, grows with tokens.
- **48 `linear_attention` (GDN) layers** -> a fixed-size recurrent state per
  *sequence*, constant in context length.

So three quarters of the layers cost nothing as context grows. This model is far
cheaper per token than a dense 64-layer equivalent.

### Cost per token

Per full-attention layer, per token:

```
2 (K and V) x 4 kv_heads x 256 head_dim = 2,048 values
```

| KV dtype | bytes/value | per layer | x 16 layers | 262,144 tokens | 1M tokens |
|---|---|---|---|---|---|
| `bfloat16` | 2 | 4 KiB | **64 KiB** | 16 GiB | 64 GiB |
| `fp8_e4m3` | 1 | 2 KiB | **32 KiB** | 8 GiB | 32 GiB |

`--kv-cache-dtype fp8_e4m3` halves every number in this table.

### Worst-case KV, fp8 (GiB)

Rows = context length per request. Columns = concurrent requests at that length.
Multiply by 2 for `bfloat16`.

| context/request | x1 | x2 | x4 | x8 |
|---|---|---|---|---|
| 8,192 | 0.25 | 0.5 | 1 | 2 |
| 32,768 | 1 | 2 | 4 | 8 |
| 65,536 | 2 | 4 | 8 | 16 |
| 131,072 | 4 | 8 | 16 | 32 |
| 262,144 | 8 | 16 | 32 | 64 |

### Reverse: how many tokens fit in a pool

| KV pool | fp8_e4m3 | bfloat16 |
|---|---|---|
| 8 GiB | 262,144 | 131,072 |
| 16 GiB | 524,288 | 262,144 |
| 32 GiB | 1,048,576 | 524,288 |
| 64 GiB | 2,097,152 | 1,048,576 |

### Capacity planning

```
KV pool GiB ~= (gpu_mem_util x total_VRAM) - weights - overhead
```

Weights: 29.44 GiB for this INT8 checkpoint, split across TP ranks.
Overhead: ~1-2 GiB/card, higher at long context (CUDA graphs grow with length).

**The authoritative number is the one vLLM prints at boot**, not this arithmetic:

```
grep -iE "GPU KV cache size|maximum concurrency" /tmp/vllm-int8.log
```

The model card reports a 266,537-token pool at TP2 / 262144 context — about one
full-length request — which is why it warns "simultaneous full-native-context
capacity: 1.02x". More cards means a bigger pool and more genuine parallelism.

---

## Why fp8 on a 3090 that "doesn't support FP8"

Two different things share the name:

| | FP8 **compute** | FP8 **storage** |
|---|---|---|
| What it is | Running the GEMMs in 8-bit | Storing values as 8-bit, converting back to bf16 before the math |
| Needs FP8 tensor cores | Yes | No |
| sm_86 (3090) | Unavailable | **Works fine** |

The 3090 lacks FP8 tensor cores, which bans FP8 *arithmetic* — not FP8 *storage*.
`--kv-cache-dtype fp8_e4m3` only affects how the cache is stored; the attention
kernel upcasts before computing. Hence it works on Ampere.

That is also why this checkpoint is W8A16 rather than FP8: on sm_86, FP8 weight
checkpoints fall back to Marlin weight-only kernels anyway, so W8A16 keeps more
weight precision for the same decode class.

**Accuracy caveat:** the card verified fp8 E4M3 KV *runtime compatibility*, not
long-context KV *accuracy*. The checkpoint declares `kv_cache_scheme: null`, so it
ships no calibrated K/V scales. Treat long-context fp8 KV quality as unmeasured.

---

## Verification after boot

```bash
grep -iE "CompressedTensorsWNA16|MarlinLinearKernel" /tmp/vllm-int8.log  # weights are really INT8
grep -iE "GPU KV cache size|maximum concurrency"     /tmp/vllm-int8.log  # pool in tokens
grep -iE "speculative|accept"                        /tmp/vllm-int8.log  # MTP wired up
curl -s localhost:8000/metrics | grep -iE "spec_decode|accepted"         # live MTP acceptance
```

If the first grep shows `CompressedTensorsWNA16`, the INT8 path is live — not a
silent BF16 fallback.
