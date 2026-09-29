Source of truth: `experiments/run-int8.sh`
Boot log: `/tmp/vllm-int8.log`

## Config

| Setting | Value |
|---|---|
| Model | `lued/Qwen3.8-27B-INT8-W8A16-MTP` |
| Served name | `Qwen3.8-27B` |
| GPUs / TP | `0,1,2,3` / 4 |
| dtype | bfloat16 |
| gpu-memory-utilization | 0.92 |
| max-model-len | 262144 |
| max-num-seqs | 4 |
| max-num-batched-tokens | 8192 |
| kv-cache-dtype | fp8_e4m3 |
| mamba-cache-mode | align |
| speculative-config | `{"method":"mtp","num_speculative_tokens":3}` |
| reasoning-parser | qwen3 |
| tool-call-parser | qwen3_coder |
| prefix caching | off |

## Result

vibe check: accepted tokens per second: 40-50

Baseline (`exp1.md`): 15-18 tok/s, so roughly 2.5-3x.

## Verified at boot

| Claim | Evidence |
|---|---|
| INT8 path is live, not a BF16 fallback | `Using MarlinLinearKernel for CompressedTensorsWNA16`, all 4 workers |
| KV pool | `GPU KV cache size: 1,406,045 tokens` |
| 262144 x 4 fits | `Maximum concurrency for 262,144 tokens per request: 5.36x` |
| MTP is drafting | 2682 draft tokens = 894 drafts x 3, matching `num_speculative_tokens` |
| MTP acceptance | 1778 accepted / 2682 drafted = **66.3%** |

Acceptance by draft position (of 894 drafts), which decays as expected since later
positions are less accurate:

| position | accepted | rate |
|---|---|---|
| 0 | 715 | 80.0% |
| 1 | 571 | 63.9% |
| 2 | 492 | 55.0% |

Positions sum to 1778, matching the total counter. The model card reported 65.5%
overall draft acceptance; this run measured 66.3%.

Memory per GPU at TP4: 8.01 GiB weights + non-torch, 1.41 GiB peak activation,
0.09 GiB CUDAGraph, 12.25 GiB KV cache.

## Notes

- Measured KV cost is ~36.5 KiB/token (4 x 12.25 GiB across 1,406,045 tokens),
  against ~32 KiB/token predicted from the config. The gap matches the log's own
  `Add 3 padding layers, may waste at most 6.25% KV cache memory` -- the page-size
  alignment that `--mamba-cache-mode align` requires.
- 262144 x 4 = 1,048,576 tokens of worst-case demand against a 1,406,045 pool, so
  four concurrent full-length requests do fit. An earlier estimate that the pool
  would be too small was wrong.
- vLLM offers `--kv-cache-memory=14487141888` (13.49 GiB) to use the remaining card
  memory instead of tuning `--gpu-memory-utilization`.

## Config drift

`scripts/serve.py` calls itself "the single definition of how the model is served",
but it cannot express this config. Its option list has no way to pass
`--kv-cache-dtype`, `--mamba-cache-mode`, `--speculative-config`,
`--max-num-batched-tokens`, `--no-enable-prefix-caching`,
`--disable-custom-all-reduce`, `--trust-remote-code`, `--performance-mode` or
`--default-chat-template-kwargs`.

That is why this experiment needed its own launcher, and it leaves two sources of
truth. `serve.py` still defaults to the BF16 checkpoint (`Qwen/Qwen3.8-27B`,
`max_model_len 131072`, `tool_call_parser qwen3_xml`), so `inference serve` and
`run-int8.sh` start different servers.

Resolve one way or the other: promote the INT8 flags into `serve.py`, or document
the launcher as the real path and scope `serve.py` down to the BF16 baseline.
