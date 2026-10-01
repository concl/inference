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

## Bench

| Metric | Value |
|---|---|
| Accepted tokens/s (vibe check) | 40-50 |
| `exp1.md` BF16 baseline (vibe check) | 15-18 |

Issue: high latency, prefix caching not effective.