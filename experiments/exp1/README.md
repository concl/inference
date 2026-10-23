# exp1

BF16 baseline, prefix caching off.

## Config

`inference serve --config experiments/exp1/config.yaml`

| Setting | Value |
|---|---|
| model | `Qwen/Qwen3.8-27B` |
| served-model-name | `Qwen3.8-27B` |
| tensor-parallel-size | 4 |
| dtype | bfloat16 |
| gpu-memory-utilization | 0.95 |
| max-model-len | 131072 |
| max-num-seqs | 32 |
| host / port | 127.0.0.1 / 8000 |
| reasoning-parser | qwen3 |
| tool-call-parser | qwen3_xml |
| enable-auto-tool-choice | true |
| enable-prefix-caching | false |
| disable-custom-all-reduce | true |

## Bench

`vllm bench serve`, 20 prompts, `--max-concurrency 2`.

| Dataset | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|
| random | 1226.5 | 30.9 | 49.3 |
| sharegpt | 281.3 | 28.9 | 62.3 |

```bash
vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions --model Qwen3.8-27B \
  --tokenizer Qwen/Qwen3.8-27B \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --num-prompts 20 --max-concurrency 2 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 \
  --save-result --result-dir experiments/exp1 --result-filename bench-random.json

vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions --model Qwen3.8-27B \
  --tokenizer Qwen/Qwen3.8-27B \
  --dataset-name sharegpt \
  --dataset-path ~/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-output-len 128 \
  --num-prompts 20 --max-concurrency 2 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 \
  --save-result --result-dir experiments/exp1 --result-filename bench-sharegpt.json
```
