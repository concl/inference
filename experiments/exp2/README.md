# exp2

INT8 W8A16 with MTP speculative decoding.

## Config

`inference serve --config experiments/exp2/config.yaml`

| Setting | Value |
|---|---|
| model | `lued/Qwen3.8-27B-INT8-W8A16-MTP` |
| served-model-name | `Qwen3.8-27B` |
| tensor-parallel-size | 4 |
| dtype | bfloat16 |
| gpu-memory-utilization | 0.92 |
| max-model-len | 262144 |
| max-num-seqs | 4 |
| max-num-batched-tokens | 8192 |
| kv-cache-dtype | fp8_e4m3 |
| mamba-cache-mode | align |
| prefix-match-unit | 16 |
| host / port | 127.0.0.1 / 8000 |
| reasoning-parser | qwen3 |
| tool-call-parser | qwen3_coder |
| enable-auto-tool-choice | true |
| trust-remote-code | true |
| disable-custom-all-reduce | true |
| enable-prefix-caching | true |
| speculative-config | `{"method": "mtp", "num_speculative_tokens": 3}` |
| default-chat-template-kwargs | `{"enable_thinking": true, "preserve_thinking": true}` |
| override-generation-config | `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0}` |

## Bench

`vllm bench serve`, 20 prompts, `--max-concurrency 2`.

| Dataset | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|
| random | 884.6 | 21.3 | 61.6 |
| sharegpt | 377.0 | 15.8 | 102.8 |

```bash
vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions --model Qwen3.8-27B \
  --tokenizer lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --num-prompts 20 --max-concurrency 2 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 \
  --save-result --result-dir experiments/exp2 --result-filename bench-random.json

vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions --model Qwen3.8-27B \
  --tokenizer lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --dataset-name sharegpt \
  --dataset-path ~/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-output-len 128 \
  --num-prompts 20 --max-concurrency 2 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 \
  --save-result --result-dir experiments/exp2 --result-filename bench-sharegpt.json
```
