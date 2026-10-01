# inference

## Serving

`scripts/serve.py` is the definition of how the model is served, including the
option list.

```bash
inference serve --help          # option list
inference serve                 # defaults
inference serve --port 8080     # any option as a flag
PORT=8080 inference serve       # or as an environment variable

# long-lived, detached
screen -S vllm_server -dm inference serve
```

Flags take precedence over environment variables. `python -m scripts.serve` also works.

## Client

```bash
inference check                          # one prompt
inference check --interactive            # chat
inference check --model Qwen3.8-27B --prompt "hi"
```

## Benchmarking

`vllm bench serve` ships with vLLM, so there is nothing extra to install. It
talks to the running server over HTTP, which means these commands also work from
a laptop through the SSH tunnel.

Both write JSON to `experiments/results/` (`--label` names the run). To compare
configurations, run the same command before and after a change and diff the JSON.

### ShareGPT — realistic chat traffic

Needs the ShareGPT JSON once (672 MB):

```bash
mkdir -p ~/benchmarks
curl -L -o ~/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

```bash
.venv/bin/vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions \
  --model Qwen3.8-27B \
  --tokenizer lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --dataset-name sharegpt \
  --dataset-path ~/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-output-len 128 \
  --num-prompts 20 --max-concurrency 2 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 \
  --save-result --result-dir experiments/results --label sharegpt
```

Drop `--sharegpt-output-len` to use the dataset's own output lengths rather than
capping them.

### BFCL — tool calling

Downloads its category files on first run. `--backend openai-chat` is required,
because BFCL attaches per-request tool schemas to chat completions.

```bash
.venv/bin/vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions \
  --model Qwen3.8-27B \
  --tokenizer lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --dataset-name hf \
  --dataset-path gorilla-llm/Berkeley-Function-Calling-Leaderboard \
  --bfcl-categories simple,live_simple,multiple \
  --num-prompts 20 --max-concurrency 2 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 \
  --save-result --result-dir experiments/results --label bfcl
```

### Baseline

2026-10-01, prefix caching on, 20 prompts each, `--max-concurrency 2`:

| Metric | ShareGPT | BFCL |
|---|---|---|
| TTFT P50 / P95 (ms) | 405 / 907 | 458 / 778 |
| TPOT P50 (ms) | 53.1 | 56.2 |
| ITL P50 / P95 (ms) | 52.4 / 56.8 | 54.0 / 109.5 |
| Output token throughput (tok/s) | 35.4 | 32.9 |
| Failed requests | 0 | 0 |

### Notes

- Keep `--max-concurrency` at or below the server's `--max-num-seqs`.
- **Do not use `--dataset-name random` for throughput claims.** Its prompts are
  non-linguistic, so MTP draft acceptance collapses and decode numbers come out
  far below reality. It is fine for TTFT-vs-length sweeps and nothing else.
- For prefix-caching changes use `--dataset-name prefix_repetition` with the
  `--prefix-repetition-*` flags, not either dataset above.
