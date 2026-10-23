# inference

vLLM serving experiments on 4×3090.

## Layout

| Path | Contents |
|---|---|
| `configs/` | Serving definitions, in `vllm serve --config` YAML format |
| `scripts/` | `serve.py` launcher, `inference` CLI, OpenAI client helper |
| `experiments/` | One dir per run: config, serve log, bench JSON, write-up |
| `notes/` | vLLM cheatsheet, PR workflow notes |
| `ignore/` | Scratch, not tracked |

## Serve

```bash
inference serve                                   # configs/serve.yaml
inference serve --config configs/serve_profiler.yaml
inference serve --dry-run                         # print the command instead
screen -S vllm -dm inference serve                # detached
```

Flags given on the command line override the YAML. `--gpus` (default `0,1,2,3`)
sets `CUDA_VISIBLE_DEVICES`. `python -m scripts.serve` works too.

## Bench

`vllm bench serve` drives the running server over HTTP, so there is nothing to
install.

```bash
.venv/bin/vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions \
  --model Qwen3.8-27B \
  --tokenizer lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --dataset-name sharegpt \
  --dataset-path ~/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 20 --max-concurrency 2 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 \
  --save-result --result-dir experiments/exp<N> --result-filename bench-sharegpt.json
```

For BFCL tool calling, swap the dataset flags for `--dataset-name hf
--dataset-path gorilla-llm/Berkeley-Function-Calling-Leaderboard
--bfcl-categories simple,live_simple,multiple`.

ShareGPT, once (672 MB):

```bash
mkdir -p ~/benchmarks
curl -L -o ~/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

- Keep `--max-concurrency` at or below the server's `--max-num-seqs`.
- Don't use `--dataset-name random` for throughput claims: MTP acceptance
  collapses and decode comes out far below reality. Fine for TTFT sweeps.
- Prefix-cache changes: use `--dataset-name prefix_repetition`.
- Bench output omits spec-decode acceptance and prefix-cache hit rate; scrape
  `/metrics` while the server is up.

Results are written up in `experiments/exp{1,2,3}/README.md`.
