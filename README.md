# inference

A workspace for experimenting with vLLM serving, and for turning anything that
pans out into an upstream pull request.

## Layout

| Path | What lives there |
|---|---|
| `configs/` | Serving configs, in vLLM's `vllm serve --config` YAML format |
| `scripts/` | `serve.py` (a thin launcher around `vllm serve --config`), the `inference` CLI, and the client helper |
| `experiments/` | Experiment write-ups (`exp*.md`) and their raw results (`results/`) |
| `notes/` | Reference notes |
| `notes/WORKING_ON_VLLM.md` | How to work on vLLM internals and submit a PR |

## Working on vLLM

vLLM is a **uv dependency**, not a copy vendored into this repo. Which one you get
is one line in `pyproject.toml` — currently commented, so `uv sync` gives the PyPI
release:

```toml
[tool.uv.sources]
vllm = { git = "https://github.com/concl/vllm.git", branch = "acmai/<change>" }
```

Work for a PR is made in a checkout of the fork, on a branch, and submitted as that
branch — there is no patch file to keep in sync. See `notes/WORKING_ON_VLLM.md`.

The rule: **if it ships in the PR, it lives in the fork.** vLLM's CI runs `pytest`
inside the vLLM repo, so a test or microbenchmark written in this repo cannot be
submitted. Changes go in `vllm/`, their correctness tests in `tests/kernels/`, and
shipped microbenchmarks in `benchmarks/kernels/` — all inside the checkout. That is
why this repo has no `tests/`: there is nothing here a vLLM PR could include.

## Serving

`configs/serve.yaml` is the definition of how the model is served, written in
vLLM's own `vllm serve --config` format: keys are flag names without the leading
`--`, and nested maps are passed through as JSON.

`scripts/serve.py` is only a launcher. It picks the `vllm` next to the
interpreter, sets `CUDA_VISIBLE_DEVICES`, forwards everything it does not
recognise, and execs `vllm serve --config <file>`.

```bash
inference serve --help              # vLLM's option list, not ours
inference serve                     # configs/serve.yaml
inference serve --port 8080         # CLI overrides the config file
inference serve --config configs/other.yaml
inference serve --dry-run           # print the command instead of running it

# long-lived, detached
screen -S vllm_server -dm inference serve
```

Command-line arguments beat the config file, which beats vLLM's defaults.
`python -m scripts.serve` also works. GPUs come from `--gpus` or
`CUDA_VISIBLE_DEVICES` (default `0,1,2,3`).

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

We use vLLM's own harness rather than a home-grown one, so the numbers are the ones
a reviewer will re-run. Results land in `experiments/results/`.

Two things to hold constant. **Change one variable per run** against a named
baseline, and **give each server config its own `--served-model-name`** — vLLM's
bench JSON cannot otherwise tell two servers apart, which is what made
`experiments/exp3.md` ambiguous.

`vllm bench` does not record spec-decode acceptance or prefix-cache hit rate. Grab
them from the server while it is still up:

```bash
curl -s localhost:8000/metrics \
  | grep -E 'spec_decode|prefix_cache|gpu_cache_usage' \
  > experiments/results/<label>-metrics.txt
```

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
