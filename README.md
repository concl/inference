# inference

vLLM serving experiments on 4×3090.

## Layout

| Path | Contents |
|---|---|
| `configs/` | Reusable default server configuration |
| `scripts/` | Reusable `serve` and `bench` commands, CLI, and OpenAI client |
| `experiments/` | Experiment-owned code, configs, notes, summaries, and ignored `artifacts/` |
| `experiments/_shared/` | Utilities shared within an experiment family |
| `notes/` | vLLM cheatsheet, PR workflow notes |
| `ignore/` | Scratch, not tracked |

## Serve

```bash
inference serve                                   # configs/serve.yaml
inference serve --config experiments/gdn_sequential_initial/config.yaml --gpus 1,3
inference serve --dry-run                         # print the command instead
screen -S vllm -dm inference serve                # detached
```

Flags given on the command line override the YAML. `--gpus` (default `0,1,2,3`)
sets `CUDA_VISIBLE_DEVICES`. `python -m scripts.serve` works too.

## Bench

`vllm bench serve` drives the running server over HTTP, so there is nothing to
install.

For repeatable launches, store benchmark flags in YAML:

```bash
inference bench --config experiments/gdn_sequential_initial/benchmark-sharegpt.yaml
inference bench --config experiments/gdn_sequential_initial/benchmark-sharegpt.yaml --num-prompts 10
inference bench --config experiments/gdn_sequential_initial/benchmark-sharegpt.yaml --dry-run
```

`python -m scripts.bench` works too. Command-line flags override YAML values.
The launcher uses vLLM's benchmark and config parser directly, avoiding the
installed CLI's argument-ordering bug with `vllm bench serve --config`.
The supplied config enables profiling, so start a profiler-enabled server first.
Dataset paths in YAML are literal; `~` and environment variables are not expanded.
Use a distinct `--result-dir` or `--result-filename` to preserve each run.

For profiling, keep notes and small curated summaries in
`experiments/<experiment>/`, and raw traces, logs, benchmark output, generated IR,
and config snapshots in `experiments/<experiment>/artifacts/`. Only `artifacts/`
is gitignored. See [the profiling index](experiments/README.md) for existing runs.

```bash
.venv/bin/vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions \
  --model Qwen3.8-27B \
  --tokenizer lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --dataset-name sharegpt \
  --dataset-path ../benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 20 --max-concurrency 2 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 \
  --save-result --result-dir experiments/my_run/artifacts/bench --result-filename bench-sharegpt.json
```

For BFCL tool calling, swap the dataset flags for `--dataset-name hf
--dataset-path gorilla-llm/Berkeley-Function-Calling-Leaderboard
--bfcl-categories simple,live_simple,multiple`.

ShareGPT, once (672 MB):

```bash
mkdir -p ../benchmarks
curl -L -o ../benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

- Keep `--max-concurrency` at or below the server's `--max-num-seqs`.
- Don't use `--dataset-name random` for throughput claims: MTP acceptance
  collapses and decode comes out far below reality. Fine for TTFT sweeps.
- Prefix-cache changes: use `--dataset-name prefix_repetition`.
- Bench output can report speculative-decode acceptance from server metrics;
  scrape `/metrics` for prefix-cache hit rates while the server is up.

See [the experiment index](experiments/README.md) for complete run recipes and
recorded results.
