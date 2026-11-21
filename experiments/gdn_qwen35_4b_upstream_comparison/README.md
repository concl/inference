# gdn_qwen35_4b_upstream_comparison

## Purpose

Compare `Qwen/Qwen3.5-4B` serving latency and throughput between the baseline unfused GDN chunked prefill pipeline (`vllm-og`) and the updated fused intra-chunk `kkt + solve_tril` and `exp2`-scaled pipeline (`vllm-updated`) across ShareGPT and 1024-token random workloads.

## Configuration

| Setting | Recorded value |
|---|---|
| Source commit / kernel selection | vLLM `898a7c68c2` (`vllm-og` baseline vs `vllm-updated` fused `chunk_gated_delta_rule_fwd_intra` + `RCP_LN2` / `exp2` + int64 pointer masks in `vllm/third_party/flash_linear_attention/ops/`) |
| Software versions | vLLM `0.34.0rc1.dev993+g898a7c68c.d20261120.precompiled`, PyTorch `2.14.0`, Triton `3.8.0`, Transformers `5.23.0`, NCCL `2.30.7` ([summary.json](summary.json)) |
| Hardware / GPU IDs / TP | 2 × NVIDIA GeForce RTX 3090 / physical GPUs 2,3 / TP=2 |
| Model/tokenizer revisions | `Qwen/Qwen3.5-4B` (`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) |
| Workload / dataset hash | [config.yaml](config.yaml), [benchmark-sharegpt.yaml](benchmark-sharegpt.yaml), [benchmark-random.yaml](benchmark-random.yaml); seed 0, 20 prompts, concurrency 2, 128 output tokens (ShareGPT: 4,564 input tokens, SHA256 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`; Random: 1,024 input tokens/prompt, 20,689 total input tokens) |
| Warmup / cache policy | Prefix caching disabled (`no-enable-prefix-caching: true`); FP8 (`fp8_e4m3`) KV cache; `vllm-og` warmed by prior benchmark pass on same server instance, `vllm-updated` warmed with `--num-warmups 2` |
| Profiler / instrumentation settings | None (uninstrumented end-to-end serving benchmark; JIT monitor in `warn` mode) |

## Replication instructions

1. Read the [replication prerequisites](../README.md#replication-prerequisites), verify GPUs 2 and 3 and port 8001 are free, and create a fresh timestamped output directory. Run all steps in the same Bash terminal from the repository root.

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   nvidia-smi
   ss -ltnp 'sport = :8001'
   EXPERIMENT_RUN_DIR="$PWD/experiments/gdn_qwen35_4b_upstream_comparison/artifacts/replications/$(date -u +%Y%m%dT%H%M%SZ)"
   mkdir -p "$EXPERIMENT_RUN_DIR/bench"
   .venv/bin/python -c 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["vllm","torch","triton","transformers"]},indent=2))' > "$EXPERIMENT_RUN_DIR/environment.json"
   nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv > "$EXPERIMENT_RUN_DIR/gpus.txt"
   sha256sum ../benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json > "$EXPERIMENT_RUN_DIR/dataset.sha256"
   ```

2. Snapshot the server and benchmark configurations and record the kernel diff between `../vllm-og` and `../vllm-updated`. Validate both benchmark configs with `--dry-run`.

   ```bash
   cp experiments/gdn_qwen35_4b_upstream_comparison/config.yaml "$EXPERIMENT_RUN_DIR/config.yaml"
   cp experiments/gdn_qwen35_4b_upstream_comparison/benchmark-sharegpt.yaml "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml"
   cp experiments/gdn_qwen35_4b_upstream_comparison/benchmark-random.yaml "$EXPERIMENT_RUN_DIR/benchmark-random.yaml"
   git -C ../vllm-og rev-parse HEAD > "$EXPERIMENT_RUN_DIR/vllm-og.commit"
   git -C ../vllm-updated diff > "$EXPERIMENT_RUN_DIR/fla_ops.patch"
   cp ../vllm-updated/vllm/third_party/flash_linear_attention/ops/chunk_fwd.py "$EXPERIMENT_RUN_DIR/chunk_fwd.py"
   .venv/bin/inference bench --config "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml" --dry-run > "$EXPERIMENT_RUN_DIR/bench-sharegpt-dry-run.txt" 2>&1
   .venv/bin/inference bench --config "$EXPERIMENT_RUN_DIR/benchmark-random.yaml" --dry-run > "$EXPERIMENT_RUN_DIR/bench-random-dry-run.txt" 2>&1
   ```

3. Point the editable `vllm` package at `../vllm-og` (or set `PYTHONPATH="$PWD/../vllm-og"`), start the baseline server on GPUs 2,3, and wait for `Application startup complete`.

   ```bash
   PYTHONPATH="$PWD/../vllm-og" .venv/bin/python -m scripts.serve \
     --config "$EXPERIMENT_RUN_DIR/config.yaml" --gpus 2,3 \
     > "$EXPERIMENT_RUN_DIR/serve-og.log" 2>&1 &
   EXPERIMENT_SERVER_PID=$!
   until curl -sf http://127.0.0.1:8001/v1/models > /dev/null; do sleep 2; done
   ```

4. Run the warmed ShareGPT and Random workloads against the `vllm-og` server, then stop the server.

   ```bash
   .venv/bin/inference bench \
     --config "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml" \
     --num-warmups 2 \
     --result-dir "$EXPERIMENT_RUN_DIR/bench" \
     --result-filename bench-qwen35-4b-sharegpt-og.json \
     > "$EXPERIMENT_RUN_DIR/bench-sharegpt-og.log" 2>&1
   .venv/bin/inference bench \
     --config "$EXPERIMENT_RUN_DIR/benchmark-random.yaml" \
     --num-warmups 2 \
     --result-dir "$EXPERIMENT_RUN_DIR/bench" \
     --result-filename bench-qwen35-4b-random-og.json \
     > "$EXPERIMENT_RUN_DIR/bench-random-og.log" 2>&1
   kill -TERM "$EXPERIMENT_SERVER_PID"
   wait "$EXPERIMENT_SERVER_PID"
   ```

5. Start the server with `../vllm-updated` on GPUs 2,3, wait for readiness, run the warmed ShareGPT and Random workloads, then stop the server and verify all four output files report 20 completed and 0 failed requests.

   ```bash
   PYTHONPATH="$PWD/../vllm-updated" .venv/bin/python -m scripts.serve \
     --config "$EXPERIMENT_RUN_DIR/config.yaml" --gpus 2,3 \
     > "$EXPERIMENT_RUN_DIR/serve-updated.log" 2>&1 &
   EXPERIMENT_SERVER_PID=$!
   until curl -sf http://127.0.0.1:8001/v1/models > /dev/null; do sleep 2; done
   .venv/bin/inference bench \
     --config "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml" \
     --num-warmups 2 \
     --result-dir "$EXPERIMENT_RUN_DIR/bench" \
     --result-filename bench-qwen35-4b-sharegpt-updated.json \
     > "$EXPERIMENT_RUN_DIR/bench-sharegpt-updated.log" 2>&1
   .venv/bin/inference bench \
     --config "$EXPERIMENT_RUN_DIR/benchmark-random.yaml" \
     --num-warmups 2 \
     --result-dir "$EXPERIMENT_RUN_DIR/bench" \
     --result-filename bench-qwen35-4b-random-updated.json \
     > "$EXPERIMENT_RUN_DIR/bench-random-updated.log" 2>&1
   kill -TERM "$EXPERIMENT_SERVER_PID"
   wait "$EXPERIMENT_SERVER_PID"
   rg 'Successful requests|Failed requests' "$EXPERIMENT_RUN_DIR"/bench-*.log
   ```

## Results

### Summary table

| Dataset | Variant | Completed / failed | Duration (s) | TTFT mean / p50 / p95 (ms) | TPOT mean / p50 / p95 (ms) | ITL mean / p50 / p95 (ms) | Output tok/s | Total tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sharegpt | `vllm-og` | 20 / 0 | 12.15 | 81.18 / 79.31 / 136.72 | 8.92 / 8.84 / 9.35 | 8.93 / 8.77 / 9.20 | 210.73 | 586.41 |
| sharegpt | `vllm-updated` | 20 / 0 | 12.20 | 78.55 / 79.61 / 133.87 | 8.98 / 8.88 / 9.39 | 8.98 / 8.82 / 9.30 | 209.92 | 584.17 |
| random | `vllm-og` | 20 / 0 | 14.04 | 220.04 / 218.67 / 284.95 | 9.31 / 9.32 / 9.83 | 9.31 / 8.81 / 9.22 | 182.40 | 1656.49 |
| random | `vllm-updated` | 20 / 0 | 14.18 | 220.91 / 219.67 / 286.11 | 9.42 / 9.40 / 9.94 | 9.42 / 8.92 / 9.40 | 180.55 | 1639.71 |

On the warmed ShareGPT workload (4,564 input tokens, 2,560 output tokens), `vllm-updated` reduced mean TTFT by 3.2% (`78.55 ms` vs `81.18 ms`), p95 TTFT by 2.1% (`133.87 ms` vs `136.72 ms`), and TTFT standard deviation by 6.2% (`28.88 ms` vs `30.79 ms`). On the 1,024-token random workload (20,689 input tokens, 2,560 output tokens), both implementations performed within 1.0% across TTFT, TPOT, and throughput (`218.67 ms` vs `219.67 ms` TTFT p50; `182.40 tok/s` vs `180.55 tok/s` output throughput). Because `Qwen/Qwen3.5-4B` on 2 × RTX 3090 spends over 85–93% of request wall-clock time in decode steps and PCIe all-reduce, prefill kernel changes shift end-to-end throughput by less than 1%. All values come from the four saved benchmark JSON files in `artifacts/bench/` and are preserved in [summary.json](summary.json).

## Artifacts

Curated metrics and SHA256 hashes: [summary.json](summary.json).

Historical files under gitignored `artifacts/`:
- `artifacts/bench/bench-qwen35-4b-sharegpt-og.json`
- `artifacts/bench/bench-qwen35-4b-sharegpt-updated.json`
- `artifacts/bench/bench-qwen35-4b-random-og.json`
- `artifacts/bench/bench-qwen35-4b-random-updated.json`
- `artifacts/serve-og.log` and `artifacts/serve-updated.log` (plus initial cold run log `artifacts/serve-initial.log`)
- `artifacts/kernel_diff/fla_ops.patch` and `artifacts/kernel_diff/chunk_fwd.py`

New replications write to `artifacts/replications/<UTC timestamp>/`, keeping original captures untouched.

## Limitations

1. **Multi-file kernel change:** Unlike single-file `chunk.py` experiments that use `experiments/_shared/gdn/import_overlay`, `vllm-updated` modifies 8 files and adds `chunk_fwd.py` in `vllm/third_party/flash_linear_attention/ops/`. Replication therefore switches the Python package root between `../vllm-og` and `../vllm-updated` (or applies `artifacts/kernel_diff/fla_ops.patch` and `chunk_fwd.py` to commit `898a7c68c2`).
2. **Warmup procedure symmetry:** In the recorded session, `vllm-og` was already warmed by earlier ShareGPT and Random runs on the same live server process before the recorded `bench-qwen35-4b-*-og.json` runs executed without `--num-warmups`, whereas `vllm-updated` was freshly started and warmed via `--num-warmups 2` in each `inference bench` invocation. During the first ShareGPT pass on `vllm-updated`, `_fused_gdn_gated_rms_norm_kernel` logged one Triton JIT compilation (`serve-updated.log`), which hit the disk cache warmed by the earlier `vllm-og` run. The replication recipe standardizes `--num-warmups 2` for both variants.
3. **Decode-dominated regime and shared host:** At 228 average input tokens (ShareGPT) and 1,024 input tokens (Random) with 128 generated tokens per prompt on 2 × RTX 3090 over PCIe, decode latency (`~8.8–9.4 ms/token`) dominates total request duration. In addition, the fused `chunk_gated_delta_rule_fwd_kkt_solve_kernel` keeps ten `16×16` FP32 accumulators in registers, trading HBM traffic for higher register pressure on `sm_86`. Other GPUs on the machine were in use by other processes during capture.
