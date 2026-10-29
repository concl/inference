# exp4

## Purpose

Repeat the exp1 BF16 workload with the educational sequential GDN prefill kernel. The config requested prefix caching off, but the recorded server enabled it.

## Configuration

| Setting | Recorded value |
|---|---|
| Server config | [config.yaml](config.yaml) |
| Benchmark configs | [benchmark-random.yaml](benchmark-random.yaml), [benchmark-sharegpt.yaml](benchmark-sharegpt.yaml) |
| Model | `Qwen/Qwen3.8-27B` |
| Tensor parallelism | 4 × RTX 3090 |
| Context limit / active sequences | 131072 / 32 |
| Prefix caching, config / recorded server | false / true |
| Prefill implementation | sequential |
| Workload | Random first, then ShareGPT; 20 prompts each, concurrency 2, 128 output-token cap |
| vLLM version in server log | 0.29.0 |

## Replication instructions

1. Read the [replication prerequisites](../README.md#replication-prerequisites), then create a new output directory. Run all steps in the same Bash terminal.

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   EXPERIMENT_RUN_DIR="$PWD/experiments/exp4/artifacts/replications/$(date -u +%Y%m%dT%H%M%SZ)"
   mkdir -p "$EXPERIMENT_RUN_DIR"
   .venv/bin/python -c 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["vllm","torch","triton","transformers"]},indent=2))' > "$EXPERIMENT_RUN_DIR/environment.json"
   ```

2. Snapshot the kernel implementation specified for replication. The import overlay selects this file without editing the installed package.

   ```bash
   EXPERIMENT_KERNEL_SOURCE="$EXPERIMENT_RUN_DIR/kernel-db0084219d62.py"
   git -C ../vllm show \
     db0084219d62a7b7b564adeedb9e6536cadcf0d8:vllm/third_party/flash_linear_attention/ops/chunk.py \
     > "$EXPERIMENT_KERNEL_SOURCE"
   ```

3. Start the server in the background and save its configuration and log.

   ```bash
   cp experiments/exp4/config.yaml "$EXPERIMENT_RUN_DIR/config.yaml"
   PYTHONPATH="$PWD/experiments/_shared/gdn/import_overlay" \
   GDN_CHUNK_SOURCE="$EXPERIMENT_KERNEL_SOURCE" \
     .venv/bin/python -m scripts.serve \
     --config "$EXPERIMENT_RUN_DIR/config.yaml" --gpus 0,1,2,3 --enable-prefix-caching \
     > "$EXPERIMENT_RUN_DIR/serve.log" 2>&1 &
   EXPERIMENT_SERVER_PID=$!
   ```

4. Wait for `Application startup complete` in the log. Press Ctrl+C to stop following the log, leaving the background server running. Do not send a warmup request unless a later step specifies one.

   ```bash
   tail -f "$EXPERIMENT_RUN_DIR/serve.log"
   ```

5. Run the random benchmark and wait for completion. Save the benchmark config with the outputs.

   ```bash
   cp experiments/exp4/benchmark-random.yaml "$EXPERIMENT_RUN_DIR/benchmark-random.yaml"
   .venv/bin/inference bench \
     --config "$EXPERIMENT_RUN_DIR/benchmark-random.yaml" \
     --result-dir "$EXPERIMENT_RUN_DIR" \
     > "$EXPERIMENT_RUN_DIR/bench-random.log" 2>&1
   ```

6. Run the sharegpt benchmark and wait for completion. Save the benchmark config with the outputs.

   ```bash
   cp experiments/exp4/benchmark-sharegpt.yaml "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml"
   .venv/bin/inference bench \
     --config "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml" \
     --result-dir "$EXPERIMENT_RUN_DIR" \
     > "$EXPERIMENT_RUN_DIR/bench-sharegpt.log" 2>&1
   ```

7. Stop the server after the benchmark or trace export finishes.

   ```bash
   kill -TERM "$EXPERIMENT_SERVER_PID"
   wait "$EXPERIMENT_SERVER_PID"
   ```

## Results

| Dataset | Completed / failed | TTFT p50 (ms) | TPOT p50 (ms) | Output tokens/s |
|---|---|---|---|---|
| random | 20 / 0 | 1372.6 | 30.5 | 49.9 |
| sharegpt | 20 / 0 | 296.2 | 28.2 | 63.1 |

## Artifacts

Historical raw results: `artifacts/bench/bench-random.json`,
`artifacts/bench/bench-sharegpt.json`, and `artifacts/serve.log`.
All benchmark metrics are preserved in tracked [summary.json](summary.json).

New replications: `artifacts/replications/<UTC timestamp>/` contains copied configs, environment versions, the kernel snapshot, server/benchmark logs, and benchmark JSON. Raw data remains gitignored.

## Limitations

The exact installed dependency commits, model/tokenizer revisions, dataset hash, physical GPU IDs, cache state, and benchmark seed were not all saved. The YAML reconstructs the documented workload with explicit seed 0; it does not establish the historical seed. The kernel overlay fixes the implementation, but does not recreate every historical vLLM file. Exact historical timings/output text cannot be guaranteed. Random inputs are not representative of MTP acceptance. The earlier failed startup in `artifacts/serve-failed-headmismatch.log` returned 4 state heads where 12 were required; the successful capture used the corrected grouped-head implementation. The failed startup is not part of the successful replication recipe. The server log records `enable_prefix_caching=True` despite `enable-prefix-caching: false` in the historical YAML. The replication command explicitly enables it to reproduce the actual run. The original “prefix caching off” description was inaccurate; exp1 versus exp3 is not a valid off/on comparison.
