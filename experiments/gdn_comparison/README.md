# gdn_comparison

## Purpose

Compare the original chunked server path with the educational sequential GDN implementation, including inside-kernel Proton captures.

## Configuration

| Setting | Recorded value |
|---|---|
| Server / benchmark configs | [config.yaml](config.yaml), [benchmark-sharegpt.yaml](benchmark-sharegpt.yaml) |
| Proton harness | [shared GDN harness](../_shared/gdn/profile_intra.py) |
| Torch server | Original chunked FLA; GPUs 1,3; TP=2; INT8 model with MTP |
| Torch workload | ShareGPT seed 0, 64 input tokens, 32 output-token cap, concurrency 1 |
| Torch extra warmups / readiness timeout | 1 request / 600 seconds; endpoint check also sends a request |
| Proton workload | Synthetic layer inputs: B=1, T=2048, Hqk=8, Hv=24, K=V=128, seed=1337 |
| Proton input types | BF16 normalized q/k and BF16 values; FP32 log gates, beta, and zero initial state |
| Proton GPU / sampling | Physical GPU 4; CTA (0,0,0), warp 0 |
| Proton extra instrumented warmups / buffer | 0 / 524288 bytes per CTA for both implementations |
| Triton | 3.7.1 |

## Replication instructions

1. Read the [replication prerequisites](../README.md#replication-prerequisites), then create a new output directory. Run all steps in the same Bash terminal.

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   EXPERIMENT_RUN_DIR="$PWD/experiments/gdn_comparison/artifacts/replications/$(date -u +%Y%m%dT%H%M%SZ)"
   mkdir -p "$EXPERIMENT_RUN_DIR"
   .venv/bin/python -c 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["vllm","torch","triton","transformers"]},indent=2))' > "$EXPERIMENT_RUN_DIR/environment.json"
   ```

2. Snapshot the kernel implementation specified for replication. The import overlay selects this file without editing the installed package.

   ```bash
   EXPERIMENT_KERNEL_SOURCE="$EXPERIMENT_RUN_DIR/kernel-58d8a013b2dd.py"
   git -C ../vllm show \
     58d8a013b2dd33adb99f75a1c7d12ce1babf8640:vllm/third_party/flash_linear_attention/ops/chunk.py \
     > "$EXPERIMENT_KERNEL_SOURCE"
   ```

3. Start the server in the background and save its configuration and log.

   ```bash
   cp experiments/gdn_comparison/config.yaml "$EXPERIMENT_RUN_DIR/config.yaml"
   PYTHONPATH="$PWD/experiments/_shared/gdn/import_overlay" \
   GDN_CHUNK_SOURCE="$EXPERIMENT_KERNEL_SOURCE" \
     .venv/bin/python -m scripts.serve \
     --config "$EXPERIMENT_RUN_DIR/config.yaml" --gpus 1,3 \
     --profiler-config.torch_profiler_dir "$EXPERIMENT_RUN_DIR/torch" \
     > "$EXPERIMENT_RUN_DIR/serve.log" 2>&1 &
   EXPERIMENT_SERVER_PID=$!
   ```

4. Wait for `Application startup complete` in the log. Press Ctrl+C to stop following the log, leaving the background server running. Do not send a warmup request unless a later step specifies one.

   ```bash
   tail -f "$EXPERIMENT_RUN_DIR/serve.log"
   ```

5. Run the sharegpt benchmark and wait for completion. Save the benchmark config with the outputs.

   ```bash
   cp experiments/gdn_comparison/benchmark-sharegpt.yaml "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml"
   .venv/bin/inference bench \
     --config "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml" \
     --result-dir "$EXPERIMENT_RUN_DIR" \
     > "$EXPERIMENT_RUN_DIR/bench-sharegpt.log" 2>&1
   ```

6. Stop the server after the benchmark or trace export finishes.

   ```bash
   kill -TERM "$EXPERIMENT_SERVER_PID"
   wait "$EXPERIMENT_SERVER_PID"
   ```

7. Select the educational source snapshot for the isolated kernel harness. It retains both the original chunked function and the sequential replacement.

   ```bash
   EXPERIMENT_KERNEL_SOURCE="$EXPERIMENT_RUN_DIR/kernel-db0084219d62.py"
   git -C ../vllm show \
     db0084219d62a7b7b564adeedb9e6536cadcf0d8:vllm/third_party/flash_linear_attention/ops/chunk.py \
     > "$EXPERIMENT_KERNEL_SOURCE"
   ```

8. Capture sequential Proton scopes with 0 instrumented warmup(s) and 524288 bytes per CTA.

   ```bash
   PYTHONPATH="$PWD/experiments/_shared/gdn/import_overlay" \
   GDN_CHUNK_SOURCE="$EXPERIMENT_KERNEL_SOURCE" \
   CUDA_VISIBLE_DEVICES=4 .venv/bin/python experiments/_shared/gdn/profile_intra.py \
     --implementation sequential --tokens 2048 \
     --instrumented-warmups 0 --instrumentation-buffer-bytes 524288 \
     --output-dir "$EXPERIMENT_RUN_DIR" \
     > "$EXPERIMENT_RUN_DIR/proton-sequential.log" 2>&1
   ```

9. Capture chunked Proton scopes with 0 instrumented warmup(s) and 524288 bytes per CTA.

   ```bash
   PYTHONPATH="$PWD/experiments/_shared/gdn/import_overlay" \
   GDN_CHUNK_SOURCE="$EXPERIMENT_KERNEL_SOURCE" \
   CUDA_VISIBLE_DEVICES=4 .venv/bin/python experiments/_shared/gdn/profile_intra.py \
     --implementation chunked --tokens 2048 \
     --instrumented-warmups 0 --instrumentation-buffer-bytes 524288 \
     --output-dir "$EXPERIMENT_RUN_DIR" \
     > "$EXPERIMENT_RUN_DIR/proton-chunked.log" 2>&1
   ```

10. Open the two `proton_*/intra.chrome_trace` files and a `torch/rank0*.pt.trace.json.gz` file in Perfetto. Check for 48 chunked prefill calls in the server capture and no `_gdn_sequential_kernel` events there.

## Results

| Capture | Recorded result |
|---|---|
| Torch chunked server | 1 successful request; 64 input tokens; 32 output tokens; CPU prefill 207.865 ms |
| Proton sequential | 16,387 events in one kernel; uninstrumented baseline 3.460 ms |
| Proton chunked | 468 events across six kernels; uninstrumented baseline 0.463 ms |

The exploratory baseline ratio is 7.48×. The Torch trace has no sequential kernels; cumsum, scaled-dot, recompute, recurrent-state, and output kernels each occur 48 times. The implementations' max output/state differences were 0.0004883 / 0.004070; instrumented results exactly matched their own uninstrumented baseline. No buffer-overflow warnings were emitted.

## Artifacts

Curated results and hashes: [summary.json](summary.json).

Historical files under `artifacts/`: `torch_chunked/` holds two rank traces, profiler tables, logs, `bench.json`, and the historical server config; `proton_sequential/` and `proton_chunked/` each hold `intra.chrome_trace`, `summary.json`, IR snapshots, and a run log. `original_chunk.py` is the original source snapshot.

New replications use `artifacts/replications/<UTC timestamp>/`; Torch traces go to `torch/` and Proton traces to `proton_<implementation>/`. Each contains or sits beside the corresponding logs and configs.

## Limitations

Another user started a server on GPU 4 during the recorded experiment. Timings are exploratory, not isolated benchmark claims. The first Proton invocations included instrumented compilation and large-buffer collection overhead; the current defaults would not reproduce that recipe without the explicit warmup/buffer flags above.

Proton inputs are synthetic per-layer tensors, not extracted ShareGPT activations. Scopes sample one CTA and one warp; instrumentation changes execution. `load_issue_*`/`matmul_issue_*` do not necessarily include asynchronous completion. Scope names are local to each kernel. Proton displays cycles using an assumed 1 GHz scale, so displayed durations are not calibrated wall-clock nanoseconds. Use uninstrumented `baseline_ms` for comparisons. The 64-token server traces are not a controlled speedup comparison because the initial sequential trace included first-use compilation.

For easier chunked inspection, use [gdn_comparison_warm_smallbuf](../gdn_comparison_warm_smallbuf/README.md). Dependency/model/dataset revisions and cache states were not fully captured; exact bytes and timings are not guaranteed.
