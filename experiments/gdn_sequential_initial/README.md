# gdn_sequential_initial

## Purpose

Capture the educational sequential GDN kernel through the two-GPU server with Torch profiler.

## Configuration

| Setting | Recorded value |
|---|---|
| Server config | [config.yaml](config.yaml) |
| Benchmark config | [benchmark-sharegpt.yaml](benchmark-sharegpt.yaml) |
| GPUs / TP | Physical GPUs 1 and 3 / TP=2 |
| Model | `lued/Qwen3.8-27B-INT8-W8A16-MTP` |
| Request | Seed 0 ShareGPT sample; 64 input tokens, 32 output-token cap, concurrency 1 |
| Extra request warmups / endpoint check | 0 / disabled |
| MTP / prefix caching | 3 draft tokens / disabled |
| Profiler | Torch; shapes on, stacks and memory off |

## Replication instructions

1. Read the [replication prerequisites](../README.md#replication-prerequisites), then create a new output directory. Run all steps in the same Bash terminal.

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   EXPERIMENT_RUN_DIR="$PWD/experiments/gdn_sequential_initial/artifacts/replications/$(date -u +%Y%m%dT%H%M%SZ)"
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
   cp experiments/gdn_sequential_initial/config.yaml "$EXPERIMENT_RUN_DIR/config.yaml"
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
   cp experiments/gdn_sequential_initial/benchmark-sharegpt.yaml "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml"
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

## Results

One successful request generated 32 tokens. Rank 0 contains 48 `_gdn_sequential_kernel` GPU events, approximately 78 µs each. CPU prefill spans about 393 ms and includes a first-use `layer_norm_fwd_kernel` compilation spike.

## Artifacts

Historical captures: `artifacts/torch/rank0.1793223620414335339.pt.trace.json.gz` and `rank1.1793223620416460087.pt.trace.json.gz`. Profiler tables and recovered server/benchmark logs are in `artifacts/torch/`; the original server config is `artifacts/serve-config.yaml`.

New replications: `artifacts/replications/<UTC timestamp>/torch/` contains rank traces; sibling files contain configs, environment versions, logs, and benchmark JSON. The historical run did not save benchmark JSON; the replication recipe adds that output without changing the request.

## Limitations

No explicit request warmup ran, but startup already performed model/kernel warmup. Disk JIT-cache contents were not recorded, so the same first-use spike cannot be guaranteed by these steps. Do not clear shared caches to force it. Dependency/model/dataset revisions were not completely pinned, and sampling is stochastic. Compare execution structure, not exact timestamps or output text.
