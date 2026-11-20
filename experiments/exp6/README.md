# exp6

## Purpose

Measure the effect of vLLM's multimodal encoder mode (`weights` versus `data`) on VisionArena image requests. Tensor parallelism stays at 2 on the same two Ynez RTX 3090s.

## Configuration

| Setting | Recorded value |
|---|---|
| Server configs | [config-weights.yaml](config-weights.yaml), [config-data.yaml](config-data.yaml); reconstructed from saved CLI commands |
| Benchmark config | [benchmark-vision.yaml](benchmark-vision.yaml); concurrency and result filename overridden for each run |
| vLLM source | `2de0531d81525b3fd652ca2f89810b2ff038395e`; server log reports `v0.1.dev23615+g2de0531d8` |
| Hardware | Ynez, physical GPUs 0 and 1, 2 × RTX 3090 (24 GiB each), TP=2 |
| Model/tokenizer | `Qwen/Qwen3-VL-4B-Instruct`, BF16; historical checkpoint revision not recorded |
| Software versions | Historical vLLM build recorded above; exact installed Torch, Triton, Transformers, and Datasets versions not saved |
| Dataset | `lmarena-ai/vision-arena-bench-v0.1` train; revision observed at run start: `3abe18efa30d674a8903386b463ce278e7f41f2d` |
| Workload | Seed 42, 100 image requests, 64 output tokens, temperature 0; concurrency 1, 4, 8; two measured repeats per condition |
| Server / cache | Max model len 8192, max sequences 8, GPU memory utilization 0.90, prefix caching off |
| Warmup / profiler | Five unmeasured requests at concurrency 2 before each mode; no profiler |

## Replication instructions

1. Follow [replication prerequisites](../README.md#replication-prerequisites). Reserve two free RTX 3090s and port 8134. From the repository root, create a fresh output directory and record installed versions and current Hub revisions. Historical model and dependency revisions are not fully known.

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   EXPERIMENT_RUN_DIR="$PWD/experiments/exp6/artifacts/replications/$(date -u +%Y%m%dT%H%M%SZ)"
   mkdir -p "$EXPERIMENT_RUN_DIR"
   .venv/bin/python -c 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["vllm","torch","triton","transformers","datasets"]},indent=2))' > "$EXPERIMENT_RUN_DIR/environment.json"
   .venv/bin/python -c 'from huggingface_hub import HfApi; import json; a=HfApi(); print(json.dumps({"model_revision":a.model_info("Qwen/Qwen3-VL-4B-Instruct").sha,"dataset_revision":a.dataset_info("lmarena-ai/vision-arena-bench-v0.1").sha},indent=2))' > "$EXPERIMENT_RUN_DIR/revisions.json"
   ```

2. Use a clean personal checkout of the vLLM fork at commit `2de0531d81525b3fd652ca2f89810b2ff038395e`. No custom kernel option was recorded. The historical run imported a private copy of this fork; the exact copy was not archived. Verify the selected source and validate the YAML through this repository's wrappers before loading the model.

   ```bash
   test "$(git -C ../vllm rev-parse HEAD)" = 2de0531d81525b3fd652ca2f89810b2ff038395e
   test -z "$(git -C ../vllm status --porcelain)"
   git -C ../vllm rev-parse HEAD > "$EXPERIMENT_RUN_DIR/vllm-commit.txt"
   .venv/bin/inference serve --config experiments/exp6/config-weights.yaml --gpus 0,1 --dry-run
   .venv/bin/inference serve --config experiments/exp6/config-data.yaml --gpus 0,1 --dry-run
   .venv/bin/inference bench --config experiments/exp6/benchmark-vision.yaml --dry-run
   ```

3. Run `weights` first, then `data`, as in the original capture. Confirm GPUs 0 and 1 and port 8134 are free before starting. The loop starts only its own server, waits up to 15 minutes for readiness, runs the explicit warmup, saves two 100-request repetitions at each concurrency, and stops that server before changing mode.

   ```bash
   set -e
   server_pid=
   stop_server() {
     if [ -n "$server_pid" ]; then
       kill -TERM "$server_pid" 2>/dev/null || true
       wait "$server_pid" 2>/dev/null || true
       server_pid=
     fi
   }
   trap stop_server EXIT
   for mode in weights data; do
     mode_dir="$EXPERIMENT_RUN_DIR/$mode"
     mkdir -p "$mode_dir"
     cp "experiments/exp6/config-$mode.yaml" "$mode_dir/config.yaml"
     cp experiments/exp6/benchmark-vision.yaml "$mode_dir/benchmark-vision.yaml"
     PYTHONPATH="$PWD/../vllm" .venv/bin/inference serve \
       --config "$mode_dir/config.yaml" --gpus 0,1 > "$mode_dir/server.log" 2>&1 &
     server_pid=$!
     for attempt in $(seq 1 180); do
       if curl -fsS http://127.0.0.1:8134/health >/dev/null; then break; fi
       sleep 5
     done
     curl -fsS http://127.0.0.1:8134/health >/dev/null
     PYTHONPATH="$PWD/../vllm" .venv/bin/inference bench \
       --config "$mode_dir/benchmark-vision.yaml" --num-prompts 5 \
       --max-concurrency 2 --result-dir "$mode_dir" \
       --result-filename warmup.json > "$mode_dir/warmup.log" 2>&1
     for concurrency in 1 4 8; do
       for repeat in 1 2; do
         PYTHONPATH="$PWD/../vllm" .venv/bin/inference bench \
           --config "$mode_dir/benchmark-vision.yaml" \
           --max-concurrency "$concurrency" --result-dir "$mode_dir" \
           --result-filename "c${concurrency}-r${repeat}.json" \
           > "$mode_dir/c${concurrency}-r${repeat}.log" 2>&1
       done
     done
     stop_server
   done
   trap - EXIT
   ```

4. Check all 12 measured JSON files for 100 completed and zero failed requests. Save the server logs and the environment/revision manifests with the raw artifacts. Verify the two selected GPUs are released with `nvidia-smi`.

## Results

Each cell is the arithmetic mean of two 100-request runs; the p95 columns average the two per-run p95 values, rather than pooling all 200 requests. Per-run TTFT, ITL, request rate, and throughput values are in [summary.json](summary.json).

| Concurrency | `weights` output tok/s | `data` output tok/s | `weights` p95 TTFT (ms) | `data` p95 TTFT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 88.74 | 94.99 | 337 | 256 |
| 4 | 197.43 | 234.56 | 811 | 667 |
| 8 | 237.68 | 298.84 | 1708 | 1149 |

All 1,200 measured requests completed with zero failures. In this workload, `data` had higher output throughput and lower p95 TTFT at all three concurrency settings.

## Artifacts

Historical raw JSON, server/benchmark logs, metadata, and exact CLI command arrays are under ignored `artifacts/original-20261105/`. They are not part of the tracked summary. New replications use `artifacts/replications/<UTC timestamp>/` and keep each mode's logs and JSON separately. Model weights and dataset images are not stored in this repository.

## Limitations

The successful run folder was named as a 27B fallback because the 27B server failed first; every result in this experiment is from the 4B model. The historical 4B model revision and exact installed dependency versions were not captured. The historical private vLLM source copy was not saved, although the fork commit and server version were recorded. The benchmark recorded the dataset revision but did not pin it in the loader or save selected image IDs; seed 42 reproduces the subset only when dataset contents/order and sampling implementation match. The YAMLs use a portable Hub model name, while the original command used a local checkpoint path. The earlier single-GPU 4B run used a different memory setting and one repeat, so it is not a controlled TP comparison. There are only two repeats per setting, the modes ran in a fixed order on a shared server, and no profiler traces were collected.
