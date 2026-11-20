# exp7

## Purpose

Measure the effect of tensor parallelism (TP=2 versus TP=4) and vLLM's multimodal encoder mode (`weights` versus `data`) on VisionArena image requests with Qwen3.8-27B-FP8. Compare encoder modes at each fixed TP size, then compare TP sizes with the encoder mode held fixed on the same Ynez server.

## Configuration

| Setting | Recorded value |
|---|---|
| Server configs | [config-tp2-weights.yaml](config-tp2-weights.yaml), [config-tp2-data.yaml](config-tp2-data.yaml), [config-tp4-weights.yaml](config-tp4-weights.yaml), [config-tp4-data.yaml](config-tp4-data.yaml); reconstructed from saved CLI commands |
| Benchmark config | [benchmark-vision.yaml](benchmark-vision.yaml); concurrency and result filename overridden for each run |
| vLLM source | `2de0531d81525b3fd652ca2f89810b2ff038395e`; server log reports `v0.1.dev23615+g2de0531d8` |
| Hardware | Ynez RTX 3090s (24 GiB each); TP=2: physical GPUs 0, 1; TP=4: physical GPUs 0, 1, 2, 3 |
| Model/tokenizer | `Qwen/Qwen3.8-27B-FP8`, FP8 checkpoint with BF16 dtype; recorded checkpoint revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` |
| Software versions | Historical vLLM build recorded above; exact installed Torch, Triton, Transformers, and Datasets versions not saved |
| Dataset | `lmarena-ai/vision-arena-bench-v0.1` train; revision observed at run start: `3abe18efa30d674a8903386b463ce278e7f41f2d` |
| Workload | Seed 42, 100 image requests, 64 output tokens, temperature 0; concurrency 1, 4, 8; two measured repeats per condition |
| Server / cache | Max model len 8192, max sequences 8, GPU memory utilization 0.90, prefix caching off; no speculative decoding or pipeline parallelism configured |
| Warmup / profiler | Five unmeasured requests at concurrency 2 before each mode; no profiler |

## Replication instructions

1. Follow [replication prerequisites](../README.md#replication-prerequisites). Reserve four free RTX 3090s and port 8134. From the repository root, create a fresh output directory and record installed versions and the dataset revision. Download or reuse the recorded model revision; exact historical dependency versions are not fully known.

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   EXPERIMENT_RUN_DIR="$PWD/experiments/exp7/artifacts/replications/$(date -u +%Y%m%dT%H%M%SZ)"
   mkdir -p "$EXPERIMENT_RUN_DIR"
   .venv/bin/python -c 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["vllm","torch","triton","transformers","datasets"]},indent=2))' > "$EXPERIMENT_RUN_DIR/environment.json"
   .venv/bin/python -c 'from huggingface_hub import HfApi; import json; print(json.dumps({"model_revision":"c3a462cc3d38cf351f3b26cbd763469eeb74ef81","dataset_revision":HfApi().dataset_info("lmarena-ai/vision-arena-bench-v0.1").sha},indent=2))' > "$EXPERIMENT_RUN_DIR/revisions.json"
   .venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.8-27B-FP8",revision="c3a462cc3d38cf351f3b26cbd763469eeb74ef81"))' > "$EXPERIMENT_RUN_DIR/model-path.txt"
   EXPERIMENT_MODEL_DIR="$(cat "$EXPERIMENT_RUN_DIR/model-path.txt")"
   ```

2. Use a clean personal checkout of the vLLM fork at commit `2de0531d81525b3fd652ca2f89810b2ff038395e`. No custom kernel option was recorded. The historical run imported a private copy of this fork; the exact copy was not archived. Verify the selected source and validate the YAML through this repository's wrappers before loading the model.

   ```bash
   test "$(git -C ../vllm rev-parse HEAD)" = 2de0531d81525b3fd652ca2f89810b2ff038395e
   test -z "$(git -C ../vllm status --porcelain)"
   git -C ../vllm rev-parse HEAD > "$EXPERIMENT_RUN_DIR/vllm-commit.txt"
   for tp in 4 2; do
     if [ "$tp" = 4 ]; then gpus=0,1,2,3; else gpus=0,1; fi
     for mode in weights data; do
       .venv/bin/inference serve --config "experiments/exp7/config-tp${tp}-${mode}.yaml" --gpus "$gpus" --dry-run
     done
   done
   .venv/bin/inference bench --config experiments/exp7/benchmark-vision.yaml --tokenizer "$EXPERIMENT_MODEL_DIR" --dry-run
   ```

3. Run TP=4 first, then TP=2; within each TP size, run `weights` first, then `data`, as in the original captures. Confirm all selected GPUs and port 8134 are free before each server launch. The loop starts only its own server, waits up to 15 minutes for readiness, runs the explicit warmup, saves two 100-request repetitions at each concurrency, and stops that server before changing mode or TP size.

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
   for tp in 4 2; do
     if [ "$tp" = 4 ]; then gpus=0,1,2,3; else gpus=0,1; fi
     for mode in weights data; do
       mode_dir="$EXPERIMENT_RUN_DIR/tp${tp}/$mode"
       mkdir -p "$mode_dir"
       cp "experiments/exp7/config-tp${tp}-${mode}.yaml" "$mode_dir/config.yaml"
       cp experiments/exp7/benchmark-vision.yaml "$mode_dir/benchmark-vision.yaml"
       PYTHONPATH="$PWD/../vllm" .venv/bin/inference serve \
         --config "$mode_dir/config.yaml" --gpus "$gpus" > "$mode_dir/server.log" 2>&1 &
       server_pid=$!
       for attempt in $(seq 1 180); do
         if curl -fsS http://127.0.0.1:8134/health >/dev/null; then break; fi
         sleep 5
       done
       curl -fsS http://127.0.0.1:8134/health >/dev/null
       PYTHONPATH="$PWD/../vllm" .venv/bin/inference bench \
         --config "$mode_dir/benchmark-vision.yaml" --tokenizer "$EXPERIMENT_MODEL_DIR" --num-prompts 5 \
         --max-concurrency 2 --result-dir "$mode_dir" \
         --result-filename warmup.json > "$mode_dir/warmup.log" 2>&1
       for concurrency in 1 4 8; do
         for repeat in 1 2; do
           PYTHONPATH="$PWD/../vllm" .venv/bin/inference bench \
             --config "$mode_dir/benchmark-vision.yaml" --tokenizer "$EXPERIMENT_MODEL_DIR" \
             --max-concurrency "$concurrency" --result-dir "$mode_dir" \
             --result-filename "c${concurrency}-r${repeat}.json" \
             > "$mode_dir/c${concurrency}-r${repeat}.log" 2>&1
         done
       done
       stop_server
     done
   done
   trap - EXIT
   ```

4. Check all 24 measured JSON files for 100 completed and zero failed requests. Save the server logs and the environment/revision manifests with the raw artifacts. Verify the selected GPUs are released with `nvidia-smi`.

## Results

Each cell is the arithmetic mean of two 100-request runs; the p95 columns average the two per-run p95 values, rather than pooling all 200 requests. Per-run TTFT, ITL, request rate, and throughput values are in [summary.json](summary.json), grouped by TP size with the same measurement and average fields as the 4B benchmark.

With TP=2:

| Concurrency | `weights` output tok/s | `data` output tok/s | `weights` p95 TTFT (ms) | `data` p95 TTFT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 33.50 | 34.53 | 932 | 875 |
| 4 | 67.99 | 73.10 | 2811 | 2690 |
| 8 | 83.36 | 91.00 | 4925 | 4334 |

With TP=4:

| Concurrency | `weights` output tok/s | `data` output tok/s | `weights` p95 TTFT (ms) | `data` p95 TTFT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 42.42 | 45.17 | 936 | 885 |
| 4 | 77.35 | 84.92 | 3162 | 2974 |
| 8 | 89.43 | 100.27 | 4981 | 4257 |

All 2,400 measured requests completed with zero failures. In this workload, `data` had higher output throughput and lower p95 TTFT than `weights` at all three concurrency settings for both TP sizes. TP=4 had higher total output throughput than TP=2 at each matching mode and concurrency, but did not consistently reduce p95 TTFT. At concurrency 8 with `data`, doubling GPUs increased total output throughput from 91.00 to 100.27 tok/s (about 10%); this is not a doubling in throughput per GPU.

## Artifacts

Historical raw JSON, server/benchmark logs, metadata, exact CLI command arrays, GPU snapshots, and the driver script are under ignored `artifacts/original-20261112/`. The `tp2/` and `tp4/` directories preserve the two captures separately. They are not part of the tracked summary. New replications use `artifacts/replications/<UTC timestamp>/tp<TP>/<mode>/` and keep each mode's logs and JSON separately. Model weights and dataset images are not stored in this repository.

## Limitations

The exact installed dependency versions and historical private vLLM source copy were not saved, although the fork commit and server version were recorded. The benchmark recorded the dataset revision but did not pin it in the loader or save selected image IDs; seed 42 reproduces the subset only when dataset contents/order and sampling implementation match. The YAMLs use a portable Hub model name and pin the recorded checkpoint revision, while the original command used a local checkpoint path. There are only two repeats per setting, TP=4 preceded TP=2, the encoder modes ran in a fixed order on a shared server, and no profiler traces were collected. The comparison changes GPU count and topology along with TP size; it measures total throughput on the selected layouts rather than an isolated communication or per-GPU efficiency effect. These FP8 results do not isolate model size from quantization relative to the BF16 4B benchmark.
