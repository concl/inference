# gdn_sequential_torch_warm

## Purpose

Repeat the sequential PyTorch serving capture after request-level warmup, and
check whether the first-use compilation gap inside prefill disappears. No trace
timestamps or gaps are edited. The initial comparison is
[gdn_sequential_initial](../gdn_sequential_initial/README.md).

## Configuration

| Setting | Recorded value |
|---|---|
| Capture | `artifacts/20261029T215922Z/` |
| Source / kernel | `db0084219d62a7b7b564adeedb9e6536cadcf0d8`, selected with the import overlay |
| Software / dataset SHA256 / source SHA256 / GPU UUIDs | `artifacts/20261029T215922Z/environment.json` |
| Hardware / physical GPUs / TP | RTX 3090 / 1,3 / 2 |
| Model / tokenizer | `lued/Qwen3.8-27B-INT8-W8A16-MTP` |
| Model and tokenizer revision | `7c12373712d1363e2b76655cb3332c9c124627d7`; cached files, offline mode |
| Server | 8,192 context; one active sequence; 2,048 batched-token budget; port 8010 |
| Workload | ShareGPT seed 0, one request, 32 output-token cap, temperature 0 |
| Request warmup | Successful endpoint-check request, then 2 additional warmup requests before profiling |
| Cache policy | Prefix caching off; existing compilation caches retained |
| MTP | Enabled, 3 speculative tokens |
| Torch profiler | CPU + CUDA, shapes on; stacks, memory, and frontend profiling off |
| Proton buffer | Not applicable; this is a Torch capture |

The experiment owns [config.yaml](config.yaml) and
[benchmark-sharegpt.yaml](benchmark-sharegpt.yaml). Historical snapshots remain
inside the capture directory; the tracked copies preserve their settings with
portable output paths.

## Replication instructions

1. Read [the prerequisites](../README.md#replication-prerequisites), check that
   GPUs 1 and 3 and port 8010 are available, and create a fresh directory. Run all
   commands in the same Bash terminal, starting inside this repository.

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   nvidia-smi
   ss -ltnp 'sport = :8010'
   EXPERIMENT_RUN_DIR="$PWD/experiments/gdn_sequential_torch_warm/artifacts/replications/$(date -u +%Y%m%dT%H%M%SZ)"
   mkdir -p "$EXPERIMENT_RUN_DIR"
   .venv/bin/python -c 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["vllm","torch","triton","transformers"]},indent=2))' > "$EXPERIMENT_RUN_DIR/environment.json"
   nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv > "$EXPERIMENT_RUN_DIR/gpus.txt"
   sha256sum ../benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json > "$EXPERIMENT_RUN_DIR/dataset.sha256"
   ```

2. Snapshot the exact sequential kernel and copy the recorded configuration.
   Update output paths in the copies, keeping all other settings explicit.

   ```bash
   git -C ../vllm show \
     db0084219d62a7b7b564adeedb9e6536cadcf0d8:vllm/third_party/flash_linear_attention/ops/chunk.py \
     > "$EXPERIMENT_RUN_DIR/kernel-db0084219d62.py"
   sha256sum "$EXPERIMENT_RUN_DIR/kernel-db0084219d62.py" > "$EXPERIMENT_RUN_DIR/kernel.sha256"
   cp experiments/gdn_sequential_torch_warm/config.yaml "$EXPERIMENT_RUN_DIR/config.yaml"
   cp experiments/gdn_sequential_torch_warm/benchmark-sharegpt.yaml "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml"
   .venv/bin/python - "$EXPERIMENT_RUN_DIR" <<'PY'
   import pathlib, sys, yaml
   root = pathlib.Path(sys.argv[1])
   for name in ["config.yaml", "benchmark-sharegpt.yaml"]:
       path = root / name
       config = yaml.safe_load(path.read_text())
       if name == "config.yaml":
           config["profiler_config"]["torch_profiler_dir"] = str(root / "torch")
       else:
           config["result-dir"] = str(root / "bench")
       path.write_text(yaml.safe_dump(config, sort_keys=False))
   PY
   .venv/bin/inference bench --config "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml" \
     --dry-run > "$EXPERIMENT_RUN_DIR/bench-dry-run.txt" 2>&1
   ```

3. Start the server with only the snapshotted chunk module overridden. The
   installed package and fork checkout are not modified. Offline mode requires
   the pinned model and tokenizer revision to be cached already.

   ```bash
   HF_HUB_OFFLINE=1 \
   PYTHONPATH="$PWD/experiments/_shared/gdn/import_overlay" \
   GDN_CHUNK_SOURCE="$EXPERIMENT_RUN_DIR/kernel-db0084219d62.py" \
     .venv/bin/python -m scripts.serve \
     --config "$EXPERIMENT_RUN_DIR/config.yaml" --gpus 1,3 \
     > "$EXPERIMENT_RUN_DIR/serve.log" 2>&1 &
   EXPERIMENT_SERVER_PID=$!
   ```

4. Run the benchmark. It waits up to 600 seconds for readiness and a successful
   unprofiled endpoint-check request, then sends two additional unprofiled
   requests. Only after those finish does it start the profiler and send the
   measured request. Prefix caching remains disabled, so prefill still executes.

   ```bash
   HF_HUB_OFFLINE=1 .venv/bin/inference bench \
     --config "$EXPERIMENT_RUN_DIR/benchmark-sharegpt.yaml" \
     > "$EXPERIMENT_RUN_DIR/bench-sharegpt.log" 2>&1
   ```

5. Wait for `Profiler stopped`, check one successful measured request, and stop
   the server. Preserve the logs, benchmark JSON, and both rank traces.

   ```bash
   rg 'Successful requests|Failed requests|Profiler stopped' "$EXPERIMENT_RUN_DIR/bench-sharegpt.log"
   kill -TERM "$EXPERIMENT_SERVER_PID"
   wait "$EXPERIMENT_SERVER_PID"
   find "$EXPERIMENT_RUN_DIR/torch" -name '*.pt.trace.json.gz'
   ```

6. Quantify the largest gap between GPU activities belonging to prefill. Match
   profiler `External id` values to CPU events inside the annotation rather than
   clipping at CPU scope exit: asynchronous GPU work may finish after that exit.
   Include kernels and GPU copies/memsets across streams, merge overlapping
   intervals, and report internal gaps rather than padding around the GPU work.
   Compare the historical rank-0 trace using the same analyzer if it is present.

   ```bash
   .venv/bin/python experiments/gdn_sequential_torch_warm/analyze.py \
     "$EXPERIMENT_RUN_DIR"/torch/rank*.pt.trace.json.gz \
     > "$EXPERIMENT_RUN_DIR/prefill-analysis.json"
   .venv/bin/python experiments/gdn_sequential_torch_warm/analyze.py \
     experiments/gdn_sequential_initial/artifacts/torch/rank0.1793223620414335339.pt.trace.json.gz
   ```

7. Before sharing, redact the local home-directory prefix from text artifacts
   and trace JSON strings without changing timing values, then hash the files.
   Run this after shutdown so the server cannot append new unsanitized log lines.

   ```bash
   .venv/bin/python - "$EXPERIMENT_RUN_DIR" <<'PY'
   import gzip, hashlib, json, pathlib, sys
   root = pathlib.Path(sys.argv[1])
   prefix = str(pathlib.Path.home())
   for path in root.rglob("*"):
       if not path.is_file():
           continue
       if path.suffix == ".pyc":
           path.unlink()  # Generated bytecode embeds the original source path.
           continue
       if path.name.endswith(".pt.trace.json.gz"):
           text = gzip.decompress(path.read_bytes()).decode()
           path.write_bytes(gzip.compress(text.replace(prefix, "<USER_HOME>").encode(), mtime=0))
       elif path.suffix in {".log", ".json", ".yaml", ".txt"}:
           path.write_text(path.read_text().replace(prefix, "<USER_HOME>"))
   hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "hashes.json"}
   (root / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
   PY
   ```

   The original capture's curated results and trace hashes are in `summary.json`.

## Results

The benchmark completed one measured request successfully, with 64 input tokens
and 32 generated tokens. The readiness request and two extra warmup requests ran
before profiling. The server logged first-use compilation during warmup but
**zero JIT warnings during the measured capture**.

| Rank-0 measurement | Initial capture | Warmed capture |
|---|---:|---:|
| CPU prefill annotation | 393.436 ms | 191.212 ms |
| Correlated GPU activity span | 369.964 ms | 181.337 ms |
| Largest gap between prefill GPU activities | 228.860 ms | 18.388 ms |
| Longest enclosed CPU operation | 230.156 ms | 10.912 ms |
| Sequential GDN kernel executions | 48 | 48 |

The isolated first-use spike is absent, and the largest recorded internal gap
is about 12.45x smaller. Residual host/profiler delays remain; gaps were not
removed or shifted in postprocessing. Rank 1 also contains 48 sequential kernels
and has a largest correlated GPU gap of 18.382 ms. Values come from the saved
`prefill-analysis.json`, `baseline-prefill-analysis.json`, and benchmark JSON.

Perfetto imports the trace and reports the known overlapping-complete-event
layout diagnostic (`slice_spill_overlapping_complete_event`, 1,227 occurrences
on rank 0). Its diagnostic says the events are moved to overflow tracks without
data loss. This capture addresses the compilation spike, not that layout issue.

## Artifacts

`artifacts/20261029T215922Z/` contains the exact YAML snapshots, kernel source,
environment manifest, dry-run validation, server/benchmark logs, benchmark JSON,
prefill analysis, and per-rank traces in `torch/`. Raw artifacts are gitignored.
The preferred rank-0 trace is
`artifacts/20261029T215922Z/torch/rank0.1793311672023744218.pt.trace.json.gz`;
rank 1 is `rank1.1793311672058058161.pt.trace.json.gz` in the same directory.
`summary.json` contains curated results and sanitized-artifact hashes.
Replications go into a new `artifacts/replications/<UTC timestamp>/` directory.

## Limitations

This captures the educational sequential kernel, not the original chunked
implementation. The import overlay pins that module only; other installed vLLM
files are described by the environment manifest rather than a clean source tree.
No shared compilation caches were cleared. The original cold trace used
stochastic generation; this experiment uses temperature 0, so decode behavior
and output text are not directly comparable. PyTorch recording still perturbs
execution, and a gap between recorded GPU activities does not by itself explain
its cause. Other GPUs and the host are shared. The intent is to remove the
first-use prefill spike, not to claim a model throughput improvement.
