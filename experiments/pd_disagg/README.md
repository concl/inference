# pd_disagg

## Purpose

Preserve and analyze the prefill/decode disaggregation setup supplied in
`reqwen3_6int4trace.zip`. **This is a configuration-only, incomplete experiment
bundle.** It describes disaggregated prefill but contains no measured results,
server logs, or profiler traces. The six supplied files are imported unchanged.

The configured topology is a benchmark client on port 8300, a proxy, a prefill
producer on one GPU, and a decode consumer on another GPU. KV transfer is
configured through NIXL with UCX. Both instances use TP=1 and PP=1, so this is
prefill/decode phase separation rather than layer-wise pipeline parallelism.
Actual routing and successful KV transfer cannot be verified without the proxy
implementation and runtime evidence.

## Configuration

| Setting | Supplied value |
|---|---|
| Launcher | [run.sh](run.sh); `PD_MTP=0`, `PROFILE=0`, `PROFILE_ONLY=0` by default |
| Prefill producer | [prefill.yaml](prefill.yaml), GPU 0 by default, port 8100, `kv_role: kv_producer` |
| Decode consumer | [decode.yaml](decode.yaml), GPU 1 by default, port 8200, `kv_role: kv_consumer` |
| Proxy | `proxy.py`, port 8300; **implementation missing from archive** |
| Transfer | `NixlConnector`, UCX backend, `kv_load_failure_policy: fail`; separate side-channel ports 5600/5601 |
| Per-instance parallelism | Tensor parallelism 1; pipeline parallelism 1 |
| Hardware | Two different CUDA GPUs requested; actual GPU models/UUIDs unknown |
| Model / tokenizer | `/home/kaleox.dev/models-cache/qwen3.8-27b-int4-autoround`; served as `qwen3.8-27b`; checkpoint/revision not supplied |
| Quantization / activations | `auto_round`, BF16 dtype; path labels the checkpoint INT4, but weights/bit width were not inspected |
| Scheduler / memory | Context limit 8192; max sequences 2; batched-token budget 2048; GPU memory utilization 0.92 on each instance |
| Cache / generation | Prefix caching explicitly disabled; thinking disabled; other generation defaults unknown |
| Optional MTP | [prefill-mtp.yaml](prefill-mtp.yaml), [decode-mtp.yaml](decode-mtp.yaml); `PD_MTP=1` selects MTP with 3 draft tokens on both instances |
| Workload | [bench-vision-arena.yaml](bench-vision-arena.yaml); VisionArena train, 20 image requests, requested output length 128, concurrency 2, seed 0, 1 configured warmup |
| Revisions / versions | Model, tokenizer, dataset, and software revisions unknown; the error message's `nixl==1.5.0` suggestion is not proof of an installed version |
| Extra import path | `no_mm_warmup/` added to both servers' `PYTHONPATH`; **overlay missing, so its behavior is unknown** |
| Profiling | Optional Torch profiler on both servers; shapes on, stack/memory off, uncompressed traces, explicit buffer sizes absent |

The archive name suggests Qwen3.6, while every supplied checkpoint path and
served-model name identifies **Qwen3.8-27B**. The imported files retain that
discrepancy; model identity must be established from the actual checkpoint.

The first comment in `prefill.yaml` says text-only, but the benchmark and later
comments specify VisionArena/vision. No explicit text-only server flag is present.
The supplied comments themselves flag multimodal PD and hybrid-SSM/Mamba MTP
compatibility as unverified; no runtime evidence resolves those questions.

## Replication instructions

1. Inspect the supplied files and metadata from the repository root. These checks
   do not start a server or allocate model memory.

   ```bash
   cd /home/lukezeng/inference
   EXPERIMENT_DIR="$PWD/experiments/pd_disagg"
   EXPERIMENT_RUN_DIR="$EXPERIMENT_DIR/artifacts/replications/$(date -u +%Y%m%dT%H%M%SZ)"
   mkdir -p "$EXPERIMENT_RUN_DIR"
   bash -n "$EXPERIMENT_DIR/run.sh"
   .venv/bin/python -m json.tool "$EXPERIMENT_DIR/summary.json"
   ```

2. Restore the original proxy and import overlay before running the launcher.
   This preflight **fails for the supplied bundle**, identifying the missing
   pieces. Their absence is a replication limitation, not an instruction to
   substitute a guessed proxy or silently omit the overlay.

   ```bash
   .venv/bin/python - <<'PY'
   from pathlib import Path
   root = Path('experiments/pd_disagg')
   missing = [str(p) for p in [root/'proxy.py', root/'no_mm_warmup'] if not p.exists()]
   if missing:
       raise SystemExit('Missing supplied-run dependencies: ' + ', '.join(missing))
   PY
   ```

3. Once those dependencies are restored, verify the checkpoint exists at the
   recorded path and NIXL imports in the existing environment. If the checkpoint
   is elsewhere, reconcile the launcher's `MODEL`, all four server YAMLs, and the
   benchmark tokenizer path consistently and retain the changed copies. Record
   exact source, dependency, checkpoint, and dataset identities for the new run.
   The archive contains no GDN kernel override or exact vLLM commit; do not assume
   that the current fork's kernel selection reproduces the original environment.

   ```bash
   test -f /home/kaleox.dev/models-cache/qwen3.8-27b-int4-autoround/config.json
   .venv/bin/python -c 'import nixl; import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["vllm","torch","triton","nixl"]},indent=2))' > "$EXPERIMENT_RUN_DIR/environment.json"
   cp "$EXPERIMENT_DIR/"*.yaml "$EXPERIMENT_DIR/run.sh" "$EXPERIMENT_RUN_DIR/"
   .venv/bin/inference bench --config "$EXPERIMENT_DIR/bench-vision-arena.yaml" --dry-run
   nvidia-smi
   ss -ltn 'sport = :8100 or sport = :8200 or sport = :8300'
   ```

4. On two available GPUs, start with MTP disabled and run an unprofiled smoke
   benchmark. The launcher starts both servers, polls their `/v1/models`, starts
   the proxy, waits for `/healthcheck`, and submits the 20-request workload. It
   requests one benchmark warmup. Select the GPU IDs explicitly; no other user's
   processes should be stopped or bypassed using `FORCE=1`.

   ```bash
   OUT="$EXPERIMENT_RUN_DIR" GPU_PREFILL=0 GPU_DECODE=1 \
     PD_MTP=0 PROFILE=0 PROFILE_ONLY=0 bash "$EXPERIMENT_DIR/run.sh"
   ```

   Confirm successful image requests, actual producer/consumer routing, and KV
   transfer in logs/connector metrics. Compare generated outputs against ordinary
   serving before treating the combined setup as correct. For a performance
   comparison, add an ordinary-serving baseline with the same hardware budget,
   model, workload, and cache policy; none was supplied in this archive.

5. Capture traces separately from performance measurements, into a fresh output
   directory. The profiling branch requests one unprofiled warmup, explicitly
   starts both profilers, submits two requests with no extra configured warmups,
   and stops/flushes both profilers. Profiler buffers and iteration limits are not
   set in the supplied launcher; record the effective settings of the chosen
   runtime before relying on its defaults. MTP is a separate compatibility test.

   ```bash
   EXPERIMENT_TRACE_DIR="$EXPERIMENT_DIR/artifacts/captures/$(date -u +%Y%m%dT%H%M%SZ)"
   OUT="$EXPERIMENT_TRACE_DIR" GPU_PREFILL=0 GPU_DECODE=1 \
     PD_MTP=0 PROFILE=1 PROFILE_ONLY=1 bash "$EXPERIMENT_DIR/run.sh"
   ```

6. The launcher's exit trap terminates its proxy and both server processes.
   Inspect all logs, benchmark completed/failed counts, and both trace directories;
   verify the selected GPUs are released. A final `done` message is insufficient:
   the supplied launcher continues after several warmup, profiling, and benchmark
   failures. Keep the original archive and previous captures unchanged.

   ```bash
   find "$EXPERIMENT_RUN_DIR" -maxdepth 2 -type f
   nvidia-smi
   ```

## Results

**Static classification: configured disaggregated prefill, not runtime-verified.**
The classification follows the distinct `kv_producer` and `kv_consumer` roles,
different server/GPU assignments, shared NIXL/UCX transfer configuration, and the
benchmark targeting a proxy rather than either worker directly.

The ZIP contains exactly six small shell/YAML files. Despite `trace` in its name,
it contains **zero profiler traces, benchmark result JSONs, or server logs**.
There are no observed TTFT, throughput, transfer timings, correctness results, or
speedups to report. [summary.json](summary.json) records the configuration analysis,
missing dependencies, and hashes; its status is not a successful benchmark run.

## Artifacts

- Tracked: all six supplied files, imported byte-for-byte; this README; and
  [summary.json](summary.json), including individual SHA256 hashes.
- Original ZIP SHA256:
  `c0e2372cd9ce2f38b4a8284d1200827927017a7eb7faca8d465a237cdc3dd8f9`.
- Ignored original archive copy: `artifacts/original/reqwen3_6int4trace.zip`.
- The upload at the repository root is retained; it is not added to Git.
- Future results: fresh `artifacts/replications/<UTC timestamp>/` and
  `artifacts/captures/<UTC timestamp>/` directories.

## Limitations

The missing proxy prevents inspection of transfer metadata, prompt forwarding,
image handling, streaming, and actual phase assignment. The missing import overlay
prevents identification of all effective code changes. Neither dependency is
reconstructed by this import. Model weights, original environment, and runtime
outputs are absent, so correctness, compatibility, successful disaggregation,
and performance are unverified.

The benchmark requests only 20 prompts at concurrency 2, without repeated trials
or an ordinary-serving baseline. That is a smoke-test specification, not strong
evidence about latency tails, statistical significance, or performance benefits.
Dataset images and prompt lengths may vary; achieved output lengths are unknown.

In the local vLLM benchmark parser, `no-stream: true` controls Hugging Face dataset
loading; it does not itself disable HTTP response streaming. Actual proxy
streaming cannot be inspected. CUDA Graph execution is also not explicitly set:
the comment about a "safe first default" has no corresponding `enforce-eager`
flag. Versions, model/dataset revisions, GPU identities, and profiler settings
must be recorded for any new run rather than inferred from these comments.
