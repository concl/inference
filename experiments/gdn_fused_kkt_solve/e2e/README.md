# Quantized Qwen3.5-4B end-to-end GDN comparison

## Purpose

Test whether the KKT/solve port improves client-observed request latency in a
complete quantized model, compared with the original chunked GDN implementation.
The installed educational sequential implementation is explicitly bypassed in
both conditions. This study changes only GDN prefill dispatch inside one server.

## Configuration

- Model: `axteam/Qwen3.5-4B-AWQ-4bit`, pinned to
  `ba968c07f40fa0bce733c2264d860d2fa671cfcf`. Compressed-tensors W4A16 AWQ,
  asymmetric 4-bit weights, group size 32, Marlin linear kernels, BF16 activations.
- One RTX 3090, TP=1, text-only. No MTP, no prefix caching, no profiler or GPU
  clock changes. Both variants use the same quantized weights and server.
- [serve.yaml](serve.yaml) records scheduling, memory, and compilation settings.
  PyTorch compilation is disabled; full CUDA Graphs are enabled for decode only.
  Prefill stays dynamic so an RPC can select the tested variant between blocks.
- The import overlay and worker extension are subprocess-local. Installed vLLM
  and the sibling fork are unchanged. Per-variant call counters verify dispatch.
- Before timing, real-activation GDN outputs and states are compared with the
  baseline at approximately 2048 and 8192 input tokens. Existing parity gates:
  max absolute error <=0.02 and relative L2 error <=0.01, all values finite.
- Prompts concatenate/truncate passages from the cached ShareGPT corpus into
  long documents. This is a constructed long-input workload, not a replay of
  unmodified ShareGPT conversations. Actual token counts and prompt text are saved.
- Pilot: 2048/32, 8192/32, and 8192/128 approximate input/output token workloads;
  four paired blocks per workload, two serial requests per arm.
- Confirmation: fixed in advance at approximately 8192 input tokens, exactly
  32 generated tokens, concurrency 1, 24 paired blocks, four requests per arm.
  A new prompt seed is used; pilot requests are excluded from confirmation.
- Every pair uses the same prompts for baseline and fused. AB/BA order is
  independently randomized. Both variants and all measured shapes are warmed
  before the first measured block. GPU contention is checked at block boundaries.
- Statistical unit: paired block, not individual request. Report geometric mean
  baseline/fused latency ratio, paired-block bootstrap 95% interval, and two-sided
  sign-flip permutation p-value. Completion latency is primary; TTFT is secondary.
  Holm correction controls the family of two predeclared endpoint tests. Intervals
  are marginal 95% intervals, not multiplicity-adjusted simultaneous intervals.
  No optional stopping based on p-values; all 24 confirmation pairs must finish.

## Replication instructions

1. From the repository root, use the existing `.venv`. Download the pinned
   checkpoint into the Hugging Face cache, select one idle GPU, and create a fresh
   artifact directory. No package installation is required.

   ```bash
   cd /home/lukezeng/inference
   EXPERIMENT_E2E_DIR="$PWD/experiments/gdn_fused_kkt_solve/e2e"
   EXPERIMENT_E2E_RUN="$PWD/experiments/gdn_fused_kkt_solve/artifacts/e2e-$(date -u +%Y%m%dT%H%M%SZ)"
   mkdir -p "$EXPERIMENT_E2E_RUN"
   .venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("axteam/Qwen3.5-4B-AWQ-4bit",revision="ba968c07f40fa0bce733c2264d860d2fa671cfcf",allow_patterns=["*.json","*.safetensors","*.jinja","*.txt","README.md"]))' > "$EXPERIMENT_E2E_RUN/model-path.txt"
   EXPERIMENT_MODEL_DIR="$(tail -n 1 "$EXPERIMENT_E2E_RUN/model-path.txt")"
   nvidia-smi
   # Set this to the UUID of the idle GPU, not a GPU owned by another job.
   EXPERIMENT_GPU_UUID=GPU-d6327ec5-dcea-859e-51cf-875c92cbc0a8
   .venv/bin/python "$EXPERIMENT_E2E_DIR/prepare.py" \
     --output-dir "$EXPERIMENT_E2E_RUN/source" --gpu-uuid "$EXPERIMENT_GPU_UUID"
   .venv/bin/python "$EXPERIMENT_E2E_DIR/test_analysis.py"
   ```

2. Start the server on loopback and retain its PID. The developer RPC endpoint
   is enabled only to switch GDN variants; the server binds to `127.0.0.1`.
   The explicit negative prefix-cache flag avoids the top-level YAML parser's
   omission of false-valued flags.

   ```bash
   env CUDA_VISIBLE_DEVICES="$EXPERIMENT_GPU_UUID" CUDA_DEVICE_ORDER=PCI_BUS_ID \
     PYTHONPATH="$EXPERIMENT_E2E_RUN/source" \
     GDN_CHUNK_SOURCE="$EXPERIMENT_E2E_RUN/source/chunk_overlay.py" \
     VLLM_SERVER_DEV_MODE=1 FLA_TRIL_PRECISION=ieee \
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     TRITON_CACHE_DIR="$EXPERIMENT_E2E_RUN/triton_cache" \
     TORCHINDUCTOR_CACHE_DIR="$EXPERIMENT_E2E_RUN/inductor_cache" \
     .venv/bin/inference serve --config "$EXPERIMENT_E2E_RUN/source/serve.yaml" \
     --gpus "$EXPERIMENT_GPU_UUID" > "$EXPERIMENT_E2E_RUN/server.log" 2>&1 &
   EXPERIMENT_SERVER_PID=$!
   ```

3. Wait for `Application startup complete` and verify `/health` succeeds. Startup
   includes kernel autotuning. Confirm the log records `enable_prefix_caching=False`
   and Triton/FLA GDN. Then run pilot, followed by confirmation. Failures invalidate
   a run; do not substitute missing pairs or merge partial attempts.

   ```bash
   tail -f "$EXPERIMENT_E2E_RUN/server.log"
   # Ctrl+C stops tail only; leave the background server running.
   curl -fsS http://127.0.0.1:8157/health
   .venv/bin/python -u "$EXPERIMENT_E2E_RUN/source/run.py" \
     --model-path "$EXPERIMENT_MODEL_DIR" \
     --dataset "$PWD/../benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json" \
     --gpu-uuid "$EXPERIMENT_GPU_UUID" --server-pid "$EXPERIMENT_SERVER_PID" \
     --phase pilot --seed 20261124 --output-dir "$EXPERIMENT_E2E_RUN/pilot"
   .venv/bin/python -u "$EXPERIMENT_E2E_RUN/source/run.py" \
     --model-path "$EXPERIMENT_MODEL_DIR" \
     --dataset "$PWD/../benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json" \
     --gpu-uuid "$EXPERIMENT_GPU_UUID" --server-pid "$EXPERIMENT_SERVER_PID" \
     --phase confirmation --seed 20261125 --output-dir "$EXPERIMENT_E2E_RUN/confirmation"
   ```

4. Analyze the fixed confirmation sample and preserve pilot results separately.
   Check numerical validation, matching greedy outputs, actual token counts,
   dispatch counters, and hardware metadata before interpreting p-values.

   ```bash
   .venv/bin/python "$EXPERIMENT_E2E_RUN/source/analyze.py" \
     "$EXPERIMENT_E2E_RUN/confirmation/results.json" \
     --output "$EXPERIMENT_E2E_RUN/confirmation/analysis.json"
   .venv/bin/python "$EXPERIMENT_E2E_RUN/source/check_outputs.py" \
     --results "$EXPERIMENT_E2E_RUN/confirmation/results.json" \
     --prompts "$EXPERIMENT_E2E_RUN/confirmation/prompts.json" \
     --output "$EXPERIMENT_E2E_RUN/output-control.json"
   ```

5. Stop only the experiment's own server and verify its GPU memory is released.

   ```bash
   kill -TERM "$EXPERIMENT_SERVER_PID"
   wait "$EXPERIMENT_SERVER_PID"
   nvidia-smi
   .venv/bin/python "$EXPERIMENT_E2E_RUN/source/summarize.py" \
     --run-dir "$EXPERIMENT_E2E_RUN" --output "$EXPERIMENT_E2E_RUN/summary.json"
   ```

## Results

Completed November 24, 2026 (Pacific time), run
`../artifacts/e2e-20261125T031821Z/`. See [summary.json](summary.json) for tracked
results and raw-artifact hashes. All **192 confirmation requests** succeeded in
**24 randomized paired blocks**, four requests per arm. Actual prompt lengths
were 8175–8176 tokens, with exactly 32 generated tokens. The confirmation prompt
set had no exact overlap with the pilot prompt set.

| HTTP endpoint metric | Baseline mean (ms) | Fused mean (ms) | Paired time reduction | Bootstrap 95% interval | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Full response latency, primary | 1276.705 | 1273.019 | 0.289% | 0.166–0.409% | 0.000280 |
| Time to first token, secondary | 1043.847 | 1040.456 | 0.325% | 0.217–0.444% | 0.000040 |

Time reductions use the geometric mean of the 24 paired baseline/fused ratios;
the two mean-latency columns are arithmetic means. Intervals use 40,000 paired
block bootstrap resamples. Permutation p-values use 100,000 seeded sign flips,
the plus-one correction, and Holm adjustment across both endpoints. All planned
pairs were collected before analysis; pilot measurements were excluded.
An independent supplementary exhaustive check of all 2^24 sign assignments gave
raw p-values 0.0001667 (completion latency) and 0.00000715 (TTFT), consistent with
the same significance conclusion. The preplanned Monte Carlo analysis remains
the analysis reported in the table.

The full-response effect is statistically detectable in this session but **small
in practice: about 3.7 ms on a 1.28-second request**. It does not support a claim
of a substantial serving improvement. Both randomized-order groups had positive
point estimates: 10 baseline-first pairs, 14 fused-first pairs. Other GPUs and
host CPUs were active; selected-GPU contention checks passed.

Both variants executed **2304 GDN prefill layer calls** during confirmation,
showing that the runtime switch affected real model execution rather than a
cached prefill graph. Real-activation parity checks passed on all 48 layer cases:
maximum output/state absolute errors 0.0078125 / 0.0122151, maximum relative L2
error 0.0035512. Tolerances were unchanged from the operator experiment.

**Quality limitation:** 84/96 paired greedy responses matched exactly; 12 differed.
A separate post-hoc repeatability control selected four differing and two matching
prompts. Each variant reproduced its own response on both repeats in all six
cases, and the four selected cross-variant differences persisted. Passing tensor
error tolerances therefore does not establish exact output equivalence or task
accuracy preservation. The control requests were excluded from latency inference.
Broad accuracy evaluation remains necessary before recommending this port.

The pilot also tested shorter inputs and longer outputs; its measurements remain
exploratory and are saved separately. The confirmation claim applies only to the
predeclared long-input, short-output, concurrency-1 setup above.

## Artifacts

`../artifacts/e2e-<UTC timestamp>/` contains source/config snapshots, server logs,
environment and model revision metadata, prompt text, per-request responses,
paired-block metrics, telemetry, validation errors, and analyses. Raw artifacts
remain ignored. Tracked summaries contain aggregate results and artifact hashes.

## Limitations

One quantized checkpoint, one GPU, one host, one server session, and constructed
long-input text workloads. Other GPUs and host CPUs are shared. No broad model
accuracy evaluation is performed. Greedy output agreement and activation parity
are narrower checks. Decode CUDA Graphs remain enabled, but this study does not
establish behavior under the default PyTorch compilation configuration, other
concurrency levels, multimodal requests, or deployment over a remote network.
Statistical significance alone does not establish practical importance.
