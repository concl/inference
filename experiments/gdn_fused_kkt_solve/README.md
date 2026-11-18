# gdn_fused_kkt_solve

## Purpose

Test whether porting upstream FLA's fused KKT/triangular-solve kernel improves
vLLM's existing chunked Gated Delta Rule on an RTX 3090. Replace only these two
stages, keep all other vLLM kernels, and validate numerical behavior before timing.
The Qwen3.5-4B study uses the model's GDN geometry on two independent GPUs;
it does not load model weights or execute a distributed model.

## Configuration

- Existing `/home/lukezeng/inference/.venv`; no package installs or serving changes.
- Installed vLLM 0.29.0, PyTorch 2.13.0, Triton 3.7.1 at preparation time.
- Baseline: explicitly call `chunk_gated_delta_rule_fwd`, bypassing the installed
  fork's educational sequential wrapper. Snapshots and SHA256s in `provenance.json`.
- Upstream FLA source pinned at `75fdd25e670afe27859eb586d2e0c7caf3a59234`.
- `candidate.py` preserves the upstream kernel body and autotune configurations.
  Integration binds its `exp2` symbol to vLLM's natural exponential and matches
  vLLM's default IEEE triangular-solve precision. The FLA cache autotuner is
  replaced by ordinary Triton autotuning. This is a contract-matched port, not a
  comparison with an unmodified complete upstream FLA package.
- Only the triangular-solve precision is explicitly matched. Upstream forms KKT
  with BF16 operands and applies beta afterward; vLLM scales beta before a
  promoted-precision dot product. Any measured gain belongs to the combined
  upstream stage replacement, not necessarily launch fusion alone.
- `candidate_full.py` is generated from the baseline forward function, replacing
  only the two A assignments; inspect `adapter.diff`.
- BF16 q/k/v, normalized q/k, FP32 gates/beta/state, K=V=128, seed 1337.
- Qwen3.5-4B config revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, copied
  unchanged into `qwen35-4b-config.json`: 16 key heads, 32 value heads, key/value
  dimensions 128. Per-rank TP=1/2 shapes are 16/32 and 8/16. Two RTX 3090s each
  run both geometries as independent replicas, not tensor-parallel workers.
  Config contents and SHA256 are retained in each result.
- Without `--model-config`, the legacy Qwen3.8-27B geometry remains available:
  16 key / 48 value heads, with per-rank TP=4/2/1 shapes 4/12, 8/24, 16/48.
- Qwen3.5-4B full suite: 22 independent-reference cases and 12 performance shapes
  per GPU (legacy default: 23 cases and 14 shapes), covering
  16/32/64-token boundaries, partial chunks, ragged inputs, no/zero/nonzero initial
  states, and stronger negative gates. Describe exact cases with `--describe`.
- Correctness uses an independent CPU FP64 token recurrence and an independent
  FP64 triangular solve. Both old and new implementations must have max absolute
  error <=0.02 and relative L2 error <=0.01; differences are recorded separately.
- Seven paired rounds in seeded randomized order, 20 calls per sample, ten
  warmups. Compilation/autotuning is completed before timing. Both eager and
  isolated CUDA Graph modes are measured for the replaced region and full GDN.
- GPU clocks are not changed, other users' processes are not stopped, and each
  measurement round rejects any other compute process on the selected GPU.
- No Proton/in-kernel instrumentation is enabled. CUDA events measure timing.

## Replication instructions

1. From the inference repository, select this experiment and inspect the changes.

   ```bash
   cd /home/lukezeng/inference
   EXPERIMENT_DIR="$PWD/experiments/gdn_fused_kkt_solve"
   cat "$EXPERIMENT_DIR/adapter.diff"
   .venv/bin/python "$EXPERIMENT_DIR/benchmark.py" \
     --model-config "$EXPERIMENT_DIR/qwen35-4b-config.json" --tp-sizes 1 2 --describe
   ```

2. Recreate the adapters from the already snapshotted sources if needed, then
   check syntax. No dependencies are installed by this step.

   ```bash
   .venv/bin/python "$EXPERIMENT_DIR/prepare.py"
   .venv/bin/python -m py_compile "$EXPERIMENT_DIR/benchmark.py" \
     "$EXPERIMENT_DIR/candidate.py" "$EXPERIMENT_DIR/candidate_full.py" \
     "$EXPERIMENT_DIR/run_pair.py" "$EXPERIMENT_DIR/analyze_pair.py"
   ```

3. Select two idle RTX 3090s and run smoke then full suites. The launcher rejects
   GPUs with compute processes, >5% utilization, or >=256 MiB in use. Each GPU
   receives a separate process and compiler caches; both run the same suites.
   A lock coordinates with the legacy watcher. Source and model config snapshots
   are saved before execution. Compilation/autotuning is outside timed regions.

   ```bash
   EXPERIMENT_RUN_DIR="$EXPERIMENT_DIR/artifacts/runs/qwen35-4b-$(date -u +%Y%m%dT%H%M%SZ)"
   nvidia-smi
   .venv/bin/python -u "$EXPERIMENT_DIR/run_pair.py" \
     --model-config "$EXPERIMENT_DIR/qwen35-4b-config.json" \
     --gpus 0 1 --tp-sizes 1 2 --output-dir "$EXPERIMENT_RUN_DIR"
   ```

4. Inspect status, logs, numerical errors, raw timings, environment, and chosen
   autotune configurations. Each run uses a fresh directory; no server is started.

   ```bash
   cat "$EXPERIMENT_RUN_DIR/status.json"
   cat "$EXPERIMENT_RUN_DIR/gpu0/full.log"
   cat "$EXPERIMENT_RUN_DIR/gpu1/full.log"
   .venv/bin/python -m json.tool "$EXPERIMENT_RUN_DIR/gpu0/full/results.json"
   .venv/bin/python "$EXPERIMENT_DIR/analyze_pair.py" \
     --run-dir "$EXPERIMENT_RUN_DIR" --model-id Qwen/Qwen3.5-4B \
     --output "$EXPERIMENT_RUN_DIR/curated-summary.json"
   ```

5. The launcher and benchmarks exit automatically. Ctrl+C stops the launcher's
   own child processes. Never terminate another user's GPU job to make space.

The original single-GPU `wait_and_run.py --output-dir <fresh-directory>` remains
available for the legacy 27B shapes. It waits for 60 continuous idle seconds.

## Results

Completed on November 17, 2026 (Pacific time; artifact timestamps are UTC), on
physical RTX 3090 GPUs 0 and 1. Run:
`artifacts/runs/qwen35-4b-20261118T045102Z/`. Both GPUs passed smoke and full
suites: **22/22 independent-reference cases and 12/12 performance-shape parity
checks per GPU**. No other compute process was detected on either selected GPU.
There was an active job on GPU 4; the host was shared.

Curated results, per-GPU ratios, paired rounds, clock ranges, source provenance,
and raw-result hashes are in [summary-qwen35-4b.json](summary-qwen35-4b.json).
The recorded model config SHA256 is
`ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

Selected **full GDN operator CUDA Graph** timings, not model serving speedups:

| Per-rank TP geometry | Sequence lengths | GPU 0 baseline → fused (us) | GPU 0 speedup | GPU 1 baseline → fused (us) | GPU 1 speedup |
|---|---|---:|---:|---:|---:|
| TP=1, 16/32 heads | 64 | 53.811 → 56.064 | 0.960x | 50.176 → 52.992 | 0.947x |
| TP=1, 16/32 heads | 65 | 56.934 → 59.846 | 0.951x | 53.043 → 55.859 | 0.950x |
| TP=1, 16/32 heads | 256 | 84.634 → 83.610 | 1.012x | 80.998 → 79.514 | 1.019x |
| TP=1, 16/32 heads | 2048 | 513.331 → 462.438 | 1.110x | 498.227 → 449.075 | 1.109x |
| TP=1, 16/32 heads | 4 × 256 | 294.395 → 263.885 | 1.116x | 283.034 → 255.533 | 1.108x |
| TP=2, 8/16 heads | 2048 | 302.285 → 271.970 | 1.111x | 270.848 → 262.963 | 1.030x |
| TP=1, 16/32 heads | packed: 17, 63, 129, 2048 | 555.520 → 540.109 | 1.029x | 594.995 → 543.590 | 1.095x |

Values are ratios of median durations from seven randomized paired rounds,
20 calls/sample. Below 1x is a regression. At TP=1 and 2048 tokens, the replaced
KKT/solve region improves by **1.450x / 1.469x** on GPUs 0/1; the full operator
improves by **1.110x / 1.109x**, about 10% less operator time. The 64–65-token
full-operator cases regress by roughly 4–6%; every paired graph round is below
1x on both GPUs. At 256 tokens, paired graph rounds straddle 1x, so the small
median improvement is not a clear win. Benefits are shape-dependent.

Long full-operator cases (single 8192 tokens, 4 × 2048, TP=2 single 8192, and
the 7424-token packed batch) have **inconclusive timing**. For example, TP=1
single-8192 graph ratios span 0.861–2.014x on GPU 0 despite a 1.774x ratio of
medians. Its telemetry records SM clocks from 510 to 1725 MHz. Clock swings and
large timing variation were observed; their exact cause was not established.
Do not use the large median gains or regressions for these cases as reliable
performance claims. All samples are retained. Some TP=2 and packed-case gains
also differ materially between GPUs; autotune selections and per-GPU results
are retained rather than pooling their absolute timings.

Eager and graph modes are both saved. Eager short-shape times include substantial
host submission overhead and should not be described as intrinsic kernel speedups.
CUDA Graph mode is an isolated operator experiment; it does not establish that
vLLM captures this prefill path or that model throughput improves by these ratios.

Across the reference cases on both GPUs, the candidate's maximum output/state
absolute errors against FP64 were 0.000746 / 0.008128; maximum relative L2 errors
were 0.003887 / 0.003164. These pass the pre-existing 0.02 / 0.01 gates without
changing tolerances. Long performance shapes separately passed baseline/candidate
output and final-state agreement checks.

The earlier legacy-geometry run at `artifacts/runs/20261104T232742Z/` was aborted
when another compute process entered its GPU. It is not a completed performance
result; its logs and partial checks are preserved.

Preparation passed: Python syntax checks, an independent closed-form recurrence
check, the one-token inverse check, and AST identity of the extracted upstream
kernel. All 12 upstream autotune configurations (BK=32/64, 1/2/4 warps, fixed and
ragged sequence paths) compiled for SM86 without creating a GPU context. Codegen
metadata and PTX using the upstream autotune defaults are in
`artifacts/offline_compile_upstream_defaults/`; this is not runtime validation.

## Artifacts

- `provenance.json`: source commit, paths, SHA256s, and adapter details.
- `qwen35-4b-config.json`: exact cached model config used for the 4B geometry.
- `run_pair.py`: independent two-GPU smoke/full launcher with source snapshots.
- `analyze_pair.py`: validates completion and curates per-GPU results.
- `summary-qwen35-4b.json`: tracked results for the completed two-GPU 4B study.
- `UPSTREAM_LICENSE`: upstream MIT license for the extracted kernel.
- `artifacts/sources/`: exact original upstream and installed baseline snapshots.
- `artifacts/runs/qwen35-4b-<timestamp>/`: launcher status and source/config
  snapshots; each `gpu<index>/` contains smoke/full logs, result JSON, readable
  summaries, and isolated compiler caches.
  Raw artifacts are gitignored by repository policy.
- Installed packages, the sibling vLLM checkout, existing experiments, and the
  user's existing `pyproject.toml` modification are preserved.

## Limitations

This experiment assesses one port on Ampere, using synthetic BF16 inputs. It
does not measure model accuracy, end-to-end serving throughput, inter-GPU
communication, or other hardware. The reference tests cover short sequences;
long performance cases check output/state agreement between implementations.
FP64 reference tolerance is fixed before GPU tests. It is an experiment gate,
not a claim of equivalence for every supported input.

CUDA Graph timings isolate a captured operator and do not establish that vLLM
captures this prefill path. Eager event measurements can include host-submission
gaps. No GPU clock locking or cache flushing is used; telemetry and all paired
rounds are retained. Availability checks cannot reserve hardware against a job
that starts later; detected compute contention stops the run.
The two independent GPUs share a host, so concurrent compilation and CPU
activity may affect eager submission timing. Model geometry comes from the
config; synthetic inputs are not activations captured from Qwen3.5-4B.
