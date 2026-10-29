# gdn_comparison_warm

## Purpose

Capture original chunked GDN after instrumented warmup with the original large profiling buffers.

## Configuration

| Setting | Recorded value |
|---|---|
| Implementation | Original chunked GDN retained in the educational source snapshot |
| Input | Synthetic B=1, T=2048, Hqk=8, Hv=24, K=V=128; seed 1337 |
| Input types | BF16 normalized q/k and BF16 values; FP32 log gates/beta and zero initial state |
| Instrumented warmups | 1, recorded in a separate session |
| Buffer bytes per CTA | 524288 |
| Sampling | CTA (0,0,0), warp 0 |
| Hardware / Triton | RTX 3090 / 3.7.1 |

## Replication instructions

1. Read the [replication prerequisites](../README.md#replication-prerequisites), then create a new output directory. Run all steps in the same Bash terminal.

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   EXPERIMENT_RUN_DIR="$PWD/experiments/gdn_comparison_warm/artifacts/replications/$(date -u +%Y%m%dT%H%M%SZ)"
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

3. Capture chunked Proton scopes with 1 instrumented warmup(s) and 524288 bytes per CTA.

   ```bash
   PYTHONPATH="$PWD/experiments/_shared/gdn/import_overlay" \
   GDN_CHUNK_SOURCE="$EXPERIMENT_KERNEL_SOURCE" \
   CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/_shared/gdn/profile_intra.py \
     --implementation chunked --tokens 2048 \
     --instrumented-warmups 1 --instrumentation-buffer-bytes 524288 \
     --output-dir "$EXPERIMENT_RUN_DIR" \
     > "$EXPERIMENT_RUN_DIR/proton-chunked.log" 2>&1
   ```

4. Open `proton_chunked/intra.chrome_trace` in Perfetto. Keep `warmup.chrome_trace` separate; inspect all six `kernel_body` scopes.

## Results

Uninstrumented baseline: 0.480729 ms. Instrumented output and state matched their baseline exactly. The measured trace contains 468 internal events across six kernels, with no reported buffer overflow. This intermediate capture warms instrumented binaries but retains large per-launch collection buffers; prefer the smaller-buffer follow-up for inspection.

## Artifacts

Historical files: `artifacts/proton_chunked/intra.chrome_trace`, `warmup.chrome_trace`, `summary.json`, `run.log`, and `ir/`.

New replications: `artifacts/replications/<UTC timestamp>/proton_chunked/`, with the environment and source snapshot in its parent directory.

## Limitations

The physical GPU for the historical warm capture was not recorded in its summary; GPU 1 in the command is a suggested available device, not a recovered historical identifier. Choose an idle RTX 3090. Match the explicit warmup count and buffer size: the chunked harness now defaults to the smaller buffer. Dependency commits and cache state were not fully captured. Sampling/clock/instrumentation limitations are the same as [gdn_comparison](../gdn_comparison/README.md#limitations). These are instrumented traces, not normal end-to-end serving timelines.
