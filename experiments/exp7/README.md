# exp7

## Purpose

Measure output throughput, p95 time to first token (TTFT), and p95 inter-token latency (ITL) while varying client concurrency, batched-token budget, and maximum active sequences. This is a multi-parameter serving sweep, not an isolated kernel comparison or model-quality evaluation. The selected historical run is `20261119T070255-154489Z`, copied from `experiments/exp5/artifacts/20261119T070255-154489Z` into exp7 without rerunning the benchmark.

## Configuration

| Setting | Recorded value |
|---|---|
| Server / benchmark configs | [config.yaml](config.yaml), [benchmark-vision-arena.yaml](benchmark-vision-arena.yaml). Historical effective configs are linked below; current files may differ. |
| Source commit / kernel selection | Inference commit: eb67e56f4af5e59487b1ba5df185eeb657b7d57a. Installed vLLM; no kernel override applied by this runner. Exact vLLM source commit unknown. Saved driver: [run.py](artifacts/20261119T070255-154489Z/run.py) |
| Software versions | [environment.json](artifacts/20261119T070255-154489Z/environment.json); vllm=0.33.0, torch=2.14.0, triton=3.8.0, transformers=5.17.0, optuna=5.0.0, plotly=7.1.0, PyYAML=6.0.3, datasets=5.2.0, huggingface-hub=1.32.0, Pillow=13.0.0 |
| Hardware / GPU IDs / TP | GPU IDs=0,1,2; TP=1, PP=3, DP=1. Inventory/UUIDs: [environment.json](artifacts/20261119T070255-154489Z/environment.json) |
| Model/tokenizer revisions | Model=Qwen/Qwen3.8-27B; model revision=unpinned; resolved revision unknown; tokenizer revision=unpinned; resolved revision unknown. Resolved revisions not independently verified. |
| Workload / dataset hash | lmarena-ai/vision-arena-bench-v0.1, split=train; text + image; measured prompts=100; seed=42. Dataset hash unknown. Lengths and effective options: [benchmark-vision-arena.yaml](artifacts/20261119T070255-154489Z/benchmark-vision-arena.yaml) |
| Warmup / cache policy | 5 warmup requests; runner uses measurement seed + 1 for warmup; prefix caching retained between warmup and measurement, fresh server per trial, disk compilation caches retained. Exact commands saved per trial. |
| Profiler / instrumentation settings | No explicit profiler enabled by the runner; profiler buffer sizes not applicable. External instrumentation unknown. |
| Server settings / search space | [config.yaml](artifacts/20261119T070255-154489Z/config.yaml); {"concurrency": [1, 2, 4, 8, 16, 32], "max_batched_tokens": [1024, 2048, 4096, 16384], "max_sequences": [4, 8, 16, 32, 64]} |

## Replication instructions

1. From the repository root on a host with the recorded GPU layout, use the existing environment. Compare versions and available GPU inventory with the historical manifest. Ensure the selected GPUs and port are available; these commands do not reserve hardware. This Vision Arena runner uses installed vLLM and Hugging Face data, without the older GDN overlay or local ShareGPT prerequisites.

   ```bash
   cd /home/snorlax/inference
   .venv/bin/python -c 'import optuna, yaml, plotly; print("Reporting/runner imports OK")'
   nvidia-smi
   ss -ltnp 'sport = :8000'
   ```

2. Verify the current runner, configs, and launchers match the historical snapshots before replaying them. Stop and reconcile any differences. Compare the installed package versions against `environment.json`; exact model/dataset revisions remain unavailable.

   ```bash
   cmp experiments/exp7/run.py experiments/exp7/artifacts/20261119T070255-154489Z/exports/20261120T025100-976945Z/replication-run.py || exit 1
   cmp experiments/exp7/config.yaml experiments/exp7/artifacts/20261119T070255-154489Z/source-config.yaml || exit 1
   cmp experiments/exp7/benchmark-vision-arena.yaml experiments/exp7/artifacts/20261119T070255-154489Z/source-benchmark-vision-arena.yaml || exit 1
   cmp scripts/serve.py experiments/exp7/artifacts/20261119T070255-154489Z/source/serve.py || exit 1
   cmp scripts/bench.py experiments/exp7/artifacts/20261119T070255-154489Z/source/bench.py || exit 1
   cmp scripts/cli.py experiments/exp7/artifacts/20261119T070255-154489Z/source/cli.py || exit 1
   ```

3. Validate the launch arguments. This produces a fresh capture directory and does not start a model server.

   ```bash
   .venv/bin/python experiments/exp7/run.py --gpus 0,1,2 --parallelism pipeline --port 8000 --trials 120 --num-prompts 100 --warmup-prompts 5 --dry-run
   ```

4. Run the sweep into a new timestamped directory. The runner snapshots inputs and environment, starts each server, polls `/health`, performs the configured warmup, and measures the workload. The recorded measurement seed is 42; the archived runner uses the next seed for warmup. It preserves caches as described above. These are performance measurements; no profiler capture is requested.

   ```bash
   .venv/bin/python experiments/exp7/run.py --gpus 0,1,2 --parallelism pipeline --port 8000 --trials 120 --num-prompts 100 --warmup-prompts 5
   ```

   Exact historical argv for an included trial: [server](artifacts/20261119T070255-154489Z/trial-0000/server-command.txt), [warmup](artifacts/20261119T070255-154489Z/trial-0000/warmup-command.txt), [measurement](artifacts/20261119T070255-154489Z/trial-0000/result-command.txt). Historical output paths are evidence, not destinations to reuse.

5. The runner stops its server process group after each trial. Record the newly printed output directory, inspect its success/failure counts, and check each included result completed 100 requests with no reported failures. Check the server port after the runner exits. A new sweep does not resume the historical study.

   ```bash
   ss -ltnp 'sport = :8000'
   ```

## Results

Status: **partial**.  Recorded trial states: 50 complete (server died halfway through). 

All included trials completed 100 requests, reported no failures, and have finite positive objective metrics. The 17 Pareto points are relative only to the included trials. JSON retains full precision; the table rounds to three decimals.

All fields from each included raw benchmark JSON are preserved in [summary.json](summary.json), using the same `experiment` / `benchmarks` structure as exp1. Benchmark keys identify trial number, concurrency (`c`), batched-token budget (`bt`), and maximum sequences (`seq`). No results from different configurations are averaged together.

## Results


| Concurrency | Batched tokens | Max sequences | Output tokens/s | p95 TTFT (ms) | p95 ITL (ms) | Pareto | Source |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1024 | 4 | 14.371 | 901.722 | 72.346 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0017/result.json) |
| 1 | 1024 | 64 | 14.281 | 978.500 | 72.953 | | [JSON](artifacts/20261119T070255-154489Z/trial-0002/result.json) |
| 1 | 2048 | 4 | 13.935 | 1149.349 | 83.330 | | [JSON](artifacts/20261119T070255-154489Z/trial-0046/result.json) |
| 1 | 2048 | 16 | 13.806 | 1214.053 | 84.645 | | [JSON](artifacts/20261119T070255-154489Z/trial-0056/result.json) |
| 1 | 2048 | 32 | 14.332 | 894.742 | 71.639 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0065/result.json) |
| 1 | 2048 | 64 | 14.087 | 1118.539 | 79.549 | | [JSON](artifacts/20261119T070255-154489Z/trial-0039/result.json) |
| 1 | 4096 | 4 | 14.354 | 987.484 | 71.050 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0007/result.json) |
| 1 | 4096 | 8 | 14.298 | 974.108 | 71.490 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0012/result.json) |
| 1 | 4096 | 16 | 14.124 | 998.720 | 77.532 | | [JSON](artifacts/20261119T070255-154489Z/trial-0025/result.json) |
| 2 | 1024 | 16 | 25.572 | 3225.449 | 76.662 | | [JSON](artifacts/20261119T070255-154489Z/trial-0032/result.json) |
| 2 | 1024 | 64 | 25.952 | 2688.361 | 72.869 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0030/result.json) |
| 2 | 2048 | 4 | 24.991 | 2824.706 | 81.033 | | [JSON](artifacts/20261119T070255-154489Z/trial-0041/result.json) |
| 2 | 2048 | 8 | 26.331 | 2755.029 | 72.070 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0004/result.json) |
| 2 | 2048 | 16 | 25.048 | 3054.425 | 82.571 | | [JSON](artifacts/20261119T070255-154489Z/trial-0061/result.json) |
| 2 | 2048 | 32 | 25.029 | 3078.497 | 80.586 | | [JSON](artifacts/20261119T070255-154489Z/trial-0043/result.json) |
| 2 | 4096 | 4 | 25.294 | 3642.715 | 79.204 | | [JSON](artifacts/20261119T070255-154489Z/trial-0036/result.json) |
| 2 | 4096 | 8 | 25.859 | 3649.710 | 72.065 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0022/result.json) |
| 2 | 4096 | 32 | 25.154 | 3706.801 | 80.447 | | [JSON](artifacts/20261119T070255-154489Z/trial-0040/result.json) |
| 2 | 4096 | 64 | 23.826 | 4282.444 | 93.655 | | [JSON](artifacts/20261119T070255-154489Z/trial-0054/result.json) |
| 4 | 1024 | 4 | 43.364 | 2438.990 | 101.890 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0008/result.json) |
| 4 | 1024 | 16 | 39.139 | 3051.150 | 115.673 | | [JSON](artifacts/20261119T070255-154489Z/trial-0024/result.json) |
| 4 | 1024 | 32 | 33.480 | 3542.106 | 137.811 | | [JSON](artifacts/20261119T070255-154489Z/trial-0057/result.json) |
| 4 | 1024 | 64 | 42.049 | 2633.599 | 97.585 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0000/result.json) |
| 4 | 2048 | 4 | 41.542 | 3345.570 | 100.577 | | [JSON](artifacts/20261119T070255-154489Z/trial-0021/result.json) |
| 4 | 2048 | 16 | 41.191 | 3339.424 | 99.816 | | [JSON](artifacts/20261119T070255-154489Z/trial-0001/result.json) |
| 4 | 2048 | 64 | 35.357 | 2437.520 | 133.474 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0050/result.json) |
| 4 | 4096 | 32 | 38.301 | 2578.619 | 117.954 | | [JSON](artifacts/20261119T070255-154489Z/trial-0037/result.json) |
| 8 | 1024 | 16 | 67.501 | 4217.228 | 98.323 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0011/result.json) |
| 8 | 1024 | 64 | 67.995 | 3803.122 | 100.142 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0005/result.json) |
| 8 | 2048 | 4 | 43.590 | 16253.868 | 100.056 | | [JSON](artifacts/20261119T070255-154489Z/trial-0020/result.json) |
| 8 | 2048 | 8 | 58.703 | 4666.296 | 126.597 | | [JSON](artifacts/20261119T070255-154489Z/trial-0053/result.json) |
| 8 | 2048 | 16 | 55.865 | 5818.674 | 123.682 | | [JSON](artifacts/20261119T070255-154489Z/trial-0059/result.json) |
| 8 | 2048 | 32 | 66.352 | 4725.488 | 115.378 | | [JSON](artifacts/20261119T070255-154489Z/trial-0048/result.json) |
| 8 | 2048 | 64 | 63.220 | 3967.273 | 115.054 | | [JSON](artifacts/20261119T070255-154489Z/trial-0042/result.json) |
| 8 | 4096 | 4 | 42.327 | 17388.847 | 111.896 | | [JSON](artifacts/20261119T070255-154489Z/trial-0023/result.json) |
| 8 | 4096 | 16 | 53.579 | 6142.328 | 128.196 | | [JSON](artifacts/20261119T070255-154489Z/trial-0058/result.json) |
| 8 | 4096 | 32 | 67.457 | 4944.091 | 89.983 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0006/result.json) |
| 16 | 1024 | 32 | 100.538 | 5991.554 | 113.772 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0060/result.json) |
| 16 | 2048 | 4 | 40.320 | 48783.910 | 127.787 | | [JSON](artifacts/20261119T070255-154489Z/trial-0045/result.json) |
| 16 | 2048 | 32 | 109.198 | 6851.280 | 94.691 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0018/result.json) |
| 16 | 2048 | 64 | 105.523 | 7413.320 | 98.347 | | [JSON](artifacts/20261119T070255-154489Z/trial-0014/result.json) |
| 16 | 4096 | 4 | 37.542 | 56139.248 | 135.027 | | [JSON](artifacts/20261119T070255-154489Z/trial-0047/result.json) |
| 16 | 4096 | 8 | 70.520 | 21105.621 | 90.483 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0015/result.json) |
| 32 | 1024 | 4 | 42.939 | 91905.445 | 104.200 | | [JSON](artifacts/20261119T070255-154489Z/trial-0064/result.json) |
| 32 | 1024 | 64 | 81.433 | 55129.946 | 795.020 | | [JSON](artifacts/20261119T070255-154489Z/trial-0019/result.json) |
| 32 | 2048 | 16 | 110.973 | 23643.851 | 93.260 | yes | [JSON](artifacts/20261119T070255-154489Z/trial-0009/result.json) |
| 32 | 2048 | 64 | 89.640 | 52335.886 | 697.323 | | [JSON](artifacts/20261119T070255-154489Z/trial-0016/result.json) |
| 32 | 4096 | 4 | 36.725 | 111557.567 | 118.142 | | [JSON](artifacts/20261119T070255-154489Z/trial-0035/result.json) |
| 32 | 4096 | 32 | 102.537 | 31579.746 | 380.651 | | [JSON](artifacts/20261119T070255-154489Z/trial-0055/result.json) |
| 32 | 4096 | 64 | 68.734 | 78875.534 | 775.881 | | [JSON](artifacts/20261119T070255-154489Z/trial-0027/result.json) |

## Artifacts

- Copies of the historical capture: `artifacts/20261119T070255-154489Z/`, including per-trial params, logs, raw result JSON, command files, `study.db`, configs, launcher/driver snapshots, and environment manifest. These remain under gitignored `artifacts/`. The exp5 originals remain unchanged.
- [capture-origin.json](artifacts/20261119T070255-154489Z/capture-origin.json) records the copy's origin. Absolute paths in historical command files and manifests still identify the original exp5 run; no historical evidence is relabeled as a new measurement. The top-level `run.py` adjusts only the experiment label so future summaries identify exp7; [replication-run.py](artifacts/20261119T070255-154489Z/exports/20261120T025100-976945Z/replication-run.py) records that runnable version for comparison.
- Tracked curated files: this [README.md](README.md), [summary.json](summary.json), [config.yaml](config.yaml), [benchmark-vision-arena.yaml](benchmark-vision-arena.yaml), and the sweep driver [run.py](run.py).
- Newly generated audit reports: [run-summary.json](artifacts/20261119T070255-154489Z/exports/20261120T025100-976945Z/run-summary.json), [trials.csv](artifacts/20261119T070255-154489Z/exports/20261120T025100-976945Z/trials.csv), [pareto.json](artifacts/20261119T070255-154489Z/exports/20261120T025100-976945Z/pareto.json), and [hashes.json](artifacts/20261119T070255-154489Z/exports/20261120T025100-976945Z/hashes.json). These are derived from the historical captures, not newly measured results.
- `hashes.json` records hashes of the source files consumed by this export. It does not hash the remote dataset or claim an atomic snapshot of a running study.
- Prior README/summary versions, if present, are backed up under `artifacts/20261119T070255-154489Z/exports/20261120T025100-976945Z/previous/` before replacement.
- Future replication output belongs in a fresh timestamped artifact directory; never overwrite the original capture.

## Limitations

This interrupted/partial sweep cannot establish performance for missing or excluded configurations. Failed and RUNNING trials are excluded from the performance frontier. File timestamps and database states alone cannot establish why a remote host became unreachable or whether another user terminated a process.

Resolved model, tokenizer, dataset revisions and dataset-content hashes were not captured reliably. Installed package versions do not identify an exact vLLM source commit. Source/config identity checks above do not recreate uncaptured dependencies. Review any dirty-worktree details in the saved environment manifest. Do not infer historical versions from current defaults.

The runner sets dataset sampling seeds, but generation defaults and scheduling may still vary; inspect saved benchmark command/config/logs for temperature and token limits. Warmup and endpoint-test requests can populate prefix and multimodal caches. Different seeds do not guarantee disjoint prefixes. Disk compilation caches are retained. Input image sizes and prompt lengths vary; requested output length is not necessarily achieved output length.

Shared hardware/host load may affect results and was not continuously captured. GPU IDs, TP/PP layout and model settings must match when comparing runs; three-GPU PP and four-GPU TP runs are different configurations. p95 estimates from 100 requests are noisy; there are no repeated-run confidence intervals. No explicit profiler was enabled; external instrumentation overhead is unknown. Inspect logs for configuration discrepancies before curating these results.
