# Replication campaign

## Purpose

Replay existing serving and GDN profiling/kernel experiments as a bounded queue
on idle GPUs. Preserve new configs, commands, logs, traces, result JSONs, source
snapshots, and verification outcomes without overwriting historical captures.
The controller can run in tmux so a user can monitor or stop it after this chat.

## Configuration

Controller: [`scripts/replicate_queue.py`](../../scripts/replicate_queue.py).
Default pool: physical GPUs 0–3. GPU 4 is excluded. Each job claims only the
number of idle GPUs it needs, checks for other compute processes, and releases
them on completion/failure. Cooperative file locks coordinate our controllers;
they do not reserve hardware against other users. A detected foreign compute
process stops only the campaign's own process groups.

Default queue, in order:

| Job | GPUs | Work |
|---|---:|---|
| exp1 | 4 | Original chunked BF16 baseline; random and ShareGPT workloads |
| exp2 | 4 | INT8 + MTP; random and ShareGPT workloads |
| exp3 | 4 | Chunked BF16 with recorded prefix caching/context settings |
| exp4 | 4 | Educational sequential GDN serving workloads |
| gdn_sequential_initial | 2 | Initial sequential Torch capture recipe |
| gdn_sequential_torch_warm | 2 | Warmed sequential Torch capture recipe |
| gdn_comparison | 2 | Original chunked Torch capture recipe |
| gdn_fused_kkt_solve | 2 | Qwen3.5-4B-shaped smoke/full numerical and timing suites |
| gdn_comparison_proton_sequential | 1 | 2048-token Proton capture; 1 instrumented warmup, 524288-byte buffer |
| gdn_comparison_proton_chunked | 1 | 2048-token Proton capture; 1 instrumented warmup, 8192-byte buffer |

Serving performance jobs use two within-server repetitions by default; profile
jobs run once. Caches remain populated across within-server repetitions, so these
are not independent fresh-server trials. Server recipes and benchmark settings
come from each experiment's tracked YAMLs. The original chunk module is selected
by an import overlay, with the specified source commit and file hash saved.
No installed packages or sibling checkouts are modified by the controller.

The controller resolves models to complete cached snapshots, runs offline, and
records those choices. It does not silently download new revisions. Explicit
prefix-cache flags avoid the CLI's false-valued YAML omission. exp1 and exp4 explicitly
enable caching to reproduce their recorded servers rather than their inaccurate
historical "off" label. Each job records effective configs and adaptations.

Experiments requiring uncached models, another vLLM environment, or unavailable
`vllm-og`/`vllm-updated` source trees are not in this default queue. In particular,
this is not a rerun of every VisionArena sweep or the separate upstream-fusion
comparison. Failures are retained and the queue proceeds to the next entry.

## Replication instructions

1. From the repository root, inspect the plan and GPU availability. Use a fresh
   campaign output path and an unused loopback server port.

   ```bash
   cd /home/lukezeng/inference
   EXPERIMENT_CAMPAIGN_DIR="$PWD/experiments/replications/artifacts/$(date -u +%Y%m%dT%H%M%SZ)"
   mkdir -p "$(dirname "$EXPERIMENT_CAMPAIGN_DIR")"
   .venv/bin/python scripts/replicate_queue.py --plan-only \
     --output-dir "$EXPERIMENT_CAMPAIGN_DIR" --gpus 0 1 2 3
   nvidia-smi
   ss -ltn 'sport = :8160'
   ```

2. Start a detached tmux session after the current PD replication has finished.
   The queue waits up to eight hours for sufficient idle GPUs before failing a
   job; it never terminates another user's process to make space.

   ```bash
   tmux new-session -d -s inference-verify -c "$PWD" \
     ".venv/bin/python -u scripts/replicate_queue.py --gpus 0 1 2 3 --output-dir '$EXPERIMENT_CAMPAIGN_DIR' --port 8160 --repeats 2 > '$EXPERIMENT_CAMPAIGN_DIR.controller.log' 2>&1"
   ```

3. Attach to the session, inspect the controller log and atomic status file, and
   open the current job's `server.log` or benchmark log. `status.json` includes
   host, controller PID, per-job state, command arrays, log paths, and active PIDs.

   ```bash
   tmux attach -t inference-verify
   # Ctrl+B, then D detaches without stopping the queue.
   cat "$EXPERIMENT_CAMPAIGN_DIR/status.json"
   .venv/bin/python scripts/replicate_queue.py --status "$EXPERIMENT_CAMPAIGN_DIR/status.json"
   tail -f "$EXPERIMENT_CAMPAIGN_DIR.controller.log"
   ```

4. To stop gracefully after the current job, create `STOP`. To interrupt the
   current job and clean up its server/worker process groups, create `STOP_NOW`
   or send SIGTERM to the controller PID recorded in the status file on the
   recorded host. GPU ownership is released; idle holding jobs are not created.

   ```bash
   touch "$EXPERIMENT_CAMPAIGN_DIR/STOP"
   # Alternatively, interrupt current work:
   touch "$EXPERIMENT_CAMPAIGN_DIR/STOP_NOW"
   ```

5. Inspect each job's `verification.json` or `failure.json`, raw benchmark success
   counts, traces, and numerical checks. Rerun selected failed entries into a new
   output directory with `--only`, retaining the original failed attempt.

   ```bash
   .venv/bin/python scripts/replicate_queue.py --gpus 0 1 2 3 \
     --only exp2 gdn_comparison \
     --output-dir "$PWD/experiments/replications/artifacts/retry-$(date -u +%Y%m%dT%H%M%SZ)"
   ```

## Results

Runtime status and verification outcomes are stored per campaign. A passed job
means the configured requests completed or the numerical/profiling harness
succeeded; it does not establish historical timing equality, a speedup, or
statistical significance. Read raw outputs and recorded adaptations before
comparing historical and new results.

## Artifacts

Ignored `artifacts/<UTC timestamp>/` contains `status.json`, `plan.json`, software
and dataset manifests, the exact controller source, and one directory per job.
Serving directories contain original/effective YAMLs, selected kernel source and
hash, command/log records, raw benchmark JSON, and profiler traces when requested.
Kernel harness directories retain their own numerical checks and raw samples.
The adjacent `.controller.log` captures queue-level progress.

## Limitations

Exact historical software and model/dataset revisions were not always saved.
Current vLLM behavior and resolved cached revisions may differ. Initial-versus-
warmed captures cannot reconstruct an unknown historical JIT/cache state. The
Proton jobs deliberately warm once and should not be described as cold captures.
Host and GPU clocks are shared and unlocked; contention monitoring itself can
add overhead. Historical result files are never used as a pass/fail timing target.
