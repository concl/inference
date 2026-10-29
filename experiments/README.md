# Experiments

Keep each experiment self-contained:

```text
experiments/<name>/
  README.md             # purpose, commands, interpretation
  summary.json          # optional curated results and artifact hashes
  artifacts/            # gitignored raw data
    torch/              # or torch_chunked/
    proton_sequential/
    proton_chunked/
```

Traces, generated IR, profiler summaries, logs, raw benchmark JSON, and historical
config/source snapshots belong under `artifacts/`. Each experiment owns its
tracked launch YAML, code specific to that experiment, and curated summaries.
`configs/` contains only the reusable default serving configuration.
Choose a new experiment name or capture subdirectory to preserve earlier runs.

Shared GDN utilities live in [`_shared/gdn/`](_shared/gdn/README.md). Keep one copy
there and pass explicit experiment parameters and output paths. Experiments must
not import utility code from a sibling experiment. General project commands
(`serve`, `bench`, and the OpenAI client) belong in `scripts/`.

## README format

Every experiment README uses these exact level-two headings, in order:

1. `Purpose`
2. `Configuration`
3. `Replication instructions`
4. `Results`
5. `Artifacts`
6. `Limitations`

Use [TEMPLATE.md](TEMPLATE.md) for a new experiment. Replication instructions
must be numbered, executable steps covering setup, the exact kernel selection,
server startup/readiness when needed, workload, warmup, capture, output paths,
and server shutdown. Set seeds and profiler buffer sizes explicitly. Put results
in a fresh timestamped directory so replication cannot overwrite prior captures.
Record unknown historical settings under `Limitations`; do not infer an exact
dependency or dataset revision from a current default.

## Replication prerequisites

Run each README's steps from inside the repository in one Bash terminal. The
first command selects the repository root. The recipes assume a sibling fork
checkout at `../vllm` and a sibling dataset directory at `../benchmarks`.
Use the existing `.venv`; the recipes do not
install packages, switch branches, or reset shared caches. See
[the vLLM setup notes](../notes/WORKING_ON_VLLM.md) if the environment is absent.

The fork must contain the two kernel commits referenced by the recipes:
`58d8a013b2dd33adb99f75a1c7d12ce1babf8640` (original chunked module) and
`db0084219d62a7b7b564adeedb9e6536cadcf0d8` (educational sequential module,
which retains the original chunked function). The import overlay selects only
that module; it does not pin all other vLLM files.

For server experiments, reserve the required number of RTX 3090s, ensure port
8000 is free, and have the named model/tokenizer available. Run `nvidia-smi`
before starting; physical GPU IDs in commands may be occupied. Changing GPU IDs
preserves the layout only if hardware/topology match; document the change.

ShareGPT experiments require
`../benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json`.
YAML paths do not expand `~` or environment variables. The original dataset hash
and model revisions were not captured consistently. Record `sha256sum` of the
dataset and resolved model revisions for future experiments.

User-specific absolute paths have been redacted from saved logs, IR, and traces.
Recorded event timings and counts are unchanged. Artifact hashes, when present,
refer to the sanitized files.

The current `vllm bench serve --config` parser mishandles nested commands, so
the recipes use `.venv/bin/inference bench --config ...`. Before starting a
server, validate a workload using that command with `--dry-run`. This validates
arguments without sending requests.

Matching a recipe means matching its recorded configuration and procedure.
Timing, scheduling, sampled output text, and trace bytes are not deterministic.
Each README lists any further barriers to exact historical replication.

## Serving benchmarks

| Experiment | Contents |
|---|---|
| [exp1](exp1/README.md) | BF16 baseline; prefix caching actually enabled despite historical YAML |
| [exp2](exp2/README.md) | INT8 W8A16 with MTP |
| [exp3](exp3/README.md) | BF16 with prefix caching enabled and a larger context limit |
| [exp4](exp4/README.md) | BF16 with the sequential GDN prefill kernel |

## Profiling captures

| Experiment | Contents |
|---|---|
| [gdn_sequential_initial](gdn_sequential_initial/README.md) | First TP=2 sequential Torch capture, including first-use compilation |
| [gdn_sequential_torch_warm](gdn_sequential_torch_warm/README.md) | TP=2 sequential Torch capture after request warmup, with measured prefill-gap comparison |
| [gdn_comparison](gdn_comparison/README.md) | Original chunked Torch server capture and sequential/chunked Proton comparison |
| [gdn_comparison_warm](gdn_comparison_warm/README.md) | Chunked Proton capture after instrumented warmup, with the original large buffers |
| [gdn_comparison_warm_smallbuf](gdn_comparison_warm_smallbuf/README.md) | Warmed chunked Proton capture with smaller profiling buffers; preferred for inspection |

The older `exp1`–`exp4` benchmark metrics are preserved in each tracked
`summary.json`; their original raw JSON and logs live under ignored `artifacts/`.
