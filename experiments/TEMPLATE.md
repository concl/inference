# <experiment_id>

## Purpose

State the question and the single change being measured.

## Configuration

| Setting | Recorded value |
|---|---|
| Source commit / kernel selection | Fill in exact identifiers |
| Software versions | Link to the saved environment manifest |
| Hardware / GPU IDs / TP | Fill in values |
| Model/tokenizer revisions | Fill in resolved revisions, or mark unknown |
| Workload / dataset hash | Link to YAML and record seed and lengths |
| Warmup / cache policy | Fill in values |
| Profiler / instrumentation settings | Fill in values |

## Replication instructions

1. Follow the replication prerequisites in `experiments/README.md` and
   create a fresh output directory. Record environment versions.
2. Select the exact source and snapshot all configs and inputs needed to repeat it.
3. Start the server if applicable; provide a copyable command and readiness check.
4. Run the explicitly configured warmup, then workload/capture. Provide commands,
   seeds, buffer sizes, and output paths; do not rely on changing defaults.
5. Stop the server if applicable and verify expected output files and success counts.

## Results

Record measured values, numerical checks, and the source of each result.

## Artifacts

List curated files and paths under gitignored `artifacts/`. Distinguish original
captures from newly generated replication output.

## Limitations

Record missing provenance, configuration/log discrepancies, shared hardware,
instrumentation overhead, and any barriers to exact historical replication.
