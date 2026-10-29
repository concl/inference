# Shared GDN experiment utilities

This directory contains code shared by the GDN experiment family. It is not an
experiment and does not own captures, benchmark configs, or recorded results.

- `profile_intra.py`: single-layer sequential/chunked Proton instrumentation.
  Call it from the repository root with an explicit `--output-dir` owned by the
  experiment. Token count, warmup count, and buffer size belong in each recipe.
- `import_overlay/sitecustomize.py`: selects a snapshotted FLA chunk module when
  this directory's `import_overlay/` is on `PYTHONPATH` and `GDN_CHUNK_SOURCE`
  names the snapshot. No installed package or fork checkout is modified.

Keep one implementation here rather than copying it into each experiment.
Capture experiment-specific changes to these utilities in git, and record the
repository commit with future runs. Experiment-specific analyzers stay in their
own experiment directory.

See [the experiment index](../../README.md) for callers and complete replication
instructions. These tools are deliberately outside the reusable `inference`
package under `scripts/`.
