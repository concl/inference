#!/usr/bin/env python3
"""Copy an exp5 capture into a new exp7 and format its reports like exp1."""
import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

METRICS = ('output_throughput', 'p95_ttft_ms', 'p95_itl_ms')
PARAMS = ('concurrency', 'max_batched_tokens', 'max_sequences')


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    original_run = args.run.resolve()
    if original_run.parent.name != 'artifacts':
        parser.error('Expected experiments/exp5/artifacts/<run-id>')
    original_experiment = original_run.parent.parent
    root = original_experiment.parent.parent
    experiment = root / 'experiments' / 'exp7'
    if experiment.exists():
        parser.error(f'{experiment} already exists. No files changed; choose a fresh destination before retrying.')
    required = ['run.json', 'environment.json', 'study.db', 'run.py',
                'source-config.yaml', 'source-benchmark-vision-arena.yaml',
                'config.yaml', 'benchmark-vision-arena.yaml']
    for name in required:
        if not (original_run / name).is_file():
            parser.error(f'Missing historical input: {original_run / name}')
    # Check basic inputs before creating the new experiment.
    read_json(original_run / 'run.json')
    read_json(original_run / 'environment.json')
    run = experiment / 'artifacts' / original_run.name
    def ignore_database(directory, names):
        if Path(directory) == original_run:
            return set(names) & {'study.db', 'study.db-wal', 'study.db-shm', 'study.db-journal'}
        return set()
    print(f'Copying {original_run} to {run}', flush=True)
    shutil.copytree(original_run, run, ignore=ignore_database)
    # SQLite backup includes committed WAL data without changing the source DB.
    with sqlite3.connect((original_run / 'study.db').as_uri() + '?mode=ro', uri=True) as source:
        with sqlite3.connect(run / 'study.db') as target:
            source.backup(target)
    for source_name, dest_name in [
        ('source-config.yaml', 'config.yaml'),
        ('source-benchmark-vision-arena.yaml', 'benchmark-vision-arena.yaml'),
    ]:
        shutil.copy2(run / source_name, experiment / dest_name)
    # Preserve the historical snapshot; only the new executable driver's label changes.
    driver = (run / 'run.py').read_text()
    driver = driver.replace('"experiment": "exp5"', '"experiment": EXPERIMENT_DIR.name')
    (experiment / 'run.py').write_text(driver)
    (run / 'capture-origin.json').write_text(json.dumps({
        'source_run': str(original_run),
        'copied_at': datetime.now(timezone.utc).isoformat(),
        'note': 'Copied historical capture, not a new measurement. Original absolute paths in snapshots and commands are retained. SQLite copied with its backup API; other files are not an atomic live snapshot.',
    }, indent=2) + '\n')
    meta = read_json(run / 'run.json')
    env = read_json(run / 'environment.json')
    with sqlite3.connect((run / 'study.db').as_uri() + '?mode=ro', uri=True) as db:
        states = db.execute('SELECT number, state FROM trials ORDER BY number').fetchall()
    counts = Counter(state for _, state in states)
    expected = meta.get('num_prompts')
    if type(expected) is not int or expected < 1:
        raise ValueError('run.json must record a positive num_prompts')
    valid, rows, problems = [], [], []
    raw_results = {}
    sources = {run / 'run.json', run / 'environment.json'}
    for number, state in states:
        folder = run / f'trial-{number:04d}'
        row = {'trial': number, 'state': state, 'included': False}
        try:
            params = read_json(folder / 'params.json')
            row.update({key: params[key] for key in PARAMS})
            sources.add(folder / 'params.json')
            if state == 'COMPLETE':
                result = read_json(folder / 'result.json')
                if result.get('completed') != expected or result.get('failed', 0) != 0:
                    raise ValueError('Request count mismatch or failed requests')
                metrics = {key: result[key] for key in METRICS}
                if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in metrics.values()):
                    raise ValueError('Missing, nonfinite, or nonpositive metrics')
                valid.append({'trial': number, 'params': {k: params[k] for k in PARAMS},
                              'metrics': metrics, 'completed': result['completed']})
                raw_results[number] = result
                row.update(metrics, included=True)
                sources.add(folder / 'result.json')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row['error'] = str(exc)
            problems.append(f'trial-{number:04d}: {exc}')
        if (folder / 'error.txt').is_file():
            row['error'] = (folder / 'error.txt').read_text().strip()
            sources.add(folder / 'error.txt')
        rows.append(row)
    if not valid:
        raise SystemExit('No COMPLETE trials with valid result.json; no reports written.')

    def dominates(a, b):
        x, y = a['metrics'], b['metrics']
        weak = x[METRICS[0]] >= y[METRICS[0]] and all(x[k] <= y[k] for k in METRICS[1:])
        return weak and any(x[k] != y[k] for k in METRICS)

    pareto = [a for a in valid if not any(dominates(b, a) for b in valid)]
    front = {t['trial'] for t in pareto}
    grid = math.prod(len(v) for v in meta['search_space'].values())
    partial = len(valid) != grid
    created = datetime.now(timezone.utc).isoformat()
    def rel(path):
        return Path(os.path.relpath(path, experiment)).as_posix()
    def link(path, title=None):
        return f'[{title or path.name}]({rel(path)})'
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    def command(*values):
        return shlex.join([str(x) for x in values])

    run_summary = {
        'experiment': experiment.name, 'dataset': meta.get('dataset', {}),
        'source_run_directory': str(original_run.relative_to(root)),
        'status': 'partial' if partial else 'complete_grid',
        'run_directory': str(run.relative_to(root)), 'generated_at': created,
        'attempted_trials': len(states), 'completed_trials': len(valid),
        'failed_trials': counts['FAIL'], 'state_counts': dict(counts),
        'excluded_or_unreadable': problems,
        'benchmarks': {
            f"trial-{t['trial']:04d}": {
                'params': t['params'], 'completed': t['completed'], **t['metrics'],
                'artifact': rel(run / f"trial-{t['trial']:04d}" / 'result.json')
            } for t in valid
        },
        'pareto_trials': [{k: t[k] for k in ('trial', 'params', 'metrics')} for t in pareto],
        'note': 'Export from recorded COMPLETE trials; not a new benchmark.',
    }
    # Match exp1 exactly: experiment + benchmarks, preserving full result objects.
    # The key identifies the scheduler settings not stored in vLLM result JSON.
    benchmark_keys = {
        t['trial']: f"vision-arena-trial-{t['trial']:04d}"
                    f"-c{t['params']['concurrency']}"
                    f"-bt{t['params']['max_batched_tokens']}"
                    f"-seq{t['params']['max_sequences']}"
        for t in valid
    }
    summary = {
        'experiment': experiment.name,
        'benchmarks': {benchmark_keys[t['trial']]: raw_results[t['trial']] for t in valid},
    }
    export_dir = run / 'exports' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    versions = ', '.join(f'{k}={v}' for k, v in env.get('versions', {}).items())
    git = env.get('inference_git', {})
    commit = git.get('stdout') if git.get('returncode') == 0 else 'unknown'
    layout = ', '.join(f'{label}={meta.get(key, "unknown")}' for label, key in
                      [('TP', 'tensor_parallel_size'), ('PP', 'pipeline_parallel_size'),
                       ('DP', 'data_parallel_size')])
    dataset = env.get('dataset', meta.get('dataset', {}))
    config_rows = [
        ('Server / benchmark configs', f'{link(experiment / "config.yaml")}, {link(experiment / "benchmark-vision-arena.yaml")}. Historical effective configs are linked below; current files may differ.'),
        ('Source commit / kernel selection', f'Inference commit: {commit}. Installed vLLM; no kernel override applied by this runner. Exact vLLM source commit unknown. Saved driver: {link(run / "run.py")}'),
        ('Software versions', f'{link(run / "environment.json")}; {versions}'),
        ('Hardware / GPU IDs / TP', f'GPU IDs={meta.get("gpus", "unknown")}; {layout}. Inventory/UUIDs: {link(run / "environment.json")}'),
        ('Model/tokenizer revisions', f'Model={env.get("model", "unknown")}; model revision={env.get("model_revision_requested", "unknown")}; tokenizer revision={env.get("tokenizer_revision_requested", "unknown")}. Resolved revisions not independently verified.'),
        ('Workload / dataset hash', f'{dataset.get("id", "unknown")}, split={dataset.get("split", "unknown")}; text + image; measured prompts={expected}; seed={meta.get("seed", "unknown")}. Dataset hash unknown. Lengths and effective options: {link(run / "benchmark-vision-arena.yaml")}'),
        ('Warmup / cache policy', f'{meta.get("warmup_prompts", "unknown")} warmup requests; runner uses measurement seed + 1 for warmup; prefix caching retained between warmup and measurement, fresh server per trial, disk compilation caches retained. Exact commands saved per trial.'),
        ('Profiler / instrumentation settings', 'No explicit profiler enabled by the runner; profiler buffer sizes not applicable. External instrumentation unknown.'),
        ('Server settings / search space', f'{link(run / "config.yaml")}; {json.dumps(meta["search_space"], sort_keys=True)}'),
    ]
    config_table = '| Setting | Recorded value |\n|---|---|\n' + ''.join(
        f'| {cell(k)} | {cell(v)} |\n' for k, v in config_rows)
    result_table = '| Trial | Concurrency | Batched tokens | Max sequences | Output tokens/s | p95 TTFT (ms) | p95 ITL (ms) | Pareto | Source |\n|---|---:|---:|---:|---:|---:|---:|---|---|\n'
    for t in valid:
        result_table += '| ' + ' | '.join([
            str(t['trial']), *(str(t['params'][k]) for k in PARAMS),
            *(f"{t['metrics'][k]:.3f}" for k in METRICS),
            'yes' if t['trial'] in front else '',
            link(run / f"trial-{t['trial']:04d}" / 'result.json', 'JSON')]) + ' |\n'

    # Replicate with the repository runner only after verifying historical inputs.
    # This avoids executing the archived driver at the wrong directory depth.
    checks = []
    checks.append(command('cmp', str((experiment / 'run.py').relative_to(root)),
                          str((export_dir / 'replication-run.py').relative_to(root))) + ' || exit 1')
    for current, saved in [('config.yaml', 'source-config.yaml'),
                           ('benchmark-vision-arena.yaml', 'source-benchmark-vision-arena.yaml')]:
        checks.append(command('cmp', str(experiment.relative_to(root) / current),
                              str((run / saved).relative_to(root))) + ' || exit 1')
    for name in ('serve.py', 'bench.py', 'cli.py'):
        checks.append(command('cmp', f'scripts/{name}', str((run / 'source' / name).relative_to(root))) + ' || exit 1')
    launch = command('.venv/bin/python', str(experiment.relative_to(root) / 'run.py'),
                     '--gpus', meta['gpus'], '--parallelism', meta['parallelism'],
                     '--port', meta.get('port', 8000), '--trials', grid,
                     '--num-prompts', expected, '--warmup-prompts', meta['warmup_prompts'])
    example = run / f"trial-{valid[0]['trial']:04d}"
    replication = f'''1. From the repository root on a host with the recorded GPU layout, use the existing environment. Compare versions and available GPU inventory with the historical manifest. Ensure the selected GPUs and port are available; these commands do not reserve hardware. This Vision Arena runner uses installed vLLM and Hugging Face data, without the older GDN overlay or local ShareGPT prerequisites.

   ```bash
   cd {shlex.quote(str(root))}
   .venv/bin/python -c 'import optuna, yaml, plotly; print("Reporting/runner imports OK")'
   nvidia-smi
   ss -ltnp 'sport = :{meta.get("port", 8000)}'
   ```

2. Verify the current runner, configs, and launchers match the historical snapshots before replaying them. Stop and reconcile any differences. Compare the installed package versions against `environment.json`; exact model/dataset revisions remain unavailable.

   ```bash
   {chr(10).join('   ' + s for s in checks).lstrip()}
   ```

3. Validate the launch arguments. This produces a fresh capture directory and does not start a model server.

   ```bash
   {launch} --dry-run
   ```

4. Run the sweep into a new timestamped directory. The runner snapshots inputs and environment, starts each server, polls `/health`, performs the configured warmup, and measures the workload. The recorded measurement seed is {meta.get('seed', 'unknown')}; the archived runner uses the next seed for warmup. It preserves caches as described above. These are performance measurements; no profiler capture is requested.

   ```bash
   {launch}
   ```

   Exact historical argv for an included trial: {link(example / 'server-command.txt', 'server')}, {link(example / 'warmup-command.txt', 'warmup')}, {link(example / 'result-command.txt', 'measurement')}. Historical output paths are evidence, not destinations to reuse.

5. The runner stops its server process group after each trial. Record the newly printed output directory, inspect its success/failure counts, and check each included result completed {expected} requests with no reported failures. Check the server port after the runner exits. A new sweep does not resume the historical study.

   ```bash
   ss -ltnp 'sport = :{meta.get("port", 8000)}'
   ```
'''
    failures = Counter(row.get('error', 'No error.txt recorded') for row in rows if row['state'] == 'FAIL')
    failure_text = '\n'.join(f'- {n} trial(s): {cell(reason)}' for reason, n in failures.items()) or 'No recorded failed trials.'
    readme = f'''# {experiment.name}

## Purpose

Measure output throughput, p95 time to first token (TTFT), and p95 inter-token latency (ITL) while varying client concurrency, batched-token budget, and maximum active sequences. This is a multi-parameter serving sweep, not an isolated kernel comparison or model-quality evaluation. The selected historical run is `{run.name}`, copied from `{original_run.relative_to(root)}` into exp7 without rerunning the benchmark.

## Configuration

{config_table}
## Replication instructions

{replication}
## Results

Status: **{'partial' if partial else 'complete grid'}**. Exported {created}. Recorded trial states: `{dict(counts)}`. Included {len(valid)} validated completed trials from {len(states)} attempted, out of {grid} grid combinations. A RUNNING database state is not evidence that a process is still alive.

All included trials completed {expected} requests, reported no failures, and have finite positive objective metrics. The {len(pareto)} Pareto points are relative only to the included trials. JSON retains full precision; the table rounds to three decimals.

All fields from each included raw benchmark JSON are preserved in [summary.json](summary.json), using the same `experiment` / `benchmarks` structure as exp1. Benchmark keys identify trial number, concurrency (`c`), batched-token budget (`bt`), and maximum sequences (`seq`). No results from different configurations are averaged together.

{result_table}
Recorded failure reasons:

{failure_text}

Export exclusions/read errors: {json.dumps(problems)}.

## Artifacts

- Copies of the historical capture: `{rel(run)}/`, including per-trial params, logs, raw result JSON, command files, `study.db`, configs, launcher/driver snapshots, and environment manifest. These remain under gitignored `artifacts/`. The exp5 originals remain unchanged.
- {link(run / 'capture-origin.json')} records the copy's origin. Absolute paths in historical command files and manifests still identify the original exp5 run; no historical evidence is relabeled as a new measurement. The top-level `run.py` adjusts only the experiment label so future summaries identify exp7; {link(export_dir / 'replication-run.py')} records that runnable version for comparison.
- Tracked curated files: this [README.md](README.md), [summary.json](summary.json), [config.yaml](config.yaml), [benchmark-vision-arena.yaml](benchmark-vision-arena.yaml), and the sweep driver [run.py](run.py).
- Newly generated audit reports: {link(export_dir / 'run-summary.json')}, {link(export_dir / 'trials.csv')}, {link(export_dir / 'pareto.json')}, and {link(export_dir / 'hashes.json')}. These are derived from the historical captures, not newly measured results.
- `hashes.json` records hashes of the source files consumed by this export. It does not hash the remote dataset or claim an atomic snapshot of a running study.
- Prior README/summary versions, if present, are backed up under `{rel(export_dir / 'previous')}/` before replacement.
- Future replication output belongs in a fresh timestamped artifact directory; never overwrite the original capture.

## Limitations

This {'interrupted/partial' if partial else 'completed'} sweep cannot establish performance for missing or excluded configurations. Failed and RUNNING trials are excluded from the performance frontier. File timestamps and database states alone cannot establish why a remote host became unreachable or whether another user terminated a process.

Resolved model, tokenizer, dataset revisions and dataset-content hashes were not captured reliably. Installed package versions do not identify an exact vLLM source commit. Source/config identity checks above do not recreate uncaptured dependencies. Review any dirty-worktree details in the saved environment manifest. Do not infer historical versions from current defaults.

The runner sets dataset sampling seeds, but generation defaults and scheduling may still vary; inspect saved benchmark command/config/logs for temperature and token limits. Warmup and endpoint-test requests can populate prefix and multimodal caches. Different seeds do not guarantee disjoint prefixes. Disk compilation caches are retained. Input image sizes and prompt lengths vary; requested output length is not necessarily achieved output length.

Shared hardware/host load may affect results and was not continuously captured. GPU IDs, TP/PP layout and model settings must match when comparing runs; three-GPU PP and four-GPU TP runs are different configurations. p95 estimates from {expected} requests are noisy; there are no repeated-run confidence intervals. No explicit profiler was enabled; external instrumentation overhead is unknown. Inspect logs for configuration discrepancies before curating these results.
'''
    for path in [run / 'run.py', run / 'config.yaml', run / 'benchmark-vision-arena.yaml',
                 run / 'source-config.yaml', run / 'source-benchmark-vision-arena.yaml',
                 *run.glob('source/*'), *example.glob('*-command.txt')]:
        if path.is_file():
            sources.add(path)
    hashes = {rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    # Serialize all JSON before replacing any curated files.
    outputs = {'README.md': readme,
               'summary.json': json.dumps(summary, indent=2, allow_nan=False) + '\n'}
    audit = {
        'run-summary.json': json.dumps(run_summary, indent=2, allow_nan=False) + '\n',
        'pareto.json': json.dumps(run_summary['pareto_trials'], indent=2, allow_nan=False) + '\n',
        'hashes.json': json.dumps(hashes, indent=2) + '\n',
    }
    export_dir.mkdir(parents=True, exist_ok=False)
    (export_dir / 'replication-run.py').write_text(driver)
    backup_dir = export_dir / 'previous'
    backup_dir.mkdir()
    for name in outputs:
        path = experiment / name
        if path.exists():
            shutil.copy2(path, backup_dir / name)
    for name, content in audit.items():
        (export_dir / name).write_text(content)
    for name, content in outputs.items():
        (experiment / name).write_text(content)
    with (export_dir / 'trials.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['trial', 'state', 'included', *PARAMS, *METRICS, 'error'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Included {len(valid)} completed trials; {len(pareto)} Pareto points.')
    for name in outputs:
        print(experiment / name)
    print('Audit reports and prior README/summary backups:', export_dir)
    print('exp7 created. The entire source exp5 experiment was left untouched. No benchmarks were rerun.')


if __name__ == '__main__':
    main()
