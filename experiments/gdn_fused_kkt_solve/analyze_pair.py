"""Curate completed two-GPU GDN results while retaining per-GPU measurements."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark import save


def curate(run_dir, model_id):
    status = json.loads((run_dir / 'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('Only completed runs may be summarized')
    reports = {}
    artifacts = {}
    for device in status['devices']:
        label = f'gpu{device[0]}'
        path = run_dir / label / 'full/results.json'
        report = json.loads(path.read_text())
        if report['status'] != 'complete' or not report.get('correctness_passed'):
            raise ValueError(f'{label} did not complete correctness and performance')
        if not all(c['passed'] for c in report['correctness']):
            raise ValueError(f'{label} has failed reference checks')
        reports[label] = report
        artifacts[label] = dict(path=str(path.relative_to(run_dir)),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    first = next(iter(reports.values()))
    for report in reports.values():
        if (report['environment']['model_profile'] != first['environment']['model_profile']
                or report['environment']['harness_sha256'] != first['environment']['harness_sha256']):
            raise ValueError('Runs must use identical model geometry and harnesses')
    summary = dict(
        model_id=model_id, run_directory=str(run_dir), started_utc=status['started_utc'],
        ended_utc=status['ended_utc'], status='complete',
        model_profile=status['model_profile'], artifacts=artifacts,
        provenance=first['environment']['provenance'],
        harness_sha256=first['environment']['harness_sha256'],
        rounds=first['arguments']['rounds'], iterations=first['arguments']['iterations'],
        correctness={}, measurements=[],
        limitations=first['limitations'] + [
            'Two GPUs run independent operator replicas, not a distributed model',
            'Both replicas share the host; eager submission timing may be affected by concurrent CPU activity',
            'Model config determines geometry only; weights and model activations are not loaded',
            'Candidate changes KKT arithmetic as well as launch fusion; speedup cannot be assigned solely to fusion'])
    for label, report in reports.items():
        checks = report['correctness']
        names = checks[0]['checks']
        summary['correctness'][label] = dict(
            gpu_uuid=report['environment']['gpu_uuid'],
            gpu_name=report['environment']['gpu_name'], versions=report['environment']['versions'],
            cases_passed=len(checks), performance_shapes=len(report['performance']),
            maximum_errors={name: dict(
                max_abs=max(c['checks'][name]['max_abs'] for c in checks),
                relative_l2=max(c['checks'][name]['relative_l2'] for c in checks)) for name in names})
    expected_names = [row['case']['name'] for row in first['performance']]
    for report in reports.values():
        if [row['case']['name'] for row in report['performance']] != expected_names:
            raise ValueError('Performance case coverage differs between GPUs')
    for i, first_row in enumerate(first['performance']):
        for j, measurement in enumerate(first_row['measurements']):
            row = dict(case=first_row['case'], region=measurement['region'],
                       mode=measurement['mode'], devices={})
            for label, report in reports.items():
                other_row = report['performance'][i]
                other = other_row['measurements'][j]
                if (other['region'], other['mode']) != (row['region'], row['mode']):
                    raise ValueError('Measurement modes differ between GPUs')
                if not all(c['passed'] for c in other_row['parity'].values()):
                    raise ValueError('Performance-shape parity check failed')
                row['devices'][label] = dict(
                    baseline_us=other['baseline']['median_us'], fused_us=other['fused']['median_us'],
                    speedup=other['speedup'], paired_round_speedups=other['paired_round_speedups'],
                    paired_speedup_range=[min(other['paired_round_speedups']),
                                          max(other['paired_round_speedups'])],
                    rounds_include_both_speedup_and_slowdown=(
                        min(other['paired_round_speedups']) < 1 < max(other['paired_round_speedups'])),
                    clock_sm_mhz_range=[min(int(t[5]) for t in other['telemetry']),
                                        max(int(t[5]) for t in other['telemetry'])],
                    parity=other_row['parity'])
            summary['measurements'].append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = curate(args.run_dir.resolve(), args.model_id)
    save(args.output, summary)
    print(f'Saved {len(summary["measurements"])} paired measurements to {args.output}')


if __name__ == '__main__':
    main()
