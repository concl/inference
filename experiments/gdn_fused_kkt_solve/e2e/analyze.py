"""Analyze fixed-size paired confirmation blocks, excluding exploratory pilot data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(path):
    report = json.loads(path.read_text())
    if report['status'] != 'complete':
        raise ValueError('Run must complete before analysis')
    result = dict(phase=report['phase'], source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  inference=report['inference'], analysis_plan=report['analysis_plan'], workloads=[])
    for workload in sorted({tuple(b['workload']) for b in report['blocks']}):
        blocks = [b for b in report['blocks'] if tuple(b['workload'])==workload]
        if report['phase']=='confirmation' and (
                len(blocks)!=24 or {b['pair'] for b in blocks}!=set(range(24))):
            raise ValueError('Confirmation requires all 24 distinct preplanned pairs')
        row = dict(workload=workload, pairs=len(blocks),
                   requests_per_arm=report['requests_per_arm'],
                   matching_outputs=sum(b['matching_outputs'] for b in blocks), metrics={})
        for metric in ['mean_e2e_ms','mean_ttft_ms']:
            a = np.array([b['arms']['baseline'][metric] for b in blocks])
            b = np.array([b['arms']['fused'][metric] for b in blocks])
            logs = np.log(a/b)
            rng = np.random.default_rng(1337)
            resampled = logs[rng.integers(0,len(logs),size=(40000,len(logs)))].mean(axis=1)
            simulated = (rng.choice([-1,1],size=(100000,len(logs)))*logs).mean(axis=1)
            p = float((1+np.sum(np.abs(simulated)>=abs(logs.mean())))/100001)
            row['metrics'][metric] = dict(
                baseline_mean_ms=float(a.mean()), fused_mean_ms=float(b.mean()),
                geometric_mean_speedup=float(np.exp(logs.mean())),
                speedup_ci95=[float(x) for x in np.exp(np.quantile(resampled,[.025,.975]))],
                permutation_p_two_sided=p, paired_log_ratios=logs.tolist(),
                endpoint='primary' if metric=='mean_e2e_ms' else 'secondary',
                confirmatory=(report['phase']=='confirmation'))
        ordered=sorted(row['metrics'].values(),key=lambda m:m['permutation_p_two_sided'])
        adjusted=0.0
        for i,metric in enumerate(ordered):
            adjusted=max(adjusted,min(1.,(2-i)*metric['permutation_p_two_sided']))
            metric['holm_adjusted_p']=adjusted
            metric['significant_at_05']=(report['phase']=='confirmation' and adjusted < .05)
        result['workloads'].append(row)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value=analyze(args.results)
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))
