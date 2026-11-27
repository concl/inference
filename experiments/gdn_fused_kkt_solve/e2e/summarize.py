"""Validate and curate a completed end-to-end comparison without raw prompt text."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze import analyze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.run_dir
    pilot=json.loads((root/'pilot/results.json').read_text())
    report=json.loads((root/'confirmation/results.json').read_text())
    control=json.loads((root/'output-control.json').read_text())
    environment=json.loads((root/'environment.json').read_text())
    result=analyze(root/'confirmation/results.json')
    assert pilot['status']=='complete' and report['status']=='complete'
    checks=pilot['validation']['validation']
    assert len(checks)==48
    assert all(c['finite'] and c['max_abs']<=.02 and c['relative_l2']<=.01
               for layer in checks for c in layer['checks'])
    requests=0
    for block in report['blocks']:
        for a,b in zip(block['arms']['baseline']['samples'],block['arms']['fused']['samples']):
            assert a['usage']['prompt_tokens']==b['usage']['prompt_tokens']
            assert a['usage']['completion_tokens']==b['usage']['completion_tokens']==32
            requests+=2
    assert requests==192
    artifacts={}
    for name in ['environment.json','server.log','pilot/results.json','pilot/prompts.json',
                 'confirmation/results.json','confirmation/prompts.json','output-control.json']:
        path=root/name
        artifacts[name]=dict(path=str(Path('artifacts')/root.name/name),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    # Verify that confirmation requests are distinct from the exploratory requests.
    old=json.loads((root/'pilot/prompts.json').read_text())
    new=json.loads((root/'confirmation/prompts.json').read_text())
    old_hashes={hashlib.sha256(p.encode()).hexdigest() for pool in old.values() for p in pool}
    new_hashes={hashlib.sha256(p.encode()).hexdigest() for pool in new.values() for p in pool}
    assert not old_hashes.intersection(new_hashes)
    usages=[s['usage'] for b in report['blocks'] for a in b['arms'].values() for s in a['samples']]
    summary=dict(status='complete', model_id=environment['model_id'],
        model_revision=environment['model_revision'], gpu_uuid=environment['gpu_uuid'],
        environment=environment, started_utc=report['started_utc'], ended_utc=report['ended_utc'],
        confirmation=result, requests=requests, failed_requests=0,
        matching_greedy_outputs=sum(b['matching_outputs'] for b in report['blocks']),
        comparisons=requests//2,
        input_token_range=[min(u['prompt_tokens'] for u in usages),max(u['prompt_tokens'] for u in usages)],
        validation=dict(layer_cases=len(checks),max_abs_limit=.02,relative_l2_limit=.01,
                        maximum_output_error=max(v['checks'][0]['max_abs'] for v in checks),
                        maximum_state_error=max(v['checks'][1]['max_abs'] for v in checks),
                        maximum_relative_l2=max(c['relative_l2'] for v in checks for c in v['checks'])),
        dispatch_calls={variant:sum(b['arms'][variant]['calls'] for b in report['blocks'])
                        for variant in ['baseline','fused']},
        output_repeatability=dict(cases=len(control['cases']),
            baseline_repeatable=sum(c['baseline']['repeatable'] for c in control['cases']),
            fused_repeatable=sum(c['fused']['repeatable'] for c in control['cases']),
            variants_match=sum(c['variants_match'] for c in control['cases']),
            note=control['purpose']),
        artifacts=artifacts,
        limitations=['One AWQ checkpoint, one GPU and host, one server session, concurrency 1',
                     'Constructed long-input prompts from ShareGPT; exactly 32 output tokens forced',
                     'PyTorch compilation disabled; decode-only full CUDA Graphs enabled',
                     'Shared host and other GPUs active; no clock locking',
                     'Activation parity passed; exact greedy output agreement and task accuracy are separate questions',
                     'Pilot excluded; marginal bootstrap intervals and Holm-adjusted endpoint p-values reported'])
    exact=root/'exact-test-check.json'
    if exact.exists():
        summary['supplementary_exact_permutation_check']=json.loads(exact.read_text())
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(requests=requests,matching_outputs=summary['matching_greedy_outputs'],
                          metrics=result['workloads'][0]['metrics']),indent=2))


if __name__=='__main__':
    main()
