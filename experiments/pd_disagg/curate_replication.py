"""Curate a completed adapted PD run, requiring transfer and trace evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def metric(path,name):
    values=[]
    for line in path.read_text().splitlines():
        if line.startswith(name+'{') or line.startswith(name+' '):
            values.append(float(line.rsplit(' ',1)[1]))
    if not values:raise ValueError(f'Metric {name} missing from {path}')
    return sum(values)


def trace_summary(path):
    data=json.loads(path.read_text())
    events=data['traceEvents']
    kernels=[event for event in events if event.get('cat')=='kernel']
    if not kernels:raise ValueError(f'No GPU kernels in {path}')
    durations=Counter()
    for event in kernels:durations[event['name']]+=event.get('dur',0)
    return dict(file=path.name,source_bytes=path.stat().st_size,source_sha256=digest(path),
        gpu_kernel_count=len(kernels),event_count=len(events),
        top_kernel_summed_durations_ms=[dict(name=name,summed_duration_ms=duration/1000)
                                       for name,duration in durations.most_common(8)],
        note='Summed kernel durations are diagnostic, not wall time or a critical-path decomposition')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir.resolve()
    status=json.loads((root/'status.json').read_text())
    if status['status']!='complete':raise ValueError('Only completed runs can be curated')
    if len(status['checks'])!=2 or not all(c['exact_output_match'] for c in status['checks']):
        raise ValueError('Text correctness checks did not pass')
    handoffs=[]
    for line in (root/'proxy.log').read_text().splitlines():
        if line.startswith('{'):
            item=json.loads(line)
            if item.get('event')=='kv_handoff':handoffs.append(item)
    if not handoffs or any(not x['remote_engine_id'] or x['remote_block_count']<=0 for x in handoffs):
        raise ValueError('Missing or empty KV handoff evidence')
    metrics=root/'decode-metrics-after.txt'
    successful=metric(metrics,'vllm:nixl_bytes_transferred_count')
    failed=metric(metrics,'vllm:nixl_num_failed_transfers_total')
    if successful<=0 or failed!=0:raise ValueError('Transfer metrics do not show clean successful KV transfers')
    rows=[]
    for benchmark in status['benchmarks']:
        raw=json.loads((root/benchmark['result_file']).read_text())
        if raw.get('completed')!=benchmark['completed'] or raw.get('failed',0)!=0:
            raise ValueError('Failed or incomplete benchmark')
        rows.append(dict(**benchmark,profiled=benchmark['label']=='vision-trace',
            output_throughput=raw.get('output_throughput'),mean_ttft_ms=raw.get('mean_ttft_ms'),
            p95_ttft_ms=raw.get('p95_ttft_ms'),p95_itl_ms=raw.get('p95_itl_ms'),
            total_input_tokens=raw.get('total_input_tokens'),total_output_tokens=raw.get('total_output_tokens')))
    traces={}
    cached=root/'trace-integrity-check.json'
    cached=json.loads(cached.read_text()) if cached.exists() else {}
    for role in ['prefill','decode']:
        paths=list((root/role/'torch').glob('*.pt.trace.json'))
        if not paths:raise ValueError(f'Missing {role} trace')
        previous=cached.get(role,{})
        if (len(paths)==1 and previous.get('file')==paths[0].name
                and previous.get('source_sha256')==digest(paths[0])):
            traces[role]=[previous]
        else:traces[role]=[trace_summary(path) for path in paths]
    artifacts={}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(part in ['triton_cache','inductor_cache','__pycache__'] for part in path.parts):continue
        artifacts[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=digest(path))
    summary=dict(experiment='pd_disagg',status='completed_adapted_replication',
        run_directory=str(Path('artifacts/replications')/root.name),model=status['model'],revision=status['revision'],
        gpu_indices=status['gpu_indices'],gpu_uuids=status['gpu_uuids'],
        started_utc=status['started_utc'],ended_utc=status['ended_utc'],adaptations=status['adaptations'],
        versions=json.loads((root/'versions.json').read_text()),
        classification=dict(runtime_disaggregation_verified=True,successful_transfers=int(successful),
            failed_transfers=int(failed),handoff_records=len(handoffs),
            bytes_transferred=metric(metrics,'vllm:nixl_bytes_transferred_sum'),
            mean_transfer_ms=1000*metric(metrics,'vllm:nixl_xfer_time_seconds_sum')/successful),
        text_correctness_checks=status['checks'],benchmarks=rows,traces=traces,artifacts=artifacts,
        limitations=['Adapted Qwen3.5-4B AWQ replication, not the inaccessible original Qwen3.8-27B AutoRound checkpoint',
            'Ordinary baseline generates on one GPU; PD uses two GPUs, so this is not a hardware-budget-matched comparison',
            'Two small ordered repeats; shared host and unlocked clocks; no statistical significance claim',
            'Vision correctness not established by two text equality checks; measured request success is distinct from image-task accuracy',
            'Profiler measurements excluded from unprofiled performance comparisons',
            'Native JIT and multimodal processor caches may differ by run order; inspect logs',
            'Dataset revision was not pinned by the supplied benchmark interface'])
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['classification'],indent=2))


if __name__=='__main__':main()
