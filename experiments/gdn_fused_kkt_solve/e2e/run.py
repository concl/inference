"""Paired HTTP end-to-end GDN experiment, with separate pilot/confirmation data."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import urllib.request

import aiohttp
import numpy as np
from transformers import AutoTokenizer


def rpc(base, method, *args):
    request = urllib.request.Request(base+'/collective_rpc',
        data=json.dumps(dict(method=method, args=list(args))).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)['results'][0]


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def hardware(args):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True)
    for line in raw.splitlines():
        uuid, pid = [v.strip() for v in line.split(',')]
        if uuid != args.gpu_uuid:
            continue
        ancestor = int(pid)
        while ancestor > 1 and ancestor != args.server_pid:
            fields = Path(f'/proc/{ancestor}/stat').read_text().rsplit(')',1)[1].split()
            ancestor = int(fields[1])
        if ancestor != args.server_pid:
            raise RuntimeError(f'GPU contention from PID {pid}')
    return subprocess.check_output(['nvidia-smi', '-i', args.gpu_uuid,
        '--query-gpu=uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used',
        '--format=csv,noheader,nounits'],text=True).strip()


async def request(session, base, prompt, output_len):
    started = time.perf_counter()
    first = None
    text = ''
    usage = None
    async with session.post(base+'/v1/completions', json=dict(
            model='qwen35-4b-awq', prompt=prompt, max_tokens=output_len,
            temperature=0, seed=1337, ignore_eos=True, stream=True,
            stream_options={'include_usage': True})) as response:
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {await response.text()}')
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            data = json.loads(line[5:].strip())
            if data.get('error'):
                raise RuntimeError(data['error'])
            for choice in data.get('choices', []):
                content = choice.get('text', '')
                if content:
                    first = first or time.perf_counter()
                    text += content
            usage = data.get('usage') or usage
    elapsed = time.perf_counter()-started
    if first is None or usage is None or usage['completion_tokens'] != output_len:
        raise RuntimeError(f'Invalid streamed response: {usage}')
    return dict(e2e_ms=elapsed*1000, ttft_ms=(first-started)*1000,
                text=text, usage=usage)


async def main_async(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    records = json.loads(args.dataset.read_text())
    passages = [turn['value'] for row in records[:1500] for turn in row['conversations']
                if isinstance(turn.get('value'), str) and len(turn['value']) > 500]
    rng = random.Random(args.seed)
    rng.shuffle(passages)
    del records
    pairs = 4 if args.phase == 'pilot' else 24
    requests_per_arm = 2 if args.phase == 'pilot' else 4
    workloads = [(2048, 32), (8192, 32), (8192, 128)] if args.phase == 'pilot' else [(8192, 32)]
    report = dict(status='running', phase=args.phase, seed=args.seed,
                  primary_metric='mean HTTP end-to-end latency per arm; paired block is the statistical unit',
                  primary_workload=dict(input_tokens_approx=8192, output_tokens=32, concurrency=1),
                  planned_pairs=pairs, requests_per_arm=requests_per_arm,
                  dataset_sha256=dataset_hash, blocks=[], warmups=[], validation=None,
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  inference=dict(temperature=0, ignore_eos=True, prefix_caching=False),
                  gpu_uuid=args.gpu_uuid, initial_hardware=hardware(args),
                  analysis_plan='Fixed 24 confirmation pairs; two-sided paired sign-flip permutation tests and paired-block bootstrap 95% CIs on geometric mean latency ratios. Pilot excluded. Completion latency primary, TTFT secondary; Holm correction over these two predeclared endpoints.')
    save(args.output_dir/'results.json', report)
    prompts = {}
    for target, output in workloads:
        pool = []
        for i in range(pairs*requests_per_arm+4):
            selected = [passages[(i*13+j) % len(passages)] for j in range(32)]
            text = '\n\n'.join(selected)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            while len(tokens) < target:
                tokens += tokens
            prompt = tokenizer.decode(tokens[:target-32], skip_special_tokens=True)
            prompt += '\n\nSummarize the important information above in a concise paragraph.\nSummary:'
            pool.append(prompt)
        prompts[(target, output)] = pool
    save(args.output_dir/'prompts.json',
         {f'{t}-{o}': pool for (t,o), pool in prompts.items()})
    timeout = aiohttp.ClientTimeout(total=300)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if args.phase == 'pilot':
                rpc(args.base_url, 'gdn_e2e_set_variant', 'validate')
                for target in [2048, 8192]:
                    await request(session, args.base_url, prompts[(target,32)][0], 1)
                report['validation'] = rpc(args.base_url, 'gdn_e2e_stats')
                save(args.output_dir/'results.json', report)
            # Warm both variants and all measured shapes before timed blocks.
            for variant in ['baseline', 'fused']:
                rpc(args.base_url, 'gdn_e2e_set_variant', variant)
                for workload in workloads:
                    for prompt in prompts[workload][-3:]:
                        r = await request(session, args.base_url, prompt, workload[1])
                        report['warmups'].append(dict(variant=variant, workload=workload,
                                                     e2e_ms=r['e2e_ms']))
            for target, output in workloads:
                for pair in range(pairs):
                    order = ['baseline', 'fused']
                    rng.shuffle(order)
                    block = dict(pair=pair, workload=[target,output], order=order, arms={})
                    block['hardware_before'] = hardware(args)
                    chosen = prompts[(target,output)][pair*requests_per_arm:(pair+1)*requests_per_arm]
                    for variant in order:
                        before = rpc(args.base_url, 'gdn_e2e_set_variant', variant)
                        samples = [await request(session, args.base_url, p, output) for p in chosen]
                        after = rpc(args.base_url, 'gdn_e2e_stats')
                        if after['calls'].get(variant,0) <= before['calls'].get(variant,0):
                            raise RuntimeError('Requested GDN variant was not executed')
                        block['arms'][variant] = dict(samples=samples,
                            mean_e2e_ms=float(np.mean([s['e2e_ms'] for s in samples])),
                            mean_ttft_ms=float(np.mean([s['ttft_ms'] for s in samples])),
                            calls=after['calls'][variant]-before['calls'].get(variant,0))
                    block['matching_outputs'] = sum(a['text']==b['text'] for a,b in zip(
                        block['arms']['baseline']['samples'], block['arms']['fused']['samples']))
                    block['hardware_after'] = hardware(args)
                    report['blocks'].append(block)
                    save(args.output_dir/'results.json', report)
                    print(json.dumps(dict(workload=[target,output],pair=pair,order=order,
                        baseline_ms=block['arms']['baseline']['mean_e2e_ms'],
                        fused_ms=block['arms']['fused']['mean_e2e_ms'],
                        matching_outputs=block['matching_outputs'])), flush=True)
            report['final_worker_stats'] = rpc(args.base_url, 'gdn_e2e_stats')
            report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save(args.output_dir/'results.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8157')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['pilot','confirmation'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--server-pid', type=int, required=True)
    asyncio.run(main_async(parser.parse_args()))
