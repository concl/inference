"""Post-benchmark repeatability control; never included in latency inference."""
import argparse
import asyncio
import json
from pathlib import Path

import aiohttp

from run import request, rpc, save


async def main(args):
    report=json.loads(args.results.read_text())
    prompts=json.loads(args.prompts.read_text())
    matching=[]
    differing=[]
    for block in report['blocks']:
        key='-'.join(map(str,block['workload']))
        for i,(a,b) in enumerate(zip(block['arms']['baseline']['samples'],
                                    block['arms']['fused']['samples'])):
            entry=dict(pair=block['pair'], sample=i, workload=block['workload'],
                       prompt=prompts[key][block['pair']*report['requests_per_arm']+i],
                       original_baseline=a['text'], original_fused=b['text'])
            (matching if a['text']==b['text'] else differing).append(entry)
    selected=differing[:4]+matching[:2]
    output=dict(purpose='Post-hoc output repeatability control, excluded from performance statistics', cases=[])
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        for entry in selected:
            row={k:v for k,v in entry.items() if k!='prompt'}
            for variant in ['baseline','fused']:
                rpc(args.base_url,'gdn_e2e_set_variant',variant)
                repeats=[await request(session,args.base_url,entry['prompt'],entry['workload'][1]) for _ in range(2)]
                row[variant]=dict(texts=[r['text'] for r in repeats],
                                  repeatable=repeats[0]['text']==repeats[1]['text'],
                                  matches_original=all(r['text']==entry['original_'+variant] for r in repeats))
            row['variants_match']=row['baseline']['texts'][0]==row['fused']['texts'][0]
            output['cases'].append(row)
            save(args.output,output)
    rpc(args.base_url,'gdn_e2e_set_variant','baseline')
    print(json.dumps(dict(cases=len(selected),baseline_repeatable=sum(c['baseline']['repeatable'] for c in output['cases']),
                          fused_repeatable=sum(c['fused']['repeatable'] for c in output['cases']),
                          variants_match=sum(c['variants_match'] for c in output['cases']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--prompts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:8157')
    asyncio.run(main(parser.parse_args()))
