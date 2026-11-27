"""Run and retain a labeled PD replication using accessible model/source inputs."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import yaml

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def save(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)


def http(url,data=None,timeout=30):
    request=urllib.request.Request(url,data=None if data is None else json.dumps(data).encode(),
                                   headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        return response.read().decode()


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs=2,required=True)
    parser.add_argument('--model',default='axteam/Qwen3.5-4B-AWQ-4bit')
    parser.add_argument('--revision',default='ba968c07f40fa0bce733c2264d860d2fa671cfcf')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--num-prompts',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--cache-dir',type=Path,help='Reuse compiler caches while preserving fresh measurement artifacts')
    args=parser.parse_args()
    args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    out=args.output_dir
    def terminate(*_): raise KeyboardInterrupt('Controller termination requested')
    signal.signal(signal.SIGTERM,terminate)
    snapshot=out/'source';snapshot.mkdir()
    for name in ['replicate.py','reference_proxy.py','reference-proxy-provenance.json']:
        shutil.copy2(HERE/name,snapshot/name)
    for port in [8100,8200,8300,5600,5601]:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',port))==0:
                raise RuntimeError(f'Port {port} already occupied')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices={line.split(',')[0].strip():[v.strip() for v in line.split(',')] for line in inventory.splitlines()}
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if len(set(args.gpus))!=2: raise ValueError('Two distinct GPUs required')
    for gpu in args.gpus:
        if devices[gpu][1] in occupied or int(devices[gpu][2])>=256 or int(devices[gpu][3])>5:
            raise RuntimeError(f'GPU {gpu} is occupied')
    deps=HERE/'artifacts/runtime_deps'
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(REPO/'experiments/_shared/gdn/import_overlay'),str(deps)])
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',UCX_TLS='all',UCX_NET_DEVICES='all',
               VLLM_KV_CACHE_LAYOUT='HND',VLLM_SSM_CONV_STATE_LAYOUT='DS',
               TRITON_CACHE_DIR=str((args.cache_dir or out)/'triton_cache'),
               TORCHINDUCTOR_CACHE_DIR=str((args.cache_dir or out)/'inductor_cache'))
    kernel=out/'original_chunk.py'
    kernel.write_text(subprocess.check_output(['git','-C',str(REPO.parent/'vllm'),'show',
        '58d8a013b2dd33adb99f75a1c7d12ce1babf8640:vllm/third_party/flash_linear_attention/ops/chunk.py'],text=True))
    env['GDN_CHUNK_SOURCE']=str(kernel)
    state=dict(status='preparing',pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               gpu_indices=args.gpus,gpu_uuids=[devices[g][1] for g in args.gpus],
               model=args.model,revision=args.revision,replication_type='adapted_model_and_reference_proxy',
               adaptations=['Cached Qwen3.5-4B AWQ replaces unreadable Qwen3.8-27B AutoRound checkpoint',
                            'Reference proxy with explicit handoff validation replaces absent original proxy',
                            'No unknown no_mm_warmup overlay; native multimodal warmup remains enabled',
                            'Original chunked GDN module selected explicitly; educational sequential path bypassed',
                            'GPU memory utilization 0.70, image limit 1, video limit 0, compilation disabled'],
               children=[],commands=[],checks=[],benchmarks=[])
    save(out/'status.json',state)
    (out/'gpu-inventory.csv').write_text(inventory)
    (out/'gpu-processes-before.csv').write_text(occupied)
    processes=[]; logs=[]

    def launch(command,label,child_env):
        log=(out/f'{label}.log').open('w'); logs.append(log)
        process=subprocess.Popen(command,env=child_env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process)
        state['children'].append(dict(label=label,pid=process.pid))
        state['commands'].append(command);save(out/'status.json',state)
        return process

    def ready(port,process,path='/v1/models'):
        for _ in range(900):
            if process.poll() is not None: raise RuntimeError(f'Server on {port} exited {process.returncode}')
            try:
                http(f'http://127.0.0.1:{port}{path}',timeout=2)
                return
            except Exception: time.sleep(2)
        raise TimeoutError(f'Server on {port} was not ready in 30 minutes')

    def bench(label,port,prompts):
        config=dict(backend='openai-chat',**{'base-url':f'http://127.0.0.1:{port}',
            'endpoint':'/v1/chat/completions','model':'pd-qwen4b','tokenizer':args.model,
            'dataset-name':'hf','dataset-path':'lmarena-ai/vision-arena-bench-v0.1',
            'hf-name':'lmarena-ai/vision-arena-bench-v0.1','hf-split':'train','hf-output-len':128,
            'num-prompts':prompts,'num-warmups':1,'max-concurrency':2,'seed':0,'temperature':0.,
            'no-stream':True,'percentile-metrics':'ttft,tpot,itl','metric-percentiles':'50,95',
            'save-result':True,'result-dir':str(out),'result-filename':label+'.json'})
        path=out/(label+'.yaml');path.write_text(yaml.safe_dump(config,sort_keys=False))
        command=[str(REPO/'.venv/bin/inference'),'bench','--config',str(path)]
        state['commands'].append(command);save(out/'status.json',state)
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(command,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
        result=json.loads((out/(label+'.json')).read_text())
        if result.get('completed')!=prompts or result.get('failed',0)!=0:
            raise RuntimeError(f'Incomplete benchmark {label}')
        state['benchmarks'].append(dict(label=label,port=port,completed=result['completed'],result_file=label+'.json'))
        save(out/'status.json',state)

    try:
        subprocess.run([sys.executable,'-c','import datasets,nixl; print("Benchmark and transfer imports passed")'],
                       env=env,check=True,stdout=(out/'dependency-check.log').open('w'),stderr=subprocess.STDOUT)
        (out/'versions.json').write_text(subprocess.check_output([sys.executable,'-c',
          'import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ["vllm","torch","triton","transformers","nixl","nixl-cu13","datasets","pandas","pyarrow","multiprocess","xxhash"]},indent=2))'],env=env,text=True))
        for label,gpu,port,role,side in [('prefill',args.gpus[0],8100,'kv_producer',5600),('decode',args.gpus[1],8200,'kv_consumer',5601)]:
            config={'model':args.model,'revision':args.revision,'served-model-name':'pd-qwen4b',
                    'dtype':'bfloat16','tensor-parallel-size':1,'gpu-memory-utilization':.70,
                    'max-model-len':8192,'max-num-seqs':2,'max-num-batched-tokens':2048,
                    'no-enable-prefix-caching':True,'host':'127.0.0.1','port':port,'seed':0,
                    'limit-mm-per-prompt':{'image':1,'video':0},'enforce-eager':True,
                    'additional-config':{'gdn_prefill_backend':'triton'},
                    'default-chat-template-kwargs':{'enable_thinking':False},
                    'kv-transfer-config':{'kv_connector':'NixlConnector','kv_role':role,
                       'kv_load_failure_policy':'fail','kv_connector_extra_config':{'backends':['UCX']}}}
            if args.profile:
                config['profiler-config']={'profiler':'torch','torch_profiler_dir':str(out/label/'torch'),
                     'ignore_frontend':True,'torch_profiler_with_stack':False,'torch_profiler_record_shapes':True,
                     'torch_profiler_with_memory':False,'torch_profiler_use_gzip':False}
            path=out/(label+'.yaml');path.write_text(yaml.safe_dump(config,sort_keys=False))
            child=env|{'CUDA_VISIBLE_DEVICES':devices[gpu][1],'VLLM_NIXL_SIDE_CHANNEL_PORT':str(side)}
            launch([str(REPO/'.venv/bin/inference'),'serve','--config',str(path),'--gpus',devices[gpu][1]],label,child)
        state['status']='starting_servers';save(out/'status.json',state)
        ready(8100,processes[0]);ready(8200,processes[1])
        proxy=launch([sys.executable,str(HERE/'reference_proxy.py'),'--host','127.0.0.1','--port','8300',
            '--prefiller-hosts','127.0.0.1','--prefiller-ports','8100','--decoder-hosts','127.0.0.1','--decoder-ports','8200'],'proxy',env)
        ready(8300,proxy,'/healthcheck')
        state['status']='checking_transfer';save(out/'status.json',state)
        for i,text in enumerate(['What is two plus two? Answer briefly.','Name one primary color.']):
            payload=dict(model='pd-qwen4b',messages=[{'role':'user','content':text}],temperature=0,seed=0,max_tokens=16,stream=False,
                         chat_template_kwargs={'enable_thinking':False})
            baseline=json.loads(http('http://127.0.0.1:8100/v1/chat/completions',payload,timeout=300))
            disagg=json.loads(http('http://127.0.0.1:8300/v1/chat/completions',payload,timeout=300))
            save(out/f'correctness-{i}.json',dict(prompt=text,baseline=baseline,disaggregated=disagg))
            matched=baseline['choices'][0]['message']['content']==disagg['choices'][0]['message']['content']
            state['checks'].append(dict(prompt=i,exact_output_match=matched));save(out/'status.json',state)
            if not matched: raise RuntimeError('PD/ordinary serving output mismatch')
        for label,port in [('prefill',8100),('decode',8200)]:
            (out/(label+'-metrics-before.txt')).write_text(http(f'http://127.0.0.1:{port}/metrics'))
        state['status']='benchmarking';save(out/'status.json',state)
        for repeat in range(args.repeats):
            bench(f'vision-baseline-r{repeat}',8100,args.num_prompts)
            bench(f'vision-pd-r{repeat}',8300,args.num_prompts)
        if args.profile:
            state['status']='profiling';save(out/'status.json',state)
            for port in [8100,8200]: http(f'http://127.0.0.1:{port}/start_profile',{},timeout=60)
            bench('vision-trace',8300,2)
            for port in [8100,8200]: http(f'http://127.0.0.1:{port}/stop_profile',{},timeout=900)
            if not list((out/'prefill/torch').glob('*.pt.trace.json')) or not list((out/'decode/torch').glob('*.pt.trace.json')):
                raise RuntimeError('One or both profiler traces are missing')
        for label,port in [('prefill',8100),('decode',8200)]:
            (out/(label+'-metrics-after.txt')).write_text(http(f'http://127.0.0.1:{port}/metrics'))
        state['status']='complete'
    except BaseException as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        save(out/'status.json',state)
        for process in reversed(processes):stop(process)
        for log in logs:log.close()
        state['ended_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        save(out/'status.json',state)


if __name__=='__main__':
    main()
