"""Run a bounded, inspectable queue of existing GPU experiments in a tmux session."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import yaml

REPO=Path(__file__).resolve().parent.parent
CACHE=Path('/home/lukezeng/.cache/huggingface/hub')
CHUNKED='58d8a013b2dd33adb99f75a1c7d12ce1babf8640'
SEQUENTIAL='db0084219d62a7b7b564adeedb9e6536cadcf0d8'


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def inventory():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {r[0]:r for r in csv.reader(text.splitlines(),skipinitialspace=True)}


def compute_processes():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return list(csv.reader(text.splitlines(),skipinitialspace=True))


def cached_model(name,revision=None):
    path=Path(name)
    if path.is_dir(): return path,None
    root=CACHE/('models--'+name.replace('/','--'))
    preferred=[]
    if revision: preferred.append(revision)
    else:
        if (root/'refs/main').exists(): preferred.append((root/'refs/main').read_text().strip())
        preferred += [p.name for p in sorted((root/'snapshots').glob('*'))]
    for rev in dict.fromkeys(preferred):
        folder=root/'snapshots'/rev
        if not (folder/'config.json').exists(): continue
        index=folder/'model.safetensors.index.json'
        if index.exists():
            names=set(json.loads(index.read_text())['weight_map'].values())
            if all((folder/name).is_file() for name in names): return folder,rev
        elif (folder/'model.safetensors').is_file(): return folder,rev
    raise FileNotFoundError(f'No complete cached checkpoint for {name}; queue will not silently download another revision')


def plan():
    jobs=[]
    for name in ['exp1','exp2','exp3','exp4']:
        jobs.append(dict(name=name,kind='serve',gpus=4,config=f'experiments/{name}/config.yaml',
            benchmarks=[f'experiments/{name}/benchmark-random.yaml',f'experiments/{name}/benchmark-sharegpt.yaml'],
            kernel=SEQUENTIAL if name=='exp4' else CHUNKED,
            prefix_caching=True))
    for name,kernel in [('gdn_sequential_initial',SEQUENTIAL),('gdn_sequential_torch_warm',SEQUENTIAL),('gdn_comparison',CHUNKED)]:
        jobs.append(dict(name=name,kind='serve',gpus=2,config=f'experiments/{name}/config.yaml',
            benchmarks=[f'experiments/{name}/benchmark-sharegpt.yaml'],kernel=kernel,
            prefix_caching=False,profile=True))
    jobs.append(dict(name='gdn_fused_kkt_solve',kind='fusion',gpus=2))
    for variant in ['sequential','chunked']:
        jobs.append(dict(name=f'gdn_comparison_proton_{variant}',kind='proton',gpus=1,variant=variant))
    return jobs


class Campaign:
    def __init__(self,args):
        self.args=args;self.out=args.output_dir.resolve();self.out.mkdir(parents=True,exist_ok=False)
        self.children=[];self.known_roots=set();self.logs=[];self.locks=[]
        self.status=dict(status='starting',host=socket.gethostname(),pid=os.getpid(),
            started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu_pool=args.gpus,
            jobs=[dict(**job,status='pending') for job in plan() if not args.only or job['name'] in args.only],active_processes=[])
        save(self.out/'status.json',self.status)
        versions={n:importlib.metadata.version(n) for n in ['vllm','torch','triton','transformers']}
        save(self.out/'environment.json',dict(versions=versions,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            queue_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            warning='Historical exact dependency/model/dataset revisions are incomplete; effective configs and all deviations are saved'))
        (self.out/'queue-source.py').write_bytes(Path(__file__).read_bytes())
        save(self.out/'plan.json',self.status['jobs'])
        dataset=REPO.parent/'benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json'
        digest=hashlib.sha256()
        with dataset.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        save(self.out/'dataset.json',dict(path=str(dataset),sha256=digest.hexdigest()))

    def update(self,**values):
        self.status.update(values);save(self.out/'status.json',self.status)

    def stop_requested(self):
        if (self.out/'STOP_NOW').exists(): raise KeyboardInterrupt('STOP_NOW requested')

    def acquire(self,count):
        deadline=time.monotonic()+self.args.wait_hours*3600
        while time.monotonic()<deadline:
            self.stop_requested()
            if (self.out/'STOP').exists(): raise KeyboardInterrupt('STOP requested before next job')
            devices=inventory();occupied={r[0] for r in compute_processes()}
            chosen=[]
            for gpu in self.args.gpus:
                row=devices[gpu]
                if row[1] in occupied or int(row[2])>=256 or int(row[3])>5: continue
                handle=open(f'/tmp/inference-replication-{row[1]}.lock','a')
                try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: handle.close();continue
                self.locks.append(handle);chosen.append(gpu)
                if len(chosen)==count:
                    self.update(status='running',selected_gpus=chosen)
                    return chosen
            self.release()
            self.update(status='waiting_for_gpus',needed_gpus=count,last_inventory=devices)
            time.sleep(15)
        raise TimeoutError('Timed out waiting for idle GPUs')

    def release(self):
        for handle in self.locks: handle.close()
        self.locks=[]

    def check_contention(self,gpus):
        uuids={inventory()[g][1] for g in gpus}
        for uuid,pid in compute_processes():
            if uuid not in uuids: continue
            ancestor=int(pid)
            while ancestor>1 and ancestor not in self.known_roots:
                try: ancestor=int(Path(f'/proc/{ancestor}/stat').read_text().rsplit(')',1)[1].split()[1])
                except FileNotFoundError: ancestor=1
            if ancestor not in self.known_roots:
                raise RuntimeError(f'Foreign GPU process {pid} appeared; stopping only campaign processes')

    def launch(self,command,log,env):
        handle=log.open('w');self.logs.append(handle)
        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        self.children.append(child);self.known_roots.add(child.pid)
        self.status['active_processes'].append(dict(pid=child.pid,command=command,log=str(log)))
        self.update();return child

    def wait(self,child,gpus,timeout=3600):
        deadline=time.monotonic()+timeout
        while child.poll() is None:
            self.stop_requested();self.check_contention(gpus)
            if time.monotonic()>deadline: raise TimeoutError('Child process exceeded timeout')
            time.sleep(3)
        if child.returncode: raise RuntimeError(f'Child {child.pid} exited {child.returncode}')

    def cleanup(self):
        for child in reversed(self.children):
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: continue
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait()
        for handle in self.logs: handle.close()
        self.children=[];self.known_roots=set();self.logs=[];self.release()
        self.update(active_processes=[])

    def kernel_env(self,job,directory,gpus):
        source=directory/'chunk.py'
        commit=job.get('kernel',SEQUENTIAL)
        source.write_text(subprocess.check_output(['git','-C',str(REPO.parent/'vllm'),'show',
            commit+':vllm/third_party/flash_linear_attention/ops/chunk.py'],text=True))
        env={k:v for k,v in os.environ.items() if not k.startswith(('GDN_','FLA_'))}
        env.update(PYTHONPATH=str(REPO/'experiments/_shared/gdn/import_overlay'),GDN_CHUNK_SOURCE=str(source),
            CUDA_VISIBLE_DEVICES=','.join(inventory()[g][1] for g in gpus),
            TRITON_CACHE_DIR=str(directory/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(directory/'inductor_cache'),
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1')
        return env

    def serve(self,job,directory,gpus):
        env=self.kernel_env(job,directory,gpus)
        source=REPO/job['config'];config=yaml.safe_load(source.read_text())
        (directory/'source-config.yaml').write_bytes(source.read_bytes())
        model,revision=cached_model(config['model'],config.get('revision'))
        adaptations=[f'Resolved model to cached snapshot {model.name}; this does not prove historical revision identity']
        config.update(model=str(model),revision=revision,**{'tokenizer-revision':revision,'port':self.args.port})
        config.pop('enable-prefix-caching',None);config.pop('no-enable-prefix-caching',None)
        config['enable-prefix-caching' if job['prefix_caching'] else 'no-enable-prefix-caching']=True
        if job['name'] in ['exp1','exp4']: adaptations.append('Explicitly enabled prefix cache to match recorded server, despite historical YAML false')
        if job.get('profile'):
            profiler=config.pop('profiler_config',config.pop('profiler-config',{}))
            profiler['torch_profiler_dir']=str(directory/'torch')
            config['profiler-config']=profiler
        path=directory/'server.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
        save(directory/'adaptations.json',adaptations)
        save(directory/'kernel.json',dict(commit=job['kernel'],sha256=hashlib.sha256((directory/'chunk.py').read_bytes()).hexdigest()))
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',self.args.port))==0: raise RuntimeError('Campaign server port is occupied')
        server=self.launch([str(REPO/'.venv/bin/inference'),'serve','--config',str(path),'--gpus',','.join(gpus)],directory/'server.log',env)
        deadline=time.monotonic()+1800
        while True:
            self.stop_requested();self.check_contention(gpus)
            if server.poll() is not None: raise RuntimeError('Server exited before readiness')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.args.port}/health',timeout=2): break
            except Exception:
                if time.monotonic()>deadline: raise TimeoutError('Server readiness timeout')
                time.sleep(3)
        results=[]
        repeats=1 if job.get('profile') else self.args.repeats
        for repeat in range(repeats):
            for source_name in job['benchmarks']:
                original=REPO/source_name;bench=yaml.safe_load(original.read_text())
                label=f'{original.stem}-r{repeat}'
                (directory/(label+'-source.yaml')).write_bytes(original.read_bytes())
                tokenizer=bench.get('tokenizer')
                bench['tokenizer']=str(model) if tokenizer in [None,config.get('model'),yaml.safe_load(source.read_text())['model']] else str(cached_model(tokenizer)[0])
                bench.update(**{'base-url':f'http://127.0.0.1:{self.args.port}',
                    'result-dir':str(directory),'result-filename':label+'.json','save-result':True})
                bpath=directory/(label+'.yaml');bpath.write_text(yaml.safe_dump(bench,sort_keys=False))
                process=self.launch([str(REPO/'.venv/bin/inference'),'bench','--config',str(bpath)],directory/(label+'.log'),env)
                self.wait(process,gpus)
                data=json.loads((directory/(label+'.json')).read_text())
                if data.get('completed')!=bench['num-prompts'] or data.get('failed',0)!=0:
                    raise RuntimeError(f'Incomplete benchmark {label}')
                results.append(dict(file=label+'.json',completed=data['completed'],
                    mean_ttft_ms=data.get('mean_ttft_ms'),output_throughput=data.get('output_throughput')))
        if job.get('profile') and not list((directory/'torch').glob('*.pt.trace.json*')):
            raise RuntimeError('Profile requested but no trace exported')
        save(directory/'verification.json',dict(status='passed',results=results,
            note='Success/workload verification; matching historical performance is not asserted'))

    def other(self,job,directory,gpus):
        env=self.kernel_env(job,directory,gpus)
        if job['kind']=='fusion':
            command=[str(REPO/'.venv/bin/python'),str(REPO/'experiments/gdn_fused_kkt_solve/run_pair.py'),
                '--model-config',str(REPO/'experiments/gdn_fused_kkt_solve/qwen35-4b-config.json'),
                '--gpus',*gpus,'--output-dir',str(directory/'run')]
        else:
            command=[str(REPO/'.venv/bin/python'),str(REPO/'experiments/_shared/gdn/profile_intra.py'),
                '--implementation',job['variant'],'--tokens','2048','--instrumented-warmups','1',
                '--instrumentation-buffer-bytes','524288' if job['variant']=='sequential' else '8192',
                '--output-dir',str(directory/'capture')]
        child=self.launch(command,directory/'run.log',env);self.wait(child,gpus,7200)
        save(directory/'verification.json',dict(status='passed',command=command,
            note='Harness exited successfully; inspect numerical checks and raw timing/trace limitations'))

    def run(self):
        for job in self.status['jobs']:
            if (self.out/'STOP').exists(): self.update(status='stopped_after_job');return
            directory=self.out/job['name'];directory.mkdir()
            job.update(status='waiting',artifact_directory=str(directory));self.update(current_job=job['name'])
            try:
                gpus=self.acquire(job['gpus']);job['status']='running';self.update()
                if job['kind']=='serve':self.serve(job,directory,gpus)
                else:self.other(job,directory,gpus)
                job['status']='passed'
            except KeyboardInterrupt:
                job['status']='interrupted';self.update(status='interrupted');return
            except Exception as exc:
                job.update(status='failed',error=repr(exc));save(directory/'failure.json',dict(error=repr(exc)))
            finally:
                self.cleanup();save(self.out/'status.json',self.status)
            print(json.dumps(dict(job=job['name'],status=job['status'],error=job.get('error'))),flush=True)
        self.update(status='complete',ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs='+',default=['0','1','2','3'])
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--port',type=int,default=8160)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--wait-hours',type=float,default=8)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--only',nargs='+',help='Run only these plan entries in their existing order')
    parser.add_argument('--status',type=Path,help='Print a compact status report without starting work')
    args=parser.parse_args()
    if args.status:
        value=json.loads(args.status.read_text())
        print(f"Host: {value['host']} | controller PID: {value['pid']} | state: {value['status']}")
        print('Current job:',value.get('current_job'),'| GPUs:',value.get('selected_gpus',[]))
        for job in value['jobs']:print(f"{job['status']:12s} {job['name']} {job.get('error','')}")
        for process in value['active_processes']:print('PID',process['pid'],'log:',process['log'])
        return
    if not args.output_dir:parser.error('--output-dir is required unless --status is used')
    selected=[job for job in plan() if not args.only or job['name'] in args.only]
    if args.only and set(args.only)-{job['name'] for job in plan()}:parser.error('Unknown --only entry')
    if len(set(args.gpus))!=len(args.gpus) or any(job['gpus']>len(args.gpus) for job in selected):
        parser.error('GPU pool must contain distinct IDs and enough GPUs for every selected job')
    if args.plan_only: print(json.dumps(selected,indent=2));return
    def terminate(*_): raise KeyboardInterrupt('Controller termination requested')
    signal.signal(signal.SIGTERM,terminate)
    Campaign(args).run()


if __name__=='__main__':main()
