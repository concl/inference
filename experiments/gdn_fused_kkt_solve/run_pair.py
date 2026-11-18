"""Run independent model-shaped GDN smoke/full suites on two idle GPUs."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmark import model_profile, nvidia_rows, save

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--gpus', nargs=2, required=True, help='Two physical GPU indices')
    parser.add_argument('--tp-sizes', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    model_profile(args.model_config, args.tp_sizes)
    rows = nvidia_rows('index,uuid,name,utilization.gpu,memory.used')
    devices = {r[0]: r for r in rows}
    occupied = {r[0] for r in nvidia_rows('gpu_uuid,pid', 'compute-apps')
                if len(r) > 1 and r[1].isdigit()}
    if len(set(args.gpus)) != 2:
        parser.error('Select two distinct GPUs')
    for index in args.gpus:
        row = devices[index]
        if row[1] in occupied or int(row[3]) > 5 or int(row[4]) >= 256:
            parser.error(f'GPU {index} is not idle: {row}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Coordinate with the existing experiment watcher, without reserving other users' GPUs.
    lock = (ROOT / 'artifacts/watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = args.output_dir / 'source'
    snapshot.mkdir()
    for name in ['benchmark.py', 'candidate.py', 'candidate_full.py',
                 'run_pair.py', 'wait_and_run.py', 'provenance.json']:
        shutil.copy2(ROOT / name, snapshot / name)
    shutil.copy2(args.model_config, snapshot / 'model-config.json')
    status = dict(status='running', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  pid=os.getpid(), devices=[devices[g] for g in args.gpus],
                  model_profile=model_profile(args.model_config, args.tp_sizes), stages=[])
    path = args.output_dir / 'status.json'
    save(path, status)
    children = []
    logs = []
    try:
        for suite in ['smoke', 'full']:
            children, logs = [], []
            for index in args.gpus:
                directory = args.output_dir / f'gpu{index}'
                directory.mkdir(exist_ok=True)
                env = {k: v for k, v in os.environ.items()
                       if not k.startswith(('FLA_', 'GDN_'))}
                env.update(CUDA_VISIBLE_DEVICES=devices[index][1], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           TRITON_CACHE_DIR=str(directory / 'triton_cache'),
                           TORCHINDUCTOR_CACHE_DIR=str(directory / 'inductor_cache'),
                           FLA_TRIL_PRECISION='ieee', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           PYTHONUNBUFFERED='1')
                command = [sys.executable, '-u', str(snapshot / 'benchmark.py'),
                           '--gpu-uuid', devices[index][1], '--output-dir', str(directory / suite),
                           '--model-config', str(snapshot / 'model-config.json'),
                           '--tp-sizes', *map(str, args.tp_sizes), '--suite', suite,
                           '--rounds', '7', '--iterations', '20']
                log = (directory / f'{suite}.log').open('w')
                logs.append(log)
                child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                         stderr=subprocess.STDOUT)
                children.append(child)
                status['stages'].append(dict(gpu=index, suite=suite, pid=child.pid,
                                             command=command, returncode=None))
                save(path, status)
                print(f'GPU {index}: {suite} started (PID {child.pid})', flush=True)
            failed = False
            for child, stage in zip(children, status['stages'][-2:]):
                stage['returncode'] = child.wait(timeout=7200)
                failed |= stage['returncode'] != 0
                save(path, status)
                print(f"GPU {stage['gpu']}: {suite} exited {stage['returncode']}", flush=True)
            for log in logs:
                log.close()
            if failed:
                raise RuntimeError(f'{suite} failed; inspect GPU logs and results.json')
        status['status'] = 'complete'
    except BaseException as exc:
        status.update(status='failed', error=str(exc))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for log in logs:
            log.close()
        status['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save(path, status)


if __name__ == '__main__':
    main()
