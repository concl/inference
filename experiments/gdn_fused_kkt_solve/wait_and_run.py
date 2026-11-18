"""Wait for an unused GPU, then run the isolated GDN comparison."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from benchmark import nvidia_rows, save

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=12)
    parser.add_argument('--idle-seconds', type=int, default=60)
    parser.add_argument('--poll-seconds', type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'artifacts/watcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = args.output_dir/'status.json'
    status = dict(status='waiting_for_idle_gpu', pid=os.getpid(),
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  deadline_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time()+args.wait_hours*3600)),
                  idle_policy='No compute processes, <=5% utilization, <256 MiB used, continuously for 60 seconds',
                  output_dir=str(args.output_dir))
    stable = {}
    deadline = time.monotonic() + args.wait_hours*3600
    try:
        chosen = None
        while time.monotonic() < deadline:
            gpu_rows = nvidia_rows('index,uuid,name,utilization.gpu,memory.used')
            proc_rows = nvidia_rows('gpu_uuid,pid', 'compute-apps')
            occupied = {r[0] for r in proc_rows if len(r)>1 and r[1].isdigit()}
            now = time.monotonic()
            for row in gpu_rows:
                index, uuid, name, utilization, memory = row
                idle = (uuid not in occupied and int(utilization)<=5 and int(memory)<256
                        and 'RTX 3090' in name)
                if idle:
                    stable.setdefault(uuid, now)
                    if now-stable[uuid] >= args.idle_seconds:
                        chosen = dict(index=int(index), uuid=uuid, name=name)
                        break
                else:
                    stable.pop(uuid, None)
            status.update(last_check_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          last_gpus=gpu_rows, idle_duration_seconds={u:round(now-t,1) for u,t in stable.items()})
            save(status_path, status)
            if chosen:
                break
            time.sleep(args.poll_seconds)
        if chosen is None:
            status['status'] = 'timed_out_waiting_for_gpu'
            save(status_path, status)
            return
        status.update(status='running', gpu=chosen)
        save(status_path, status)
        env = os.environ.copy()
        # Keep caches and compiler artifacts owned by this experiment/run.
        for key in list(env):
            if key.startswith(('FLA_', 'GDN_')):
                env.pop(key)
        env.update(CUDA_VISIBLE_DEVICES=chosen['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   TRITON_CACHE_DIR=str(args.output_dir/'triton_cache'),
                   TORCHINDUCTOR_CACHE_DIR=str(args.output_dir/'inductor_cache'),
                   FLA_TRIL_PRECISION='ieee', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1')
        for suite in ['smoke', 'full']:
            command = [sys.executable, '-u', str(ROOT/'benchmark.py'),
                       '--gpu-uuid', chosen['uuid'], '--output-dir', str(args.output_dir/suite),
                       '--suite', suite, '--rounds', '7', '--iterations', '20']
            status.update(active_suite=suite, command=command)
            save(status_path, status)
            with (args.output_dir/f'{suite}.log').open('w') as log:
                result = subprocess.run(command, env=env, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=7200)
            if result.returncode:
                status.update(status='failed', failed_suite=suite, returncode=result.returncode)
                save(status_path, status)
                return
        status['status'] = 'complete'
        status['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save(status_path, status)
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc())
        save(status_path, status)
        raise


if __name__ == '__main__':
    main()
