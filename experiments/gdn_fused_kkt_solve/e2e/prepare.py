"""Snapshot the local GDN module and add an explicit chunked/fused dispatcher."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

import yaml

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT.parent/'provenance.json').read_text())
    for item in manifest['baseline_files'].values():
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError(f'Baseline source changed: {item["path"]}')
    source = Path(manifest['baseline_files']['chunk.py']['path']).read_text()
    source += '\nfrom gdn_e2e_worker import install as _install_e2e\n_install_e2e(globals())\n'
    (args.output_dir/'chunk_overlay.py').write_text(source)
    for name in ['candidate.py', 'candidate_full.py', 'provenance.json']:
        shutil.copy2(ROOT.parent/name, args.output_dir/name)
    for name in ['gdn_e2e_worker.py', 'run.py', 'analyze.py', 'serve.yaml',
                 'check_outputs.py', 'test_analysis.py', 'summarize.py']:
        shutil.copy2(ROOT/name, args.output_dir/name)
    shutil.copy2(ROOT.parent.parent/'_shared/gdn/import_overlay/sitecustomize.py', args.output_dir/'sitecustomize.py')
    config=yaml.safe_load((ROOT/'serve.yaml').read_text())
    environment=dict(model_id=config['model'], model_revision=config['revision'],
        gpu_uuid=args.gpu_uuid, host=platform.node(), python=platform.python_version(),
        git_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        versions={n:importlib.metadata.version(n) for n in
                  ['vllm','torch','triton','transformers','compressed-tensors','numpy','aiohttp']},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in args.output_dir.iterdir() if p.is_file()},
        statistics_plan=dict(confirmation_pairs=24,requests_per_arm=4,
            prompt_tokens_approx=8192,output_tokens=32,concurrency=1,
            endpoints=['HTTP completion latency','HTTP TTFT'],
            multiplicity='Holm correction across two endpoints',
            unit='paired block, not individual request',pilot_excluded=True))
    (args.output_dir.parent/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    print(args.output_dir)


if __name__ == '__main__':
    main()
