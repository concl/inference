"""Compile the adapted kernel for SM86 without creating a GPU context."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # vLLM's op module normally imports hardware-discovery utilities. Substitute
    # only the default exp symbol in this isolated codegen process; the generated
    # candidate file remains unchanged. Runtime benchmarks import actual vLLM.
    op = types.ModuleType('vllm.third_party.flash_linear_attention.ops.op')
    op.exp = tl.exp
    sys.modules[op.__name__] = op
    os.environ['FLA_TRIL_PRECISION'] = 'ieee'
    candidate = importlib.import_module('candidate')
    autotuner = candidate.chunk_gated_delta_rule_fwd_kkt_solve_kernel.fn
    fn = autotuner.fn
    signatures = dict(k='*bf16', g='*fp32', beta='*fp32', A='*bf16',
                      cu_seqlens='*i64', chunk_indices='*i64', T='i32')
    result = dict(target='cuda:sm86', gpu_work_executed=False, triton=triton.__version__,
                  note='Offline compilation only; does not validate numerical behavior or speed', kernels=[])
    for varlen in [False, True]:
        for config in autotuner.configs:
                bk, warps = config.kwargs['BK'], config.num_warps
                label = f'varlen{int(varlen)}_bk{bk}_warps{warps}'
                print('COMPILE', label, flush=True)
                constants = dict(H=4, HV=12, K=128, BT=64, BC=16, BK=bk,
                                 USE_G=True, IS_VARLEN=varlen)
                source = ASTSource(fn, signatures, constexprs=constants)
                kernel = triton.compile(source, target=GPUTarget('cuda', 86, 32),
                                        options=dict(num_warps=warps, num_stages=config.num_stages))
                (args.output_dir/f'{label}.ptx').write_text(kernel.asm['ptx'])
                result['kernels'].append(dict(label=label, constants=constants,
                    metadata={k:str(v) for k,v in kernel.metadata._asdict().items()}))
                (args.output_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    result['status'] = 'passed'
    (args.output_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: all 12 configurations compiled without a GPU context', flush=True)


if __name__ == '__main__':
    main()
