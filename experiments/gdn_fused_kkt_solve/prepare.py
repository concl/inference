"""Extract one pinned upstream kernel and adapt its gate/precision contract."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "provenance.json").read_text())
source = (ROOT / "artifacts/sources/upstream_chunk_fwd.py").read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == "chunk_gated_delta_rule_fwd_kkt_solve_kernel")
lines = source.splitlines(keepends=True)
kernel = "".join(lines[min(d.lineno for d in node.decorator_list)-1:node.end_lineno])
header = f'''# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Derived from FLA commit {manifest['upstream_commit']}.
# MIT license: see UPSTREAM_LICENSE.
# Kernel body is unchanged. Integration adaptations:
# - use ordinary Triton autotuning instead of FLA's cache wrapper;
# - bind exp2 to vLLM's natural exponential (inputs remain natural-log gates);
# - match vLLM's FLA_TRIL_PRECISION, default ieee, instead of upstream tf32.
import os
import torch
import triton
import triton.language as tl
from vllm.third_party.flash_linear_attention.ops.op import exp as exp2
fla_cache_autotune = triton.autotune
autotune_cache_kwargs = {{}}
SOLVE_TRIL_DOT_PRECISION = tl.constexpr(os.environ.get('FLA_TRIL_PRECISION', 'ieee'))

'''
wrapper = '''
def fused_kkt_solve(k, beta, g, cu_seqlens=None, chunk_indices=None):
    B, T, H, K = k.shape
    HV = beta.shape[-1]
    if H <= 0 or HV % H or not k.is_contiguous():
        raise ValueError("Requires contiguous keys and an integral value/key head ratio")
    BT, BC = 64, 16
    if cu_seqlens is not None and chunk_indices is None:
        raise ValueError("Pass precomputed chunk_indices for packed inputs")
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    # Upstream zeros the full allocation: the kernel only writes lower blocks.
    A = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * HV)](
        k=k, g=g, beta=beta, A=A, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC,
    )
    return A
'''
candidate = header + kernel + "\n" + wrapper
(ROOT / "candidate.py").write_text(candidate)

baseline = (ROOT / "artifacts/sources/baseline_ops/chunk.py").read_text()
node = next(n for n in ast.parse(baseline).body if isinstance(n, ast.FunctionDef)
            and n.name == 'chunk_gated_delta_rule_fwd')
body = ast.get_source_segment(baseline, node)
body_tree = ast.parse(body).body[0]
assigns = [n for n in body_tree.body if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == 'A' for t in n.targets)]
assert len(assigns) == 2
oldlines = body.splitlines(keepends=True)
newbody = ''.join(oldlines[:assigns[0].lineno-1]) + '''    A = fused_kkt_solve(
        k=k, beta=beta, g=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
''' + ''.join(oldlines[assigns[1].end_lineno:])
newbody = newbody.replace('def chunk_gated_delta_rule_fwd(', 'def full_fused(', 1)
imports = '''# Experiment adapter; all other stages use the installed vLLM implementation.
import torch
from candidate import fused_kkt_solve
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL

'''
(ROOT / "candidate_full.py").write_text(imports + newbody + '\n')
diff = ''.join(difflib.unified_diff(body.splitlines(True), newbody.splitlines(True),
                                  fromfile='baseline_forward', tofile='fused_forward'))
(ROOT / "adapter.diff").write_text(diff)
manifest['adaptations'] = [
    'Kernel body byte-for-byte extracted from pinned upstream source',
    'exp2 symbol bound to vLLM natural exponential; no gate-conversion kernel',
    'SOLVE_TRIL_DOT_PRECISION matches vLLM FLA_TRIL_PRECISION (ieee by default)',
    'FLA cache autotuner replaced by triton.autotune; upstream configurations retained',
    'Only KKT and triangular solve replaced in copied full forward function',
]
for name in ['candidate.py', 'candidate_full.py', 'adapter.diff', 'UPSTREAM_LICENSE']:
    manifest.setdefault('experiment_sha256', {})[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
(ROOT / 'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(diff)
