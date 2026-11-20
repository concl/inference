"""Compare vLLM's chunked GDN with a contract-matched upstream FLA port.

No installation, model server, checkout mutation, or GPU clock changes required.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
SEED = 1337


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def nvidia_rows(query, kind='gpu'):
    text = subprocess.check_output(
        ['nvidia-smi', f'--query-{kind}={query}', '--format=csv,noheader,nounits'],
        text=True, timeout=15)
    return list(csv.reader(text.splitlines(), skipinitialspace=True))


def check_exclusive(uuid):
    rows = nvidia_rows('gpu_uuid,pid,process_name,used_gpu_memory', 'compute-apps')
    others = [r for r in rows if len(r) >= 2 and r[0] == uuid
              and r[1].isdigit() and int(r[1]) != os.getpid()]
    if others:
        raise RuntimeError(f'GPU contention detected; stopping measurements: {others}')
    return rows


def telemetry(uuid):
    rows = nvidia_rows('uuid,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw')
    return next((r for r in rows if r[0] == uuid), None)


DEFAULT_HEAD_SHAPES = [(4, 12), (8, 24), (16, 48)]


def model_profile(config_path, tp_sizes):
    """Read model geometry without loading weights or creating a GPU context."""
    config = json.loads(config_path.read_text())
    text = config.get('text_config', config)
    hq, hv = text['linear_num_key_heads'], text['linear_num_value_heads']
    if text['linear_key_head_dim'] != 128 or text['linear_value_head_dim'] != 128:
        raise ValueError('This harness currently requires key/value dimensions of 128')
    if not tp_sizes or len(set(tp_sizes)) != len(tp_sizes):
        raise ValueError('TP sizes must be nonempty and distinct')
    if any(tp < 1 or hq % tp or hv % tp for tp in tp_sizes):
        raise ValueError('TP sizes must be positive and divide both head counts')
    return dict(config_path=str(config_path.absolute()),
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                config=config, key_heads=hq, value_heads=hv,
                key_head_dim=128, value_head_dim=128, tp_sizes=tp_sizes,
                head_shapes=[(hq // tp, hv // tp) for tp in tp_sizes])


def correctness_cases(smoke=False, head_shapes=None):
    head_shapes = head_shapes or DEFAULT_HEAD_SHAPES
    # Preserve the original boundary/state coverage for the primary geometry.
    result = _correctness_cases(smoke)
    primary_hq, primary_hv = head_shapes[0]
    result = [dict(c, hq=primary_hq, hv=primary_hv) for c in result
              if not c['name'].startswith('heads')]
    if not smoke:
        for hq, hv in head_shapes[1:]:
            result.append(dict(name=f'heads{hq}_{hv}', lengths=[129], packed=False,
                               hq=hq, hv=hv, initial='random', gate=0.01))
    return result


def _correctness_cases(smoke=False):
    if smoke:
        return [dict(name='boundary65', lengths=[65], packed=False, hq=4, hv=12,
                     initial='random', gate=0.01),
                dict(name='ragged_strong', lengths=[1, 17, 65], packed=True,
                     hq=4, hv=12, initial='random', gate=1.0)]
    result = []
    for i, length in enumerate([1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129]):
        result.append(dict(name=f'boundary{length}', lengths=[length], packed=False,
                           hq=4, hv=12, initial=['none', 'zero', 'random'][i % 3], gate=0.01))
    for gate in [0.01, 1.0]:
        for initial in ['none', 'zero', 'random']:
            result.append(dict(name=f'ragged_g{gate}_{initial}', lengths=[1, 17, 63, 65, 129],
                               packed=True, hq=4, hv=12, initial=initial, gate=gate))
    for length in [17, 65, 129]:
        result.append(dict(name=f'strong{length}', lengths=[length], packed=False,
                           hq=4, hv=12, initial='random', gate=1.0))
    for hq, hv in [(8, 24), (16, 48)]:
        result.append(dict(name=f'heads{hq}_{hv}', lengths=[129], packed=False,
                           hq=hq, hv=hv, initial='random', gate=0.01))
    return result


def performance_cases(smoke=False, head_shapes=None):
    head_shapes = head_shapes or DEFAULT_HEAD_SHAPES
    primary_hq, primary_hv = head_shapes[0]
    if smoke:
        shapes = [(primary_hq, primary_hv, [256], False),
                  (primary_hq, primary_hv, [2048], False)]
    else:
        shapes = [(primary_hq, primary_hv, [t] * n, False)
                  for n in [1, 4] for t in [64, 256, 2048]]
        shapes += [(primary_hq, primary_hv, [65], False),
                   (primary_hq, primary_hv, [8192], False)]
        shapes += [(hq, hv, [t], False)
                   for hq, hv in head_shapes[1:] for t in [2048, 8192]]
        shapes += [(primary_hq, primary_hv, [17, 63, 129, 2048], True),
                   (primary_hq, primary_hv, [256, 1023, 2048, 4097], True)]
    return [dict(name=f'h{hq}_{hv}_n{len(ls)}_t{sum(ls)}_{"packed" if packed else "fixed"}',
                 lengths=ls, packed=packed, hq=hq, hv=hv, initial='random', gate=0.01)
            for hq, hv, ls, packed in shapes]


def make_inputs(case):
    import torch
    import torch.nn.functional as F
    ls, packed = case['lengths'], case['packed']
    b, t = (1, sum(ls)) if packed else (len(ls), ls[0])
    if not packed:
        assert len(set(ls)) == 1
    hq, hv, dim = case['hq'], case['hv'], 128
    q = F.normalize(torch.randn(b, t, hq, dim, device='cuda'), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(b, t, hq, dim, device='cuda'), dim=-1).to(torch.bfloat16)
    v = torch.randn(b, t, hv, dim, device='cuda', dtype=torch.bfloat16)
    g = -case['gate'] * F.softplus(torch.randn(b, t, hv, device='cuda'))
    beta = torch.sigmoid(torch.randn_like(g))
    initial = None
    if case['initial'] != 'none':
        initial = torch.zeros(len(ls), hv, dim, dim, device='cuda')
        if case['initial'] == 'random':
            initial.normal_(0, 0.05)
    cu = indices = offsets = None
    if packed:
        cumulative = [0]
        chunk_rows, chunk_totals = [], [0]
        for seq, length in enumerate(ls):
            cumulative.append(cumulative[-1] + length)
            count = (length + 63) // 64
            chunk_rows.extend((seq, i) for i in range(count))
            chunk_totals.append(chunk_totals[-1] + count)
        cu = torch.tensor(cumulative, device='cuda', dtype=torch.int64)
        indices = torch.tensor(chunk_rows, device='cuda', dtype=torch.int64)
        offsets = torch.tensor(chunk_totals, device='cuda', dtype=torch.int64)
    return dict(q=q, k=k, v=v, g=g, beta=beta, scale=dim ** -0.5,
                initial_state=initial, output_final_state=True,
                cu_seqlens=cu, chunk_indices=indices, chunk_offsets=offsets)


def cpu_reference(inputs, case):
    """Independent FP64 token recurrence; no FLA kernels or chunk algebra."""
    import torch
    q, k, v, g, beta = [inputs[n].detach().cpu().double() for n in ['q', 'k', 'v', 'g', 'beta']]
    hv, vd, kd = v.shape[-2], v.shape[-1], k.shape[-1]
    mapping = torch.arange(hv) // (hv // k.shape[-2])
    out = torch.empty_like(v)
    initial = inputs['initial_state']
    states = (initial.detach().cpu().double().clone() if initial is not None
              else torch.zeros(len(case['lengths']), hv, vd, kd, dtype=torch.float64))
    cursor = 0
    for seq, length in enumerate(case['lengths']):
        b, start = (0, cursor) if case['packed'] else (seq, 0)
        state = states[seq]
        for t in range(start, start + length):
            kt = k[b, t, mapping]
            state.mul_(g[b, t].exp()[:, None, None])
            predicted = torch.einsum('hvk,hk->hv', state, kt)
            delta = (v[b, t] - predicted) * beta[b, t, :, None]
            state.add_(delta[:, :, None] * kt[:, None, :])
            out[b, t] = torch.einsum('hvk,hk->hv', state, q[b, t, mapping]) * inputs['scale']
        cursor += length
    return out, states


def inverse_reference(inputs, case, g_cumsum):
    """Independent FP64 unit-lower-triangular solve for each actual chunk."""
    import torch
    k = inputs['k'].detach().cpu().double()
    beta = inputs['beta'].detach().cpu().double()
    gates = g_cumsum.detach().cpu().double()
    hv = beta.shape[-1]
    mapping = torch.arange(hv) // (hv // k.shape[-2])
    out = torch.zeros(*beta.shape, 64, dtype=torch.float64)
    cursor = 0
    for seq, length in enumerate(case['lengths']):
        b, start = (0, cursor) if case['packed'] else (seq, 0)
        for local in range(0, length, 64):
            a, end = start + local, start + min(local + 64, length)
            keys = k[b, a:end, mapping].permute(1, 0, 2)
            gg = gates[b, a:end].T
            kk = keys @ keys.transpose(-1, -2)
            lower = torch.tril(kk * beta[b, a:end].T[:, :, None]
                               * (gg[:, :, None] - gg[:, None, :]).exp(), diagonal=-1)
            ident = torch.eye(end-a, dtype=torch.float64).expand(hv, -1, -1)
            solved = torch.linalg.solve_triangular(lower + ident, ident, upper=False, unitriangular=True)
            out[b, a:end, :, :end-a] = solved.permute(1, 0, 2)
        cursor += length
    return out


def error_metrics(actual, expected, max_abs_limit=0.02, relative_l2_limit=0.01):
    import torch
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(passed=False, finite=False, max_abs=None, relative_l2=None)
    delta = a - b
    maximum = delta.abs().max().item()
    relative = (delta.norm() / b.norm().clamp_min(1e-30)).item()
    return dict(passed=maximum <= max_abs_limit and relative <= relative_l2_limit,
                finite=True, max_abs=maximum, relative_l2=relative,
                max_abs_limit=max_abs_limit, relative_l2_limit=relative_l2_limit)


def measure(fn, iterations):
    import torch
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        output = fn()
    end.record()
    end.synchronize()
    wall_us = (time.perf_counter() - wall) * 1e6 / iterations
    elapsed = begin.elapsed_time(end) * 1000 / iterations
    del output
    return dict(gpu_timeline_us=elapsed, wall_us=wall_us)


def summarize(samples):
    vals = [s['gpu_timeline_us'] for s in samples]
    return dict(median_us=statistics.median(vals), min_us=min(vals), max_us=max(vals),
                median_wall_us=statistics.median(s['wall_us'] for s in samples), samples=samples)


def compare_timings(functions, mode, rounds, iterations, rng, uuid):
    import torch
    for fn in functions.values():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    graphs, retained = {}, {}
    if mode == 'cuda_graph':
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for fn in functions.values():
                for _ in range(3):
                    fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for name, fn in functions.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                retained[name] = fn()
            graphs[name] = graph
        functions = {name: graph.replay for name, graph in graphs.items()}
    samples = {name: [] for name in functions}
    orders, sensors = [], []
    for _ in range(rounds):
        check_exclusive(uuid)
        names = list(functions)
        rng.shuffle(names)
        orders.append(names)
        sensors.append(telemetry(uuid))
        for name in names:
            samples[name].append(measure(functions[name], iterations))
        check_exclusive(uuid)
    results = {name: summarize(rows) for name, rows in samples.items()}
    return dict(mode=mode, **results,
                speedup=results['baseline']['median_us'] / results['fused']['median_us'],
                paired_round_speedups=[a['gpu_timeline_us']/b['gpu_timeline_us']
                                       for a, b in zip(samples['baseline'], samples['fused'])],
                orders=orders, telemetry=sensors)


def collect_autotune(module, name):
    obj = getattr(module, name)
    for _ in range(4):
        if hasattr(obj, 'cache') and isinstance(obj.cache, dict):
            return {str(k): str(v) for k, v in obj.cache.items()}
        obj = getattr(obj, 'fn', None)
        if obj is None:
            break
    return {}


def render_summary(report):
    env = report.get('environment', {})
    cases = report['correctness']
    passed = sum(c['passed'] for c in cases)
    lines = ['# GDN KKT/solve fusion experiment', '',
             f"Status: **{report['status']}**. Reference checks: {passed}/{len(cases)} passed.", '',
             f"GPU: {env.get('gpu_name', 'unknown')} ({env.get('gpu_uuid', 'unknown')}).",
             f"Software: `{env.get('versions', {})}`.",
             f"Triangular solve: `{env.get('precision', 'unknown')}` in both implementations.", '',
             'The candidate replaces only KKT and triangular solve with an adapted pinned upstream kernel.',
             'KKT arithmetic also changes (BF16 dot before beta scaling); gains cannot be assigned solely to fewer launches.', '',
             'Outputs and final states are checked against an independent CPU FP64 recurrence; inverse matrices',
             'are checked against an independent FP64 triangular solve. Fixed gates: max abs <=0.02, relative L2 <=0.01.', '']
    if report.get('error'):
        lines += ['Run failed: ' + report['error'], '', 'Do not use this incomplete run as a performance claim.', '']
    if report['performance']:
        lines += ['| Shape | Region | Execution | Baseline us | Fused us | Speedup |',
                  '|---|---|---|---:|---:|---:|']
        for row in report['performance']:
            for m in row['measurements']:
                lines.append(f"| {row['case']['name']} | {m['region']} | {m['mode']} | "
                             f"{m['baseline']['median_us']:.3f} | {m['fused']['median_us']:.3f} | {m['speedup']:.3f}x |")
        lines += ['', f"Medians from {report['arguments']['rounds']} randomized paired rounds, "
                  f"{report['arguments']['iterations']} calls/sample. Raw samples, ordering, and",
                  'telemetry are in results.json. Values below 1x are regressions.', '']
    lines += ['No model-level throughput or accuracy claim. Per-rank TP shapes run on one GPU.',
              'Graph measurements do not establish that vLLM captures this prefill path.',
              'Source provenance, per-file hashes, arguments, error metrics, and autotuning choices are in results.json.', '']
    return '\n'.join(lines)


def run(args, report):
    import torch
    import triton
    import importlib
    check_exclusive(args.gpu_uuid)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    baseline = importlib.import_module('vllm.third_party.flash_linear_attention.ops.chunk')
    kkt = importlib.import_module('vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt')
    solve = importlib.import_module('vllm.third_party.flash_linear_attention.ops.solve_tril')
    from candidate import fused_kkt_solve
    import candidate
    from candidate_full import full_fused
    manifest = json.loads((ROOT/'provenance.json').read_text())
    for name, expected in manifest['baseline_files'].items():
        digest = hashlib.sha256(Path(expected['path']).read_bytes()).hexdigest()
        if digest != expected['sha256']:
            raise RuntimeError(f'Installed baseline changed after snapshot: {name}')
    report['environment'] = dict(
        versions={n: importlib.metadata.version(n) for n in ['torch', 'triton', 'vllm']},
        python=sys.version, hostname=platform.node(), pid=os.getpid(), seed=SEED,
        gpu_name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        gpu_uuid=args.gpu_uuid, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        torch_cuda=torch.version.cuda, baseline_module=baseline.__file__,
        baseline_entry='chunk_gated_delta_rule_fwd (original chunked function; custom sequential wrapper bypassed)',
        precision=solve.FLA_TRIL_PRECISION, candidate_precision=str(candidate.SOLVE_TRIL_DOT_PRECISION),
        kkt_arithmetic='Upstream computes BF16 KKT then scales beta; baseline scales beta before a promoted dot. Differences are validated, not attributed solely to fusion.',
        driver_version=nvidia_rows('uuid,driver_version'),
        telemetry_columns=['uuid','name','utilization.gpu','memory.used_MiB','temperature.gpu_C','clocks.sm_MHz','clocks.mem_MHz','power.draw_W'],
        harness_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                        for name in ['benchmark.py','candidate.py','candidate_full.py','wait_and_run.py']},
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        gates='natural log in both implementations', dtype='bfloat16', state_dtype='float32',
        provenance=manifest, model_profile=args.model_profile,
        initial_telemetry=telemetry(args.gpu_uuid),
        environment_overrides={k:v for k,v in os.environ.items() if k.startswith(('FLA_', 'GDN_', 'TRITON_'))})
    if baseline.FLA_CHUNK_SIZE != 64:
        raise RuntimeError('This candidate requires chunk size 64')
    save(args.output_dir/'results.json', report)

    def stage(inputs):
        return baseline.chunk_local_cumsum(inputs['g'], chunk_size=64,
                    cu_seqlens=inputs['cu_seqlens'], chunk_indices=inputs['chunk_indices'])

    def intra_functions(inputs, gates):
        kw = dict(k=inputs['k'], beta=inputs['beta'], g=gates,
                  cu_seqlens=inputs['cu_seqlens'], chunk_indices=inputs['chunk_indices'])
        def old():
            a = baseline.chunk_scaled_dot_kkt_fwd(**kw, output_dtype=torch.float32)
            return baseline.solve_tril(a, cu_seqlens=kw['cu_seqlens'],
                                       chunk_indices=kw['chunk_indices'], output_dtype=inputs['k'].dtype)
        return dict(baseline=old, fused=lambda: fused_kkt_solve(**kw))

    with torch.inference_mode():
        for case in correctness_cases(args.suite == 'smoke', args.head_shapes):
            check_exclusive(args.gpu_uuid)
            print('CORRECTNESS', case['name'], flush=True)
            inputs = make_inputs(case)
            original_initial = (inputs['initial_state'].clone()
                                if inputs['initial_state'] is not None else None)
            gates = stage(inputs)
            funcs = intra_functions(inputs, gates)
            a_ref = inverse_reference(inputs, case, gates)
            output_ref, state_ref = cpu_reference(inputs, case)
            old = baseline.chunk_gated_delta_rule_fwd(**inputs)
            new = full_fused(**inputs)
            torch.cuda.synchronize()
            checks = dict(
                baseline_inverse=error_metrics(funcs['baseline'](), a_ref),
                fused_inverse=error_metrics(funcs['fused'](), a_ref),
                baseline_output=error_metrics(old[1], output_ref),
                fused_output=error_metrics(new[1], output_ref),
                baseline_state=error_metrics(old[3], state_ref),
                fused_state=error_metrics(new[3], state_ref),
                output_delta=error_metrics(new[1], old[1]),
                state_delta=error_metrics(new[3], old[3]),
            )
            if original_initial is not None and not torch.equal(original_initial, inputs['initial_state']):
                raise RuntimeError('Forward modified initial_state')
            passed = all(c['passed'] for c in checks.values())
            report['correctness'].append(dict(case=case, passed=passed, checks=checks))
            save(args.output_dir/'results.json', report)
            if not passed:
                raise RuntimeError(f'Correctness failed for {case["name"]}: {checks}')
            del inputs, gates, funcs, a_ref, output_ref, state_ref, old, new
        report['correctness_passed'] = True
        save(args.output_dir/'results.json', report)
        if not args.correctness_only:
            rng = random.Random(SEED)
            for case in performance_cases(args.suite == 'smoke', args.head_shapes):
                check_exclusive(args.gpu_uuid)
                print('BENCHMARK', case['name'], flush=True)
                inputs = make_inputs(case)
                gates = stage(inputs)
                old, new = baseline.chunk_gated_delta_rule_fwd(**inputs), full_fused(**inputs)
                checks = dict(output=error_metrics(new[1], old[1]), state=error_metrics(new[3], old[3]))
                if not all(c['passed'] for c in checks.values()):
                    raise RuntimeError(f'Long-shape parity failed: {case}: {checks}')
                del old, new
                funcs = intra_functions(inputs, gates)
                full = dict(baseline=lambda: baseline.chunk_gated_delta_rule_fwd(**inputs),
                            fused=lambda: full_fused(**inputs))
                row = dict(case=case, parity=checks, measurements=[])
                for region, functions in [('kkt_solve', funcs), ('full_gdn', full)]:
                    for mode in ['eager', 'cuda_graph']:
                        timed = compare_timings(functions, mode, args.rounds, args.iterations, rng, args.gpu_uuid)
                        timed['region'] = region
                        row['measurements'].append(timed)
                        print(json.dumps(dict(case=case['name'], region=region, mode=mode,
                            baseline_us=timed['baseline']['median_us'], fused_us=timed['fused']['median_us'],
                            speedup=timed['speedup'])), flush=True)
                report['performance'].append(row)
                save(args.output_dir/'results.json', report)
                del inputs, gates, funcs, full, row
                gc.collect()
                torch.cuda.empty_cache()
        report['autotune'] = dict(
            baseline_kkt=collect_autotune(kkt, 'chunk_scaled_dot_kkt_fwd_kernel'),
            baseline_solve=collect_autotune(solve, 'merge_16x16_to_64x64_inverse_kernel'),
            fused=collect_autotune(candidate, 'chunk_gated_delta_rule_fwd_kkt_solve_kernel'))
        report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--suite', choices=['smoke', 'full'], default='full')
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--model-config', type=Path,
                        help='Local model config.json; uses its GDN geometry without loading weights')
    parser.add_argument('--tp-sizes', nargs='+', type=int, default=[1, 2],
                        help='Per-rank geometries to test with --model-config; no distributed execution')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    try:
        args.model_profile = model_profile(args.model_config, args.tp_sizes) if args.model_config else None
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    args.head_shapes = args.model_profile['head_shapes'] if args.model_profile else DEFAULT_HEAD_SHAPES
    if args.describe:
        print(json.dumps(dict(model_profile=args.model_profile,
                              correctness=correctness_cases(args.suite=='smoke', args.head_shapes),
                              performance=performance_cases(args.suite=='smoke', args.head_shapes)), indent=2))
        return
    if not args.output_dir or not args.gpu_uuid:
        parser.error('--gpu-uuid and --output-dir are required for GPU work')
    if args.rounds < 3 or args.iterations < 1:
        parser.error('Use at least 3 rounds and one iteration')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', correctness=[], performance=[],
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  limitations=['Single-GPU operator benchmark; TP shapes are per-rank equivalents, not distributed runs',
                               'Synthetic BF16 inputs, normalized q/k; no model-level throughput or accuracy claim',
                               'Graph mode is an isolated captured operator, not proof vLLM captures this path',
                               'Eager CUDA-event timelines can include GPU idle gaps caused by host submission',
                               'No cache flushing or clock locking; randomized paired rounds and telemetry recorded'])
    try:
        run(args, report)
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save(args.output_dir/'results.json', report)
        (args.output_dir/'summary.md').write_text(render_summary(report))
    print('COMPLETE', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
