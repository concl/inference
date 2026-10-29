"""Compare GDN kernels and capture Proton scopes after TTGIR optimization.

Run twice, with --implementation sequential and --implementation chunked.
Only CTA (0,0,0), warp 0 is recorded to keep loop-heavy traces manageable.
The installed kernels and the vLLM checkout are never modified.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.profiler as proton
from triton._C.libtriton import ir

from vllm.third_party.flash_linear_attention.ops.chunk import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_fwd,
)


class InstrumentTTGIR:
    def __init__(self, output):
        self.output = output

    @staticmethod
    def record(start, name, indent):
        direction = "start" if start else "end"
        return [f'{indent}proton.record {direction} "{name}"']

    def instrument_operations(self, text):
        lines = text.splitlines()
        result = []
        index = 0
        op_number = 0
        while index < len(lines):
            line = lines[index]
            indent = line[:len(line) - len(line.lstrip())]
            match = re.search(r"\b(tt\.load|tt\.store|tt\.dot|tt\.reduce|ttng\.mma|ttng\.warp_group_dot)\b", line)
            if match:
                kind = {
                    "tt.load": "load_issue", "tt.store": "store_issue",
                    "tt.reduce": "reduction", "tt.dot": "matmul_issue",
                }.get(match[1], "matmul_issue")
                name = f"{kind}_{op_number:03d}"
                op_number += 1
                result.extend(self.record(True, name, indent))
                result.append(line)
                # Reductions have a nested region; enclose the entire operation.
                depth = line.count("{") - line.count("}")
                while depth > 0:
                    index += 1
                    line = lines[index]
                    result.append(line)
                    depth += line.count("{") - line.count("}")
                result.extend(self.record(False, name, indent))
            else:
                result.append(line)
            index += 1
        return "\n".join(result) + "\n"

    def instrument(self, text):
        lines = text.splitlines()
        header = next(i for i, line in enumerate(lines) if "tt.func public @" in line)
        returns = [i for i, line in enumerate(lines) if "tt.return" in line]
        if len(returns) != 1:
            raise ValueError("Expected one inlined, void-returning Triton kernel")
        end = returns[0]
        body = "\n".join(lines[header+1:end])
        arguments = set(re.findall(r"%([\w.$]+)", lines[header]))
        # One uniform branch selects an instrumented body for CTA zero. Other
        # CTAs execute a copy of the same optimized IR without recording.
        plain = re.sub(r"%([\w.$]+)", lambda m: m[0] if m[1] in arguments else f"%plain_{m[1]}", body)
        predicate = [
            "    %gdn_profile_zero = arith.constant 0 : i32",
            "    %gdn_profile_x = tt.get_program_id x : i32",
            "    %gdn_profile_y = tt.get_program_id y : i32",
            "    %gdn_profile_z = tt.get_program_id z : i32",
            "    %gdn_profile_x0 = arith.cmpi eq, %gdn_profile_x, %gdn_profile_zero : i32",
            "    %gdn_profile_y0 = arith.cmpi eq, %gdn_profile_y, %gdn_profile_zero : i32",
            "    %gdn_profile_z0 = arith.cmpi eq, %gdn_profile_z, %gdn_profile_zero : i32",
            "    %gdn_profile_xy0 = arith.andi %gdn_profile_x0, %gdn_profile_y0 : i1",
            "    %gdn_profile_cta = arith.andi %gdn_profile_xy0, %gdn_profile_z0 : i1",
            "    scf.if %gdn_profile_cta {",
            '      proton.record start "kernel_body"',
        ]
        return "\n".join([
            *lines[:header+1], *predicate, self.instrument_operations(body),
            '      proton.record end "kernel_body"', "      scf.yield",
            "    } else {", plain, "      scf.yield", "    }",
            *lines[end:],
        ]) + "\n"

    def __call__(self, *args):
        if not args:
            digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            return ("gdn_intra_ttgir_" + digest, digest)
        _, stages, _, _, _ = args
        original = stages["ttgir"]

        def ttgir(module, metadata):
            module = original(module, metadata)
            source = self.instrument(str(module))
            name = re.search(r"tt.func public @([^\(]+)", source)[1]
            digest = hashlib.sha256(source.encode()).hexdigest()[:12]
            path = self.output / f"{name}-{digest}.ttgir"
            path.write_text(source)
            parsed = ir.parse_mlir_module(str(path), module.context)
            parsed.context = module.context
            return parsed

        stages["ttgir"] = ttgir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["sequential", "chunked"], required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--instrumented-warmups", type=int, default=1)
    parser.add_argument(
        "--instrumentation-buffer-bytes", type=int,
        help="Profiling buffer bytes per CTA; defaults to 16384 chunked or 524288 sequential.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Owning experiment's output directory; no shared default capture location.",
    )
    args = parser.parse_args()
    if args.instrumentation_buffer_bytes is not None and args.instrumentation_buffer_bytes <= 0:
        parser.error("--instrumentation-buffer-bytes must be positive")
    output = args.output_dir.resolve() / f"proton_{args.implementation}"
    (output / "ir").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1337)
    # One layer on a TP=2 worker: 8 query/key heads, 24 value heads.
    q = F.normalize(torch.randn(1, args.tokens, 8, 128, device="cuda", dtype=torch.bfloat16), dim=-1)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, args.tokens, 24, 128, device="cuda", dtype=torch.bfloat16)
    g = -F.softplus(torch.randn(1, args.tokens, 24, device="cuda"))
    beta = torch.sigmoid(torch.randn_like(g))
    initial = torch.zeros(1, 24, 128, 128, device="cuda")
    kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, scale=128**-0.5,
                  initial_state=initial, output_final_state=True)

    def sequential():
        return chunk_gated_delta_rule(**kwargs)

    def chunked():
        result = chunk_gated_delta_rule_fwd(**kwargs)
        return result[1], result[3]

    fn = sequential if args.implementation == "sequential" else chunked
    with torch.inference_mode():
        expected = fn()
        other = chunked() if args.implementation == "sequential" else sequential()
        torch.cuda.synchronize()
        errors = [(a.float()-b.float()).abs().max().item() for a,b in zip(expected, other)]
        torch.testing.assert_close(expected[0], other[0], atol=0.02, rtol=0.02)
        torch.testing.assert_close(expected[1], other[1], atol=0.02, rtol=0.02)
        baseline = triton.testing.do_bench(fn, warmup=100, rep=200)
        print(json.dumps({"implementation": args.implementation, "baseline_ms": baseline,
                          "output_state_max_abs_error": errors}), flush=True)

        # Instrument optimized GPU IR, avoiding changes to Triton loop pipelining.
        triton.knobs.runtime.add_stages_inspection_hook = InstrumentTTGIR(output / "ir")
        buffer_size = args.instrumentation_buffer_bytes
        if buffer_size is None:
            buffer_size = 16384 if args.implementation == "chunked" else 524288
        mode = proton.mode.Default(
            buffer_type="global", buffer_size=buffer_size,
            sampling_strategy="selective", sampling_options="0",
            optimizations="clock32,time_shift",
        )
        # Warm the actual instrumented binaries, not just the original kernels.
        # A separate session keeps compilation and first-use data out of intra.
        if args.instrumented_warmups > 0:
            print("Warming instrumented kernels...", flush=True)
            warmup = proton.start(str(output / "warmup"), backend="instrumentation", data="trace", mode=mode)
            try:
                for _ in range(args.instrumented_warmups):
                    fn()
                torch.cuda.synchronize()
            finally:
                proton.finalize(warmup)
        print("Recording warmed instrumented kernels...", flush=True)
        session = proton.start(str(output / "intra"), backend="instrumentation", data="trace", mode=mode)
        try:
            actual = fn()
            torch.cuda.synchronize()
        finally:
            proton.finalize(session)
            triton.knobs.runtime.add_stages_inspection_hook = None
        instrumented_errors = []
        for actual_tensor, expected_tensor in zip(actual, expected):
            instrumented_errors.append((actual_tensor.float()-expected_tensor.float()).abs().max().item())
            # Instrumentation can change register allocation and FP rounding.
            torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-3, rtol=1e-3)
        summary = dict(implementation=args.implementation, tokens=args.tokens,
                       qk_heads=8, value_heads=24, head_dim=128, seed=1337,
                       gpu=torch.cuda.get_device_name(), triton=triton.__version__,
                       baseline_ms=baseline, output_state_max_abs_error=errors,
                       instrumentation="post-optimization TTGIR", sampled_cta=[0,0,0],
                       sampled_warp=0, instrumented_validation_atol=1e-3,
                       instrumented_warmups=args.instrumented_warmups,
                       instrumentation_buffer_bytes_per_cta=buffer_size,
                       instrumented_output_state_max_abs_error=instrumented_errors)
        (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
