"""Options and defaults for the vLLM server.

This module is the single definition of how the model is served. `README.md`
points here rather than repeating the options.

The default checkpoint is the INT8 W8A16 build of Qwen3.8-27B. That needs more
than a model path: its MTP head, its GDN linear-attention layers and its fp8 KV
cache each need their own flag, so those defaults live here rather than in a
separate launcher.

Run it as `inference serve`, or standalone as `python -m scripts.serve`.
"""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

DEFAULT_MODEL = "lued/Qwen3.8-27B-INT8-W8A16-MTP"
DEFAULT_SERVED_NAME = "Qwen3.8-27B"
DEFAULT_GPUS = "0,1,2,3"
DEFAULT_TP_SIZE = 4
DEFAULT_DTYPE = "bfloat16"
DEFAULT_GPU_MEM_UTIL = 0.92
DEFAULT_MAX_MODEL_LEN = 262144
DEFAULT_MAX_NUM_SEQS = 4
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_REASONING_PARSER = "qwen3"
DEFAULT_TOOL_CALL_PARSER = "qwen3_coder"

# Checkpoint wiring rather than tuning. Qwen3.8 interleaves GDN
# linear-attention layers with full attention and carries an MTP head, so these
# are what make the checkpoint work at all.
DEFAULT_KV_CACHE_DTYPE = "fp8_e4m3"
DEFAULT_MAMBA_CACHE_MODE = "align"
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_NUM_SPECULATIVE_TOKENS = 3
DEFAULT_PREFIX_MATCH_UNIT = 16
DEFAULT_CHAT_TEMPLATE_KWARGS = '{"enable_thinking":true,"preserve_thinking":true}'
DEFAULT_OVERRIDE_GENERATION_CONFIG = (
    '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,'
    '"repetition_penalty":1.0,"presence_penalty":0.0}'
)


def env_default(env_name, fallback):
    """Environment variable beats the built-in default; a flag beats both."""
    return os.environ.get(env_name, fallback)


def env_flag(env_name, fallback):
    """Same precedence as env_default, for on/off options."""
    raw = os.environ.get(env_name)
    if raw is None:
        return fallback
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def vllm_binary():
    """Prefer the vllm next to this interpreter, so no venv activation needed."""
    candidate = Path(sys.executable).with_name("vllm")
    return str(candidate) if candidate.exists() else "vllm"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="inference serve",
        description=(
            "Launch the vLLM server. This replaces the current process, so run "
            "it under screen or systemd for a long-lived server."
        ),
        epilog="Every option also reads an environment variable of the same name.",
    )
    parser.add_argument(
        "--model", default=env_default("MODEL", DEFAULT_MODEL),
        help="HuggingFace repo to load (default: %(default)s)",
    )
    parser.add_argument(
        "--served-model-name", default=env_default("SERVED_MODEL_NAME", DEFAULT_SERVED_NAME),
        help="name clients request (default: %(default)s)",
    )
    parser.add_argument(
        "--gpus", default=env_default("CUDA_VISIBLE_DEVICES", DEFAULT_GPUS),
        help="GPUs to use (default: %(default)s)",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=env_default("TENSOR_PARALLEL_SIZE", DEFAULT_TP_SIZE),
        help="GPUs per copy of the model (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype", default=env_default("DTYPE", DEFAULT_DTYPE),
        help="weight dtype (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float,
        default=env_default("GPU_MEMORY_UTILIZATION", DEFAULT_GPU_MEM_UTIL),
        help="fraction of each GPU vLLM may use (default: %(default)s)",
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int,
        default=env_default("MAX_NUM_BATCHED_TOKENS", DEFAULT_MAX_NUM_BATCHED_TOKENS),
        help="prompt tokens per scheduler step (default: %(default)s)",
    )
    parser.add_argument(
        "--kv-cache-dtype", default=env_default("KV_CACHE_DTYPE", DEFAULT_KV_CACHE_DTYPE),
        help=(
            "KV cache storage dtype. fp8_e4m3 halves the bytes per token and works "
            "on Ampere because the attention kernel upcasts before the math, so no "
            "fp8 tensor cores are involved (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--mamba-cache-mode", default=env_default("MAMBA_CACHE_MODE", DEFAULT_MAMBA_CACHE_MODE),
        help="required by the Qwen3.8 GDN layers (default: %(default)s)",
    )
    parser.add_argument(
        "--num-speculative-tokens", type=int,
        default=env_default("NUM_SPECULATIVE_TOKENS", DEFAULT_NUM_SPECULATIVE_TOKENS),
        help="MTP draft depth; 0 disables speculative decoding (default: %(default)s)",
    )
    parser.add_argument(
        "--prefix-match-unit", type=int,
        default=env_default("PREFIX_MATCH_UNIT", DEFAULT_PREFIX_MATCH_UNIT),
        help="prefix-cache hash block size (default: %(default)s)",
    )
    parser.add_argument(
        "--default-chat-template-kwargs",
        default=env_default("DEFAULT_CHAT_TEMPLATE_KWARGS", DEFAULT_CHAT_TEMPLATE_KWARGS),
        help="JSON chat-template kwargs applied to every request (default: %(default)s)",
    )
    parser.add_argument(
        "--override-generation-config",
        default=env_default("OVERRIDE_GENERATION_CONFIG", DEFAULT_OVERRIDE_GENERATION_CONFIG),
        help="JSON sampling defaults (default: %(default)s)",
    )
    parser.add_argument(
        "--prefix-caching", action=argparse.BooleanOptionalAction,
        default=env_flag("PREFIX_CACHING", True),
        help=(
            "reuse KV across shared prefixes. Only safe while the vllm#48375 patch "
            "is applied to the installed vllm -- without it an MTP prefix-cache hit "
            "can resume from a rejected-draft state and serve wrong output silently"
        ),
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction,
        default=env_flag("TRUST_REMOTE_CODE", True),
        help="the Qwen3.8 architecture needs its own remote code (default: %(default)s)",
    )
    parser.add_argument(
        "--disable-custom-all-reduce", action=argparse.BooleanOptionalAction,
        default=env_flag("DISABLE_CUSTOM_ALL_REDUCE", True),
        help="the 3090s here have no NVLink, and the model card validates this for INT8",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=env_default("MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN),
        help="max context length (default: %(default)s)",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=env_default("MAX_NUM_SEQS", DEFAULT_MAX_NUM_SEQS),
        help="max concurrent sequences (default: %(default)s)",
    )
    parser.add_argument(
        "--host", default=env_default("HOST", DEFAULT_HOST),
        help="bind address (default: %(default)s)",
    )
    parser.add_argument(
        "--port", type=int, default=env_default("PORT", DEFAULT_PORT),
        help="bind port (default: %(default)s)",
    )
    parser.add_argument(
        "--reasoning-parser", default=env_default("REASONING_PARSER", DEFAULT_REASONING_PARSER),
        help="reasoning parser (default: %(default)s)",
    )
    parser.add_argument(
        "--tool-call-parser", default=env_default("TOOL_CALL_PARSER", DEFAULT_TOOL_CALL_PARSER),
        help="tool parser; 'none' disables tool calling (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key", default=env_default("API_KEY", None),
        help="require this bearer token (default: none)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the command instead of running it",
    )
    return parser


def vllm_args(args):
    """Turn parsed arguments into the argv tail passed to `vllm serve`."""
    argv = [
        "serve", args.model,
        "--served-model-name", args.served_model_name,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--mamba-cache-mode", args.mamba_cache_mode,
        "--host", args.host,
        "--port", str(args.port),
        "--reasoning-parser", args.reasoning_parser,
    ]
    argv += ["--prefix-match-unit", str(args.prefix_match_unit)]
    argv += ["--default-chat-template-kwargs", args.default_chat_template_kwargs]
    argv += ["--override-generation-config", args.override_generation_config]
    if args.trust_remote_code:
        argv.append("--trust-remote-code")
    if args.disable_custom_all_reduce:
        argv.append("--disable-custom-all-reduce")
    argv.append("--enable-prefix-caching" if args.prefix_caching else "--no-enable-prefix-caching")
    if args.num_speculative_tokens:
        argv += [
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": args.num_speculative_tokens}),
        ]
    if args.tool_call_parser != "none":
        argv += ["--enable-auto-tool-choice", "--tool-call-parser", args.tool_call_parser]
    if args.api_key:
        argv += ["--api-key", args.api_key]
    return argv


def run(argv=None):
    """Launch vLLM, replacing this process. Returns early only to dry-run."""
    # Anything this parser does not define is forwarded to `vllm serve`
    # untouched, so vLLM-only flags (--kv-cache-dtype, --speculative-config,
    # --mamba-cache-mode, ...) work without being restated here. `inference
    # serve` depends on this: it hands its unrecognised arguments straight in.
    args, passthrough = build_parser().parse_known_args(argv)
    binary = vllm_binary()
    args_list = vllm_args(args) + passthrough

    if args.dry_run:
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpus)} " if args.gpus else ""
        print(prefix + shlex.join([binary, *args_list]))
        return 0

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    # FlashInfer compiles kernels at startup and shells out to the bare `ninja`
    # command. vllm_binary() picks the vllm beside this interpreter so that no
    # venv activation is needed, so put that same bin directory on PATH as well
    # -- otherwise the workers die with FileNotFoundError: 'ninja'.
    bin_dir = str(Path(sys.executable).parent)
    if bin_dir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    sys.stdout.flush()
    os.execvpe(binary, [binary, *args_list], env)  # never returns


if __name__ == "__main__":
    sys.exit(run())
