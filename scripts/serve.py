"""Options and defaults for the vLLM server.

This module is the single definition of how the model is served. `README.md`
points here rather than repeating the options.

Run it as `inference serve`, or standalone as `python -m scripts.serve`.
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_SERVED_NAME = "Qwen3.8-27B"
DEFAULT_GPUS = "0,1,2,3"
DEFAULT_TP_SIZE = 4
DEFAULT_DTYPE = "bfloat16"
DEFAULT_GPU_MEM_UTIL = 0.95
DEFAULT_MAX_MODEL_LEN = 262144
DEFAULT_MAX_NUM_SEQS = 32
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_REASONING_PARSER = "qwen3"
DEFAULT_TOOL_CALL_PARSER = "qwen3_xml"


def env_default(env_name, fallback):
    """Environment variable beats the built-in default; a flag beats both."""
    return os.environ.get(env_name, fallback)


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
        "--host", args.host,
        "--port", str(args.port),
        "--reasoning-parser", args.reasoning_parser,
    ]
    if args.tool_call_parser != "none":
        argv += ["--enable-auto-tool-choice", "--tool-call-parser", args.tool_call_parser]
    if args.api_key:
        argv += ["--api-key", args.api_key]
    return argv


def run(argv=None):
    """Launch vLLM, replacing this process. Returns early only to dry-run."""
    args = build_parser().parse_args(argv)
    binary = vllm_binary()
    args_list = vllm_args(args)

    if args.dry_run:
        env = f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpus)} " if args.gpus else ""
        print(env + shlex.join([binary, *args_list]))
        return 0

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    sys.stdout.flush()
    os.execvpe(binary, [binary, *args_list], env)  # never returns


if __name__ == "__main__":
    sys.exit(run())
