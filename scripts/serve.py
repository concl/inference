"""Launch the vLLM server from `configs/serve.yaml`.

The serving definition lives in YAML, in vLLM's own `vllm serve --config` format:
keys are flag names without the leading `--`, nested maps are passed as JSON, and
command-line arguments override the file. So this module is only a launcher.

It does the parts YAML cannot:

  * pick the `vllm` next to this interpreter, so no venv activation is needed
  * choose GPUs from `--gpus` / `CUDA_VISIBLE_DEVICES`
  * put that same bin directory on PATH, because FlashInfer's JIT shells out to
    the bare `ninja` command and otherwise dies with FileNotFoundError
  * replace this process with vLLM via `os.execvpe`

Anything this parser does not recognise -- including `--help` -- is forwarded to
`vllm serve`, so vLLM's own option list stays authoritative.

Run it as `inference serve`, or standalone as `python -m scripts.serve`.
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "serve.yaml"
DEFAULT_GPUS = "0,1,2,3"


def vllm_binary():
    """Prefer the vllm next to this interpreter, so no venv activation needed."""
    candidate = Path(sys.executable).with_name("vllm")
    return str(candidate) if candidate.exists() else "vllm"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="inference serve",
        # --help falls through to `vllm serve --help`, which owns the options.
        add_help=False,
        description=(
            "Launch the vLLM server from a YAML config. Unrecognised arguments "
            "are forwarded to `vllm serve` and override the config file."
        ),
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="vLLM serve-args YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_GPUS),
        help="GPUs to expose, as CUDA_VISIBLE_DEVICES (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the command instead of running it",
    )
    return parser


def build_argv(config, passthrough):
    """`vllm serve --config <file>` plus everything the caller passed through."""
    return [vllm_binary(), "serve", "--config", str(config), *passthrough]


def run(argv=None):
    """Launch vLLM, replacing this process. Returns early only to dry-run."""
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)

    if not Path(args.config).is_file():
        parser.error(f"config file not found: {args.config}")

    command = build_argv(args.config, passthrough)

    if args.dry_run:
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpus)} " if args.gpus else ""
        print(prefix + shlex.join(command))
        return 0

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    # FlashInfer compiles kernels at startup and shells out to the bare `ninja`
    # command, so the interpreter's bin directory must be on PATH.
    bin_dir = str(Path(sys.executable).parent)
    if bin_dir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    sys.stdout.flush()
    os.execvpe(command[0], command, env)  # never returns


if __name__ == "__main__":
    sys.exit(run())
