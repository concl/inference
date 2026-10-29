"""Run vLLM's serving benchmark with native YAML parsing and CLI overrides.

Using a single parser avoids the installed CLI's nested `bench serve --config`
argument-ordering bug. The benchmark implementation remains vLLM's own.
"""

import shlex
import sys

from scripts.serve import vllm_binary


def run(argv=None):
    from vllm.benchmarks.serve import add_cli_args, main
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(prog="inference bench")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate arguments and print the equivalent vllm command.",
    )
    add_cli_args(parser)
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_args)
    if args.dry_run:
        expanded = (
            parser._pull_args_from_config(raw_args)
            if "--config" in raw_args else raw_args
        )
        expanded = [arg for arg in expanded if arg != "--dry-run"]
        print(shlex.join([vllm_binary(), "bench", "serve", *expanded]))
        return 0

    main(args)
    return 0


if __name__ == "__main__":
    sys.exit(run())
