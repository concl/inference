
"""The `inference` command line.

Only this module parses arguments that belong to the CLI itself. Subcommands
call plain functions directly -- no subprocesses. The one exception is the vLLM
server, which `scripts.serve` execs in place of this process.

To add a subcommand: write a `cmd_*` function that takes the parsed namespace
and returns an exit code, then register it in `build_parser()`.
"""

import argparse
import sys

from scripts import serve, vibe_check


def cmd_check(args):
    """Send a single prompt, or start an interactive chat."""
    client = vibe_check.make_client()
    if args.interactive:
        vibe_check.interactive(client, args.model)
        return 0

    print(f"Model: {args.model}")
    print(f"Response: {vibe_check.ask(client, args.model, args.prompt)}")
    return 0


def cmd_serve(args):
    """Launch the vLLM server, replacing this process.

    `scripts/serve.py` owns the option list, so unrecognised arguments (and
    --help, hence add_help=False below) are handed straight to it.
    """
    return serve.run(args.extra)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="inference",
        description="Talk to the club's OpenAI-compatible model endpoint.",
        epilog=(
            "Connection settings come from the environment: "
            f"OPENAI_BASE_URL (default {vibe_check.DEFAULT_BASE_URL}) "
            "and OPENAI_API_KEY."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check", help="Send a prompt to the model and print the reply."
    )
    check.add_argument(
        "--model",
        default=vibe_check.DEFAULT_MODEL,
        help="Model to request (default: %(default)s).",
    )
    check.add_argument(
        "--prompt",
        default="What is 967 + 32?",
        help="Prompt to send (default: %(default)s).",
    )
    check.add_argument(
        "--interactive",
        action="store_true",
        help="Chat interactively instead of sending one prompt.",
    )
    check.set_defaults(func=cmd_check)

    serve_cmd = subparsers.add_parser(
        "serve",
        # add_help=False so `inference serve --help` falls through to
        # scripts.serve, which owns the option list and the defaults.
        add_help=False,
        help="Launch the vLLM server (runs scripts/serve.py).",
        description=(
            "Launch the vLLM server. Every option is defined in "
            "scripts/serve.py -- run `inference serve --help` to see them."
        ),
    )
    serve_cmd.set_defaults(func=cmd_serve)

    return parser


def main(argv=None):
    parser = build_parser()
    # `serve` hands unrecognised arguments to scripts.serve, which owns that
    # option list. Every other subcommand rejects them.
    args, extra = parser.parse_known_args(argv)
    if extra and args.command != "serve":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.extra = extra
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
    
