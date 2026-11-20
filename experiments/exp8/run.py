#!/usr/bin/env python3
"""Three-GPU Pareto sweep; see this experiment's README.md.

Run: .venv/bin/python experiments/exp5/run.py --trials 1
Validate arguments without starting a server: add --dry-run.
All raw output goes under this experiment's artifacts/<UTC timestamp>/.
Uses the existing repository environment; never installs packages or edits it.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import optuna
import yaml
from optuna.visualization import plot_pareto_front


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parent.parent
INFERENCE = ROOT / ".venv" / "bin" / "inference"
SERVER_CONFIG = EXPERIMENT_DIR / "config.yaml"
BENCH_CONFIG = EXPERIMENT_DIR / "benchmark-vision-arena.yaml"
DATASET_ID = "lmarena-ai/vision-arena-bench-v0.1"
DATASET_SPLIT = "train"

HOST = "127.0.0.1"
PORT = 8000
GPUS = "0,1,2"
NUM_GPUS = 3
SEED = 42

SEARCH_SPACE = {
    "concurrency": [1, 2, 4, 8, 16, 32],
    "max_batched_tokens": [1024, 2048, 4096, 16384],
    "max_sequences": [4, 8, 16, 32, 64],
}
GRID_SIZE = math.prod(len(values) for values in SEARCH_SPACE.values())
METRICS = ("output_throughput", "p95_ttft_ms", "p95_itl_ms")

STARTUP_TIMEOUT = 7200
BENCHMARK_TIMEOUT = 1800
SHUTDOWN_TIMEOUT = 30


class TrialFailure(RuntimeError):
    """Expected server/benchmark failure: record it and continue the grid."""


def gpu_environment(gpus):
    # Pass a copy so the parent process environment is not modified.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    return env


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yaml(path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def capture_command(command):
    try:
        result = subprocess.run(
            command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, check=False,
        )
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def prepare_run(results_dir, args, server_config, bench_config):
    """Snapshot inputs and record the installed serving environment."""
    # Keep the HF repository ID intact: it is NOT a local filesystem path.
    # The native VisionArenaDataset loader sends turns[0][0].content plus images[0].
    required = {
        "dataset-name": "hf", "dataset-path": DATASET_ID,
        "hf-split": DATASET_SPLIT, "backend": "openai-chat",
        "endpoint": "/v1/chat/completions",
    }
    for name, expected in required.items():
        if bench_config.get(name) != expected:
            raise ValueError(f"{BENCH_CONFIG.name}: {name} must be {expected!r}")
    # CLI overrides are reflected in the snapshots actually used by subprocesses.
    server_config.update({
        "host": HOST, "port": PORT,
        "tensor-parallel-size": args.tensor_parallel_size,
        "pipeline-parallel-size": args.pipeline_parallel_size,
        "data-parallel-size": 1, "distributed-executor-backend": "mp",
    })
    bench_config.update({
        "base-url": f"http://{HOST}:{PORT}",
        "num-prompts": args.num_prompts, "seed": SEED,
    })
    args.server_config = results_dir / "config.yaml"
    args.bench_config = results_dir / "benchmark-vision-arena.yaml"
    args.server_config.write_text(yaml.safe_dump(server_config, sort_keys=False))
    args.bench_config.write_text(yaml.safe_dump(bench_config, sort_keys=False))
    shutil.copy2(SERVER_CONFIG, results_dir / "source-config.yaml")
    shutil.copy2(BENCH_CONFIG, results_dir / "source-benchmark-vision-arena.yaml")
    shutil.copy2(__file__, results_dir / "run.py")

    source_dir = results_dir / "source"
    source_dir.mkdir()
    for name in ("serve.py", "bench.py", "cli.py"):
        shutil.copy2(ROOT / "scripts" / name, source_dir / name)
    versions = {}
    for name in ("vllm", "torch", "triton", "transformers", "optuna", "plotly", "PyYAML", "datasets", "huggingface-hub", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed in runner environment"
    manifest = {
        "python": sys.version, "python_executable": sys.executable,
        "versions": versions,
        "inference_git": capture_command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "inference_worktree": capture_command(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "gpus": capture_command([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,driver_version",
            "--format=csv", "--id=" + args.gpus,
        ]),
        "dataset": {
            "id": DATASET_ID, "split": DATASET_SPLIT,
            "loader": "vLLM VisionArenaDataset", "modalities": ["text", "image"],
            "source": "Hugging Face Hub (downloaded/cached by the benchmark loader)",
            "revision": "unpinned; resolved dataset revision not captured",
        },
        "model": server_config.get("model"),
        "model_revision_requested": server_config.get("revision", "unpinned; resolved revision unknown"),
        "tokenizer_revision_requested": bench_config.get("tokenizer-revision", "unpinned; resolved revision unknown"),
        "note": "Uses the installed vLLM implementation. Package versions are recorded; this runner does not pin an exact vLLM source revision.",
    }
    (results_dir / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")


def port_is_in_use():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((HOST, PORT)) == 0


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def stop_process_group(process):
    """Terminate the whole owned group, even if its launcher already exited.

    Children must remain in the launcher's process group (the usual behavior).
    This deliberately never kills unrelated processes using the same GPUs.
    """
    if process is None:
        return

    pgid = process.pid  # Every Popen below uses start_new_session=True.
    signal_group(pgid, signal.SIGTERM)
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT
    while time.monotonic() < deadline:
        process.poll()  # Reap the launcher; group membership is checked separately.
        if not group_exists(pgid):
            break
        time.sleep(0.2)

    if group_exists(pgid):
        signal_group(pgid, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired as exc:
        # Do not start another GPU trial when cleanup itself has failed.
        raise RuntimeError(f"Could not reap process {process.pid}; stopping study.") from exc


def check_server(server):
    code = server.poll()
    if code is not None:
        raise TrialFailure(f"Server exited with code {code}; see server.log.")


def wait_until_ready(server):
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        check_server(server)
        try:
            with urlopen(f"http://{HOST}:{PORT}/health", timeout=2) as response:
                if response.status == 200:
                    check_server(server)
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise TrialFailure(f"Server startup exceeded {STARTUP_TIMEOUT}s; see server.log.")


def record_command(command, path):
    path.write_text(shlex.join(command) + "\n", encoding="utf-8")


def read_result(path, expected_requests, require_metrics=True):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrialFailure(f"Cannot read {path.name}: {exc}") from exc

    if not isinstance(result, dict):
        raise TrialFailure(f"{path.name} must contain one JSON object.")
    completed = result.get("completed")
    if type(completed) is not int or completed != expected_requests:
        raise TrialFailure(
            f"{path.name}: completed {completed!r}/{expected_requests} requests."
        )
    # Older benchmark versions may not include `failed`.
    if result.get("failed", 0) != 0:
        raise TrialFailure(f"{path.name}: benchmark reported failed requests.")

    if require_metrics:
        for name in METRICS:
            value = result.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise TrialFailure(
                    f"{path.name}: invalid/missing {name}={value!r}. "
                    "Check percentile flags, streaming, and generated token counts."
                )
    return result


def benchmark_command(concurrency, trial_dir, name, num_prompts, seed, args):
    return [
        str(INFERENCE), "bench",
        "--config", str(args.bench_config),
        "--base-url", f"http://{HOST}:{PORT}",
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--seed", str(seed),
        "--save-result",
        "--percentile-metrics", "ttft,tpot,itl",
        "--metric-percentiles", "50,95",
        "--result-dir", str(trial_dir),
        "--result-filename", f"{name}.json",
    ]


def run_benchmark(server, concurrency, trial_dir, name, num_prompts, seed, args):
    command = benchmark_command(concurrency, trial_dir, name, num_prompts, seed, args)
    record_command(command, trial_dir / f"{name}-command.txt")
    process = None
    with (trial_dir / f"{name}.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=gpu_environment(args.gpus),
            )
            deadline = time.monotonic() + BENCHMARK_TIMEOUT
            while process.poll() is None:
                check_server(server)
                if time.monotonic() >= deadline:
                    raise TrialFailure(f"{name} exceeded {BENCHMARK_TIMEOUT}s.")
                time.sleep(0.5)
            if process.returncode != 0:
                raise TrialFailure(
                    f"{name} exited with code {process.returncode}; see {name}.log."
                )
            check_server(server)
        finally:
            stop_process_group(process)

    return read_result(
        trial_dir / f"{name}.json", num_prompts, require_metrics=(name == "result")
    )


def server_command(params, args):
    return [
        str(INFERENCE), "serve",
        "--config", str(args.server_config),
        "--gpus", args.gpus,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--pipeline-parallel-size", str(args.pipeline_parallel_size),
        "--data-parallel-size", "1",
        "--distributed-executor-backend", "mp",
        "--host", HOST,
        "--port", str(PORT),
        "--max-num-batched-tokens", str(params["max_batched_tokens"]),
        "--max-num-seqs", str(params["max_sequences"]),
        "--enable-prefix-caching",
    ]


def validate_commands(results_dir, args):
    """Use the repository's YAML-aware parser before loading the model."""
    params = {name: values[0] for name, values in SEARCH_SPACE.items()}
    commands = {
        "serve-dry-run": server_command(params, args),
        "bench-dry-run": benchmark_command(
            params["concurrency"], results_dir, "preflight", args.num_prompts, SEED, args
        ),
    }
    for name, command in commands.items():
        command = [*command, "--dry-run"]
        record_command(command, results_dir / f"{name}-command.txt")
        with (results_dir / f"{name}.log").open("w") as log:
            try:
                subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=gpu_environment(args.gpus),
                    check=True, timeout=900,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"Preflight failed; see {results_dir / (name + '.log')}") from exc


def objective(trial, results_dir, args):
    params = {
        name: trial.suggest_categorical(name, values)
        for name, values in SEARCH_SPACE.items()
    }
    trial_dir = results_dir / f"trial-{trial.number:04d}"
    trial_dir.mkdir()
    trial.set_user_attr("results_dir", str(trial_dir))
    (trial_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")

    # Do not benchmark or kill an unrelated server.
    if port_is_in_use():
        raise RuntimeError(f"Port {PORT} is already in use; stopping study.")
    command = server_command(params, args)
    record_command(command, trial_dir / "server-command.txt")
    server = None
    with (trial_dir / "server.log").open("w", encoding="utf-8") as log:
        try:
            server = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=gpu_environment(args.gpus),
            )
            wait_until_ready(server)
            if args.warmup_prompts:
                # A different seed reduces identical sampling but does NOT
                # guarantee disjoint prefixes. Cache stays enabled/warm.
                run_benchmark(
                    server, params["concurrency"], trial_dir, "warmup",
                    args.warmup_prompts, SEED + 1, args,
                )
            result = run_benchmark(
                server, params["concurrency"], trial_dir, "result",
                args.num_prompts, SEED, args,
            )
            values = tuple(float(result[name]) for name in METRICS)
            trial.set_user_attr("completed_requests", result["completed"])
            print(
                f"Trial {trial.number}: {values[0]:.2f} tokens/s | "
                f"p95 TTFT {values[1]:.2f} ms | p95 ITL {values[2]:.2f} ms",
                flush=True,
            )
            return values
        except TrialFailure as exc:
            trial.set_user_attr("error", str(exc))
            (trial_dir / "error.txt").write_text(str(exc) + "\n")
            raise
        finally:
            stop_process_group(server)


def save_reports(study, results_dir):
    fields = ["trial", "state", *SEARCH_SPACE, *METRICS, "error", "results_dir"]
    with (results_dir / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trial in study.trials:
            writer.writerow({
                "trial": trial.number,
                "state": trial.state.name,
                **trial.params,
                **dict(zip(METRICS, trial.values or [])),
                "error": trial.user_attrs.get("error", ""),
                "results_dir": trial.user_attrs.get("results_dir", ""),
            })

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nCompleted {len(completed)}/{len(study.trials)} attempted trials.")
    pareto = []
    if completed:
        pareto = [
            {"trial": t.number, "params": t.params, "metrics": dict(zip(METRICS, t.values))}
            for t in study.best_trials
        ]
        (results_dir / "pareto.json").write_text(json.dumps(pareto, indent=2) + "\n")
        try:
            fig = plot_pareto_front(study, target_names=[
                "Output throughput (tokens/s)", "p95 TTFT (ms)", "p95 ITL (ms)",
            ])
            fig.write_html(str(results_dir / "pareto.html"), include_plotlyjs=True)
        except ImportError:
            print("Install plotly to generate the HTML chart. CSV/JSON/SQLite are saved.")
        print(f"Pareto-efficient trials: {len(pareto)}")
        for entry in pareto:
            print(f"  {entry}")
    summary = {
        "experiment": EXPERIMENT_DIR.name,
        "dataset": {"id": DATASET_ID, "split": DATASET_SPLIT, "modalities": ["text", "image"]},
        "status": "complete_grid" if len(completed) == GRID_SIZE else "partial",
        "run_directory": str(results_dir.relative_to(ROOT)),
        "attempted_trials": len(study.trials),
        "completed_trials": len(completed),
        "failed_trials": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
        "benchmarks": {
            f"trial-{t.number:04d}": {
                "params": t.params, "completed": t.user_attrs["completed_requests"],
                **dict(zip(METRICS, t.values)),
                "artifact": f"trial-{t.number:04d}/result.json",
            } for t in completed
        },
        "pareto_trials": pareto,
        "note": "Candidate for curation; this file does not replace the tracked summary.json.",
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    hashes = {
        str(path.relative_to(results_dir)): sha256(path)
        for path in sorted(results_dir.rglob("*"))
        if path.is_file() and path.name != "hashes.json" and "__pycache__" not in path.parts
    }
    (results_dir / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Results: {results_dir}", flush=True)


def main():
    global HOST, PORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=GRID_SIZE)
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Override num-prompts in benchmark-vision-arena.yaml")
    parser.add_argument("--warmup-prompts", type=int, default=5)
    parser.add_argument("--gpus", default=GPUS, help="Exactly three GPU IDs, e.g. 0,1,2")
    parser.add_argument("--port", type=int, default=None, help="Override port in config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Snapshot inputs and validate launcher arguments; start no server")
    parser.add_argument(
        "--parallelism", choices=("pipeline", "tensor"), default="pipeline",
        help="pipeline: TP=1/PP=3 (default); tensor: TP=3/PP=1 (only for compatible models)",
    )
    args = parser.parse_args()
    for path in (INFERENCE, SERVER_CONFIG, BENCH_CONFIG):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Place this folder in experiments/exp5/.")
    server_config = read_yaml(SERVER_CONFIG)
    bench_config = read_yaml(BENCH_CONFIG)
    HOST = server_config.get("host", "127.0.0.1")
    PORT = args.port if args.port is not None else server_config.get("port", 8000)
    if HOST != "127.0.0.1":
        parser.error("This local experiment expects host: 127.0.0.1 in config.yaml")
    if type(PORT) is not int or not 1 <= PORT <= 65535:
        parser.error("Port must be an integer from 1 to 65535")
    if args.num_prompts is None:
        args.num_prompts = bench_config.get("num-prompts", 100)
    if type(args.num_prompts) is not int:
        parser.error("num-prompts must be an integer")
    gpu_ids = [part.strip() for part in args.gpus.split(",")]
    if len(gpu_ids) != NUM_GPUS or not all(
        part.isascii() and part.isdigit() for part in gpu_ids
    ):
        parser.error("--gpus must contain exactly three non-negative integer GPU IDs")
    gpu_ids = [str(int(part)) for part in gpu_ids]
    if len(set(gpu_ids)) != NUM_GPUS:
        parser.error("--gpus must contain three distinct GPU IDs")
    args.gpus = ",".join(gpu_ids)
    args.tensor_parallel_size = NUM_GPUS if args.parallelism == "tensor" else 1
    args.pipeline_parallel_size = NUM_GPUS if args.parallelism == "pipeline" else 1
    if not 1 <= args.trials <= GRID_SIZE:
        parser.error(f"--trials must be between 1 and {GRID_SIZE}")
    if args.num_prompts < 1 or args.warmup_prompts < 0:
        parser.error("--num-prompts must be positive; --warmup-prompts must be >= 0")
    if os.name != "posix":
        parser.error("This script requires POSIX process groups (run on the GPU Linux host).")
    if not os.access(INFERENCE, os.X_OK):
        raise PermissionError(f"Not executable: {INFERENCE}")
    if not args.dry_run and port_is_in_use():
        raise RuntimeError(f"Port {PORT} is already in use. Stop the existing server first.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    results_dir = EXPERIMENT_DIR / "artifacts" / run_id
    results_dir.mkdir(parents=True)
    prepare_run(results_dir, args, server_config, bench_config)
    (results_dir / "run.json").write_text(json.dumps({
        **vars(args), "seed": SEED, "search_space": SEARCH_SPACE,
        "dataset": {"id": DATASET_ID, "split": DATASET_SPLIT},
        "root": str(ROOT), "host": HOST, "port": PORT,
        "data_parallel_size": 1, "distributed_executor_backend": "mp",
        "startup_timeout": STARTUP_TIMEOUT, "benchmark_timeout": BENCHMARK_TIMEOUT,
        "optuna_version": optuna.__version__,
    }, indent=2, default=str) + "\n")
    validate_commands(results_dir, args)
    if args.dry_run:
        print(f"Argument preflight passed. No server started. Snapshots/logs: {results_dir}")
        print("serve --dry-run prints flags; model compatibility and memory are checked only at startup.")
        return 0

    study = optuna.create_study(
        study_name="vllm-pareto",
        storage=f"sqlite:///{results_dir / 'study.db'}",
        directions=["maximize", "minimize", "minimize"],
        sampler=optuna.samplers.GridSampler(SEARCH_SPACE, seed=SEED),
    )
    print(f"Running {args.trials}/{GRID_SIZE} grid combinations. Results: {results_dir}")
    print(f"Dataset: {DATASET_ID}, split={DATASET_SPLIT}, text + image", flush=True)
    print(
        f"GPUs: {args.gpus}; TP={args.tensor_parallel_size}, "
        f"PP={args.pipeline_parallel_size}, DP=1 (local multiprocessing)",
        flush=True,
    )

    # SIGTERM, like Ctrl+C, unwinds the finally blocks and saves partial reports.
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    status = 0
    try:
        study.optimize(
            lambda trial: objective(trial, results_dir, args),
            n_trials=args.trials, n_jobs=1, catch=(TrialFailure,),
        )
    except KeyboardInterrupt:
        print("\nInterrupted; saving completed results.", flush=True)
        status = 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        save_reports(study, results_dir)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        return status or 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
