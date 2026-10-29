"""Report prefill CPU duration and gaps between recorded GPU activities."""

import argparse
import gzip
import json
from pathlib import Path


def analyze(path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    spans = [
        event for event in events
        if event.get("cat") == "user_annotation"
        and event["name"].startswith("execute_context_1(")
    ]
    if len(spans) != 1:
        raise ValueError(f"Expected one prefill in {path}, found {len(spans)}")
    prefill = spans[0]
    start, end = prefill["ts"], prefill["ts"] + prefill["dur"]
    # CPU scopes measure submission, so GPU work can finish after scope exit.
    # Match profiler external IDs instead of truncating at the CPU timestamp.
    external_ids = {
        event.get("args", {}).get("External id") for event in events
        if event.get("cat") in {"cpu_op", "cuda_runtime", "cuda_driver", "user_annotation"}
        and start <= event.get("ts", -1) < end
    }
    external_ids.discard(None)
    activity = [
        event for event in events
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        and event.get("args", {}).get("External id") in external_ids
    ]
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in activity)
    gaps = []
    busy_end = intervals[0][1]
    for activity_start, activity_end in intervals[1:]:
        if activity_start > busy_end:
            gaps.append(activity_start - busy_end)
        busy_end = max(busy_end, activity_end)
    sequential = [e for e in activity if e["name"] == "_gdn_sequential_kernel"]
    cpu = [
        event for event in events
        if event.get("cat") == "cpu_op" and start <= event["ts"] < end
    ]
    longest = max(cpu, key=lambda e: e["dur"])
    return {
        "trace": str(path),
        "prefill_annotation": prefill["name"],
        "prefill_cpu_ms": prefill["dur"] / 1000,
        "gpu_activity_selection": "External id correlation to CPU events inside prefill",
        "prefill_gpu_activity_span_ms": (max(e[1] for e in intervals) - intervals[0][0]) / 1000,
        "sequential_kernel_count": len(sequential),
        "sequential_kernel_total_ms": sum(e["dur"] for e in sequential) / 1000,
        "largest_internal_gpu_activity_gap_ms": max(gaps, default=0) / 1000,
        "five_largest_internal_gpu_activity_gaps_ms": [g / 1000 for g in sorted(gaps, reverse=True)[:5]],
        "longest_cpu_op": longest["name"],
        "longest_cpu_op_ms": longest["dur"] / 1000,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([analyze(path) for path in args.trace], indent=2))
