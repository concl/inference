"""Draw the two historical rank-0 prefill captures with uv run plot_prefill.py."""

# /// script
# dependencies = ["matplotlib>=3.8"]
# ///

import json
from pathlib import Path

import matplotlib.pyplot as plt
from analyze import read_prefill

HERE = Path(__file__).resolve().parent


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.29, top=0.72)
    labels = ["Before warmup", "After warmup"]
    for y, capture in zip([1, 0], [summary["baseline_rank0"], summary["warmed_ranks"][0]]):
        _, _, activity = read_prefill(HERE.parents[1] / capture["trace"])
        origin = min(e["ts"] for e in activity)
        intervals = sorted(((e["ts"] - origin) / 1000, e["dur"] / 1000) for e in activity)
        ax.broken_barh(intervals, (y - 0.12, 0.24), facecolors="#007394", edgecolors="none")
        gaps = []
        end = 0
        for start, duration in intervals:
            if start > end:
                gaps.append((start - end, end, start))
            end = max(end, start + duration)
        gap, left, right = max(gaps)
        ax.plot([left, right], [y + 0.2] * 2, marker="|", color="#b34b22", lw=1.4)
        ax.text((left + right) / 2, y + 0.29, f"{gap:.1f} ms gap", ha="center", color="#963b1b")
    ax.set_yticks([1, 0], labels)
    ax.set_ylim(-0.35, 1.62)
    ax.set_xlim(left=-3)
    ax.set_xlabel("Time since first correlated GPU activity (ms)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("The largest gap during prefill shrinks after warmup", x=0.055, y=0.97,
                 ha="left", fontweight="bold", fontsize=17)
    fig.text(0.055, 0.82, "Qwen3.8-27B INT8 + MTP  /  64 input tokens  /  sequential GDN  /  rank 0 of 2")
    fig.text(0.055, 0.06, "Each strip shows kernels, copies and memsets linked to the prefill CPU scope. Both captures contain 48 sequential GDN launches.")
    fig.savefig(HERE / "prefill-timeline.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
