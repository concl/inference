"""Plot the recorded VisionArena latency tradeoffs with uv run plot_latency.py."""

# /// script
# dependencies = ["matplotlib>=3.8"]
# ///

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
COLORS = {2: "#007394", 4: "#bc5326"}
MARKERS = {"weights": "o", "data": "^"}


def dominates(a, b):
    """Both plotted metrics are minimized."""
    return (
        a["p95_ttft_ms"] <= b["p95_ttft_ms"]
        and a["p95_itl_ms"] <= b["p95_itl_ms"]
        and (a["p95_ttft_ms"] < b["p95_ttft_ms"] or a["p95_itl_ms"] < b["p95_itl_ms"])
    )


def main():
    runs = json.loads((HERE / "summary.json").read_text())["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.6))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.32, top=0.77, wspace=0.34)
    fig.suptitle("First-token and inter-token latency under load", x=0.073, y=0.97,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(0.075, 0.875, "Qwen3.8-27B-FP8  /  VisionArena  /  RTX 3090", fontsize=11)
    for ax, concurrency in zip(axes, (1, 4, 8)):
        conditions = [(r, m) for r in runs for m in r["averages"] if m["concurrency"] == concurrency]
        for run, mean in conditions:
            color = COLORS[len(run["physical_gpus"])]
            marker = MARKERS[mean["mode"]]
            repeats = [m for m in run["measurements"]
                       if m["concurrency"] == concurrency and m["mode"] == mean["mode"]]
            xs = [m["p95_ttft_ms"] / 1000 for m in repeats]
            ys = [m["p95_itl_ms"] for m in repeats]
            ax.plot(xs, ys, color=color, alpha=0.5, marker=marker, markersize=4, linewidth=1)
            x, y = mean["p95_ttft_ms"] / 1000, mean["p95_itl_ms"]
            ax.scatter(x, y, color=color, marker=marker, s=68, edgecolor="white", linewidth=0.8, zorder=3)
            if not any(dominates(other, mean) for _, other in conditions):
                ax.scatter(x, y, marker="o", facecolor="none", edgecolor="#20242b", s=180, zorder=4)
        ax.set_title(f"Concurrency {concurrency}", loc="left", fontweight="bold")
        ax.set_xlabel("p95 time to first token (s)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.margins(x=0.13, y=0.13)
        ax.grid(color="#dce1e4", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("p95 inter-token latency (ms)")
    legend = [Line2D([], [], color=c, lw=3, label=f"{n} GPUs") for n, c in COLORS.items()]
    legend += [Line2D([], [], color="#454b50", ls="", marker=m, label=f"{k} mode")
               for k, m in MARKERS.items()]
    legend.append(Line2D([], [], color="#20242b", ls="", marker="o", mfc="none",
                         markersize=11, label="Observed frontier"))
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.065, 0.135),
               ncol=len(legend), frameon=False)
    fig.text(0.075, 0.09, "Lines join two 100-request repeats. Large symbols show the mean of the two run-level p95 values.")
    fig.text(0.075, 0.049, "Lower and further left is better. Frontier rings compare the four means within each panel. Axis ranges vary by load.")
    fig.savefig(HERE / "latency-frontier.png", dpi=160)


if __name__ == "__main__":
    main()
