"""Two combined-arena views of the top TBA-contracted clusters per dataset.

Same cluster selection as plot_feature_contraction_lines.py (rank by the pooled
2D+3D slope of mean TBA vs week, steepest decline first, top `n`). For each
dataset (mouse, batch) we draw:

  1. POOLED   one line per cluster, weekly mean TBA over BOTH arenas combined.
              output/tba_combined_lines/<mouse>/week<N>_pooled.png

  2. REGRESSION   mean TBA regressed on week SEPARATELY within each arena, both
              fitted lines on one axis: 2D dashed, 3D solid, one colour per
              cluster (faint markers show the weekly means behind each fit).
              output/tba_combined_lines/<mouse>/week<N>_regression.png

Run:
    uv run python arena_analysis/plot_tba_combined_lines.py
    uv run python arena_analysis/plot_tba_combined_lines.py --n 6 --feature TotAccelBA
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_feature_contraction_lines import (ARENAS, FEATURES, ROOT,
                                            cluster_week_feat, contraction_slope)

OUT = ROOT / "output" / "tba_combined_lines"
ARENA_STYLE = {"2D": "--", "3D": "-"}


def weekly_pivot(frames, feature, clusters, weeks, min_frames):
    cw = cluster_week_feat(frames, feature, min_frames, use_abs=False)
    cw = cw[cw["cluster"].isin(clusters)]
    return cw.pivot(index="week", columns="cluster", values="mean_v") \
             .reindex(index=weeks, columns=clusters)


def plot_pooled(frames, top, weeks, feature, label, colors, min_frames, title, path):
    piv = weekly_pivot(frames, feature, top, weeks, min_frames)
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in top:
        ax.plot(piv.index, piv[c], marker="o", ms=5, color=colors[c], label=f"cluster {c}")
    ax.set(xlabel="week", ylabel=f"mean {label}  (pooled 2D + 3D)", xticks=weeks, title=title)
    ax.grid(alpha=0.3)
    ax.legend(title="cluster", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_regression(frames, top, weeks, feature, label, colors, min_frames, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for arena in ARENAS:
        cw = cluster_week_feat(frames[frames["arena"] == arena], feature, min_frames, False)
        for c in top:
            sub = cw[cw["cluster"] == c].sort_values("week")
            if len(sub) < 2:
                continue
            x, y = sub["week"].to_numpy(float), sub["mean_v"].to_numpy(float)
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, slope * xs + intercept, ARENA_STYLE[arena], color=colors[c], lw=2)
            ax.plot(x, y, "o", ms=3, color=colors[c], alpha=0.35)
    ax.set(xlabel="week", ylabel=f"mean {label}", xticks=weeks, title=title)
    ax.grid(alpha=0.3)
    cluster_leg = ax.legend(handles=[Line2D([0], [0], color=colors[c], lw=2, label=f"cluster {c}")
                                     for c in top], title="cluster", fontsize=8, loc="upper right")
    ax.add_artist(cluster_leg)
    ax.legend(handles=[Line2D([0], [0], color="k", ls="--", lw=2, label="2D (flat)"),
                       Line2D([0], [0], color="k", ls="-", lw=2, label="3D")],
              title="arena", fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def run_dataset(frames, mouse, batch, feature, n, min_frames, min_weeks):
    weeks = sorted(frames["week"].unique())
    start = int(min(weeks))
    _, label = FEATURES[feature]

    pooled = cluster_week_feat(frames, feature, min_frames, use_abs=False)
    slopes = contraction_slope(pooled, min_weeks)
    if slopes.empty:
        print(f"  {mouse}/{batch}: no clusters fit; skipped")
        return
    top = slopes.sort_values().head(n).index.tolist()
    colors = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(top)}

    out_dir = OUT / mouse
    out_dir.mkdir(parents=True, exist_ok=True)
    head = f"{mouse} {batch} (weeks {start}+): top {len(top)} TBA-contracted clusters"
    plot_pooled(frames, top, weeks, feature, label, colors, min_frames,
                f"{head}\npooled 2D+3D weekly mean", out_dir / f"week{start}_pooled.png")
    plot_regression(frames, top, weeks, feature, label, colors, min_frames,
                    f"{head}\nper-arena regression (2D dashed, 3D solid)",
                    out_dir / f"week{start}_regression.png")
    print(f"  {mouse}/{batch}: clusters {top} -> {out_dir}/week{start}_(pooled|regression).png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature", default="TotAccelBA", choices=list(FEATURES))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--min-frames", type=int, default=25)
    ap.add_argument("--min-weeks", type=int, default=4)
    args = ap.parse_args()

    ff = pd.read_csv(ROOT / "frame_features.csv")
    for (mouse, batch), frames in ff.groupby(["mouse", "batch"]):
        run_dataset(frames, mouse, batch, args.feature, args.n, args.min_frames, args.min_weeks)
    print(f"\nWrote figures under {OUT}/")


if __name__ == "__main__":
    main()
