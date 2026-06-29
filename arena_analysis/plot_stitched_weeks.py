"""Stitch the per-batch transition statistics into one continuous w8->w24
trajectory per mouse.

Fan-out and rarefied richness are label-free GROUP statistics (one value per
week, independent of the cluster ids), so the three batches -- which sample
interleaved, non-overlapping weeks -- can be tiled onto a single week axis to
approximate "one big clustering per mouse". Each point is marked by its source
batch so any between-batch level offset (different codebooks K) is visible.

Reads the per-week CSVs already written by the transition scripts:
    --view clustertype --mode {home,all,all_bins}   (primarily-2D vs primarily-3D)
    --view arena                                     (2D vs 3D, all clusters)

One figure per mouse: fan-out (left) and rarefied richness (right) vs week, with
the stitched Spearman trend per group in the legend.

Run:
    uv run python arena_analysis/plot_stitched_weeks.py
    uv run python arena_analysis/plot_stitched_weeks.py --view arena
    uv run python arena_analysis/plot_stitched_weeks.py --view clustertype --mode all_bins
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "stitched_weeks"
BATCH_MARKER = {"w8": "o", "w9": "s", "w10": "^"}
STYLE = {  # group label -> (colour, linestyle)
    "primarily-2D": ("#d62728", "--"), "primarily-3D": ("#1f77b4", "-"),
    "2D": ("#d62728", "--"), "3D": ("#1f77b4", "-"),
}


def sources(view, mode):
    if view == "arena":
        d = ROOT / "output" / "transitions"
        return d / "fanout_by_arena.csv", d / "diversity_by_arena.csv", "arena"
    d = ROOT / "output" / "transitions_by_clustertype" / mode
    return d / "fanout_by_clustertype.csv", d / "diversity_by_clustertype.csv", "subset"


def stitched(df, gcol, group, value):
    """Week-indexed series for one group, averaging any duplicate weeks."""
    s = df[df[gcol] == group]
    return s.groupby("week")[value].mean().sort_index()


def trend_label(group, series):
    if len(series) >= 3 and series.std() > 0:
        rho, p = spearmanr(series.index, series.values)
        return f"{group}  (rho={rho:+.2f}, p={p:.3f})"
    return group


def panel(ax, df, gcol, value, ylabel, title):
    for group, (color, ls) in STYLE.items():
        if group not in df[gcol].unique():
            continue
        s = stitched(df, gcol, group, value)
        ax.plot(s.index, s.values, ls, color=color, lw=1.8, zorder=1,
                label=trend_label(group, s))
        g = df[df[gcol] == group]
        for b, mk in BATCH_MARKER.items():
            bb = g[g["batch"] == b]
            ax.scatter(bb["week"], bb[value], marker=mk, color=color, s=38,
                       zorder=2, edgecolors="white", linewidths=0.5)
    ax.set(xlabel="disease week", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", choices=["clustertype", "arena"], default="clustertype")
    ap.add_argument("--mode", choices=["home", "all", "all_bins"], default="home",
                    help="clustertype subset mode (ignored for --view arena)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="weeks to drop, as 'WEEK' (all mice) or 'MOUSE:WEEK' "
                         "(e.g. 2mp:20) -- a low-quality/outlier recording")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    excl_global, excl_mouse = set(), {}
    for tok in args.exclude:
        if ":" in tok:
            mse, wk = tok.split(":")
            excl_mouse.setdefault(mse, set()).add(int(wk))
        else:
            excl_global.add(int(tok))

    fan_path, div_path, gcol = sources(args.view, args.mode)
    fan, div = pd.read_csv(fan_path), pd.read_csv(div_path)
    tag = args.view + (f"-{args.mode}" if args.view == "clustertype" else "")

    for mouse in sorted(fan["mouse"].unique()):
        drop = excl_global | excl_mouse.get(mouse, set())
        fan_m = fan[(fan.mouse == mouse) & (~fan.week.isin(drop))]
        div_m = div[(div.mouse == mouse) & (~div.week.isin(drop))]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        panel(axes[0], fan_m, gcol, "mean_successors",
              "mean distinct successors / source", "Fan-out by week (stitched)")
        panel(axes[1], div_m, gcol, "median_richness",
              "median rarefied richness", "Successor richness by week (stitched)")
        for ax in axes:
            grp_leg = ax.legend(title="group  (stitched trend)", fontsize=8, loc="upper right")
            ax.add_artist(grp_leg)
        # batch-marker legend (which clustering each point came from)
        axes[1].legend(handles=[Line2D([0], [0], marker=mk, color="grey", ls="",
                                       label=b) for b, mk in BATCH_MARKER.items()],
                       title="source batch", fontsize=8, loc="lower left")
        ex = f"  (excl wk {sorted(drop)})" if drop else ""
        fig.suptitle(f"{mouse}: stitched w8->w24 transition trajectory  [{tag}]{ex}", y=1.0)
        fig.tight_layout()
        suffix = f"_excl-{'-'.join(map(str, sorted(drop)))}" if drop else ""
        path = OUT / f"{mouse}_{tag}{suffix}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        weeks = sorted(fan_m["week"].unique())
        print(f"  {mouse}: weeks {weeks[0]}-{weeks[-1]} ({len(weeks)} pts){ex} -> {path}")

    print(f"\nWrote per-mouse figures under {OUT}/")


if __name__ == "__main__":
    main()
