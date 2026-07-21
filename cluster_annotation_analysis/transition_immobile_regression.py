"""Every behavior category's transition coupling with "immobile", regressed to a
line, split by direction.

For each category C (behavior by default) we take the two weekly series from
``transition_pair_coupling`` -- C's mean probability of going TO immobile and
coming FROM immobile across its clusters -- and, instead of drawing the wiggly
averaged line, fit a straight line over the progression weeks and plot just that.
The two directions go in two stacked panels sharing axes:

  * top  panel = C -> immobile  (where C goes)
  * bottom panel = immobile -> C  (where immobile came from / went)

Each category keeps one full-strength colour, identical in both panels, so a
category reads as a colour and a direction reads as which panel.

Two figures are written:

  * raw            -- y is the mean transition probability.
  * immobile-presence -- y is that probability divided by immobile's weekly
                      presence (the fraction of frames in immobile clusters that
                      week). This controls for immobile simply occupying more of
                      the repertoire as the disease advances: a flat line near the
                      dashed y = 1 means "couples with immobile exactly as much as
                      immobile is around", above 1 means preferential coupling.

"Moving": per-cluster probabilities are count-pooled over a centred ``--window``
(default 1 = raw weeks; the regression does its own smoothing). Self-transitions
dropped; progression weeks (week_8..week_24) only.

Run:  uv run python cluster_annotation_analysis/transition_immobile_regression.py
      uv run python cluster_annotation_analysis/transition_immobile_regression.py --pivot immobile --window 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import linregress

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # repo root
from dataset_config import CSV_NAME, data_root                 # noqa: E402
from utils import save_figure                                   # noqa: E402
from presence_heatmap import (                                  # noqa: E402
    BEHAVIOR_COLORS, BEHAVIOR_ORDER, PHASE_COLORS, PHASE_ORDER)
from transition_flow import load_annotation, make_meta          # noqa: E402
from transition_category_lines import progression_weeks         # noqa: E402
from transition_pair_coupling import cluster_probs              # noqa: E402


def series_mean(df, focal_ids, partner_ids, direction, weeks, window):
    """Mean over `focal_ids` of each cluster's within-week probability toward /
    from the partner set (a weeks-indexed Series)."""
    mat = pd.DataFrame({a: cluster_probs(df, a, partner_ids, direction, weeks, window)
                        for a in focal_ids})
    return mat.mean(axis=1, skipna=True)


def pivot_presence(detail_csv, pivot_ids, weeks):
    """Weekly occupancy of the pivot ('immobile') clusters: fraction of that
    week's frames that fall in a pivot cluster."""
    d = pd.read_csv(detail_csv).dropna(subset=["Folder_Name"])
    counts = pd.crosstab(d["ClusterIdx"], d["Folder_Name"])
    occ = counts.reindex(pivot_ids).sum() / counts.sum()
    return occ.reindex(weeks)


def fit_endpoints(xs, ys):
    """Least-squares line over the finite points; returns ((x0,x1),(y0,y1)) to
    draw, or None if fewer than two usable points."""
    m = np.isfinite(ys)
    if m.sum() < 2:
        return None
    r = linregress(xs[m], ys[m])
    ends = np.array([xs[m].min(), xs[m].max()])
    return ends, r.intercept + r.slope * ends, r


def plot_regressions(df, meta, by, pivot, weeks, window, occ, normalize, path):
    colors = BEHAVIOR_COLORS if by == "behavior" else PHASE_COLORS
    order = BEHAVIOR_ORDER if by == "behavior" else PHASE_ORDER
    pivot_ids = meta.index[meta[by] == pivot].tolist()
    categories = [c for c in order if c != pivot and (meta[by] == c).any()]
    xs = np.array([int(w.split("_")[1]) for w in weeks], dtype=float)
    occ_arr = occ.to_numpy(float) if normalize else None
    ylabel = (f"mean P / {pivot} presence   (enrichment)" if normalize
              else "mean transition probability")

    # one panel per direction, sharing axes so the two read on the same scale;
    # each category keeps ONE full-strength colour in both panels
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, sharey=True)
    panels = [("to", f"category → {pivot}"), ("from", f"{pivot} → category")]
    ymax = 0.0
    for ax, (direction, ptitle) in zip(axes, panels):
        for C in categories:
            focal_ids = meta.index[meta[by] == C].tolist()
            ys = series_mean(df, focal_ids, pivot_ids, direction, weeks, window).to_numpy(float)
            if normalize:
                ys = ys / occ_arr
            fit = fit_endpoints(xs, ys)
            if fit is None:
                continue
            ends, yhat, _ = fit
            ax.plot(ends, yhat, color=colors[C], lw=2.4, solid_capstyle="round")
            ymax = max(ymax, float(np.nanmax(yhat)))
        if normalize:
            ax.axhline(1.0, ls="--", color="#999999", lw=1.2, zorder=0)
        ax.set_title(ptitle, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.02)

    axes[0].set_ylim(0, ymax * 1.06)          # headroom so no line rides the frame
    axes[-1].set_xlabel("week")
    axes[-1].set_xticks(xs.astype(int))
    handles = [Line2D([0], [0], color=colors[C], lw=2.6, label=C)
               for C in categories]
    fig.legend(handles=handles, ncol=1, fontsize=8, loc="center left",
               bbox_to_anchor=(1.0, 0.5), title=by, title_fontsize=9)
    note = f"  (normalized by {pivot}'s weekly presence)" if normalize else ""
    fig.suptitle(f"each {by} category's transition coupling with '{pivot}', "
                 f"linear fit over weeks{note}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", default="1mp")
    ap.add_argument("--by", choices=["behavior", "phase"], default="behavior")
    ap.add_argument("--pivot", default="immobile", help="category everything is "
                    "compared against (default immobile)")
    ap.add_argument("--window", type=int, default=1,
                    help="centred moving window before the fit (default 1 = raw)")
    args = ap.parse_args()

    annot = load_annotation(HERE / "data" / f"{args.mouse}.json")
    df = pd.read_csv(data_root() / args.mouse / "cluster_transition_by_week.csv")
    df = df[df["source"] != df["target"]]
    weeks = progression_weeks(df)
    meta = make_meta(annot, sorted(set(df["source"]) | set(df["target"])))
    pivot_ids = meta.index[meta[args.by] == args.pivot].tolist()
    if not pivot_ids:
        raise SystemExit(f"pivot '{args.pivot}' has no clusters under --by {args.by}")

    occ = pivot_presence(data_root() / args.mouse / CSV_NAME, pivot_ids, weeks)
    out = HERE / "output" / "transition_immobile_regression"
    out.mkdir(parents=True, exist_ok=True)

    for normalize, tag in [(False, "raw"), (True, "norm_presence")]:
        plot_regressions(df, meta, args.by, args.pivot, weeks, args.window, occ,
                         normalize, out / f"{args.mouse}_{args.by}_{args.pivot}_{tag}.jpeg")
        print(f"  wrote {tag} regression overlay")
    print(f"{args.mouse}: {len(pivot_ids)} '{args.pivot}' clusters; output in {out}")


if __name__ == "__main__":
    main()
