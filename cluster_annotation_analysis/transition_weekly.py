"""Week-by-week transition heatmaps for the uncategorized climbing (ascent)
clusters -- how each focal cluster's strongest transition partners come and go
across the recording weeks.

Companion to ``transition_flow.py``. The aggregate fan diagram there shows a
focal cluster's incoming and outgoing partners pooled over all weeks; here we
unfold the time axis: for one focal cluster and one *direction* we draw a
partner x week heatmap, each cell the transition probability that week
(matplotlib ``magma``), grey where the focal has no transitions that week.

Two flags pick what is drawn:

  ``--direction`` -- which flow, relative to the focal cluster:
      * ``towards``  incoming: partner --> focal   (who feeds the focal)
      * ``away``     outgoing: focal --> partner   (where the focal goes)
      * ``both``     one figure each (default)

  ``--metric`` -- which conditional probability colours the cells (same two
  readings as the fan diagram):
      * ``forward``  P(step | current):  towards -> P(focal | partner),
                                          away    -> P(partner | focal)
      * ``backward`` P(origin | current): towards -> P(partner | focal),
                                          away    -> P(focal | partner)

Partners and their row order are taken from the *aggregate* fan diagram (top
``--top-k`` by the same metric), so a weekly figure and its pooled fan diagram
line up row-for-row. A left strip carries each partner's temporal class (phase).
Self-transitions are dropped; every probability is conditional on *changing*
cluster and normalised within the week.

Run:  uv run python cluster_annotation_analysis/transition_weekly.py
      uv run python cluster_annotation_analysis/transition_weekly.py --direction towards
      uv run python cluster_annotation_analysis/transition_weekly.py --metric backward
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # repo root
from dataset_config import data_root          # noqa: E402
from utils import save_figure                 # noqa: E402
from presence_heatmap import (                 # noqa: E402
    PHASE_COLORS, PHASE_ORDER, UNANNOTATED, week_sort_key)
from transition_flow import (                  # noqa: E402
    edge_weights, load_annotation, load_transitions, make_meta, node_label)

WEEKLY_CMAP = plt.cm.magma

# short colourbar label per (direction, metric) -- what a cell's height means
CELL_LABEL = {
    ("towards", "forward"):  "P(focal | partner)   per week",
    ("towards", "backward"): "P(partner | focal)   per week",
    ("away", "forward"):     "P(partner | focal)   per week",
    ("away", "backward"):    "P(focal | partner)   per week",
}
DIR_DESC = {"towards": "incoming  (partner → focal)",
            "away": "outgoing  (focal → partner)"}


def partners_for(T, focal, direction, metric, top_k):
    """Top-`top_k` partner ids for `focal`, ordered exactly as the aggregate fan
    diagram orders them, so the weekly heatmap rows match it."""
    incoming, outgoing = edge_weights(T, focal, metric)
    side = incoming if direction == "towards" else outgoing
    return side.sort_values("prob", ascending=False).head(top_k).index.tolist()


def weekly_matrix(df, focal, partners, direction, metric, weeks):
    """partner x week matrix of the chosen within-week transition probability.

    NaN where the denominator is empty (the focal, or the partner, has no
    transitions in that direction that week) so "absent" reads apart from a true
    zero. A present-but-never cell is a real 0.
    """
    out_tot = df.groupby(["week", "source"])["count"].sum()   # leaving each cluster
    in_tot = df.groupby(["week", "target"])["count"].sum()    # entering each cluster
    if direction == "towards":
        edges, pcol = df[df["target"] == focal], "source"     # partner -> focal
    else:
        edges, pcol = df[df["source"] == focal], "target"     # focal -> partner
    cnt = edges.groupby(["week", pcol])["count"].sum()

    mat = np.full((len(partners), len(weeks)), np.nan)
    for j, w in enumerate(weeks):
        for i, p in enumerate(partners):
            if direction == "towards" and metric == "forward":
                denom = out_tot.get((w, p), 0.0)              # per partner
            elif direction == "towards":                       # backward
                denom = in_tot.get((w, focal), 0.0)           # per column (focal)
            elif metric == "forward":                          # away, forward
                denom = out_tot.get((w, focal), 0.0)          # per column (focal)
            else:                                              # away, backward
                denom = in_tot.get((w, p), 0.0)               # per partner
            if denom > 0:
                mat[i, j] = cnt.get((w, p), 0.0) / denom
    return mat


def _phase_strip(ax, phases):
    """Left colour strip, one row per partner, keyed to the phase palette."""
    rgba = np.array([plt.matplotlib.colors.to_rgba(
        PHASE_COLORS[p if p in PHASE_COLORS else UNANNOTATED]) for p in phases])
    ax.imshow(rgba[:, None, :], aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("phase", rotation=90, fontsize=8)
    ax.xaxis.set_label_position("top")


def plot_weekly(focal, T, df, meta, direction, metric, top_k, weeks, path):
    """One partner x week heatmap for `focal` in one direction."""
    partners = partners_for(T, focal, direction, metric, top_k)
    mat = weekly_matrix(df, focal, partners, direction, metric, weeks)
    masked = np.ma.masked_invalid(mat)

    valid = mat[np.isfinite(mat) & (mat > 0)]
    vmax = float(np.percentile(valid, 98)) if valid.size else 1.0
    cmap = WEEKLY_CMAP.copy()
    cmap.set_bad("#cfcfcf")                     # focal absent that week

    wk_labels = [w.replace("week_", "") for w in weeks]
    n = len(partners)
    fig = plt.figure(figsize=(max(8, 0.46 * len(weeks) + 3), max(4.5, 0.42 * n + 1.5)))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.5, 24, 0.8], wspace=0.05)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    _phase_strip(ax_p, meta.loc[partners, "phase"].tolist())
    im = ax_h.imshow(masked, aspect="auto", cmap=cmap, norm=Normalize(0, vmax),
                     interpolation="nearest")
    ax_h.set_xticks(range(len(weeks)))
    ax_h.set_xticklabels(wk_labels, rotation=90, fontsize=8)
    ax_h.set_yticks([])
    # partner labels ride the leftmost (strip) axis so they clear the strip
    ax_p.set_yticks(range(n))
    ax_p.set_yticklabels([node_label(p, meta) for p in partners], fontsize=8)
    for tick, p in zip(ax_p.get_yticklabels(), partners):
        ph = meta.loc[p, "phase"]
        tick.set_color(PHASE_COLORS[ph] if ph in PHASE_COLORS and ph != UNANNOTATED
                       else "#666666")
    ax_h.set_xlabel("week")
    ax_h.set_title(f"{node_label(focal, meta)}  —  {DIR_DESC[direction]},  "
                   f"week by week   [{metric}]", fontsize=12)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label(CELL_LABEL[(direction, metric)] + "\n(grey = focal absent that week)",
                 fontsize=8)

    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", default="1mp")
    ap.add_argument("--direction", choices=["towards", "away", "both"],
                    default="both",
                    help="flow relative to focal: towards (incoming), away "
                         "(outgoing), or both (default)")
    ap.add_argument("--metric", choices=["forward", "backward"], default="forward",
                    help="within-week probability that colours the cells")
    ap.add_argument("--behaviors", nargs="+", default=["ascent"])
    ap.add_argument("--phases", nargs="+", default=["uncategorized"],
                    help="restrict focal clusters to these temporal classes "
                         "(default: uncategorized)")
    ap.add_argument("--top-k", type=int, default=12)
    args = ap.parse_args()

    annot = load_annotation(HERE / "data" / f"{args.mouse}.json")
    T = load_transitions(data_root() / args.mouse / "cluster_transition_pairs.csv")
    df = pd.read_csv(data_root() / args.mouse / "cluster_transition_by_week.csv")
    df = df[df["source"] != df["target"]]
    weeks = sorted(df["week"].unique(), key=week_sort_key)
    meta = make_meta(annot, list(T.index))

    focal = sorted(annot.loc[annot["behavior"].isin(args.behaviors)
                             & annot["phase"].isin(args.phases), "cluster"])
    focal = [c for c in focal if c in T.index]
    if not focal:
        raise SystemExit(f"no {args.phases} {args.behaviors} clusters found")

    directions = ["towards", "away"] if args.direction == "both" else [args.direction]
    out = HERE / "output" / f"transition_weekly_{args.metric}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.mouse}: focal {focal}, directions={directions}, metric={args.metric}")

    for c in focal:
        for d in directions:
            plot_weekly(c, T, df, meta, d, args.metric, args.top_k, weeks,
                        out / f"{args.mouse}_cl{c}_{d}_weekly.jpeg")
            print(f"  cl{c} {d}: wrote weekly heatmap")
    print(f"  output in {out}")


if __name__ == "__main__":
    main()
