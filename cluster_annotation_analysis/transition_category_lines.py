"""Moving transition-probability line graphs for a focal cluster, aggregated to
annotation *categories* -- the simple readout behind the per-partner weekly
heatmaps.

For each focal cluster (default the uncategorized ascent / climbing clusters) two
overlaid line graphs, one panel each:

  * "goes to" (forward):  moving P(next category = X | in focal) across weeks --
    of all the transitions leaving the focal cluster in a window of weeks, the
    fraction that land in a cluster of category X.
  * "came from" (backward): moving P(previous category = X | in focal) -- of all
    the transitions entering the focal cluster, the fraction that started in X.

One line per category (behavior by default, or phase with ``--by phase``); every
week the lines sum to 1, so the panel reads as the focal cluster's transition
makeup evolving over the disease timeline.

"Moving" = a centred rolling window (``--window``, default 3 weeks) pooled at the
count level: numerator and denominator are each summed over the window before
dividing, so low-traffic weeks don't swing the ratio. Only the progression weeks
(week_8..week_24) are drawn; the two week-24 drug conditions are interventions
rather than points on the timeline and are left out of the line view (they remain
in the fan / weekly-heatmap tools).

Run:  uv run python cluster_annotation_analysis/transition_category_lines.py
      uv run python cluster_annotation_analysis/transition_category_lines.py --by phase
      uv run python cluster_annotation_analysis/transition_category_lines.py --window 5
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # repo root
from dataset_config import data_root          # noqa: E402
from utils import save_figure                 # noqa: E402
from presence_heatmap import (                 # noqa: E402
    BEHAVIOR_COLORS, BEHAVIOR_ORDER, PHASE_COLORS, PHASE_ORDER)
from transition_flow import (                  # noqa: E402
    load_annotation, make_meta, node_label)


def progression_weeks(df):
    """Ordered plain progression weeks (week_8..week_24), dropping the week-24
    drug variants that don't sit on the timeline."""
    plain = [w for w in df["week"].unique() if re.fullmatch(r"week_\d+", str(w))]
    return sorted(plain, key=lambda w: int(w.split("_")[1]))


def moving_probs(df, focal, direction, meta, by, weeks, window):
    """weeks x category DataFrame of the moving transition probability.

    direction 'forward' = transitions leaving focal grouped by the target's
    category; 'backward' = transitions entering focal grouped by the source's
    category. Pooled over a centred `window` (count-weighted) then normalised so
    each week's categories sum to 1.
    """
    if direction == "forward":
        edges, pcol = df[df["source"] == focal].copy(), "target"
    else:
        edges, pcol = df[df["target"] == focal].copy(), "source"
    edges["cat"] = edges[pcol].map(meta[by])

    cc = (edges.groupby(["week", "cat"])["count"].sum().unstack("cat")
               .reindex(weeks).fillna(0.0))
    num = cc.rolling(window, center=True, min_periods=1).sum()
    # weeks with no transitions in the window -> NaN (a float gap in the line),
    # not pd.NA which would make the column a non-float nullable dtype
    den = num.sum(axis=1).where(lambda s: s > 0, other=float("nan"))
    return num.div(den, axis=0)


def plot_lines(focal, df, meta, by, weeks, window, path):
    """Two stacked panels (goes-to / came-from) of moving category probabilities."""
    colors = BEHAVIOR_COLORS if by == "behavior" else PHASE_COLORS
    order = BEHAVIOR_ORDER if by == "behavior" else PHASE_ORDER
    xs = [int(w.split("_")[1]) for w in weeks]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    panels = [("forward", "goes to  (focal → category)"),
              ("backward", "came from  (category → focal)")]
    seen = []
    for ax, (direction, title) in zip(axes, panels):
        probs = moving_probs(df, focal, direction, meta, by, weeks, window)
        for cat in order:
            if cat not in probs.columns or probs[cat].fillna(0).sum() == 0:
                continue
            if cat not in seen:
                seen.append(cat)
            ax.plot(xs, probs[cat].to_numpy(dtype=float), color=colors[cat],
                    lw=2.2, marker="o", ms=3.5, label=cat)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel(f"moving P({by})")
        ax.set_ylim(0, None)
        ax.margins(x=0.01)

    axes[-1].set_xlabel("week")
    axes[-1].set_xticks(xs)
    handles = [plt.Line2D([0], [0], color=colors[c], lw=2.4, marker="o", ms=4,
                          label=c) for c in order if c in seen]
    fig.legend(handles=handles, title=by, loc="center left",
               bbox_to_anchor=(0.99, 0.5), fontsize=9, title_fontsize=10)
    fig.suptitle(f"{node_label(focal, meta)}  —  transition makeup by {by}, "
                 f"{window}-week moving probability", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.99, 0.97])
    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", default="1mp")
    ap.add_argument("--by", choices=["behavior", "phase"], default="behavior",
                    help="category axis for the lines (default behavior)")
    ap.add_argument("--behaviors", nargs="+", default=["ascent"])
    ap.add_argument("--phases", nargs="+", default=["uncategorized"])
    ap.add_argument("--window", type=int, default=3,
                    help="centred moving-average window in weeks (default 3)")
    args = ap.parse_args()

    annot = load_annotation(HERE / "data" / f"{args.mouse}.json")
    df = pd.read_csv(data_root() / args.mouse / "cluster_transition_by_week.csv")
    df = df[df["source"] != df["target"]]
    weeks = progression_weeks(df)
    meta = make_meta(annot, sorted(set(df["source"]) | set(df["target"])))

    focal = sorted(annot.loc[annot["behavior"].isin(args.behaviors)
                             & annot["phase"].isin(args.phases), "cluster"])
    focal = [c for c in focal if c in df["source"].values or c in df["target"].values]
    if not focal:
        raise SystemExit(f"no {args.phases} {args.behaviors} clusters found")

    out = HERE / "output" / "transition_category_lines"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.mouse}: focal {focal}, by={args.by}, window={args.window}, "
          f"{len(weeks)} progression weeks")
    for c in focal:
        plot_lines(c, df, meta, args.by, weeks, args.window,
                   out / f"{args.mouse}_cl{c}_{args.by}_lines.jpeg")
        print(f"  cl{c}: wrote category line graph")
    print(f"  output in {out}")


if __name__ == "__main__":
    main()
