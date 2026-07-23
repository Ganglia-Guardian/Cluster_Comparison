"""Annotational transition makeup of a cluster (or a whole category), split into
three stretches of the timeline and drawn as pies.

Like the makeup pies in ``transition_flow.py``, but over time: the progression
weeks (week_8..week_24) are cut into

    first 6  (weeks 8-13) | mid 5 (weeks 14-18) | final 6 (weeks 19-24)

and each period gets one pie of the annotational makeup of the transitions
touching the focal. Two figures are written:

  * ``_into``  -- makeup of what transitions INTO the focal (by the source's
                  category), one pie per period, and
  * ``_outof`` -- makeup of what the focal transitions OUT to (by the target's
                  category).

Focal is either a single cluster (``--cluster 54``) or a whole category
(``--category ascent``); slices are grouped by ``--by`` (behavior or phase), the
same vocabulary the category is named in. ``--highlight`` renders the named
categories in full colour and mutes the rest (pale, desaturated, and slightly
exploded), to follow a few categories across the three periods.

Weighting is by raw transition count; self-transitions are dropped (so for a
category focal, only cross-cluster moves count).

Run:  uv run python cluster_annotation_analysis/transition_makeup_periods.py --category immobile
      uv run python cluster_annotation_analysis/transition_makeup_periods.py --cluster 54 --highlight immobile locomotion
      uv run python cluster_annotation_analysis/transition_makeup_periods.py --category ascent --by phase --highlight early late
"""
from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # repo root
from dataset_config import data_root          # noqa: E402
from utils import save_figure                 # noqa: E402
from presence_heatmap import (                 # noqa: E402
    BEHAVIOR_COLORS, BEHAVIOR_ORDER, PHASE_COLORS, PHASE_ORDER)
from transition_flow import load_annotation, make_meta, node_label   # noqa: E402
from transition_category_lines import progression_weeks              # noqa: E402


def mute(color):
    """A pale, desaturated version of `color` for non-highlighted slices."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
    return colorsys.hls_to_rgb(h, 0.82, s * 0.25)


def split_periods(weeks, first_n, last_n):
    """[(label, [weeks])] for first `first_n`, the middle, and last `last_n`."""
    nums = [int(w.split("_")[1]) for w in weeks]
    spans = [("first", weeks[:first_n], nums[:first_n]),
             ("mid", weeks[first_n:len(weeks) - last_n], nums[first_n:len(weeks) - last_n]),
             ("final", weeks[len(weeks) - last_n:], nums[len(weeks) - last_n:])]
    out = []
    for tag, wk, nn in spans:
        if not wk:
            continue
        out.append((f"{tag} {len(wk)}  (wk {nn[0]}–{nn[-1]})", wk))
    return out


def makeup(df, focal_ids, direction, period_weeks, meta, by, order):
    """Ordered {category: total count} of transitions into/out of `focal_ids`
    during `period_weeks`, grouped by the neighbour's `by` category."""
    if direction == "in":
        e = df[df["target"].isin(focal_ids) & df["week"].isin(period_weeks)]
        pcol = "source"
    else:
        e = df[df["source"].isin(focal_ids) & df["week"].isin(period_weeks)]
        pcol = "target"
    agg = e["count"].groupby(e[pcol].map(meta[by]).values).sum()
    return {c: float(agg.get(c, 0.0)) for c in order if agg.get(c, 0.0) > 0}


def plot_direction(df, focal_ids, focal_label, direction, periods, meta, by,
                   highlight, path):
    palette = BEHAVIOR_COLORS if by == "behavior" else PHASE_COLORS
    order = BEHAVIOR_ORDER if by == "behavior" else PHASE_ORDER
    hi = set(highlight or [])

    def wedge_color(cat):
        return palette[cat] if (not hi or cat in hi) else mute(palette[cat])

    fig, axes = plt.subplots(1, len(periods), figsize=(4.6 * len(periods), 5.2),
                             squeeze=False)
    axes = axes[0]
    seen = []
    for ax, (label, wk) in zip(axes, periods):
        m = makeup(df, focal_ids, direction, wk, meta, by, order)
        total = sum(m.values())
        if total == 0:
            ax.text(0.5, 0.5, "no transitions", ha="center", va="center",
                    fontsize=10, color="#888888")
            ax.set_axis_off()
            ax.set_title(f"{label}\n(n=0)", fontsize=10)
            continue
        cats = list(m)
        for c in cats:
            if c not in seen:
                seen.append(c)
        explode = [0.06 if (hi and c in hi) else 0.0 for c in cats]
        ax.pie(list(m.values()), colors=[wedge_color(c) for c in cats],
               explode=explode, startangle=90, counterclock=False,
               wedgeprops=dict(edgecolor="white", linewidth=0.8),
               autopct=lambda p: f"{p:.0f}%" if p >= 7 else "",
               textprops=dict(fontsize=8, color="#222222"))
        ax.set_title(f"{label}\n(n={int(total)})", fontsize=10)

    arrow = "into" if direction == "in" else "out of"
    handles = [Patch(facecolor=palette[c],
                     edgecolor=("black" if (hi and c in hi) else "none"),
                     label=(c + "  ★" if (hi and c in hi) else c))
               for c in order if c in seen]
    fig.legend(handles=handles, title=by, loc="center left",
               bbox_to_anchor=(0.99, 0.5), fontsize=9, title_fontsize=10)
    fig.suptitle(f"transition makeup {arrow}  {focal_label}   "
                 f"(by {by}, weighted by count)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.99, 0.95])
    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", default="1mp")
    ap.add_argument("--by", choices=["behavior", "phase"], default="behavior")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cluster", type=int, help="focal single cluster id")
    g.add_argument("--category", help="focal category (under --by)")
    ap.add_argument("--highlight", nargs="+", default=None,
                    help="categories to keep bright; the rest are muted")
    ap.add_argument("--first-weeks", type=int, default=6)
    ap.add_argument("--last-weeks", type=int, default=6)
    args = ap.parse_args()

    order = BEHAVIOR_ORDER if args.by == "behavior" else PHASE_ORDER
    for h in (args.highlight or []):
        if h not in order:
            raise SystemExit(f"--highlight '{h}' is not a {args.by} category; "
                             f"choose from {order}")

    annot = load_annotation(HERE / "data" / f"{args.mouse}.json")
    df = pd.read_csv(data_root() / args.mouse / "cluster_transition_by_week.csv")
    df = df[df["source"] != df["target"]]
    weeks = progression_weeks(df)
    meta = make_meta(annot, sorted(set(df["source"]) | set(df["target"])))
    periods = split_periods(weeks, args.first_weeks, args.last_weeks)

    if args.cluster is not None:
        focal_ids = [args.cluster]
        focal_label = node_label(args.cluster, meta)
        slug = f"cl{args.cluster}"
    else:
        cat = args.category or "immobile"
        focal_ids = meta.index[meta[args.by] == cat].tolist()
        if not focal_ids:
            raise SystemExit(f"category '{cat}' has no clusters under --by {args.by}")
        focal_label = f"{cat} ({len(focal_ids)} clusters)"
        slug = f"{args.by}_{cat}"

    out = HERE / "output" / "transition_makeup_periods"
    out.mkdir(parents=True, exist_ok=True)
    for direction, tag in [("in", "into"), ("out", "outof")]:
        plot_direction(df, focal_ids, focal_label, direction, periods, meta,
                       args.by, args.highlight,
                       out / f"{args.mouse}_{slug}_{tag}.jpeg")
        print(f"  wrote {tag} makeup ({len(periods)} periods)")
    hl = f", highlight={args.highlight}" if args.highlight else ""
    print(f"{args.mouse}: focal {focal_label}, by={args.by}{hl}; output in {out}")


if __name__ == "__main__":
    main()
