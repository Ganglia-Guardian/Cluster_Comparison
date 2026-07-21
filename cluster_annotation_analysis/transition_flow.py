"""Per-cluster incoming/outgoing transition "flow" diagrams for the annotated
climbing clusters, plus makeup summaries.

For every focal cluster (by default the *ascent* / climbing clusters of the hand
annotation) this draws a fan diagram:

    incoming clusters   -->   [ focal cluster ]   -->   outgoing clusters
        (left column)                                     (right column)

Each edge is coloured (matplotlib ``magma``) and the neighbours on each side are
ordered, top = strongest, by one of two conditional probabilities -- pick with
``--metric``:

  * ``forward`` (the flow direction): for an incoming edge s->C the weight is
    P(C | s) = "given you are in source s, the chance you step to C"; for an
    outgoing edge C->t it is P(t | C). This is the ordinary transition
    probability, read in the direction the arrows point.

  * ``backward`` (the origin direction): for an incoming edge the weight is
    P(s | C) = "given you are now in C, the chance the cluster you came from was
    s"; for an outgoing edge it is P(C | t) = "given you are in t, the chance you
    arrived from C". This is the time-reversed conditional.

Both are conditional probabilities in [0, 1]; a single ``magma`` colourbar sits
outside the two columns. Self-transitions (C->C) are dropped, and each edge's
probability is taken among cluster-*change* transitions only (the row/column sum
excludes the diagonal). Node markers carry each cluster's temporal class (phase);
labels read "Behavior (idx)".

Because the climbing clusters have 40-80 neighbours each, only the top
``--top-k`` per side are drawn; the caption reports how many neighbours and how
much transition mass that covers.

Two summary figures group every focal cluster's incoming and outgoing transition
*makeup* (weighted by raw transition count) as pies -- one split by behavior, one
by temporal phase.

Run:  uv run python cluster_annotation_analysis/transition_flow.py                 # ascent, forward
      uv run python cluster_annotation_analysis/transition_flow.py --metric backward
      uv run python cluster_annotation_analysis/transition_flow.py --behaviors ascent descent
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # repo root
from dataset_config import data_root          # noqa: E402
from utils import save_figure                 # noqa: E402
from presence_heatmap import (                 # noqa: E402
    BEHAVIOR_COLORS, BEHAVIOR_ORDER, PHASE_COLORS, PHASE_ORDER, UNANNOTATED,
    load_annotation)

FLOW_CMAP = plt.cm.magma

# How each metric reads on each side, for titles/colourbar labels.
METRIC_LABEL = {
    "forward":  ("forward transition probability",
                 "left  edge s→C:  P(C | s)   (given source s, step to focal)\n"
                 "right edge C→t:  P(t | C)   (given focal, step to target t)"),
    "backward": ("origin (time-reversed) probability",
                 "left  edge s→C:  P(s | C)   (given focal, came from source s)\n"
                 "right edge C→t:  P(C | t)   (given target t, came from focal)"),
}


def load_transitions(pairs_csv):
    """Directed cluster x cluster transition-count matrix (source rows, target
    cols), self-transitions removed and squared over the union of cluster ids."""
    p = pd.read_csv(pairs_csv)[["source", "target", "total_count"]]
    p = p[p["source"] != p["target"]]                     # drop self-loops
    T = p.pivot_table(index="source", columns="target", values="total_count",
                      aggfunc="sum", fill_value=0)
    ids = sorted(set(T.index) | set(T.columns))
    return T.reindex(index=ids, columns=ids, fill_value=0).astype(float)


def make_meta(annot, ids):
    """Per-cluster [phase, behavior] aligned to `ids`; untouched clusters folded
    in as 'unannotated' so a neighbour is never dropped for lacking a label."""
    return (annot.set_index("cluster").reindex(ids)
                 .fillna({"phase": UNANNOTATED, "behavior": UNANNOTATED}))


def node_label(cid, meta):
    """"Behavior (idx)" with the behavior title-cased, e.g. 'Ascent (6)'."""
    return f"{str(meta.loc[cid, 'behavior']).title()} ({cid})"


def edge_weights(T, focal, metric):
    """Return (incoming, outgoing) frames for `focal`, each indexed by neighbour
    id with columns [count, prob], where `prob` is the chosen conditional metric.

    incoming = clusters that step INTO focal; outgoing = clusters focal steps to.
    Row sums (outgoing totals) and column sums (incoming totals) are over the
    self-loop-free matrix, so every probability is conditional on *changing*
    cluster.
    """
    row_out = T.sum(axis=1)         # total change-transitions leaving each cluster
    col_in = T.sum(axis=0)          # total change-transitions entering each cluster

    inc_counts = T[focal]           # T[s, focal] over sources s
    inc_counts = inc_counts[inc_counts > 0]
    out_counts = T.loc[focal]       # T[focal, t] over targets t
    out_counts = out_counts[out_counts > 0]

    if metric == "forward":
        inc_prob = inc_counts / row_out[inc_counts.index]        # P(focal | s)
        out_prob = out_counts / row_out[focal]                   # P(t | focal)
    else:  # backward
        inc_prob = inc_counts / col_in[focal]                    # P(s | focal)
        out_prob = out_counts / col_in[out_counts.index]         # P(focal | t)

    incoming = pd.DataFrame({"count": inc_counts, "prob": inc_prob})
    outgoing = pd.DataFrame({"count": out_counts, "prob": out_prob})
    return incoming, outgoing


def _phase_of(cid, meta):
    ph = meta.loc[cid, "phase"]
    return ph if ph in PHASE_COLORS else UNANNOTATED


def plot_flow(focal, T, meta, metric, top_k, path):
    """One incoming/focal/outgoing fan diagram for `focal`."""
    incoming, outgoing = edge_weights(T, focal, metric)
    inc = incoming.sort_values("prob", ascending=False)
    out = outgoing.sort_values("prob", ascending=False)
    inc_top, out_top = inc.head(top_k), out.head(top_k)

    vmax = max(inc_top["prob"].max(), out_top["prob"].max())
    norm = Normalize(0, float(vmax))
    n = max(len(inc_top), len(out_top), 1)

    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.62 * n + 1.6)))
    ax.set_xlim(-0.55, 1.7)
    ax.set_ylim(-0.06, 1.14)
    ax.axis("off")

    def ys(k):
        return np.linspace(1.0, 0.0, k) if k > 1 else np.array([0.5])

    yc = 0.5
    xL, xR, xC = 0.0, 1.0, 0.5

    # --- edges (draw first so node markers sit on top) ---
    for (cid, r), y in zip(inc_top.iterrows(), ys(len(inc_top))):
        c = FLOW_CMAP(norm(r["prob"]))
        ax.annotate("", xy=(xC - 0.03, yc), xytext=(xL + 0.02, y),
                    arrowprops=dict(arrowstyle="-|>", color=c,
                                    lw=1.0 + 3.0 * norm(r["prob"]),
                                    shrinkA=2, shrinkB=2))
    for (cid, r), y in zip(out_top.iterrows(), ys(len(out_top))):
        c = FLOW_CMAP(norm(r["prob"]))
        ax.annotate("", xy=(xR - 0.02, y), xytext=(xC + 0.03, yc),
                    arrowprops=dict(arrowstyle="-|>", color=c,
                                    lw=1.0 + 3.0 * norm(r["prob"]),
                                    shrinkA=2, shrinkB=2))

    # --- nodes: phase-coloured marker + label ---
    for (cid, r), y in zip(inc_top.iterrows(), ys(len(inc_top))):
        ax.scatter(xL, y, s=90, color=PHASE_COLORS[_phase_of(cid, meta)],
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(xL - 0.04, y, node_label(cid, meta), ha="right", va="center",
                fontsize=8.5)
    for (cid, r), y in zip(out_top.iterrows(), ys(len(out_top))):
        ax.scatter(xR, y, s=90, color=PHASE_COLORS[_phase_of(cid, meta)],
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(xR + 0.04, y, node_label(cid, meta), ha="left", va="center",
                fontsize=8.5)

    # focal node, boxed and tinted by its own phase
    fph = _phase_of(focal, meta)
    ax.scatter(xC, yc, s=260, color=PHASE_COLORS[fph], edgecolor="black",
               linewidth=1.2, zorder=4)
    ax.text(xC, yc - 0.075, node_label(focal, meta), ha="center", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PHASE_COLORS[fph],
                      lw=1.6))
    ax.text(xL, 1.08, "incoming", ha="center", fontsize=11, fontweight="bold")
    ax.text(xR, 1.08, "outgoing", ha="center", fontsize=11, fontweight="bold")

    # coverage caption -- never hide the tail silently
    inc_cov = inc_top["count"].sum() / inc["count"].sum()
    out_cov = out_top["count"].sum() / out["count"].sum()
    ax.text(xC, -0.045,
            f"showing top {len(inc_top)} of {len(inc)} incoming "
            f"({inc_cov:.0%} of transition mass)   |   "
            f"top {len(out_top)} of {len(out)} outgoing ({out_cov:.0%})",
            ha="center", va="top", fontsize=8, color="#555555")

    metric_name, metric_desc = METRIC_LABEL[metric]
    fig.suptitle(f"{node_label(focal, meta)}  —  transition flow "
                 f"[{metric_name}]", fontsize=13, y=0.99)

    # colourbar outside the two columns
    cax = fig.add_axes([0.9, 0.22, 0.02, 0.56])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=FLOW_CMAP), cax=cax)
    cb.set_label(metric_desc, fontsize=8)

    # phase legend (temporal class of each cluster): a horizontal row along the
    # bottom so it never collides with the left/right node labels
    handles = [Patch(facecolor=PHASE_COLORS[p], edgecolor="black", label=p)
               for p in PHASE_ORDER
               if p == fph or (meta.reindex(list(inc_top.index) + list(out_top.index))
                               ["phase"] == p).any()]
    ax.legend(handles=handles, title="temporal class", fontsize=8,
              title_fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=len(handles), framealpha=0.9)

    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(focal=focal, n_in=len(inc), n_out=len(out),
                inc_cov=inc_cov, out_cov=out_cov)


def _makeup(counts, meta, by, order):
    """Sum `counts` (Series indexed by neighbour id) into an ordered category ->
    total mapping, for a pie. `by` is 'behavior' or 'phase'."""
    cats = meta.loc[counts.index, by]
    agg = counts.groupby(cats.values).sum()
    return {c: float(agg.get(c, 0.0)) for c in order if agg.get(c, 0.0) > 0}


def plot_makeup(clusters, T, meta, by, path):
    """Grid of incoming/outgoing makeup pies (one row per focal cluster), grouped
    by `by` ('behavior' or 'phase') and coloured by that scheme."""
    colors = BEHAVIOR_COLORS if by == "behavior" else PHASE_COLORS
    order = BEHAVIOR_ORDER if by == "behavior" else PHASE_ORDER
    nrows = len(clusters)
    fig, axes = plt.subplots(nrows, 2, figsize=(7.5, 2.5 * nrows),
                             squeeze=False)
    seen = set()
    for i, focal in enumerate(clusters):
        incoming, outgoing = edge_weights(T, focal, "forward")
        for j, (counts, side) in enumerate([(incoming["count"], "incoming"),
                                            (outgoing["count"], "outgoing")]):
            ax = axes[i][j]
            m = _makeup(counts, meta, by, order)
            seen.update(m)
            ax.pie(list(m.values()), colors=[colors[c] for c in m],
                   startangle=90, counterclock=False,
                   wedgeprops=dict(edgecolor="white", linewidth=0.6),
                   autopct=lambda p: f"{p:.0f}%" if p >= 10 else "",
                   textprops=dict(fontsize=7))
            if i == 0:
                ax.set_title(side, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(node_label(focal, meta), fontsize=9,
                              rotation=0, ha="right", va="center", labelpad=30)
    handles = [Patch(facecolor=colors[c], label=c) for c in order if c in seen]
    fig.legend(handles=handles, title=by, loc="center left",
               bbox_to_anchor=(0.99, 0.5), fontsize=8, title_fontsize=9)
    fig.suptitle(f"transition makeup by {by} (weighted by transition count)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.98, 0.97])
    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", default="1mp")
    ap.add_argument("--metric", choices=["forward", "backward"], default="forward",
                    help="edge weight: forward = P(step | current), "
                         "backward = P(origin | current). Default forward.")
    ap.add_argument("--behaviors", nargs="+", default=["ascent"],
                    help="which annotated behaviors are the focal 'climbing' "
                         "clusters (default: ascent)")
    ap.add_argument("--top-k", type=int, default=12,
                    help="neighbours drawn per side (default 12)")
    args = ap.parse_args()

    annot = load_annotation(HERE / "data" / f"{args.mouse}.json")
    T = load_transitions(data_root() / args.mouse / "cluster_transition_pairs.csv")
    meta = make_meta(annot, list(T.index))

    focal = sorted(annot.loc[annot["behavior"].isin(args.behaviors), "cluster"])
    focal = [c for c in focal if c in T.index]
    if not focal:
        raise SystemExit(f"no {args.behaviors} clusters found in the transitions")

    out = HERE / "output" / f"transition_flow_{args.metric}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.mouse}: {len(focal)} focal ({'+'.join(args.behaviors)}) "
          f"clusters {focal}, metric={args.metric}")

    for c in focal:
        info = plot_flow(c, T, meta, args.metric, args.top_k,
                         out / f"{args.mouse}_cl{c}_flow.jpeg")
        print(f"  cl{c}: {info['n_in']} in / {info['n_out']} out  "
              f"(top-{args.top_k} covers {info['inc_cov']:.0%} / {info['out_cov']:.0%})")

    plot_makeup(focal, T, meta, "behavior",
                out / f"{args.mouse}_makeup_by_behavior.jpeg")
    plot_makeup(focal, T, meta, "phase",
                out / f"{args.mouse}_makeup_by_phase.jpeg")
    print(f"  wrote flow diagrams + makeup summaries to {out}")


if __name__ == "__main__":
    main()
