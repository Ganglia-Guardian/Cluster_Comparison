"""Cluster-presence heatmaps and a validation dendrogram, sorted / coloured by a
hand annotation of the clusters.

The annotation (``data/<mouse>.json``) tags every cluster on two axes:

  * a *phase* -- when across the recording weeks the cluster is present
    (early / mid / late / sustained / uncategorized), and
  * a *behavior* -- what the animal is doing in it
    (descent / ascent / rearing / grooming / immobile / locomotion / miscellaneous).

This is a deliberately human, non-numerical labelling: it is a memorable example
for readers who are not interested in the clustering internals, not a
replacement for them. So the job here is to *lay the numbers next to the labels*
and let the two agree (or not):

  1. Presence heatmap = a cluster x week matrix of each cluster's weekly
     occupancy, ROW-NORMALISED so every row is that cluster's temporal profile
     (fraction of its own frames landing in each week). Rows are then sorted by
     the annotation. If the phase labels mean anything, sorting by phase makes an
     early->late diagonal band fall out of the matrix on its own. Two figures:
     one sorted phase-then-behavior, one behavior-then-phase.

  2. Validation dendrogram = Ward linkage over those same temporal profiles, with
     no annotation in the distance. Leaf-aligned strips recolour the tree by
     phase and behavior; if the hand phases track the presence structure, the
     phase strip comes out in contiguous blocks. A cut into as many groups as
     there are phases is scored against the phase labels (adjusted Rand) as a
     one-number readout.

Run:  uv run python cluster_annotation_analysis/presence_heatmap.py            # mouse 1mp
      uv run python cluster_annotation_analysis/presence_heatmap.py --mouse 1mp
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # repo root
from dataset_config import CSV_NAME, data_root  # noqa: E402
from utils import save_figure                    # noqa: E402

# Annotation vocab, kept in the order we want to read it on the page. `early`
# -> `sustained` is a rough temporal progression; `uncategorized` (annotated but
# no phase call) and `unannotated` (never touched) trail after, greyed out.
PHASE_ORDER = ["early", "mid", "late", "sustained", "uncategorized", "unannotated"]
BEHAVIOR_ORDER = ["descent", "ascent", "rearing", "grooming", "immobile",
                  "locomotion", "miscellaneous", "unannotated"]

PHASE_COLORS = {
    "early": "#4575B4", "mid": "#FDAE61", "late": "#D73027",
    "sustained": "#1A9850", "uncategorized": "#7F7F7F", "unannotated": "#DDDDDD",
}
BEHAVIOR_COLORS = {
    **{b: plt.cm.tab10(i) for i, b in enumerate(BEHAVIOR_ORDER[:-1])},
    "unannotated": "#DDDDDD",
}
PRESENCE_CMAP = plt.cm.magma   # 0 -> near-white, high -> dark
UNANNOTATED = "unannotated"


def load_annotation(path):
    """The ``{phase: {behavior: [cluster_ids]}}`` JSON -> tidy long DataFrame
    with columns [cluster, phase, behavior]. A cluster listed under more than one
    (phase, behavior) -- it should not be -- keeps its first listing and warns."""
    tree = json.loads(Path(path).read_text())
    rows, seen = [], {}
    for phase, behaviors in tree.items():
        for behavior, ids in behaviors.items():
            for cid in ids:
                if cid in seen:
                    print(f"  warn: cluster {cid} annotated twice "
                          f"({seen[cid]} and {phase}/{behavior}); keeping first")
                    continue
                seen[cid] = f"{phase}/{behavior}"
                rows.append({"cluster": int(cid), "phase": phase,
                             "behavior": behavior})
    return pd.DataFrame(rows)


def week_sort_key(week):
    """Order weeks chronologically: ``week_8`` < ... < ``week_24`` < its treatment
    variants (``week_24_saline`` < ``week_24_ldop``), which share week 24 but come
    after the plain session."""
    m = re.match(r"week_(\d+)(?:_(.*))?$", str(week))
    if not m:
        return (10**6, 9, str(week))          # unknown labels sort last, stably
    n = int(m.group(1))
    suffix = m.group(2) or ""
    suffix_rank = {"": 0, "saline": 1, "ldop": 2}.get(suffix, 3)
    return (n, suffix_rank, suffix)


def build_presence(detail_csv, annot):
    """Return (presence, meta): a row-normalised cluster x week presence matrix
    and a per-cluster annotation frame aligned to it.

    Every cluster that occurs in the data gets a row -- clusters the annotator
    never touched are folded in as phase/behavior = 'unannotated' rather than
    dropped, so the figure never silently hides part of the animal's repertoire.
    """
    df = pd.read_csv(detail_csv).dropna(subset=["Folder_Name"])
    counts = pd.crosstab(df["ClusterIdx"], df["Folder_Name"])
    counts = counts[sorted(counts.columns, key=week_sort_key)]
    presence = counts.div(counts.sum(axis=1), axis=0)      # row-normalise
    presence.index = presence.index.astype(int)

    meta = (annot.set_index("cluster")
                 .reindex(presence.index)
                 .fillna({"phase": UNANNOTATED, "behavior": UNANNOTATED}))
    return presence, meta


def order_rows(meta, keys):
    """Cluster ids ordered by the annotation `keys` (e.g. ['phase','behavior']),
    each key following its canonical vocab order, ties broken by cluster id."""
    orders = {"phase": PHASE_ORDER, "behavior": BEHAVIOR_ORDER}
    tmp = meta.copy()
    for k in keys:
        tmp[k + "_rank"] = tmp[k].map({v: i for i, v in enumerate(orders[k])})
    return tmp.sort_values([k + "_rank" for k in keys] + [tmp.index.name or "index"],
                           kind="stable").index.tolist()


def _annotation_strip(ax, values, colors, order, label):
    """One vertical colour strip, one row per cluster (top = first row), plus its
    y-label. `values` is the per-row category; `colors`/`order` fix the mapping."""
    idx = {v: i for i, v in enumerate(order)}
    rgba = np.array([plt.matplotlib.colors.to_rgba(colors[v]) for v in values])
    ax.imshow(rgba[:, None, :], aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(label, rotation=90, fontsize=8, labelpad=4)
    ax.xaxis.set_label_position("top")


def _group_separators(ax, values, n_cols):
    """Draw thin white lines between consecutive runs of a category down the rows,
    so the annotation blocks read as blocks in the heatmap itself."""
    boundaries = np.where(values[1:] != values[:-1])[0] + 1
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=1.4)


def plot_heatmap(presence, meta, keys, title, path):
    """Presence heatmap with rows sorted by `keys` and two annotation strips
    (phase, behavior) down the left margin."""
    order = order_rows(meta, keys)
    mat = presence.loc[order].to_numpy()
    m = meta.loc[order]
    weeks = [w.replace("week_", "") for w in presence.columns]
    n = len(order)

    # robust ceiling so a few concentrated clusters don't wash the rest out
    nz = mat[mat > 0]
    vmax = float(np.percentile(nz, 98)) if nz.size else 1.0

    fig = plt.figure(figsize=(max(7, 0.42 * len(weeks) + 2.5), max(6, 0.11 * n)))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.6, 0.6, 26, 0.8], wspace=0.06)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    _annotation_strip(ax_p, m["phase"].to_numpy(), PHASE_COLORS, PHASE_ORDER, "phase")
    _annotation_strip(ax_b, m["behavior"].to_numpy(), BEHAVIOR_COLORS,
                      BEHAVIOR_ORDER, "behavior")

    im = ax_h.imshow(mat, aspect="auto", cmap=PRESENCE_CMAP, vmin=0, vmax=vmax,
                     interpolation="nearest")
    _group_separators(ax_h, m[keys[0]].to_numpy(), len(weeks))
    ax_h.set_xticks(range(len(weeks)))
    ax_h.set_xticklabels(weeks, rotation=90, fontsize=7)
    ax_h.set_yticks(range(n))
    ax_h.set_yticklabels(m.index, fontsize=4)
    for tick, ph in zip(ax_h.get_yticklabels(), m["phase"]):
        tick.set_color(PHASE_COLORS[ph] if ph != UNANNOTATED else "#999999")
    ax_h.set_xlabel("week")
    ax_h.set_title(title, fontsize=11)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("within-cluster weekly presence\n(row-normalised)", fontsize=8)

    # legends: phase and behavior, off to the right of the colorbar
    phase_handles = [Patch(facecolor=PHASE_COLORS[p], label=p)
                     for p in PHASE_ORDER if (m["phase"] == p).any()]
    beh_handles = [Patch(facecolor=BEHAVIOR_COLORS[b], label=b)
                   for b in BEHAVIOR_ORDER if (m["behavior"] == b).any()]
    leg1 = ax_h.legend(handles=phase_handles, title="phase", fontsize=7,
                       title_fontsize=8, loc="upper left",
                       bbox_to_anchor=(1.14, 1.0), borderaxespad=0)
    ax_h.add_artist(leg1)
    ax_h.legend(handles=beh_handles, title="behavior", fontsize=7,
                title_fontsize=8, loc="upper left",
                bbox_to_anchor=(1.14, 0.55), borderaxespad=0)

    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return order


def _leaf_strip(ax, values, leaves, colors, order, label):
    """A leaf-aligned categorical strip under the dendrogram. scipy places leaf i
    at x = 10*i + 5, so the strip spans [0, 10*n] to line up under the tree."""
    n = len(leaves)
    rgba = np.array([plt.matplotlib.colors.to_rgba(colors[v]) for v in values[leaves]])
    ax.imshow(rgba[None, :, :], aspect="auto", extent=[0, 10 * n, 0, 1],
              interpolation="nearest")
    ax.set_yticks([])
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)


def plot_dendrogram(presence, meta, path):
    """Ward tree over the temporal presence profiles, recoloured by annotation.

    The annotation is NOT in the distance -- only the cluster x week profiles are
    -- so the phase/behavior strips underneath are an independent check on the
    labelling. Reports the adjusted-Rand agreement between a phase-count cut of
    the tree and the phase labels (unannotated clusters excluded from the score).
    """
    ids = presence.index.to_numpy()
    X = presence.to_numpy()
    Z = linkage(X, method="ward")
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    n = len(ids)

    phase_vals = meta["phase"].to_numpy()
    beh_vals = meta["behavior"].to_numpy()

    # quantitative check: cut into (#real phases) groups, score vs phase labels
    real = np.array([p != UNANNOTATED for p in phase_vals])
    k = phase_vals[real]
    n_phase = len({p for p in phase_vals if p != UNANNOTATED})
    cut = fcluster(Z, t=max(n_phase, 2), criterion="maxclust")
    ari = adjusted_rand_score(phase_vals[real], cut[real]) if real.any() else float("nan")

    fig = plt.figure(figsize=(max(10, 0.16 * n), 6.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[6, 0.5, 0.5], hspace=0.12)
    ax_d = fig.add_subplot(gs[0])
    ax_ph = fig.add_subplot(gs[1], sharex=ax_d)
    ax_be = fig.add_subplot(gs[2], sharex=ax_d)

    dendrogram(Z, ax=ax_d, no_labels=True, above_threshold_color="#999999",
               color_threshold=0)
    ax_d.set_ylabel("Ward distance")
    ax_d.set_title(f"cluster temporal-presence dendrogram, coloured by annotation "
                   f"(phase vs {n_phase}-group cut: adj. Rand = {ari:.2f})",
                   fontsize=11)
    ax_d.tick_params(labelbottom=False)

    _leaf_strip(ax_ph, phase_vals, leaves, PHASE_COLORS, PHASE_ORDER, "phase")
    _leaf_strip(ax_be, beh_vals, leaves, BEHAVIOR_COLORS, BEHAVIOR_ORDER, "behavior")

    # cluster-id labels under the bottom strip, coloured by phase
    ax_be.set_xticks([10 * i + 5 for i in range(n)])
    ax_be.set_xticklabels([str(ids[l]) for l in leaves], rotation=90, fontsize=4)
    for tick, l in zip(ax_be.get_xticklabels(), leaves):
        ph = phase_vals[l]
        tick.set_color(PHASE_COLORS[ph] if ph != UNANNOTATED else "#999999")
    ax_be.tick_params(labelbottom=True)
    ax_be.set_xlabel("cluster id (leaf order)")

    phase_handles = [Patch(facecolor=PHASE_COLORS[p], label=p)
                     for p in PHASE_ORDER if (phase_vals == p).any()]
    beh_handles = [Patch(facecolor=BEHAVIOR_COLORS[b], label=b)
                   for b in BEHAVIOR_ORDER if (beh_vals == b).any()]
    ax_d.legend(handles=phase_handles + beh_handles, ncol=2, fontsize=7,
                title="annotation", title_fontsize=8, loc="upper right")

    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ari


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", default="1mp",
                    help="dataset folder / annotation stem (default: 1mp)")
    args = ap.parse_args()

    annot_path = HERE / "data" / f"{args.mouse}.json"
    detail_csv = data_root() / args.mouse / CSV_NAME
    out = HERE / "output"
    out.mkdir(parents=True, exist_ok=True)

    annot = load_annotation(annot_path)
    presence, meta = build_presence(detail_csv, annot)
    n_un = int((meta["phase"] == UNANNOTATED).sum())
    print(f"{args.mouse}: {len(presence)} clusters x {presence.shape[1]} weeks "
          f"({len(annot)} annotated, {n_un} unannotated)")

    plot_heatmap(presence, meta, ["phase", "behavior"],
                 f"{args.mouse}: cluster presence, sorted by phase then behavior",
                 out / f"{args.mouse}_presence_by_phase.jpeg")
    plot_heatmap(presence, meta, ["behavior", "phase"],
                 f"{args.mouse}: cluster presence, sorted by behavior then phase",
                 out / f"{args.mouse}_presence_by_behavior.jpeg")
    ari = plot_dendrogram(presence, meta,
                          out / f"{args.mouse}_presence_dendrogram.jpeg")

    print(f"  phase vs presence-tree cut: adjusted Rand = {ari:.3f}")
    print(f"  wrote heatmaps + dendrogram to {out}")


if __name__ == "__main__":
    main()
