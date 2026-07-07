"""Per-WEEK dendrograms: split each mouse into its individual recording weeks
(8, 9, 10, ...) and merge that week's clusters into their own tree, using
week-specific kinematic centroids (a cluster's mean features drift over disease
weeks, so each week gets its own centroid rather than the batch-pooled one).

Each week belongs to exactly one batch (w8->8,11,14,..; w9->9,12,..; w10->10,..),
so a week's clusters are that batch's clustering restricted to that week's frames.
Clusters with < MIN_FRAMES frames that week are dropped (noisy centroids).

For every (mouse, week) we z-score the 4 features within the week, Ward-link, cut
into k branches, and draw the dendrogram with aligned occ3d + TBA strips
(output/week_dendrogram/<mouse>/w<week>.png). We also collect per-week stats and
a single summary figure so the trajectory across weeks is visible without paging
through every tree:
    - median TBA per week            (bradykinesia: MP should decline, LC flat)
    - occ3d eta^2 per week           (how strongly kinematic branches segregate arena)
    - low-TBA & 2D frame share/week  (the persistent low-vigour flat-arena group)

Run:  uv run python cluster_group_analysis/week_dendrogram.py
      uv run python cluster_group_analysis/week_dendrogram.py --k 5 --min-frames 15
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

from branch_analysis import eta_squared
from common import ROOT
from dendrogram import (OCC_CMAP, TBA_CMAP, _strip, cut_threshold,
                        relabel_by_leaf_order, zscore)
from feature_extraction import FEATURE_NAMES

OUT = ROOT / "output" / "week_dendrogram"
TBA_LOW, OCC_2D = 0.15, 0.45      # absolute thresholds for the low-TBA / 2D group
MOUSE_COLORS = {"1mp": "#d62728", "2mp": "#ff7f0e", "3mp": "#8c564b",
                "1lc": "#1f77b4", "2lc": "#17becf"}


def week_centroids(wf):
    """Collapse per-(cluster, week, arena) rows for ONE (mouse, week) into one
    row per cluster: frame-weighted mean features + per-week occ3d + counts."""
    rows = []
    tot2 = wf.loc[wf.arena == "2D", "n"].sum()
    tot3 = wf.loc[wf.arena == "3D", "n"].sum()
    for c, g in wf.groupby("cluster"):
        n = g["n"].to_numpy()
        n2 = int(g.loc[g.arena == "2D", "n"].sum())
        n3 = int(g.loc[g.arena == "3D", "n"].sum())
        p2 = n2 / tot2 if tot2 else 0.0
        p3 = n3 / tot3 if tot3 else 0.0
        row = {"cluster": int(c), "n_frames": n2 + n3, "n_2d": n2, "n_3d": n3,
               "occ3d": p3 / (p2 + p3) if (p2 + p3) else 0.5}
        for f in FEATURE_NAMES:                      # frame-weighted mean over arenas
            row[f] = float(np.average(g[f].to_numpy(), weights=n))
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def plot_week(sub, Z, leaves, k, title, path):
    n = len(leaves)
    fig = plt.figure(figsize=(max(6, 0.18 * n), 4.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[50, 1],
                          height_ratios=[5, 0.6, 0.6], hspace=0.12, wspace=0.03)
    ax_d = fig.add_subplot(gs[0, 0])
    ax_o = fig.add_subplot(gs[1, 0], sharex=ax_d)
    ax_t = fig.add_subplot(gs[2, 0], sharex=ax_d)
    dendrogram(Z, ax=ax_d, color_threshold=cut_threshold(Z, k),
               above_threshold_color="#999999", no_labels=True)
    ax_d.set(ylabel="Ward dist", title=title)
    ax_d.tick_params(labelbottom=False)
    ax_d.spines[["top", "right"]].set_visible(False)

    occ = sub["occ3d"].to_numpy(float)
    tba = sub["TotAccelBA"].to_numpy(float)
    _strip(ax_o, occ, leaves, OCC_CMAP, Normalize(0, 1), "3D occ")
    _strip(ax_t, tba, leaves, TBA_CMAP, Normalize(np.nanmin(tba), np.nanmax(tba)),
           "TBA")
    ax_t.set_xlabel("clusters (leaf order)")
    fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 1), OCC_CMAP),
                 cax=fig.add_subplot(gs[1, 1])).set_label("occ3d", fontsize=7)
    fig.colorbar(plt.cm.ScalarMappable(
        Normalize(np.nanmin(tba), np.nanmax(tba)), TBA_CMAP),
        cax=fig.add_subplot(gs[2, 1])).set_label("TBA", fontsize=7)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def summary_plot(st, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [("tba_median", "median TBA (per-week centroids)", "bradykinesia axis"),
              ("occ3d_eta2", "occ3d eta^2", "branch-arena segregation"),
              ("lowtba2d_frac", "low-TBA & 2D frame share", f"TBA<{TBA_LOW}, occ3d<{OCC_2D}")]
    for ax, (col, ylab, title) in zip(axes, panels):
        for mouse, g in st.groupby("mouse"):
            g = g.sort_values("week")
            ax.plot(g["week"], g[col], "-o", ms=4, color=MOUSE_COLORS.get(mouse),
                    label=mouse)
        ax.set(xlabel="week", ylabel=ylab, title=title)
        ax.grid(alpha=0.3)
    axes[0].legend(title="mouse", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5, help="branches per week tree (default 5)")
    ap.add_argument("--min-frames", type=int, default=15,
                    help="drop clusters with fewer frames that week (default 15)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    wf = pd.read_csv(ROOT / "cluster_week_features.csv")
    stats = []
    for (mouse, week), g in wf.groupby(["mouse", "week"], sort=True):
        batch = g["batch"].iloc[0]
        sub = week_centroids(g)
        sub = sub[sub["n_frames"] >= args.min_frames].reset_index(drop=True)
        if len(sub) < 6:
            print(f"  {mouse}/w{week}: only {len(sub)} clusters >= {args.min_frames} "
                  "frames, skipped")
            continue
        k = min(args.k, len(sub) - 1)
        X = zscore(sub[FEATURE_NAMES].to_numpy(float))
        Z = linkage(X, method="ward")
        leaves = dendrogram(Z, no_plot=True)["leaves"]
        branch = relabel_by_leaf_order(fcluster(Z, t=k, criterion="maxclust"), leaves)

        (OUT / mouse).mkdir(exist_ok=True)
        plot_week(sub, Z, leaves, k,
                  f"{mouse} week {week} ({batch}, {len(sub)} clusters -> {k} br.)",
                  OUT / mouse / f"w{week:02d}.png")

        low = sub[(sub["TotAccelBA"] < TBA_LOW) & (sub["occ3d"] < OCC_2D)]
        stats.append({
            "mouse": mouse, "week": int(week), "batch": batch,
            "n_clusters": len(sub),
            "tba_median": sub["TotAccelBA"].median(),
            "occ3d_eta2": eta_squared(sub["occ3d"], branch),
            "n_lowtba2d": len(low),
            "lowtba2d_frac": low["n_frames"].sum() / sub["n_frames"].sum(),
        })
        print(f"  {mouse}/w{week}: {len(sub)} clusters -> {k} branches, "
              f"TBA_med={sub['TotAccelBA'].median():.3f}, "
              f"lowTBA2D={len(low)}")

    st = pd.DataFrame(stats)
    st.to_csv(OUT / "week_stats.csv", index=False)
    summary_plot(st, OUT / "summary.png")
    print(f"\nWrote per-week trees under {OUT}/<mouse>/, week_stats.csv, summary.png")


if __name__ == "__main__":
    main()
