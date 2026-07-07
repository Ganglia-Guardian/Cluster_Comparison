"""Merge each MOUSE's clusters (pooled across its w8/w9/w10 batches) into one
dendrogram from their 4 mean kinematic features, cut it into a handful of
branches, and record the branch membership for downstream arena/TBA/week
analysis.

Pooling across batches: the 3 batches per mouse are separate clusterings over
interleaved weeks (w8->8,11,14,17,20,23; w9->9,12,...; w10->10,13,...), so
pooling their clusters gives full week coverage (8-24) and a more holistic tree.
Each cluster is only meaningful within its own batch's clustering, so a leaf is
identified by (batch, cluster) and we merge purely on kinematics -- a cluster's
source batch never enters the distance.

Distance basis (user choice): the 4 per-cluster mean func-features
    anterior_posterior_x_accel, dorsal_ventral_y_accel, y_gyro, TotAccelBA
z-scored WITHIN each mouse (across all its pooled clusters) so no single
feature's native scale dominates, then Ward linkage on Euclidean distance.

Because TBA is one of the merge features, "avg TBA per branch" is a partly
circular readout (branches are somewhat TBA-homogeneous by construction) -- the
branch analysis flags this. Arena purity (occ3d) and week distribution are NOT
in the distance, so they are independent readouts.

Each tree is cut into `--k` branches (default 6) via a maxclust cut and drawn
with three aligned strips under the leaves: source batch, 3D occupancy
(red=2D .. blue=3D), and mean TBA. Branch membership -> cluster_branches.csv.

Run:  uv run python cluster_group_analysis/dendrogram.py            # k=6
      uv run python cluster_group_analysis/dendrogram.py --k 8
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

from common import ROOT
from feature_extraction import FEATURE_NAMES

OUT = ROOT / "output" / "dendrogram"
OCC_CMAP = plt.cm.RdBu          # 0 -> red (2D), 1 -> blue (3D)
TBA_CMAP = plt.cm.viridis
BATCH_CMAP = plt.cm.tab10       # categorical, for the source-batch strip


def zscore(mat):
    """Column-wise z-score; a zero-variance column collapses to 0 (not NaN)."""
    mu = mat.mean(0)
    sd = mat.std(0)
    sd[sd == 0] = 1.0
    return (mat - mu) / sd


def cut_threshold(Z, k):
    """Link-color threshold that yields exactly k colored groups: the midpoint
    between the (k-1)th and kth largest merge heights."""
    n = Z.shape[0] + 1
    if k >= n or k < 2:
        return 0.0
    h = np.sort(Z[:, 2])
    return 0.5 * (h[n - k - 1] + h[n - k])


def relabel_by_leaf_order(branch, leaves):
    """Renumber branch ids 1..k left-to-right as they appear in the dendrogram
    leaf order, so branch numbers read naturally across the tree."""
    mapping, nxt = {}, 1
    for leaf in leaves:
        b = branch[leaf]
        if b not in mapping:
            mapping[b] = nxt
            nxt += 1
    return np.array([mapping[b] for b in branch])


def _strip(ax, values, leaves, cmap, norm, label):
    """Draw one leaf-aligned heatmap strip. scipy places leaf i at x=10*i+5, so
    the strip must span [0, 10*n] to line up under the dendrogram (shared x)."""
    n = len(leaves)
    ax.imshow(values[leaves][None, :], aspect="auto", cmap=cmap, norm=norm,
              extent=[0, 10 * n, 0, 1], interpolation="nearest")
    ax.set_yticks([])
    ax.tick_params(labelbottom=False)      # only the bottom strip labels leaves
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)


def plot_tree(sub, Z, branch, leaves, k, batches, title, path):
    """Dendrogram + aligned batch / occ3d / TBA strips, with colorbars in a
    dedicated right-hand column so no strip is shrunk out of alignment."""
    n = len(leaves)
    fig = plt.figure(figsize=(max(9, 0.16 * n), 6.6))
    gs = fig.add_gridspec(4, 2, width_ratios=[60, 1],
                          height_ratios=[6, 0.45, 0.55, 0.55],
                          hspace=0.1, wspace=0.02)
    ax_d = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_d)
    ax_o = fig.add_subplot(gs[2, 0], sharex=ax_d)
    ax_t = fig.add_subplot(gs[3, 0], sharex=ax_d)

    dendrogram(Z, ax=ax_d, color_threshold=cut_threshold(Z, k),
               above_threshold_color="#999999", no_labels=True)
    ax_d.set(ylabel="Ward distance", title=title)
    ax_d.tick_params(labelbottom=False)
    ax_d.spines[["top", "right"]].set_visible(False)

    # source-batch strip (categorical)
    bidx = np.array([batches.index(b) for b in sub["batch"]])
    bcmap = ListedColormap([BATCH_CMAP(i) for i in range(len(batches))])
    bnorm = BoundaryNorm(np.arange(-0.5, len(batches) + 0.5), len(batches))
    _strip(ax_b, bidx.astype(float), leaves, bcmap, bnorm, "batch")
    occ = sub["occ3d"].to_numpy(float)
    tba = sub["TotAccelBA"].to_numpy(float)
    _strip(ax_o, occ, leaves, OCC_CMAP, Normalize(0, 1), "3D occ")
    _strip(ax_t, tba, leaves, TBA_CMAP, Normalize(np.nanmin(tba), np.nanmax(tba)),
           "TBA")

    ax_t.set_xlabel("clusters in leaf order  (batch:id mapping in cluster_branches.csv)")

    # colorbars in the reserved right column (do not steal strip width)
    fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 1), OCC_CMAP),
                 cax=fig.add_subplot(gs[2, 1])).set_label("occ3d", fontsize=7)
    fig.colorbar(plt.cm.ScalarMappable(
        Normalize(np.nanmin(tba), np.nanmax(tba)), TBA_CMAP),
        cax=fig.add_subplot(gs[3, 1])).set_label("TBA", fontsize=7)
    # batch legend on the dendrogram
    ax_d.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BATCH_CMAP(i))
                         for i in range(len(batches))],
                labels=batches, title="batch", fontsize=8, loc="upper right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=6, help="branches per tree (default 6)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(ROOT / "cluster_features.csv")
    out_parts = []
    for mouse, sub in feats.groupby("mouse", sort=False):
        sub = sub.reset_index(drop=True)
        batches = sorted(sub["batch"].unique(), key=lambda b: int(b[1:]))
        if len(sub) <= args.k:
            print(f"  {mouse}: only {len(sub)} clusters <= k, skipped")
            continue
        X = zscore(sub[FEATURE_NAMES].to_numpy(float))
        Z = linkage(X, method="ward")
        leaves = dendrogram(Z, no_plot=True)["leaves"]
        branch = relabel_by_leaf_order(
            fcluster(Z, t=args.k, criterion="maxclust"), leaves)

        plot_tree(sub, Z, branch, leaves, args.k, batches,
                  f"{mouse}  ({len(sub)} clusters from {'+'.join(batches)} "
                  f"-> {args.k} branches)",
                  OUT / f"{mouse}_dendro.png")

        rec = sub.copy()
        rec["branch"] = branch
        out_parts.append(rec)
        print(f"  {mouse}: {len(sub)} clusters ({'+'.join(batches)}) -> "
              f"{len(np.unique(branch))} branches")

    allb = pd.concat(out_parts, ignore_index=True)
    cols = ["mouse", "batch", "branch", "cluster", "n_frames", "n_2d", "n_3d",
            "occ3d", *FEATURE_NAMES]
    allb = allb[cols].sort_values(["mouse", "branch", "batch", "cluster"])
    allb.to_csv(ROOT / "cluster_branches.csv", index=False)
    print(f"\nWrote {ROOT/'cluster_branches.csv'} ({len(allb)} clusters) and "
          f"per-mouse dendrograms in {OUT}")


if __name__ == "__main__":
    main()
