"""Feature-space embedding vs time: x = week, y = 1D MDS of feature distance,
color = mean total body acceleration (TBA) of the cluster. Places each cluster at
its temporal centroid with a thin line spanning its 10th-90th percentile weeks.

Reading the map:
    same y, different x   -> CHAIN (same behavior region, across time)
    overlapping in x AND y -> DEGENERACY (redundant, co-temporal)
    color (TBA) gradient along y -> the feature axis tracks vigor/acceleration

The y axis is a 1D metric-MDS embedding of the cluster x cluster feature distance
Dfeat (diagonal zeroed). 1D is very lossy for ~100 clusters (stress reported); the
embedding is per mouse -> y is NOT comparable across mice. Color is mean *linear*
TBA per cluster from StructData/func feature 3 (see feature_extraction.py).

Run (after presence_similarity.py + feature_similarity.py):
    C:/ProgramData/anaconda3/python.exe degeneracy_analysis/feature_time_map.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extraction import load_funct_features, bin_features, FEATURE_NAMES
from presence_similarity import MICE, DATA, OUT
from feature_similarity import _row_labels


def _mds(D, n, seed=0):
    """Metric MDS embedding + Kruskal stress-1 (normalized, comparable across mice)."""
    D = D.copy()
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    emb = MDS(n_components=n, dissimilarity="precomputed", random_state=seed,
              n_init=8, max_iter=500, normalized_stress=False).fit_transform(D)
    d_in = squareform(D, checks=False)
    d_emb = pdist(emb)
    stress1 = np.sqrt(((d_emb - d_in) ** 2).sum() / (d_in ** 2).sum())
    return emb, stress1


def _pctl_weeks(P, weeks, qs=(0.1, 0.9)):
    """Interpolated week at each cumulative-prob quantile, per cluster."""
    cdf = np.cumsum(P, axis=1)
    out = np.empty((P.shape[0], len(qs)))
    for i in range(P.shape[0]):
        out[i] = np.interp(qs, cdf[i], weeks)
    return out


def cluster_tba(mouse, lab, K):
    """Mean linear total body acceleration per cluster, over the kept windows."""
    binned = bin_features(load_funct_features(f"{DATA}/{mouse}/session_1_out.mat"))
    tba = binned[FEATURE_NAMES.index("TotAccelBA")]      # (N,), one per window
    if len(tba) != len(lab):
        raise ValueError(f"{mouse}: TBA windows {len(tba)} != detail rows {len(lab)}")
    return np.array([tba[lab == k].mean() for k in range(K)])


def analyze(mouse):
    pres = np.load(f"{OUT}/{mouse}/presence.npz", allow_pickle=True)
    feat = np.load(f"{OUT}/{mouse}/feature.npz", allow_pickle=True)
    clusters = pres["clusters"].astype(int)
    weeks = pres["weeks"].astype(int)
    P, cen = pres["presence"], pres["centroid"]
    Dfeat = feat["Dfeat"]

    emb1, s1 = _mds(Dfeat, 1)
    y = emb1[:, 0]
    # orient the (arbitrary-sign) feature axis to point with time, then quantify
    rho = spearmanr(y, cen).correlation
    if rho < 0:
        y, rho = -y, -rho

    lab = _row_labels(mouse, clusters, weeks)
    tba = cluster_tba(mouse, lab, len(clusters))
    rho_tba = spearmanr(y, tba).correlation
    span = _pctl_weeks(P, weeks)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for i in range(len(clusters)):
        ax.plot(span[i], [y[i], y[i]], color="0.75", lw=0.6, zorder=1)
    sc = ax.scatter(cen, y, c=tba, s=45, cmap="viridis", edgecolor="k",
                    linewidth=0.3, zorder=2)
    for i in range(len(clusters)):
        ax.annotate(str(clusters[i]), (cen[i], y[i]), fontsize=5,
                    ha="center", va="center", zorder=3)
    fig.colorbar(sc, ax=ax, label="mean total body acceleration")
    grp = "control" if mouse.endswith("lc") else "MitoPark"
    ax.set(xlabel="week (centroid; line = 10th-90th pctl temporal span)",
           ylabel="1D feature embedding (MDS, oriented with time)",
           title=f"{mouse} ({grp}): 1D feature embedding vs time   "
                 f"|corr(feat, week)|={rho:.2f}   |corr(feat, TBA)|={abs(rho_tba):.2f}   "
                 f"(stress={s1:.2f})")
    ax.set_xticks(weeks[::2])
    fig.tight_layout()
    fig.savefig(f"{OUT}/{mouse}/feature_time_map.png", dpi=130)
    plt.close(fig)
    print(f"{mouse:5s} ({grp:8s}): corr(feat,week)={rho:.2f}  corr(feat,TBA)={rho_tba:+.2f}  "
          f"stress1D={s1:.2f}  ({len(clusters)} clusters)")


if __name__ == "__main__":
    for m in MICE:
        if os.path.exists(f"{OUT}/{m}/feature.npz"):
            analyze(m)
