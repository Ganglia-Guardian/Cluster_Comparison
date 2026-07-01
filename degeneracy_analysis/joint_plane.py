"""Join the two axes into the cluster-degeneracy plane and read structure off it.

Per mouse, each point is a cluster PAIR (a, b), a<b:
    X  feature distance   Dfeat[a,b]  = -mean affinity   (0 == identical features)
    Y  presence distance  logW[a,b]   = log CDF-L1 over weeks (temporal displacement)

Interpretation by region (feature-similar = small X):
    small X, small Y   same features, same timing   -> DEGENERACY (redundant/merge)
    small X, large Y   same features, diff timing    -> CHAIN (behavior across time)
    large X, small Y   diff features, co-temporal
    large X, large Y   unrelated

Degeneracy is judged relative to each cluster's OWN affinity radius (self-distance):
    Dfeat_norm[a,b] = 2*Dfeat[a,b] / (self_d[a] + self_d[b])   (~1 => as close as a
    cluster is to itself -> truly degenerate).

Three views are produced per mouse:
    (1) the raw plane scatter (the point cloud), colored by |delta centroid|
    (2) the feature-similar CONDITIONAL slice: histogram of Y for low-X pairs;
        a low-Y mode = degeneracy, a high-Y mode = temporal chains
    (3) the unit-circle angular density: both axes rank->normal standardized, each
        pair projected to its angle; density of angles (the requested view)

Plus CSVs of the top degeneracy and chain candidate pairs.

Run (after presence_similarity.py and feature_similarity.py):
    C:/ProgramData/anaconda3/python.exe degeneracy_analysis/joint_plane.py
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from presence_similarity import MICE, OUT

FEAT_SIMILAR_PCT = 20     # "feature-similar" = pairs in the lowest this-% of Dfeat_norm
DEGEN_PRES_PCT = 25       # within feature-similar, low-Y (degenerate) vs high-Y (chain)


def load_pairs(mouse):
    pres = np.load(f"{OUT}/{mouse}/presence.npz", allow_pickle=True)
    feat = np.load(f"{OUT}/{mouse}/feature.npz", allow_pickle=True)
    cp, cf = pres["clusters"].astype(int), feat["clusters"].astype(int)
    if not np.array_equal(cp, cf):
        raise ValueError(f"{mouse}: presence/feature cluster sets differ")
    clusters = cp
    logW = pres["log_wasserstein"]
    delta = pres["delta"]
    cen = pres["centroid"]
    Dfeat = feat["Dfeat"]
    self_d = feat["self_dist"]
    Dnorm = 2 * Dfeat / (self_d[:, None] + self_d[None, :])

    iu, ju = np.triu_indices(len(clusters), k=1)
    return dict(
        mouse=mouse, clusters=clusters, cen=cen,
        a=iu, b=ju,
        Dfeat=Dfeat[iu, ju], Dnorm=Dnorm[iu, ju],
        logW=logW[iu, ju], delta=delta[iu, ju],
    )


def analyze(mouse):
    P = load_pairs(mouse)
    X, Y, Dnorm, delta = P["Dfeat"], P["logW"], P["Dnorm"], P["delta"]
    rho, _ = spearmanr(X, Y)

    # feature-similar column and its degeneracy/chain split
    xthr = np.percentile(Dnorm, FEAT_SIMILAR_PCT)
    fsim = Dnorm <= xthr
    ythr = np.percentile(Y[fsim], DEGEN_PRES_PCT)
    degen = fsim & (Y <= ythr)
    chain = fsim & (Y > np.percentile(Y[fsim], 100 - DEGEN_PRES_PCT))

    _plot(mouse, P, fsim, degen, chain, xthr, rho)
    _tables(mouse, P, degen, chain)

    print(f"\n{'='*68}\n{mouse}: joint plane\n{'='*68}")
    print(f"  pairs: {len(X)}   Spearman(Dfeat, logW) = {rho:+.3f}  "
          f"({'axes carry independent info' if abs(rho) < 0.5 else 'axes correlated -> plane collapsing'})")
    print(f"  feature-similar (Dnorm <= {xthr:.2f}, lowest {FEAT_SIMILAR_PCT}%): {fsim.sum()} pairs")
    print(f"    -> degeneracy candidates (also low presence dist): {degen.sum()}")
    print(f"    -> chain candidates (feature-similar, temporally displaced): {chain.sum()}")
    return P


def _plot(mouse, P, fsim, degen, chain, xthr, rho):
    X, Y, delta = P["Dfeat"], P["logW"], P["delta"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    sc = ax[0].scatter(X, Y, c=np.abs(delta), s=8, alpha=0.5, cmap="viridis")
    ax[0].scatter(X[degen], Y[degen], s=22, facecolors="none", edgecolors="red", lw=0.8, label="degeneracy")
    ax[0].scatter(X[chain], Y[chain], s=22, facecolors="none", edgecolors="orange", lw=0.8, label="chain")
    ax[0].set(xscale="log", xlabel="feature distance Dfeat (log)", ylabel="presence dist  log CDF-L1",
              title=f"{mouse}: the plane (rho={rho:+.2f})")
    fig.colorbar(sc, ax=ax[0], label="|delta centroid| (weeks)")
    ax[0].legend(fontsize=7, loc="lower right")

    # (2) feature-similar conditional slice
    ax[1].hist(Y[fsim], bins=40, color="#69c", edgecolor="none")
    ax[1].axvline(np.percentile(Y[fsim], 25), color="red", ls=":", label="degeneracy side")
    ax[1].axvline(np.percentile(Y[fsim], 75), color="orange", ls=":", label="chain side")
    ax[1].set(xlabel="presence dist  log CDF-L1", ylabel="feature-similar pairs",
              title=f"{mouse}: conditional slice (X in lowest {FEAT_SIMILAR_PCT}%)")
    ax[1].legend(fontsize=7)

    # (3) unit-circle angular density (rank->normal standardized axes)
    xz = norm.ppf((rankdata(X) - 0.5) / len(X))
    yz = norm.ppf((rankdata(Y) - 0.5) / len(Y))
    theta = np.arctan2(yz, xz)
    axp = fig.add_subplot(1, 3, 3, projection="polar")
    ax[2].remove()
    axp.hist(theta, bins=48, color="#3a7", alpha=0.85)
    axp.set_title(f"{mouse}: angular density\n(0=high Dfeat, pi/2=high presence dist)", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT}/{mouse}/joint_plane.png", dpi=115)
    plt.close(fig)


def _tables(mouse, P, degen, chain):
    cl, cen = P["clusters"], P["cen"]

    def rows(mask, order_desc):
        a, b = P["a"][mask], P["b"][mask]
        df = pd.DataFrame({
            "cluster_a": cl[a], "cluster_b": cl[b],
            "Dfeat": P["Dfeat"][mask], "Dnorm": P["Dnorm"][mask],
            "logW": P["logW"][mask], "delta_centroid": P["delta"][mask],
            "centroid_a": cen[a], "centroid_b": cen[b],
        })
        return df.sort_values("logW", ascending=not order_desc)

    rows(degen, False).to_csv(f"{OUT}/{mouse}/degeneracy_candidates.csv", index=False)
    rows(chain, True).to_csv(f"{OUT}/{mouse}/chain_candidates.csv", index=False)


if __name__ == "__main__":
    import os
    for m in MICE:
        if os.path.exists(f"{OUT}/{m}/feature.npz"):
            analyze(m)
