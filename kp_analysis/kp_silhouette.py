"""Silhouette of the pose clustering, and a transitional-vs-sustained split.

Pose space is continuous, so windows caught mid-transition sit between cluster
cores and blur the boundaries. We test that intuition directly:
  * silhouette_samples on the pose embedding  -> per-window "core-ness"
  * pose-change rate = mean |deformation rate| across the 105 distance pairs
    (the motion feature) -> a continuous per-window "transitional-ness" score
If the intuition holds, transitional (high pose-change) windows have lower
silhouette. We report the rank correlation, silhouette by transitional tercile,
and whether restricting to sustained windows sharpens both the clustering
(silhouette) and the func alignment (AMI). A window is flagged TRANSITIONAL if
its pose-change is in the top third OR its silhouette is negative.

Run from repo root:
    C:/ProgramData/anaconda3/python.exe kp_analysis/kp_silhouette.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import adjusted_mutual_info_score as ami
from scipy.stats import spearmanr

import kp_cluster_compare as cc

K = 30


def run(session):
    post, mot, comb, func = cc.build_windows(session)
    Xp = cc.embed(post)
    labels = KMeans(K, n_init=10, random_state=0).fit_predict(Xp)
    sil = silhouette_samples(Xp, labels)
    change = mot.mean(axis=1)            # per-window pose-change rate

    overall = silhouette_score(Xp, labels)
    frac_neg = float((sil < 0).mean())
    rho, prho = spearmanr(change, sil)

    # silhouette by transitional tercile (low / mid / high pose-change)
    q = np.quantile(change, [1 / 3, 2 / 3])
    tert = np.digitize(change, q)
    tert_sil = [float(sil[tert == g].mean()) for g in range(3)]

    # sustained (low-change) half vs transitional (high-change) half
    med = np.median(change)
    lo, hi = change <= med, change > med
    sil_sus = silhouette_score(Xp[lo], labels[lo])
    sil_tra = silhouette_score(Xp[hi], labels[hi])
    ami_full = ami(func, labels)
    ami_sus = ami(func[lo], labels[lo])
    ami_tra = ami(func[hi], labels[hi])

    print(f"\n{'='*64}\n{session}  (K={K}, {len(func)} windows)\n{'='*64}")
    print(f"  overall silhouette        = {overall:+.3f}")
    print(f"  windows with negative sil = {frac_neg:.1%}  (boundary/transitional)")
    print(f"  spearman(change, sil)     = {rho:+.3f}  (p={prho:.1e})")
    print(f"  silhouette by change tercile  low={tert_sil[0]:+.3f} "
          f"mid={tert_sil[1]:+.3f} high={tert_sil[2]:+.3f}")
    print(f"  silhouette  sustained-half={sil_sus:+.3f}  transitional-half={sil_tra:+.3f}")
    print(f"  func AMI     full={ami_full:.3f}  sustained={ami_sus:.3f}  "
          f"transitional={ami_tra:.3f}")

    # per pose cluster: crisp (sustained) vs blurry, and how transitional it is
    print("  crispest / blurriest pose clusters (mean silhouette):")
    means = [(c, sil[labels == c].mean(), change[labels == c].mean(),
              int((labels == c).sum())) for c in range(K)]
    means.sort(key=lambda r: r[1])
    for tag, rows in [("blurry", means[:3]), ("crisp", means[-3:][::-1])]:
        for c, s, ch, n in rows:
            print(f"    [{tag:6s}] pose {c:2d}  sil={s:+.3f}  change={ch:.3f}  n={n}")

    _plot(session, sil, change, labels)
    return dict(session=session, silhouette=overall, frac_neg=frac_neg, rho=rho,
                ami_full=ami_full, ami_sustained=ami_sus, ami_transitional=ami_tra)


def _plot(session, sil, change, labels):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    hb = ax[0].hexbin(change, sil, gridsize=40, cmap="viridis", mincnt=1)
    ax[0].axhline(0, color="r", lw=0.8, ls="--")
    ax[0].set_xlabel("pose-change rate (transitional-ness)")
    ax[0].set_ylabel("silhouette")
    ax[0].set_title(f"{session}: silhouette vs transitional-ness")
    fig.colorbar(hb, ax=ax[0], label="windows")
    ax[1].hist(sil, bins=50, color="steelblue")
    ax[1].axvline(0, color="r", lw=0.8, ls="--")
    ax[1].set_xlabel("silhouette")
    ax[1].set_ylabel("windows")
    ax[1].set_title("silhouette distribution")
    fig.tight_layout()
    fig.savefig(f"kp_analysis/{session}_silhouette.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    for session in cc.SESSIONS:
        run(session)
    print("\nSaved per-session *_silhouette.png")
