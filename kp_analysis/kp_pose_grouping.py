"""Are the pose modes recruited by IMU clusters grouped by chance, or do some
pose clusters co-occur under the same IMU behaviors far more than chance?

Method
------
1. Global association: chi-square on the IMU(func) x pose contingency table, with
   Cramer's V as effect size. Tests whether pose occupancy depends on IMU cluster
   at all (vs independence / pure chance).
2. Co-recruitment: describe each pose cluster by its column of *standardized
   Pearson residuals* over IMU clusters (how enriched/depleted it is in each IMU
   behavior, vs the independence expectation). Two pose clusters co-recruit when
   their residual profiles correlate -- they are enriched in the SAME IMU
   behaviors. A permutation null (shuffle pose labels, keep func fixed) gives a
   per-pair z-score, so we can say which pairs group beyond chance.
3. Grouping: hierarchically cluster the pose clusters on 1 - profile-correlation
   (average linkage) and plot the dendrogram, cutting into super-groups.
4. Cross-reference: Mantel-style rank correlation between co-recruitment distance
   and pose-centroid GEOMETRIC distance -- do co-recruited poses also look alike?

Run from repo root (uses pose geometry labels from kp_aligned_clusters):
    C:/ProgramData/anaconda3/python.exe kp_analysis/kp_pose_grouping.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency, spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import kp_cluster_compare as cc
import kp_aligned_clusters as ac

K = 30
N_PERM = 1000
N_GROUPS = 6


def _count_matrix(f_idx, pose, n_imu, k):
    M = np.zeros((n_imu, k))
    np.add.at(M, (f_idx, pose), 1)
    return M


def _residual_profiles(C):
    """Standardized Pearson residuals; columns are per-pose profiles over IMU."""
    tot = C.sum()
    E = C.sum(1, keepdims=True) * C.sum(0, keepdims=True) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        res = (C - E) / np.sqrt(E)
    return np.nan_to_num(res)


def _profile_corr(res):
    S = np.corrcoef(res.T)
    S = np.nan_to_num(S)            # constant (empty) columns -> 0
    np.fill_diagonal(S, 0.0)
    return S


def analyze(session):
    post, mot, comb, func = cc.build_windows(session)
    Xp = cc.embed(post)
    pose = KMeans(K, n_init=10, random_state=0).fit_predict(Xp)
    labels = ac.pair_names(session)
    geo = ac.describe_pose(post, pose, list(range(K)), labels)

    f_idx, funcs = pd.factorize(func)
    n_imu = len(funcs)
    C = _count_matrix(f_idx, pose, n_imu, K)

    chi2, p, dof, _ = chi2_contingency(C)
    cramers_v = np.sqrt((chi2 / C.sum()) / (min(C.shape) - 1))

    S = _profile_corr(_residual_profiles(C))

    # permutation null for co-recruitment correlation
    rng = np.random.default_rng(0)
    null = np.empty((N_PERM, K, K))
    for t in range(N_PERM):
        null[t] = _profile_corr(_residual_profiles(
            _count_matrix(f_idx, rng.permutation(pose), n_imu, K)))
    mu, sd = null.mean(0), null.std(0) + 1e-9
    Z = (S - mu) / sd

    iu = np.triu_indices(K, 1)
    z_pairs = Z[iu]
    n_pairs = len(z_pairs)
    n_sig_pos = int((z_pairs > 2.58).sum())          # p<0.01, co-grouped
    n_sig_neg = int((z_pairs < -2.58).sum())          # avoid same IMU clusters
    exp_chance = 0.005 * n_pairs                       # one tail of p<0.01

    # geometry cross-reference (Mantel: co-recruit distance vs centroid distance)
    cent = np.vstack([Xp[pose == c].mean(0) for c in range(K)])
    d_geom = pdist(cent)
    d_core = squareform(np.clip(1.0 - S, 0, 2), checks=False)
    rho_mantel, p_mantel = spearmanr(d_core, d_geom)

    # dendrogram + super-groups on co-recruitment distance
    Zlink = linkage(d_core, method="average")
    groups = fcluster(Zlink, N_GROUPS, criterion="maxclust")
    _dendro(session, Zlink, geo, pose)

    print(f"\n{'='*70}\n{session}  (K={K} pose, {n_imu} IMU clusters, {len(func)} windows)\n{'='*70}")
    print(f"  chi-square IMU x pose : chi2={chi2:,.0f}  dof={dof}  p={p:.1e}  "
          f"Cramer's V={cramers_v:.3f}")
    print(f"  co-recruited pose pairs beyond chance (|z|>2.58, p<0.01):")
    print(f"     positive (co-grouped) = {n_sig_pos}/{n_pairs}   "
          f"negative (anti) = {n_sig_neg}/{n_pairs}   chance~{exp_chance:.0f} each tail")
    print(f"  geometry cross-ref (Mantel rho, co-recruit dist vs centroid dist) = "
          f"{rho_mantel:+.3f} (p={p_mantel:.1e})")

    # strongest co-grouped pairs
    order = np.argsort(z_pairs)[::-1][:8]
    print("  top co-grouped pose pairs (z, then their distinctive segments):")
    for o in order:
        a, b = iu[0][o], iu[1][o]
        ga = geo[a]["stretched"][0][0]
        gb = geo[b]["stretched"][0][0]
        print(f"     pose {a:2d} ~ pose {b:2d}  z={z_pairs[o]:5.1f}  "
              f"[{ga}] / [{gb}]")

    # super-group membership
    print(f"  {N_GROUPS} pose super-groups (cut of the dendrogram):")
    for g in sorted(set(groups)):
        members = [c for c in range(K) if groups[c] == g]
        sizes = sum((pose == c).sum() for c in members)
        tags = ", ".join(f"P{c}:{geo[c]['stretched'][0][0]}" for c in members[:4])
        print(f"     group {g} (n_pose={len(members)}, {sizes} windows): {tags}"
              + (" ..." if len(members) > 4 else ""))

    pd.DataFrame(Z, columns=[f"P{c}" for c in range(K)],
                 index=[f"P{c}" for c in range(K)]).to_csv(
        f"kp_analysis/{session}_pose_corecruit_z.csv")
    return dict(session=session, cramers_v=cramers_v, n_sig_pos=n_sig_pos,
                rho_mantel=rho_mantel)


def _dendro(session, Zlink, geo, pose):
    labels = [f"P{c}·{geo[c]['stretched'][0][0]}·n{int((pose == c).sum())}"
              for c in range(K)]
    fig, ax = plt.subplots(figsize=(10, 9))
    dendrogram(Zlink, labels=labels, orientation="right", ax=ax,
               color_threshold=0.7 * Zlink[:, 2].max(), leaf_font_size=7)
    ax.set_title(f"{session}: pose clusters grouped by IMU co-recruitment\n"
                 "(distance = 1 - residual-profile correlation)")
    ax.set_xlabel("co-recruitment distance")
    fig.tight_layout()
    fig.savefig(f"kp_analysis/{session}_pose_dendrogram.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    for session in cc.SESSIONS:
        analyze(session)
    print("\nSaved per-session *_pose_dendrogram.png and *_pose_corecruit_z.csv")
