"""Rarefaction of the pose-grouping structure (kp_pose_grouping).

The grouping headline -- Cramer's V of the IMU x pose table and the number of
pose pairs co-recruited beyond chance -- both depend on how many windows we feed
in (chi-square grows with N; rare IMU clusters vanish at low depth). Rarefaction
subsamples windows at increasing depths (pose & func labels FIXED from the full
fit) and recomputes those two statistics, so we can see whether the structure is
saturated at the real sample size or still accumulating, and compare wk8lc vs
wk8mp at equal depth.

For each depth M we draw `reps` random subsets without replacement; for each we
compute Cramer's V and run a per-subset permutation null (shuffle pose labels) to
count pose pairs with co-recruitment z > 2.58. Curves show mean +/- 5-95 pct band.

Run from repo root:
    C:/ProgramData/anaconda3/python.exe kp_analysis/kp_pose_grouping_rarefaction.py
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
from scipy.stats import chi2_contingency

import kp_cluster_compare as cc
from kp_pose_grouping import _count_matrix, _residual_profiles, _profile_corr, K

REPS = 12
N_PERM = 200
DEPTHS = [500, 1000, 1500, 2000, 3000, 4000, 5000]
IU = np.triu_indices(K, 1)


def _prep(session):
    post, mot, comb, func = cc.build_windows(session)
    pose = KMeans(K, n_init=10, random_state=0).fit_predict(cc.embed(post))
    f_idx, funcs = pd.factorize(func)
    return pose, f_idx, len(funcs)


def _stats(pose_s, f_idx_s, n_imu, rng, n_perm):
    C = _count_matrix(f_idx_s, pose_s, n_imu, K)
    Cf = C[C.sum(1) > 0][:, C.sum(0) > 0]
    chi2 = chi2_contingency(Cf)[0]
    V = np.sqrt((chi2 / Cf.sum()) / (min(Cf.shape) - 1))
    S = _profile_corr(_residual_profiles(C))
    null = np.empty((n_perm, K, K))
    for t in range(n_perm):
        null[t] = _profile_corr(_residual_profiles(
            _count_matrix(f_idx_s, rng.permutation(pose_s), n_imu, K)))
    Z = (S - null.mean(0)) / (null.std(0) + 1e-9)
    return V, int((Z[IU] > 2.58).sum())


def rarefy(session):
    pose, f_idx, n_imu = _prep(session)
    N = len(pose)
    depths = sorted(set([d for d in DEPTHS if d < N] + [N]))
    rng = np.random.default_rng(0)
    rows = []
    for M in depths:
        reps = 1 if M == N else REPS
        Vs, Ps = [], []
        for _ in range(reps):
            sub = rng.choice(N, M, replace=False) if M < N else np.arange(N)
            V, nsig = _stats(pose[sub], f_idx[sub], n_imu, rng, N_PERM)
            Vs.append(V); Ps.append(nsig)
        rows.append({"depth": M, "cramers_v": np.mean(Vs), "v_lo": np.percentile(Vs, 5),
                     "v_hi": np.percentile(Vs, 95), "sig_pairs": np.mean(Ps),
                     "p_lo": np.percentile(Ps, 5), "p_hi": np.percentile(Ps, 95)})
    df = pd.DataFrame(rows)
    print(f"\n=== {session}  (N={N}, {n_imu} IMU clusters) ===")
    print(df.round(3).to_string(index=False))
    df.to_csv(f"kp_analysis/{session}_grouping_rarefaction.csv", index=False)
    return df


def _plot(results):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for session, df in results.items():
        ax[0].plot(df.depth, df.cramers_v, marker="o", label=session)
        ax[0].fill_between(df.depth, df.v_lo, df.v_hi, alpha=0.2)
        ax[1].plot(df.depth, df.sig_pairs, marker="o", label=session)
        ax[1].fill_between(df.depth, df.p_lo, df.p_hi, alpha=0.2)
    ax[0].set(xlabel="windows subsampled", ylabel="Cramer's V (IMU x pose)",
              title="effect size vs depth")
    ax[1].set(xlabel="windows subsampled",
              ylabel="co-recruited pose pairs (z>2.58)",
              title="grouping structure vs depth")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("kp_analysis/pose_grouping_rarefaction.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    results = {s: rarefy(s) for s in cc.SESSIONS}
    _plot(results)
    print("\nSaved kp_analysis/pose_grouping_rarefaction.png and *_grouping_rarefaction.csv")
