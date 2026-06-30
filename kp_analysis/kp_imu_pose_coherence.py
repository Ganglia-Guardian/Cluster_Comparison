"""Does pose-recruitment similarity between IMU clusters reflect their IMU-space
similarity, or are the two independent?

Each IMU (func) cluster gets:
  * a POSE PROFILE = P(pose | imu) over the 30 pose clusters -- the "pose mode
    set" it recruits.
  * a position in IMU space via the provided window x window IMU similarity,
    aggregated to an IMU-cluster x IMU-cluster mean-similarity matrix.
We Mantel-test (permutation) IMU-space distance against pose-profile distance
(Jensen-Shannon). Positive correlation => IMU clusters that are similar in IMU
space recruit similar postures (cross-modally coherent); ~0 => the pose mode set
an IMU behavior recruits is independent of where it sits in IMU space.

Run from repo root:
    C:/ProgramData/anaconda3/python.exe kp_analysis/kp_imu_pose_coherence.py \
        --sim <imu_sim.npy> --session wk8lc
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

import kp_cluster_compare as cc

K = 30
MIN_WINDOWS = 20
N_PERM = 2000
REST_SHARE_THRESH = 0.60    # IMU clusters with >= this rest occupancy are dropped


def _rest_poses(pose, change):
    """Rest postures = large AND low-motion (size > median, pose-change in the
    bottom quartile). Returns a set of pose-cluster ids."""
    sizes = np.array([(pose == c).sum() for c in range(K)])
    ch = np.array([change[pose == c].mean() if (pose == c).any() else np.inf
                   for c in range(K)])
    big = sizes > np.median(sizes)
    lowmo = ch <= np.quantile(ch[np.isfinite(ch)], 0.25)
    return set(np.where(big & lowmo)[0].tolist())


def analyze(session, sim, exclude_rest=False):
    post, mot, comb, func = cc.build_windows(session)
    pose = KMeans(K, n_init=10, random_state=0).fit_predict(cc.embed(post))
    change = mot.mean(axis=1)
    N = len(func)
    if sim.shape != (N, N):
        raise ValueError(f"sim {sim.shape} != ({N},{N})")

    rest = _rest_poses(pose, change)
    keep_pose = np.array([c for c in range(K) if c not in rest]) if exclude_rest \
        else np.arange(K)

    funcs = []
    for c in np.unique(func):
        ix = func == c
        if ix.sum() < MIN_WINDOWS:
            continue
        if exclude_rest:
            rest_share = np.isin(pose[ix], list(rest)).mean()
            if rest_share >= REST_SHARE_THRESH:
                continue            # drop rest-dominated IMU clusters
        funcs.append(c)
    k = len(funcs)
    idxs = [np.where(func == c)[0] for c in funcs]

    # pose-recruitment profiles P(pose | imu), optionally over non-rest poses only
    P = np.zeros((k, len(keep_pose)))
    for i, ix in enumerate(idxs):
        h = np.bincount(pose[ix], minlength=K).astype(float)[keep_pose]
        P[i] = h / h.sum() if h.sum() > 0 else h

    # IMU-cluster x IMU-cluster mean similarity (off-diagonal blocks of sim)
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            m = sim[np.ix_(idxs[i], idxs[j])].mean()
            M[i, j] = M[j, i] = m

    # distances
    D_imu = M.max() - M
    np.fill_diagonal(D_imu, 0.0)
    D_pose = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = jensenshannon(P[i], P[j])      # JS distance (sqrt of divergence)
            D_pose[i, j] = D_pose[j, i] = d

    iu = np.triu_indices(k, 1)
    a, b = D_imu[iu], D_pose[iu]
    r0 = spearmanr(a, b).correlation

    rng = np.random.default_rng(0)
    null = np.empty(N_PERM)
    for t in range(N_PERM):
        p = rng.permutation(k)
        null[t] = spearmanr(a, D_pose[np.ix_(p, p)][iu]).correlation
    pval = (np.sum(np.abs(null) >= abs(r0)) + 1) / (N_PERM + 1)

    tag = "EXCLUDING rest family" if exclude_rest else "all clusters"
    print(f"\n{'='*66}\n{session}: IMU-space vs pose-recruitment similarity ({tag})\n{'='*66}")
    if exclude_rest:
        print(f"  rest poses dropped: {sorted(rest)}; "
              f"IMU clusters kept: {k} (rest-dominated removed)")
    print(f"  {k} IMU clusters (>= {MIN_WINDOWS} windows), {len(a)} pairs")
    print(f"  Mantel spearman(IMU-distance, pose-profile-distance) = {r0:+.3f}")
    print(f"  permutation p = {pval:.4f}  (null mean {null.mean():+.3f}, "
          f"sd {null.std():.3f})")
    verdict = ("COHERENT: IMU-similar clusters recruit similar poses"
               if (pval < 0.05 and r0 > 0) else
               "INDEPENDENT: pose mode set ~ unrelated to IMU similarity"
               if pval >= 0.05 else
               "INVERSE (unexpected)")
    print(f"  -> {verdict}")

    # concrete: among the most IMU-similar pairs, how alike are their pose sets?
    near = np.argsort(a)[:max(10, len(a) // 50)]      # closest in IMU space
    far = np.argsort(a)[-max(10, len(a) // 50):]      # farthest in IMU space
    print(f"  mean pose-distance among IMU-nearest pairs  = {b[near].mean():.3f}")
    print(f"  mean pose-distance among IMU-farthest pairs = {b[far].mean():.3f}")
    print(f"  (overall mean pose-distance = {b.mean():.3f})")

    _plot(session, a, b, r0, pval, exclude_rest)
    return dict(session=session, k=k, mantel_r=r0, p=pval, exclude_rest=exclude_rest)


def _plot(session, a, b, r0, pval, exclude_rest):
    suffix = "_norest" if exclude_rest else ""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(a, b, gridsize=35, cmap="viridis", mincnt=1)
    ax.set_xlabel("IMU-space distance (1 - mean similarity)")
    ax.set_ylabel("pose-profile distance (Jensen-Shannon)")
    ax.set_title(f"{session}{' (no rest)' if exclude_rest else ''}: "
                 f"IMU vs pose-recruitment\nMantel rho={r0:+.3f}, p={pval:.3f}")
    fig.tight_layout()
    fig.savefig(f"kp_analysis/{session}_imu_pose_coherence{suffix}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--session", required=True, choices=list(cc.SESSIONS))
    ap.add_argument("--exclude-rest", action="store_true")
    args = ap.parse_args()
    analyze(args.session, np.load(args.sim), exclude_rest=args.exclude_rest)
    print("\nSaved *_imu_pose_coherence.png")
