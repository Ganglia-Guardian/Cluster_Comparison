"""Condition on each IMU func cluster, then sub-cluster the poses beneath it.

For each func (IMU) cluster we take its member windows in the pose embedding and
run HDBSCAN -- density-based, so it surfaces a few dense pose modes and labels
the rest as noise (-1), which is exactly the "a few large clusters + some noise"
shape hoped for. Per func cluster we report the number of pose modes, the share
of windows in the largest / top-3 modes, and the noise fraction, then flag the
"clean" ones (few modes, high coverage, low noise).

As a synthesis we also correlate sub-cluster cleanliness against the global
posture<->func alignment (lift, from kp_aligned_clusters): well-aligned IMU
behaviors should resolve into tighter pose modes.

Run from repo root (after kp_aligned_clusters.py, for the merge):
    C:/ProgramData/anaconda3/python.exe kp_analysis/kp_conditional_pose.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from scipy.stats import spearmanr

import kp_cluster_compare as cc

MIN_WINDOWS = 40        # need enough windows to look for sub-structure


def run(session):
    post, mot, comb, func = cc.build_windows(session)
    Xp = cc.embed(post)

    rows = []
    for fc in np.unique(func):
        idx = np.where(func == fc)[0]
        n = len(idx)
        if n < MIN_WINDOWS:
            continue
        mcs = max(5, int(0.10 * n))
        lab = HDBSCAN(min_cluster_size=mcs).fit_predict(Xp[idx])
        noise = float((lab == -1).mean())
        sizes = np.sort(np.bincount(lab[lab >= 0])) [::-1] if (lab >= 0).any() else np.array([])
        n_modes = len(sizes)
        rows.append({
            "func": int(fc), "n": n, "mcs": mcs, "n_modes": n_modes,
            "top1_share": float(sizes[0] / n) if n_modes else 0.0,
            "top3_share": float(sizes[:3].sum() / n) if n_modes else 0.0,
            "noise_frac": noise, "mode_sizes": sizes[:5].tolist(),
        })
    df = pd.DataFrame(rows)
    df["clean"] = ((df.n_modes.between(1, 4)) & (df.top3_share >= 0.4)
                   & (df.noise_frac <= 0.6))

    # merge global alignment lift, if the aligned-clusters table exists
    apath = f"kp_analysis/{session}_aligned_clusters.csv"
    if os.path.exists(apath):
        al = pd.read_csv(apath)[["func", "purity", "lift"]]
        df = df.merge(al, on="func", how="left")

    print(f"\n{'='*72}\n{session}  ({len(df)} func clusters with >= {MIN_WINDOWS} windows)\n{'='*72}")
    print(f"  clean (few modes, >=40% covered, <=60% noise): {int(df.clean.sum())}/{len(df)}")
    print(f"  median noise fraction = {df.noise_frac.median():.2f}   "
          f"median n_modes = {df.n_modes.median():.0f}")
    if "lift" in df:
        m = df.dropna(subset=["lift"])
        if len(m) > 3:
            r1, p1 = spearmanr(m.top1_share, m.lift)
            r2, p2 = spearmanr(m.noise_frac, m.lift)
            print(f"  spearman(top1_share, lift)={r1:+.2f} (p={p1:.1e})   "
                  f"spearman(noise_frac, lift)={r2:+.2f} (p={p2:.1e})")

    cols = ["func", "n", "n_modes", "top1_share", "top3_share", "noise_frac",
            "mode_sizes"] + (["lift"] if "lift" in df else [])
    show = df.sort_values("top3_share", ascending=False).head(12)[cols].copy()
    for c in ["top1_share", "top3_share", "noise_frac", "lift"]:
        if c in show:
            show[c] = show[c].round(3)
    print("  cleanest func clusters (highest top-3 pose-mode coverage):")
    print(show.to_string(index=False))

    df.to_csv(f"kp_analysis/{session}_conditional_pose.csv", index=False)
    return df


if __name__ == "__main__":
    for session in cc.SESSIONS:
        run(session)
    print("\nSaved per-session *_conditional_pose.csv")
