"""Cluster the internal-geometry keypoint features and compare them, per window,
to the existing IMU/func clustering via adjusted mutual information.

For each session (wk8lc<->1lc, wk8mp<->1mp) we:
    1. build per-frame posture (pairwise distances) and motion (deformation
       rate) features            [kp_features]
    2. aggregate them onto the func window grid (shared-start, 30 fps)
    3. cluster each feature set (standardize -> PCA -> KMeans), sweeping K
    4. score the keypoint partition against the func ClusterIdx partition with
       adjusted_mutual_info_score (chance-adjusted, so comparing across K is fair)

AMI ~ 0 means the geometry clusters carry no information about the func clusters;
higher means the internal geometry recovers the accelerometer-based behavior
structure. A label-shuffle baseline is reported as a floor.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

import kp_features as kpf

SESSIONS = {
    "wk8lc": ("kp_analysis/data/wk8lc/multicam_3d_results.csv",
              "data/2lc/Cluster_detail_results.csv", "w8"),
    "wk8mp": ("kp_analysis/data/wk8mp/multicam_3d_results.csv",
              "data/2mp/Cluster_detail_results.csv", "week8"),
}


def build_windows(session):
    """Return (posture, motion, combined, func_labels) aligned on valid windows.

    Each feature array is (n_valid_windows, n_features); func_labels is the
    matching func ClusterIdx per window. Windows with no frames are dropped.
    """
    kp_csv, func_csv, folder = SESSIONS[session]
    coords, _, _ = kpf.load_keypoints(kp_csv)
    dists, _ = kpf.pairwise_distances(coords)
    norm, _ = kpf.normalize_scale(dists)
    motion = kpf.motion_features(norm)

    win = kpf.load_func_windows(func_csv, folder)
    owner = kpf.assign_frames_to_windows(len(coords), win)
    post_w = kpf.aggregate_to_windows(norm, owner, len(win), "mean")
    mot_w = kpf.aggregate_to_windows(motion, owner, len(win), "mean_abs")

    valid = ~np.isnan(post_w).any(axis=1)
    post_w, mot_w = post_w[valid], mot_w[valid]
    labels = win.cluster.to_numpy()[valid]
    combined = np.hstack([post_w, mot_w])
    return post_w, mot_w, combined, labels


def embed(X, pca_var=0.95):
    """standardize -> PCA(retain pca_var fraction of variance). Returns the
    embedding the clustering / silhouette operate in."""
    Xs = StandardScaler().fit_transform(X)
    return PCA(n_components=pca_var, svd_solver="full").fit_transform(Xs)


def cluster(X, k, pca_var=0.95, seed=0):
    """standardize -> PCA(retain pca_var) -> KMeans(k). Returns label array."""
    Xp = embed(X, pca_var)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xp)


def compare_session(session, k_list):
    post, mot, comb, func = build_windows(session)
    n_func = len(np.unique(func))
    feats = {"posture": post, "motion": mot, "posture+motion": comb}

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(func)
    floor = adjusted_mutual_info_score(func, shuffled)

    rows = []
    for name, X in feats.items():
        for k in k_list:
            lab = cluster(X, k)
            rows.append({
                "session": session, "features": name, "k": k,
                "AMI": adjusted_mutual_info_score(func, lab),
                "NMI": normalized_mutual_info_score(func, lab),
            })
    df = pd.DataFrame(rows)
    return df, n_func, floor


def contingency_plot(session, k, out_png):
    post, mot, comb, func = build_windows(session)
    lab = cluster(comb, k)
    ct = pd.crosstab(lab, func)
    # row-normalize so each keypoint cluster shows its func-label composition
    frac = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(min(18, 0.3 * ct.shape[1] + 3), 0.35 * k + 2))
    im = ax.imshow(frac.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("func ClusterIdx")
    ax.set_ylabel("keypoint cluster")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns, fontsize=6, rotation=90)
    ax.set_title(f"{session}: keypoint(posture+motion, k={k}) vs func composition")
    fig.colorbar(im, ax=ax, label="row fraction")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print("Saved:", out_png)


if __name__ == "__main__":
    k_list = [5, 8, 10, 12, 15, 20, 25, 30]
    all_res = []
    for session in SESSIONS:
        df, n_func, floor = compare_session(session, k_list)
        all_res.append(df)
        print(f"\n=== {session}  (func has {n_func} week-8 clusters; "
              f"shuffle-baseline AMI={floor:+.4f}) ===")
        best = df.loc[df.groupby("features").AMI.idxmax()]
        print(df.pivot(index="k", columns="features", values="AMI").round(4).to_string())
        print("best per feature set:")
        for _, r in best.iterrows():
            print(f"  {r.features:14s} AMI={r.AMI:.4f} NMI={r.NMI:.4f} @ k={int(r.k)}")
        contingency_plot(session, int(best.set_index("features").loc["posture+motion", "k"]),
                         f"kp_analysis/{session}_contingency.png")
    pd.concat(all_res).to_csv("kp_analysis/ami_results.csv", index=False)
    print("\nSaved: kp_analysis/ami_results.csv")
