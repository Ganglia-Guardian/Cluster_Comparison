"""Reverse conditioning: take each POSE cluster as a prior and look at the IMU
data underneath it.

Two layers:

(A) reverse_contingency() -- uses the discrete func ClusterIdx we already have.
    For each pose cluster it reports how many distinct IMU behaviors live under
    it (richness), the dominant IMU share, and the effective number of IMU
    behaviors (perplexity = exp(entropy)). Low perplexity => that posture nearly
    determines the IMU behavior; high => the posture is behaviorally ambiguous.

(B) ap_within_poses() -- re-clusters the IMU data *continuously* within each pose
    using Affinity Propagation on a CALLER-PROVIDED window x window IMU similarity
    matrix (affinity='precomputed'). AP picks its own number of exemplars, so it
    answers "how many IMU sub-modes does this posture contain", independent of the
    existing func codebook. Each pose's AP labelling is scored against func
    ClusterIdx (AMI) to see whether AP recovers the codebook or finds new splits.

Similarity-matrix contract (for B):
    * shape (N, N), N = number of week-8 windows for the session
      (5964 for wk8lc/2lc, 5977 for wk8mp/2mp)
    * row i == the i-th week-8 window in Timestamp order == row order of the
      session's Cluster_detail_results.csv week-folder (already chronological)
    * IMU-derived, higher = more similar
    export_window_order() writes the canonical order so alignment can be checked.

Run from repo root:
    # immediate (discrete) result + write window order:
    C:/ProgramData/anaconda3/python.exe kp_analysis/kp_pose_prior_imu.py
    # once you have a matrix:
    ...kp_pose_prior_imu.py --sim path/to/wk8lc_imu_sim.npy --session wk8lc
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import adjusted_mutual_info_score as ami

import kp_cluster_compare as cc

K = 30


def pose_labels(session):
    """Return (pose, func) aligned to the session's week-8 windows."""
    post, mot, comb, func = cc.build_windows(session)
    pose = KMeans(K, n_init=10, random_state=0).fit_predict(cc.embed(post))
    return pose, func


def export_window_order(session):
    """Write the canonical window order so a provided similarity matrix can be
    checked / aligned against it."""
    kp_csv, func_csv, folder = cc.SESSIONS[session]
    win = pd.read_csv(func_csv)
    win = win[win.Folder_Name == folder].sort_values("Timestamp").reset_index(drop=True)
    out = pd.DataFrame({
        "window_idx": np.arange(len(win)),
        "timestamp_rel": win.Timestamp.to_numpy() - win.Timestamp.iloc[0],
        "func_cluster": win.ClusterIdx.to_numpy(),
    })
    path = f"kp_analysis/{session}_window_order.csv"
    out.to_csv(path, index=False)
    return path, len(out)


def extract_sim_from_mat(mat_path, session, out_npy):
    """Pull the aligned week-8 window x window IMU similarity out of a
    session_*_out.mat into build_windows() order and save as float32 .npy.

    /Clusters/sim is all-weeks (e.g. 113627^2 for 2lc). /Clusters/idx is verified
    to equal the detail-CSV ClusterIdx, so sim rows align to detail rows; we take
    the week-folder rows in Timestamp order (matching build_windows) out of the
    enclosing square block (cheap contiguous hyperslab), then symmetrize.
    """
    import h5py

    _, func_csv, folder = cc.SESSIONS[session]
    det = pd.read_csv(func_csv)
    order = det[det.Folder_Name == folder].sort_values("Timestamp").index.to_numpy()
    lo, hi = int(order.min()), int(order.max()) + 1
    with h5py.File(mat_path, "r") as f:
        idx = np.array(f["/Clusters/idx"][()]).squeeze().astype(int)
        if not np.array_equal(idx, det.ClusterIdx.to_numpy()):
            raise ValueError("/Clusters/idx != detail ClusterIdx order; not aligned")
        block = f["/Clusters/sim"][lo:hi, lo:hi]
    S = block[np.ix_(order - lo, order - lo)].astype(np.float64)
    S = (S + S.T) / 2.0
    np.save(out_npy, S.astype(np.float32))
    print(f"{session}: saved {S.shape} similarity -> {out_npy}  "
          f"(min={S.min():.5f}, the AP 'min' preference)")
    return S.shape, float(S.min())


def reverse_contingency(session):
    pose, func = pose_labels(session)
    rows = []
    for c in range(K):
        sub = func[pose == c]
        n = len(sub)
        if n == 0:
            continue
        p = np.bincount(sub) / n
        p = p[p > 0]
        H = -(p * np.log(p)).sum()
        rows.append({
            "pose": c, "n": n, "n_distinct_imu": len(p),
            "dominant_imu_share": float(p.max()),
            "eff_n_imu": float(np.exp(H)),     # perplexity
        })
    df = pd.DataFrame(rows).sort_values("eff_n_imu").reset_index(drop=True)
    print(f"\n{'='*64}\n{session}: IMU behaviors under each pose (pose as prior)\n{'='*64}")
    print(f"  median effective # IMU behaviors per pose = {df.eff_n_imu.median():.1f}")
    print("  most behaviorally DETERMINATE postures (low eff-n):")
    print(df.head(5).round(3).to_string(index=False))
    print("  most behaviorally AMBIGUOUS postures (high eff-n):")
    print(df.tail(5).round(3).to_string(index=False))
    df.to_csv(f"kp_analysis/{session}_reverse_contingency.csv", index=False)
    return df


def ap_within_poses(session, sim, preference="min", min_windows=30, damping=0.9):
    """Affinity Propagation on the IMU similarity submatrix of each pose cluster.

    sim : (N, N) precomputed window x window IMU similarity aligned to this
          session's windows.
    preference: "min" -> global minimum of the matrix (the lab convention; fewest
          exemplars), "median" -> sklearn default (per-subset median), float ->
          fixed value, ('pct', q) -> q-th percentile of the subset.
    """
    pose, func = pose_labels(session)
    N = len(pose)
    if sim.shape != (N, N):
        raise ValueError(f"similarity is {sim.shape}, expected ({N}, {N}) for "
                         f"{session}; check ordering/alignment vs window_order.csv")
    gmin = float(sim.min())

    rows = []
    for c in range(K):
        idx = np.where(pose == c)[0]
        n = len(idx)
        if n < min_windows:
            rows.append({"pose": c, "n": n, "ap_submodes": None, "note": "too small"})
            continue
        S = sim[np.ix_(idx, idx)].astype(float)
        if preference == "min":
            pref = gmin                       # minimum value in the matrix
        elif preference == "median":
            pref = None                       # sklearn uses per-subset median
        elif isinstance(preference, tuple) and preference[0] == "pct":
            pref = np.percentile(S[~np.eye(n, dtype=bool)], preference[1])
        else:
            pref = preference
        ap = AffinityPropagation(affinity="precomputed", preference=pref,
                                 damping=damping, max_iter=1000, convergence_iter=50,
                                 random_state=0)
        try:
            lab = ap.fit_predict(S)
            n_modes = len(set(lab)) if ap.cluster_centers_indices_ is not None \
                and len(ap.cluster_centers_indices_) else 1
            converged = ap.n_iter_ < 1000 and n_modes >= 1
        except Exception as e:  # non-convergence / degenerate
            lab = np.zeros(n, int); n_modes = 1; converged = False
        sizes = np.sort(np.bincount(lab))[::-1]
        rows.append({
            "pose": c, "n": n, "ap_submodes": int(n_modes),
            "top_mode_share": float(sizes[0] / n),
            "ami_vs_func": float(ami(func[idx], lab)) if n_modes > 1 else 0.0,
            "converged": bool(converged),
        })
    df = pd.DataFrame(rows)
    print(f"\n{'='*64}\n{session}: Affinity Propagation on IMU under each pose\n{'='*64}")
    done = df.dropna(subset=["ap_submodes"])
    print(f"  poses clustered: {len(done)}/{K}  "
          f"(median AP sub-modes = {done.ap_submodes.median():.0f}, "
          f"median AMI vs func = {done.ami_vs_func.median():.3f})")
    print(done.sort_values("ami_vs_func", ascending=False).round(3).to_string(index=False))
    df.to_csv(f"kp_analysis/{session}_ap_within_pose.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", help="path to (N,N) IMU similarity .npy")
    ap.add_argument("--mat", help="extract similarity from a session_*_out.mat first")
    ap.add_argument("--session", choices=list(cc.SESSIONS))
    ap.add_argument("--pref", default="min",
                    help="AP preference: min | median | a float (default min)")
    args = ap.parse_args()

    if args.mat:
        args.sim = args.sim or f"kp_analysis/data/{args.session}/imu_sim.npy"
        extract_sim_from_mat(args.mat, args.session, args.sim)
    if args.sim:
        S = np.load(args.sim)
        try:
            pref = float(args.pref)
        except ValueError:
            pref = args.pref
        ap_within_poses(args.session, S, preference=pref)
    else:
        for session in cc.SESSIONS:
            path, n = export_window_order(session)
            print(f"{session}: {n} windows; canonical order -> {path}")
            reverse_contingency(session)
        print("\nProvide an IMU window x window similarity .npy (see contract) to "
              "run Affinity Propagation:\n  --sim <file> --session <wk8lc|wk8mp>")
