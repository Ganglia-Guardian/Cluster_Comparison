"""Feature (affinity) half of the cluster-degeneracy plane.

Aggregates the per-mouse observation x observation IMU affinity in
data/<mouse>/session_1_out.mat:/Clusters/sim  (113707 x 113707 float64, gzip,
diagonal 0 == identical, off-diagonals small NEGATIVE == AP affinity) into a
cluster x cluster feature matrix, keyed to the SAME kept clusters/weeks that
presence_similarity.py wrote to out/<mouse>/presence.npz.

    Sagg[a,b] = mean_{i in a, j in b} sim[i,j]           (mean affinity)
    Dfeat[a,b] = -Sagg[a,b]                               (feature DISTANCE >= 0)

Diagonal Sagg[a,a] is the within-cluster self-affinity -> the reference scale for
"how degenerate is degenerate" (cross-cluster Dfeat approaching self Dfeat).

Cost control (exact contiguous streaming):
  sim is chunked (8121, 1). Benchmarks: contiguous column blocks read at ~366
  cols/s, but FANCY/scattered column indexing at ~6 cols/s (60x slower) -- so a
  scattered subsample is far slower than reading everything contiguously. We
  therefore stream the FULL matrix in contiguous column blocks and aggregate
  exactly (~5 min/mouse, no sampling error):

    for each contiguous column block [c0:c1] (read full-height sim[:, c0:c1]):
        M += (R^T @ sim[:, c0:c1]) @ Cj    # R: N x K row one-hot, Cj: w x K col one-hot
    Sagg = M / (n[:,None] * n[None,:])      # n[a] = # kept obs in cluster a

  Rows/cols outside the kept clusters/natural weeks get no one-hot entry (lab=-1)
  and are excluded from every block sum.

Alignment: /Clusters/idx is asserted == detail-CSV ClusterIdx, so sim index i is
detail row i (its cluster and week). Rows outside the kept clusters/natural weeks
get no one-hot entry and are excluded from every block sum.

Run from repo root (after presence_similarity.py):
    C:/ProgramData/anaconda3/python.exe degeneracy_analysis/feature_similarity.py --mouse 1mp
    C:/ProgramData/anaconda3/python.exe degeneracy_analysis/feature_similarity.py   # all mice
"""
import argparse
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py

from presence_similarity import MICE, DATA, OUT, natural_week

COLBLOCK = 2000    # columns read per contiguous block (N x COLBLOCK float64 ~ 1.8 GB)


def _row_labels(mouse, clusters, weeks):
    """Per-sim-row cluster column index (0..K-1) or -1 if excluded, verified
    aligned to sim via /Clusters/idx == detail ClusterIdx."""
    det = pd.read_csv(f"{DATA}/{mouse}/Cluster_detail_results.csv")
    with h5py.File(f"{DATA}/{mouse}/session_1_out.mat", "r") as f:
        idx = np.array(f["/Clusters/idx"][()]).squeeze().astype(int)
    if not (len(idx) == len(det) and np.array_equal(idx, det.ClusterIdx.to_numpy())):
        raise ValueError(f"{mouse}: /Clusters/idx != detail ClusterIdx; not aligned")

    col_of = {c: k for k, c in enumerate(clusters)}
    week_ok = det.Folder_Name.map(natural_week).isin(set(weeks)).to_numpy()
    lab = np.full(len(det), -1, int)
    for k, c in enumerate(clusters):
        lab[(det.ClusterIdx.to_numpy() == c) & week_ok] = k
    return lab


def aggregate(mouse, colblock=COLBLOCK, verbose=True):
    pres = np.load(f"{OUT}/{mouse}/presence.npz", allow_pickle=True)
    clusters = pres["clusters"].astype(int)
    weeks = pres["weeks"].astype(int)
    K = len(clusters)

    lab = _row_labels(mouse, clusters, weeks)
    N = len(lab)
    valid = lab >= 0
    rows = np.where(valid)[0]
    R = sp.csr_matrix((np.ones(rows.size), (rows, lab[valid])), shape=(N, K))  # N x K
    n = np.asarray(R.sum(axis=0)).ravel()                                      # kept obs/cluster

    M = np.zeros((K, K))
    t0 = time.time()
    with h5py.File(f"{DATA}/{mouse}/session_1_out.mat", "r") as f:
        sim = f["/Clusters/sim"]
        for c0 in range(0, N, colblock):
            c1 = min(c0 + colblock, N)
            block = sim[:, c0:c1]                     # (N, w) CONTIGUOUS -> fast
            left = R.T @ block                        # (K, w) rows summed by cluster
            clab = lab[c0:c1]
            m = clab >= 0
            Cj = sp.csr_matrix((np.ones(int(m.sum())), (np.where(m)[0], clab[m])),
                               shape=(c1 - c0, K))     # w x K, excluded cols are empty
            M += left @ Cj                            # (K, K)
            if verbose:
                print(f"  {mouse}: {c1}/{N} cols  ({time.time()-t0:.0f}s)", flush=True)

    Sagg = M / (n[:, None] * n[None, :])               # mean affinity over kept obs (exact)
    Sagg = 0.5 * (Sagg + Sagg.T)                       # guard tiny fp asymmetry
    Dfeat = -Sagg                                      # feature distance >= 0
    self_d = np.diag(Dfeat).copy()                     # within-cluster self-distance

    np.savez(f"{OUT}/{mouse}/feature.npz",
             clusters=clusters, Sagg=Sagg, Dfeat=Dfeat, self_dist=self_d, n=n)

    if verbose:
        off = Dfeat[~np.eye(K, dtype=bool)]
        print(f"\n{'='*68}\n{mouse}: feature axis\n{'='*68}")
        print(f"  kept clusters: {K}   ({time.time()-t0:.0f}s total)")
        print(f"  self-distance (diag): median {np.median(self_d):.4g}, "
              f"max {self_d.max():.4g}")
        print(f"  cross Dfeat: median {np.median(off):.4g}, "
              f"range [{off.min():.4g}, {off.max():.4g}]")
        print(f"  cross >= self scale?  min cross {off.min():.4g} vs "
              f"median self {np.median(self_d):.4g}  "
              f"({'some cross below self -> degeneracy candidates' if off.min() < np.median(self_d) else 'all cross above self'})")
    return dict(mouse=mouse, clusters=clusters, Sagg=Sagg, Dfeat=Dfeat, self_dist=self_d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", choices=MICE, help="one mouse (default: all)")
    args = ap.parse_args()
    for m in ([args.mouse] if args.mouse else MICE):
        aggregate(m)
