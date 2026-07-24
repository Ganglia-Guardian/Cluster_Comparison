"""Precompute a cluster x cluster distance from the datapoint similarity matrix.

``session_1_out.mat`` holds ``Clusters/sim``: a datapoint x datapoint affinity
matrix (affinity-propagation style -- symmetric, <= 0, with a ~0 diagonal, so
higher = more similar and ``-sim`` is a nonnegative dissimilarity). Its row/col
order is the row order of ``Cluster_detail_results.csv``; ``Clusters/idx`` (==
that CSV's ``ClusterIdx``) gives the cluster of each datapoint.

We collapse that ~114k x 114k matrix to a small (n_clusters x n_clusters)
average-dissimilarity matrix -- the mean of ``-sim`` between the members of each
pair of clusters -- and cache it. That is exactly the average-linkage distance
between clusters under the pipeline's own similarity, so the presence dendrogram
can be built on the model's geometry instead of on the weekly presence profiles.

The full matrix is ~36 GB on disk (gzip) and chunked column-wise (chunks are
(rows, 1)), so we stream it in COLUMN blocks -- a row-block read would touch
every column chunk and is ~10x slower. Each block updates
``S = G.T @ sim @ G`` where ``G`` is the (n_datapoints x n_clusters) cluster
indicator; the cluster means are then ``S / (n_a * n_b)``.

Run (one-time, reads the whole matrix, a few minutes):
    uv run python cluster_annotation_analysis/build_sim_distance.py --mouse 1mp
Writes ``data/<mouse>_sim_distance.csv`` (a labelled n_clusters x n_clusters
matrix) next to the annotation JSON, ready for presence_heatmap.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # repo root
from dataset_config import CSV_NAME, data_root  # noqa: E402


def find_mat(mouse):
    """Locate the session .mat holding ``Clusters/sim`` for `mouse`, trying the
    annotation data folder first (where it was dropped) then the mouse folder."""
    candidates = [
        HERE / "data" / f"{mouse}_session_1_out.mat",
        HERE / "data" / "session_1_out.mat",
        data_root() / mouse / "session_1_out.mat",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "no session_1_out.mat found; looked in:\n  " +
        "\n  ".join(str(c) for c in candidates))


def cluster_distance(mat_path, labels, col_block=2048):
    """Mean ``-sim`` between every pair of clusters, as an (K x K) array aligned
    to ``sorted(unique(labels))``.

    `labels[i]` is the cluster of datapoint i, in the matrix's row/col order.
    Streams the similarity matrix in column blocks (see module docstring) and
    accumulates ``S = G.T @ sim @ G`` for the cluster indicator ``G``.
    """
    labels = np.asarray(labels)
    clusters = np.unique(labels)
    K = clusters.size
    col_of = {c: k for k, c in enumerate(clusters)}
    codes = np.array([col_of[v] for v in labels])          # datapoint -> 0..K-1
    N = codes.size

    G = np.zeros((N, K), dtype=np.float64)                 # cluster indicator
    G[np.arange(N), codes] = 1.0
    sizes = G.sum(axis=0)                                  # datapoints per cluster

    with h5py.File(mat_path, "r") as f:
        sim = f["Clusters"]["sim"]
        if sim.shape[0] != N or sim.shape[1] != N:
            raise ValueError(f"sim is {sim.shape} but got {N} datapoint labels; "
                             "the .mat and the CSV are not aligned.")
        S = np.zeros((K, K), dtype=np.float64)
        t0 = time.time()
        for c0 in range(0, N, col_block):
            c1 = min(c0 + col_block, N)
            cb = np.asarray(sim[:, c0:c1], dtype=np.float64)   # (N x w)
            S += (G.T @ cb) @ G[c0:c1]                          # (K x K)
            done = c1 / N
            print(f"  cols {c1:>7}/{N}  ({done*100:5.1f}%)  "
                  f"{time.time()-t0:6.1f}s", flush=True)

    counts = np.outer(sizes, sizes)                       # n_a * n_b
    mean_sim = S / counts
    dist = -mean_sim                                      # sim <= 0 -> dist >= 0
    dist = (dist + dist.T) / 2.0                          # kill fp asymmetry
    return clusters, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", default="1mp",
                    help="dataset folder / annotation stem (default: 1mp)")
    ap.add_argument("--col-block", type=int, default=2048,
                    help="columns read per streaming block (default: 2048)")
    args = ap.parse_args()

    mat_path = find_mat(args.mouse)
    detail_csv = data_root() / args.mouse / CSV_NAME
    labels = pd.read_csv(detail_csv)["ClusterIdx"].to_numpy().astype(int)
    print(f"{args.mouse}: sim = {mat_path}")
    print(f"         csv = {detail_csv}  ({labels.size} datapoints, "
          f"{np.unique(labels).size} clusters)")

    clusters, dist = cluster_distance(mat_path, labels, col_block=args.col_block)

    out = HERE / "data" / f"{args.mouse}_sim_distance.csv"
    pd.DataFrame(dist, index=clusters, columns=clusters).to_csv(out)
    off = dist[~np.eye(dist.shape[0], dtype=bool)]
    print(f"  wrote {out}  ({dist.shape[0]}x{dist.shape[1]}; "
          f"off-diagonal dist range {off.min():.4g}..{off.max():.4g})")


if __name__ == "__main__":
    main()
