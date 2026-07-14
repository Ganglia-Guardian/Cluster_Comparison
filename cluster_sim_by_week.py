"""Silhouette score per week for the session-1 clustering.

Inputs (in data/1mp/):
  - session_1_out.mat  : MATLAB v7.3 (HDF5). We use
        Clusters/sim   (N x N) pairwise dissimilarity between bins
                       (diagonal 0, off-diagonal <= 0, so distance = -sim)
        Clusters/idx   (1 x N) cluster label per bin (the 103-cluster run
                       that matches `sim`)
  - Cluster_detail_results.csv : per-window (Timestamp, Folder_Name=week).
        The Timestamp column is a single, strictly monotonic global clock,
        so each week is a contiguous span of time -> a contiguous block of
        bins. We use it only to define the week boundaries.

Why this works without loading 103 GB: the bins are 60-sample windows tiling
the recording in time order (binSize=60, N*60 = number of raw samples), and
weeks are contiguous in time. So week w is a contiguous bin range [start:end),
and silhouette for that week only needs the diagonal block sim[start:end,
start:end] -- we read just those blocks from the HDF5 file.

Run:
    uv run python cluster_sim_by_week.py
(If a stray VIRTUAL_ENV points at anaconda, prefix with `VIRTUAL_ENV= ` or
deactivate it so uv uses the project .venv.)
"""

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import save_figure, similarity_to_distance
from sklearn.metrics import silhouette_samples, silhouette_score

DATA_DIR = Path("data/1mp")
MAT_PATH = DATA_DIR / "session_1_out.mat"
CSV_PATH = DATA_DIR / "Cluster_detail_results.csv"
OUT_DIR = DATA_DIR / "silhouette_out"

SIM_KEY = "Clusters/sim"
IDX_KEY = "Clusters/idx"
WEEK_COL = "Folder_Name"


def week_sort_key(week):
    """Numeric ordering by the week number embedded in the label, with the
    ldop/saline conditions sorted just after their week. Robust to label
    formats like 'week_10', 'week_w10', or 'week_24_ldop'."""
    match = re.search(r"\d+", week)
    num = float(match.group()) if match else float("inf")
    low = week.lower()
    if "saline" in low:
        num += 0.5
    elif "ldop" in low:
        num += 1
    return num


def week_bin_ranges(csv_path, n_bins):
    """Map the 113,597 bins to contiguous week ranges.

    The CSV rows are time-ordered windows; dropping the NaN-week rows leaves one
    contiguous block per week (recording order). We turn the per-week window
    counts into bin-index boundaries, scaled to n_bins so the (tiny, <0.1%)
    count difference between the CSV and the matrix is absorbed proportionally.

    Returns a list of (week, start, end) in recording (time) order.
    """
    df = pd.read_csv(csv_path)
    counts = df.dropna(subset=[WEEK_COL]).groupby(WEEK_COL, sort=False).size()
    weeks = counts.index.tolist()
    cum = counts.to_numpy().cumsum()
    # scale cumulative window counts onto the bin axis
    ends = np.round(cum * n_bins / cum[-1]).astype(int)
    ends[-1] = n_bins
    starts = np.concatenate([[0], ends[:-1]])

    n_csv = int(counts.sum())
    if n_csv != n_bins:
        print(f"Note: CSV has {n_csv} non-NaN windows vs {n_bins} bins "
              f"(diff {n_csv - n_bins}); week boundaries scaled proportionally.")
    return list(zip(weeks, starts, ends))


def silhouette_by_week_from_mat(mat_path=MAT_PATH, csv_path=CSV_PATH):
    """Compute silhouette per week, reading only each week's diagonal block."""
    with h5py.File(mat_path, "r") as f:
        sim = f[SIM_KEY]
        n_bins = sim.shape[0]
        idx_all = np.asarray(f[IDX_KEY]).ravel().astype(int)

        ranges = week_bin_ranges(csv_path, n_bins)

        rows = []
        per_bin = np.full(n_bins, np.nan)
        for week, start, end in ranges:
            labels = idx_all[start:end]
            n_clusters = int(np.unique(labels).size)
            score = np.nan

            if n_clusters < 2:
                print(f"{week}: {n_clusters} cluster(s); silhouette undefined, skipping.")
            elif n_clusters > len(labels) - 1:
                print(f"{week}: {n_clusters} clusters for {len(labels)} bins; skipping.")
            else:
                # contiguous diagonal block -> efficient HDF5 read
                block = sim[start:end, start:end]
                dist = similarity_to_distance(block)  # distance = -sim, diag 0
                score = float(silhouette_score(dist, labels, metric="precomputed"))
                per_bin[start:end] = silhouette_samples(dist, labels, metric="precomputed")
                print(f"{week}: bins={len(labels)} clusters={n_clusters} "
                      f"silhouette={score:.4f}")

            rows.append({"week": week, "start": start, "end": end,
                         "n_bins": len(labels), "n_clusters": n_clusters,
                         "silhouette": score,
                         "silhouette_per_cluster": (score / n_clusters
                                                    if np.isfinite(score) and n_clusters
                                                    else np.nan)})

    result = pd.DataFrame(rows)
    result["order"] = result["week"].map(week_sort_key)
    result = result.sort_values("order").reset_index(drop=True)
    return result, per_bin, ranges


# week_24_ldop / week_24_saline are conditions, not normal weeks. Plot them as
# standalone labelled dots rather than as points on the weekly line.
_VARIANT_STYLE = {
    "ldop": dict(label="L-DOPA", color="tab:red", marker="D"),
    "saline": dict(label="saline", color="tab:purple", marker="^"),
}


def _variant_style(week):
    low = week.lower()
    for key, style in _VARIANT_STYLE.items():
        if key in low:
            return style
    return None


def _draw_silhouette(ax, weeks, x, y, valid):
    """Line through the regular weeks; ldopa/saline as separate labelled dots."""
    is_variant = np.array([_variant_style(w) is not None for w in weeks])
    regular = valid & ~is_variant
    ax.plot(x[regular], y[regular], "o-", color="tab:blue", label="weekly")
    for i, week in enumerate(weeks):
        style = _variant_style(week)
        if style is None or not valid[i]:
            continue
        ax.plot(x[i], y[i], style["marker"], color=style["color"],
                markersize=11, linestyle="none", label=style["label"])
        ax.annotate(style["label"], (x[i], y[i]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=style["color"])
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.legend(loc="best", fontsize=8)


def make_plots(result, per_bin, ranges, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    weeks = result["week"].tolist()
    x = np.arange(len(weeks))
    y = result["silhouette"].to_numpy()
    valid = result["silhouette"].notna().to_numpy()

    # 1) silhouette over weeks (ldopa/saline as individual labelled dots)
    fig, ax = plt.subplots(figsize=(11, 4))
    _draw_silhouette(ax, weeks, x, y, valid)
    ax.set_xticks(x); ax.set_xticklabels(weeks, rotation=45, ha="right")
    ax.set_ylabel("mean silhouette"); ax.set_title("Silhouette score by week")
    fig.tight_layout(); save_figure(fig, out_dir / "silhouette_by_week.jpeg", dpi=150)
    plt.close(fig)

    # 2) number of clusters per week
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, result["n_clusters"], color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(weeks, rotation=45, ha="right")
    ax.set_ylabel("n clusters"); ax.set_title("Number of clusters present by week")
    fig.tight_layout(); save_figure(fig, out_dir / "n_clusters_by_week.jpeg", dpi=150)
    plt.close(fig)

    # 3) per-bin silhouette distribution by week (spread, not just the mean)
    data = [per_bin[s:e][np.isfinite(per_bin[s:e])]
            for (_, s, e) in sorted(ranges, key=lambda r: week_sort_key(r[0]))]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.boxplot(data, showfliers=False)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x + 1); ax.set_xticklabels(weeks, rotation=45, ha="right")
    ax.set_ylabel("per-bin silhouette")
    ax.set_title("Distribution of per-bin silhouette by week")
    fig.tight_layout(); save_figure(fig, out_dir / "silhouette_distribution_by_week.jpeg", dpi=150)
    plt.close(fig)

    # combined overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    _draw_silhouette(axes[0, 0], weeks, x, y, valid)
    axes[0, 0].set_title("Silhouette by week")
    axes[0, 1].bar(x, result["n_clusters"], color="tab:orange")
    axes[0, 1].set_title("n clusters by week")
    axes[1, 0].boxplot(data, showfliers=False)
    axes[1, 0].set_title("Per-bin silhouette distribution")
    axes[1, 0].axhline(0, color="grey", ls="--", lw=0.8)
    axes[1, 1].axis("off")
    for a in (axes[0, 0], axes[0, 1]):
        a.set_xticks(x); a.set_xticklabels(weeks, rotation=45, ha="right", fontsize=8)
    axes[1, 0].set_xticks(x + 1)
    axes[1, 0].set_xticklabels(weeks, rotation=45, ha="right", fontsize=8)
    fig.tight_layout(); save_figure(fig, out_dir / "overview.jpeg", dpi=150)
    plt.close(fig)

    print(f"Saved 4 plots to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Silhouette score per week from the session clustering.")
    parser.add_argument("--mat", type=Path, default=MAT_PATH,
                        help=f"input .mat file with {SIM_KEY} and {IDX_KEY} "
                             f"(default: {MAT_PATH})")
    parser.add_argument("--csv", type=Path, default=CSV_PATH,
                        help=f"cluster-detail CSV defining the weeks "
                             f"(default: {CSV_PATH})")
    parser.add_argument("--out", type=Path, default=OUT_DIR,
                        help=f"output directory for the CSV and plots "
                             f"(default: {OUT_DIR})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result, per_bin, ranges = silhouette_by_week_from_mat(args.mat, args.csv)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_out = args.out / "silhouette_by_week.csv"
    result.drop(columns=["order"]).to_csv(csv_out, index=False)
    print(f"\nWrote {csv_out}")
    print(result.drop(columns=["order", "start", "end"]).to_string(index=False))
    make_plots(result, per_bin, ranges, args.out)
