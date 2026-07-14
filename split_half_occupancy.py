"""Split-half occupancy reproducibility per week, for each dataset.

Question this answers
---------------------
"Is a single weekly recording long enough to describe a mouse's behavioral
repertoire?" -- and, downstream, does the across-week variation we see reflect
disease progression or just sampling noise from too-short recordings?

We answer it WITHOUT re-clustering. We reuse the cluster labels already in each
`session_1_out.mat` (`Clusters/idx`, one label per 60-sample bin) and the weeks
defined by `Cluster_detail_results.csv`. So this runs in megabytes, not the
hundreds of GB the clustering itself needs.

Method
------
For each weekly recording we estimate how reproducible the *occupancy
distribution* (fraction of time in each cluster) is, by splitting the recording
in half and comparing the two halves:

  * Block-interleaved split, NOT first-half/second-half. Bins are
    autocorrelated (a behavior lasts many bins) and a recording drifts over its
    duration (habituation, fatigue). A midpoint cut would blame that drift on
    irreproducibility. Instead we cut each week into contiguous blocks of
    `block` bins and deal alternate blocks to halves A and B: both halves span
    the whole recording, and blocks longer than a bout stop the autocorrelation
    from inflating the agreement. We sweep `block` so the dependence is visible
    (block=1 = inflated bin-level number; large block -> temporal-split number).

  * Metrics between the two half-occupancy vectors (support = all K clusters):
      cosine            1 = identical shape
      js                Jensen-Shannon distance, 0 = identical distribution
      spearman          rank agreement of cluster usage
      jaccard_presence  agreement on *which* clusters appear at all (coverage)
    plus how many clusters land in both halves / one half only.

Saturation (rarefaction)
------------------------
Cumulative distinct clusters vs time within each recording (temporal order). If
the curve plateaus well before the end, the recording saturates the repertoire;
if clusters keep first-appearing late, it does not. `late_first_seen` counts
clusters whose first appearance is in the last 10% of the recording.

Interpreting the result (ties back to the 2D/3D puzzle)
-------------------------------------------------------
  * High split-half agreement + plateaued rarefaction, yet behaviors still don't
    separate the way you expect -> a *representation/sensor ceiling* (head IMU
    can't resolve them). More recording time will not help.
  * Low split-half agreement / no plateau -> *undersampling*. More time, or
    pooling, helps.

Run:
    uv run python split_half_occupancy.py
    uv run python split_half_occupancy.py --datasets 1mp 1lc --block 200
(If a stray VIRTUAL_ENV points at anaconda, prefix with `VIRTUAL_ENV= `.)
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from cluster_sim_by_week import week_bin_ranges, week_sort_key
from utils import save_figure

DATA_ROOT = Path("data")
IDX_KEY = "Clusters/idx"
MAT_NAME = "session_1_out.mat"
CSV_NAME = "Cluster_detail_results.csv"

DEFAULT_BLOCK = 200            # bins per block for the headline split
BLOCK_SWEEP = [1, 30, 100, 300, 1000, 3000]
MIN_BLOCKS_PER_HALF = 4        # need >= this many blocks each side to trust a week
RAREFACTION_POINTS = 60


def load_idx(mat_path):
    """Read only the per-bin cluster labels (cheap; ignores the huge `sim`)."""
    with h5py.File(mat_path, "r") as f:
        return np.asarray(f[IDX_KEY]).ravel().astype(int)


def occupancy(labels, k):
    """Normalised time-in-cluster vector over clusters 1..k (sums to 1)."""
    counts = np.bincount(labels, minlength=k + 1)[1:]
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def block_split(n, block):
    """Boolean mask selecting half A under a block-interleaved split of n bins.

    Block i (bins [i*block:(i+1)*block]) goes to A if i is even, else B. Returns
    (maskA, n_blocks). Both halves therefore span the whole recording.
    """
    block_id = np.arange(n) // block
    return (block_id % 2 == 0), int(block_id[-1] + 1) if n else 0


def split_half_metrics(labels, k, block):
    """Compare the two block-interleaved halves' occupancy. NaNs if too few blocks."""
    n = labels.size
    mask_a, n_blocks = block_split(n, block)
    nan = dict(cosine=np.nan, js=np.nan, spearman=np.nan, jaccard_presence=np.nan,
               n_both=np.nan, n_a_only=np.nan, n_b_only=np.nan, n_blocks=n_blocks)
    if n_blocks // 2 < MIN_BLOCKS_PER_HALF:
        return nan

    a = occupancy(labels[mask_a], k)
    b = occupancy(labels[~mask_a], k)
    pa, pb = a > 0, b > 0

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(a @ b / denom) if denom else np.nan
    js = float(jensenshannon(a, b))                     # base e; 0 = identical
    rho = spearmanr(a, b).correlation if (pa.any() and pb.any()) else np.nan
    union = (pa | pb).sum()
    jaccard = float((pa & pb).sum() / union) if union else np.nan

    return dict(cosine=cosine, js=js, spearman=float(rho),
                jaccard_presence=jaccard, n_both=int((pa & pb).sum()),
                n_a_only=int((pa & ~pb).sum()), n_b_only=int((~pa & pb).sum()),
                n_blocks=n_blocks)


def rarefaction(labels, k, n_points=RAREFACTION_POINTS):
    """Cumulative distinct clusters in temporal order.

    Returns (frac_time, frac_repertoire, counts, sat50, late_first_seen):
      frac_time        x in [0,1]
      frac_repertoire  distinct-so-far / total distinct in this recording
      counts           distinct-so-far (absolute cluster count at each cut)
      sat50            distinct@50% / distinct@100%  (1.0 = saturated by halfway)
      late_first_seen  # clusters first appearing in the last 10% of the recording
    """
    n = labels.size
    cuts = np.unique(np.linspace(1, n, n_points).astype(int))
    seen_counts = [np.unique(labels[:c]).size for c in cuts]
    total = np.unique(labels).size
    frac_time = cuts / n
    frac_rep = np.asarray(seen_counts) / total if total else np.zeros_like(cuts, float)

    half = np.unique(labels[: n // 2]).size
    sat50 = half / total if total else np.nan

    # first-appearance index of each label, then count those in the last 10%
    first = {}
    for i, lab in enumerate(labels):
        if lab not in first:
            first[lab] = i
    late = sum(1 for v in first.values() if v >= 0.9 * n)
    return frac_time, frac_rep, np.asarray(seen_counts), float(sat50), int(late)


def analyse_dataset(ds_dir):
    """Per-week split-half + rarefaction for one dataset. Returns (rows, rares, bout)."""
    idx = load_idx(ds_dir / MAT_NAME)
    k = int(idx.max())
    ranges = week_bin_ranges(ds_dir / CSV_NAME, idx.size)

    # data-driven sanity check on block size: how long does one behavior persist?
    runs = np.diff(np.flatnonzero(np.r_[True, idx[1:] != idx[:-1], True]))
    bout = dict(median=int(np.median(runs)), p95=int(np.percentile(runs, 95)))

    rows, rares = [], []
    for week, start, end in ranges:
        labels = idx[start:end]
        m = split_half_metrics(labels, k, DEFAULT_BLOCK)
        ft, fr, counts, sat50, late = rarefaction(labels, k)
        rares.append((week, ft, fr, counts))
        sweep = {f"cos_b{b}": split_half_metrics(labels, k, b)["cosine"]
                 for b in BLOCK_SWEEP}
        rows.append(dict(dataset=ds_dir.name, week=week, n_bins=labels.size,
                         n_clusters_week=int(np.unique(labels).size), **m,
                         sat50=sat50, late_first_seen=late, **sweep))

    df = pd.DataFrame(rows)
    df["order"] = df["week"].map(week_sort_key)
    df = df.sort_values("order").drop(columns="order").reset_index(drop=True)
    return df, rares, bout, k


def plot_rarefaction(rares, out_dir, ds_name, k):
    """Dedicated rarefaction figure: fraction of clusters discovered over time.

    Left  -- y = fraction of the recording's repertoire seen so far (normalised).
    Right -- y = absolute count of distinct clusters seen so far.
    One line per weekly recording, coloured early->late so disease progression
    is visible; the bold black line is the across-week mean (normalised panel).
    A plateau before x=1 means the recording saturates its repertoire.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rares = sorted(rares, key=lambda r: week_sort_key(r[0]))
    n = len(rares)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, (_week, ft, fr, counts) in enumerate(rares):
        color = cmap(i / max(n - 1, 1))
        axes[0].plot(ft, fr, color=color, lw=1.3, alpha=0.85)
        axes[1].plot(ft, counts, color=color, lw=1.3, alpha=0.85)

    grid = np.linspace(0, 1, RAREFACTION_POINTS)
    mean_fr = np.mean([np.interp(grid, ft, fr) for _, ft, fr, _ in rares], axis=0)
    axes[0].plot(grid, mean_fr, color="k", lw=2.5, label="across-week mean")
    axes[0].legend(fontsize=9, loc="lower right")

    for ax in axes:
        ax.axvline(0.5, color="grey", ls="--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of recording")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("fraction of repertoire seen")
    axes[0].set_title("Normalised (fraction of clusters)")
    axes[1].set_ylabel("distinct clusters seen")
    axes[1].set_title("Absolute count")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n - 1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("week (early -> late)")
    cbar.set_ticks([0, n - 1]); cbar.set_ticklabels([rares[0][0], rares[-1][0]])

    fig.suptitle(f"{ds_name}: cluster discovery over time (rarefaction), "
                 f"K={k} clusters", fontsize=13)
    save_figure(fig, out_dir / "rarefaction_by_week.jpeg", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dataset(df, rares, bout, k, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    weeks = df["week"].tolist()
    x = np.arange(len(weeks))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # 1) split-half agreement across weeks
    ax = axes[0, 0]
    ax.plot(x, df["cosine"], "o-", label="cosine", color="tab:blue")
    ax.plot(x, 1 - df["js"], "s-", label="1 - JS dist", color="tab:green")
    ax.plot(x, df["jaccard_presence"], "^-", label="Jaccard (presence)",
            color="tab:orange")
    ax.set_ylim(0, 1.02); ax.set_ylabel("split-half agreement")
    ax.set_title(f"Split-half occupancy agreement by week (block={DEFAULT_BLOCK} bins)")
    ax.set_xticks(x); ax.set_xticklabels(weeks, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    # 2) rarefaction curves (one faint line per week)
    ax = axes[0, 1]
    for week, ft, fr, _ in rares:
        ax.plot(ft, fr, color="tab:blue", alpha=0.25, lw=1)
    ax.axvline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("fraction of recording"); ax.set_ylabel("fraction of repertoire seen")
    ax.set_title("Rarefaction per week (plateau = saturated)")
    ax.set_ylim(0, 1.02)

    # 3) block-size sensitivity (mean cosine over weeks +/- spread)
    ax = axes[1, 0]
    means = [df[f"cos_b{b}"].mean() for b in BLOCK_SWEEP]
    stds = [df[f"cos_b{b}"].std() for b in BLOCK_SWEEP]
    ax.errorbar(BLOCK_SWEEP, means, yerr=stds, fmt="o-", color="tab:purple")
    ax.axvline(bout["p95"], color="tab:red", ls="--", lw=1,
               label=f"bout p95 = {bout['p95']} bins")
    ax.set_xscale("log"); ax.set_xlabel("block size (bins, log)")
    ax.set_ylabel("mean split-half cosine"); ax.set_ylim(0, 1.02)
    ax.set_title("Agreement vs block size (small = autocorr-inflated)")
    ax.legend(fontsize=8)

    # 4) saturation index + late discovery across weeks
    ax = axes[1, 0].twinx() if False else axes[1, 1]
    ax.plot(x, df["sat50"], "o-", color="tab:blue", label="distinct@50% / @100%")
    ax.set_ylabel("saturation index", color="tab:blue"); ax.set_ylim(0, 1.02)
    ax2 = ax.twinx()
    ax2.bar(x, df["late_first_seen"], alpha=0.3, color="tab:red")
    ax2.set_ylabel("clusters first seen in last 10%", color="tab:red")
    ax.set_title("Saturation by week")
    ax.set_xticks(x); ax.set_xticklabels(weeks, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"{out_dir.parent.name}: split-half occupancy diagnostic "
                 f"(K={k} clusters, bout median={bout['median']} bins)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out_dir / "split_half_occupancy.jpeg", dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="dataset folder names under data/ (default: all that "
                        f"contain {MAT_NAME})")
    p.add_argument("--block", type=int, default=DEFAULT_BLOCK,
                   help=f"bins per block for the headline split (default {DEFAULT_BLOCK})")
    return p.parse_args()


def main():
    args = parse_args()
    global DEFAULT_BLOCK
    DEFAULT_BLOCK = args.block

    if args.datasets:
        dirs = [args.data_root / d for d in args.datasets]
    else:
        dirs = sorted(d for d in args.data_root.iterdir()
                      if d.is_dir() and (d / MAT_NAME).exists())

    all_df, summary_rares = [], {}
    for ds_dir in dirs:
        print(f"\n=== {ds_dir.name} ===")
        df, rares, bout, k = analyse_dataset(ds_dir)
        out_dir = ds_dir / "split_half_out"
        plot_dataset(df, rares, bout, k, out_dir)
        plot_rarefaction(rares, out_dir, ds_dir.name, k)
        df.to_csv(out_dir / "split_half_by_week.csv", index=False)
        summary_rares[ds_dir.name] = rares
        all_df.append(df)

        cols = ["week", "n_bins", "n_clusters_week", "cosine", "js",
                "jaccard_presence", "sat50", "late_first_seen"]
        print(df[cols].to_string(index=False))
        print(f"  mean cosine={df['cosine'].mean():.3f}  mean JS={df['js'].mean():.3f}  "
              f"mean sat50={df['sat50'].mean():.3f}  bout median/p95="
              f"{bout['median']}/{bout['p95']} bins")
        print(f"  -> plots+csv in {out_dir}")

    if all_df:
        combined = pd.concat(all_df, ignore_index=True)
        combined.to_csv(args.data_root / "split_half_summary.csv", index=False)

        # overlay: mean rarefaction per dataset
        fig, ax = plt.subplots(figsize=(8, 5))
        grid = np.linspace(0, 1, RAREFACTION_POINTS)
        for ds, rares in summary_rares.items():
            curves = [np.interp(grid, ft, fr) for _, ft, fr, _ in rares]
            ax.plot(grid, np.mean(curves, axis=0), "o-", ms=3, label=ds)
        ax.axvline(0.5, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("fraction of recording"); ax.set_ylabel("fraction of repertoire seen")
        ax.set_title("Mean rarefaction per dataset"); ax.set_ylim(0, 1.02)
        ax.legend(title="dataset")
        fig.tight_layout(); save_figure(fig, args.data_root / "rarefaction_overlay.jpeg", dpi=150)
        plt.close(fig)
        print(f"\nWrote {args.data_root / 'split_half_summary.csv'} and "
              f"{args.data_root / 'rarefaction_overlay.jpeg'}")


if __name__ == "__main__":
    main()
