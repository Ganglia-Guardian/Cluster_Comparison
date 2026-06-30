"""Fork 1: per-cluster occupancy over progression weeks, separated by arena.

For each MATLAB clustering (per mouse, per batch -- cluster ids are only
meaningful within one batch) we split frames by arena (weekN = 3D, weekN_O =
flat/2D) and, within each arena, build the occupancy of every cluster across the
batch's progression weeks. We then ask, per cluster, how much that occupancy
changes over weeks -- and crucially whether it changes DIFFERENTLY in the two
arenas (a cluster drifting in 3D but flat in 2D is a candidate "volumetric"
vulnerability).

Per-cluster change is summarized two ways:
  * trend  -- Spearman rho of occupancy vs week (signed monotonic drift)
  * delta  -- late-third mean occupancy minus early-third mean (magnitude/direction)

Because single clusters can drift noisily, we also track two COLLECTIVE measures
per arena, week by week:
  * effective behaviours  exp(H) of the occupancy distribution over clusters
    (how many states are meaningfully occupied)
  * JS drift  Jensen-Shannon distance of each week's occupancy from an early
    baseline (how far the whole repertoire has moved)

Headline test: for clusters present in both arenas, is |occupancy change|
systematically larger in 3D than 2D? Paired Wilcoxon, per batch.

Run:
    uv run python arena_analysis/arena_occupancy_drift.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wilcoxon

from cluster_arena_exclusivity import (BATCHES, MICE, OUT as EXCL_OUT,
                                       load_mat_frames)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "occupancy_drift"
ARENAS = ["2D", "3D"]
ARENA_COLOR = {"2D": "#d62728", "3D": "#1f77b4"}
MIN_WEEK_FRAMES = 200    # drop an arena-week with fewer frames than this


def occupancy_matrix(frames_arena, clusters, weeks):
    """(week x cluster) occupancy: each row sums to 1 over the clusters present
    that arena-week. Weeks with too few frames become NaN rows."""
    mat = pd.DataFrame(index=weeks, columns=clusters, dtype=float)
    for w, g in frames_arena.groupby("week"):
        if len(g) < MIN_WEEK_FRAMES:
            continue
        counts = g["cluster"].value_counts()
        mat.loc[w] = counts.reindex(clusters).fillna(0).to_numpy() / len(g)
    return mat


def per_cluster_change(occ, weeks):
    """For an occupancy matrix (week x cluster), per-cluster trend + early/late delta."""
    wk = np.array(weeks, float)
    valid = occ.notna().all(axis=1)        # weeks that were well-sampled
    wk_v = wk[valid.to_numpy()]
    third = max(1, len(wk_v) // 3)
    rows = []
    for c in occ.columns:
        y = occ.loc[valid, c].to_numpy(float)
        if len(y) < 3 or np.allclose(y, y[0]):
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(wk_v, y)
        early = y[:third].mean() if len(y) else np.nan
        late = y[-third:].mean() if len(y) else np.nan
        rows.append({"cluster": int(c), "rho": rho, "p": p,
                     "early_occ": early, "late_occ": late,
                     "delta": late - early, "abs_delta": abs(late - early),
                     "mean_occ": y.mean() if len(y) else np.nan})
    return pd.DataFrame(rows)


def collective(occ, weeks):
    """Per-week effective-behaviour count exp(H) and JS drift from an early baseline."""
    valid = occ.notna().all(axis=1)
    wk = np.array(weeks)[valid.to_numpy()]
    M = occ.loc[valid].to_numpy(float)
    if len(M) < 2:
        return pd.DataFrame(columns=["week", "eff_behaviours", "js_drift"])
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(M > 0, M * np.log(M), 0.0), axis=1)
    eff = np.exp(H)
    third = max(1, len(M) // 3)
    baseline = M[:third].mean(axis=0)
    baseline = baseline / baseline.sum()
    js = [jensenshannon(row, baseline, base=2) for row in M]
    return pd.DataFrame({"week": wk, "eff_behaviours": eff, "js_drift": js})


def plot_collective(coll_by_arena, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for arena, coll in coll_by_arena.items():
        if coll.empty:
            continue
        c = ARENA_COLOR[arena]
        axes[0].plot(coll["week"], coll["eff_behaviours"], "-o", color=c, label=arena)
        axes[1].plot(coll["week"], coll["js_drift"], "-o", color=c, label=arena)
    axes[0].set(xlabel="week", ylabel="effective behaviours  exp(H)",
                title="Repertoire breadth by week")
    axes[1].set(xlabel="week", ylabel="JS drift from early baseline",
                title="Repertoire drift by week")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(title="arena")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_arena_scatter(merged, title, path):
    """Per-cluster occupancy trend in 2D vs 3D; off-diagonal = arena-divergent."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.plot([-1, 1], [-1, 1], ls="--", color="grey", lw=0.8)
    sz = 30 + 600 * merged["mean_occ_3D"].fillna(0)
    ax.scatter(merged["rho_2D"], merged["rho_3D"], s=sz, alpha=0.6,
               color="#6a3d9a", edgecolors="white", linewidths=0.5)
    ax.set(xlabel="occupancy trend rho (2D/flat)",
           ylabel="occupancy trend rho (3D)", xlim=(-1.05, 1.05), ylim=(-1.05, 1.05),
           title=title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    long_rows, change_rows, coll_rows, headline = [], [], [], []
    for mouse_dir, mouse in MICE.items():
        for batch in BATCHES:
            frames = load_mat_frames(mouse_dir, batch)
            if frames is None:
                continue
            weeks = sorted(frames["week"].unique())
            clusters = sorted(frames["cluster"].unique())

            change_by_arena, coll_by_arena = {}, {}
            for arena in ARENAS:
                fa = frames[frames["arena"] == arena]
                occ = occupancy_matrix(fa, clusters, weeks)
                # long occupancy record
                lo = occ.reset_index(names="week").melt(
                    id_vars="week", var_name="cluster", value_name="occ").dropna()
                lo.insert(0, "arena", arena)
                lo.insert(0, "batch", batch); lo.insert(0, "mouse", mouse)
                long_rows.append(lo)

                ch = per_cluster_change(occ, weeks)
                change_by_arena[arena] = ch
                coll = collective(occ, weeks)
                coll.insert(0, "arena", arena)
                coll.insert(0, "batch", batch); coll.insert(0, "mouse", mouse)
                coll_by_arena[arena] = coll.drop(columns=["mouse", "batch", "arena"])
                coll_rows.append(coll)

            # merge per-cluster change across arenas for the paired comparison
            merged = change_by_arena["2D"].merge(
                change_by_arena["3D"], on="cluster", suffixes=("_2D", "_3D"))
            merged.insert(0, "batch", batch); merged.insert(0, "mouse", mouse)
            change_rows.append(merged)

            # headline: is |occupancy change| larger in 3D than 2D? (paired)
            pair = merged.dropna(subset=["abs_delta_2D", "abs_delta_3D"])
            d2, d3 = pair["abs_delta_2D"].to_numpy(), pair["abs_delta_3D"].to_numpy()
            if len(pair) >= 6 and np.any(d3 - d2):
                stat, pval = wilcoxon(d3, d2)
            else:
                stat, pval = np.nan, np.nan
            headline.append({"mouse": mouse, "batch": batch, "n_clusters": len(pair),
                             "median_absdelta_2D": np.median(d2) if len(d2) else np.nan,
                             "median_absdelta_3D": np.median(d3) if len(d3) else np.nan,
                             "wilcoxon_p": pval})

            plot_collective(coll_by_arena, f"{mouse} {batch}",
                            OUT / f"{mouse}_{batch}_collective.png")
            plot_arena_scatter(merged, f"{mouse} {batch}: occupancy trend by arena",
                               OUT / f"{mouse}_{batch}_trend_scatter.png")
            print(f"  {mouse}/{batch}: {len(clusters)} clusters, {len(weeks)} weeks")

    pd.concat(long_rows, ignore_index=True).to_csv(OUT / "occupancy_long.csv", index=False)
    pd.concat(change_rows, ignore_index=True).to_csv(OUT / "cluster_change_by_arena.csv", index=False)
    pd.concat(coll_rows, ignore_index=True).to_csv(OUT / "collective_by_arena.csv", index=False)
    hl = pd.DataFrame(headline)
    hl.to_csv(OUT / "headline_3d_vs_2d.csv", index=False)

    print(f"\nWrote {OUT}/ (occupancy_long, cluster_change_by_arena, "
          f"collective_by_arena, headline_3d_vs_2d + plots)")
    print("\n=== |occupancy change| 3D vs 2D (paired, per batch) ===")
    print(hl.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
