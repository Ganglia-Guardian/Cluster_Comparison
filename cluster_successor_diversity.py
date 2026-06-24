"""Week-by-week distribution of per-cluster successor diversity, occupancy-controlled
by rarefaction.

The question: as disease progresses, does the *population* of behavioural states
change how it branches -- and crucially, does the change split (some states
fragmenting into many successors while others sharpen into few)? The mean fan-out
line plots hide that; here we plot the whole distribution, week by week.

For each week and each present source cluster we measure two things about its
outgoing transitions:

  * RICHNESS  -- number of distinct successor clusters.
  * PERPLEXITY (effective successors) = exp(entropy of the successor distribution).
    Counts a rare successor far less than a frequent one, so it tracks how
    *deterministic* a state's exits are rather than how many states it ever touches.

THE CONFOUND, AND RAREFACTION. Both measures grow with how often a cluster is
visited (occupancy): a state entered 500x will show more distinct successors than
one entered 5x purely from sampling. As disease redistributes occupancy, a naive
comparison across weeks would partly measure occupancy, not branching. So we
*rarefy*: every included cluster is subsampled to exactly `depth` outgoing
transitions (multivariate hypergeometric, i.e. draw `depth` without replacement
from its successor multiset), averaged over `reps` draws. Clusters with fewer than
`depth` transitions are dropped. After rarefaction every cluster is compared at the
same sampling effort, in any week and any dataset -- and because richness is then
capped at `depth` (not at K), the distributions are comparable across cohorts with
different codebook sizes too.

Outputs per dataset (data/<ds>/):
    successor_diversity.csv             week, source, richness, perplexity (rarefied)
    successor_richness_ridgeline.png    per-week distribution of rarefied richness
    successor_perplexity_ridgeline.png  per-week distribution of effective successors
Combined (data/):
    successor_diversity_over_time.png   median / spread / bimodality of the richness
                                        distribution vs week, all datasets overlaid,
                                        to test the "splits into two modes" hypothesis.

Run:
    python cluster_successor_diversity.py
    python cluster_successor_diversity.py --datasets 1mp 2mp 3mp --depth 20
    python cluster_successor_diversity.py --depth 15 --reps 200
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kurtosis, skew, spearmanr

from cluster_transition_compare import (DATA_ROOT, DEFAULT_DATASETS,
                                        MIN_FRAME_FRAC, cohort, load,
                                        progression_frames)
from cluster_transition_labels import build_transitions, is_variant, week_sort_key

DEPTH = 20    # rarefaction depth: each cluster subsampled to this many transitions
REPS = 100    # subsample draws averaged per cluster
SEED = 0


def rarefied_diversity(df, depth=DEPTH, reps=REPS, seed=SEED):
    """Per (week, source) rarefied richness and perplexity, restricted to
    well-sampled progression weeks and to clusters with >= depth out-transitions.
    Returns a DataFrame and the fraction of (week, source) groups kept."""
    rng = np.random.default_rng(seed)
    trans = build_transitions(df)
    frames = progression_frames(df)
    if frames.empty:
        return pd.DataFrame(columns=["week", "source", "richness", "perplexity", "wn"]), 0.0
    med = float(np.median(frames.to_numpy()))
    keep = {w for w in frames.index if frames[w] >= MIN_FRAME_FRAC * med}
    prog = trans[trans["week"].isin(keep)]

    rows, n_groups, n_kept = [], 0, 0
    for (w, s), g in prog.groupby(["week", "source"]):
        n_groups += 1
        counts = g["target"].value_counts().to_numpy()
        if counts.sum() < depth:
            continue
        n_kept += 1
        # multivariate-hypergeometric subsample: draw `depth` successors without
        # replacement from this cluster's successor multiset, `reps` times.
        draws = rng.multivariate_hypergeometric(counts, depth, size=reps)  # (reps, S)
        richness = float((draws > 0).sum(axis=1).mean())
        p = draws / depth
        with np.errstate(divide="ignore", invalid="ignore"):
            plogp = np.where(draws > 0, p * np.log(p), 0.0)
        perplexity = float(np.exp(-plogp.sum(axis=1)).mean())
        rows.append((w, int(s), richness, perplexity))

    res = pd.DataFrame(rows, columns=["week", "source", "richness", "perplexity"])
    if not res.empty:
        res["wn"] = res["week"].map(week_sort_key).astype(int)
    return res, (n_kept / n_groups if n_groups else 0.0)


def bimodality_coef(x):
    """Sarle's bimodality coefficient. > ~0.555 suggests a bimodal (split)
    distribution; the uniform value is 5/9. Heuristic, but dependency-free."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4 or np.std(x) == 0:
        return np.nan
    g = skew(x, bias=False)
    k = kurtosis(x, fisher=True, bias=False)  # excess kurtosis
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g ** 2 + 1.0) / denom


def plot_ridgeline(res, metric, name, depth, out_path):
    """One filled density curve per week, stacked and time-coloured, so the shape
    of the per-cluster diversity distribution can be compared across weeks."""
    weeks = sorted(res["wn"].unique())
    if len(weeks) < 2:
        return None
    vals_all = res[metric].to_numpy()
    grid = np.linspace(vals_all.min(), vals_all.max(), 200)
    cmap = plt.get_cmap("viridis")
    offset = 0.9 * (len(weeks))  # vertical spacing between ridges

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(weeks) + 2))
    for i, wn in enumerate(weeks):
        v = res.loc[res["wn"] == wn, metric].to_numpy()
        base = (len(weeks) - 1 - i) * 1.0  # earliest week at top
        color = cmap(i / max(1, len(weeks) - 1))
        if len(v) >= 3 and np.std(v) > 0:
            dens = gaussian_kde(v)(grid)
            dens = dens / dens.max() * 0.9
            ax.fill_between(grid, base, base + dens, color=color, alpha=0.8, lw=0.5,
                            edgecolor="white")
        ax.plot([np.median(v)], [base], "|", color="black", ms=8, mew=1.2)
        ax.text(grid[0], base + 0.05, f"w{wn}", fontsize=7, va="bottom")

    ax.set_yticks([])
    label = "distinct successors (rarefied)" if metric == "richness" \
        else "effective successors  exp(H)  (rarefied)"
    ax.set_xlabel(label)
    ax.set_title(f"{name}: per-cluster {metric} distribution by week\n"
                 f"(rarefied to {depth} transitions/cluster; tick = weekly median)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary(per_dataset, depth, out_path):
    """Median, spread (IQR) and bimodality of the rarefied-richness distribution
    vs week, every dataset overlaid. Rising spread / bimodality in disease but not
    control is the signature of the repertoire splitting into two modes."""
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (name, res) in enumerate(per_dataset):
        if res.empty:
            continue
        weeks = sorted(res["wn"].unique())
        med = [res.loc[res.wn == w, "richness"].median() for w in weeks]
        iqr = [res.loc[res.wn == w, "richness"].quantile(.75)
               - res.loc[res.wn == w, "richness"].quantile(.25) for w in weeks]
        bc = [bimodality_coef(res.loc[res.wn == w, "richness"]) for w in weeks]
        color, style = cmap(i % 10), ("--" if cohort(name) == "lc" else "-")
        for ax, y in zip(axes, (med, iqr, bc)):
            yv = np.array(y, float)
            ok = ~np.isnan(yv)
            rho, p = spearmanr(np.array(weeks)[ok], yv[ok]) if ok.sum() > 2 else (np.nan, np.nan)
            ax.plot(weeks, yv, style, marker="o", ms=4, color=color,
                    label=f"{name}: rho={rho:.2f}, p={p:.3f}")

    titles = ["Median rarefied richness", "Spread (IQR) of richness",
              "Bimodality coefficient of richness"]
    for ax, t in zip(axes, titles):
        ax.set_xlabel("disease week"); ax.set_title(t); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[2].axhline(5 / 9, ls=":", color="black", lw=1)
    axes[2].text(axes[2].get_xlim()[0], 5 / 9 + 0.01, "bimodal threshold (0.556)",
                 fontsize=7)
    fig.suptitle("Per-cluster successor-richness distribution over disease "
                 f"(rarefied to {depth})  -- dashed = control, solid = MitoPark",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--depth", type=int, default=DEPTH,
                    help="rarefaction depth (out-transitions per cluster)")
    ap.add_argument("--reps", type=int, default=REPS,
                    help="subsample draws averaged per cluster")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT)
    args = ap.parse_args()

    per_dataset = []
    for name in args.datasets:
        res, frac = rarefied_diversity(load(name), args.depth, args.reps)
        if res.empty:
            print(f"{name}: no clusters with >= {args.depth} transitions; skipped")
            continue
        ds_dir = DATA_ROOT / name
        res.to_csv(ds_dir / "successor_diversity.csv", index=False)
        print(f"{name}: kept {frac*100:.0f}% of clusters at depth {args.depth}; "
              f"wrote {ds_dir / 'successor_diversity.csv'}")
        for metric in ("richness", "perplexity"):
            p = plot_ridgeline(res, metric, name, args.depth,
                               ds_dir / f"successor_{metric}_ridgeline.png")
            if p:
                print(f"  wrote {p}")
        per_dataset.append((name, res))

    if per_dataset:
        p = plot_summary(per_dataset, args.depth,
                         args.out_dir / "successor_diversity_over_time.png")
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
