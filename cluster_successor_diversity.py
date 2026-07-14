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
    successor_richness_ridgeline.jpeg   per-week distribution of rarefied richness
    successor_perplexity_ridgeline.jpeg per-week distribution of effective successors
Combined (data/):
    successor_diversity_over_time.jpeg  median / spread / bimodality of the richness
                                        distribution vs week, all datasets overlaid,
                                        to test the "splits into two modes" hypothesis.

Run:
    python cluster_successor_diversity.py
    python cluster_successor_diversity.py --arena 3d      # high-tier only
    python cluster_successor_diversity.py --arena open    # *_open datasets only
    python cluster_successor_diversity.py --datasets 1mp 2mp 3mp --depth 20
    python cluster_successor_diversity.py --depth 15 --reps 200

The --arena {both,open,3d} flag selects open-field (*_open), 3d/high-tier
(everything else), or both (default); subset runs write a suffixed summary
(successor_diversity_over_time_{open,3d}.jpeg) so the 'both' plot is preserved.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.special import gammaln
from scipy.stats import gaussian_kde, kurtosis, skew, spearmanr

import dataset_config
from cluster_transition_compare import (DATA_ROOT, DEFAULT_DATASETS,
                                        MIN_FRAME_FRAC, cohort, load,
                                        progression_frames, set_data_root)
from cluster_transition_labels import build_transitions, is_variant, week_sort_key
from utils import cohort_colors, save_figure

DEPTH = 20    # rarefaction depth: each cluster subsampled to this many transitions
REPS = 100    # subsample draws averaged per cluster
SEED = 0


def hurlbert_richness(counts, depth):
    """Exact EXPECTED number of distinct successors in a without-replacement
    subsample of `depth` transitions (Hurlbert 1971 rarefaction). Deterministic --
    replaces the Monte-Carlo estimate of richness, so there is no seed/reps noise.

    E[distinct] = sum_i (1 - P(successor i absent)),  where
    P(i absent) = C(N - n_i, depth) / C(N, depth), computed in log-gamma space.
    A successor with fewer than `depth` 'other' transitions must appear -> P=0."""
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    if N < depth:
        return np.nan
    other = N - counts
    with np.errstate(over="ignore", invalid="ignore"):
        log_absent = (gammaln(other + 1) - gammaln(other - depth + 1)
                      - (gammaln(N + 1) - gammaln(N - depth + 1)))
        p_absent = np.exp(log_absent)
    p_absent = np.where(other < depth, 0.0, p_absent)     # must-appear successors
    return float((1.0 - p_absent).sum())


def rarefied_diversity(df, depth=DEPTH, reps=REPS, seed=SEED):
    """Per (week, source) rarefied richness and perplexity, restricted to
    well-sampled progression weeks and to clusters with >= depth out-transitions.
    Richness is the exact Hurlbert expectation (deterministic); perplexity has no
    closed form so it stays Monte-Carlo (reps draws). Returns a DataFrame and the
    fraction of (week, source) groups kept."""
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
        richness = hurlbert_richness(counts, depth)          # exact, deterministic
        # perplexity (effective successors) has no closed form -> Monte-Carlo
        draws = rng.multivariate_hypergeometric(counts, depth, size=reps)
        p = draws / depth
        with np.errstate(divide="ignore", invalid="ignore"):
            plogp = np.where(draws > 0, p * np.log(p), 0.0)
        perplexity = float(np.exp(-plogp.sum(axis=1)).mean())
        rows.append((w, int(s), richness, perplexity))

    res = pd.DataFrame(rows, columns=["week", "source", "richness", "perplexity"])
    if not res.empty:
        res["wn"] = res["week"].map(week_sort_key).astype(int)
    return res, (n_kept / n_groups if n_groups else 0.0)


def variant_richness(df, depth=DEPTH):
    """Per-source Hurlbert richness for each pharmacological arm (saline, ldopa),
    computed like the progression weeks but on the challenge-arm transitions only.
    Returns {arm: array of per-source richness}. Saline is matched first because the
    mp arm folders contain both 'ldopa' and 'saline' in the name."""
    trans = build_transitions(df)
    out = {}
    for w in pd.unique(trans["week"]):
        if not is_variant(w):
            continue
        arm = "saline" if "saline" in str(w).lower() else "ldopa"
        vals = []
        for _, g in trans[trans["week"] == w].groupby("source"):
            counts = g["target"].value_counts().to_numpy()
            if counts.sum() >= depth:
                vals.append(hurlbert_richness(counts, depth))
        if vals:
            out[arm] = np.array(vals, float)
    return out


def boot_median_ci(vals, rng, B=2000, q=(2.5, 97.5)):
    """Percentile-bootstrap CI for the median: resample clusters with replacement."""
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3:
        return np.nan, np.nan
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    meds = np.median(vals[idx], axis=1)
    return tuple(np.percentile(meds, q))


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
    save_figure(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _legend_order_key(name):
    """Sort datasets *lc-ascending then *mp-ascending: (cohort, numeric prefix,
    name). cohort 'lc' < 'mp' alphabetically; leading digits give within-cohort
    order. Robust to added datasets / multi-digit prefixes."""
    lead = ""
    for ch in name:
        if ch.isdigit():
            lead += ch
        else:
            break
    return (cohort(name), int(lead) if lead else 0, name)


def plot_summary(per_dataset, depth, out_path):
    """Median rarefied-richness vs week, every dataset overlaid in its UTD cohort
    shade (orange = control, green = MitoPark). No CI band: with every line a
    shade of the same two hues, translucent bands smear together and bury the
    medians. The saline (triangle) and L-DOPA (diamond) week-24 challenge arms are
    drawn after the last week -- same colour per dataset, not line-connected, each
    with its own bootstrap CI (isolated markers, so those don't overlap). Legend
    sorted *lc then *mp, ascending. (Spread/IQR and bimodality were checked before
    and showed no correlation, so only the median is plotted.)"""
    rng = np.random.default_rng(0)
    ordered = sorted((d for d in per_dataset if not d[1].empty),
                     key=lambda d: _legend_order_key(d[0]))
    if not ordered:
        return out_path
    colors = cohort_colors([name for name, _, _ in ordered])
    weeks_all = sorted({int(w) for _, res, _ in ordered for w in res["wn"].unique()})
    wmax = weeks_all[-1]

    # Only draw a challenge column/marker/legend for an arm that some dataset
    # actually has (cohorts without a wk24 saline/l-dopa challenge, e.g. weeks
    # 8-13, get a plain progression plot with no challenge apparatus).
    ARM_ORDER = [("saline", "^", "saline"), ("ldopa", "D", "l-dopa")]

    def _arm_present(arm):
        return any(variants and variants.get(arm) is not None and len(variants[arm])
                   for _, _, variants in ordered)

    present_arms = [a for a in ARM_ORDER if _arm_present(a[0])]
    arm_x = {arm: (wmax + 1 + j, mk)
             for j, (arm, mk, _lab) in enumerate(present_arms)}

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for name, res, variants in ordered:
        color = colors[name]
        weeks = sorted(res["wn"].unique())
        med = np.array([np.median(res.loc[res.wn == w, "richness"].to_numpy())
                        for w in weeks])
        ok = ~np.isnan(med)
        rho, p = spearmanr(np.array(weeks)[ok], med[ok]) if ok.sum() > 2 else (np.nan, np.nan)
        ax.plot(weeks, med, "-", marker="o", ms=4, color=color,
                label=f"{name}: rho={rho:.2f}, p={p:.3f}")
        # week-24 challenge arms: standalone markers + bootstrap CI, no connecting line
        for arm, (xpos, mk) in arm_x.items():
            v = variants.get(arm) if variants else None
            if v is not None and len(v):
                m = float(np.median(v))
                l, h = boot_median_ci(v, rng)
                yerr = None if np.isnan(l) else [[m - l], [h - m]]
                ax.errorbar([xpos], [m], yerr=yerr, fmt=mk, color=color, ms=8,
                            capsize=3, elinewidth=1, mec="black", mew=0.4, zorder=5)

    arm_ticks = [arm_x[arm][0] for arm, _, _ in present_arms]
    if present_arms:
        ax.axvline(wmax + 0.5, ls=":", color="0.6", lw=1)   # progression | challenge divider
    prog_ticks = list(range(weeks_all[0], wmax + 1, 2))
    ticks = prog_ticks + arm_ticks
    ax.set_xticks(ticks)
    labels = ax.set_xticklabels([str(t) for t in prog_ticks]
                                + [lab for _, _, lab in present_arms])
    for lab, t in zip(labels, ticks):
        if t in arm_ticks:
            lab.set_rotation(45); lab.set_ha("right")
            lab.set_style("italic"); lab.set_fontsize(8)

    ax.set_xlabel("disease week   →   wk24 challenge" if present_arms
                  else "disease week")
    ax.set_ylabel("median rarefied successor richness")
    ax.set_title(f"Median rarefied successor richness over disease "
                 f"(rarefied to {depth})\n"
                 f"orange = control (lc), green = MitoPark (mp)")
    leg1 = ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1),
                     title=f"dataset:  Spearman rho (weeks {weeks_all[0]}-{wmax})")
    ax.add_artist(leg1)
    extra = [leg1]
    if present_arms:
        arm_label = {"saline": "saline arm", "ldopa": "L-DOPA arm"}
        shape_handles = [
            Line2D([0], [0], marker=mk, color="0.4", ls="none", mec="black",
                   mew=0.4, label=arm_label[arm])
            for arm, mk, _lab in present_arms
        ]
        leg2 = ax.legend(handles=shape_handles, fontsize=8, loc="lower left",
                         title="wk24 challenge")
        ax.add_artist(leg2)
    save_figure(fig, out_path, dpi=150, bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    dataset_config.add_dataset_args(ap)
    ap.add_argument("--arena", choices=["both", "open", "3d"], default="both",
                    help="which arena to include: open (*_open datasets), 3d "
                         "(high-tier, everything else), or both (default)")
    ap.add_argument("--depth", type=int, default=DEPTH,
                    help="rarefaction depth (out-transitions per cluster)")
    ap.add_argument("--reps", type=int, default=REPS,
                    help="subsample draws averaged per cluster")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the combined figure (default: data root)")
    args = ap.parse_args()

    root, all_datasets = dataset_config.resolve_datasets(args)
    if not all_datasets:
        raise SystemExit(f"no datasets found under {root}")
    set_data_root(root)                       # so the imported load() uses it
    out_dir = args.out_dir or root

    is_open = {"open": lambda n: n.endswith("_open"),
               "3d": lambda n: not n.endswith("_open"),
               "both": lambda n: True}[args.arena]
    datasets = [d for d in all_datasets if is_open(d)]
    print(f"data root: {root}   arena={args.arena}: {datasets}")

    per_dataset = []
    for name in datasets:
        df = load(name)
        res, frac = rarefied_diversity(df, args.depth, args.reps)
        if res.empty:
            print(f"{name}: no clusters with >= {args.depth} transitions; skipped")
            continue
        ds_dir = root / name
        res.to_csv(ds_dir / "successor_diversity.csv", index=False)
        print(f"{name}: kept {frac*100:.0f}% of clusters at depth {args.depth}; "
              f"wrote {ds_dir / 'successor_diversity.csv'}")
        for metric in ("richness", "perplexity"):
            p = plot_ridgeline(res, metric, name, args.depth,
                               ds_dir / f"successor_{metric}_ridgeline.jpeg")
            if p:
                print(f"  wrote {p}")
        variants = variant_richness(df, args.depth)
        per_dataset.append((name, res, variants))

    if per_dataset:
        suffix = "" if args.arena == "both" else f"_{args.arena}"
        p = plot_summary(per_dataset, args.depth,
                         out_dir / f"successor_diversity_over_time{suffix}.jpeg")
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
