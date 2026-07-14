"""Cross-animal behavior sharing & clustering validity for the wildtype cohort.

Question this answers
---------------------
We have ONE CSV of cluster-labeled behavioral bouts pooled over 13 wildtype mice
(15 min of head-IMU data each, ~0.3 s bouts, K clusters from an upstream
unsupervised clustering). The mice are concatenated end-to-end; a mouse boundary
is wherever `startTime` resets toward 0. We answer two things:

  1. To what degree is the behavioral repertoire SHARED across the cohort --
     core behaviors that every mouse performs vs. idiosyncratic clusters seen in
     only one animal -- and how much of each mouse's time is spent in the shared
     part?

  2. Did the clustering produce VALID results -- i.e. do the labels describe
     real, reproducible behavior rather than per-animal noise or arbitrary cuts?

What we can and cannot test here
--------------------------------
The CSV holds only bout timing and an integer `clusterLabel`; it does NOT hold
the raw IMU feature vectors. So we cannot compute feature-space INTERNAL indices
(silhouette, Davies-Bouldin) -- `utils.silhouette_*` needs a similarity matrix we
do not have. Instead we validate the codebook the way that actually matters for
behavior: by EXTERNAL reproducibility across 13 independent animals. A real motif
recurs across animals with consistent prevalence, has a characteristic dwell
time, and sits in structured transitions; a noise/over-split cluster does none of
these. Every headline number below is compared against a label-shuffle null, so
"shared" and "valid" come out as numbers with a p-value, not a vibe.

The three validity tests (each vs. its own null)
------------------------------------------------
  reproducible prevalence  Mean pairwise correlation of the per-mouse occupancy
                           vectors, + Kendall's W concordance. Null: relabel
                           clusters independently per mouse (destroys cross-animal
                           correspondence while keeping each mouse's marginals).
                           Observed >> null  ==>  cluster identity carries the
                           same meaning across animals.
  dwell persistence        Mean run length (consecutive windows keeping the same
                           label) vs. a shuffled-sequence null. Bouts here are a
                           FIXED ~0.295 s window, not behavior-defined durations,
                           so the right timescale test is persistence, not
                           duration spread. Observed >> null  ==>  labels mark
                           behaviors that last, not frame-by-frame flicker.
  structured transitions   Normalised mutual information of consecutive *distinct*
                           labels I(curr;next)/H(next), within-mouse, self-
                           transitions excluded so this measures syntax beyond
                           mere persistence. Null: shuffle the sequence. Observed
                           >> null  ==>  transitions between behaviors are
                           predictable, an ethogram rather than an i.i.d. stream.

Run:
    uv run python wt_collection.py
    uv run python wt_collection.py --csv data/all_wt/wildtype_062425_labels.csv
(If a stray VIRTUAL_ENV points at anaconda, prefix with `VIRTUAL_ENV= `.)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import jensenshannon, squareform
from scipy.stats import spearmanr

from utils import save_figure

DATA_ROOT = Path("data")
DEFAULT_CSV = DATA_ROOT / "all_wt" / "wildtype_062425_labels.csv"
OUT_DIRNAME = "wt_collection_out"

N_PERM = 1000            # permutations for every shuffle null
RNG_SEED = 0
CORE_FRAC = 1.0          # present in this fraction of mice -> "core" behavior
RARE_MAX_MICE = 1        # present in <= this many mice -> "idiosyncratic"


# --------------------------------------------------------------------------- #
# Load & split the concatenated cohort into individual mice
# --------------------------------------------------------------------------- #
def load_bouts(csv_path):
    """Read the labels CSV and tag each bout with the mouse it belongs to.

    Mice are concatenated; a new mouse starts wherever `startTime` drops below
    the previous bout's `startTime` (it resets toward 0). Adds integer columns
    `mouse` (0-based, in order of appearance) and `duration` (endTime-startTime).
    """
    df = pd.read_csv(csv_path)
    df["clusterLabel"] = df["clusterLabel"].astype(int)
    df["duration"] = df["endTime"] - df["startTime"]

    # boundary = time runs backwards relative to the previous row
    reset = df["startTime"].to_numpy() < df["startTime"].shift(1).to_numpy()
    reset[0] = False
    df["mouse"] = np.cumsum(reset)
    return df


def occupancy_matrices(df, n_clusters):
    """(mice x clusters) occupancy, both time-weighted and bout-count-weighted.

    Time-weighted is the fraction of each mouse's recorded time spent in each
    cluster (long immobility bouts count for their full length); count-weighted
    is the fraction of bouts. Each row sums to 1. Returns (time_occ, count_occ,
    mouse_ids).
    """
    mice = np.sort(df["mouse"].unique())
    time_occ = np.zeros((len(mice), n_clusters))
    count_occ = np.zeros((len(mice), n_clusters))
    for i, m in enumerate(mice):
        sub = df[df["mouse"] == m]
        t = np.bincount(sub["clusterLabel"], weights=sub["duration"],
                        minlength=n_clusters)
        c = np.bincount(sub["clusterLabel"], minlength=n_clusters)
        time_occ[i] = t / t.sum() if t.sum() else t
        count_occ[i] = c / c.sum() if c.sum() else c
    return time_occ, count_occ, mice


# --------------------------------------------------------------------------- #
# 1. Behavior sharing
# --------------------------------------------------------------------------- #
def sharing_stats(time_occ):
    """How shared is the repertoire? Operates on the (mice x clusters) matrix.

    Returns a dict with, per cluster: in how many mice it appears (`prevalence`),
    its cohort-mean time fraction, and the effective number of mice contributing
    to it (exp of the entropy of its normalised across-mouse usage -- 1 means a
    single mouse owns the cluster, n_mice means perfectly even). Plus cohort-level
    summaries: counts of core / rare clusters and the share of total time they
    explain.
    """
    n_mice, n_clusters = time_occ.shape
    present = time_occ > 0
    prevalence = present.sum(axis=0)                       # mice using each cluster
    mean_use = time_occ.mean(axis=0)

    # effective # mice per cluster from the entropy of its across-mouse profile
    col_tot = time_occ.sum(axis=0, keepdims=True)
    col = time_occ / np.where(col_tot > 0, col_tot, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(col > 0, col * np.log(col), 0.0), axis=0)
    eff_mice = np.exp(ent)

    per_cluster = pd.DataFrame({
        "cluster": np.arange(n_clusters),
        "prevalence": prevalence,
        "mean_time_frac": mean_use,
        "eff_mice": eff_mice,
    }).sort_values("mean_time_frac", ascending=False).reset_index(drop=True)

    seen = prevalence > 0
    core = prevalence >= np.ceil(CORE_FRAC * n_mice)
    rare = (prevalence > 0) & (prevalence <= RARE_MAX_MICE)

    # how much of an average mouse's time lives in core vs rare clusters
    core_time = time_occ[:, core].sum(axis=1).mean()
    rare_time = time_occ[:, rare].sum(axis=1).mean()

    summary = dict(
        n_mice=n_mice, n_clusters_total=n_clusters,
        n_clusters_seen=int(seen.sum()),
        n_core=int(core.sum()), n_rare=int(rare.sum()),
        mean_clusters_per_mouse=float(present.sum(axis=1).mean()),
        core_time_frac=float(core_time), rare_time_frac=float(rare_time),
    )
    return per_cluster, summary, dict(core=core, rare=rare, seen=seen)


def mouse_similarity(time_occ):
    """Pairwise cosine and 1-JS between mice (on time occupancy). Returns both
    (n_mice x n_mice) matrices plus their mean off-diagonal values."""
    n = time_occ.shape[0]
    norm = np.linalg.norm(time_occ, axis=1)
    cos = (time_occ @ time_occ.T) / np.outer(norm, norm)
    js = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(time_occ[i], time_occ[j])
            js[i, j] = js[j, i] = d
    off = ~np.eye(n, dtype=bool)
    return cos, js, float(cos[off].mean()), float((1 - js)[off].mean())


# --------------------------------------------------------------------------- #
# 2. Validity tests (each vs. a label-shuffle null)
# --------------------------------------------------------------------------- #
def reproducible_prevalence(time_occ, rng):
    """Cross-mouse agreement of cluster usage, vs. a per-mouse relabel null.

    Observed = mean pairwise Spearman of the occupancy vectors (rank agreement of
    'which clusters are used a lot'). Kendall's W summarises full-cohort
    concordance (0 = none, 1 = identical rankings). Null independently permutes
    the cluster axis of each mouse, which keeps every mouse's marginal usage but
    destroys the across-animal correspondence of cluster identities.
    """
    n_mice, k = time_occ.shape

    def mean_pairwise_spearman(mat):
        rs = [spearmanr(mat[i], mat[j]).correlation
              for i in range(n_mice) for j in range(i + 1, n_mice)]
        return float(np.nanmean(rs))

    def kendalls_w(mat):
        ranks = np.apply_along_axis(lambda r: pd.Series(r).rank().to_numpy(),
                                    1, mat)
        s = np.sum((ranks.sum(axis=0) - ranks.sum() / k) ** 2)
        return float(12 * s / (n_mice ** 2 * (k ** 3 - k)))

    obs_rho = mean_pairwise_spearman(time_occ)
    obs_w = kendalls_w(time_occ)

    null = np.empty(N_PERM)
    for b in range(N_PERM):
        perm = np.array([time_occ[i, rng.permutation(k)] for i in range(n_mice)])
        null[b] = mean_pairwise_spearman(perm)
    p = float((np.sum(null >= obs_rho) + 1) / (N_PERM + 1))
    return dict(mean_pairwise_spearman=obs_rho, kendalls_w=obs_w,
                null_spearman_mean=float(null.mean()),
                null_spearman_p95=float(np.percentile(null, 95)), p_value=p)


def _run_lengths(seq):
    """Lengths of maximal runs of identical values in a 1-D integer array."""
    if seq.size == 0:
        return np.array([], dtype=int)
    change = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1], True])
    return np.diff(change)


def dwell_persistence(df, rng):
    """Do labels persist across windows? Mean run length vs. a shuffle null.

    Bouts are a fixed ~0.295 s window, so a real behavior shows up as several
    consecutive windows with the same label (a long run). We pool run lengths
    across mice (runs never cross a mouse boundary) and compare the mean to a null
    that shuffles each mouse's sequence -- which keeps the marginal label counts
    but destroys temporal contiguity. Observed >> null means the labels capture
    behaviors that dwell, not per-window noise. `dwell_ratio` = observed/null mean.
    """
    seqs = [df[df["mouse"] == m]["clusterLabel"].to_numpy()
            for m in np.sort(df["mouse"].unique())]

    def mean_run(sequences):
        runs = np.concatenate([_run_lengths(s) for s in sequences])
        return float(runs.mean()) if runs.size else np.nan

    obs = mean_run(seqs)
    null = np.array([mean_run([rng.permutation(s) for s in seqs])
                     for _ in range(N_PERM)])
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    null_mean = float(null.mean())
    return dict(mean_run_len=obs, mean_run_len_sec=obs * 0.295,
                null_run_len_mean=null_mean,
                null_run_len_p95=float(np.percentile(null, 95)),
                dwell_ratio=obs / null_mean if null_mean else np.nan, p_value=p)


def transition_structure(df, n_clusters, rng):
    """Is the label sequence a structured ethogram? Normalised MI of (curr,next).

    Builds first-order transitions between *distinct* consecutive labels WITHIN
    each mouse (self-transitions dropped, so this measures syntax beyond the mere
    persistence already captured by dwell_persistence; no cross-mouse boundaries).
    I(curr;next)/H(next) = the fraction of next-cluster uncertainty removed by
    knowing the current cluster. Null shuffles each mouse's sequence. Observed >>
    null means transitions are predictable, as real behavioral syntax is.
    """
    def nmi_from_seqs(seqs):
        joint = np.zeros((n_clusters, n_clusters))
        for s in seqs:
            if s.size > 1:
                a, b = s[:-1], s[1:]
                keep = a != b                 # drop self-transitions (persistence)
                np.add.at(joint, (a[keep], b[keep]), 1)
        tot = joint.sum()
        if tot == 0:
            return np.nan
        pj = joint / tot
        pc = pj.sum(axis=1)               # current
        pn = pj.sum(axis=0)               # next
        outer = np.outer(pc, pn)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(pj > 0, pj * np.log(pj / outer), 0.0)
            mi = np.nansum(terms)
            hn = -np.nansum(np.where(pn > 0, pn * np.log(pn), 0.0))
        return float(mi / hn) if hn > 0 else np.nan

    seqs = [df[df["mouse"] == m]["clusterLabel"].to_numpy()
            for m in np.sort(df["mouse"].unique())]
    obs = nmi_from_seqs(seqs)
    null = np.array([nmi_from_seqs([rng.permutation(s) for s in seqs])
                     for _ in range(N_PERM)])
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    return dict(normalised_mi=obs, null_nmi_mean=float(np.nanmean(null)),
                null_nmi_p95=float(np.nanpercentile(null, 95)), p_value=p)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_overview(time_occ, mice, per_cluster, summary, cos, js, masks, out_dir):
    """Six-panel figure: sharing on the left, cross-animal validity on the right."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_mice = time_occ.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1) occupancy heatmap (mice x clusters), clusters ordered by mean usage
    order = per_cluster["cluster"].to_numpy()
    ax = axes[0, 0]
    im = ax.imshow(time_occ[:, order] ** 0.5, aspect="auto", cmap="magma")
    ax.set_xlabel("cluster (sorted by cohort mean usage)")
    ax.set_ylabel("mouse"); ax.set_yticks(range(n_mice))
    ax.set_title("Time occupancy (sqrt scale)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2) prevalence histogram: in how many mice does each cluster appear
    ax = axes[0, 1]
    ax.hist(per_cluster["prevalence"], bins=np.arange(0, n_mice + 2) - 0.5,
            color="tab:blue", edgecolor="k")
    ax.axvline(n_mice, color="tab:green", ls="--", label=f"all {n_mice} mice (core)")
    ax.axvline(1, color="tab:red", ls="--", label="single mouse (idiosyncratic)")
    ax.set_xlabel("# mice the cluster appears in"); ax.set_ylabel("# clusters")
    ax.set_title("Cluster prevalence across the cohort")
    ax.legend(fontsize=8)

    # 3) per-mouse time in core vs rare clusters
    ax = axes[0, 2]
    core_t = time_occ[:, masks["core"]].sum(axis=1)
    rare_t = time_occ[:, masks["rare"]].sum(axis=1)
    mid_t = 1 - core_t - rare_t
    x = np.arange(n_mice)
    ax.bar(x, core_t, label="core (all mice)", color="tab:green")
    ax.bar(x, mid_t, bottom=core_t, label="shared (2..n-1)", color="tab:blue")
    ax.bar(x, rare_t, bottom=core_t + mid_t, label="idiosyncratic", color="tab:red")
    ax.set_xlabel("mouse"); ax.set_ylabel("fraction of time"); ax.set_ylim(0, 1)
    ax.set_xticks(x); ax.set_title("Where each mouse spends its time")
    ax.legend(fontsize=8, loc="lower right")

    # 4) mouse-mouse cosine similarity, ordered by hierarchical clustering
    ax = axes[1, 0]
    dist = squareform(js, checks=False)
    Z = linkage(dist, method="average")
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    sim = cos[np.ix_(leaves, leaves)]
    im = ax.imshow(sim, cmap="viridis", vmin=float(np.min(cos)), vmax=1)
    ax.set_xticks(range(n_mice)); ax.set_xticklabels(np.array(mice)[leaves], fontsize=7)
    ax.set_yticks(range(n_mice)); ax.set_yticklabels(np.array(mice)[leaves], fontsize=7)
    ax.set_title("Mouse-mouse occupancy cosine\n(ordered by JS dendrogram)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 5) dendrogram of mice
    ax = axes[1, 1]
    dendrogram(Z, labels=[str(m) for m in mice], ax=ax, color_threshold=0)
    ax.set_xlabel("mouse"); ax.set_ylabel("JS distance")
    ax.set_title("Do the mice form one cohort or subgroups?")

    # 6) effective # mice per cluster vs how much time it gets (single-mouse = suspect)
    ax = axes[1, 2]
    sc = ax.scatter(per_cluster["eff_mice"], per_cluster["mean_time_frac"],
                    c=per_cluster["prevalence"], cmap="plasma", s=25)
    ax.set_yscale("log")
    ax.axvline(1.5, color="tab:red", ls="--", lw=1, label="~single-mouse clusters")
    ax.set_xlabel("effective # mice (entropy of usage)")
    ax.set_ylabel("cohort mean time fraction (log)")
    ax.set_title("Per-cluster spread vs. prevalence")
    ax.legend(fontsize=8); fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                                        label="# mice present")

    fig.suptitle(
        f"Wildtype cohort: behavior sharing & clustering validity  "
        f"(n={n_mice} mice, {summary['n_clusters_seen']}/{summary['n_clusters_total']} "
        f"clusters used)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out_dir / "wt_collection_overview.jpeg", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--n-perm", type=int, default=N_PERM)
    return p.parse_args()


def main():
    args = parse_args()
    global N_PERM
    N_PERM = args.n_perm
    rng = np.random.default_rng(RNG_SEED)

    df = load_bouts(args.csv)
    n_clusters = int(df["clusterLabel"].max()) + 1
    out_dir = args.csv.parent / OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    time_occ, count_occ, mice = occupancy_matrices(df, n_clusters)
    print(f"Loaded {len(df)} bouts, {len(mice)} mice, K={n_clusters} clusters "
          f"(used: {int((time_occ.sum(0) > 0).sum())}).")

    # 1. sharing
    per_cluster, summary, masks = sharing_stats(time_occ)
    cos, js, mean_cos, mean_1mjs = mouse_similarity(time_occ)
    summary["mean_pairwise_cosine"] = mean_cos
    summary["mean_pairwise_1_minus_js"] = mean_1mjs

    # 2. validity
    rep = reproducible_prevalence(time_occ, rng)
    dwell = dwell_persistence(df, rng)
    trans = transition_structure(df, n_clusters, rng)

    # save tables
    per_cluster.to_csv(out_dir / "per_cluster_sharing.csv", index=False)
    validity = pd.DataFrame([
        dict(test="reproducible_prevalence", statistic="mean_pairwise_spearman",
             observed=rep["mean_pairwise_spearman"], null_p95=rep["null_spearman_p95"],
             p_value=rep["p_value"], extra=f"Kendall_W={rep['kendalls_w']:.3f}"),
        dict(test="dwell_persistence", statistic="mean_run_len_windows",
             observed=dwell["mean_run_len"], null_p95=dwell["null_run_len_p95"],
             p_value=dwell["p_value"],
             extra=f"ratio={dwell['dwell_ratio']:.2f}x, "
                   f"{dwell['mean_run_len_sec']:.2f}s"),
        dict(test="transition_structure", statistic="normalised_MI",
             observed=trans["normalised_mi"], null_p95=trans["null_nmi_p95"],
             p_value=trans["p_value"], extra=""),
    ])
    validity.to_csv(out_dir / "validity_tests.csv", index=False)

    plot_overview(time_occ, mice, per_cluster, summary, cos, js, masks, out_dir)

    # report
    print("\n--- behavior sharing ---")
    print(f"  clusters used by ALL {summary['n_mice']} mice (core): "
          f"{summary['n_core']}  |  single-mouse (idiosyncratic): {summary['n_rare']}")
    print(f"  mean clusters per mouse: {summary['mean_clusters_per_mouse']:.1f}")
    print(f"  avg mouse spends {summary['core_time_frac']*100:.1f}% of time in core, "
          f"{summary['rare_time_frac']*100:.2f}% in idiosyncratic clusters")
    print(f"  mean pairwise mouse similarity: cosine={mean_cos:.3f}, "
          f"1-JS={mean_1mjs:.3f}")

    print("\n--- clustering validity (vs. label-shuffle null) ---")
    def verdict(p): return "PASS" if p < 0.05 else "n.s."
    print(f"  reproducible prevalence: Spearman={rep['mean_pairwise_spearman']:.3f} "
          f"(null<= {rep['null_spearman_p95']:.3f}), Kendall W={rep['kendalls_w']:.3f}, "
          f"p={rep['p_value']:.3g}  [{verdict(rep['p_value'])}]")
    print(f"  dwell persistence: mean run={dwell['mean_run_len']:.2f} windows "
          f"({dwell['mean_run_len_sec']:.2f}s) = {dwell['dwell_ratio']:.2f}x null "
          f"(null<= {dwell['null_run_len_p95']:.2f}), p={dwell['p_value']:.3g}  "
          f"[{verdict(dwell['p_value'])}]")
    print(f"  structured transitions: nMI={trans['normalised_mi']:.3f} "
          f"(null<= {trans['null_nmi_p95']:.3f}), p={trans['p_value']:.3g}  "
          f"[{verdict(trans['p_value'])}]")
    print(f"\n  -> plots + CSVs in {out_dir}")
    print("  NOTE: feature-space indices (silhouette/Davies-Bouldin) need the raw "
          "IMU\n        vectors, which are not in this CSV; these are external "
          "cross-animal\n        reproducibility tests instead.")


if __name__ == "__main__":
    main()
