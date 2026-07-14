"""Presence-distribution (temporal) half of the cluster-degeneracy plane.

The plan (per mouse, clustering is per-mouse so everything is within-mouse):

    Two pairwise metrics per cluster pair (a, b), plotted as a plane:
      X  feature distance   = -mean( obs-obs affinity over i in a, j in b )   [built later]
      Y  presence distance  = log( CDF-L1 between weekly occupancy of a, b )

    CDF-L1 on the *ordered* week axis == 1-D Wasserstein-1 (Earth Mover), so two
    clusters peaking in adjacent weeks read as closer than two peaking far apart.
    log() spreads [0, inf) onto the real line; degenerate/co-temporal pairs fall
    into the left (-inf) tail. A signed centroid displacement (delta mean-week)
    is computed alongside so feature-similar pairs can be *ordered* into chains
    (CDF-L1 is symmetric and cannot say which cluster comes first).

This module owns the Y axis + chain ordering. It needs only the detail CSVs.

Design choices (see conversation):
  * per mouse, one plane each (cluster indices are not shared across mice).
  * shape-only: each cluster's weekly vector is normalized to sum 1, so timing
    shape matters, not cluster size.
  * within-week option: divide each week by its total first (cluster's *share* of
    the week) so weeks with more data don't dominate. Default on; reported either
    way so the effect is visible.
  * challenge weeks (week_24_ldop, week_24_saline) are dropped from the time axis
    -- they are acute drug challenges, not natural progression.
  * min-count guard: clusters with too few observations get spiky histograms that
    inflate Wasserstein as noise; drop below MIN_COUNT and report how many.

Run from repo root:
    C:/ProgramData/anaconda3/python.exe degeneracy_analysis/presence_similarity.py
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# repo root on the path so the shared config imports whether this stage is run
# via the pipeline runner (which sets PYTHONPATH) or directly from repo root.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import dataset_config
from utils import save_figure

# Data root + degeneracy out dir are env-driven (CLUSTER_DATA_ROOT /
# CLUSTER_DEGEN_OUT), so the pipeline runner can point every stage at another
# cohort (e.g. early_analysis/data) at once; defaults reproduce the old paths.
DATA = str(dataset_config.data_root())
OUT = dataset_config.degen_out_root(DATA)


def discover_mice(data_dir=None):
    """Mouse folders under the data root with a Cluster_detail_results.csv, sorted.

    Honours CLUSTER_DATASETS / CLUSTER_DATASET_GLOB so a run can be narrowed to
    specific folders or to the <n>lc / <n>mp cohorts. Auto-discovers otherwise,
    so dropping in a new <root>/<mouse>/ folder is picked up with no code change.
    """
    return dataset_config.discover_datasets(data_dir if data_dir is not None else DATA)


MICE = discover_mice()

MIN_COUNT = 200          # drop clusters with fewer total obs over natural weeks
MIN_WEEK_FRAC = 0.1      # drop weeks with < this fraction of the median week's obs
# Weekly totals are ~uniform (~5970) except failed recordings (1lc wk10 = 16 obs),
# so dividing each week by its total buys nothing and would over-weight the sparse
# weeks. Off by default; unequal-spacing Wasserstein handles the dropped-week gaps.
WITHIN_WEEK_NORM = False

# Folder naming differs by mouse: w8 / week_8 / week8, and drug challenges named
# w24_ldopa, week_24_ldop, or LDOPA_week24_hightier_{ldopa,saline}. A natural week
# has a week number and is NOT a drug challenge.
_WEEK_RE = re.compile(r"w(?:eek)?_?(\d+)", re.IGNORECASE)
_CHALLENGE = re.compile(r"ldop|salin", re.IGNORECASE)


def natural_week(folder):
    """Integer week for a natural-progression folder, else None (challenge weeks)."""
    s = str(folder)
    if _CHALLENGE.search(s):
        return None
    m = _WEEK_RE.search(s)
    return int(m.group(1)) if m else None


def load_counts(mouse):
    """cluster x week count matrix over natural weeks only.

    Returns (counts, clusters, weeks, dropped_challenge_frac) where counts is
    (n_clusters, n_weeks) of raw observation counts.
    """
    det = pd.read_csv(f"{DATA}/{mouse}/Cluster_detail_results.csv")
    det["wk"] = det.Folder_Name.map(natural_week)
    n_total = len(det)
    det = det.dropna(subset=["wk"]).copy()
    det["wk"] = det.wk.astype(int)
    dropped = 1.0 - len(det) / n_total

    clusters = np.sort(det.ClusterIdx.unique())
    weeks = np.sort(det.wk.unique())
    ci = {c: i for i, c in enumerate(clusters)}
    wi = {w: j for j, w in enumerate(weeks)}
    counts = np.zeros((len(clusters), len(weeks)), float)
    g = det.groupby(["ClusterIdx", "wk"]).size().reset_index(name="n")
    for c, w, n in g.itertuples(index=False):
        counts[ci[c], wi[w]] = n

    # drop failed/partial recording weeks (e.g. 1lc wk10 = 16 obs)
    wk_tot = counts.sum(axis=0)
    good = wk_tot >= MIN_WEEK_FRAC * np.median(wk_tot)
    dropped_weeks = list(weeks[~good])
    counts, weeks = counts[:, good], weeks[good]
    return counts, clusters, weeks, dropped, dropped_weeks


def presence_distributions(counts, weeks, within_week_norm=WITHIN_WEEK_NORM):
    """Shape-normalized weekly occupancy per cluster (rows sum to 1)."""
    C = counts.copy()
    if within_week_norm:
        col = C.sum(axis=0, keepdims=True)
        col[col == 0] = 1.0
        C = C / col                       # cluster's share of each week
    row = C.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return C / row                        # across-week shape, sums to 1


def cdf_l1_matrix(P, weeks):
    """Pairwise 1-D Wasserstein-1 (== CDF-L1 on the ordered, possibly unequally
    spaced week axis). P is (K, W) rows summing to 1."""
    w = np.asarray(weeks, float)
    gaps = np.diff(w)                      # spacing between successive weeks
    cdf = np.cumsum(P, axis=1)[:, :-1]     # drop last col (its |diff| weight is 0)
    K = P.shape[0]
    D = np.zeros((K, K))
    for a in range(K):
        # |CDF_a - CDF_b| weighted by gap, summed over weeks; vectorized over b
        D[a] = (np.abs(cdf[a][None, :] - cdf) * gaps[None, :]).sum(axis=1)
    return D


def centroids(P, weeks):
    """Temporal centroid (mean week) per cluster; delta gives chain direction."""
    return P @ np.asarray(weeks, float)


def analyze(mouse, min_count=MIN_COUNT, within_week_norm=WITHIN_WEEK_NORM):
    counts, clusters, weeks, dropped, dropped_weeks = load_counts(mouse)
    totals = counts.sum(axis=1)
    keep = totals >= min_count
    n_drop = int((~keep).sum())

    counts_k = counts[keep]
    clusters_k = clusters[keep]
    P = presence_distributions(counts_k, weeks, within_week_norm)

    W = cdf_l1_matrix(P, weeks)                 # Wasserstein / CDF-L1 distance
    cen = centroids(P, weeks)
    delta = cen[:, None] - cen[None, :]         # signed mean-week displacement

    off = W[~np.eye(len(W), dtype=bool)]
    pos = off[off > 0]
    eps = 0.5 * pos.min() if pos.size else 1e-6
    n_zero_pairs = int((off == 0).sum())
    logW = np.log(W + eps)
    np.fill_diagonal(logW, np.nan)              # self-pairs are not data points

    os.makedirs(f"{OUT}/{mouse}", exist_ok=True)
    np.savez(f"{OUT}/{mouse}/presence.npz",
             clusters=clusters_k, weeks=weeks, presence=P,
             wasserstein=W, log_wasserstein=logW, centroid=cen, delta=delta,
             eps=eps, within_week_norm=within_week_norm)

    weekly_tot = counts.sum(axis=0)
    print(f"\n{'='*68}\n{mouse}: presence axis\n{'='*68}")
    print(f"  natural weeks: {list(weeks)}  ({len(weeks)} weeks)")
    if dropped_weeks:
        print(f"  dropped failed/partial weeks (< {MIN_WEEK_FRAC:.0%} median obs): {dropped_weeks}")
    print(f"  obs in challenge weeks (dropped from time axis): {dropped*100:.1f}%")
    print(f"  clusters: {len(clusters)} total, dropped {n_drop} below {min_count} obs, "
          f"{len(clusters_k)} kept")
    print(f"  weekly totals min/median/max: {weekly_tot.min():.0f} / "
          f"{np.median(weekly_tot):.0f} / {weekly_tot.max():.0f}  "
          f"(ratio {weekly_tot.max()/max(weekly_tot.min(),1):.1f}x -> within-week "
          f"norm {'matters' if weekly_tot.max()/max(weekly_tot.min(),1) > 1.5 else 'minor'})")
    print(f"  log-eps = {eps:.4g}   exact-zero off-diagonal pairs = {n_zero_pairs}")
    print(f"  Wasserstein (weeks): median {np.median(pos):.3f}, "
          f"range [{pos.min():.4f}, {off.max():.3f}]")

    _sanity_plot(mouse, clusters_k, totals[keep], W, logW, cen, weeks)
    return dict(mouse=mouse, clusters=clusters_k, W=W, logW=logW,
                centroid=cen, delta=delta, presence=P, weeks=weeks)


def _sanity_plot(mouse, clusters, totals, W, logW, cen, weeks):
    off_log = logW[~np.eye(len(logW), dtype=bool)]
    off_log = off_log[np.isfinite(off_log)]
    # per-cluster mean Wasserstein to all others (low-count noise check)
    mean_w = (W.sum(axis=1)) / (len(W) - 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(off_log, bins=60, color="#3b6", edgecolor="none")
    ax[0].set(title=f"{mouse}: log CDF-L1 (presence axis marginal)",
              xlabel="log Wasserstein (weeks)", ylabel="cluster pairs")
    ax[0].axvline(np.log(0.5 * W[W > 0].min()), color="k", ls=":", lw=1)

    ax[1].scatter(totals, mean_w, s=14, alpha=0.6, color="#36b")
    ax[1].set(xscale="log", title=f"{mouse}: low-count noise check",
              xlabel="cluster total obs (log)", ylabel="mean Wasserstein to others")

    # cluster centroids binned by half-week: tighter => more temporally centered
    bins = np.arange(np.floor(weeks.min()), np.ceil(weeks.max()) + 0.5, 0.5)
    ax[2].hist(cen, bins=bins, color="#b63", edgecolor="white", lw=0.4)
    ax[2].axvline(cen.mean(), color="k", ls="--", lw=1)
    ax[2].set(title=f"{mouse}: centroid weeks  (mean={cen.mean():.1f}, std={cen.std():.2f})",
              xlabel="centroid week", ylabel="# clusters")
    ax[2].set_xticks(weeks[::2])
    fig.tight_layout()
    save_figure(fig, f"{OUT}/{mouse}/presence_sanity.jpeg", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    for m in MICE:
        analyze(m)
