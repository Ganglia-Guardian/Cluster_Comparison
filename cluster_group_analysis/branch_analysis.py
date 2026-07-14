"""Characterize each dendrogram branch by arena purity, TBA, and week
distribution, and answer the three questions the branches were built to probe:

  Q1  Are the 2D/3D ratios the same for clusters in the same branch?
        -> within-branch spread of cluster occ3d, and eta^2 = fraction of
           cluster-level occ3d variance explained by branch (+ Kruskal-Wallis).
           High eta^2 / low within-branch spread => a branch's clusters DO share
           an arena preference (kinematic branches carry arena structure).
  Q2  Are there branches that are primarily 2D or primarily 3D?
        -> per-branch frame-aggregated occ3d (arena-time-normalized). Verdict
           2D if <0.40, 3D if >0.60, else mixed.
  Q3  What does the week distribution look like for each branch?
        -> per-branch frame share by week + a branch x week chi-square
           (Cramer's V) testing whether branches have distinct temporal profiles.

TBA is one of the merge features, so its branch eta^2 is high by construction;
we report it only as a ceiling to read the occ3d eta^2 against (how much arena
structure comes along "for free" with a purely kinematic merge).

Branches are pooled per mouse (w8+w9+w10 together), matching dendrogram.py.

Inputs:  cluster_branches.csv, cluster_week_counts.csv  (built by the prior steps)
Outputs (cluster_group_analysis/output/branches/):
    branch_summary.csv   one row per (mouse, branch)
    mouse_stats.csv      one row per mouse: eta^2 / test results
    <mouse>_branches.jpeg   3-panel: occ3d, TBA, week heatmap per branch
    overview.jpeg        eta^2(occ3d) vs eta^2(TBA) across mice + verdict counts

Run:  uv run python cluster_group_analysis/branch_analysis.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

from common import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from utils import save_figure  # noqa: E402

OUT = ROOT / "output" / "branches"
OCC_CMAP = plt.cm.RdBu
WEEK_CMAP = plt.cm.magma


def eta_squared(values, groups):
    """Correlation ratio eta^2: fraction of total variance of `values` between
    `groups` (0 = branch tells you nothing, 1 = branch fully determines it)."""
    values = np.asarray(values, float)
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    if ss_tot == 0:
        return np.nan
    ss_between = sum(len(v) * (v.mean() - grand) ** 2
                     for v in (values[groups == g] for g in np.unique(groups)))
    return ss_between / ss_tot


def arena_verdict(occ3d_agg):
    return "2D" if occ3d_agg < 0.40 else ("3D" if occ3d_agg > 0.60 else "mixed")


def branch_summary(sub, total_2d, total_3d):
    """One row per branch for a (mouse, batch): purity, TBA, frame share."""
    rows = []
    for b, g in sub.groupby("branch"):
        P2 = g["n_2d"].sum() / total_2d if total_2d else 0.0
        P3 = g["n_3d"].sum() / total_3d if total_3d else 0.0
        occ_agg = P3 / (P3 + P2) if (P3 + P2) else 0.5
        rows.append({
            "branch": int(b), "n_clusters": len(g),
            "n_frames": int(g["n_frames"].sum()),
            "occ3d_agg": occ_agg,                       # frame-aggregated (Q2)
            "occ3d_mean": g["occ3d"].mean(),            # unweighted cluster mean
            "occ3d_std": g["occ3d"].std(ddof=0),        # within-branch spread (Q1)
            "tba_mean": g["TotAccelBA"].mean(),
            "tba_std": g["TotAccelBA"].std(ddof=0),
            "arena_verdict": arena_verdict(occ_agg),
        })
    return pd.DataFrame(rows).sort_values("branch").reset_index(drop=True)


def week_matrix(wk_sub, branches):
    """branch x week frame-count matrix (rows=branch sorted, cols=week sorted)."""
    m = (wk_sub.groupby(["branch", "week"])["n"].sum()
               .unstack(fill_value=0).sort_index())
    return m.reindex(sorted(branches), fill_value=0)


def plot_batch(bs, wm, title, path):
    weeks = wm.columns.to_numpy()
    branches = bs["branch"].to_numpy()
    fig, (a0, a1, a2) = plt.subplots(
        1, 3, figsize=(13, 0.5 + 0.42 * len(branches) + 1.5),
        gridspec_kw={"width_ratios": [1, 1, 1.4]})

    y = np.arange(len(branches))
    # Q2: arena purity per branch
    a0.barh(y, bs["occ3d_agg"], color=[OCC_CMAP(v) for v in bs["occ3d_agg"]],
            edgecolor="k", lw=0.4)
    a0.axvline(0.5, color="k", lw=0.8)
    for xv in (0.40, 0.60):
        a0.axvline(xv, color="k", ls=":", lw=0.7, alpha=0.6)
    # Q1: overlay cluster-occ3d spread as an error bar (mean +/- std)
    a0.errorbar(bs["occ3d_mean"], y, xerr=bs["occ3d_std"], fmt="none",
                ecolor="k", elinewidth=1, capsize=3, alpha=0.7)
    a0.set(yticks=y, yticklabels=branches, xlim=(0, 1),
           xlabel="occ3d (bar=agg, whisker=cluster mean±std)", ylabel="branch",
           title="Arena purity")
    a0.invert_yaxis()

    # TBA per branch
    a1.barh(y, bs["tba_mean"], xerr=bs["tba_std"], color="#4c72b0",
            edgecolor="k", lw=0.4, error_kw=dict(elinewidth=1, capsize=3))
    a1.set(yticks=y, yticklabels=[], xlabel="mean TBA (±std)", title="TBA")
    a1.invert_yaxis()

    # Q3: week distribution per branch (row-normalized frame share)
    frac = wm.to_numpy(float)
    frac = frac / frac.sum(1, keepdims=True).clip(min=1)
    im = a2.imshow(frac, aspect="auto", cmap=WEEK_CMAP)
    a2.set(xticks=np.arange(len(weeks)), xticklabels=weeks,
           yticks=y, yticklabels=branches, xlabel="week",
           title="Week distribution (row-normalized)")
    a2.invert_yaxis()
    fig.colorbar(im, ax=a2, fraction=0.04, pad=0.02).set_label("frame share")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    save_figure(fig, path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    br = pd.read_csv(ROOT / "cluster_branches.csv")
    wk = pd.read_csv(ROOT / "cluster_week_counts.csv")
    wk = wk.merge(br[["mouse", "batch", "cluster", "branch"]],
                  on=["mouse", "batch", "cluster"], how="inner")

    summaries, stats = [], []
    for mouse, sub in br.groupby("mouse", sort=False):
        sub = sub.reset_index(drop=True)
        total_2d, total_3d = sub["n_2d"].sum(), sub["n_3d"].sum()
        bs = branch_summary(sub, total_2d, total_3d)
        bs.insert(0, "mouse", mouse)
        summaries.append(bs)

        wk_sub = wk[wk.mouse == mouse]
        wm = week_matrix(wk_sub, sub["branch"].unique())
        plot_batch(bs, wm, f"{mouse}  branch characterization (w8+w9+w10 pooled)",
                   OUT / f"{mouse}_branches.jpeg")

        # Q1: does occ3d differ by branch? eta^2 + Kruskal-Wallis
        groups = sub["branch"].to_numpy()
        occ_eta = eta_squared(sub["occ3d"], groups)
        tba_eta = eta_squared(sub["TotAccelBA"], groups)
        parts = [sub["occ3d"][groups == g] for g in np.unique(groups)]
        kw_p = kruskal(*parts).pvalue if all(len(p) for p in parts) else np.nan
        # Q3: branch x week independence
        wm_ct = wm.to_numpy()
        wm_ct = wm_ct[:, wm_ct.sum(0) > 0]
        chi2, chi_p, _, _ = chi2_contingency(wm_ct)
        n_tot = wm_ct.sum()
        cramers_v = np.sqrt(chi2 / (n_tot * (min(wm_ct.shape) - 1)))
        stats.append({
            "mouse": mouse, "n_clusters": len(sub),
            "occ3d_eta2": occ_eta, "occ3d_kw_p": kw_p, "tba_eta2": tba_eta,
            "week_cramers_v": cramers_v, "week_chi2_p": chi_p,
            "n_2d": arena_verdict_count(bs, "2D"),
            "n_3d": arena_verdict_count(bs, "3D"),
            "n_mixed": arena_verdict_count(bs, "mixed"),
        })

    allbs = pd.concat(summaries, ignore_index=True)
    allbs.to_csv(OUT / "branch_summary.csv", index=False)
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "mouse_stats.csv", index=False)
    plot_overview(st, OUT / "overview.jpeg")
    report(st, allbs)


def arena_verdict_count(bs, verdict):
    return int((bs["arena_verdict"] == verdict).sum())


def plot_overview(st, path):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4.6))
    lab = st["mouse"]
    x = np.arange(len(st))
    a0.bar(x - 0.2, st["tba_eta2"], 0.4, label="TBA (merge feature, ceiling)",
           color="#bbbbbb")
    a0.bar(x + 0.2, st["occ3d_eta2"], 0.4, label="occ3d (arena, independent)",
           color="#d62728")
    a0.set(xticks=x, ylabel="eta^2 (variance explained by branch)",
           title="How much does the kinematic branch determine the feature?")
    a0.set_xticklabels(lab, rotation=90, fontsize=7)
    a0.legend(fontsize=8)

    bottom = np.zeros(len(st))
    for col, c in (("n_2d", "#d62728"), ("n_mixed", "#999999"), ("n_3d", "#1f77b4")):
        a1.bar(x, st[col], bottom=bottom, color=c, label=col.replace("n_", ""))
        bottom += st[col].to_numpy()
    a1.set(xticks=x, ylabel="# branches", title="Branch arena verdicts per mouse")
    a1.set_xticklabels(lab, rotation=90, fontsize=7)
    a1.legend(fontsize=8, title="verdict")
    fig.tight_layout()
    save_figure(fig, path, dpi=140)
    plt.close(fig)


def report(st, allbs):
    print("\n=== per-mouse stats (w8+w9+w10 pooled) ===")
    print(st.round(3).to_string(index=False))
    print("\n=== Q1: do clusters in a branch share arena preference? ===")
    print(f"  occ3d eta^2 (branch explains cluster occ3d): "
          f"median={st['occ3d_eta2'].median():.3f}, "
          f"range {st['occ3d_eta2'].min():.3f}-{st['occ3d_eta2'].max():.3f}")
    print(f"  vs TBA eta^2 (merge-feature ceiling): median={st['tba_eta2'].median():.3f}")
    sig = (st["occ3d_kw_p"] < 0.05).sum()
    print(f"  Kruskal-Wallis occ3d~branch significant (p<0.05) in {sig}/{len(st)} mice")
    print("\n=== Q2: primarily-2D / 3D branches ===")
    print(f"  branches by verdict: 2D={allbs.eval('arena_verdict==\"2D\"').sum()}, "
          f"mixed={allbs.eval('arena_verdict==\"mixed\"').sum()}, "
          f"3D={allbs.eval('arena_verdict==\"3D\"').sum()} "
          f"(of {len(allbs)} branches)")
    print("\n=== Q3: distinct week profiles per branch ===")
    sigw = (st["week_chi2_p"] < 0.05).sum()
    print(f"  branch x week chi-square significant in {sigw}/{len(st)} mice; "
          f"Cramer's V median={st['week_cramers_v'].median():.3f}")
    print(f"\nWrote {OUT}/branch_summary.csv, mouse_stats.csv, overview.jpeg, "
          "and per-mouse panels")


if __name__ == "__main__":
    main()
