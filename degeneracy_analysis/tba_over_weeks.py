"""Week-by-week distribution of per-window total body acceleration (TBA), one
ridgeline per mouse (style mirrors cluster_successor_diversity.py: one KDE curve
per week, stacked earliest-at-top, time-coloured, tick = weekly median).

Each 60-sample window has one linear TBA value (feature_extraction.py) and a week;
we plot the full per-window TBA distribution for every natural week, so a shift of
mass toward the resting line over weeks (MitoPark hypokinesia) is visible directly.

For the MitoPark mice we also split by INDIVIDUAL temporal class (before merging,
the `label` column of temporal_classes.csv): a plot from the transient early+mid+
late clusters, one from sustained-only, one from uncategorized-only. These land in
out/<mouse>/tba_by_category/. Every plot uses a fixed x-range (XLIM = 0.0-0.6) so
all mice and subsets compare directly.

Run (needs the session_*.mat + temporal_classes.csv for the category split):
    C:/ProgramData/anaconda3/python.exe degeneracy_analysis/tba_over_weeks.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extraction import load_funct_features, bin_features, FEATURE_NAMES
from presence_similarity import MICE, DATA, OUT, natural_week, MIN_WEEK_FRAC

RESTING_TBA = 0.10
XLIM = (0.0, 0.6)    # fixed x-range for every plot, all mice, for comparability


def window_data(mouse):
    """(week, TBA, cluster) per window over natural weeks only."""
    det = pd.read_csv(f"{DATA}/{mouse}/Cluster_detail_results.csv")
    wk = det.Folder_Name.map(natural_week).to_numpy()
    cidx = det.ClusterIdx.to_numpy()
    binned = bin_features(load_funct_features(f"{DATA}/{mouse}/session_1_out.mat"))
    tba = binned[FEATURE_NAMES.index("TotAccelBA")]
    if len(tba) != len(det):
        raise ValueError(f"{mouse}: TBA windows {len(tba)} != detail rows {len(det)}")
    m = ~pd.isna(wk)
    return wk[m].astype(int), tba[m], cidx[m]


def plot(mouse, wk, tba, out_path, subtitle=""):
    if len(tba) < 10:
        print(f"    {subtitle or 'all'}: only {len(tba)} windows; skipped")
        return None
    uw, cnt = np.unique(wk, return_counts=True)
    weeks = uw[cnt >= MIN_WEEK_FRAC * np.median(cnt)]      # drop failed/partial weeks
    if len(weeks) < 2:
        print(f"    {subtitle or 'all'}: <2 usable weeks; skipped")
        return None
    grid = np.linspace(*XLIM, 200)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(weeks) + 2))
    meds = []
    for i, w in enumerate(weeks):
        v = tba[wk == w]
        base = (len(weeks) - 1 - i) * 1.0                  # earliest week at top
        color = cmap(i / max(1, len(weeks) - 1))
        if len(v) >= 3 and np.std(v) > 0:
            dens = gaussian_kde(v)(grid)
            dens = dens / dens.max() * 0.9
            ax.fill_between(grid, base, base + dens, color=color, alpha=0.8,
                            lw=0.5, edgecolor="white")
        meds.append(np.median(v))
        ax.plot([np.median(v)], [base], "|", color="black", ms=8, mew=1.2)
        ax.text(grid[0], base + 0.05, f"w{w}", fontsize=7, va="bottom")

    ax.axvline(RESTING_TBA, ls=":", color="red", lw=1.2)
    ax.text(RESTING_TBA, len(weeks) - 0.4, " resting", color="red", fontsize=7, va="top")
    rho, p = spearmanr(weeks, meds)
    grp = "control" if mouse.endswith("lc") else "MitoPark"
    sub = f"  [{subtitle}]" if subtitle else ""
    ax.set_yticks([])
    ax.set_xlim(*XLIM)
    ax.set_xlabel("total body acceleration (per 60-sample window)")
    ax.set_title(f"{mouse} ({grp}): TBA distribution by week{sub}\n"
                 f"weekly-median trend rho={rho:+.2f}, p={p:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rho, p


# individual temporal-class subsets (before merging), MitoPark only
CATEGORY_SUBSETS = {
    "early_mid_late": ["early", "mid", "late"],
    "sustained": ["sustained"],
    "uncategorized": ["uncategorized"],
}


def category_plots(mouse, wk, tba, cidx):
    """Split the windows by their cluster's INDIVIDUAL temporal label and plot each
    subset with a shared x-range, into out/<mouse>/tba_by_category/."""
    tc = pd.read_csv(f"{OUT}/{mouse}/temporal_classes.csv")
    lab = dict(zip(tc.cluster.astype(int), tc.label))
    wlabel = np.array([lab.get(int(c)) for c in cidx], dtype=object)
    folder = f"{OUT}/{mouse}/tba_by_category"
    os.makedirs(folder, exist_ok=True)
    for name, keep in CATEGORY_SUBSETS.items():
        sel = np.isin(wlabel, keep)
        plot(mouse, wk[sel], tba[sel], f"{folder}/{name}.png",
             subtitle=f"{name} clusters, {int(sel.sum())} windows")


def main():
    for m in MICE:
        if not os.path.exists(f"{DATA}/{m}/session_1_out.mat"):
            continue
        wk, tba, cidx = window_data(m)
        os.makedirs(f"{OUT}/{m}", exist_ok=True)
        rp = plot(m, wk, tba, f"{OUT}/{m}/tba_over_weeks.png")
        print(f"{m}: tba_over_weeks.png  rho={rp[0]:+.2f} p={rp[1]:.3f}"
              if rp else f"{m}: base plot skipped")
        if not m.endswith("lc") and os.path.exists(f"{OUT}/{m}/temporal_classes.csv"):
            category_plots(m, wk, tba, cidx)
            print(f"  wrote tba_by_category/ (early_mid_late, sustained, uncategorized)")


if __name__ == "__main__":
    main()
