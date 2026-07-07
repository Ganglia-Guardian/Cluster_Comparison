"""Quick analysis of the low-TBA, 2D-leaning cluster group in the two controls
(1lc, 2lc). In the pooled per-mouse dendrograms these form a distinct branch
whose mean TBA (~0.06) is an order of magnitude below every other branch
(~0.25-0.39) -- a low-vigour, flat-arena repertoire.

For each control we take the "low-vigour branch" = the branch with the lowest
mean TBA, then characterize it:
  - size (clusters, % of behavior time) and arena lean (occ3d)
  - how it differs kinematically from the rest (all 4 func-features)
  - its frame-share trajectory across weeks 8-24 (controls have no disease, so is
    it a stable trait or does it drift?)

Outputs (output/lc_low_tba/):
    low_vigour_clusters.csv   the member clusters (mouse, batch, cluster, ...)
    lc_low_tba.png            2 rows (1lc, 2lc) x 3 panels: occ3d-vs-TBA scatter,
                              feature contrast, week trajectory

Run:  uv run python cluster_group_analysis/lc_low_tba_analysis.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT
from feature_extraction import FEATURE_NAMES

OUT = ROOT / "output" / "lc_low_tba"
MICE = ["1lc", "2lc"]
FEAT_SHORT = {"anterior_posterior_x_accel": "AP accel",
              "dorsal_ventral_y_accel": "DV accel", "y_gyro": "y-gyro",
              "TotAccelBA": "TBA"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    br = pd.read_csv(ROOT / "cluster_branches.csv")
    bs = pd.read_csv(ROOT / "output" / "branches" / "branch_summary.csv")
    wc = pd.read_csv(ROOT / "cluster_week_counts.csv")

    fig, axes = plt.subplots(len(MICE), 3, figsize=(15, 4.4 * len(MICE)))
    members, summary_lines = [], []
    for r, mouse in enumerate(MICE):
        sub = br[br.mouse == mouse].reset_index(drop=True)
        # low-vigour branch = lowest mean TBA
        lowb = int(bs[bs.mouse == mouse].sort_values("tba_mean")["branch"].iloc[0])
        grp = sub[sub.branch == lowb]
        rest = sub[sub.branch != lowb]
        members.append(grp.assign(low_vigour_branch=lowb))

        f_share = grp["n_frames"].sum() / sub["n_frames"].sum()
        summary_lines.append(
            f"{mouse}: low-vigour branch {lowb} = {len(grp)} clusters, "
            f"{f_share*100:.1f}% of time, mean occ3d={grp['occ3d'].mean():.2f} "
            f"(2D lean), mean TBA={grp['TotAccelBA'].mean():.3f} "
            f"vs rest {rest['TotAccelBA'].mean():.3f}")

        # -- panel A: occ3d vs TBA, group highlighted
        a = axes[r, 0]
        a.scatter(rest["occ3d"], rest["TotAccelBA"], s=rest["n_frames"] / 40,
                  c="#cccccc", edgecolor="none", label="other clusters")
        a.scatter(grp["occ3d"], grp["TotAccelBA"], s=grp["n_frames"] / 40,
                  c="#d62728", edgecolor="k", lw=0.3, label="low-vigour branch")
        a.axvline(0.5, color="k", lw=0.7, alpha=0.5)
        a.set(xlabel="occ3d (0=2D flat, 1=3D)", ylabel="mean TBA", xlim=(0, 1),
              title=f"{mouse}: occ3d vs TBA  (size ~ #frames)")
        a.legend(fontsize=8)

        # -- panel B: feature contrast, group vs rest (z-scored within mouse)
        b = axes[r, 1]
        z = (sub[FEATURE_NAMES] - sub[FEATURE_NAMES].mean()) / sub[FEATURE_NAMES].std()
        gmean = z[sub.branch == lowb].mean()
        rmean = z[sub.branch != lowb].mean()
        x = np.arange(len(FEATURE_NAMES))
        b.bar(x - 0.2, gmean, 0.4, color="#d62728", label="low-vigour branch")
        b.bar(x + 0.2, rmean, 0.4, color="#999999", label="rest")
        b.axhline(0, color="k", lw=0.7)
        b.set(xticks=x, ylabel="z-score (within mouse)",
              title=f"{mouse}: kinematic profile")
        b.set_xticklabels([FEAT_SHORT[f] for f in FEATURE_NAMES], rotation=20)
        b.legend(fontsize=8)

        # -- panel C: week trajectory of group frame-share
        c = axes[r, 2]
        gclu = set(zip(grp.batch, grp.cluster))
        wsub = wc[wc.mouse == mouse].copy()
        wsub["in_grp"] = [(bt, cl) in gclu for bt, cl in zip(wsub.batch, wsub.cluster)]
        per_week = wsub.groupby("week").apply(
            lambda d: d.loc[d.in_grp, "n"].sum() / d["n"].sum(), include_groups=False)
        c.plot(per_week.index, per_week.values, "-o", color="#d62728")
        c.set(xlabel="week", ylabel="group frame-share",
              title=f"{mouse}: low-vigour share over weeks", ylim=(0, None))
        c.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "lc_low_tba.png", dpi=140)
    plt.close(fig)

    mem = pd.concat(members, ignore_index=True)
    mem.to_csv(OUT / "low_vigour_clusters.csv", index=False)
    print("=== low-TBA / 2D-leaning group in controls ===")
    for line in summary_lines:
        print(" ", line)
    print("\n=== kinematic contrast (raw means, group vs rest) ===")
    for mouse in MICE:
        sub = br[br.mouse == mouse]
        lowb = int(bs[bs.mouse == mouse].sort_values("tba_mean")["branch"].iloc[0])
        g, rst = sub[sub.branch == lowb], sub[sub.branch != lowb]
        print(f"  {mouse}:")
        for f in FEATURE_NAMES:
            print(f"    {FEAT_SHORT[f]:9s}: group={g[f].mean():+.3f}  rest={rst[f].mean():+.3f}")
    print(f"\nWrote {OUT}/low_vigour_clusters.csv and lc_low_tba.png")


if __name__ == "__main__":
    main()
