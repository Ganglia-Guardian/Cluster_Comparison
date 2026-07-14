"""Do the evolution candidates share a signature?

Pools every cluster across mice (from temporal_classes.csv, which now carries the
IMU func signature + resting flag), then asks what distinguishes the evolution
candidates (`changed` = individual temporal class differs from its feature-family's
pooled class) from the rest. Emphasis on the INTRINSIC IMU signature (TBA, gyro,
per-axis accel) because that is measurable early -- independent of the full time
course -- which is the whole point: a signature that flags a would-be evolver
before all weeks are collected.

Main comparison is within pooled MitoPark (that is where evolution lives; controls
are almost all sustained). Reports Mann-Whitney U with an AUC effect size (P[a
candidate's metric > a non-candidate's]; 0.5 = no difference) per metric, plus the
fraction flagged by individual temporal class and by resting/active.

Run (after temporal_classify.py):
    C:/ProgramData/anaconda3/python.exe degeneracy_analysis/evolution_commonalities.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_figure
from presence_similarity import MICE, OUT

METRICS = ["TBA", "gyro", "ap_accel", "dv_accel", "size", "self_dist",
           "centroid_week", "margin"]
BOOL = ["changed", "strong", "resting"]


def load_all():
    frames = []
    for m in MICE:
        p = f"{OUT}/{m}/temporal_classes.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        for c in BOOL:
            if d[c].dtype == object:
                d[c] = d[c].map({"True": True, "False": False})
        feat = np.load(f"{OUT}/{m}/feature.npz", allow_pickle=True)
        d["self_dist"] = feat["self_dist"]
        d["mouse"] = m
        d["group"] = "control" if m.endswith("lc") else "MitoPark"
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def compare(df, flag="changed", metrics=METRICS):
    a, b = df[df[flag]], df[~df[flag]]
    rows = []
    for mt in metrics:
        x, y = a[mt].dropna(), b[mt].dropna()
        if len(x) < 3 or len(y) < 3:
            continue
        U, p = mannwhitneyu(x, y, alternative="two-sided")
        rows.append({"metric": mt, "med_cand": x.median(), "med_other": y.median(),
                     "AUC": U / (len(x) * len(y)), "p": p})
    r = pd.DataFrame(rows)
    r["effect"] = (r.AUC - 0.5).abs()
    return r.sort_values("effect", ascending=False).reset_index(drop=True)


def _plot(df, res, flag, path):
    top = res.metric.head(4).tolist()
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax = ax.ravel()
    for j, mt in enumerate(top):
        cand = df[df[flag]][mt].dropna()
        other = df[~df[flag]][mt].dropna()
        ax[j].boxplot([other, cand], labels=["rest", "candidate"], showfliers=False)
        auc = res.loc[res.metric == mt, "AUC"].iloc[0]
        pv = res.loc[res.metric == mt, "p"].iloc[0]
        ax[j].set_title(f"{mt}   AUC={auc:.2f}  p={pv:.1e}", fontsize=10)

    # fraction flagged by individual temporal class
    order = ["early", "mid", "late", "sustained", "uncategorized"]
    frac = df.groupby("label")[flag].mean().reindex(order).fillna(0)
    ax[4].bar(range(len(order)), frac.values, color="#c33")
    ax[4].set_xticks(range(len(order)))
    ax[4].set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax[4].set(title=f"fraction {flag} by individual class", ylim=(0, 1))

    # fraction flagged: resting vs active
    fr = df.groupby("resting")[flag].mean().reindex([False, True]).fillna(0)
    ax[5].bar(["active", "resting"], fr.values, color=["#39c", "#11224a"])
    ax[5].set(title=f"fraction {flag}: resting vs active", ylim=(0, 1))

    fig.suptitle(f"Evolution-candidate signature (pooled MitoPark, flag={flag})",
                 fontsize=12)
    fig.tight_layout()
    save_figure(fig, path, dpi=130)
    plt.close(fig)


def main():
    df = load_all()
    df.to_csv(f"{OUT}/all_clusters.csv", index=False)
    mp = df[df.group == "MitoPark"].copy()

    print(f"\n{'='*70}\nPooled MitoPark: {len(mp)} clusters, "
          f"{int(mp.changed.sum())} evolution candidates, "
          f"{int(mp.resting.sum())} resting\n{'='*70}")

    res = compare(mp, "changed")
    print("\nWhat distinguishes evolution candidates (changed) -- ranked by effect:")
    print(res.round(3).to_string(index=False))

    print("\nFraction flagged `changed` by individual temporal class:")
    print(mp.groupby("label").changed.agg(["mean", "size"]).round(3).to_string())

    print("\nResting vs active -- fraction flagged, and fraction resting overall:")
    print(mp.groupby("resting").changed.mean().round(3).to_string())
    print(f"  MitoPark resting fraction: {mp.resting.mean():.3f}   "
          f"control resting fraction: {df[df.group=='control'].resting.mean():.3f}")

    _plot(mp, res, "changed", f"{OUT}/evolution_commonalities.jpeg")
    print(f"\nsaved {OUT}/evolution_commonalities.jpeg and {OUT}/all_clusters.csv")


if __name__ == "__main__":
    main()
