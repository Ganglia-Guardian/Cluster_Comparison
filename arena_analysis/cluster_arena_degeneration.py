"""Per-cluster 2D-vs-3D degeneration: does an individual cluster's frequency rise
or fall over disease weeks, and does that trend differ between the flat (2D) and
3D arenas?

Companion to temporal_arena_frequency.py (which pools clusters within a temporal
class). Here each cluster is treated on its own. Cluster ids are only meaningful
within one stitched batch, so the batches are UNSTITCHED: a cluster's Spearman
runs over just its own batch's weeks (w8={8,11,14,17,20,23}, w9={9,12,15,18,21},
w10={10,13,16,19,24}), separately for the 2D and the 3D series.

Per cluster c and arena a, the weekly value is the within-(week, arena) share

    freq(c, week, a) = n(c, week, a) / N(week, a)      (N = all frames that week+arena)

so 2D and 3D are comparable despite unequal logged time; weeks where the cluster
is absent count as 0 (that is the fade we want to capture). Spearman(week, freq)
gives rho_2D and rho_3D -- negative = the cluster degenerates over disease.

Plot (per mouse): one panel per temporal class (from temporal_classes.csv's
`label`). Within a panel the class's clusters -- pooled across batches, each a
distinct (batch, id) -- are ordered by rho_3D low->high, drawn as a red 3D bar
next to a blue 2D bar. Missing bar = flat/absent series (rho undefined).

Run:
    C:/ProgramData/anaconda3/python.exe arena_analysis/cluster_arena_degeneration.py
    ... --data-root E:/arena_analysis --label-col combined_label --min-frames 50
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from utils import save_figure                                          # noqa: E402

from cluster_arena_exclusivity import parse_segment                    # noqa: E402
from temporal_arena_frequency import (discover, spear, CATEGORIES,     # noqa: E402
                                      ARENA_COLORS)

OUT = ROOT / "output" / "cluster_arena_degeneration"
ARENAS = ["2D", "3D"]


def batch_cluster_trends(detail_csv, temporal_csv, label_col, min_frames):
    """One row per (labelled) cluster in a batch with its 2D & 3D week trends."""
    det = pd.read_csv(detail_csv)
    wk, ar = zip(*(parse_segment(s) for s in det["Folder_Name"]))
    det = det.assign(week=wk, arena=ar).dropna(subset=["arena"])
    weeks = sorted(det["week"].unique())

    Ntot = det.groupby(["arena", "week"]).size()          # (arena, week) -> frames
    cnt = det.groupby(["ClusterIdx", "arena", "week"]).size()

    cat = pd.read_csv(temporal_csv).set_index("cluster")[label_col]

    rows = []
    for c, category in cat.items():
        if c not in det["ClusterIdx"].values:
            continue                                       # labelled but no frames here
        rec = {"cluster": int(c), "category": category}
        n_total = 0
        for a in ARENAS:
            aw = [w for w in weeks if Ntot.get((a, w), 0) > 0]
            series = [cnt.get((c, a, w), 0) / Ntot[(a, w)] for w in aw]
            frames = int(sum(cnt.get((c, a, w), 0) for w in aw))
            n_total += frames
            rho, p = spear(aw, series)
            rec[f"rho_{a}"] = rho
            rec[f"p_{a}"] = p
            rec[f"frames_{a}"] = frames
        rec["n_frames"] = n_total
        rec["n_weeks"] = len(weeks)
        if n_total >= min_frames:
            rows.append(rec)
    return rows


def plot_mouse(mouse, df, path):
    """Stacked panels (one temporal class each); clusters sorted by rho_3D."""
    fig, axes = plt.subplots(len(CATEGORIES), 1,
                             figsize=(max(12, df["category"].value_counts().max()
                                          * 0.22 if not df.empty else 12),
                                      2.7 * len(CATEGORIES)))
    for ax, cat in zip(axes, CATEGORIES):
        sub = df[df["category"] == cat].sort_values(
            "rho_3D", ascending=True, na_position="last").reset_index(drop=True)
        if sub.empty:
            ax.set_title(f"{cat}  (no clusters)", fontsize=9)
            ax.set_ylim(-1, 1)
            continue
        x = np.arange(len(sub))
        w = 0.4
        ax.bar(x - w / 2, sub["rho_3D"], w, color=ARENA_COLORS["3D"], label="3D")
        ax.bar(x + w / 2, sub["rho_2D"], w, color=ARENA_COLORS["2D"], label="2D")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}:{c}" for b, c in zip(sub["batch"], sub["cluster"])],
                           rotation=90, fontsize=6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Spearman ρ\n(week vs freq)")
        ax.set_title(f"{cat}  (n={len(sub)} clusters, ordered by 3D ρ)", fontsize=9)
        ax.legend(fontsize=8, title="arena", loc="upper left")
    grp = "control" if mouse.endswith("lc") else "MitoPark"
    fig.suptitle(f"{mouse} ({grp}): per-cluster degeneration, 2D vs 3D "
                 f"(week-vs-frequency Spearman ρ)", y=1.0, fontsize=13)
    fig.tight_layout()
    save_figure(fig, path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("E:/arena_analysis"),
                    help="root holding the <mouse>_arena_compare dirs")
    ap.add_argument("--label-col", choices=["label", "combined_label"],
                    default="label",
                    help="temporal_classes.csv column defining the subsets")
    ap.add_argument("--min-frames", type=int, default=0,
                    help="drop clusters with fewer than this many total (2D+3D) "
                         "frames in their batch (noisy rho); default 0 = keep all")
    args = ap.parse_args()

    mice = discover(args.data_root)
    if not mice:
        raise SystemExit(f"no *_arena_compare mice found under {args.data_root}")
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for mouse, batches in mice.items():
        rows = []
        for tag, detail, temporal in batches:
            for r in batch_cluster_trends(detail, temporal, args.label_col,
                                          args.min_frames):
                r["mouse"], r["batch"] = mouse, tag
                rows.append(r)
        df = pd.DataFrame(rows)
        all_rows.extend(rows)
        plot_mouse(mouse, df, OUT / f"{mouse}_cluster_degeneration.jpeg")
        n_neg3d = int((df["rho_3D"] < 0).sum())
        print(f"  {mouse}: {len(df)} clusters "
              f"({n_neg3d} degenerating in 3D, rho_3D<0)")

    cols = ["mouse", "batch", "cluster", "category", "n_weeks", "n_frames",
            "frames_2D", "frames_3D", "rho_2D", "p_2D", "rho_3D", "p_3D"]
    pd.DataFrame(all_rows)[cols].to_csv(
        OUT / "cluster_arena_degeneration.csv", index=False)
    print(f"\nWrote {OUT}/ ({args.label_col}, min_frames={args.min_frames})")


if __name__ == "__main__":
    main()
