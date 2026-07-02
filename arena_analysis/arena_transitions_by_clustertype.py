"""Fork 2 variant: transition analysis on cluster-TYPE subsets (median 3D-occupancy
split), comparing how the primarily-2D vs primarily-3D repertoire's successor
structure changes over disease.

Clusters are split by their 3D occupancy (occ3d from the exclusivity analysis):
    primarily-2D   occ3d <  --split (default 0.5)
    primarily-3D   occ3d >= --split
The split is complementary, so each group is ~half the codebook (more power than
the earlier narrow bands).

Two arena treatments (run both to compare):
    --arena-mode home   ignore the opposite arena -- primarily-2D analysed only
                        in 2D frames, primarily-3D only in 3D frames.
    --arena-mode all    take every member cluster's frames regardless of arena
                        (2D+3D of the same week pooled by relabelling to the week
                        number, so fanout_by_week doesn't drop one arena).

Frames outside the subset are removed: with --link bridge (default) the surviving
member frames within a recording are linked across the gaps (sub-repertoire
transition structure); --link adjacent counts only temporally adjacent member
transitions. Segment/week boundaries always break the sequence. Then the repo's
fanout_by_week / rarefied_diversity run on each subset.

Run:
    uv run python arena_analysis/arena_transitions_by_clustertype.py --arena-mode home
    uv run python arena_analysis/arena_transitions_by_clustertype.py --arena-mode all
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from cluster_transition_compare import fanout_by_week           # noqa: E402
from cluster_successor_diversity import rarefied_diversity      # noqa: E402

from cluster_arena_exclusivity import (BATCHES, MICE, mat_csv,  # noqa: E402
                                       parse_segment)

OUT = ROOT / "output" / "transitions_by_clustertype"
# subset -> (home arena, plot colour, plot style)
SUBSETS = {"primarily-2D": ("2D", "#d62728", "--"),
           "primarily-3D": ("3D", "#1f77b4", "-")}


def load_full(mouse_dir, batch):
    csv = mat_csv(mouse_dir, batch)
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    seg = df["Folder_Name"].map(parse_segment)
    df["_week"] = [s[0] for s in seg]
    df["_arena"] = [s[1] for s in seg]
    return df


def subset_df(full, home_arena, members, link, arena_mode):
    """Restrict frames to the subset; mode controls the arena filter, link controls
    whether removed bins break the sequence (adjacent) or are bridged over."""
    if arena_mode == "home":
        keep = (full["_arena"] == home_arena) & full["ClusterIdx"].isin(members)
    else:                                   # all: any arena, member cluster
        keep = full["ClusterIdx"].isin(members)

    if link == "bridge":
        df = full[keep].reset_index(drop=True)
    else:
        df = full.copy()
        df.loc[~keep, "Folder_Name"] = np.nan
        df = df.reset_index(drop=True)

    if arena_mode == "all":
        # collapse the two arenas of a week into one label so they pool by week
        m = df["Folder_Name"].notna()
        df.loc[m, "Folder_Name"] = df.loc[m, "_week"].map(lambda w: f"week{int(w)}")
    return df


def spear(d):
    if len(d) < 3:
        return np.nan, np.nan
    w = sorted(d)
    y = [d[k] for k in w]
    if np.std(y) == 0:
        return np.nan, np.nan
    return spearmanr(w, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--membership", choices=["split", "all"], default="split",
                    help="'split': group clusters by occ3d (default). 'all': both "
                         "groups are ALL clusters, so the only differentiator is the "
                         "home arena -> all 2D bins vs all 3D bins (forces home mode).")
    ap.add_argument("--split", type=float, default=0.5,
                    help="occ3d < split -> primarily-2D, >= split -> primarily-3D")
    ap.add_argument("--arena-mode", choices=["home", "all"], default="home")
    ap.add_argument("--link", choices=["bridge", "adjacent"], default="bridge")
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    if args.membership == "all":
        args.arena_mode = "home"          # arena is the only differentiator
    out_dir = OUT / ("all_bins" if args.membership == "all" else args.arena_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    # legend/console labels: cluster-type groups, or plain arena groups
    labels = ({"primarily-2D": "all 2D bins", "primarily-3D": "all 3D bins"}
              if args.membership == "all"
              else {s: s for s in SUBSETS})

    verd = pd.read_csv(ROOT / "output" / "exclusivity" / "cluster_verdicts.csv")
    fan_rows, div_rows, summary = [], [], []
    for mouse_dir, mouse in MICE.items():
        for batch in BATCHES:
            full = load_full(mouse_dir, batch)
            if full is None:
                print(f"  {mouse}/{batch}: no MATLAB output, skipped")
                continue
            if args.membership == "all":
                allc = set(full["ClusterIdx"].unique())
                members = {"primarily-2D": allc, "primarily-3D": allc}
            else:
                occ = verd[(verd.mouse == mouse) & (verd.batch == batch)] \
                    .set_index("cluster")["occ3d"]
                members = {"primarily-2D": set(occ[occ < args.split].index),
                           "primarily-3D": set(occ[occ >= args.split].index)}

            fan_sub, rich_sub = {}, {}
            for sub, (home, _, _) in SUBSETS.items():
                df = subset_df(full, home, members[sub], args.link, args.arena_mode)
                fan, _ = fanout_by_week(df)
                res, frac = rarefied_diversity(df, args.depth, args.reps)
                rich = res.groupby("wn")["richness"].median().to_dict() if not res.empty else {}
                fan_sub[sub], rich_sub[sub] = fan, rich
                for wn, v in fan.items():
                    fan_rows.append({"mouse": mouse, "batch": batch, "subset": sub,
                                     "week": wn, "mean_successors": v})
                for wn, g in (res.groupby("wn") if not res.empty else []):
                    div_rows.append({"mouse": mouse, "batch": batch, "subset": sub,
                                     "week": int(wn), "median_richness": g["richness"].median(),
                                     "n_sources": len(g), "kept_frac": round(frac, 2)})
                fr, fp = spear(fan)
                rr, rp = spear(rich)
                summary.append({"mouse": mouse, "batch": batch, "subset": sub,
                                "n_member_clusters": len(members[sub]),
                                "kept_frac": round(frac, 2),
                                "fanout_rho": fr, "fanout_p": fp,
                                "richness_rho": rr, "richness_p": rp})
            tag = "all-bins-by-arena" if args.membership == "all" \
                else f"split={args.split}, arena-mode={args.arena_mode}"
            plot_dataset(fan_sub, rich_sub, labels, f"{mouse} {batch}  ({tag})",
                         out_dir / f"{mouse}_{batch}_clustertype_transitions.png")
            print(f"  {mouse}/{batch}: "
                  + ", ".join(f"{labels[s]}={len(members[s])}cl" for s in SUBSETS))

    pd.DataFrame(fan_rows).to_csv(out_dir / "fanout_by_clustertype.csv", index=False)
    pd.DataFrame(div_rows).to_csv(out_dir / "diversity_by_clustertype.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(out_dir / "transition_trends_by_clustertype.csv", index=False)
    print(f"\nWrote {out_dir}/")
    print(f"\n=== disease-week trend (Spearman rho), arena-mode={args.arena_mode} ===")
    print(summ.round(3).to_string(index=False))


def plot_dataset(fan_sub, rich_sub, labels, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for sub, (_, color, ls) in SUBSETS.items():
        f, r = fan_sub.get(sub, {}), rich_sub.get(sub, {})
        if f:
            axes[0].plot(sorted(f), [f[w] for w in sorted(f)], ls, marker="o",
                         color=color, label=labels[sub])
        if r:
            axes[1].plot(sorted(r), [r[w] for w in sorted(r)], ls, marker="o",
                         color=color, label=labels[sub])
    axes[0].set(xlabel="disease week", ylabel="mean distinct successors / source",
                title="Fan-out by week")
    axes[1].set(xlabel="disease week", ylabel="median rarefied richness",
                title="Successor richness by week (rarefied)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(title="cluster subset")
    fig.suptitle(title, y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
