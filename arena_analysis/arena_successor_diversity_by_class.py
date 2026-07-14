"""Rarefied successor-diversity (cluster_successor_diversity.py) run on each
temporal-class x arena subset, per mouse.

For every mouse we take the temporal classes assigned in each batch's
degeneracy_out/<batch>/temporal_classes.csv (early / mid / late / sustained /
uncategorized) and, separately for the flat (2D) and 3D arena, ask how the
per-cluster successor diversity of that subset evolves over disease weeks.

Two things force a PER-BATCH run that is then stitched back together:
  * cluster ids are only meaningful within one stitched batch, and
  * build_transitions must not join frames across a Folder_Name (week+arena)
    boundary.
So for each (mouse, arena, class) we filter each batch's frames to that arena and
that batch's member clusters (bridging over the removed frames -- the
sub-repertoire's transition structure), run rarefied_diversity on the batch, and
concatenate the per-(week, source) results. Because the w8/w9/w10 batches cover
disjoint weeks, the concatenation is a clean weeks-8..24 timeline; source ids may
repeat across batches but only the per-week richness DISTRIBUTION is used, so that
is harmless.

Filtering to one arena keeps 2D (weekN_O) and 3D (weekN) as distinct weeks, so no
transition ever crosses arenas and the 2D vs 3D comparison stays clean.

Outputs (arena_analysis/output/successor_diversity_by_class/<mouse>/):
    <arena>_<class>_successor_diversity.csv        week, source, richness, perplexity
    <arena>_<class>_richness_ridgeline.jpeg        per-week rarefied-richness distribution
    <arena>_<class>_perplexity_ridgeline.jpeg      per-week effective-successors distribution
    <arena>_class_overlay_richness.jpeg            all 5 classes' median richness vs week
And a top-level trend summary: class_arena_richness_trends.csv (Spearman per subset).

Run:
    C:/ProgramData/anaconda3/python.exe arena_analysis/arena_successor_diversity_by_class.py
    ... --depth 15 --reps 200 --label-col combined_label --data-root E:/arena_analysis
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from cluster_arena_exclusivity import parse_segment                        # noqa: E402
from temporal_arena_frequency import discover, CATEGORIES                  # noqa: E402
from cluster_successor_diversity import (rarefied_diversity, plot_ridgeline,  # noqa: E402
                                         boot_median_ci, DEPTH, REPS)
from utils import save_figure                                             # noqa: E402

OUT = ROOT / "output" / "successor_diversity_by_class"
ARENAS = ["2D", "3D"]
# temporal-class colours matching degeneracy_analysis/temporal_classify.py
CAT_COLORS = {"early": "#d62728", "mid": "#ff7f0e", "late": "#1f77b4",
              "sustained": "#2ca02c", "uncategorized": "#bbbbbb"}


def subset_res(batches, category, arena, label_col, depth, reps):
    """Concatenate rarefied_diversity over a mouse's batches for one
    (class, arena). Each batch is filtered to that arena and that batch's member
    clusters, then bridged (removed frames close the gap)."""
    parts = []
    for tag, detail, temporal in batches:
        det = pd.read_csv(detail)
        arenas = [parse_segment(s)[1] for s in det["Folder_Name"]]
        det = det.assign(_arena=arenas).dropna(subset=["_arena"])
        cats = pd.read_csv(temporal).set_index("cluster")[label_col]
        members = set(cats[cats == category].index)
        sub = det[(det["_arena"] == arena) & det["ClusterIdx"].isin(members)] \
            .reset_index(drop=True)                       # bridge link
        if sub.empty:
            continue
        res, _ = rarefied_diversity(sub, depth, reps)
        if not res.empty:
            res = res.assign(batch=tag)
            parts.append(res)
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(columns=["week", "source", "richness", "perplexity", "wn"])


def trend(res):
    """Spearman(week, median-richness) over the subset's weeks -> (rho, p)."""
    if res.empty:
        return np.nan, np.nan
    med = res.groupby("wn")["richness"].median()
    if len(med) < 3 or med.std() == 0:
        return np.nan, np.nan
    return spearmanr(med.index.to_numpy(), med.to_numpy())


def plot_overlay(mouse, arena, cat_res, depth, path):
    """All temporal classes' median rarefied richness vs week on one axis, with a
    cluster-bootstrap 95% CI band and a Spearman rho per class in the legend."""
    rng = np.random.default_rng(0)
    grp = "control" if mouse.endswith("lc") else "MitoPark"
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    drew = False
    for cat in CATEGORIES:
        res = cat_res.get(cat)
        if res is None or res.empty:
            continue
        weeks = sorted(res["wn"].unique())
        med, lo, hi = [], [], []
        for w in weeks:
            vals = res.loc[res.wn == w, "richness"].to_numpy()
            med.append(np.median(vals))
            l, h = boot_median_ci(vals, rng)
            lo.append(l); hi.append(h)
        med, lo, hi = np.array(med), np.array(lo), np.array(hi)
        ok = ~np.isnan(med)
        if ok.sum() < 2:
            continue
        rho, p = spearmanr(np.array(weeks)[ok], med[ok]) if ok.sum() > 2 else (np.nan, np.nan)
        c = CAT_COLORS[cat]
        ax.fill_between(weeks, lo, hi, color=c, alpha=0.12, lw=0)
        ax.plot(weeks, med, marker="o", ms=4, color=c,
                label=f"{cat}: rho={rho:.2f}, p={p:.3f}")
        drew = True
    ax.set(xlabel="disease week", ylabel="median rarefied successor richness",
           title=f"{mouse} ({grp}) — {arena}: successor richness by temporal class\n"
                 f"(rarefied to {depth}; shaded = bootstrap 95% CI)")
    if drew:
        ax.legend(fontsize=8, title="temporal class", loc="best")
    fig.tight_layout()
    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("E:/arena_analysis"),
                    help="root holding the <mouse>_arena_compare dirs")
    ap.add_argument("--label-col", choices=["label", "combined_label"],
                    default="label", help="temporal_classes.csv column for subsets")
    ap.add_argument("--depth", type=int, default=DEPTH,
                    help="rarefaction depth (out-transitions per cluster)")
    ap.add_argument("--reps", type=int, default=REPS,
                    help="subsample draws averaged per cluster (perplexity)")
    args = ap.parse_args()

    mice = discover(args.data_root)
    if not mice:
        raise SystemExit(f"no *_arena_compare mice found under {args.data_root}")

    trend_rows = []
    for mouse, batches in mice.items():
        mdir = OUT / mouse
        mdir.mkdir(parents=True, exist_ok=True)
        for arena in ARENAS:
            cat_res = {}
            for cat in CATEGORIES:
                res = subset_res(batches, cat, arena, args.label_col,
                                 args.depth, args.reps)
                cat_res[cat] = res
                rho, p = trend(res)
                trend_rows.append({"mouse": mouse, "arena": arena, "category": cat,
                                   "n_source_weeks": res["wn"].nunique(),
                                   "n_points": len(res),
                                   "richness_rho": rho, "richness_p": p})
                if res.empty:
                    continue
                res.to_csv(mdir / f"{arena}_{cat}_successor_diversity.csv", index=False)
                plot_ridgeline(res, "richness", f"{mouse} {arena} · {cat}",
                               args.depth, mdir / f"{arena}_{cat}_richness_ridgeline.jpeg")
                plot_ridgeline(res, "perplexity", f"{mouse} {arena} · {cat}",
                               args.depth, mdir / f"{arena}_{cat}_perplexity_ridgeline.jpeg")
            plot_overlay(mouse, arena, cat_res, args.depth,
                         mdir / f"{arena}_class_overlay_richness.jpeg")
            filled = [c for c in CATEGORIES if not cat_res[c].empty]
            print(f"  {mouse}/{arena}: subsets with data -> {filled}")

    summ = pd.DataFrame(trend_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT / "class_arena_richness_trends.csv", index=False)
    print(f"\nWrote {OUT}/ ({args.label_col}, depth={args.depth}, reps={args.reps})")
    print("\n=== median rarefied-richness trend (Spearman) per mouse × arena × class ===")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
