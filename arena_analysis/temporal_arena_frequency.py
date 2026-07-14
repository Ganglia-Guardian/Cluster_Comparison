"""Weekly 2D-vs-3D frequency of each temporal cluster-class, per mouse.

Idea: the degeneracy pipeline already classifies every arena_compare cluster by
WHEN it appears over the batch's weeks (early / mid / late / sustained /
uncategorized -- see degeneracy_analysis/temporal_classify.py, written to each
batch's degeneracy_out/<batch>/temporal_classes.csv). Here we ask, for each
temporal class, how its share of behaviour time splits between the flat (2D) and
3D arenas and whether that share trends over disease weeks.

Per (mouse, week, arena) the class value is the POOLED share of behaviour time:

    freq(class, week, arena) = sum_{c in class} n(c, week, arena)
                               ---------------------------------------
                                     N(week, arena)   (all frames)

i.e. what fraction of that week+arena's frames belong to the class. Normalising
within (week, arena) makes 2D and 3D comparable even when the two arenas log
unequal total time. Clusters below the presence MIN_COUNT never enter
temporal_classes.csv, so they sit in the denominator but no class -- the per-week
class shares therefore sum to <= 1 (the remainder is unclassified low-count time).

Weeks are partitioned across the w8/w9/w10 batches (each a separate stitched
clustering: w8={8,11,14,17,20,23}, w9={9,12,15,18,21}, w10={10,13,16,19,24}), so
every natural week is defined by exactly one batch and the class labels for that
week come from that batch's temporal_classes.csv.

Output (per mouse): a grid of double-bar plots, one panel per temporal class,
each week showing a blue 2D bar and a red 3D bar, with a Spearman week-vs-freq
test annotated for the 2D and the 3D trend separately. Plus two CSVs -- the long
per-(mouse,class,week,arena) frequency table and the Spearman trend summary.

Run:
    C:/ProgramData/anaconda3/python.exe arena_analysis/temporal_arena_frequency.py
    ... --data-root E:/arena_analysis --label-col combined_label
"""
import argparse
import re
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
from utils import save_figure                                # noqa: E402

from cluster_arena_exclusivity import parse_segment          # noqa: E402

DEFAULT_DATA_ROOT = Path("E:/arena_analysis")
OUT = ROOT / "output" / "temporal_arena_frequency"

# Temporal classes, in the canonical section order used across the degeneracy code.
CATEGORIES = ["early", "mid", "late", "sustained", "uncategorized"]
# User-specified arena colours for THIS plot: 2D blue, 3D red. (Note this is the
# opposite of cluster_arena_exclusivity, where 3D is blue -- kept deliberately.)
ARENA_COLORS = {"2D": "#1f77b4", "3D": "#d62728"}
ARENAS = ["2D", "3D"]

_ARENA_LABEL = re.compile(r"(\d+(?:mp|lc)[a-z0-9_]*)_arena_compare", re.I)
_BATCH = re.compile(r"arena_compare_(w\d+)", re.I)


def discover(data_root):
    """mouse-label -> [(batch_tag, detail_csv, temporal_csv), ...] in week order.

    Walks the E:/arena_analysis layout: <...>_arena_compare/ holds the stitched
    batch dirs (arena_compare_w8[_mouse]/) with Cluster_detail_results.csv, and a
    parallel degeneracy_out/<same batch dir>/temporal_classes.csv. Batches missing
    either file (e.g. an empty w10) are skipped."""
    out = {}
    root = Path(data_root)
    if not root.is_dir():
        return out
    for mdir in sorted(root.iterdir()):
        m = _ARENA_LABEL.search(mdir.name)
        if not (mdir.is_dir() and m):
            continue
        mouse = m.group(1)
        degen = mdir / "degeneracy_out"
        batches = []
        for bdir in sorted(mdir.iterdir()):
            bm = _BATCH.search(bdir.name)
            if not (bdir.is_dir() and bm):
                continue
            detail = bdir / "Cluster_detail_results.csv"
            temporal = degen / bdir.name / "temporal_classes.csv"
            if detail.is_file() and temporal.is_file():
                batches.append((bm.group(1), detail, temporal))
        if batches:
            out[mouse] = sorted(batches, key=lambda b: int(b[0][1:]))
    return out


def batch_frequencies(detail_csv, temporal_csv, label_col):
    """Long rows {category, week, arena, freq} for one stitched batch.

    freq = pooled share of the (week, arena) frames belonging to the class."""
    det = pd.read_csv(detail_csv)
    wk, ar = zip(*(parse_segment(s) for s in det["Folder_Name"]))
    det = det.assign(week=wk, arena=ar).dropna(subset=["arena"])

    tmp = pd.read_csv(temporal_csv)
    cat = tmp.set_index("cluster")[label_col]
    det["category"] = det["ClusterIdx"].map(cat)      # NaN for low-count clusters

    rows = []
    for (week, arena), grp in det.groupby(["week", "arena"]):
        total = len(grp)                              # all frames incl. unclassified
        if total == 0:
            continue
        by_cat = grp["category"].value_counts()
        for c in CATEGORIES:
            rows.append({"category": c, "week": int(week), "arena": arena,
                         "freq": by_cat.get(c, 0) / total})
    return rows


def spear(weeks, vals):
    """Spearman(week, freq) guarding tiny / constant series -> (rho, p)."""
    if len(weeks) < 3 or np.std(vals) == 0:
        return np.nan, np.nan
    rho, p = spearmanr(weeks, vals)
    return rho, p


def plot_mouse(mouse, df, path):
    """Grid of double-bar panels (one per temporal class) for one mouse."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    trend_rows = []
    for ax, cat in zip(axes, CATEGORIES):
        sub = df[df["category"] == cat]
        weeks = sorted(sub["week"].unique())
        x = np.arange(len(weeks))
        w = 0.4
        series = {}
        for arena in ARENAS:
            a = sub[sub["arena"] == arena].set_index("week")["freq"]
            series[arena] = [float(a.get(wk, 0.0)) for wk in weeks]
        ax.bar(x - w / 2, series["2D"], w, color=ARENA_COLORS["2D"], label="2D")
        ax.bar(x + w / 2, series["3D"], w, color=ARENA_COLORS["3D"], label="3D")

        r2, p2 = spear(weeks, series["2D"])
        r3, p3 = spear(weeks, series["3D"])
        trend_rows.append({"mouse": mouse, "category": cat, "n_weeks": len(weeks),
                           "rho_2D": r2, "p_2D": p2, "rho_3D": r3, "p_3D": p3})
        ax.set_title(f"{cat}\n2D ρ={r2:.2f} p={p2:.3f}   "
                     f"3D ρ={r3:.2f} p={p3:.3f}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(weeks)
        ax.set_xlabel("disease week")
        ax.set_ylabel("share of week+arena time")
        ax.legend(fontsize=8, title="arena")
    for ax in axes[len(CATEGORIES):]:
        ax.set_visible(False)
    grp = "control" if mouse.endswith("lc") else "MitoPark"
    fig.suptitle(f"{mouse} ({grp}): temporal-class frequency by week, 2D vs 3D",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    save_figure(fig, path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return trend_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help=f"root holding the <mouse>_arena_compare dirs "
                         f"(default {DEFAULT_DATA_ROOT})")
    ap.add_argument("--label-col", choices=["label", "combined_label"],
                    default="label",
                    help="temporal_classes.csv column defining the subsets "
                         "('label' = individual class, keeps uncategorized)")
    args = ap.parse_args()

    mice = discover(args.data_root)
    if not mice:
        raise SystemExit(f"no *_arena_compare mice found under {args.data_root}")
    OUT.mkdir(parents=True, exist_ok=True)

    freq_rows, trend_rows = [], []
    for mouse, batches in mice.items():
        rows = []
        for tag, detail, temporal in batches:
            for r in batch_frequencies(detail, temporal, args.label_col):
                r["mouse"], r["batch"] = mouse, tag
                rows.append(r)
        df = pd.DataFrame(rows)
        freq_rows.extend(rows)
        trend_rows.extend(
            plot_mouse(mouse, df, OUT / f"{mouse}_temporal_arena_frequency.jpeg"))
        print(f"  {mouse}: {len(batches)} batches, "
              f"weeks {sorted(df['week'].unique())}")

    pd.DataFrame(freq_rows)[["mouse", "batch", "category", "week", "arena", "freq"]] \
        .to_csv(OUT / "temporal_arena_frequency.csv", index=False)
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(OUT / "temporal_arena_trends.csv", index=False)

    print(f"\nWrote {OUT}/ ({args.label_col})")
    print("\n=== Spearman week-vs-frequency (per mouse x class) ===")
    print(trends.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
