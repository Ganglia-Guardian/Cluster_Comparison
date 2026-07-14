"""Fork 2: cluster-transition statistics, compared by arena.

The repo's transition analysis (cluster_transition_labels / _compare /
cluster_successor_diversity) runs on a frame table with columns ClusterIdx +
Folder_Name, forming transitions only between consecutive same-Folder_Name
frames. In the arena MATLAB data Folder_Name is the segment (weekN = 3D,
weekN_O = 2D), so a transition never crosses an arena -- we isolate one arena by
masking the OTHER arena's frames (and the existing NaN boundary frames stay NaN,
preserving sequence breaks), then reuse the existing functions verbatim.

Per dataset (mouse, batch) we compute, for each arena, over progression weeks:
  * fan-out      mean distinct successors per source cluster (fanout_by_week)
  * richness     rarefied distinct successors per source (rarefied_diversity)
  * perplexity   rarefied effective successors exp(H) per source

and report whether each rises/falls with disease week (Spearman), 2D vs 3D.
Within a dataset both arenas share one codebook (same clustering), so the raw
counts are directly comparable across arenas.

Run:
    uv run python arena_analysis/arena_transitions.py
    uv run python arena_analysis/arena_transitions.py --depth 15
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
from utils import save_figure                                   # noqa: E402

from cluster_arena_exclusivity import (BATCHES, MICE, mat_csv,  # noqa: E402
                                       parse_segment)

OUT = ROOT / "output" / "transitions"
ARENAS = ["2D", "3D"]
ARENA_STYLE = {"2D": "--", "3D": "-"}
ARENA_COLOR = {"2D": "#d62728", "3D": "#1f77b4"}


def masked_df(mouse_dir, batch, arena):
    """MATLAB frame table with only `arena`'s segments keeping their Folder_Name;
    all other rows masked to NaN so they act as sequence breaks."""
    csv = mat_csv(mouse_dir, batch)
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    ar = df["Folder_Name"].map(lambda s: parse_segment(s)[1])
    df = df.copy()
    df.loc[ar != arena, "Folder_Name"] = np.nan
    return df.reset_index(drop=True)


def spear(d):
    """Spearman rho/p of a {week: value} mapping vs week (nan if <3 points)."""
    if len(d) < 3:
        return np.nan, np.nan
    w = sorted(d)
    return spearmanr(w, [d[k] for k in w])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=20,
                    help="rarefaction depth (out-transitions per cluster)")
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    fan_rows, div_rows, summary = [], [], []
    for mouse_dir, mouse in MICE.items():
        for batch in BATCHES:
            fan_arena, rich_arena = {}, {}
            for arena in ARENAS:
                df = masked_df(mouse_dir, batch, arena)
                if df is None:
                    break
                fan, K = fanout_by_week(df)
                fan_arena[arena] = fan
                for wn, v in fan.items():
                    fan_rows.append({"mouse": mouse, "batch": batch, "arena": arena,
                                     "week": wn, "mean_successors": v, "K": K})

                res, frac = rarefied_diversity(df, args.depth, args.reps)
                rich = res.groupby("wn")["richness"].median().to_dict()
                rich_arena[arena] = rich
                for wn, g in res.groupby("wn"):
                    div_rows.append({"mouse": mouse, "batch": batch, "arena": arena,
                                     "week": int(wn), "median_richness": g["richness"].median(),
                                     "median_perplexity": g["perplexity"].median(),
                                     "n_sources": len(g), "kept_frac": round(frac, 2)})
            if not fan_arena:
                print(f"  {mouse}/{batch}: no MATLAB output, skipped")
                continue

            for arena in ARENAS:
                fr, fp = spear(fan_arena[arena])
                rr, rp = spear(rich_arena[arena])
                summary.append({"mouse": mouse, "batch": batch, "arena": arena,
                                "fanout_rho": fr, "fanout_p": fp,
                                "richness_rho": rr, "richness_p": rp})
            plot_dataset(fan_arena, rich_arena, f"{mouse} {batch}",
                         OUT / f"{mouse}_{batch}_transitions.jpeg")
            print(f"  {mouse}/{batch}: fan-out/diversity computed for both arenas")

    pd.DataFrame(fan_rows).to_csv(OUT / "fanout_by_arena.csv", index=False)
    pd.DataFrame(div_rows).to_csv(OUT / "diversity_by_arena.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "transition_trends_by_arena.csv", index=False)

    print(f"\nWrote {OUT}/ (fanout_by_arena, diversity_by_arena, "
          f"transition_trends_by_arena + plots)")
    print("\n=== disease-week trend (Spearman rho) of transition metrics, by arena ===")
    print(summ.round(3).to_string(index=False))


def plot_dataset(fan_arena, rich_arena, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for arena in ARENAS:
        c, ls = ARENA_COLOR[arena], ARENA_STYLE[arena]
        f = fan_arena[arena]
        axes[0].plot(sorted(f), [f[w] for w in sorted(f)], ls, marker="o",
                     color=c, label=arena)
        r = rich_arena[arena]
        axes[1].plot(sorted(r), [r[w] for w in sorted(r)], ls, marker="o",
                     color=c, label=arena)
    axes[0].set(xlabel="disease week", ylabel="mean distinct successors / source",
                title="Fan-out by week")
    axes[1].set(xlabel="disease week", ylabel="median rarefied richness",
                title="Successor richness by week (rarefied)")
    for ax in axes:
        ax.legend(title="arena  (2D dashed, 3D solid)")
    fig.suptitle(title, y=1.0)
    fig.tight_layout(); save_figure(fig, path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
