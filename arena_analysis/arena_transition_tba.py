"""Do behavioural transitions degenerate toward lower-TBA (hypoactive) states,
and does that flow differ by arena and intensify with disease?

For each arena's transitions (built exactly as in arena_transitions.py), we label
the source and target cluster with their CHARACTERISTIC TBA -- the static
per-(mouse,batch,arena,cluster) mean total body acceleration. The signed gap

    delta = TBA(target) - TBA(source)

is negative when a transition steps DOWN in vigour. Using a static per-cluster
TBA (not week-specific) isolates the STRUCTURAL question -- is the transition
graph re-routing toward low-vigour clusters -- from the within-cluster vigour
decline measured elsewhere ([[arena-tba-features]]).

Per (arena, week) we report the frequency-weighted mean delta and the "downhill
fraction" (share of transitions with delta < 0), then test whether they trend
with disease week (Spearman), 2D vs 3D.

Run:
    uv run python arena_analysis/arena_transition_tba.py
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
from cluster_transition_labels import build_transitions, week_sort_key   # noqa: E402

from arena_transitions import (ARENA_COLOR, ARENA_STYLE, ARENAS,           # noqa: E402
                               masked_df)
from cluster_arena_exclusivity import BATCHES, MICE                        # noqa: E402

OUT = ROOT / "output" / "transitions"


def arena_transition_deltas(mouse_dir, mouse, batch, arena, ctba):
    """Per-transition TBA gap for one arena, with disease week."""
    df = masked_df(mouse_dir, batch, arena)
    if df is None:
        return None
    trans = build_transitions(df)                 # week (Folder_Name), source, target
    tba = ctba.get((mouse, batch, arena), {})
    trans["tba_src"] = trans["source"].map(tba)
    trans["tba_tgt"] = trans["target"].map(tba)
    trans = trans.dropna(subset=["tba_src", "tba_tgt"])
    trans["delta"] = trans["tba_tgt"] - trans["tba_src"]
    trans["wn"] = trans["week"].map(week_sort_key).astype(int)
    return trans


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ff = pd.read_csv(ROOT / "frame_features.csv")
    # characteristic TBA per (mouse, batch, arena, cluster)
    ctba = (ff.groupby(["mouse", "batch", "arena", "cluster"])["TotAccelBA"].mean()
            .reset_index())
    ctba = {(m, b, a): grp.set_index("cluster")["TotAccelBA"].to_dict()
            for (m, b, a), grp in ctba.groupby(["mouse", "batch", "arena"])}

    week_rows, summary = [], []
    for mouse_dir, mouse in MICE.items():
        for batch in BATCHES:
            by_arena = {}
            for arena in ARENAS:
                trans = arena_transition_deltas(mouse_dir, mouse, batch, arena, ctba)
                if trans is None or trans.empty:
                    by_arena = {}
                    break
                wk = trans.groupby("wn").agg(
                    mean_delta=("delta", "mean"),
                    downhill_frac=("delta", lambda d: float((d < 0).mean())),
                    n=("delta", "size")).reset_index()
                by_arena[arena] = wk
                for r in wk.itertuples(index=False):
                    week_rows.append({"mouse": mouse, "batch": batch, "arena": arena,
                                      "week": r.wn, "mean_delta": r.mean_delta,
                                      "downhill_frac": r.downhill_frac, "n": r.n})
            if not by_arena:
                continue
            for arena, wk in by_arena.items():
                dr, dp = (spearmanr(wk["wn"], wk["mean_delta"]) if len(wk) >= 3
                          else (np.nan, np.nan))
                fr, fp = (spearmanr(wk["wn"], wk["downhill_frac"]) if len(wk) >= 3
                          else (np.nan, np.nan))
                summary.append({"mouse": mouse, "batch": batch, "arena": arena,
                                "mean_delta_overall": wk["mean_delta"].mean(),
                                "delta_rho": dr, "delta_p": dp,
                                "downhill_rho": fr, "downhill_p": fp})
            plot_dataset(by_arena, f"{mouse} {batch}",
                         OUT / f"{mouse}_{batch}_tba_flow.png")
            print(f"  {mouse}/{batch}: TBA-flow computed for both arenas")

    pd.DataFrame(week_rows).to_csv(OUT / "tba_flow_by_arena.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "tba_flow_trends.csv", index=False)
    print(f"\nWrote {OUT}/ (tba_flow_by_arena, tba_flow_trends + plots)")
    print("\n=== TBA-flow of transitions by arena "
          "(mean_delta<0 = transitions step down in vigour) ===")
    print(summ.round(4).to_string(index=False))


def plot_dataset(by_arena, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for arena, wk in by_arena.items():
        c, ls = ARENA_COLOR[arena], ARENA_STYLE[arena]
        axes[0].plot(wk["wn"], wk["mean_delta"], ls, marker="o", color=c, label=arena)
        axes[1].plot(wk["wn"], wk["downhill_frac"], ls, marker="o", color=c, label=arena)
    axes[0].axhline(0, color="k", lw=0.7)
    axes[1].axhline(0.5, color="k", lw=0.7)
    axes[0].set(xlabel="disease week", ylabel="mean TBA(target) - TBA(source)",
                title="Vigour step of transitions (neg = downhill)")
    axes[1].set(xlabel="disease week", ylabel="fraction of transitions stepping down",
                title="Downhill-transition fraction")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(title="arena  (2D dashed, 3D solid)")
    fig.suptitle(title, y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
