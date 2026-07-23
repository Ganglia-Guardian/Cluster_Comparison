"""Week-by-week distribution of per-window total body acceleration (TBA) for the
arena_compare data, one ridgeline per natural week, split by ARENA.

Same ridgeline style as degeneracy_analysis/tba_over_weeks.py (one KDE curve per
week, stacked earliest-at-top, weekly median tick, fixed x-range), with two
deliberate changes requested for the arena version:

  (1) The distributions DO NOT change colour over time -- every week for a mouse
      is drawn in that mouse's single fixed cohort colour (MitoPark green /
      control orange, per utils.cohort_colors). Time is read off the vertical
      stack order, not colour.
  (2) The analysis runs on the arena_compare tree (default E:/arena_compare) and
      each mouse gets TWO plots -- one from its 2D (flat) arena windows and one
      from its 3D arena windows. The MATLAB Folder_Name suffix (weekN vs weekN_o)
      carries the arena; parse_segment handles the per-mouse suffix variants.

Windows are pooled across a mouse's batches (w8/w9/w10 sample interleaved weeks),
so each mouse's plot spans the full disease-week range for that arena.

Run:
    C:/ProgramData/anaconda3/python.exe arena_analysis/arena_tba_by_week.py
    C:/ProgramData/anaconda3/python.exe arena_analysis/arena_tba_by_week.py --data E:/arena_compare
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from feature_extraction import combine_results          # noqa: E402
from utils import save_figure, cohort_colors            # noqa: E402
import cluster_arena_exclusivity as cae                 # noqa: E402

RESTING_TBA = 0.10
XLIM = (0.0, 0.6)        # fixed x-range for every plot, all mice/arenas, comparable
MIN_WEEK_N = 10          # a week needs at least this many windows to be drawn
ARENAS = ("2D", "3D")


def mouse_windows(mouse_dir, batches):
    """Pool (week, arena, TBA) per 60-sample window across all of a mouse's
    batches, arena/week taken from the MATLAB Folder_Name suffix."""
    parts = []
    for batch in batches:
        mat, csv = cae.session_mat(mouse_dir, batch), cae.mat_csv(mouse_dir, batch)
        if not (mat.exists() and csv.exists()):
            continue
        clu = pd.read_csv(csv)
        df = combine_results(mat, cb_matrix=clu)         # appends binned TBA
        wk, ar = zip(*(cae.parse_segment(s) for s in df["Folder_Name"]))
        df = df.assign(week=wk, arena=ar).dropna(subset=["arena", "week"])
        parts.append(df[["week", "arena", "TotAccelBA"]])
    if not parts:
        return pd.DataFrame(columns=["week", "arena", "TotAccelBA"])
    out = pd.concat(parts, ignore_index=True)
    out["week"] = out["week"].astype(int)
    return out


def plot(mouse, arena, win, color, out_path):
    """Ridgeline of per-window TBA by natural week for one (mouse, arena), all
    weeks in a single fixed colour."""
    tba = win["TotAccelBA"].to_numpy()
    wk = win["week"].to_numpy()
    if len(tba) < MIN_WEEK_N:
        print(f"    {arena}: only {len(tba)} windows; skipped")
        return None
    uw, cnt = np.unique(wk, return_counts=True)
    weeks = uw[cnt >= MIN_WEEK_N]
    if len(weeks) < 2:
        print(f"    {arena}: <2 usable weeks; skipped")
        return None
    grid = np.linspace(*XLIM, 200)

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(weeks) + 2))
    meds = []
    for i, w in enumerate(weeks):
        v = tba[wk == w]
        base = (len(weeks) - 1 - i) * 1.0                # earliest week at top
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
    ax.set_yticks([])
    ax.set_xlim(*XLIM)
    ax.set_xlabel("total body acceleration (per 60-sample window)")
    ax.set_title(f"{mouse} ({grp}): {arena} arena TBA distribution by week\n"
                 f"weekly-median trend rho={rho:+.2f}, p={p:.3f}")
    fig.tight_layout()
    save_figure(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rho, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="E:/arena_compare",
                    help="arena_compare root (default E:/arena_compare)")
    args = ap.parse_args()

    cae.DATA = Path(args.data)                            # repoint the path helpers
    mice = cae.discover_mice(cae.DATA)
    batches = cae.discover_batches(cae.DATA, mice)
    if not mice:
        sys.exit(f"No <...>_arena_compare mouse dirs under {cae.DATA}")
    colors = cohort_colors(list(mice.values()))
    out_root = ROOT / "output" / "tba_by_week"

    for mouse_dir, mouse in mice.items():
        win = mouse_windows(mouse_dir, batches)
        if win.empty:
            print(f"{mouse}: no arena windows, skipped")
            continue
        folder = out_root / mouse
        os.makedirs(folder, exist_ok=True)
        print(f"{mouse}: {len(win):,} windows "
              f"(2D={int((win.arena=='2D').sum()):,}, 3D={int((win.arena=='3D').sum()):,})")
        for arena in ARENAS:
            sub = win[win.arena == arena]
            rp = plot(mouse, arena, sub, colors[mouse], folder / f"{arena}.jpeg")
            if rp:
                print(f"    {arena}.jpeg  rho={rp[0]:+.2f} p={rp[1]:.3f}")


if __name__ == "__main__":
    main()
