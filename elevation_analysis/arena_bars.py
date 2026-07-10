"""
Period bar plots comparing 2D (open-field) vs 3D (complex) arena transition rates.

Three panels -- first six weeks (8-13), middle weeks (14-18), last six weeks
(19-24; LDOPA saline/ldopa sets are already excluded by load_with_weeks). Each
panel has two triple-bars (lc, then mp); each triple-bar is:
    blue  = 2D arena transition rate        (open-field data, all transitions)
    red   = 3D arena transition rate        (complex data, non-plateau transitions)
    grey  = 3D arena 0-slope transition rate(complex data, equal-height incl plateaus)

All rates are transitions per tracked minute. Each per-mouse value is that mouse's
mean weekly rate over the period's weeks. Bar height = mean over the group's mice;
the individual mouse values are scattered on the bar with a +/- std error bar.

Run:  python arena_bars.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import deproject_centroids as dc
import slope_transitions as st

DATA_3D = dc.DATA
DATA_2D = os.path.join(os.path.dirname(dc.DATA), "data_open")
ELEV_3D = os.path.join(DATA_3D, st.ELEV_NAME)
ELEV_2D = os.path.join(DATA_2D, st.ELEV_NAME)

PERIODS = [
    ("First six weeks (8-13)", set(range(8, 14))),
    ("Middle weeks (14-18)",   set(range(14, 19))),
    ("Last six weeks (19-24)", set(range(19, 25))),
]

# (legend label, color, which rates table, which column)
CATS = [
    ("2D arena transition rate",  "tab:blue", "2d", "rate_all"),
    ("3D arena transition rate",  "tab:red",  "3d", "rate_all"),
    ("3D arena 0-slope rate",     "grey",     "3d", "rate_zero"),
]
BAR_W = 0.8
GROUP_GAP = 1.4


def weekly_rates(mice, data_root, elev, open_field):
    """mouse label -> DataFrame(week, rate_all, rate_zero) of per-minute rates."""
    if open_field:
        arr = np.loadtxt(elev, delimiter=",")            # heights forced to 0
        X = arr[:, 0].max() + arr[:, 0].min() - arr[:, 0]
        Y = arr[:, 1].max() + arr[:, 1].min() - arr[:, 1]
        Z = np.zeros(len(X))
    else:
        X, Y, Z = dc.load_elevation(elev, orient_rot180=True)
    z_at = dc.build_elevation_lookup(X, Y, Z)
    center = (0.5 * (X.min() + X.max()), 0.5 * (Y.min() + Y.max()))
    colxy = np.column_stack([X, Y])
    tree = cKDTree(colxy)

    out = {}
    for name, folder, grp in mice:
        df = st.aligned_xy(folder, X, Y, z_at, center, data_root)
        tr = st.transitions(df, tree, colxy, Z)
        frames = df.groupby("week").size()
        minutes = frames / st.FPS / 60.0
        nonflat = tr[~tr.flat_corner].groupby("week").size().reindex(frames.index, fill_value=0)
        zero = tr[tr.dz < st.ZERO_DZ].groupby("week").size().reindex(frames.index, fill_value=0)
        out[name] = pd.DataFrame({
            "week": frames.index.values,
            "rate_all": (nonflat / minutes).values,
            "rate_zero": (zero / minutes).values,
        })
        print(f"  {name} ({'2D' if open_field else '3D'}): {df.week.nunique()} weeks")
    return out


def period_value(rates, name, weeks, col):
    """A mouse's mean weekly rate over the period's weeks (NaN if none)."""
    d = rates.get(name)
    if d is None:
        return np.nan
    sub = d[d.week.isin(weeks)]
    return sub[col].mean() if len(sub) else np.nan


def make_panel(ax, weeks, mice, rates3d, rates2d, title):
    xticks, xticklabels = [], []
    for gi, grp in enumerate(["lc", "mp"]):
        gmice = [m for m in mice if m[2] == grp]
        base = gi * (len(CATS) * BAR_W + GROUP_GAP)
        for ci, (clabel, color, arena, col) in enumerate(CATS):
            x = base + ci * BAR_W
            rates = rates2d if arena == "2d" else rates3d
            vals = np.array([period_value(rates, n, weeks, col) for n, _, _ in gmice], float)
            m, s = np.nanmean(vals), np.nanstd(vals)
            ax.bar(x, m, width=BAR_W * 0.9, color=color, alpha=0.75,
                   label=clabel if gi == 0 else None, zorder=1)
            ax.errorbar(x, m, yerr=s, fmt="none", ecolor="k", capsize=5, lw=1.4, zorder=2)
            good = ~np.isnan(vals)
            xs = x + np.linspace(-0.22, 0.22, len(vals))
            ax.scatter(xs[good], vals[good], color="k", s=24, zorder=3)
        xticks.append(base + BAR_W)
        xticklabels.append(grp)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def main():
    mice = st.discover_mice(DATA_3D)
    print("computing 3D (complex) arena rates...")
    rates3d = weekly_rates(mice, DATA_3D, ELEV_3D, open_field=False)
    print("computing 2D (open-field) arena rates...")
    rates2d = weekly_rates(mice, DATA_2D, ELEV_2D, open_field=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (title, weeks) in zip(axes, PERIODS):
        make_panel(ax, weeks, mice, rates3d, rates2d, title)
    axes[0].set_ylabel("transitions per minute")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("2D vs 3D arena transition rates by period "
                 "(bar = mean over mice, points = individual mice, error = std)",
                 y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = os.path.join(dc.OUT, "arena_transition_bars.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
