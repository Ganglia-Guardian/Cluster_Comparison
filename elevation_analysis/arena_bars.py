"""
Bar plots comparing 2D (open-field) vs 3D (complex) arena transition rates, coloured
by COHORT (lc orange, mp green -- the same palette the weekly elevation-bin
transition-rate plots use). Two layouts are produced (--layout, default both):

BYGROUP layout -- three figures, one per way of treating the 3D *0-elevation-change*
transitions (equal-height moves between adjacent bins, i.e. moving around the arena
without climbing):

  arena_transition_bars_split      2D | 3D elevation-change | 3D 0-elevation
                                   (0-elevation shown as its own third category)
  arena_transition_bars_combined   2D | 3D
                                   (0-elevation folded into the single 3D bar)
  arena_transition_bars_climbonly  2D | 3D elevation-change
                                   (0-elevation dropped entirely; climbs only)

  Three panels = the three periods (weeks 8-13, 14-18, 19-24; LDOPA saline/ldopa sets
  excluded by load_with_weeks). Within each panel, each ARENA category is a group of
  two bars -- lc and mp. Bar height = mean over the cohort's mice of each mouse's mean
  weekly rate over the period; individual mice are scattered in their own cohort shade
  with a +/- std error bar. A bracket above each group gives the lc-vs-mp significance
  (Mann-Whitney U): ns / * (p<.05) / ** (p<.01) / *** (p<.001).

PROGRESSION layout -- one figure per arena category (arena_progression_2d,
_3d_all, _3d_change, _3d_zero). The x-axis is grouped by MOUSE, and within each mouse
the first/middle/last period bars sit side by side (shaded light->dark) with a line
over the three period means, so each mouse's temporal progression reads left to right.
Cohort still sets the hue; bar height = that mouse's mean weekly rate in the period,
points = its individual weekly rates. Each also gets a cohort-averaged twin
(..._avg): just two groups (lc, mp), bars = mean over the cohort's mice (+/- std),
points = individual mice.

All rates are transitions per tracked minute. Flat-corner wandering (both endpoints
on the SAME flat plateau) is always excluded as a tracking artifact, so within a 3D
arena the categories partition cleanly:  rate_all = rate_change + rate_zero.

The 3D (complex) and 2D (open-field) centroid roots each hold one subfolder per
mouse (all_weeks_centroid.csv + manifest.csv) plus the elevation-geometry CSV. They
default to the collected-centroids layout at the repo root; override with --data-3d /
--data-2d (and --elev-3d / --elev-2d / --out) for a different layout.

Run:  python arena_bars.py [--data-3d DIR] [--data-2d DIR] [--out DIR]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deproject_centroids as dc
import slope_transitions as st
from utils import UTD_GREEN, UTD_ORANGE, cohort_colors, save_figure

# Repo root (parent of elevation_analysis/); the collected-centroids folders sit here.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_3D = os.path.join(REPO, "collected_centroids")        # complex arena
DEFAULT_DATA_2D = os.path.join(REPO, "collected_centroids_open")   # open-field arena

PERIODS = [
    ("First six weeks (8-13)", set(range(8, 14))),
    ("Middle weeks (14-18)",   set(range(14, 19))),
    ("Last six weeks (19-24)", set(range(19, 25))),
]

COHORTS = ["lc", "mp"]
BASE_COLOR = {"lc": UTD_ORANGE, "mp": UTD_GREEN}   # bar fill, one per cohort

# An arena "category" is one group of (lc, mp) bars: (x-tick label, rates table,
# rate column). Modes below pick which categories a figure shows.
CAT_2D        = ("2D arena",                "2d", "rate_all")
CAT_3D_ALL    = ("3D arena",                "3d", "rate_all")
CAT_3D_CHANGE = ("3D arena\nelev. change",  "3d", "rate_change")
CAT_3D_ZERO   = ("3D arena\n0-elevation",   "3d", "rate_zero")

# mode key -> (output stem, [categories], one-line subtitle describing the 3D choice)
MODES = {
    "split":     ("arena_transition_bars_split",
                  [CAT_2D, CAT_3D_CHANGE, CAT_3D_ZERO],
                  "3D 0-elevation moves shown as their own category"),
    "combined":  ("arena_transition_bars_combined",
                  [CAT_2D, CAT_3D_ALL],
                  "3D 0-elevation moves folded into the 3D count"),
    "climbonly": ("arena_transition_bars_climbonly",
                  [CAT_2D, CAT_3D_CHANGE],
                  "3D 0-elevation moves omitted (elevation changes only)"),
}

BAR_W = 0.8
GROUP_GAP = 1.2
MOUSE_COLOR = {}   # mouse label -> its individual cohort shade, filled in main()

# --- progression layout: one figure per arena category ------------------------
# (output stem, figure title, rates table, rate column). Each mouse becomes a
# group of first/middle/last period bars, so within-mouse progression reads left
# to right. The three periods are shaded light -> dark by these alphas.
ARENA_FIGS = [
    ("arena_progression_2d",        "2D arena",                    "2d", "rate_all"),
    ("arena_progression_3d_all",    "3D arena (all moves)",        "3d", "rate_all"),
    ("arena_progression_3d_change", "3D arena (elevation change)", "3d", "rate_change"),
    ("arena_progression_3d_zero",   "3D arena (0-elevation)",      "3d", "rate_zero"),
]
PERIOD_ALPHA = [0.40, 0.62, 0.90]   # first, middle, last


def weekly_rates(mice, data_root, elev, open_field):
    """mouse label -> DataFrame(week, rate_all, rate_change, rate_zero) of per-minute rates.

    Flat-corner (same-plateau) transitions are excluded from every column, so the
    three columns nest:  rate_all = rate_change + rate_zero, where
        rate_change = transitions with a real elevation change (|dz| >= ZERO_DZ)
        rate_zero   = 0-elevation transitions (|dz| < ZERO_DZ, non-plateau)
    In the flat 2D arena every |dz| is 0, so rate_all == rate_zero and rate_change == 0;
    only rate_all is ever read for 2D.
    """
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

        keep = ~tr.flat_corner                            # drop plateau wandering always
        change = keep & (tr.dz >= st.ZERO_DZ)             # real elevation changes
        zero = keep & (tr.dz < st.ZERO_DZ)                # equal-height, non-plateau

        def rate(mask):
            cnt = tr[mask].groupby("week").size().reindex(frames.index, fill_value=0)
            return (cnt / minutes).values

        out[name] = pd.DataFrame({
            "week": frames.index.values,
            "rate_all": rate(keep),
            "rate_change": rate(change),
            "rate_zero": rate(zero),
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


def cohort_values(rates, gmice, weeks, col):
    """Per-mouse period values for the mice in one cohort (labels aligned)."""
    labels = [n for n, _, _ in gmice]
    vals = np.array([period_value(rates, n, weeks, col) for n in labels], float)
    return labels, vals


def mouse_period_weeks(rates, name, weeks, col):
    """A mouse's individual weekly rates that fall in the period (empty if none)."""
    d = rates.get(name)
    if d is None:
        return np.array([])
    sub = d[d.week.isin(weeks)]
    return sub[col].to_numpy(float)


def cohort_weekly_trend(rates, gmice, col):
    """Spearman trend of a cohort's weekly-MEAN rate vs week -> (p, rho).

    For each week the rate is averaged over whichever of the cohort's mice were
    tracked that week (one point per week, so no per-mouse pseudoreplication).
    Returns (nan, nan) if fewer than three weeks are available.
    """
    frames = [rates[n][["week", col]] for n, _, _ in gmice if n in rates]
    if not frames:
        return np.nan, np.nan
    wk = pd.concat(frames).groupby("week")[col].mean().sort_index()
    if len(wk) < 3:
        return np.nan, np.nan
    res = spearmanr(wk.index.to_numpy(), wk.to_numpy())
    return res.pvalue, res.statistic


def stars(p):
    """p-value -> significance label."""
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def lc_vs_mp_p(vals_by_cohort):
    """Two-sided Mann-Whitney U p-value for lc vs mp (NaN if too few points)."""
    lc = vals_by_cohort["lc"][~np.isnan(vals_by_cohort["lc"])]
    mp = vals_by_cohort["mp"][~np.isnan(vals_by_cohort["mp"])]
    if len(lc) < 1 or len(mp) < 1:
        return np.nan
    try:
        return mannwhitneyu(lc, mp, alternative="two-sided").pvalue
    except ValueError:                     # e.g. all values identical
        return np.nan


def sig_bracket(ax, x1, x2, top, text, span):
    """Draw a significance bracket from x1 to x2 sitting just above `top`."""
    h = 0.03 * span
    y = top + 0.04 * span
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k", zorder=4)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=11, zorder=4)


def make_panel(ax, weeks, mice, rates3d, rates2d, cats, title, span):
    xticks, xticklabels = [], []
    stride = len(COHORTS) * BAR_W + GROUP_GAP
    for ci, (clabel, arena, col) in enumerate(cats):
        rates = rates2d if arena == "2d" else rates3d
        base = ci * stride
        vals_by_cohort, tops = {}, []
        for hi, grp in enumerate(COHORTS):
            gmice = [m for m in mice if m[2] == grp]
            labels, vals = cohort_values(rates, gmice, weeks, col)
            vals_by_cohort[grp] = vals
            x = base + hi * BAR_W
            m, s = np.nanmean(vals), np.nanstd(vals)
            ax.bar(x, m, width=BAR_W * 0.9, color=BASE_COLOR[grp], alpha=0.55,
                   edgecolor=BASE_COLOR[grp], lw=1.2, zorder=1)
            ax.errorbar(x, m, yerr=s, fmt="none", ecolor="k", capsize=5, lw=1.4, zorder=2)
            good = ~np.isnan(vals)
            xs = x + np.linspace(-0.24, 0.24, len(vals))
            ax.scatter(xs[good], vals[good],
                       color=[MOUSE_COLOR[n] for n in np.array(labels)[good]],
                       edgecolor="k", linewidth=0.6, s=30, zorder=3)
            tops.append((m + s) if not np.isnan(m) else 0.0)
        # lc-vs-mp significance bracket spanning the two cohort bars
        p = lc_vs_mp_p(vals_by_cohort)
        sig_bracket(ax, base, base + BAR_W, max(tops), stars(p), span)
        xticks.append(base + 0.5 * BAR_W)
        xticklabels.append(clabel)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_title(title)


def figure_span(mice, rates3d, rates2d):
    """A y-range estimate used to size the significance brackets consistently."""
    vals = []
    for _, weeks in PERIODS:
        for cat in (CAT_2D, CAT_3D_ALL, CAT_3D_CHANGE, CAT_3D_ZERO):
            _, arena, col = cat
            rates = rates2d if arena == "2d" else rates3d
            for grp in COHORTS:
                gmice = [m for m in mice if m[2] == grp]
                _, v = cohort_values(rates, gmice, weeks, col)
                if np.isfinite(v).any():
                    vals.append(np.nanmean(v) + np.nanstd(v))
    return max(vals) if vals else 1.0


def run_mode(key, mice, rates3d, rates2d, span, out_root):
    stem, cats, subtitle = MODES[key]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (title, weeks) in zip(axes, PERIODS):
        make_panel(ax, weeks, mice, rates3d, rates2d, cats, title, span)
    axes[0].set_ylabel("transitions per minute")

    handles = [Patch(facecolor=BASE_COLOR[g], edgecolor=BASE_COLOR[g], alpha=0.55,
                     label={"lc": "lc (littermate control)", "mp": "mp (MitoPark)"}[g])
               for g in COHORTS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("2D vs 3D arena transition rates by period  --  " + subtitle
                 + "\n(bar = mean over mice, points = individual mice, error = std; "
                 "bracket = lc vs mp Mann-Whitney U)", y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out = os.path.join(out_root, stem + ".jpeg")
    save_figure(fig, out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def run_progression(stem, title, arena, col, mice, rates3d, rates2d, out_root):
    """One figure for a single arena category: each mouse is a group of first/
    middle/last period bars (shaded light->dark), with a line over the three
    period means to stress each mouse's progression. Cohort sets the hue."""
    rates = rates2d if arena == "2d" else rates3d
    bw, intra_gap, cohort_gap = BAR_W, 0.5, 1.6

    fig, ax = plt.subplots(figsize=(15, 6))
    x, prev_grp = 0.0, None
    xticks, xticklabels = [], []
    for name, _, grp in mice:
        if prev_grp is not None and grp != prev_grp:
            x += cohort_gap                       # visual gap between cohorts
        base, centers, means = x, [], []
        for pi, (_, weeks) in enumerate(PERIODS):
            bx = base + pi * bw
            vals = mouse_period_weeks(rates, name, weeks, col)
            m = np.nanmean(vals) if len(vals) else np.nan
            ax.bar(bx, m, width=bw * 0.9, color=BASE_COLOR[grp], alpha=PERIOD_ALPHA[pi],
                   edgecolor=BASE_COLOR[grp], lw=1.0, zorder=1)
            if len(vals):
                jx = bx + np.linspace(-0.20, 0.20, len(vals))
                ax.scatter(jx, vals, color="k", s=12, alpha=0.55, zorder=3)
            centers.append(bx)
            means.append(m)
        ax.plot(centers, means, "-o", color="#333333", lw=1.3, ms=4, zorder=4)
        xticks.append(base + bw)                  # centre over the three bars
        xticklabels.append(name)                  # name already carries the cohort suffix
        x = base + 3 * bw + intra_gap
        prev_grp = grp

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("transitions per minute")
    progression_legend(fig)
    fig.suptitle(f"{title}: per-mouse weekly transition rate across study periods\n"
                 "(bars = period mean, points = individual weeks, line = progression; "
                 "cohort sets hue, period sets shade)", y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    out = os.path.join(out_root, stem + ".jpeg")
    save_figure(fig, out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def progression_legend(fig):
    """Cohort-hue + period-shade legend shared by the progression figures."""
    cohort_h = [Patch(facecolor=BASE_COLOR[g], edgecolor=BASE_COLOR[g], alpha=0.9,
                      label={"lc": "lc (control)", "mp": "mp (MitoPark)"}[g])
                for g in COHORTS]
    period_h = [Patch(facecolor="0.45", edgecolor="0.45", alpha=PERIOD_ALPHA[i],
                      label=PERIODS[i][0]) for i in range(len(PERIODS))]
    fig.legend(handles=cohort_h + period_h, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))


def run_progression_avg(stem, title, arena, col, mice, rates3d, rates2d, out_root):
    """Cohort-averaged twin of run_progression: one figure per arena category with
    just two groups (lc, mp). Within each, the first/middle/last bars are the mean
    over the cohort's mice of each mouse's period-mean rate (+/- std across mice),
    individual mice are scattered in their own shade, and a line over the three
    means shows the cohort's progression."""
    rates = rates2d if arena == "2d" else rates3d
    bw, cohort_gap = BAR_W, 1.6

    fig, ax = plt.subplots(figsize=(9, 6))
    xticks, xticklabels, annotations, ymax = [], [], [], 0.0
    for gi, grp in enumerate(COHORTS):
        gmice = [m for m in mice if m[2] == grp]
        base = gi * (3 * bw + cohort_gap)
        centers, means, top = [], [], 0.0
        for pi, (_, weeks) in enumerate(PERIODS):
            bx = base + pi * bw
            labels, vals = cohort_values(rates, gmice, weeks, col)
            m, s = np.nanmean(vals), np.nanstd(vals)
            ax.bar(bx, m, width=bw * 0.9, color=BASE_COLOR[grp], alpha=PERIOD_ALPHA[pi],
                   edgecolor=BASE_COLOR[grp], lw=1.0, zorder=1)
            ax.errorbar(bx, m, yerr=s, fmt="none", ecolor="k", capsize=4, lw=1.2, zorder=2)
            good = ~np.isnan(vals)
            jx = bx + np.linspace(-0.20, 0.20, len(vals))
            ax.scatter(jx[good], vals[good],
                       color=[MOUSE_COLOR[n] for n in np.array(labels)[good]],
                       edgecolor="k", linewidth=0.6, s=30, zorder=3)
            centers.append(bx)
            means.append(m)
            if not np.isnan(m):
                top = max(top, m + s, np.nanmax(vals))
        ax.plot(centers, means, "-o", color="#333333", lw=1.3, ms=4, zorder=4)
        xticks.append(base + bw)
        xticklabels.append({"lc": "lc (control)", "mp": "mp (MitoPark)"}[grp])
        # progression-trend star: Spearman of weekly cohort-mean rate vs week
        p, rho = cohort_weekly_trend(rates, gmice, col)
        annotations.append((base + bw, top, f"{stars(p)}  (rho={rho:+.2f})"))
        ymax = max(ymax, top)

    pad = 0.05 * ymax if ymax else 1.0
    for xc, top, txt in annotations:
        ax.text(xc, top + pad, txt, ha="center", va="bottom", fontsize=12, zorder=5)
    ax.set_ylim(top=ymax * 1.20 if ymax else None)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("transitions per minute")
    progression_legend(fig)
    fig.suptitle(f"{title}: cohort-mean weekly transition rate across study periods\n"
                 "(bars = mean over mice, points = individual mice, error = std, line = "
                 "progression; star = Spearman trend of weekly cohort-mean rate vs week)",
                 y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    out = os.path.join(out_root, stem + ".jpeg")
    save_figure(fig, out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def parse_args():
    p = argparse.ArgumentParser(
        description="2D-vs-3D arena transition-rate bar plots, grouped by arena and "
                    "coloured by cohort (lc/mp) with lc-vs-mp significance stars.")
    p.add_argument("--data-3d", default=DEFAULT_DATA_3D,
                   help=f"complex-arena centroid root (default: {DEFAULT_DATA_3D})")
    p.add_argument("--data-2d", default=DEFAULT_DATA_2D,
                   help=f"open-field centroid root (default: {DEFAULT_DATA_2D})")
    p.add_argument("--elev-3d", default=None,
                   help=f"complex elevation CSV (default: <data-3d>/{st.ELEV_NAME})")
    p.add_argument("--elev-2d", default=None,
                   help=f"open-field elevation CSV (default: <data-2d>/{st.ELEV_NAME})")
    p.add_argument("--out", default=dc.OUT,
                   help=f"output folder for figures (default: {dc.OUT})")
    p.add_argument("--layout", choices=["bygroup", "progression", "both"],
                   default="both",
                   help="bygroup: old per-period panels, arena groups of lc/mp bars "
                        "with significance stars. progression: one figure per arena "
                        "category, each mouse's first/middle/last period bars side by "
                        "side. both (default): produce both sets.")
    return p.parse_args()


def main():
    args = parse_args()
    data_3d, data_2d = args.data_3d, args.data_2d
    elev_3d = args.elev_3d or os.path.join(data_3d, st.ELEV_NAME)
    elev_2d = args.elev_2d or os.path.join(data_2d, st.ELEV_NAME)
    os.makedirs(args.out, exist_ok=True)

    # Mice are discovered from the 3D root; the same labels are looked up in the 2D
    # root, so both arenas must use matching per-mouse folder names.
    mice = st.discover_mice(data_3d)
    if not mice:
        raise SystemExit(f"no matching mice found under {data_3d}")
    MOUSE_COLOR.update(cohort_colors([name for name, _, _ in mice]))
    print("mice:", ", ".join(f"{n}({g})" for n, _, g in mice))

    print("computing 3D (complex) arena rates...")
    rates3d = weekly_rates(mice, data_3d, elev_3d, open_field=False)
    print("computing 2D (open-field) arena rates...")
    rates2d = weekly_rates(mice, data_2d, elev_2d, open_field=True)

    if args.layout in ("bygroup", "both"):
        span = figure_span(mice, rates3d, rates2d)
        for key in MODES:
            run_mode(key, mice, rates3d, rates2d, span, args.out)
    if args.layout in ("progression", "both"):
        for stem, title, arena, col in ARENA_FIGS:
            run_progression(stem, title, arena, col, mice, rates3d, rates2d, args.out)
            run_progression_avg(stem + "_avg", title, arena, col,
                                 mice, rates3d, rates2d, args.out)


if __name__ == "__main__":
    main()
