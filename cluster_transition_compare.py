"""Cross-dataset transition plots for the MitoPark cohorts.

Two figures:

1. FAN-OUT COMPARISON (--datasets). Mean number of distinct successor clusters
   per source cluster, by week, overlaid for every dataset. lc = littermate
   controls (dashed), mp = MitoPark / disease (solid). If each behavioural state
   branches to more successors as disease progresses, the mp lines rise above the
   lc lines. Because the datasets have different codebook sizes K (clusters), the
   raw count is confounded -- so the right panel normalizes by K (mean fraction of
   the repertoire reached as a successor), which is comparable across cohorts.
   Under-sampled weeks (far fewer frames than the dataset's median, e.g. lc w10
   with 16 frames) are dropped.

2. SUCCESSOR-DISTRIBUTION DRIFT (one per dataset). For the few source clusters
   whose dominant successor cleanly shifts from one target early in disease to a
   different target late, plots a heatmap per source: x = week, y = successor
   cluster, colour = fraction of that source's outgoing transitions that week
   (each week-column sums to 1). Rows are ordered by when they peak, so the
   (a,b) early -> (a,c)/(a,d) late degeneration reads as the bright band sliding
   down the heatmap. Cluster labels sit on a categorical axis -- only the colour
   (frequency) is a magnitude.

Reuses build_transitions / week_sort_key / is_variant from
cluster_transition_labels.py so the transition rules and week ordering match.

Run:
    python cluster_transition_compare.py
    python cluster_transition_compare.py --datasets 1lc 2lc 1mp 2mp 3mp
    python cluster_transition_compare.py --datasets 1mp 2mp 3mp --top 6
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cluster_transition_labels import (IDX_COL, WEEK_COL, build_transitions,
                                        is_variant, week_sort_key)

CSV_NAME = "Cluster_detail_results.csv"
DATA_ROOT = Path("data")
DEFAULT_DATASETS = ["1lc", "2lc", "1mp", "2mp", "3mp"]
MIN_FRAME_FRAC = 0.5   # drop weeks with < this fraction of the dataset's median frames


def cohort(name):
    """'lc' (control) or 'mp' (disease) from the dataset name."""
    return "lc" if "lc" in name.lower() else "mp"


def load(name):
    return pd.read_csv(DATA_ROOT / name / CSV_NAME).reset_index(drop=True)


def progression_frames(df):
    """Per-week frame counts for the non-variant (normal) weeks."""
    frames = df[df[WEEK_COL].notna()].groupby(WEEK_COL).size()
    return frames[[not is_variant(w) for w in frames.index]]


def fanout_by_week(df):
    """Map week-number -> mean distinct successors per source, dropping the
    under-sampled weeks. Also returns K (codebook size) for normalization."""
    trans = build_transitions(df)
    frames = progression_frames(df)
    if frames.empty:
        return {}, int(df[IDX_COL].max())
    med = float(np.median(frames.to_numpy()))
    keep = {w for w in frames.index if frames[w] >= MIN_FRAME_FRAC * med}

    prog = trans[trans["week"].isin(keep)]
    out = {}
    for w, gw in prog.groupby("week"):
        per_source = gw.groupby("source")["target"].nunique()
        out[int(week_sort_key(w))] = float(per_source.mean())
    return out, int(df[IDX_COL].max())


def plot_fanout(datasets, out_path):
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    for i, name in enumerate(datasets):
        series, k = fanout_by_week(load(name))
        if not series:
            continue
        weeks = sorted(series)
        y = np.array([series[w] for w in weeks])
        style = "--" if cohort(name) == "lc" else "-"
        color = cmap(i % 10)
        # Spearman of fan-out vs week. Note: dividing y by the constant K is
        # rank-preserving, so rho/p are identical for the raw and normalized panels.
        rho, p = spearmanr(weeks, y)
        stat = f"  rho={rho:.2f}, p={p:.3f}"
        label = f"{name} (K={k}){stat}"
        axes[0].plot(weeks, y, style, marker="o", ms=4, color=color, label=label)
        axes[1].plot(weeks, y / k, style, marker="o", ms=4, color=color, label=label)

    axes[0].set_ylabel("mean distinct successors / source")
    axes[0].set_title("Behavioural fan-out by week (raw)")
    axes[1].set_ylabel("mean fraction of repertoire reached  (/ K)")
    axes[1].set_title("Fan-out normalized by codebook size K")
    for ax in axes:
        ax.set_xlabel("disease week")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Transition fan-out over disease course  "
                 "(dashed = littermate control, solid = MitoPark)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def source_week_fanout(df):
    """sources x week-number table of distinct successors per source per week,
    restricted to well-sampled progression weeks (same guard as the fan-out plot)."""
    trans = build_transitions(df)
    frames = progression_frames(df)
    keep = {w for w in frames.index
            if frames[w] >= MIN_FRAME_FRAC * float(np.median(frames.to_numpy()))}
    prog = trans[trans["week"].isin(keep)]
    fan = (prog.groupby(["source", "week"])["target"].nunique()
           .rename("n").reset_index())
    fan["wn"] = fan["week"].map(week_sort_key).astype(int)
    return fan.pivot_table(index="source", columns="wn", values="n")


def top_change_series(pivot, top, direction):
    """Average weekly fan-out of the `top` sources with the largest signed
    early->late change. direction='expansion' takes the biggest increases,
    'contraction' the biggest decreases. Returns (series indexed by week, sources)."""
    weeks = sorted(pivot.columns)
    third = max(1, len(weeks) // 3)
    early = pivot[weeks[:third]].mean(axis=1)
    late = pivot[weeks[-third:]].mean(axis=1)
    change = (late - early).dropna()  # sources present in both ends
    ascending = direction == "contraction"
    sel = change.sort_values(ascending=ascending).head(top).index
    series = pivot.loc[sel, weeks].mean(axis=0).dropna()
    return series, list(map(int, sel))


def fit_curve(x, y, model):
    """Return (xs, ys, slope) for the fitted model; slope is None for non-linear."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs = np.linspace(x.min(), x.max(), 200)
    if model == "linear":
        c = np.polyfit(x, y, 1)
        return xs, np.polyval(c, xs), float(c[0])
    if model == "cubic":
        c = np.polyfit(x, y, min(3, len(x) - 1))
        return xs, np.polyval(c, xs), None
    if model == "spline":
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(x, y, k=min(3, len(x) - 1), s=len(x))
        return xs, spl(xs), None
    raise ValueError(f"unknown model {model!r}")


def plot_top_change(datasets, direction, model, top, out_path):
    """Overlay, for every dataset, the fitted weekly fan-out of its top
    expanding / contracting source clusters."""
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i, name in enumerate(datasets):
        pivot = source_week_fanout(load(name))
        if pivot.empty:
            continue
        series, sel = top_change_series(pivot, top, direction)
        if len(series) < 4:
            continue
        x, y = series.index.to_numpy(float), series.to_numpy(float)
        rho, p = spearmanr(x, y)
        color = cmap(i % 10)
        style = "--" if cohort(name) == "lc" else "-"
        ax.scatter(x, y, color=color, alpha=0.35, s=18, edgecolors="none")
        xs, ys, slope = fit_curve(x, y, model)
        extra = f", slope={slope:+.2f}" if slope is not None else ""
        ax.plot(xs, ys, style, color=color, lw=2.2,
                label=f"{name}{extra}, rho={rho:.2f}, p={p:.3f}")

    verb = "expansion" if direction == "expansion" else "contraction"
    ax.set_xlabel("disease week")
    ax.set_ylabel(f"distinct successors / source  (mean of top {top})")
    ax.set_title(f"Top early->late target {verb} by week  ({model} fit)\n"
                 "dashed = littermate control, solid = MitoPark")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def drift_sources(trans, prog_weeks, top, min_count=20):
    """Rank sources by a CLEAN early->late dominant-target shift.

    For each source: early_target = most common successor over the first third of
    weeks, late_target = most common over the last third. We keep sources where
    those differ, scored by min(early_frac, late_frac) so both ends are
    well-defined (not just noisy week-to-week flicker). Returns a list of dicts."""
    n = len(prog_weeks)
    third = max(1, n // 3)
    early_w, late_w = set(prog_weeks[:third]), set(prog_weeks[-third:])

    scored = []
    for source, g in trans.groupby("source"):
        if len(g) < min_count:
            continue
        e = g[g["week"].isin(early_w)]["target"]
        l = g[g["week"].isin(late_w)]["target"]
        if e.empty or l.empty:
            continue
        e_tgt, l_tgt = int(e.mode().iloc[0]), int(l.mode().iloc[0])
        if e_tgt == l_tgt:
            continue
        score = min((e == e_tgt).mean(), (l == l_tgt).mean())
        scored.append({"source": int(source), "early_target": e_tgt,
                       "late_target": l_tgt, "score": float(score),
                       "n": int(len(g))})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top]


def successor_distribution(g, week_nums, top_k=8):
    """For one source: a (week x successor) table of FRACTIONS that sum to 1 each
    week. Only the top_k most-used successors keep their own band; the rest are
    pooled into 'other' so the colour count stays readable. Returns (frac, bands)."""
    g = g.copy()
    g["wn"] = g["week"].map(week_sort_key).astype(int)
    counts = (g.groupby(["wn", "target"]).size().rename("n").reset_index()
              .pivot_table(index="wn", columns="target", values="n", fill_value=0)
              .reindex(week_nums, fill_value=0))
    totals = counts.sum(axis=0).sort_values(ascending=False)
    keep = list(totals.index[:top_k])
    dist = counts[keep].copy()
    rest = [t for t in counts.columns if t not in keep]
    if rest:
        dist["other"] = counts[rest].sum(axis=1)
    rowsum = dist.sum(axis=1).replace(0, np.nan)
    frac = dist.div(rowsum, axis=0).fillna(0.0)
    bands = [c for c in dist.columns]  # keep order: top successors then 'other'
    return frac, bands


def plot_drift(name, top, out_path):
    df = load(name)
    trans = build_transitions(df)
    frames = progression_frames(df)
    keep_w = {w for w in frames.index
              if frames[w] >= MIN_FRAME_FRAC * float(np.median(frames.to_numpy()))}
    prog_weeks = sorted({w for w in trans["week"].unique()
                         if not is_variant(w) and w in keep_w}, key=week_sort_key)
    prog = trans[trans["week"].isin(set(prog_weeks))]
    chosen = drift_sources(prog, prog_weeks, top)
    if not chosen:
        return None

    week_nums = np.array([int(week_sort_key(w)) for w in prog_weeks], float)

    # build each source's (successor x week) frequency matrix, rows time-sorted
    mats = []
    for info in chosen:
        s = info["source"]
        frac, bands = successor_distribution(prog[prog["source"] == s],
                                             list(week_nums.astype(int)), top_k=12)
        M = frac[bands].to_numpy().T  # successors x weeks; each week-column sums to 1
        labels = [str(b) for b in bands]
        # order rows by temporal centroid so the drift shows as a diagonal;
        # the pooled 'other' row is always parked at the bottom
        weight = M.sum(axis=1)
        centroid = np.divide((M * week_nums).sum(axis=1), weight,
                             out=np.full(len(labels), np.inf), where=weight > 0)
        centroid = [np.inf if l == "other" else c for l, c in zip(labels, centroid)]
        order = np.argsort(centroid, kind="stable")
        mats.append((s, M[order], [labels[i] for i in order]))

    vmax = max(m[1].max() for m in mats) or 1.0
    ncol = min(3, len(chosen))
    nrow = int(np.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow),
                             squeeze=False)

    im = None
    for ax, (s, M, labels) in zip(axes.ravel(), mats):
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(week_nums)))
        ax.set_xticklabels(week_nums.astype(int), fontsize=6, rotation=90)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"cluster {s}", fontsize=9)
        ax.set_xlabel("disease week"); ax.set_ylabel("successor cluster")

    for ax in axes.ravel()[len(chosen):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                 label="fraction of that week's transitions")
    fig.suptitle(f"{name}: successor-cluster frequency over disease, per source\n"
                 "(colour = fraction of that source's transitions that week; "
                 "rows sorted by when they peak)", y=1.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help="dataset folder names under data/ (default: all five)")
    ap.add_argument("--top", type=int, default=6,
                    help="how many drifting source clusters to show per dataset")
    ap.add_argument("--change-top", type=int, default=5,
                    help="how many top expanding/contracting sources to average "
                         "for the combined change plots")
    ap.add_argument("--model", choices=["linear", "cubic", "spline"],
                    default="linear", help="regression model for the change plots")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT,
                    help="where to write the combined figures")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = plot_fanout(args.datasets, args.out_dir / "compare_fanout_by_week.png")
    print(f"Wrote {p}")

    for direction in ("expansion", "contraction"):
        out = args.out_dir / f"compare_top_{direction}_by_week.png"
        print(f"Wrote {plot_top_change(args.datasets, direction, args.model, args.change_top, out)}")

    for name in args.datasets:
        out = DATA_ROOT / name / "successor_distribution_by_week.png"
        res = plot_drift(name, args.top, out)
        print(f"Wrote {res}" if res else f"{name}: no clean drifting sources found")


if __name__ == "__main__":
    main()
