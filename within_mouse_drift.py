"""Within-mouse occupancy drift across weeks, vs the sampling-noise floor.

Each weekly recording (30 min, same animal) gives an occupancy distribution over
that mouse's OWN cluster codebook. We measure how far each week drifts from an
early-week baseline (Jensen-Shannon distance) and compare that drift to the
sampling-noise floor -- the JS you'd expect between two same-distribution samples
of the same size. Drift above the floor is change beyond what 30 min of sampling
noise explains; for MitoPark that is candidate disease progression, while
littermate controls should stay near the floor.

Why this needs no shared library / no re-clustering
---------------------------------------------------
Drift is computed WITHIN a mouse, on that mouse's own labels (`Clusters/idx`), so
there is no cross-mouse cluster-matching problem. It also reuses the existing
labels, so it costs megabytes, not the hundreds of GB the clustering needs.

Floor, done honestly
--------------------
The split-half JS (~0.12) was a half-vs-half number; drift is week-vs-baseline.
To compare like with like we build the null per week by bootstrap: draw a
pseudo-week of the SAME bin count from the pooled baseline distribution and
measure its JS to the baseline. The 95th percentile of that null is the floor for
that week. Points above it are drift beyond sampling noise.

Scope
-----
mp and lc are both 30 min/week, so durations are matched and the floor applies
directly across them. Wildtype (2 h) is a separate, longer-recording cohort and
is deliberately NOT included here -- its floor would be tighter.

Run:
    uv run python within_mouse_drift.py
    uv run python within_mouse_drift.py --datasets 1mp 1lc --baseline-weeks 2
(If a stray VIRTUAL_ENV points at anaconda, prefix with `VIRTUAL_ENV= `.)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from cluster_sim_by_week import week_bin_ranges, week_sort_key
from split_half_occupancy import (CSV_NAME, DATA_ROOT, MAT_NAME, load_idx,
                                  occupancy)

N_BASELINE_WEEKS = 2     # earliest valid weeks pooled as the "before" reference
N_BOOT = 500             # bootstrap draws for the per-week null floor
FLOOR_PCT = 95           # floor = this percentile of the null JS
MIN_BINS_FRAC = 0.5      # drop weeks with < this fraction of the median bin count


def is_variant(week):
    """True for the pharmacological conditions (L-DOPA / saline), not normal weeks."""
    low = week.lower()
    return "ldop" in low or "saline" in low


def bootstrap_floor(baseline_labels, base_occ, k, sample_size, rng):
    """95th-pct JS between a same-size pseudo-week drawn from baseline and baseline."""
    js = np.empty(N_BOOT)
    for b in range(N_BOOT):
        samp = rng.choice(baseline_labels, size=sample_size, replace=True)
        js[b] = jensenshannon(occupancy(samp, k), base_occ)
    return float(np.percentile(js, FLOOR_PCT))


def compute_drift(ds_dir, baseline_weeks=N_BASELINE_WEEKS, rng=None):
    """Per-week JS drift from an early-week baseline, plus the per-week null floor."""
    rng = rng or np.random.default_rng(0)
    idx = load_idx(ds_dir / MAT_NAME)
    k = int(idx.max())
    ranges = week_bin_ranges(ds_dir / CSV_NAME, idx.size)
    ranges = sorted(ranges, key=lambda r: week_sort_key(r[0]))

    sizes = np.array([end - start for _, start, end in ranges])
    min_bins = MIN_BINS_FRAC * np.median(sizes)

    # baseline = pooled labels of the first N valid, non-variant weeks
    base_labels = []
    used = []
    for week, start, end in ranges:
        if len(used) >= baseline_weeks:
            break
        if (end - start) >= min_bins and not is_variant(week):
            base_labels.append(idx[start:end])
            used.append(week)
    base_labels = np.concatenate(base_labels)
    base_occ = occupancy(base_labels, k)

    rows = []
    for week, start, end in ranges:
        n = end - start
        if n < min_bins:
            continue                      # QC: skip truncated/empty recordings (e.g. w10)
        occ = occupancy(idx[start:end], k)
        rows.append(dict(
            dataset=ds_dir.name, week=week, week_num=week_sort_key(week),
            n_bins=int(n), variant=is_variant(week),
            drift_js=float(jensenshannon(occ, base_occ)),
            floor95=bootstrap_floor(base_labels, base_occ, k, n, rng),
            baseline=week in used))
    df = pd.DataFrame(rows)

    # progression test on the regular weeks only (variants excluded)
    reg = df[~df["variant"]]
    rho, p = spearmanr(reg["week_num"], reg["drift_js"])
    return df, k, used, float(rho), float(p)


def plot_drift(df, k, used, rho, p, ds_name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = df[~df["variant"]]
    fig, ax = plt.subplots(figsize=(11, 5))

    # sampling-noise floor band (per-week 95th pct null; ~flat since weeks ~equal)
    order = df.sort_values("week_num")
    ax.fill_between(order["week_num"], 0, order["floor95"], color="grey", alpha=0.25,
                    label=f"sampling-noise floor ({FLOOR_PCT}th pct null)")

    ax.plot(reg["week_num"], reg["drift_js"], "o-", color="tab:blue",
            label="drift from baseline (JS)")
    # pharmacological conditions as distinct markers
    for _, r in df[df["variant"]].iterrows():
        style = dict(color="tab:red", marker="D") if "ldop" in r["week"].lower() \
            else dict(color="tab:purple", marker="^")
        ax.plot(r["week_num"], r["drift_js"], linestyle="none", markersize=11, **style)
        ax.annotate("L-DOPA" if "ldop" in r["week"].lower() else "saline",
                    (r["week_num"], r["drift_js"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=style["color"])

    base_x = df[df["baseline"]]["week_num"]
    ax.scatter(base_x, df[df["baseline"]]["drift_js"], s=120, facecolors="none",
               edgecolors="tab:green", linewidths=2, zorder=5, label="baseline weeks")

    ax.set_xlabel("week"); ax.set_ylabel("Jensen-Shannon distance from baseline")
    ax.set_ylim(bottom=0)
    sig = "progressive" if (p < 0.05 and rho > 0) else "no monotonic trend"
    ax.set_title(f"{ds_name}: within-mouse occupancy drift  (K={k}, baseline="
                 f"{'+'.join(used)})\nSpearman drift vs week: rho={rho:.2f}, "
                 f"p={p:.3g}  -> {sig}")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "occupancy_drift.png", dpi=150)
    plt.close(fig)


def plot_combined(per_ds, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"1mp": "tab:blue", "1lc": "tab:red", "2lc": "tab:orange", "2mp": "tab:cyan", "3mp": "tab:purple"}
    floor_top = max(df["floor95"].max() for df, *_ in per_ds.values())
    ax.axhspan(0, floor_top, color="grey", alpha=0.18,
               label=f"sampling-noise floor (<= {floor_top:.3f})")

    for ds, (df, k, used, rho, p) in per_ds.items():
        reg = df[~df["variant"]]
        c = colors.get(ds, None)
        tag = "MitoPark" if "mp" in ds.lower() else "control"
        ax.plot(reg["week_num"], reg["drift_js"], "o-", color=c,
                label=f"{ds} ({tag}); rho={rho:.2f}, p={p:.2g}")
        for _, r in df[df["variant"]].iterrows():
            mk = "^" if "saline" in r["week"].lower() else "D"
            ax.plot(r["week_num"], r["drift_js"], mk, color=c, markersize=10,
                    markeredgecolor="k", linestyle="none")

    ax.set_xlabel("week"); ax.set_ylabel("JS distance from each mouse's own baseline")
    ax.set_ylim(bottom=0)
    ax.set_title("Within-mouse occupancy drift across weeks\n"
                 "(diamonds = L-DOPA, triangles = saline; each mouse vs its own baseline)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--datasets", nargs="*", default=None,
                   help=f"dataset folders under data/ (default: all with {MAT_NAME})")
    p.add_argument("--baseline-weeks", type=int, default=N_BASELINE_WEEKS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.datasets:
        dirs = [args.data_root / d for d in args.datasets]
    else:
        dirs = sorted(d for d in args.data_root.iterdir()
                      if d.is_dir() and (d / MAT_NAME).exists())

    per_ds, all_df = {}, []
    rng = np.random.default_rng(0)
    for ds_dir in dirs:
        print(f"\n=== {ds_dir.name} ===")
        df, k, used, rho, p = compute_drift(ds_dir, args.baseline_weeks, rng)
        out_dir = ds_dir / "drift_out"
        plot_drift(df, k, used, rho, p, ds_dir.name, out_dir)
        df.to_csv(out_dir / "occupancy_drift.csv", index=False)
        per_ds[ds_dir.name] = (df, k, used, rho, p)
        all_df.append(df)

        above = df[(~df["variant"]) & (df["drift_js"] > df["floor95"])]
        print(df[["week", "n_bins", "drift_js", "floor95", "baseline"]]
              .to_string(index=False))
        print(f"  baseline={'+'.join(used)}  Spearman(drift,week): rho={rho:.2f} "
              f"p={p:.3g}  | weeks above floor: {len(above)}/{(~df['variant']).sum()}")
        print(f"  -> {out_dir}")

    if all_df:
        pd.concat(all_df, ignore_index=True).to_csv(
            args.data_root / "occupancy_drift_summary.csv", index=False)
        plot_combined(per_ds, args.data_root / "occupancy_drift_combined.png")
        print(f"\nWrote {args.data_root / 'occupancy_drift_summary.csv'} and "
              f"{args.data_root / 'occupancy_drift_combined.png'}")


if __name__ == "__main__":
    main()
