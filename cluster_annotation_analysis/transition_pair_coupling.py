"""Category-to-category transition coupling over weeks, with error bands and a
Spearman test.

Given a focal category A and a partner category B (behavior categories by
default, or phase with ``--by phase``), this averages, across the clusters that
make up A, each cluster's within-week probability of

  * going  TO   B:   P(a → B) = of a's outgoing transitions that week, the
                     fraction that land in a cluster of category B, and
  * coming FROM B:   P(prev = B | a) = of a's incoming transitions that week, the
                     fraction that started in category B,

and draws both as lines across the progression weeks with a shaded ±SEM band
(spread over A's clusters). A Spearman correlation between the two weekly mean
series -- does A's outflow-to-B track its inflow-from-B over the disease
timeline? -- is reported as rho and p in the panel.

"Moving": each cluster's probability is pooled over a centred ``--window``
(default 3 weeks) at the count level before dividing, matching
``transition_category_lines.py``. Self-transitions are dropped; only the
progression weeks (week_8..week_24) are used.

Run:  uv run python cluster_annotation_analysis/transition_pair_coupling.py                       # ascent <-> rearing
      uv run python cluster_annotation_analysis/transition_pair_coupling.py --cat ascent --other immobile
      uv run python cluster_annotation_analysis/transition_pair_coupling.py --by phase --cat early --other late
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # repo root
from dataset_config import data_root          # noqa: E402
from utils import save_figure                 # noqa: E402
from presence_heatmap import BEHAVIOR_ORDER, PHASE_ORDER      # noqa: E402
from transition_flow import load_annotation, make_meta          # noqa: E402
from transition_category_lines import progression_weeks         # noqa: E402

TO_COLOR = "#1f77b4"       # A -> B  (outflow to partner)
FROM_COLOR = "#d62728"     # B -> A  (inflow from partner)


def cluster_probs(df, a, partner_ids, direction, weeks, window):
    """One cluster's within-week probability series toward/from a partner set.

    direction 'to'  -> P(a → partner) among a's outgoing that week;
    direction 'from'-> P(partner → a) among a's incoming that week.
    Count-pooled over a centred `window`; NaN where a has no transitions that
    week (the denominator is empty).
    """
    if direction == "to":
        num_e = df[(df["source"] == a) & (df["target"].isin(partner_ids))]
        den_e = df[df["source"] == a]
    else:
        num_e = df[(df["target"] == a) & (df["source"].isin(partner_ids))]
        den_e = df[df["target"] == a]
    num = num_e.groupby("week")["count"].sum().reindex(weeks).fillna(0.0)
    den = den_e.groupby("week")["count"].sum().reindex(weeks).fillna(0.0)
    num = num.rolling(window, center=True, min_periods=1).sum()
    den = den.rolling(window, center=True, min_periods=1).sum()
    return num.div(den.where(den > 0, other=float("nan")))


def category_band(df, focal_ids, partner_ids, direction, weeks, window):
    """Mean and SEM across `focal_ids` of the per-cluster probability series."""
    mat = pd.DataFrame({a: cluster_probs(df, a, partner_ids, direction, weeks, window)
                        for a in focal_ids})
    mean = mat.mean(axis=1, skipna=True)
    n = mat.notna().sum(axis=1)
    sem = mat.std(axis=1, ddof=0, skipna=True) / np.sqrt(n.where(n > 0))
    return mean, sem, n


def plot_coupling(df, meta, by, cat, other, weeks, window, path):
    focal_ids = meta.index[meta[by] == cat].tolist()
    partner_ids = meta.index[meta[by] == other].tolist()
    if not focal_ids or not partner_ids:
        raise SystemExit(f"empty category: {cat} has {len(focal_ids)}, "
                         f"{other} has {len(partner_ids)} clusters")

    to_mean, to_sem, _ = category_band(df, focal_ids, partner_ids, "to", weeks, window)
    fr_mean, fr_sem, _ = category_band(df, focal_ids, partner_ids, "from", weeks, window)
    xs = np.array([int(w.split("_")[1]) for w in weeks])

    both = to_mean.notna() & fr_mean.notna()
    if both.sum() >= 3:
        rho, p = spearmanr(to_mean[both], fr_mean[both])
    else:
        rho, p = float("nan"), float("nan")

    fig, ax = plt.subplots(figsize=(10, 6))
    for mean, sem, color, lbl in [
            (to_mean, to_sem, TO_COLOR, f"{cat} → {other}   (outflow to {other})"),
            (fr_mean, fr_sem, FROM_COLOR, f"{other} → {cat}   (inflow from {other})")]:
        m = mean.to_numpy(float)
        s = sem.to_numpy(float)
        fin = np.isfinite(m)
        ax.plot(xs, m, color=color, lw=2.4, marker="o", ms=4, label=lbl)
        ax.fill_between(xs, m - s, m + s, where=fin & np.isfinite(s),
                        color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel("week")
    ax.set_ylabel(f"moving P (mean ± SEM over {cat} clusters)")
    ax.set_ylim(0, None)
    ax.set_xticks(xs)
    ax.margins(x=0.01)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # category swatches + counts + Spearman, in a corner box
    p_txt = "n/a" if np.isnan(p) else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")
    rho_txt = "n/a" if np.isnan(rho) else f"{rho:+.3f}"
    ax.text(0.98, 0.97,
            f"Spearman(to, from) over weeks\nρ = {rho_txt}   p = {p_txt}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888888"))
    fig.suptitle(f"{by} coupling:  {cat}  ⇄  {other}   "
                 f"({len(focal_ids)} × {len(partner_ids)} clusters, "
                 f"{window}-week moving)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rho, p, len(focal_ids), len(partner_ids)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", default="1mp")
    ap.add_argument("--by", choices=["behavior", "phase"], default="behavior")
    ap.add_argument("--cat", default="ascent", help="focal category A")
    ap.add_argument("--other", default="rearing", help="partner category B")
    ap.add_argument("--window", type=int, default=3)
    args = ap.parse_args()

    order = BEHAVIOR_ORDER if args.by == "behavior" else PHASE_ORDER
    for c in (args.cat, args.other):
        if c not in order:
            raise SystemExit(f"'{c}' is not a {args.by} category; choose from {order}")

    annot = load_annotation(HERE / "data" / f"{args.mouse}.json")
    df = pd.read_csv(data_root() / args.mouse / "cluster_transition_by_week.csv")
    df = df[df["source"] != df["target"]]
    weeks = progression_weeks(df)
    meta = make_meta(annot, sorted(set(df["source"]) | set(df["target"])))

    out = HERE / "output" / "transition_pair_coupling"
    out.mkdir(parents=True, exist_ok=True)
    rho, p, na, nb = plot_coupling(
        df, meta, args.by, args.cat, args.other, weeks, args.window,
        out / f"{args.mouse}_{args.by}_{args.cat}_x_{args.other}.jpeg")
    print(f"{args.mouse}: {args.cat}({na}) <-> {args.other}({nb}) [{args.by}]  "
          f"Spearman rho={rho:.3f} p={p:.3g}")
    print(f"  wrote coupling plot to {out}")


if __name__ == "__main__":
    main()
