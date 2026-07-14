"""Categorize cluster-to-cluster transitions by their temporal behavior over the
MitoPark disease course.

Input
-----
`Cluster_detail_results.csv` with columns:
    ClusterIdx   integer behavioral-cluster label of each ~0.3 s frame/window
    Timestamp    frame time (only used to keep rows in recording order)
    Folder_Name  the weekly recording the frame belongs to (w8, w9, ... w24,
                 plus the w24 L-DOPA / saline treatment arms). Blank rows are
                 dropped frames and act as sequence breaks.

What this does
--------------
1. ORDERED PAIRS. For every weekly recording we read frames in order and emit the
   ordered transition (source -> target) for each *change* of cluster. Transitions
   are never formed across a week boundary or across a dropped (blank) frame, so we
   never invent a transition between two separate recordings. Self-loops (a -> a,
   i.e. staying in the same cluster) are excluded by default (--include-self keeps
   them).

2. ONE-WAY. (a -> b) is one-way if the reverse (b -> a) is never observed anywhere.
   We also report a directionality ratio  count(a->b) / (count(a->b)+count(b->a))
   so near-one-way transitions are visible too (1.0 = strictly one-way).

3. DEGENERATION OVER TIME. For each source cluster a we track, week by week, which
   targets it goes to and which target dominates. This surfaces the pattern you
   care about: a transition that starts as (a, b) but as disease progresses
   degenerates into (a, c), (a, d), ...  -- captured as the source's dominant
   target drifting and its target set gaining new late-disease members.

4. TEMPORAL CATEGORY. Each ordered transition is placed in one of:
       early       active in the early phase, gone before the late phase
       transient   a burst that comes and goes mid-course (mid and late bursts that
                   do not persist to the end are pooled here)
       late        emerges in the late phase and is still active at the end
       sustained   present across (most of) the whole course
   computed on the normal disease-progression weeks only; the L-DOPA / saline
   treatment arms are reported separately (they are interventions, not later time
   points). Thresholds are constants below and the raw per-transition metrics
   (onset/offset/span/prevalence) are written out so the labels can be retuned.

Outputs (written next to the input CSV, or --out-dir):
    cluster_transition_pairs.csv     one row per ordered (source,target): counts,
                                     one-way flag, directionality, temporal metrics,
                                     and the assigned category.
    cluster_transition_by_week.csv   long format week,source,target,count -- for
                                     heatmaps / plotting the time course.
    cluster_source_drift.csv         per source cluster: dominant-target sequence
                                     over weeks, whether it drifts, and the targets
                                     that are new in late disease.

Run:
    python cluster_transition_labels.py
    python cluster_transition_labels.py --csv data/1lc/Cluster_detail_results.csv
    python cluster_transition_labels.py --min-count 5 --include-self
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write JPEGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dataset_config
from utils import save_figure

# --- column names in the input CSV ------------------------------------------
IDX_COL = "ClusterIdx"
WEEK_COL = "Folder_Name"

DEFAULT_CSV = Path("data/1lc/Cluster_detail_results.csv")

# --- temporal-category thresholds (normalized timeline position in [0, 1]) ---
# The timeline is the ordered set of normal progression weeks; week i maps to
# pos = i / (n_weeks - 1), so the first week is 0.0 and the last is 1.0.
EARLY_BOUND = 1.0 / 3.0   # < this position counts as the "early" third
LATE_BOUND = 2.0 / 3.0    # >= this position counts as the "late" third
START_THRESH = 0.15       # onset <= this  => present at the very start
END_THRESH = 0.85         # offset >= this => still active at the very end

SUSTAINED_SPAN = 0.66     # span (offset-onset) >= this => spans most of the course
SUSTAINED_MIN_PREV = 0.40 # ...and present in >= this fraction of weeks => sustained
UBIQUITOUS_PREV = 0.80    # present in >= this fraction of weeks => sustained outright
LATE_PERSIST_SPAN = 0.20  # late onset spanning >= this stretch => "late" not "-transient"

CATEGORIES = ["early", "transient", "late", "sustained"]

# fixed colour per category, in early -> late narrative order
CATEGORY_COLORS = {
    "early": "#1f77b4",           # blue
    "transient": "#ff7f0e",       # orange
    "late": "#d62728",            # red
    "sustained": "#7f7f7f",       # grey
    "treatment-only": "#9467bd",  # purple
}


def week_sort_key(week):
    """Numeric ordering by the week number embedded in the label, with the
    saline/L-DOPA arms sorted just after their week. Mirrors the convention in
    cluster_sim_by_week.py so week ordering is consistent across the project."""
    match = re.search(r"\d+", str(week))
    num = float(match.group()) if match else float("inf")
    low = str(week).lower()
    if "saline" in low:
        num += 0.5
    elif "ldop" in low:
        num += 1
    return num


def is_variant(week):
    """True for the pharmacological arms (L-DOPA / saline), not normal weeks."""
    low = str(week).lower()
    return "ldop" in low or "saline" in low


def build_transitions(df, include_self=False):
    """Return a DataFrame of every ordered transition: columns week, source, target.

    Transitions are formed only between consecutive frames that share the same,
    non-blank week (so week boundaries and dropped frames break the sequence).
    """
    src = df[IDX_COL].to_numpy()
    tgt = np.roll(src, -1)
    week = df[WEEK_COL].to_numpy()
    next_week = np.roll(week, -1)

    # valid transition: not the wrap-around last row, same week, week not blank
    valid = np.ones(len(df), dtype=bool)
    valid[-1] = False
    valid &= (week == next_week)
    valid &= pd.notna(week)
    if not include_self:
        valid &= (src != tgt)

    return pd.DataFrame({
        "week": week[valid],
        "source": src[valid].astype(int),
        "target": tgt[valid].astype(int),
    })


def categorize(onset, offset, span, prevalence):
    """Map a transition's temporal footprint to one of CATEGORIES.

    onset/offset are the earliest/latest normalized week positions where the
    transition occurs; span = offset - onset; prevalence = fraction of progression
    weeks in which it occurs. Rules are applied in priority order."""
    reaches_start = onset <= START_THRESH
    reaches_end = offset >= END_THRESH
    mid = 0.5 * (onset + offset)

    # 1. Sustained: spans most of the course, or is present almost everywhere.
    if (span >= SUSTAINED_SPAN and prevalence >= SUSTAINED_MIN_PREV) \
            or prevalence >= UBIQUITOUS_PREV:
        return "sustained"

    # 2. Late: onset in the late third and still active at the end of the course.
    if onset >= LATE_BOUND and reaches_end:
        return "late" if span >= LATE_PERSIST_SPAN else "transient"

    # 3. Early: present at the start and gone before the late phase.
    if reaches_start and offset < LATE_BOUND:
        return "early"

    # 4. Otherwise a localized burst that comes and goes; its center decides only
    #    early vs the merged transient class (mid and late bursts are one group).
    if mid < EARLY_BOUND:
        return "early"
    return "transient"


def summarize_pairs(trans, progression_weeks, min_count):
    """One row per ordered (source, target): counts, one-way flag, directionality,
    temporal metrics, and the assigned category."""
    n_weeks = len(progression_weeks)
    pos = {w: (i / (n_weeks - 1) if n_weeks > 1 else 0.0)
           for i, w in enumerate(progression_weeks)}
    prog_set = set(progression_weeks)

    # total counts per ordered pair (across all weeks incl. treatment arms)
    totals = (trans.groupby(["source", "target"]).size()
              .rename("total_count").reset_index())
    # count_map[(a, b)] = number of a -> b transitions; reverse is count_map[(b, a)]
    count_map = {(int(s), int(t)): int(c)
                 for s, t, c in totals.itertuples(index=False)}

    # per-pair, per-week counts restricted to the progression timeline
    prog = trans[trans["week"].isin(prog_set)]
    by_week = prog.groupby(["source", "target", "week"]).size()

    rows = []
    for s, t, total in totals.itertuples(index=False):
        rev_count = count_map.get((int(t), int(s)), 0)
        directionality = total / (total + rev_count) if (total + rev_count) else np.nan

        # temporal footprint on the progression timeline
        try:
            wk_counts = by_week.loc[(s, t)]
            positions = sorted(pos[w] for w in wk_counts.index)
        except KeyError:
            positions = []

        if positions:
            onset, offset = positions[0], positions[-1]
            span = offset - onset
            prevalence = len(positions) / n_weeks
            category = categorize(onset, offset, span, prevalence)
            prog_count = int(wk_counts.sum())
        else:
            # transition seen only in the treatment arms, never in normal weeks
            onset = offset = span = prevalence = np.nan
            category = "treatment-only"
            prog_count = 0

        rows.append({
            "source": int(s),
            "target": int(t),
            "total_count": int(total),
            "progression_count": prog_count,
            "reverse_count": rev_count,
            "is_one_way": rev_count == 0,
            "directionality": round(directionality, 3) if pd.notna(directionality) else np.nan,
            "n_weeks_present": len(positions),
            "first_week": progression_weeks[int(round(onset * (n_weeks - 1)))] if positions else "",
            "last_week": progression_weeks[int(round(offset * (n_weeks - 1)))] if positions else "",
            "onset_pos": round(onset, 3) if positions else np.nan,
            "offset_pos": round(offset, 3) if positions else np.nan,
            "span": round(span, 3) if positions else np.nan,
            "prevalence": round(prevalence, 3) if positions else np.nan,
            "category": category,
        })

    pairs = pd.DataFrame(rows)
    pairs = pairs[pairs["total_count"] >= min_count]
    return pairs.sort_values(["source", "target"]).reset_index(drop=True)


def source_drift(trans, progression_weeks):
    """Per source cluster: how its set of targets and its dominant target evolve
    over disease progression -- the (a,b) -> (a,c),(a,d) degeneration."""
    n_weeks = len(progression_weeks)
    early_weeks = set(progression_weeks[: max(1, n_weeks // 3)])
    late_weeks = set(progression_weeks[-max(1, n_weeks // 3):])
    prog = trans[trans["week"].isin(set(progression_weeks))]

    rows = []
    for source, g in prog.groupby("source"):
        # dominant target within each week, in week order
        dom_seq = []
        for w in progression_weeks:
            gw = g[g["week"] == w]
            if len(gw):
                dom_seq.append((w, int(gw["target"].value_counts().idxmax())))
        dom_targets = [t for _, t in dom_seq]

        early_t = set(g[g["week"].isin(early_weeks)]["target"].astype(int))
        late_t = set(g[g["week"].isin(late_weeks)]["target"].astype(int))

        rows.append({
            "source": int(source),
            "n_distinct_targets": int(g["target"].nunique()),
            "n_distinct_dominant": len(set(dom_targets)),
            "dominant_drifts": len(set(dom_targets)) > 1,
            "dominant_sequence": " -> ".join(str(t) for t in dom_targets),
            "early_targets": ",".join(map(str, sorted(early_t))),
            "late_targets": ",".join(map(str, sorted(late_t))),
            "new_late_targets": ",".join(map(str, sorted(late_t - early_t))),
            "lost_early_targets": ",".join(map(str, sorted(early_t - late_t))),
        })
    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)


def make_plots(trans, pairs, progression_weeks, out_dir, per_cat=15):
    """Write the time-course figures. Returns the list of paths written."""
    weeks = progression_weeks
    wk_idx = {w: i for i, w in enumerate(weeks)}
    prog = trans[trans["week"].isin(set(weeks))].copy()
    # attach each observed transition's category
    cat_of = {(r.source, r.target): r.category
              for r in pairs.itertuples(index=False)}
    prog["category"] = [cat_of.get((s, t), "treatment-only")
                        for s, t in zip(prog["source"], prog["target"])]

    # per (week, category): how many DISTINCT transitions are active that week
    active = (prog.groupby(["week", "category"])[["source", "target"]]
              .apply(lambda g: len(g.drop_duplicates()))
              .rename("n").reset_index())
    pivot_active = (active.pivot(index="week", columns="category", values="n")
                    .reindex(weeks).fillna(0))

    written = []

    # --- Figure 1: active transitions per week, by category --------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(weeks))
    for cat in CATEGORIES:
        if cat in pivot_active:
            axes[0].plot(x, pivot_active[cat], marker="o", ms=4,
                         color=CATEGORY_COLORS[cat], label=cat)
    axes[0].set_xticks(x); axes[0].set_xticklabels(weeks, rotation=45, ha="right")
    axes[0].set_ylabel("distinct transitions active")
    axes[0].set_title("Active transitions per week, by temporal category")
    axes[0].legend(fontsize=8)

    # 100% stacked composition of active transitions per week
    cats_present = [c for c in CATEGORIES if c in pivot_active]
    frac = pivot_active[cats_present]
    frac = frac.div(frac.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    bottom = np.zeros(len(weeks))
    for cat in cats_present:
        axes[1].bar(x, frac[cat], bottom=bottom, color=CATEGORY_COLORS[cat],
                    label=cat, width=0.85)
        bottom += frac[cat].to_numpy()
    axes[1].set_xticks(x); axes[1].set_xticklabels(weeks, rotation=45, ha="right")
    axes[1].set_ylabel("fraction of active transitions")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Category composition of the repertoire over time")
    fig.tight_layout()
    p = out_dir / "transition_categories_by_week.jpeg"
    save_figure(fig, p, dpi=150); plt.close(fig); written.append(p)

    # --- Figure 2: presence heatmap of a sample of transitions, banded by cat --
    # take the top `per_cat` transitions by count within each category so every
    # band is visible (otherwise the abundant 'sustained' rows dominate)
    chosen, band_edges, band_labels = [], [], []
    for cat in CATEGORIES:
        sub = (pairs[pairs["category"] == cat]
               .sort_values("progression_count", ascending=False).head(per_cat))
        sub = sub.sort_values("onset_pos")
        if len(sub):
            band_edges.append(len(chosen))
            band_labels.append(cat)
            chosen.extend((r.source, r.target) for r in sub.itertuples(index=False))

    if chosen:
        counts = (prog.groupby(["source", "target", "week"]).size()
                  .rename("n").reset_index())
        mat = np.zeros((len(chosen), len(weeks)))
        row_of = {st: i for i, st in enumerate(chosen)}
        for r in counts.itertuples(index=False):
            st = (r.source, r.target)
            if st in row_of:
                mat[row_of[st], wk_idx[r.week]] = r.n
        # row-normalize so each row shows WHEN it is active, not how often
        norm = mat / mat.max(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(10, max(6, len(chosen) * 0.16)))
        ax.imshow(norm, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_xticks(np.arange(len(weeks)))
        ax.set_xticklabels(weeks, rotation=45, ha="right")
        ax.set_yticks([]); ax.set_ylabel("transitions (grouped by category)")
        ax.set_title("When each transition is active\n"
                     "(row-normalized per-week count; top "
                     f"{per_cat} per category)")
        # category band separators + labels down the left edge
        for e in band_edges[1:]:
            ax.axhline(e - 0.5, color="white", lw=1.2)
        for start, lab in zip(band_edges, band_labels):
            ax.text(-0.6, start, lab, ha="right", va="top", fontsize=8,
                    color=CATEGORY_COLORS[lab], fontweight="bold")
        fig.tight_layout()
        p = out_dir / "transition_presence_heatmap.jpeg"
        save_figure(fig, p, dpi=150); plt.close(fig); written.append(p)

    # --- Figure 3: target fan-out per source over time (degeneration) ----------
    fan = (prog.groupby(["source", "week"])["target"].nunique()
           .rename("n_targets").reset_index())
    fan_pivot = (fan.pivot(index="source", columns="week", values="n_targets")
                 .reindex(columns=weeks))
    mean_fan = fan_pivot.mean(axis=0)
    # the sources whose target set expands most from early to late
    n_third = max(1, len(weeks) // 3)
    early_fan = fan_pivot[weeks[:n_third]].mean(axis=1)
    late_fan = fan_pivot[weeks[-n_third:]].mean(axis=1)
    growth = (late_fan - early_fan).sort_values(ascending=False)
    top_sources = growth.head(8).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(x, mean_fan.to_numpy(), marker="o", color="black", lw=2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(weeks, rotation=45, ha="right")
    axes[0].set_ylabel("mean distinct targets per source")
    axes[0].set_title("Behavioural fan-out over disease course\n"
                      "(does each state branch to more successors?)")
    for s in top_sources:
        axes[1].plot(x, fan_pivot.loc[s].to_numpy(), marker=".", label=f"cluster {s}")
    axes[1].set_xticks(x); axes[1].set_xticklabels(weeks, rotation=45, ha="right")
    axes[1].set_ylabel("distinct targets")
    axes[1].set_title("Sources with the largest early->late target expansion")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = out_dir / "transition_target_fanout.jpeg"
    save_figure(fig, p, dpi=150); plt.close(fig); written.append(p)

    return written


def process(csv, out_dir, min_count, include_self, no_plots):
    """Run the transition-label analysis for one dataset CSV, writing to out_dir
    (defaults to the CSV's own folder)."""
    csv = Path(csv)
    out_dir = Path(out_dir) if out_dir else csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    df = df.reset_index(drop=True)

    trans = build_transitions(df, include_self=include_self)

    # ordered week timelines
    all_weeks = sorted(df[WEEK_COL].dropna().unique(), key=week_sort_key)
    progression_weeks = [w for w in all_weeks if not is_variant(w)]
    treatment_weeks = [w for w in all_weeks if is_variant(w)]

    print(f"Loaded {len(df):,} frames; {len(trans):,} ordered transitions "
          f"({'incl.' if include_self else 'excl.'} self-loops).")
    print(f"Progression weeks ({len(progression_weeks)}): {progression_weeks}")
    if treatment_weeks:
        print(f"Treatment arms (reported, not in timeline): {treatment_weeks}")

    pairs = summarize_pairs(trans, progression_weeks, min_count)
    drift = source_drift(trans, progression_weeks)

    by_week = (trans.groupby(["week", "source", "target"]).size()
               .rename("count").reset_index())
    by_week["__k"] = by_week["week"].map(week_sort_key)
    by_week = by_week.sort_values(["__k", "source", "target"]).drop(columns="__k")

    p_pairs = out_dir / "cluster_transition_pairs.csv"
    p_week = out_dir / "cluster_transition_by_week.csv"
    p_drift = out_dir / "cluster_source_drift.csv"
    pairs.to_csv(p_pairs, index=False)
    by_week.to_csv(p_week, index=False)
    drift.to_csv(p_drift, index=False)

    # ----- console summary -----
    print(f"\n{len(pairs):,} distinct ordered transitions "
          f"(min-count {min_count}).")
    n_one_way = int(pairs["is_one_way"].sum())
    print(f"One-way transitions (reverse never seen): {n_one_way:,} "
          f"({100 * n_one_way / max(1, len(pairs)):.0f}%).")

    print("\nTemporal categories:")
    counts = pairs["category"].value_counts()
    for cat in CATEGORIES + ["treatment-only"]:
        if cat in counts:
            print(f"  {cat:<15} {counts[cat]:>5}")

    drifters = drift[drift["dominant_drifts"]]
    print(f"\nSource clusters whose dominant target drifts over time: "
          f"{len(drifters)}/{len(drift)}.")
    gained = drift[drift["new_late_targets"] != ""]
    print(f"Source clusters that gain new targets in late disease "
          f"(the (a,b)->(a,c),(a,d) pattern): {len(gained)}.")
    if len(gained):
        print("Top examples (most new late-disease targets):")
        gained = gained.copy()
        gained["__n"] = gained["new_late_targets"].str.count(",") + 1
        for r in gained.sort_values("__n", ascending=False).head(8).itertuples():
            print(f"  cluster {r.source}: dominant {r.dominant_sequence} | "
                  f"new late targets: {r.new_late_targets}")

    print(f"\nWrote:\n  {p_pairs}\n  {p_week}\n  {p_drift}")

    if not no_plots:
        for p in make_plots(trans, pairs, progression_weeks, out_dir):
            print(f"  {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=None,
                    help=f"single Cluster_detail_results.csv (default: {DEFAULT_CSV}); "
                         "ignored when --data-root/--datasets/--dataset-glob select a cohort")
    dataset_config.add_dataset_args(ap)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write outputs (default: alongside each CSV; in "
                         "cohort mode a given --out-dir gets one <dataset>/ subdir each)")
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop ordered pairs seen fewer than this many times total")
    ap.add_argument("--include-self", action="store_true",
                    help="keep self-loops (a -> a)")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip writing the time-course figures")
    args = ap.parse_args()

    # Cohort mode: loop over every selected dataset under the data root. Single
    # mode: one --csv (back-compat; default data/1lc/...).
    cohort = args.data_root or args.datasets or args.dataset_glob
    if cohort:
        root, datasets = dataset_config.resolve_datasets(args)
        if not datasets:
            raise SystemExit(f"no datasets found under {root}")
        print(f"data root: {root}   datasets: {', '.join(datasets)}\n")
        for name in datasets:
            print(f"{'='*70}\n{name}\n{'='*70}")
            out_dir = (args.out_dir / name) if args.out_dir else None
            process(root / name / dataset_config.CSV_NAME, out_dir,
                    args.min_count, args.include_self, args.no_plots)
            print()
    else:
        process(args.csv or DEFAULT_CSV, args.out_dir,
                args.min_count, args.include_self, args.no_plots)


if __name__ == "__main__":
    main()
