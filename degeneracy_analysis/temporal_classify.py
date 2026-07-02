"""Classify clusters by WHEN they appear over weeks, then absorb uncategorized
clusters into categories via feature-similarity grouping.

Part 1 -- temporal category (per cluster, from shape-normalized weekly presence):
    early     descending ramp  (front-loaded, fades)
    late      ascending ramp
    mid       centered Gaussian bump (transient: ~0 at both ends)
    sustained flat              (present beginning -> end)
  Assign argmax cosine to these templates, but only if (best - 2nd) >= MARGIN;
  otherwise 'uncategorized'. early/mid/late mass fractions are dumped too.

Part 2 -- feature grouping + merge:
  Agglomerative clustering (average linkage) on the cluster x cluster feature
  distance Dfeat, distance_threshold = GROUP_PCT-th percentile of off-diagonal
  Dfeat (tight -> mostly singletons + small groups). Any 'uncategorized' cluster
  whose group contains categorized members inherits the group's MAJORITY category
  (marked merged).

Plot -- sectioned heatmap: rows=clusters (color = row-normalized weekly presence,
  so temporal SHAPE shows), sectioned by final category, within-section ordered by
  centroid; left strip = feature-group id; merged rows flagged red.

Run (after presence_similarity.py + feature_similarity.py):
    C:/ProgramData/anaconda3/python.exe degeneracy_analysis/temporal_classify.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, ListedColormap
from matplotlib.patches import Patch
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extraction import load_funct_features, bin_features, FEATURE_NAMES
from feature_similarity import _row_labels
from presence_similarity import MICE, OUT, DATA, MIN_COUNT, load_counts

CATS = ["early", "mid", "late", "sustained"]
MARGIN = 0.03        # min (best - 2nd) cosine to accept a category
GROUP_PCT = 10       # Dfeat percentile used as agglomerative distance threshold
RESTING_TBA = 0.10   # mean total body accel below this => low TBA
RESTING_GYRO = 8.0   # mean |gyro| below this => low rotation
SECTION_ORDER = ["early", "mid", "late", "sustained", "uncategorized"]
CAT_RANK = {c: i for i, c in enumerate(SECTION_ORDER)}
CAT_COLORS = {"early": "#d62728", "mid": "#ff7f0e", "late": "#1f77b4",
              "sustained": "#2ca02c", "uncategorized": "#bbbbbb"}

# rest strip decomposes movement into translational (TBA) vs rotational (gyro):
# 0 white = hi TBA & hi gyro (vigorous mixed); 1 green = hi TBA & lo gyro
# (translational); 2 red = lo TBA & hi gyro (rotational, rare); 3 navy = lo both
# (immobile). Order matches the code produced in analyze().
MOVE_COLORS = ["#ffffff", "#2ca02c", "#d62728", "#11224a"]
MOVE_LABELS = ["hiTBA hiGyro (vigorous)", "hiTBA loGyro (translational)",
               "loTBA hiGyro (rotational)", "loTBA loGyro (immobile)"]
MOVE_TYPES = np.array(["vigorous", "translational", "rotational", "immobile"])


def rest_code(TBA, gyro):
    """0 white / 1 green / 2 red / 3 navy from the TBA x gyro quadrants."""
    hi_t, hi_g = TBA >= RESTING_TBA, gyro >= RESTING_GYRO
    return np.where(hi_t, np.where(hi_g, 0, 1), np.where(hi_g, 2, 3))


def _templates(weeks):
    t = (weeks - weeks.min()) / (weeks.max() - weeks.min())    # 0..1
    T = {
        "early": 1 - t,
        "late": t,
        "mid": np.exp(-((t - 0.5) / 0.18) ** 2),
        "sustained": np.ones_like(t, float),
    }
    return {k: v / np.linalg.norm(v) for k, v in T.items()}


def classify(P, weeks, margin=MARGIN):
    """Return (labels, cosine table K x 4, margins)."""
    T = _templates(weeks)
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    S = np.stack([Pn @ T[c] for c in CATS], axis=1)            # K x 4
    order = np.argsort(-S, axis=1)
    rows = np.arange(len(S))
    best, second = order[:, 0], order[:, 1]
    margins = S[rows, best] - S[rows, second]
    labels = np.array([CATS[b] for b in best], dtype=object)
    labels[margins < margin] = "uncategorized"
    return labels, S, margins


def feature_groups(Dfeat, pct=GROUP_PCT):
    off = Dfeat[~np.eye(len(Dfeat), dtype=bool)]
    thr = np.percentile(off, pct)
    D = Dfeat.copy()
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    g = AgglomerativeClustering(n_clusters=None, distance_threshold=thr,
                                metric="precomputed", linkage="average").fit_predict(D)
    return g, thr


def per_cluster_func(mouse, clusters, weeks):
    """Mean IMU func-feature signature per cluster: TBA (linear total body accel)
    and the per-axis movement magnitudes (mean |accel|/|gyro|), aligned to the
    kept clusters. Used for the resting flag and the commonality search."""
    lab = _row_labels(mouse, clusters, weeks)
    binned = bin_features(load_funct_features(f"{DATA}/{mouse}/session_1_out.mat"))
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    K = len(clusters)

    def per(vals, absval=False):
        v = np.abs(vals) if absval else vals
        return np.array([v[lab == k].mean() for k in range(K)])

    return {
        "TBA": per(binned[idx["TotAccelBA"]]),
        "ap_accel": per(binned[idx["anterior_posterior_x_accel"]], True),
        "dv_accel": per(binned[idx["dorsal_ventral_y_accel"]], True),
        "gyro": per(binned[idx["y_gyro"]], True),
    }


def kept_counts(mouse, clusters, weeks):
    """Raw (cluster x week) observation counts for the kept clusters, aligned to
    presence.npz order -- so histograms can be *summed* across a feature-family."""
    counts, cl_all, wk, _, _ = load_counts(mouse)
    keep = counts.sum(axis=1) >= MIN_COUNT
    counts_k, cl_k = counts[keep], cl_all[keep]
    if not (np.array_equal(cl_k, clusters) and np.array_equal(wk, weeks)):
        raise ValueError(f"{mouse}: kept counts misaligned with presence.npz")
    return counts_k


def group_reclassify(labels, groups, counts_k, weeks, margin=MARGIN):
    """Within each multi-member feature-group, SUM the member presence histograms
    and reclassify the pooled profile; every member takes that combined label.
    A cluster whose individual label != its family's combined label is 'changed'
    (an evolution candidate); 'strong' = both labels are real categories and
    differ (e.g. early -> late), the cleanest temporal-shift signal."""
    K = len(labels)
    final = labels.copy()
    changed = np.zeros(K, bool)
    strong = np.zeros(K, bool)
    for gid in np.unique(groups):
        idx = np.where(groups == gid)[0]
        if idx.size < 2:
            continue                                   # singleton keeps its own label
        pooled = counts_k[idx].sum(axis=0)
        if pooled.sum() == 0:
            continue
        comb = classify((pooled / pooled.sum())[None, :], weeks, margin)[0][0]
        for i in idx:
            final[i] = comb
            if labels[i] != comb:
                changed[i] = True
                strong[i] = labels[i] in CATS and comb in CATS
    return final, changed, strong


def _rest_strip(axr, code, ro):
    axr.imshow(code[ro][:, None], aspect="auto", vmin=-0.5, vmax=3.5,
               cmap=ListedColormap(MOVE_COLORS))
    axr.set(xticks=[], yticks=[], title="T·G")
    axr.title.set_fontsize(7)


def _move_legend(fig):
    fig.legend(handles=[Patch(facecolor=MOVE_COLORS[i], edgecolor="0.3",
                              label=MOVE_LABELS[i]) for i in range(4)],
               loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2,
               fontsize=6, title="rest strip = TBA × gyro", title_fontsize=7)


def _section_heatmap(mouse, P, weeks, clusters, labels, groups, merged, code, path, title):
    disp = P / P.max(axis=1, keepdims=True)          # row-max norm -> shape visible
    cen = P @ weeks

    row_order, bounds, ypos = [], [], []
    for c in SECTION_ORDER:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        idx = idx[np.argsort(cen[idx])]
        ypos.append((c, len(row_order) + idx.size / 2))
        row_order.extend(idx)
        bounds.append(len(row_order))
    ro = np.array(row_order)

    fig, (axr, axg, ax) = plt.subplots(
        1, 3, figsize=(8.9, max(6, len(ro) * 0.11)),
        gridspec_kw={"width_ratios": [0.035, 0.05, 1], "wspace": 0.02})
    _rest_strip(axr, code, ro)

    # feature-group strip (recolor group ids to a compact cyclic palette)
    uniq = {g: i for i, g in enumerate(np.unique(groups))}
    gstrip = np.array([uniq[groups[i]] for i in ro])[:, None]
    axg.imshow(gstrip % 20, aspect="auto", cmap="tab20")
    axg.set(xticks=[], yticks=[], title="grp")
    axg.title.set_fontsize(7)

    im = ax.imshow(disp[ro], aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(0, len(weeks), 2))
    ax.set_xticklabels(weeks[::2])
    ax.set_yticks(range(len(ro)))
    ax.set_yticklabels(clusters[ro], fontsize=5)
    for tick, i in zip(ax.get_yticklabels(), ro):
        if merged[i]:
            tick.set_color("red")
            tick.set_fontweight("bold")
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="cyan", lw=1.2)
    for c, yc in ypos:
        ax.text(len(weeks) + 0.6, yc, c, va="center", fontsize=8, rotation=90)
    ax.set(xlabel="week", title=title)
    fig.colorbar(im, ax=ax, label="presence (row-max normalized)", pad=0.12, shrink=0.6)
    _move_legend(fig)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _group_heatmap(mouse, P, weeks, clusters, final, groups, merged, code, path, title):
    """Rows sectioned by feature-group (multi-member first, ordered by dominant
    category), within-group ordered early->mid->late->sustained, then a singleton
    block. Left strip = temporal category color -> a coherent group is one color."""
    disp = P / P.max(axis=1, keepdims=True)
    cen = P @ weeks
    gids = np.unique(groups)
    sizes = {g: int((groups == g).sum()) for g in gids}

    def dominant(g):
        labs = [final[i] for i in np.where(groups == g)[0]]
        vals, cnts = np.unique(labs, return_counts=True)
        return sorted(zip(vals, cnts), key=lambda vc: (-vc[1], CAT_RANK[vc[0]]))[0][0]

    def order_within(idx):
        cr = np.array([CAT_RANK[final[i]] for i in idx])
        return idx[np.lexsort((cen[idx], cr))]        # primary=category, then centroid

    multi = sorted([g for g in gids if sizes[g] >= 2],
                   key=lambda g: (CAT_RANK[dominant(g)], -sizes[g]))
    singles = [g for g in gids if sizes[g] == 1]

    row_order, sections = [], []          # sections: (label, center_row, end_row)
    for g in multi:
        idx = order_within(np.where(groups == g)[0])
        start = len(row_order)
        row_order.extend(idx)
        sections.append((f"g{g}·n{sizes[g]}", (start + len(row_order)) / 2 - 0.5,
                         len(row_order)))
    if singles:
        idx = order_within(np.concatenate([np.where(groups == g)[0] for g in singles]))
        start = len(row_order)
        row_order.extend(idx)
        sections.append((f"singletons·n{len(idx)}", (start + len(row_order)) / 2 - 0.5,
                         len(row_order)))
    ro = np.array(row_order)

    strip = np.array([to_rgb(CAT_COLORS[final[i]]) for i in ro])[:, None, :]
    fig, (axr, axc, ax) = plt.subplots(
        1, 3, figsize=(8.9, max(6, len(ro) * 0.12)),
        gridspec_kw={"width_ratios": [0.035, 0.045, 1], "wspace": 0.02})
    _rest_strip(axr, code, ro)
    axc.imshow(strip, aspect="auto")
    axc.set(xticks=[], yticks=[], title="cat")
    axc.title.set_fontsize(7)

    im = ax.imshow(disp[ro], aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(0, len(weeks), 2))
    ax.set_xticklabels(weeks[::2])
    ax.set_yticks(range(len(ro)))
    ax.set_yticklabels(clusters[ro], fontsize=5)
    for tick, i in zip(ax.get_yticklabels(), ro):
        if merged[i]:
            tick.set_color("red")
            tick.set_fontweight("bold")
    for _, _, end in sections[:-1]:
        ax.axhline(end - 0.5, color="cyan", lw=0.8)
    for lab, center, _ in sections:
        ax.text(len(weeks) + 0.4, center, lab, va="center", fontsize=6)
    ax.set(xlabel="week", title=title)
    ax.legend(handles=[Patch(color=CAT_COLORS[c], label=c) for c in SECTION_ORDER],
              bbox_to_anchor=(1.16, 1), loc="upper left", fontsize=6, title="category")
    fig.colorbar(im, ax=ax, label="presence (row-max normalized)", pad=0.22, shrink=0.5)
    _move_legend(fig)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def analyze(mouse):
    pres = np.load(f"{OUT}/{mouse}/presence.npz", allow_pickle=True)
    feat = np.load(f"{OUT}/{mouse}/feature.npz", allow_pickle=True)
    clusters = pres["clusters"].astype(int)
    weeks = pres["weeks"].astype(int)
    P = pres["presence"]
    Dfeat = feat["Dfeat"]

    labels, S, margins = classify(P, weeks)
    groups, thr = feature_groups(Dfeat)
    counts_k = kept_counts(mouse, clusters, weeks)
    final, changed, strong = group_reclassify(labels, groups, counts_k, weeks)
    cen = P @ weeks
    func = per_cluster_func(mouse, clusters, weeks)
    resting = func["TBA"] <= RESTING_TBA
    code = rest_code(func["TBA"], func["gyro"])
    size = counts_k.sum(axis=1)

    # thirds mass fractions for transparency
    e, m, l = [b.sum(1) for b in np.array_split(P, 3, axis=1)]
    df = pd.DataFrame({
        "cluster": clusters, "label": labels, "combined_label": final,
        "changed": changed, "strong": strong, "resting": resting,
        "feature_group": groups, "centroid_week": cen, "margin": margins,
        "size": size, "TBA": func["TBA"], "ap_accel": func["ap_accel"],
        "dv_accel": func["dv_accel"], "gyro": func["gyro"],
        "move_type": MOVE_TYPES[code],
        "frac_early": e, "frac_mid": m, "frac_late": l,
        **{f"cos_{c}": S[:, i] for i, c in enumerate(CATS)},
    })
    df.to_csv(f"{OUT}/{mouse}/temporal_classes.csv", index=False)

    # evolution candidates: clusters whose class changes once pooled with its
    # feature-family. Group context is included to hunt for commonalities.
    rows = []
    for i in np.where(changed)[0]:
        mem = np.where(groups == groups[i])[0]
        rows.append({
            "cluster": clusters[i], "feature_group": int(groups[i]),
            "individual_label": labels[i], "combined_label": final[i],
            "strong": bool(strong[i]), "group_size": mem.size,
            "centroid_week": round(float(cen[i]), 2),
            "group_centroid_span": round(float(cen[mem].max() - cen[mem].min()), 2),
            "group_members": " ".join(map(str, clusters[mem])),
            "member_labels": " ".join(labels[mem]),
        })
    evo = pd.DataFrame(rows)
    if not evo.empty:
        evo = evo.sort_values(["strong", "group_centroid_span"], ascending=False)
    evo.to_csv(f"{OUT}/{mouse}/evolution_candidates.csv", index=False)

    grp = "control" if mouse.endswith("lc") else "MitoPark"
    noflag = np.zeros(len(clusters), bool)
    _section_heatmap(mouse, P, weeks, clusters, labels, groups, noflag, code,
                     f"{OUT}/{mouse}/temporal_classes.png",
                     f"{mouse} ({grp}): temporal classes (individual)")
    _section_heatmap(mouse, P, weeks, clusters, final, groups, changed, code,
                     f"{OUT}/{mouse}/temporal_classes_merged.png",
                     f"{mouse} ({grp}): pooled-family reclassify (red label = evolution candidate)")
    _group_heatmap(mouse, P, weeks, clusters, labels, groups, changed, code,
                   f"{OUT}/{mouse}/temporal_by_group.png",
                   f"{mouse} ({grp}): rows by feature-group (red label = evolution candidate)")

    def counts(lab):
        return {c: int((lab == c).sum()) for c in SECTION_ORDER}
    c0, c1 = counts(labels), counts(final)
    n_grp = len(np.unique(groups))
    print(f"\n{mouse} ({grp}): {len(clusters)} clusters, {n_grp} feature-groups "
          f"(thr={thr:.4f})")
    print(f"  individual : " + "  ".join(f"{k}={c0[k]}" for k in SECTION_ORDER))
    print(f"  pooled     : " + "  ".join(f"{k}={c1[k]}" for k in SECTION_ORDER))
    print(f"  flagged (changed): {int(changed.sum())}   of which strong "
          f"(cat->diff cat): {int(strong.sum())}   resting: {int(resting.sum())}")
    return df


if __name__ == "__main__":
    for m in MICE:
        if os.path.exists(f"{OUT}/{m}/feature.npz"):
            analyze(m)
