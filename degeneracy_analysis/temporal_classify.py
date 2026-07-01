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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from sklearn.cluster import AgglomerativeClustering

from presence_similarity import MICE, OUT

CATS = ["early", "mid", "late", "sustained"]
MARGIN = 0.03        # min (best - 2nd) cosine to accept a category
GROUP_PCT = 10       # Dfeat percentile used as agglomerative distance threshold
SECTION_ORDER = ["early", "mid", "late", "sustained", "uncategorized"]
CAT_RANK = {c: i for i, c in enumerate(SECTION_ORDER)}
CAT_COLORS = {"early": "#d62728", "mid": "#ff7f0e", "late": "#1f77b4",
              "sustained": "#2ca02c", "uncategorized": "#bbbbbb"}


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


def merge_uncategorized(labels, groups):
    final = labels.copy()
    merged = np.zeros(len(labels), bool)
    for gid in np.unique(groups):
        idx = np.where(groups == gid)[0]
        cat = [labels[i] for i in idx if labels[i] != "uncategorized"]
        if not cat:
            continue
        vals, counts = np.unique(cat, return_counts=True)
        maj = vals[np.argmax(counts)]
        for i in idx:
            if labels[i] == "uncategorized":
                final[i], merged[i] = maj, True
    return final, merged


def _section_heatmap(mouse, P, weeks, clusters, labels, groups, merged, path, title):
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

    fig, (axg, ax) = plt.subplots(
        1, 2, figsize=(8.5, max(6, len(ro) * 0.11)),
        gridspec_kw={"width_ratios": [0.05, 1], "wspace": 0.02})

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
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _group_heatmap(mouse, P, weeks, clusters, final, groups, merged, path, title):
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
    fig, (axc, ax) = plt.subplots(
        1, 2, figsize=(8.5, max(6, len(ro) * 0.12)),
        gridspec_kw={"width_ratios": [0.045, 1], "wspace": 0.02})
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
    final, merged = merge_uncategorized(labels, groups)

    # thirds mass fractions for transparency
    e, m, l = [b.sum(1) for b in np.array_split(P, 3, axis=1)]
    df = pd.DataFrame({
        "cluster": clusters, "label": labels, "final_label": final, "merged": merged,
        "feature_group": groups, "centroid_week": P @ weeks, "margin": margins,
        "frac_early": e, "frac_mid": m, "frac_late": l,
        **{f"cos_{c}": S[:, i] for i, c in enumerate(CATS)},
    })
    df.to_csv(f"{OUT}/{mouse}/temporal_classes.csv", index=False)

    grp = "control" if mouse.endswith("lc") else "MitoPark"
    _section_heatmap(mouse, P, weeks, clusters, labels, groups, merged,
                     f"{OUT}/{mouse}/temporal_classes.png",
                     f"{mouse} ({grp}): temporal classes (original)")
    _section_heatmap(mouse, P, weeks, clusters, final, groups, merged,
                     f"{OUT}/{mouse}/temporal_classes_merged.png",
                     f"{mouse} ({grp}): after feature-merge (red = absorbed)")
    _group_heatmap(mouse, P, weeks, clusters, final, groups, merged,
                   f"{OUT}/{mouse}/temporal_by_group.png",
                   f"{mouse} ({grp}): rows by feature-group, ordered early->sustained")

    def counts(lab):
        return {c: int((lab == c).sum()) for c in SECTION_ORDER}
    c0, c1 = counts(labels), counts(final)
    n_grp = len(np.unique(groups))
    print(f"\n{mouse} ({grp}): {len(clusters)} clusters, {n_grp} feature-groups "
          f"(thr={thr:.4f})")
    print(f"  original : " + "  ".join(f"{k}={c0[k]}" for k in SECTION_ORDER))
    print(f"  merged   : " + "  ".join(f"{k}={c1[k]}" for k in SECTION_ORDER)
          + f"   (absorbed {int(merged.sum())})")
    return df


if __name__ == "__main__":
    for m in MICE:
        if os.path.exists(f"{OUT}/{m}/feature.npz"):
            analyze(m)
