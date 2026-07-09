"""Arena-resolved temporal classification: split each cluster into its 2D and 3D
components, classify each with the SAME template classifier used on the pooled
presence, and see where the two arenas disagree. Plus per-arena cluster presence
heatmaps.

Motivation. The pooled temporal class (early/mid/late/sustained/uncategorized in
temporal_classes.csv) is fit on 2D+3D presence combined. A per-cluster Spearman
over only 5-6 weeks is too underpowered to ask "does it fade in 3D but not 2D?".
The template classifier instead scores the whole weekly SHAPE against the four
templates with a margin, so re-running it on each arena's presence separately is a
far more robust way to detect an arena-specific temporal shift.

Part 1 -- CLASS SPLIT. For each (mouse, batch) we rebuild presence exactly as
degeneracy_analysis/presence_similarity.py does (raw counts per natural week,
row-normalized; no within-week norm) but restricted to one arena, and classify it.
A cluster's 2D component and 3D component each get a label; we compare them to each
other and to the pooled label. We also report each arena's presence centroid week
-- cen_3D < cen_2D means the behaviour is concentrated earlier in 3D (fades sooner
there), a threshold-free degeneration signal. Arenas with fewer than --min-arena
frames are left 'insufficient' (a half-split of a small cluster is just noise).

Part 2 -- PRESENCE HEATMAPS. Per (mouse, batch) a 2D and a 3D heatmap: rows =
that batch's clusters, columns = that batch's weeks (each batch is its own
clustering over a strided 6-week set), colour = that arena's row-normalized weekly
presence (row-max scaled so the temporal SHAPE shows). Both arenas of a batch use
the SAME row order so a cluster's 2D and 3D shape line up. A cluster absent from an
arena is an all-grey row. Row order is decided by one function -- order_rows() --
so it is easy to change (default: temporal section, then pooled centroid week).

Outputs (arena_analysis/output/arena_class_split/):
    arena_class_split.csv                     per-cluster pooled/2D/3D labels + centroids
    <mouse>_class_confusion.png               label_2D x label_3D count heatmap
    <mouse>_<batch>_2D_presence_heatmap.png   2D per-week presence, temporal-class ordered
    <mouse>_<batch>_3D_presence_heatmap.png   3D per-week presence, same row order

Run:
    C:/ProgramData/anaconda3/python.exe arena_analysis/arena_class_split.py
    ... --label-col combined_label --min-arena 100 --data-root E:/arena_analysis
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from cluster_arena_exclusivity import parse_segment                        # noqa: E402
from temporal_arena_frequency import discover                             # noqa: E402

OUT = ROOT / "output" / "arena_class_split"
ARENAS = ["2D", "3D"]

# --- classifier, copied verbatim from degeneracy_analysis/temporal_classify.py --
CATS = ["early", "mid", "late", "sustained"]
MARGIN = 0.03
SECTION_ORDER = ["early", "mid", "late", "sustained", "uncategorized"]
LABELS_ALL = SECTION_ORDER + ["insufficient"]
CAT_COLORS = {"early": "#d62728", "mid": "#ff7f0e", "late": "#1f77b4",
              "sustained": "#2ca02c", "uncategorized": "#bbbbbb",
              "insufficient": "#eeeeee"}


def templates(weeks):
    t = (weeks - weeks.min()) / (weeks.max() - weeks.min())
    T = {"early": 1 - t, "late": t,
         "mid": np.exp(-((t - 0.5) / 0.18) ** 2),
         "sustained": np.ones_like(t, float)}
    return {k: v / np.linalg.norm(v) for k, v in T.items()}


def classify_one(counts, weeks, margin=MARGIN):
    """Classify one cluster's weekly count profile -> (label, margin, centroid).

    Presence is the row-normalized shape (matches presence_similarity, no
    within-week norm); an empty profile is 'insufficient'."""
    total = counts.sum()
    if total == 0:
        return "insufficient", np.nan, np.nan
    P = counts / total                              # across-week shape, sums to 1
    n = np.linalg.norm(P)
    if n == 0:
        return "insufficient", np.nan, np.nan
    T = templates(weeks)
    S = np.array([(P / n) @ T[c] for c in CATS])
    order = np.argsort(-S)
    marg = float(S[order[0]] - S[order[1]])
    lab = CATS[order[0]] if marg >= margin else "uncategorized"
    return lab, marg, float(P @ weeks)


def arena_counts(det, clusters, weeks):
    """arena -> (cluster x week) count matrix, aligned to `clusters` and `weeks`."""
    ci = {c: i for i, c in enumerate(clusters)}
    wi = {w: j for j, w in enumerate(weeks)}
    mats = {a: np.zeros((len(clusters), len(weeks))) for a in ARENAS}
    g = det.groupby(["ClusterIdx", "_week", "_arena"]).size().reset_index(name="n")
    for c, w, a, n in g.itertuples(index=False):
        if a in mats and c in ci and w in wi:
            mats[a][ci[c], wi[w]] = n
    return mats


def load_batch(detail, temporal):
    """(det with _week/_arena, clusters, weeks, temporal-indexed-by-cluster)."""
    det = pd.read_csv(detail)
    seg = [parse_segment(s) for s in det["Folder_Name"]]
    det["_week"] = [s[0] for s in seg]
    det["_arena"] = [s[1] for s in seg]
    det = det.dropna(subset=["_arena"])
    det["_week"] = det["_week"].astype(int)
    t = pd.read_csv(temporal)
    return det, t["cluster"].to_numpy(), np.sort(det["_week"].unique()), \
        t.set_index("cluster")


def classify_batch(mats, clusters, weeks, tinfo, label_col, min_arena):
    """Per-cluster pooled label + each arena's independent label/centroid, one row
    per cluster. Shared by the heatmaps (highlighting) and the CSV."""
    wf = weeks.astype(float)
    recs = []
    for i, c in enumerate(clusters):
        rec = {"cluster": int(c), "pooled_label": tinfo.loc[c, label_col],
               "pooled_centroid": tinfo.loc[c, "centroid_week"]}
        for a in ARENAS:
            cnt = mats[a][i]
            n = int(cnt.sum())
            if n < min_arena:
                lab, marg, cen = "insufficient", np.nan, np.nan
            else:
                lab, marg, cen = classify_one(cnt, wf)
            rec[f"label_{a}"] = lab
            rec[f"n_{a}"] = n
            rec[f"margin_{a}"] = round(marg, 4) if marg == marg else np.nan
            rec[f"cen_{a}"] = round(cen, 3) if cen == cen else np.nan
        rec["cen_shift_3D_minus_2D"] = (
            round(rec["cen_3D"] - rec["cen_2D"], 3)
            if rec["cen_2D"] == rec["cen_2D"] and rec["cen_3D"] == rec["cen_3D"]
            else np.nan)
        rec["arena_discordant"] = rec["label_2D"] != rec["label_3D"]
        recs.append(rec)
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
#  Row ordering -- the single place to change how heatmap rows are ordered.    #
# --------------------------------------------------------------------------- #
def order_rows(meta, section_col="pooled_label", sort_col="pooled_centroid"):
    """Return (row_order, section_bounds, section_labels) for the presence
    heatmaps. Default: group by temporal section (SECTION_ORDER), then sort within
    a section by `sort_col` ascending. Change this function to re-order."""
    labels = meta[section_col].to_numpy()
    key = meta[sort_col].to_numpy()
    order, bounds, ypos = [], [], []
    for c in SECTION_ORDER:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        idx = idx[np.argsort(key[idx])]
        ypos.append((c, len(order) + idx.size / 2))
        order.extend(int(i) for i in idx)
        bounds.append(len(order))
    return np.array(order, int), bounds, ypos


def plot_presence_heatmap(mouse, batch, arena, disp, weeks, meta, path):
    """One (batch, arena) per-week presence (row-max normalized) for its clusters.

    Rows whose THIS-arena class differs from the pooled class (the one that orders
    the rows) are highlighted: the y-label reads 'cluster -> newclass', bold and
    tinted the new class's colour. 'insufficient' arenas are not a change."""
    ro, bounds, ypos = order_rows(meta)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#dddddd")                         # grey: cluster absent from arena
    clusters = meta["cluster"].to_numpy()
    pooled = meta["pooled_label"].to_numpy()
    alab = meta[f"label_{arena}"].to_numpy()
    changed = (alab != pooled) & (alab != "insufficient")

    tick_txt, tick_col = [], []
    for i in ro:
        if changed[i]:
            tick_txt.append(f"{clusters[i]} → {alab[i]}")
            tick_col.append(CAT_COLORS[alab[i]])
        else:
            tick_txt.append(str(clusters[i]))
            tick_col.append("black")

    fig, ax = plt.subplots(figsize=(max(5.5, len(weeks) * 0.6),
                                    max(5, len(ro) * 0.13)))
    im = ax.imshow(np.ma.masked_invalid(disp[ro]), aspect="auto", cmap=cmap,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels(weeks, fontsize=8)
    ax.set_yticks(range(len(ro)))
    ax.set_yticklabels(tick_txt, fontsize=5)
    for tick, col in zip(ax.get_yticklabels(), tick_col):
        if col != "black":
            tick.set_color(col); tick.set_fontweight("bold")
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="cyan", lw=1.0)
    for c, yc in ypos:
        ax.text(len(weeks) - 0.4, yc, c, va="center", ha="left", fontsize=8,
                rotation=90, color=CAT_COLORS[c], fontweight="bold")
    grp = "control" if mouse.endswith("lc") else "MitoPark"
    ax.set(xlabel="disease week",
           title=f"{mouse} ({grp}) {batch} — {arena}: cluster presence by week\n"
                 f"(row-max norm; bold →label = reclassifies vs pooled, "
                 f"{int(changed.sum())}/{len(clusters)} clusters)")
    fig.colorbar(im, ax=ax, label="presence (row-max normalized)", shrink=0.5, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(mouse, df, path):
    """label_2D (rows) x label_3D (cols) count heatmap for one mouse."""
    order = LABELS_ALL
    M = np.zeros((len(order), len(order)), int)
    oi = {c: i for i, c in enumerate(order)}
    for a, b in zip(df["label_2D"], df["label_3D"]):
        M[oi[a], oi[b]] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(order)):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center",
                        color="white" if M[i, j] > M.max() / 2 else "black", fontsize=9)
    ax.plot([-0.5, len(order) - 0.5], [-0.5, len(order) - 0.5], color="0.6", lw=0.8)
    grp = "control" if mouse.endswith("lc") else "MitoPark"
    both = df[(df["label_2D"] != "insufficient") & (df["label_3D"] != "insufficient")]
    n_disc = int((both["label_2D"] != both["label_3D"]).sum())
    ax.set(xlabel="3D component class", ylabel="2D component class",
           title=f"{mouse} ({grp}): arena-split temporal class\n"
                 f"off-diagonal = arena-discordant "
                 f"({n_disc}/{len(both)} classifiable in both)")
    fig.colorbar(im, ax=ax, label="clusters", shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("E:/arena_analysis"))
    ap.add_argument("--label-col", choices=["label", "combined_label"], default="label")
    ap.add_argument("--min-arena", type=int, default=100,
                    help="min frames in an arena to classify its component "
                         "(else 'insufficient'); default 100")
    args = ap.parse_args()

    mice = discover(args.data_root)
    if not mice:
        raise SystemExit(f"no *_arena_compare mice found under {args.data_root}")
    OUT.mkdir(parents=True, exist_ok=True)

    parts = []
    for mouse, batches in mice.items():
        for tag, detail, temporal in batches:
            det, clusters, weeks, tinfo = load_batch(detail, temporal)
            mats = arena_counts(det, clusters, weeks)
            bdf = classify_batch(mats, clusters, weeks, tinfo,
                                 args.label_col, args.min_arena)

            # Part 2: per-arena presence heatmap for this batch (highlighting reclassified rows)
            for a in ARENAS:
                P = mats[a]
                rowsum = P.sum(axis=1)
                pres = np.full_like(P, np.nan, dtype=float)
                nz = rowsum > 0
                pres[nz] = P[nz] / rowsum[nz, None]           # across-week shape
                rowmax = np.array([np.nanmax(r) if m else np.nan
                                   for r, m in zip(pres, nz)])
                disp = pres / np.where(rowmax > 0, rowmax, np.nan)[:, None]
                plot_presence_heatmap(mouse, tag, a, disp, weeks, bdf,
                                      OUT / f"{mouse}_{tag}_{a}_presence_heatmap.png")

            parts.append(bdf.assign(mouse=mouse, batch=tag))

    df = pd.concat(parts, ignore_index=True)
    df = df[["mouse", "batch"] + [c for c in df.columns
                                  if c not in ("mouse", "batch")]]
    df.to_csv(OUT / "arena_class_split.csv", index=False)
    for mouse, g in df.groupby("mouse"):
        plot_confusion(mouse, g, OUT / f"{mouse}_class_confusion.png")

    # --- reclassification tally: how often each arena's label differs from pooled ---
    tally = []
    for mouse, g in df.groupby("mouse"):
        c2 = g[g.label_2D != "insufficient"]
        c3 = g[g.label_3D != "insufficient"]
        both = g[(g.label_2D != "insufficient") & (g.label_3D != "insufficient")]
        tally.append({
            "mouse": mouse,
            "group": "control" if mouse.endswith("lc") else "MitoPark",
            "n_clusters": len(g),
            "n_2D_reclass": int((c2.label_2D != c2.pooled_label).sum()),
            "pct_2D_reclass": round(100 * (c2.label_2D != c2.pooled_label).mean(), 1) if len(c2) else np.nan,
            "n_3D_reclass": int((c3.label_3D != c3.pooled_label).sum()),
            "pct_3D_reclass": round(100 * (c3.label_3D != c3.pooled_label).mean(), 1) if len(c3) else np.nan,
            "n_arena_discordant": int((both.label_2D != both.label_3D).sum()),
        })
    tdf = pd.DataFrame(tally)
    tdf.to_csv(OUT / "reclassification_tally.csv", index=False)

    # console summary
    print(f"Wrote {OUT}/ ({args.label_col}, min_arena={args.min_arena})\n")
    both = df[(df.label_2D != "insufficient") & (df.label_3D != "insufficient")]
    print(f"Clusters classifiable in both arenas: {len(both)}/{len(df)}")
    print(f"  arena-concordant (same class): {(both.label_2D==both.label_3D).sum()}")
    print(f"  arena-discordant:              {(both.label_2D!=both.label_3D).sum()}")
    cs = both["cen_shift_3D_minus_2D"].dropna()
    print(f"  centroid shift (3D - 2D): mean {cs.mean():+.2f} wk  "
          f"({(cs<0).sum()} fade earlier in 3D, {(cs>0).sum()} earlier in 2D)")
    print("\n=== reclassification tally (arena label vs pooled label) ===")
    print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()
