"""Is 2D-vs-3D temporal divergence real signal or sampling noise? And are the
divergent clusters a coherent behavioural class -- within a mouse and across mice?

Companion to arena_class_split.py. Three questions:

1. SIGNIFICANCE. For each cluster we compare its 2D and 3D weekly presence SHAPE.
   Frames (~0.3 s) are heavily autocorrelated, so a frame-level null would be
   wildly anti-conservative; we resample at the BOUT level instead (a bout =
   maximal run of one cluster within one recording ~ one independent visit). The
   divergence metric is the Jensen-Shannon distance between the 2D and 3D
   bout-per-week profiles. Null: both arenas are iid multinomial draws (of the
   observed per-arena bout totals) from the POOLED weekly shape -- i.e. same shape,
   differing only by sampling. p = P(null JS >= observed JS); Benjamini-Hochberg
   FDR across clusters. A cluster whose 2D/3D shapes differ more than this null
   allows has a genuine arena x time interaction. The MitoPark-vs-control contrast
   is the bias-robust headline (both cohorts share the same autocorrelation).

2. WITHIN-MOUSE SIMILARITY. Are the divergent clusters closer in FEATURE space
   (feature.npz Dfeat) than random same-size sets of tested clusters? If so they
   are a coherent repertoire, not scattered noise.

3. CROSS-MOUSE ANALOGS. Using the physical IMU features (TBA, per-axis accel,
   gyro -- common units across mice), do divergent clusters have close analogs in
   OTHER mice? Reports the nearest cross-mouse divergent pairs and tests whether
   divergent clusters are more cross-mouse-similar than chance.

Needs arena_class_split.csv (run arena_class_split.py first) for the frame-based
class labels it merges in.

Outputs (arena_analysis/output/arena_divergence_stats/):
    divergence_significance.csv        per cluster: js_obs, p, q, labels, features
    divergence_significance.png        volcano + fraction-significant-per-mouse
    within_mouse_feature_similarity.csv  per batch: divergent vs null mean Dfeat
    cross_mouse_analogs.csv            nearest divergent cross-mouse pairs
    cross_mouse_analogs.png            divergent clusters in feature PCA, analogs linked

Run:
    C:/ProgramData/anaconda3/python.exe arena_analysis/arena_divergence_stats.py
    ... --min-bouts 15 --nperm 1000 --data-root E:/arena_analysis
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
from arena_class_split import ARENAS, CAT_COLORS                           # noqa: E402

OUT = ROOT / "output" / "arena_divergence_stats"
FEATS = ["TBA", "ap_accel", "dv_accel", "gyro"]


# ------------------------------- primitives -------------------------------- #
def bout_counts(detail, clusters, weeks):
    """arena -> (cluster x week) BOUT-count matrix. A bout = maximal run of one
    ClusterIdx within one Folder_Name recording (computed on the FULL frame order
    so dropped/blank frames and week boundaries break runs)."""
    d = pd.read_csv(detail).reset_index(drop=True)
    new = (d["ClusterIdx"].ne(d["ClusterIdx"].shift())
           | d["Folder_Name"].ne(d["Folder_Name"].shift()))
    starts = d[new].copy()
    seg = [parse_segment(s) for s in starts["Folder_Name"]]
    starts["_week"] = [s[0] for s in seg]
    starts["_arena"] = [s[1] for s in seg]
    starts = starts.dropna(subset=["_arena"])
    starts["_week"] = starts["_week"].astype(int)
    ci = {c: i for i, c in enumerate(clusters)}
    wi = {w: j for j, w in enumerate(weeks)}
    mats = {a: np.zeros((len(clusters), len(weeks))) for a in ARENAS}
    g = starts.groupby(["ClusterIdx", "_week", "_arena"]).size().reset_index(name="n")
    for c, w, a, n in g.itertuples(index=False):
        if a in mats and c in ci and w in wi:
            mats[a][ci[c], wi[w]] = n
    return mats


def js_dist_rows(P, Q):
    """Jensen-Shannon distance in [0,1] between rows of two count matrices."""
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    M = 0.5 * (P + Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        klp = np.nansum(np.where(P > 0, P * np.log(P / M), 0.0), axis=1)
        klq = np.nansum(np.where(Q > 0, Q * np.log(Q / M), 0.0), axis=1)
    return np.sqrt(np.clip(0.5 * klp + 0.5 * klq, 0, None) / np.log(2))


def js_dist(a, b):
    return float(js_dist_rows(a[None, :], b[None, :])[0])


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# ------------------------------- part 1 ------------------------------------ #
def significance(mice, min_bouts, nperm, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mouse, batches in mice.items():
        for tag, detail, temporal in batches:
            t = pd.read_csv(temporal)
            clusters = t["cluster"].to_numpy()
            det = pd.read_csv(detail)
            weeks = np.sort([w for w in
                             (parse_segment(s)[0] for s in det["Folder_Name"])
                             if w is not None])
            weeks = np.unique(weeks)
            mats = bout_counts(detail, clusters, weeks)
            ti = t.set_index("cluster")
            for i, c in enumerate(clusters):
                b2, b3 = mats["2D"][i], mats["3D"][i]
                N2, N3 = int(b2.sum()), int(b3.sum())
                rec = {"mouse": mouse, "batch": tag, "cluster": int(c),
                       "bouts_2D": N2, "bouts_3D": N3,
                       **{f: ti.loc[c, f] for f in FEATS},
                       "move_type": ti.loc[c, "move_type"]}
                if N2 < min_bouts or N3 < min_bouts:
                    rec.update(js_obs=np.nan, js_null_mean=np.nan, p_value=np.nan)
                else:
                    p = (b2 + b3) / (b2 + b3).sum()
                    d_obs = js_dist(b2, b3)
                    A = rng.multinomial(N2, p, size=nperm)
                    Bm = rng.multinomial(N3, p, size=nperm)
                    d_null = js_dist_rows(A, Bm)
                    rec.update(js_obs=round(d_obs, 4),
                               js_null_mean=round(float(d_null.mean()), 4),
                               p_value=(1 + int((d_null >= d_obs).sum())) / (nperm + 1))
                rows.append(rec)
    df = pd.DataFrame(rows)
    tested = df["p_value"].notna()
    df.loc[tested, "q_value"] = bh_fdr(df.loc[tested, "p_value"].to_numpy())
    df["group"] = np.where(df.mouse.str.endswith("lc"), "control", "MitoPark")
    df["significant"] = df["q_value"] < 0.05
    return df


def plot_significance(df, path):
    tested = df[df.p_value.notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for grp, c in [("MitoPark", "#d62728"), ("control", "#1f77b4")]:
        s = tested[tested.group == grp]
        ax.scatter(s.js_obs, -np.log10(s.p_value.clip(lower=1e-4)), s=12, alpha=0.5,
                   color=c, label=grp)
    ax.axhline(-np.log10(0.05), ls="--", color="k", lw=0.8, alpha=0.6)
    ax.set(xlabel="observed 2D–3D Jensen-Shannon distance",
           ylabel="-log10 p (bout-resample null)",
           title="Per-cluster arena divergence vs sampling null")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    agg = tested.groupby("mouse").agg(
        frac_sig=("significant", "mean"),
        grp=("group", "first")).reset_index()
    colors = ["#d62728" if g == "MitoPark" else "#1f77b4" for g in agg.grp]
    ax.bar(agg.mouse, 100 * agg.frac_sig, color=colors)
    ax.axhline(5, ls="--", color="k", lw=0.8, alpha=0.6, label="5% (null expectation)")
    ax.set(ylabel="% clusters significantly divergent (FDR<0.05)",
           title="Genuine arena divergence by mouse")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


# ------------------------------- part 2 ------------------------------------ #
def within_mouse_similarity(mice, sig, min_set=3, nperm=2000, seed=1):
    """Per batch: mean pairwise Dfeat among divergent clusters vs random same-size
    subsets of tested clusters."""
    rng = np.random.default_rng(seed)
    rows = []
    for mouse, batches in mice.items():
        for tag, detail, temporal in batches:
            f = np.load(Path(temporal).parent / "feature.npz", allow_pickle=True)
            fc = f["clusters"].astype(int)
            D = f["Dfeat"]
            fi = {c: i for i, c in enumerate(fc)}
            g = sig[(sig.mouse == mouse) & (sig.batch == tag) & sig.p_value.notna()]
            pool = [fi[c] for c in g.cluster if c in fi]
            div = [fi[c] for c in g[g.significant].cluster if c in fi]
            if len(div) < min_set or len(pool) <= len(div):
                continue

            def mean_pair(idx):
                idx = np.array(idx)
                sub = D[np.ix_(idx, idx)]
                iu = np.triu_indices(len(idx), 1)
                return float(sub[iu].mean())

            obs = mean_pair(div)
            null = np.array([mean_pair(rng.choice(pool, len(div), replace=False))
                             for _ in range(nperm)])
            rows.append({"mouse": mouse, "batch": tag, "group":
                         "control" if mouse.endswith("lc") else "MitoPark",
                         "n_divergent": len(div), "n_pool": len(pool),
                         "obs_mean_Dfeat": round(obs, 4),
                         "null_mean_Dfeat": round(float(null.mean()), 4),
                         "z": round((obs - null.mean()) / (null.std() + 1e-9), 2),
                         "p_more_similar": (1 + int((null <= obs).sum())) / (nperm + 1)})
    return pd.DataFrame(rows)


# ------------------------------- part 3 ------------------------------------ #
def cross_mouse_analogs(sig, nperm=2000, seed=2, top=25):
    """Nearest cross-mouse divergent pairs in standardized physical-feature space,
    and whether divergent clusters are more cross-mouse-similar than chance."""
    rng = np.random.default_rng(seed)
    tested = sig[sig.p_value.notna()].dropna(subset=FEATS).copy().reset_index(drop=True)
    X = tested[FEATS].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    mouse = tested.mouse.to_numpy()

    def nn_other_mouse(idx_set):
        """mean nearest-neighbour distance to a cluster in a DIFFERENT mouse."""
        d = []
        for i in idx_set:
            other = np.where(mouse != mouse[i])[0]
            dist = np.linalg.norm(X[other] - X[i], axis=1)
            d.append(dist.min())
        return np.array(d)

    div_idx = np.where(tested.significant.to_numpy())[0]
    obs = nn_other_mouse(div_idx)
    null = np.array([nn_other_mouse(rng.choice(len(tested), len(div_idx),
                                               replace=False)).mean()
                     for _ in range(nperm)])
    stat = {"n_divergent": len(div_idx),
            "obs_mean_NN": round(float(obs.mean()), 4),
            "null_mean_NN": round(float(null.mean()), 4),
            "p_more_similar": (1 + int((null <= obs.mean()).sum())) / (nperm + 1)}

    # explicit nearest cross-mouse divergent pairs
    pairs = []
    for a in div_idx:
        others = [b for b in div_idx if mouse[b] != mouse[a]]
        if not others:
            continue
        others = np.array(others)
        dist = np.linalg.norm(X[others] - X[a], axis=1)
        b = others[dist.argmin()]
        key = tuple(sorted([a, int(b)]))
        pairs.append((key, float(dist.min()), a, int(b)))
    seen, out = set(), []
    for key, dist, a, b in sorted(pairs, key=lambda x: x[1]):
        if key in seen:
            continue
        seen.add(key)
        ra, rb = tested.iloc[a], tested.iloc[b]
        out.append({
            "dist": round(dist, 3),
            "a": f"{ra.mouse}/{ra.batch}:{ra.cluster}", "a_move": ra.move_type,
            "a_js": ra.js_obs,
            "b": f"{rb.mouse}/{rb.batch}:{rb.cluster}", "b_move": rb.move_type,
            "b_js": rb.js_obs,
            "a_TBA": round(ra.TBA, 3), "b_TBA": round(rb.TBA, 3),
            "a_gyro": round(ra.gyro, 2), "b_gyro": round(rb.gyro, 2)})
    return stat, pd.DataFrame(out).head(top), tested, X


def plot_cross_mouse(tested, X, analogs, path):
    # PCA(2) of standardized features
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    mice = sorted(tested.mouse.unique())
    cmap = plt.get_cmap("tab10")
    mcol = {m: cmap(i % 10) for i, m in enumerate(mice)}
    fig, ax = plt.subplots(figsize=(8.5, 7))
    div = tested.significant.to_numpy()
    ax.scatter(P[~div, 0], P[~div, 1], s=8, color="0.8", label="tested (n.s.)")
    for m in mice:
        sel = div & (tested.mouse.to_numpy() == m)
        ax.scatter(P[sel, 0], P[sel, 1], s=32, color=mcol[m], edgecolor="k",
                   lw=0.3, label=f"{m} divergent")
    idx = {f"{r.mouse}/{r.batch}:{r.cluster}": k for k, r in tested.iterrows()}
    for _, row in analogs.iterrows():
        if row.a in idx and row.b in idx:
            ia, ib = idx[row.a], idx[row.b]
            ax.plot(P[[ia, ib], 0], P[[ia, ib], 1], "-", color="0.4", lw=0.7, alpha=0.7)
    ax.set(xlabel="feature PC1", ylabel="feature PC2",
           title="Arena-divergent clusters in physical-feature space\n"
                 "(lines = nearest cross-mouse analog pairs)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("E:/arena_analysis"))
    ap.add_argument("--min-bouts", type=int, default=15,
                    help="min bouts in an arena to test a cluster (default 15)")
    ap.add_argument("--nperm", type=int, default=1000, help="null resamples")
    args = ap.parse_args()

    mice = discover(args.data_root)
    if not mice:
        raise SystemExit(f"no *_arena_compare mice found under {args.data_root}")
    OUT.mkdir(parents=True, exist_ok=True)

    asc_path = ROOT / "output" / "arena_class_split" / "arena_class_split.csv"
    asc = pd.read_csv(asc_path)[["mouse", "batch", "cluster", "pooled_label",
                                 "label_2D", "label_3D", "arena_discordant",
                                 "cen_shift_3D_minus_2D"]]

    # --- Part 1 ---
    sig = significance(mice, args.min_bouts, args.nperm)
    sig = sig.merge(asc, on=["mouse", "batch", "cluster"], how="left")
    sig.to_csv(OUT / "divergence_significance.csv", index=False)
    plot_significance(sig, OUT / "divergence_significance.png")

    # baseline control: JS between the two arenas' TOTAL weekly bout profiles. If
    # the arenas were sampled equally each week this is ~0, so any per-cluster
    # divergence is behavioural, not a per-week arena-sampling artifact.
    base_rows = []
    for mouse, batches in mice.items():
        for tag, detail, temporal in batches:
            clusters = pd.read_csv(temporal)["cluster"].to_numpy()
            det = pd.read_csv(detail)
            weeks = np.unique([w for w in
                               (parse_segment(s)[0] for s in det["Folder_Name"])
                               if w is not None])
            m = bout_counts(detail, clusters, weeks)
            base_rows.append({"mouse": mouse, "batch": tag,
                              "baseline_arena_JS": round(js_dist(m["2D"].sum(0),
                                                                 m["3D"].sum(0)), 3)})
    base = pd.DataFrame(base_rows)
    base.to_csv(OUT / "arena_sampling_baseline.csv", index=False)

    tested = sig[sig.p_value.notna()]
    print(f"Wrote {OUT}/  (min_bouts={args.min_bouts}, nperm={args.nperm})\n")
    print("=== Part 0: arena-sampling baseline (JS of total 2D vs 3D per week) ===")
    print(f"  median {base.baseline_arena_JS.median():.3f}  "
          f"(vs per-cluster median obs {tested.js_obs.median():.3f}) — "
          f"batches with baseline>0.10 (sampling artifact): "
          f"{list(base[base.baseline_arena_JS>0.10].mouse+'/'+base[base.baseline_arena_JS>0.10].batch)}")
    print("\n=== Part 1: divergence significance (bout-resample null) ===")
    print(f"tested clusters (>= {args.min_bouts} bouts both arenas): "
          f"{len(tested)}/{len(sig)}")
    for grp, g in tested.groupby("group"):
        print(f"  {grp:8s}: {int(g.significant.sum()):3d}/{len(g):3d} "
              f"significant (FDR<0.05) = {100*g.significant.mean():.0f}%  "
              f"| median JS obs {g.js_obs.median():.3f} vs null {g.js_null_mean.median():.3f}")
    disc = tested[tested.arena_discordant == True]
    print(f"  of {len(disc)} label-discordant clusters, "
          f"{int(disc.significant.sum())} ({100*disc.significant.mean():.0f}%) "
          f"are also significantly divergent")

    # --- Part 2 ---
    wm = within_mouse_similarity(mice, sig)
    wm.to_csv(OUT / "within_mouse_feature_similarity.csv", index=False)
    print("\n=== Part 2: are divergent clusters more feature-similar than chance? ===")
    if wm.empty:
        print("  (too few divergent clusters per batch to test)")
    else:
        print(wm.to_string(index=False))
        print(f"  batches where divergent are MORE similar (p<0.05): "
              f"{int((wm.p_more_similar<0.05).sum())}/{len(wm)}")

    # --- Part 3 ---
    stat, analogs, tt, X = cross_mouse_analogs(sig)
    analogs.to_csv(OUT / "cross_mouse_analogs.csv", index=False)
    plot_cross_mouse(tt, X, analogs, OUT / "cross_mouse_analogs.png")
    print("\n=== Part 3: cross-mouse analogs of divergent clusters ===")
    print(f"  divergent clusters more cross-mouse-similar than chance: "
          f"obs NN {stat['obs_mean_NN']} vs null {stat['null_mean_NN']} "
          f"(p={stat['p_more_similar']:.3f}, n={stat['n_divergent']})")
    print("  nearest cross-mouse divergent pairs (top 12):")
    print(analogs.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
