"""Classify each cluster as 2D-exclusive, 3D-exclusive, or shared, per the
MATLAB pipeline's per-frame arena labels.

Each stitched batch is one joint clustering over (weeks x 2 arenas). The MATLAB
Cluster_detail_results.csv labels every frame with Folder_Name = weekN (the 3D
arena) or weekN_O (the flat / 2D arena). For each cluster we ask: of the time
the animal spends in that cluster, how much happens in the 3D vs the flat arena?

Arena convention (user-confirmed):
    weekN     -> 3D arena
    weekN_O   -> flat / 2D arena

Occupancy metric. Raw frame fraction would be biased if the two arenas
contribute unequal total time, so we normalize by each arena's total frames
(an enrichment, not a raw count):

    p3 = n3 / total_3d        p2 = n2 / total_2d
    occ3d = p3 / (p3 + p2)    in [0, 1]; 0.5 == arena-neutral

A cluster's verdict comes from occ3d against tunable bands (defaults):
    occ3d >= 0.80            -> 3D-exclusive
    occ3d <= 0.20            -> 2D-exclusive
    0.35 <= occ3d <= 0.65    -> shared
    otherwise               -> grey (leaning 2D/3D but not decisive)

Cluster ids are only meaningful WITHIN one (mouse, batch) clustering, so every
verdict is computed per (mouse, batch). 2mp/w10 has no MATLAB output and is
skipped. Frames whose Folder_Name is NaN (segment boundaries) are dropped.

Run:
    uv run python arena_analysis/cluster_arena_exclusivity.py
    uv run python arena_analysis/cluster_arena_exclusivity.py --excl 0.85 --shared 0.40
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "exclusivity"
_ARENA_LABEL = re.compile(r"(\d+(?:mp|lc)[a-z0-9_]*)_arena_compare", re.I)
_ARENA_BATCH = re.compile(r"arena_compare_(w\d+)_stitched", re.I)


def discover_mice(data_dir=DATA):
    """Map each '<...>_arena_compare' data dir to its short label (042025_1mp_
    arena_compare -> 1mp). Auto-discovered so a new arena_compare mouse dir is
    picked up with no code change."""
    out = {}
    if Path(data_dir).is_dir():
        for d in sorted(Path(data_dir).iterdir()):
            m = _ARENA_LABEL.search(d.name)
            if d.is_dir() and m:
                out[d.name] = m.group(1)
    return out


def discover_batches(data_dir=DATA, mice=None):
    """Union of stitched batch tags (w8, w9, ...) across the mouse dirs, in week
    order."""
    mice = mice if mice is not None else discover_mice(data_dir)
    found = set()
    for mouse_dir in mice:
        base = Path(data_dir) / mouse_dir
        if base.is_dir():
            for sub in base.iterdir():
                m = _ARENA_BATCH.search(sub.name)
                if m:
                    found.add(m.group(1))
    return sorted(found, key=lambda b: int(b[1:]))


MICE = discover_mice()
BATCHES = discover_batches(mice=MICE)

MIN_FRAMES = 50          # clusters smaller than this are flagged low_n (noisy occ)
VERDICT_COLORS = {
    "3D-exclusive": "#1f77b4", "2D-exclusive": "#d62728",
    "shared": "#2ca02c", "grey": "#999999",
}


def parse_segment(label):
    """'week8_O' -> (8, '2D');  'week8' -> (8, '3D');  else (None, None)."""
    if not isinstance(label, str):
        return None, None
    m = re.match(r"week(\d+)(_O)?$", label.strip())
    if not m:
        return None, None
    return int(m.group(1)), ("2D" if m.group(2) else "3D")


def classify(occ3d, excl, shared):
    if occ3d >= excl:
        return "3D-exclusive"
    if occ3d <= 1 - excl:
        return "2D-exclusive"
    if 0.5 - shared <= occ3d <= 0.5 + shared:
        return "shared"
    return "grey"


def load_mat_frames(mouse_dir, batch):
    """Per-frame (cluster, week, arena) for one MATLAB clustering, or None."""
    path = DATA / mouse_dir / f"arena_compare_{batch}_stitched" / "mat_results" \
        / "Cluster_detail_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    wk, ar = zip(*(parse_segment(s) for s in df["Folder_Name"]))
    df = df.assign(week=wk, arena=ar).dropna(subset=["arena"])
    return df.rename(columns={"ClusterIdx": "cluster"})[["cluster", "week", "arena"]]


def cluster_table(frames, excl, shared):
    """One row per cluster: arena counts, occupancy, verdict."""
    total = frames["arena"].value_counts()
    total_2d, total_3d = total.get("2D", 0), total.get("3D", 0)
    rows = []
    for c, grp in frames.groupby("cluster"):
        ac = grp["arena"].value_counts()
        n2, n3 = int(ac.get("2D", 0)), int(ac.get("3D", 0))
        p2 = n2 / total_2d if total_2d else 0.0
        p3 = n3 / total_3d if total_3d else 0.0
        occ3d = p3 / (p3 + p2) if (p3 + p2) else 0.5
        rows.append({
            "cluster": int(c), "n_frames": n2 + n3, "n_2d": n2, "n_3d": n3,
            "raw_f3d": n3 / (n2 + n3) if (n2 + n3) else np.nan,
            "occ3d": occ3d, "verdict": classify(occ3d, excl, shared),
            "low_n": (n2 + n3) < MIN_FRAMES,
        })
    return pd.DataFrame(rows).sort_values("occ3d").reset_index(drop=True)


def plot_batch(tbl, title, path, excl, shared):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(tbl))
    colors = [VERDICT_COLORS[v] for v in tbl["verdict"]]
    ax.bar(x, tbl["occ3d"], color=colors, width=0.9)
    for y in (1 - excl, 0.5 - shared, 0.5 + shared, excl):
        ax.axhline(y, ls="--", lw=0.8, color="k", alpha=0.4)
    ax.axhline(0.5, ls="-", lw=0.8, color="k", alpha=0.6)
    ax.set(xlabel=f"cluster (sorted by 3D occupancy, n={len(tbl)})",
           ylabel="3D occupancy  (0=flat/2D, 1=3D)", ylim=(0, 1), title=title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in VERDICT_COLORS.values()]
    ax.legend(handles, VERDICT_COLORS.keys(), ncol=4, fontsize=8,
              loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excl", type=float, default=0.80,
                    help="occ3d >= this -> 3D-exclusive (and <= 1-this -> 2D); default 0.80")
    ap.add_argument("--shared", type=float, default=0.15,
                    help="half-width of the shared band around 0.5; default 0.15")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    per_cluster, summary = [], []
    for mouse_dir, mouse in MICE.items():
        for batch in BATCHES:
            frames = load_mat_frames(mouse_dir, batch)
            if frames is None:
                print(f"  {mouse}/{batch}: no MATLAB output, skipped")
                continue
            tbl = cluster_table(frames, args.excl, args.shared)
            tbl.insert(0, "batch", batch)
            tbl.insert(0, "mouse", mouse)
            per_cluster.append(tbl)
            plot_batch(tbl, f"{mouse} {batch}  (MATLAB)",
                       OUT / f"{mouse}_{batch}_occupancy.png", args.excl, args.shared)

            # frame-share + cluster-count by verdict
            vc = tbl["verdict"].value_counts()
            fs = tbl.groupby("verdict")["n_frames"].sum()
            tot = tbl["n_frames"].sum()
            row = {"mouse": mouse, "batch": batch, "n_clusters": len(tbl)}
            for v in VERDICT_COLORS:
                row[f"k_{v}"] = int(vc.get(v, 0))
                row[f"frac_{v}"] = fs.get(v, 0) / tot
            summary.append(row)
            print(f"  {mouse}/{batch}: {len(tbl)} clusters -> "
                  + ", ".join(f"{v}={int(vc.get(v,0))}" for v in VERDICT_COLORS))

    allc = pd.concat(per_cluster, ignore_index=True)
    allc.to_csv(OUT / "cluster_verdicts.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "verdict_summary.csv", index=False)

    # overall: stacked frame-share by verdict, one bar per (mouse,batch)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = summ["mouse"] + "/" + summ["batch"]
    bottom = np.zeros(len(summ))
    for v, c in VERDICT_COLORS.items():
        ax.bar(labels, summ[f"frac_{v}"], bottom=bottom, color=c, label=v)
        bottom += summ[f"frac_{v}"].to_numpy()
    ax.set(ylabel="share of behavior time", ylim=(0, 1),
           title="Arena exclusivity of clusters (MATLAB), by mouse/batch")
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(OUT / "verdict_summary.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote {OUT}/cluster_verdicts.csv, verdict_summary.csv, and plots")
    print("\n=== verdict summary (frame-share) ===")
    cols = ["mouse", "batch", "n_clusters"] + [f"frac_{v}" for v in VERDICT_COLORS]
    print(summ[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
