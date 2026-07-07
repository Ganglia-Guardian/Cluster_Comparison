"""Shared data-access helpers for the cluster-group (dendrogram) analysis.

Same flat MATLAB layout as arena_analysis:
    data/<mouse>_arena_compare/arena_compare_w<N>/
        Cluster_detail_results.csv   per-frame ClusterIdx + Timestamp + Folder_Name
        session_1_out.mat            StructData/func -> 4 per-sample kinematic features

Folder_Name encodes both week and arena: weekN -> 3D arena, weekN_O -> flat/2D
arena (suffix written inconsistently across mice; parse_segment tolerates the
variants). Cluster ids are only meaningful WITHIN one (mouse, batch) clustering.

This module only reads/normalizes; it builds no plots and takes no CLI args.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT.parent))          # repo-root feature extractor
from feature_extraction import FEATURE_NAMES, combine_results  # noqa: E402

_ARENA_LABEL = re.compile(r"(\d+(?:mp|lc)[a-z0-9_]*)_arena_compare", re.I)
_ARENA_BATCH = re.compile(r"arena_compare_(w\d+)", re.I)
# weekN (3D) or weekN_o / week_N_o / weekN_O (2D flat); see arena-pipelines memory.
_SEG = re.compile(r"week_?(\d+)(_o)?$", re.I)


def discover_mice(data_dir=DATA):
    """{data-dir-name -> short label}, e.g. 042025_1mp_arena_compare -> 1mp."""
    out = {}
    if Path(data_dir).is_dir():
        for d in sorted(Path(data_dir).iterdir()):
            m = _ARENA_LABEL.search(d.name)
            if d.is_dir() and m:
                out[d.name] = m.group(1)
    return out


def discover_batches(data_dir=DATA, mice=None):
    """Sorted union of batch tags (w8, w9, w10) across mouse dirs."""
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


def batch_dir(mouse_dir, batch):
    return DATA / mouse_dir / f"arena_compare_{batch}"


def mat_csv(mouse_dir, batch):
    return batch_dir(mouse_dir, batch) / "Cluster_detail_results.csv"


def session_mat(mouse_dir, batch):
    return batch_dir(mouse_dir, batch) / "session_1_out.mat"


def parse_segment(label):
    """weekN[_o] -> (week:int, arena:'2D'|'3D'); non-matching -> (None, None)."""
    if not isinstance(label, str):
        return None, None
    m = _SEG.match(label.strip())
    if not m:
        return None, None
    return int(m.group(1)), ("2D" if m.group(2) else "3D")


def load_frames_with_features(mouse_dir, batch):
    """Per-frame table for one (mouse, batch): cluster, week, arena + 4 features.

    Boundary frames whose Folder_Name doesn't parse (NaN arena) are dropped.
    Returns None if either the CSV or the MAT file is missing.
    """
    mat, csv = session_mat(mouse_dir, batch), mat_csv(mouse_dir, batch)
    if not (mat.exists() and csv.exists()):
        return None
    clu = pd.read_csv(csv)
    df = combine_results(mat, cb_matrix=clu)          # appends binned features
    wk, ar = zip(*(parse_segment(s) for s in df["Folder_Name"]))
    df = (df.assign(week=wk, arena=ar)
            .dropna(subset=["arena"])
            .rename(columns={"ClusterIdx": "cluster"}))
    return df[["cluster", "week", "arena", *FEATURE_NAMES]]


def cluster_centroids(frames):
    """One row per cluster for a (mouse, batch): mean of the 4 features plus
    arena-normalized 3D occupancy and frame counts.

    occ3d normalizes each arena's frames by that arena's total so an imbalance in
    recording time doesn't bias the ratio (matches arena_analysis convention):
        p3 = n3/total_3d,  p2 = n2/total_2d,  occ3d = p3/(p3+p2).
    """
    total = frames["arena"].value_counts()
    total_2d, total_3d = total.get("2D", 0), total.get("3D", 0)
    rows = []
    for c, grp in frames.groupby("cluster"):
        ac = grp["arena"].value_counts()
        n2, n3 = int(ac.get("2D", 0)), int(ac.get("3D", 0))
        p2 = n2 / total_2d if total_2d else 0.0
        p3 = n3 / total_3d if total_3d else 0.0
        occ3d = p3 / (p3 + p2) if (p3 + p2) else 0.5
        row = {"cluster": int(c), "n_frames": n2 + n3, "n_2d": n2, "n_3d": n3,
               "occ3d": occ3d}
        for f in FEATURE_NAMES:
            row[f] = grp[f].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def week_arena_counts(frames):
    """Long table: cluster, week, arena, n  (frame counts) for one (mouse, batch)."""
    g = (frames.groupby(["cluster", "week", "arena"]).size()
                .rename("n").reset_index())
    g["cluster"] = g["cluster"].astype(int)
    g["week"] = g["week"].astype(int)
    return g


def week_arena_features(frames):
    """Per (cluster, week, arena): mean of the 4 features + frame count n, for one
    (mouse, batch). Lets downstream code build week-specific cluster centroids
    (features drift over disease weeks) and per-week arena occupancy."""
    g = frames.groupby(["cluster", "week", "arena"])
    out = g[FEATURE_NAMES].mean().reset_index()
    out["n"] = g.size().to_numpy()
    out["cluster"] = out["cluster"].astype(int)
    out["week"] = out["week"].astype(int)
    return out[["cluster", "week", "arena", "n", *FEATURE_NAMES]]
