"""Build one tidy per-frame table across mice, batches, and both pipelines.

Each stitched batch is a single joint clustering over (weeks x 2 arenas). Only
the MATLAB pipeline labels each frame's segment (Folder_Name = week8, week8_O,
...); the Python pipeline emits a single constant Folder_Name and a harp-clock
Timestamp that does not cleanly mark segment seams. But the two pipelines are
row-aligned (same windowing, same order: alphabetical by segment label), so we
borrow MATLAB's per-row segment label onto the Python rows by index.

Arena convention (confirmed by the user):
    label ending in "_O"  -> flat / 2D arena
    plain label (week8)   -> 3D arena

Output: arena_analysis/frame_table.parquet with columns
    mouse, batch, row, week, arena, py_cluster, mat_cluster
NaN-segment boundary frames are dropped. Batches without a MATLAB output
(2mp/w10) are written Python-only with week/arena = NA and a note.

Run:  uv run python arena_analysis/build_frame_table.py
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MICE = {"042025_1mp_arena_compare": "1mp", "042025_2mp_arena_compare": "2mp"}
BATCHES = ["w8", "w9", "w10"]


def parse_segment(label):
    """'week8_O' -> (8, '2D');  'week8' -> (8, '3D');  NaN -> (None, None)."""
    if not isinstance(label, str):
        return None, None
    m = re.match(r"week(\d+)(_O)?$", label.strip())
    if not m:
        return None, None
    return int(m.group(1)), ("2D" if m.group(2) else "3D")


def fit_offset(py_lab, mat_lab, max_shift=10):
    """Find the integer offset o (py row i  <->  mat row i+o) that maximizes the
    adjusted mutual information between the two pipelines' cluster-label
    sequences. If the pipelines share windowing, AMI peaks sharply at the true
    offset; the peak value is itself the cross-pipeline agreement. Returns
    (best_offset, best_ami)."""
    a = np.asarray(py_lab)
    b = np.asarray(mat_lab)
    best_o, best_ami = 0, -1.0
    for o in range(-max_shift, max_shift + 1):
        if o >= 0:
            x, y = a[: len(a) - o], b[o:]
        else:
            x, y = a[-o:], b[: len(b) + o]
        m = min(len(x), len(y))
        if m < 100:
            continue
        ami = adjusted_mutual_info_score(x[:m], y[:m])
        if ami > best_ami:
            best_o, best_ami = o, ami
    return best_o, best_ami


def load_batch(mouse_dir, mouse, batch):
    base = DATA / mouse_dir / f"arena_compare_{batch}_stitched"
    py = pd.read_csv(base / "Cluster_detail_results.csv")
    mat_path = base / "mat_results" / "Cluster_detail_results.csv"

    if not mat_path.exists():
        df = pd.DataFrame({
            "mouse": mouse, "batch": batch, "row": np.arange(len(py)),
            "week": pd.NA, "arena": pd.NA,
            "py_cluster": py["ClusterIdx"].to_numpy(), "mat_cluster": pd.NA,
        })
        return df, "MATLAB MISSING -> python-only, no segment labels"

    mat = pd.read_csv(mat_path)
    py_c = py["ClusterIdx"].to_numpy()
    mat_c = mat["ClusterIdx"].to_numpy()
    off, ami = fit_offset(py_c, mat_c)

    # Align: py row i  <->  mat row i+off. Keep the overlapping span.
    if off >= 0:
        lo_py, lo_mat = 0, off
    else:
        lo_py, lo_mat = -off, 0
    n = min(len(py_c) - lo_py, len(mat_c) - lo_mat)
    seg = mat["Folder_Name"].iloc[lo_mat: lo_mat + n].tolist()
    weeks, arenas = zip(*(parse_segment(s) for s in seg))
    df = pd.DataFrame({
        "mouse": mouse, "batch": batch, "row": np.arange(n),
        "week": weeks, "arena": arenas,
        "py_cluster": py_c[lo_py: lo_py + n],
        "mat_cluster": mat_c[lo_mat: lo_mat + n],
    })
    dropped = df["arena"].isna().sum()
    df = df.dropna(subset=["arena"]).reset_index(drop=True)
    return df, f"offset={off:+d} AMI={ami:.3f}; dropped {dropped} boundary frames"


def main():
    parts = []
    print(f"{'batch':>10}  {'rows':>6}  {'py_k':>4} {'mat_k':>5}  note")
    for mouse_dir, mouse in MICE.items():
        for batch in BATCHES:
            df, note = load_batch(mouse_dir, mouse, batch)
            parts.append(df)
            pyk = df["py_cluster"].nunique()
            matk = df["mat_cluster"].nunique() if df["mat_cluster"].notna().any() else 0
            print(f"{mouse+'/'+batch:>10}  {len(df):>6,}  {pyk:>4} {matk:>5}  {note}")

    full = pd.concat(parts, ignore_index=True)
    out = ROOT / "frame_table.csv"
    full.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(full):,} frames)")
    # quick arena balance per labeled batch
    bal = (full.dropna(subset=["arena"])
               .groupby(["mouse", "batch", "arena"]).size().unstack("arena"))
    print("\nFrames per arena:")
    print(bal.to_string())


if __name__ == "__main__":
    main()
