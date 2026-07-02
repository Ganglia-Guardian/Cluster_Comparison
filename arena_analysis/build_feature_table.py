"""Attach per-bin functional features (incl. total body acceleration) to the
MATLAB arena cluster frames, for every batch.

For each `<mouse>_arena_compare/arena_compare_w<N>/`, we bin the features in
session_1_out.mat (60 samples/window -> one value per cluster row) via
feature_extraction.combine_results, attach week/arena from Folder_Name, drop
NaN-segment boundary frames, and concatenate. Batches with an empty/missing
folder are skipped.

Output: arena_analysis/frame_features.csv with columns
    mouse, batch, week, arena, cluster, anterior_posterior_x_accel,
    dorsal_ventral_y_accel, y_gyro, TotAccelBA

Run:  uv run python arena_analysis/build_feature_table.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))          # import the repo-root extractor
from feature_extraction import FEATURE_NAMES, combine_results

from cluster_arena_exclusivity import (BATCHES, MICE, mat_csv, parse_segment,
                                       session_mat)


def main():
    parts = []
    for mouse_dir, mouse in MICE.items():
        for batch in BATCHES:
            mat, csv = session_mat(mouse_dir, batch), mat_csv(mouse_dir, batch)
            if not (mat.exists() and csv.exists()):
                print(f"  {mouse}/{batch}: no MATLAB output, skipped")
                continue
            clu = pd.read_csv(csv)
            df = combine_results(mat, cb_matrix=clu)        # appends binned features
            wk, ar = zip(*(parse_segment(s) for s in df["Folder_Name"]))
            df = df.assign(mouse=mouse, batch=batch, week=wk, arena=ar) \
                   .dropna(subset=["arena"]) \
                   .rename(columns={"ClusterIdx": "cluster"})
            parts.append(df[["mouse", "batch", "week", "arena", "cluster", *FEATURE_NAMES]])
            print(f"  {mouse}/{batch}: {len(df):,} frames, "
                  f"TBA median={df['TotAccelBA'].median():.3f}")

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(ROOT / "frame_features.csv", index=False)
    print(f"\nWrote {ROOT / 'frame_features.csv'}  ({len(out):,} frames)")


if __name__ == "__main__":
    main()
