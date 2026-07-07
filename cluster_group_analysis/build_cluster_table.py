"""Read every (mouse, batch) MATLAB clustering ONCE and cache two per-cluster
tables that the dendrogram + branch analysis run off of (the .mat files are
~3.6 GB each, so we don't want to touch them again).

Outputs (in cluster_group_analysis/):
    cluster_features.csv       one row per (mouse, batch, cluster):
                               n_frames, n_2d, n_3d, occ3d, + 4 mean features
    cluster_week_counts.csv    long: (mouse, batch, cluster, week, arena, n)
    cluster_week_features.csv  long: (mouse, batch, cluster, week, arena, n, +4
                               feature means) -- week-specific centroids for the
                               per-week dendrograms

Run:  uv run python cluster_group_analysis/build_cluster_table.py
"""
from pathlib import Path

import pandas as pd

from common import (BATCHES, MICE, ROOT, cluster_centroids,
                    load_frames_with_features, week_arena_counts,
                    week_arena_features)


def main():
    feat_parts, wk_parts, wkf_parts = [], [], []
    for mouse_dir, mouse in MICE.items():
        for batch in BATCHES:
            frames = load_frames_with_features(mouse_dir, batch)
            if frames is None:
                print(f"  {mouse}/{batch}: no MATLAB output, skipped")
                continue
            cen = cluster_centroids(frames)
            cen.insert(0, "batch", batch)
            cen.insert(0, "mouse", mouse)
            feat_parts.append(cen)

            wk = week_arena_counts(frames)
            wk.insert(0, "batch", batch)
            wk.insert(0, "mouse", mouse)
            wk_parts.append(wk)

            wkf = week_arena_features(frames)
            wkf.insert(0, "batch", batch)
            wkf.insert(0, "mouse", mouse)
            wkf_parts.append(wkf)
            print(f"  {mouse}/{batch}: {len(cen)} clusters, {len(frames):,} frames, "
                  f"TBA median={cen['TotAccelBA'].median():.3f}")

    feats = pd.concat(feat_parts, ignore_index=True)
    weeks = pd.concat(wk_parts, ignore_index=True)
    wkfeats = pd.concat(wkf_parts, ignore_index=True)
    feats.to_csv(ROOT / "cluster_features.csv", index=False)
    weeks.to_csv(ROOT / "cluster_week_counts.csv", index=False)
    wkfeats.to_csv(ROOT / "cluster_week_features.csv", index=False)
    print(f"\nWrote {ROOT/'cluster_features.csv'} ({len(feats)} clusters), "
          f"{ROOT/'cluster_week_counts.csv'} ({len(weeks)} rows), "
          f"{ROOT/'cluster_week_features.csv'} ({len(wkfeats)} rows)")


if __name__ == "__main__":
    main()
