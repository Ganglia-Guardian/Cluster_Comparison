---
name: arena-pipelines-not-frame-aligned
description: arena_analysis data — MATLAB vs Python cluster outputs are NOT frame-aligned; only MATLAB labels segments
metadata:
  type: project
---

In `arena_analysis/data`, each stitched batch (`<mouse>_arena_compare_<wN>_stitched`)
holds two cluster outputs: top-level `Cluster_detail_results.csv` (Python) and
`mat_results/Cluster_detail_results.csv` (MATLAB). They are NOT frame-aligned —
adjusted mutual information between their per-frame `ClusterIdx` is ~0.02 (flat
across ±2000-frame offsets; no peak). Both emit instantaneous per-frame labels
(mean dwell ~1.2 frames). So a frame-level cross-pipeline confusion matrix is
impossible; compare at the conclusion level (per-cluster arena occupancy / verdict
distributions) instead.

Only the MATLAB pipeline labels each frame's segment via `Folder_Name` (`weekN` =
3D arena, `weekN_O` = flat/2D arena). The Python pipeline's `Folder_Name` is a
single constant and its harp-clock `Timestamp` does not cleanly map to the twelve
900 s segments — segmenting Python frames is unsolved (user was unsure how their
PY pipeline tracks segments). `2mp/w10` has no MATLAB output. Cross-arena analysis
currently runs MATLAB-only: see [[arena-exclusivity-analysis]].
