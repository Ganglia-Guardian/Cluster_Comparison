---
name: cluster-group-dendrogram
description: cluster_group_analysis — per-(mouse,batch) kinematic dendrograms of clusters, then arena/TBA/week readouts per branch
metadata:
  type: project
---

New fork (2026-07-06) in `cluster_group_analysis/` (own copy of the flat MATLAB
data under `data/<mouse>_arena_compare/arena_compare_w<N>/`; 14 batches have data,
`042025_2mp/w10` missing csv). Approach: merge each MOUSE's clusters (POOLED across
its w8/w9/w10 batches, ~99-156 clusters) into one dendrogram, then characterize
each merged BRANCH by arena purity / TBA / week. Pooling batches gives full week
coverage 8-24 (batches sample interleaved weeks) and a holistic per-mouse tree;
a leaf is a (batch,cluster) since cluster ids aren't cross-batch comparable, but
source batch never enters the distance (dendrogram shows a batch-color strip that
confirms batches interleave across all branches, i.e. merge is behavior- not
session-driven).

Pipeline (run in order; `common.py` holds discovery + loaders, reuses repo-root
`feature_extraction` and the `parse_segment` weekN/_o logic):
1. `build_cluster_table.py` reads every ~3.6GB `session_1_out.mat` ONCE and caches
   `cluster_features.csv` (per-cluster: n_2d,n_3d,occ3d + 4 mean func-features) and
   `cluster_week_counts.csv` (long cluster×week×arena counts). 697 clusters total.
2. `dendrogram.py` — per MOUSE: z-score the 4 func-features WITHIN mouse (all
   pooled clusters), Ward linkage, cut into `--k` branches (default 6), plot tree +
   aligned batch/occ3d/TBA strips. GOTCHA: strip x-extent must be [0,10*n] to match
   scipy leaf positions 10*i+5, and colorbars must go in a reserved gridspec column
   (fig.colorbar ax=... steals width from ONE strip -> misaligns it). Writes
   `cluster_branches.csv`. Merge basis = kinematic centroids incl. TBA (user choice).
3. `branch_analysis.py` — per-mouse, per-branch purity/TBA/week + stats. Output in
   output/branches/ (branch_summary.csv, mouse_stats.csv, per-mouse panels, overview).

Per-WEEK extension (2026-07-06): `build_cluster_table.py` also caches
`cluster_week_features.csv` (per cluster,week,arena: mean features + n) since a
cluster's centroid drifts over disease weeks. Each week maps to exactly one batch
(w8->8,11,14,17,20,23; w9->9,12..24; w10->10,13,16,19,22; verified no week spans 2
batches). `week_dendrogram.py` builds a dendrogram per (mouse,week) from
week-specific centroids (drop clusters <15 frames/week, k=5), writes
output/week_dendrogram/<mouse>/w<NN>.png + week_stats.csv + summary.png. Summary
(3 line panels vs week, one line/mouse): median TBA cleanly separates controls
(~0.30, flat) from MP (decline to ~0.15) at weekly resolution — bradykinesia; 3mp
occ3d-eta² pinned near 0 every week (branches never arena-segregate).

`lc_low_tba_analysis.py` — the low-TBA/2D control group the user flagged = the
lowest-mean-TBA BRANCH in each control (1lc branch 3: 30 clusters, 11.7% time,
occ3d 0.43 mild-2D; 2lc branch 2: 20 clusters, 5.5%, occ3d 0.23 clear-2D). Both
have mean TBA ~0.06 vs ~0.34 rest. Kinematic signature: very low TBA (z~-1.6) but
PRESERVED/elevated AP-accel (z~+0.5) and near-normal y-gyro -> a specific
low-vigour flat-arena behaviour, NOT global immobility. Week frame-share is spiky
(present every week, big session-to-session swings, no monotonic trend — expected
for controls). Output: output/lc_low_tba/ (lc_low_tba.png, low_vigour_clusters.csv).

Design note: merge features INCLUDE TBA, so branch TBA-homogeneity is circular;
we report TBA eta² only as a CEILING to read the (independent) occ3d eta² against.
occ3d and week are NOT in the distance -> honest readouts.

Results (per-mouse pooled, k=6 -> 30 branches over 5 mice): occ3d eta² (branch
explains cluster occ3d) median 0.21 (range 0.07–0.38), vs TBA ceiling 0.67 —
kinematic branches capture ~a fifth to a third of arena-preference variance,
KW-significant in 3/5 mice. 3mp lowest (0.07, non-sig; branches arena-agnostic);
2lc highest (0.38). Most branches mixed (23/30), but arena-pure branches DO exist
(4 primarily-2D, 3 primarily-3D). Week profiles differ across branches in ALL 5
mice (chi² p<<0.05, Cramér's V ~0.21). Builds on [[arena-exclusivity-analysis]],
[[arena-tba-features]].
