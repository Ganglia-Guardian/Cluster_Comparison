---
name: arena-exclusivity-analysis
description: arena_analysis — how 2D/3D/shared cluster exclusivity is computed and the first result
metadata:
  type: project
---

`arena_analysis/cluster_arena_exclusivity.py` classifies each MATLAB cluster (per
mouse, per batch — cluster ids are only meaningful within one clustering) as
3D-exclusive / 2D-exclusive / shared / grey from its arena occupancy
`occ3d = p3/(p3+p2)` where `p3,p2` are frames normalized by each arena's total
(corrects for arena time imbalance). Default bands: `occ3d>=0.80` 3D-excl,
`<=0.20` 2D-excl, `0.35–0.65` shared, else grey (tunable via `--excl/--shared`).
Outputs to `arena_analysis/output/exclusivity/`.

First result: the large majority of behavior time is in **shared** clusters
(~75–91% per batch), with a consistent asymmetry — several robust 3D-exclusive
clusters exist but almost no 2D-exclusive ones (flat-arena behavior is ~a subset
of 3D-available behavior). Goal beyond this: analyze how shared-cluster behavior
changes differently by arena across the progression weeks within each batch.
Depends on [[arena-pipelines-not-frame-aligned]].
