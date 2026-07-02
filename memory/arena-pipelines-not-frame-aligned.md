---
name: arena-pipelines-not-frame-aligned
description: arena_analysis data layout (flat, MATLAB-only), the 5 mice, and the Folder_Name suffix gotcha
metadata:
  type: project
---

CURRENT layout (flat, updated 2026-07): `arena_analysis/data/<mouse>_arena_compare/
arena_compare_w<N>/` holds the MATLAB `Cluster_detail_results.csv` and
`session_1_out.mat` DIRECTLY (no `mat_results/` subdir, no `_stitched` suffix).
The Python pipeline was DROPPED — analysis is MATLAB-only. Path helpers
`batch_dir/mat_csv/session_mat` + auto-discovery (`discover_mice/discover_batches`,
`MICE`/`BATCHES`) live in `cluster_arena_exclusivity.py`; all scripts import from
there. Empty/missing batch folders are skipped (currently `042025_2mp/w10` and
`042025_3mp/w10`).

Mice (5): MitoPark = `1mp`,`2mp`,`3mp` (042025_*); littermate controls = `1lc`,
`2lc` (042425_*). Batches w8/w9/w10 sample interleaved weeks (w8→8,11,14,17,20,23;
w9→9,12,15,18,21,24; w10→10,13,16,19,22/24). Controls unlock a real MitoPark-vs-
control contrast (previously only within-mouse progression, n=2 MitoPark).

GOTCHA: the 2D-arena `Folder_Name` suffix is written inconsistently across mice —
`week8_O` (1mp/2mp), `week8_o` (3mp), `week_8_o` (1lc/2lc); 3D is always `weekN`.
`parse_segment` regex `week_?(\d+)(_o)?$` (re.I) handles all variants. If it
mis-parses, a mouse degenerates to 100% 3D-exclusive (all 2D frames dropped).

Historical: when the Python pipeline still existed its per-frame ClusterIdx was
NOT frame-aligned to MATLAB's (AMI ~0.02, flat across offsets), which is why we
went MATLAB-only. Both emit instantaneous labels (mean dwell ~1.2 frames).
See [[arena-exclusivity-analysis]], [[arena-transitions]].
