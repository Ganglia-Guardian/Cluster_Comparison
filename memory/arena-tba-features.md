---
name: arena-tba-features
description: arena_analysis — how per-bin TBA / functional features are extracted and merged
metadata:
  type: project
---

`feature_extraction.py` (repo root) extracts 4 per-sample functional features
from `mat_results/session_1_out.mat` (`StructData/func`): anterior_posterior_x_accel,
dorsal_ventral_y_accel, y_gyro, and **TotAccelBA** (total body accel, stored as
log). Each is 60 samples per cluster window; `bin_features()` averages by 60 to
one value per cluster row (TotAccelBA is exp'd before averaging -> mean linear
accel). It was refactored so output paths are arguments and `combine_results`
returns a DataFrame (was hardcoded + had an undefined `output_path` bug).

`arena_analysis/build_feature_table.py` runs it over every MAT batch and writes
`arena_analysis/frame_features.csv` (mouse, batch, week, arena, cluster + 4
features). Per-cluster mean TBA correlates with 3D occupancy (Spearman rho=0.31,
p=4e-7, n=255) — 3D-leaning clusters are more vigorous. Builds on
[[arena-exclusivity-analysis]].

`arena_analysis/arena_tba_vulnerability.py` is the key result: within-cluster /
whole-arena TBA declines over disease weeks in EVERY mouse/batch/arena
(collective Spearman rho -0.20 to -1.00; TBA ~halves first->last week) — a clean
bradykinesia signal that occupancy never gave. Crucially the decline is stronger
& more significant in 3D than 2D in 4/5 batches (tie in the 5th) — first arena
effect consistent across both mice, supporting "3D detects MitoPark better than
2D". Caveat: the per-cluster volumetric gradient (occ3d vs decline) is
inconsistent, so the 3D advantage is arena-level, not cleanly that 3D-exclusive
clusters are individually most vulnerable. Disease axis = within-mouse week
progression (no arena controls). Output: output/tba_vulnerability/.

`arena_analysis/plot_feature_contraction_lines.py` (generalizes the old
plot_tba_contraction_lines.py; `--feature` over all 4 features, `--complete`,
`--abs`, `--n`) draws side-by-side 2D-vs-3D weekly trajectories of the
steepest-declining clusters. Finding: only TBA contracts cleanly over disease;
dorsal-ventral accel (tilt) and y-gyro (turning) are flat/noisy — consistent
with bradykinesia being a movement-magnitude loss. y-gyro signed mean is
non-zero (~0-74) so --abs isn't required. Output:
output/feature_contraction_lines/<mouse>/<feature>/.
