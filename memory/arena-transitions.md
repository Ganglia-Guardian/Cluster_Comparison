---
name: arena-transitions
description: arena_analysis Fork 2 — cluster-transition stats by arena (successor contraction) + null TBA-flow
metadata:
  type: project
---

`arena_analysis/arena_transitions.py` reuses the repo's transition machinery
(fanout_by_week, rarefied_diversity from cluster_transition_compare /
cluster_successor_diversity) on per-arena subsets. Isolation trick: in the MAT
frame table Folder_Name is the segment (weekN=3D, weekN_O=2D) and build_transitions
only links same-Folder_Name consecutive frames, so masking the other arena's rows
to NaN (preserving dropped-frame breaks) cleanly isolates one arena's transitions.

Key Fork-2 result (strong, convincing — replicates the historical MitoPark
successor-contraction finding): both raw fan-out and occupancy-controlled
rarefied successor richness CONTRACT over disease weeks in BOTH arenas (richness
Spearman <= -0.77, significant in 8/10 mouse*batch*arena cells; rarefaction kept
64-80% of sources at depth 20). Arena asymmetry is the same mouse-dependent split
seen everywhere (1mp sharper in 2D, 2mp in 3D), but contraction is robust
regardless of arena. Output: output/transitions/.

`arena_analysis/arena_transitions_by_clustertype.py` is the key dissociation
(--split median 0.5 on occ3d; --arena-mode home|all; --link bridge|adjacent).
Splitting the codebook into primarily-2D (occ3d<0.5) vs primarily-3D (>=0.5),
~20-30 clusters each: the primarily-3D (volumetric) repertoire's successor
structure CONTRACTS strongly and consistently over disease (rarefied richness
Spearman more negative than primarily-2D in 5/5 datasets, BOTH modes; 3D median
rho ~ -0.9 home / -1.0 all, 2D median ~ +0.1 home / -0.3 all), while the
primarily-2D repertoire stays ~flat. This is the FIRST arena result consistent
across both mice (not the usual mouse-dependent split). 'home' mode (ignore
opposite arena) sharpens the contrast; 'all' mode dilutes 2D by importing
2D-leaning clusters' 3D-arena frames. Caveat: n=2 mice (5 batches).
Output: output/transitions_by_clustertype/<mode>/.

CONTROL (`--membership all`, = all bins split by arena only, no cluster subset;
reproduces arena_transitions.py): BOTH arenas contract strongly (richness rho
~-0.9 in both, all 5 datasets). So at the pure-arena level there is NO
dissociation. The smoking gun: all-2D-bins contracts (rho ~-0.94) but
primarily-2D-CLUSTERS is flat (rho ~+0.10) -> the 2D arena's contraction is
carried entirely by the 3D-type (volumetric) behaviours that also occur in it.
Conclusion: vulnerability (successor contraction) is a property of the BEHAVIOUR
(volumetric cluster type), not the recording arena; you must split by cluster
type to see it. Output: output/transitions_by_clustertype/all_bins/.

`arena_analysis/arena_transition_tba.py` tested "do successors degenerate into
lower-TBA clusters?" using static per-cluster TBA as delta=TBA(tgt)-TBA(src).
Result: NULL — mean_delta ~0.0000, downhill fraction 0.49-0.53 everywhere, no
consistent week/arena trend. This is largely structural: transitions are near-
reversible, so aggregate mean delta cancels and this metric can't detect the
effect. The vigour-degeneration signal lives at the occupancy/within-cluster
level ([[arena-tba-features]]), not as an aggregate transition bias. A per-source
conditional successor-TBA trend or net-flow-to-sinks metric could probe it
further. Builds on [[arena-pipelines-not-frame-aligned]].
