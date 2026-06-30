"""Find IMU/func clusters that are well-aligned to a keypoint POSE cluster.

Overall posture<->func AMI is weak (~0.05), but that is a global average; some
individual IMU behaviors may still have a sharp posture signature. This script
finds them. For each func cluster we take its best-matching pose cluster and
report:
    purity  = P(pose=best | func)         how concentrated the func cluster is
    lift    = purity / P(pose=best)        enrichment over the pose marginal
    jaccard = overlap / union              one-to-one-ness of the two window sets
    p       = hypergeometric survival      P(overlap this large by chance)
A func cluster is "well-aligned" when purity and lift are high with a tiny p.
The pose clusters that anchor the top matches are then described by their most
distinctive pairwise distances (z-scored vs the session mean), so each hit reads
as "IMU behavior #f looks like posture: <stretched / compressed segments>".

Run from repo root:
    C:/ProgramData/anaconda3/python.exe kp_analysis/kp_aligned_clusters.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import kp_features as kpf
import kp_cluster_compare as cc

K = 30              # pose clusters (posture features; best from the AMI sweep)
MIN_WINDOWS = 20    # ignore tiny func clusters (unstable purity)
TOP_N = 12


def pair_names(session):
    """Human-readable 'kpA~kpB' label per distance column (header-only read)."""
    fp = cc.SESSIONS[session][0]
    cols = pd.read_csv(fp, header=[1, 2], index_col=0, nrows=0).columns
    all_kp = list(dict.fromkeys(c[0] for c in cols))
    names = [k for k in all_kp if k not in kpf.DROP_IN_MP]
    i, j = np.triu_indices(len(names), k=1)
    return [f"{names[a]}~{names[b]}" for a, b in zip(i, j)]


def aligned_table(func, pose, min_windows=MIN_WINDOWS):
    """Per func cluster: best pose match + concentration / enrichment stats."""
    N = len(func)
    ct = pd.crosstab(pd.Series(func, name="func"), pd.Series(pose, name="pose"))
    pose_sizes = ct.sum(axis=0)
    rows = []
    for fc in ct.index:
        nj = int(ct.loc[fc].sum())
        if nj < min_windows:
            continue
        best = ct.loc[fc].idxmax()
        cnt = int(ct.loc[fc, best])
        ni = int(pose_sizes[best])
        purity = cnt / nj
        rows.append({
            "func": int(fc), "n_func": nj, "pose": int(best), "n_pose": ni,
            "overlap": cnt, "purity": purity, "lift": purity / (ni / N),
            "jaccard": cnt / (ni + nj - cnt),
            "p": float(hypergeom.sf(cnt - 1, N, ni, nj)),
        })
    res = pd.DataFrame(rows)
    res["sig_bonf"] = res["p"] * len(res) < 0.05   # family = func clusters tested
    return res.sort_values(["lift", "purity"], ascending=False).reset_index(drop=True)


def describe_pose(post, pose, clusters, labels, topm=4):
    """Most distinctive distances (z vs session mean) for each pose cluster."""
    gmean, gstd = post.mean(0), post.std(0)
    out = {}
    for c in clusters:
        z = (post[pose == c].mean(0) - gmean) / gstd
        order = np.argsort(z)
        out[c] = {
            "stretched": [(labels[t], round(float(z[t]), 2)) for t in order[::-1][:topm]],
            "compressed": [(labels[t], round(float(z[t]), 2)) for t in order[:topm]],
        }
    return out


def run(session):
    post, mot, comb, func = cc.build_windows(session)
    pose = cc.cluster(post, K)            # posture features, deterministic seed
    labels = pair_names(session)
    res = aligned_table(func, pose)

    print(f"\n{'='*70}\n{session}  (K={K} pose clusters, {len(np.unique(func))} func "
          f"clusters, {len(func)} windows; {len(res)} func clusters >= "
          f"{MIN_WINDOWS} windows)\n{'='*70}")
    show = res.head(TOP_N).copy()
    show["p"] = show["p"].map(lambda v: f"{v:.1e}")
    show[["purity", "lift", "jaccard"]] = show[["purity", "lift", "jaccard"]].round(3)
    print(show.to_string(index=False))

    top_pose = res.head(TOP_N)["pose"].unique().tolist()
    desc = describe_pose(post, pose, top_pose, labels)
    print("\n-- geometry of the anchoring pose clusters --")
    for c in top_pose:
        s = ", ".join(f"{n}({z:+.1f})" for n, z in desc[c]["stretched"][:3])
        k = ", ".join(f"{n}({z:+.1f})" for n, z in desc[c]["compressed"][:3])
        print(f"  pose {c:2d}: long  -> {s}")
        print(f"          short -> {k}")

    res.to_csv(f"kp_analysis/{session}_aligned_clusters.csv", index=False)
    return res


if __name__ == "__main__":
    for session in cc.SESSIONS:
        run(session)
    print("\nSaved per-session *_aligned_clusters.csv")
