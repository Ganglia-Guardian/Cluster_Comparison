"""Side-by-side feature trajectories (2D vs 3D arena) for the clusters whose
chosen feature declines most over the measured weeks.

Generalizes the TBA-contraction plots to any of the binned functional features.
For each dataset (mouse, batch) we rank clusters by how steeply the feature
contracts -- the POOLED (both arenas) slope of the per-week cluster mean, most
negative first -- and take the top `n`. Then two line plots side by side, one
y-axis:

    left  : each top cluster's weekly feature mean, measured in the 2D arena
    right : the same clusters, measured in the 3D arena

Each cluster keeps one colour across both panels; a shared legend names them.
Under-sampled cluster-weeks (< --min-frames) are gaps. With --complete, only
clusters well-sampled in every week of both arenas are eligible (gap-free).

IMU axes are BODY-RELATIVE (anchored to the mouse, not a global frame):
    anterior_posterior_x_accel  fore-aft accel   (gravity component ~ pitch/tilt)
    dorsal_ventral_y_accel      up-down accel     (gravity component ~ posture)
    y_gyro                      body-relative angular velocity (turning)
    TotAccelBA                  total body acceleration magnitude (vigor)
The accel/gyro axes are signed, so their per-window mean can cancel; --abs uses
the mean magnitude instead, the better readout for turning/tilt intensity.

Figures: output/feature_contraction_lines/<mouse>/<feature>/week<startweek>.jpeg

Run:
    uv run python arena_analysis/plot_feature_contraction_lines.py
    uv run python arena_analysis/plot_feature_contraction_lines.py --feature y_gyro --abs
    uv run python arena_analysis/plot_feature_contraction_lines.py --feature all --n 5
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from utils import save_figure          # noqa: E402

OUT = ROOT / "output" / "feature_contraction_lines"
ARENAS = ["2D", "3D"]

# feature -> (folder tag, human-readable axis label)
FEATURES = {
    "TotAccelBA": ("tba", "total body acceleration (TBA)"),
    "anterior_posterior_x_accel": ("ap_accel", "anterior-posterior accel (fore-aft, body-rel.)"),
    "dorsal_ventral_y_accel": ("dv_accel", "dorsal-ventral accel (up-down, body-rel.)"),
    "y_gyro": ("y_gyro", "y gyroscope (body-relative angular velocity)"),
}


def cluster_week_feat(frames, feature, min_frames, use_abs):
    """(cluster, week) -> mean of the feature (or its magnitude), well-sampled cells."""
    vals = frames[feature].abs() if use_abs else frames[feature]
    tmp = frames.assign(_v=vals)
    g = tmp.groupby(["cluster", "week"])
    cw = g["_v"].mean().rename("mean_v").reset_index()
    cw["n"] = g.size().values
    return cw[cw["n"] >= min_frames]


def complete_clusters(frames, feature, weeks, min_frames, use_abs):
    """Clusters well-sampled in EVERY week of BOTH arenas (gap-free both panels)."""
    full = None
    for arena in ARENAS:
        cw = cluster_week_feat(frames[frames["arena"] == arena], feature, min_frames, use_abs)
        cov = cw.groupby("cluster")["week"].nunique()
        present_all = set(cov[cov == len(weeks)].index)
        full = present_all if full is None else (full & present_all)
    return full or set()


def contraction_slope(cw, min_weeks):
    """Per cluster: slope of the weekly mean vs week (most negative = steepest decline)."""
    out = {}
    for c, gp in cw.groupby("cluster"):
        gp = gp.sort_values("week")
        if len(gp) < min_weeks:
            continue
        y = gp["mean_v"].to_numpy(float)
        if np.allclose(y, y[0]):
            continue
        out[int(c)] = np.polyfit(gp["week"].to_numpy(float), y, 1)[0]
    return pd.Series(out, name="slope")


def trajectory(frames_arena, feature, clusters, weeks, min_frames, use_abs):
    """(week x cluster) feature mean for the clusters in one arena; gaps stay NaN."""
    cw = cluster_week_feat(frames_arena, feature, min_frames, use_abs)
    cw = cw[cw["cluster"].isin(clusters)]
    mat = cw.pivot(index="week", columns="cluster", values="mean_v")
    return mat.reindex(index=weeks, columns=clusters)


def plot_dataset(frames, mouse, batch, feature, n, min_frames, min_weeks, complete, use_abs):
    weeks = sorted(frames["week"].unique())
    start_week = int(min(weeks))
    tag, label = FEATURES[feature]

    pooled = cluster_week_feat(frames, feature, min_frames, use_abs)
    slopes = contraction_slope(pooled, min_weeks)
    if complete:
        keep = complete_clusters(frames, feature, weeks, min_frames, use_abs)
        slopes = slopes[slopes.index.isin(keep)]
    if slopes.empty:
        print(f"  {mouse}/{batch} [{tag}]: no clusters fit; skipped")
        return
    top = slopes.sort_values().head(n).index.tolist()

    traj = {a: trajectory(frames[frames["arena"] == a], feature, top, weeks,
                          min_frames, use_abs) for a in ARENAS}
    ymax = max(t.max().max() for t in traj.values())
    ymin = min(t.min().min() for t in traj.values())
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
    colors = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(top)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, arena in zip(axes, ARENAS):
        t = traj[arena]
        for c in top:
            ax.plot(t.index, t[c], marker="o", ms=5, color=colors[c], label=f"cluster {c}")
        ax.set(xlabel="week", title=f"{arena} arena", xticks=weeks)
    ylab = f"mean |{label}|" if use_abs else f"mean {label}"
    axes[0].set_ylabel(ylab)
    axes[0].set_ylim(ymin - pad, ymax + pad)
    axes[1].legend(title="cluster", loc="best", fontsize=8)
    vis = "complete-visibility " if complete else ""
    fig.suptitle(f"{mouse}  {batch} (weeks {start_week}+): top {len(top)} {vis}"
                 f"steepest-declining clusters by {label}, 2D vs 3D", y=1.0)
    fig.text(0.5, -0.02, "IMU axes are body-relative (anchored to the mouse), "
             "not a global x-y-z frame.", ha="center", fontsize=7, style="italic")
    fig.tight_layout()

    out_dir = OUT / mouse / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"week{start_week}{'_complete' if complete else ''}.jpeg"
    save_figure(fig, path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {mouse}/{batch} [{tag}]: clusters {top} -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature", default="all", choices=["all", *FEATURES],
                    help="which feature to plot (default: all four)")
    ap.add_argument("--n", type=int, default=5, help="top declining clusters to plot")
    ap.add_argument("--min-frames", type=int, default=25,
                    help="min frames for a cluster-week mean to be plotted")
    ap.add_argument("--min-weeks", type=int, default=4,
                    help="min well-sampled weeks for a cluster to be rankable")
    ap.add_argument("--complete", action="store_true",
                    help="only clusters well-sampled in EVERY week of BOTH arenas")
    ap.add_argument("--abs", dest="use_abs", action="store_true",
                    help="rank/plot mean MAGNITUDE (use for signed gyro/tilt axes)")
    args = ap.parse_args()

    features = list(FEATURES) if args.feature == "all" else [args.feature]
    ff = pd.read_csv(ROOT / "frame_features.csv")
    for feature in features:
        for (mouse, batch), frames in ff.groupby(["mouse", "batch"]):
            plot_dataset(frames, mouse, batch, feature, args.n, args.min_frames,
                         args.min_weeks, args.complete, args.use_abs)
    print(f"\nWrote figures under {OUT}/")


if __name__ == "__main__":
    main()
