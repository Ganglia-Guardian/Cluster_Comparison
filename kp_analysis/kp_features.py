"""Internal-geometry features from EKS 3D keypoint output (multicam_3d_results.csv).

The EKS export is frame-unanchored (no global/gravitational x-y-z), so we describe
each frame by the *internal* geometry of the skeleton: the pairwise Euclidean
distances between keypoints. Pairwise distances are invariant to translation and
rotation, which lets us compare sessions recorded in different coordinate frames
(wk8lc and wk8mp live in entirely different coordinate ranges).

Two feature sets per frame:
    posture  -- the C(k, 2) pairwise distances, size-normalized
    motion   -- the time-derivative of those distances (deformation rate; a
                frame-invariant stand-in for "speed", since true locomotor speed
                is unrecoverable without a global frame)

Acquisition facts (week 8):
    * keypoint video : 30 fps, ~60 min, 108,000 frames
    * IMU/func windows: ~0.345 s each, ~5,958 windows, ~35 min, shared start
      trigger with the video (window 0 == frame 0)
So only the first ~35 min (~63k frames) of video overlaps the func labels; later
frames have no func window and are dropped at alignment time.

Missing data: paw_front_left and back_base are 100% absent in wk8mp, so by default
we restrict to the 15 keypoints common to both sessions. Remaining scattered gaps
are linearly interpolated over time per coordinate.

File layout: DLC/Lightning-Pose 3-row header
    row 0 scorer | row 1 bodyparts | row 2 coords (x, y, z, *_posterior_var)
"""
import numpy as np
import pandas as pd

FPS = 30.0
# Entirely missing in wk8mp -> excluded so wk8lc and wk8mp share a feature space.
DROP_IN_MP = ["paw_front_left", "back_base"]


def load_keypoints(filepath, keypoints=None, interpolate=True):
    """Load multicam_3d_results.csv into coordinate and variance arrays.

    Parameters
    ----------
    filepath : path to a multicam_3d_results.csv
    keypoints : list of bodypart names to keep, in order. Defaults to every
        bodypart except those in DROP_IN_MP (the set common to both sessions).
    interpolate : if True, linearly interpolate NaN gaps over time per coord
        (the ends are held flat via numpy's interp clamping).

    Returns
    -------
    coords : (n_frames, n_kp, 3) float array of x, y, z
    var    : (n_frames, n_kp, 3) float array of posterior variances
    names  : list of kept keypoint names (len n_kp)
    """
    df = pd.read_csv(filepath, header=[1, 2], index_col=0)

    all_kp = list(dict.fromkeys(c[0] for c in df.columns))
    if keypoints is None:
        keypoints = [k for k in all_kp if k not in DROP_IN_MP]
    missing = [k for k in keypoints if k not in all_kp]
    if missing:
        raise KeyError(f"Requested keypoints not in file: {missing}")

    xyz = ["x", "y", "z"]
    var_cols = ["x_posterior_var", "y_posterior_var", "z_posterior_var"]
    coords = np.stack([df[[(k, c) for c in xyz]].to_numpy() for k in keypoints], axis=1)
    var = np.stack([df[[(k, c) for c in var_cols]].to_numpy() for k in keypoints], axis=1)

    if interpolate and np.isnan(coords).any():
        coords = _interp_gaps(coords)
    return coords, var, keypoints


def _interp_gaps(coords):
    """Linearly interpolate NaNs along time (axis 0) for each (kp, axis)."""
    out = coords.copy()
    t = np.arange(out.shape[0])
    for k in range(out.shape[1]):
        for a in range(out.shape[2]):
            col = out[:, k, a]
            nan = np.isnan(col)
            if nan.all() or not nan.any():
                continue
            col[nan] = np.interp(t[nan], t[~nan], col[~nan])
    return out


def pairwise_distances(coords):
    """Pairwise Euclidean distances per frame.

    coords : (n_frames, n_kp, 3)
    Returns (dists, pairs):
        dists : (n_frames, n_pairs), n_pairs = n_kp*(n_kp-1)/2 (upper triangle)
        pairs : list of (i, j) keypoint-index tuples, column-aligned to dists
    """
    n_kp = coords.shape[1]
    i_idx, j_idx = np.triu_indices(n_kp, k=1)
    # loop over the ~105 pairs to keep memory flat vs an (F, K, K) blowup
    dists = np.empty((coords.shape[0], len(i_idx)), dtype=float)
    for col, (i, j) in enumerate(zip(i_idx, j_idx)):
        dists[:, col] = np.linalg.norm(coords[:, i, :] - coords[:, j, :], axis=1)
    return dists, list(zip(i_idx.tolist(), j_idx.tolist()))


def normalize_scale(dists):
    """Remove fixed body-size / depth scale with a single session-level scalar.

    Divides every distance by the global median distance for the session. This
    strips overall scale (making wk8lc and wk8mp comparable) while preserving
    within-session posture variation, which a per-frame normalization would
    partly destroy. Returns (normalized, scale_scalar).
    """
    scale = float(np.nanmedian(dists))
    return dists / scale, scale


def motion_features(dists, fps=FPS, window=7, polyorder=2):
    """Frame-invariant deformation rate: smoothed time-derivative of distances.

    Savitzky-Golay smooths and differentiates in one pass, avoiding the noise
    amplification of a raw two-frame difference. Result is in distance-units per
    second when fps is given, else per-frame.
    """
    from scipy.signal import savgol_filter

    deriv = savgol_filter(dists, window_length=window, polyorder=polyorder,
                          deriv=1, axis=0)
    return deriv * fps if fps is not None else deriv


# --- alignment to the IMU/func window grid (shared start trigger) ------------

def load_func_windows(detail_csv, folder):
    """Per-window func labels and time bounds for one week-folder.

    Returns a DataFrame with columns [cluster, start, end] where start/end are
    seconds relative to the first window of the folder (so they share the
    video's frame-0 origin). Window end is the next window's start; the final
    window gets the median window duration.
    """
    df = pd.read_csv(detail_csv)
    wk = df[df.Folder_Name == folder].sort_values("Timestamp").reset_index(drop=True)
    if wk.empty:
        raise ValueError(f"No rows for folder {folder!r} in {detail_csv}")
    start = wk.Timestamp.to_numpy() - wk.Timestamp.iloc[0]
    dur = np.diff(start)
    end = np.concatenate([start[1:], [start[-1] + np.median(dur)]])
    return pd.DataFrame({"cluster": wk.ClusterIdx.to_numpy(), "start": start, "end": end})


def assign_frames_to_windows(n_frames, windows, fps=FPS):
    """Map each video frame to a func window index by time (frame f at f/fps s).

    Frames falling inside a real window get that window's row index; frames past
    the last window or inside an inter-window gap get -1. Returns an int array of
    length n_frames.
    """
    t = np.arange(n_frames) / fps
    starts = windows.start.to_numpy()
    ends = windows.end.to_numpy()
    owner = np.searchsorted(starts, t, side="right") - 1
    valid = (owner >= 0) & (owner < len(starts))
    # reject frames that landed in a gap (between a window's end and the next start)
    in_win = np.zeros(n_frames, dtype=bool)
    in_win[valid] = t[valid] < ends[owner[valid]]
    owner[~in_win] = -1
    return owner


def aggregate_to_windows(features, owner, n_windows, reduce="mean"):
    """Collapse per-frame features onto windows using a frame->window owner map.

    reduce='mean' (posture) or 'mean_abs' (motion magnitude). Windows with no
    frames are NaN. Returns (n_windows, n_features).
    """
    vals = np.abs(features) if reduce == "mean_abs" else features
    out = np.full((n_windows, features.shape[1]), np.nan)
    for w in range(n_windows):
        rows = vals[owner == w]
        if len(rows):
            out[w] = rows.mean(axis=0)
    return out


if __name__ == "__main__":
    base = "kp_analysis/data"
    # placeholder func file for mechanics check; real subject (1/2/3) TBD
    func_probe = {"wk8lc": ("data/1lc/Cluster_detail_results.csv", "w8"),
                  "wk8mp": ("data/1mp/Cluster_detail_results.csv", "week_8")}
    for name in ["wk8lc", "wk8mp"]:
        fp = f"{base}/{name}/multicam_3d_results.csv"
        coords, var, names = load_keypoints(fp)
        dists, pairs = pairwise_distances(coords)
        norm, scale = normalize_scale(dists)
        motion = motion_features(norm)
        print(f"=== {name} ===")
        print(f"  frames={coords.shape[0]}  keypoints={len(names)}  pairs={dists.shape[1]}")
        print(f"  scale(median dist)={scale:.2f}  residual NaN={int(np.isnan(coords).sum())}")
        csv, folder = func_probe[name]
        win = load_func_windows(csv, folder)
        owner = assign_frames_to_windows(len(coords), win)
        post_w = aggregate_to_windows(norm, owner, len(win), "mean")
        mot_w = aggregate_to_windows(motion, owner, len(win), "mean_abs")
        print(f"  func windows={len(win)} (folder {folder!r}, PLACEHOLDER subject)")
        print(f"  frames inside IMU span={int((owner >= 0).sum())} / {len(coords)}")
        print(f"  posture windows={post_w.shape} empty={int(np.isnan(post_w).any(1).sum())}"
              f"  motion windows={mot_w.shape}")
