"""
Deproject camera-centroid points onto the arena floor for a range of assumed
camera heights `h`, and plot each result so a good `h` can be chosen by eye.

Background
----------
The camera sits directly above the center of the arena looking straight down.
A mouse standing on top of a column of height `z` is therefore *closer* to the
camera than the floor, so its image position is pushed radially outward
(away from the image center) by the factor  h / (h - z).

To undo that ("deproject"), we pull every apparent point back toward the center:

        true = center + (apparent - center) * (1 - z / h)

where `z` is the column elevation at the mouse's location (looked up from the
elevation map) and `h` is the (unknown, to-be-measured) camera height above the
arena floor. Larger h  -> weaker correction; smaller h -> stronger correction.

We don't know the true pixel<->meter calibration, so the centroid pixel cloud is
mapped onto the elevation map's metric extent by an affine bounding-box fit
(assuming the mouse explores essentially the whole arena over all weeks). This is
approximate but good enough to judge `h` visually; measure the real camera height
later for validation.

Coordinate systems
------------------
The elevation CSV is stored as a 180-degree rotation of the "normal" arena
coordinates, and the image y-axis points down. The exact flip/rotation that makes
the two clouds line up is hard to know a priori, so:
  * The main figure uses ORIENT (below) to place the centroids in the elevation
    frame.
  * An extra "orientation compare" figure shows all four options at one h, so you
    can see which one lines the mouse cloud up with the terrain and set ORIENT.

Nothing here reads the `centroid_analysis/` folder.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import save_figure

# --------------------------------------------------------------------------- #
# CONFIG - edit these
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT  = os.path.join(HERE, "output")

ELEV_CSV     = os.path.join(DATA, "5-30-hi-complex-1-points.csv")
DATASET      = "042425_1lc"                      # subfolder under data/
CENTROID_CSV = os.path.join(DATA, DATASET, "all_weeks_centroid.csv")

# Camera heights to try, in the SAME units as the elevation z (meters here).
# Column tops reach ~0.6 m, so h must be > 0.6. Increase h -> weaker correction.
# Centered on the measured candidate h = 1.915 m, with neighbors for direction.
H_VALUES = [1.9, 1.9075, 1.915, 1.9225, 1.93]

# How to place the centroid cloud into the elevation frame. One of:
#   "rot180"  (default; elevation is a 180-deg rotation of normal coords)
#   "none", "flipx", "flipy"
# If the main overlay looks mirrored / upside down, look at the
# orientation-compare figure and change this.
ORIENT = "rot180"

# Pixel -> meter calibration.
#   "physical": isotropic scale (square nadir pixels) anchored on a measured arena
#               dimension. Preferred once you have a real measurement.
#   "bbox":     older anisotropic fit of the pixel box to the elevation extent.
CALIB_MODE        = "physical"
ARENA_FRONT_BACK_M = 1.625     # measured front-to-back arena dimension (meters)
FRONT_BACK_AXIS    = "y"       # which IMAGE axis is front-to-back: "y" or "x"
# Percentile span of the (filtered) pixel cloud taken to span the arena. 0.5/99.5
# trims a few stray points; assumes the mouse explores ~the full arena.
CALIB_PCTL = (0.5, 99.5)

# Corner-artifact filter: when the tracker loses the mouse it parks the centroid
# at a fixed fallback pixel in an image corner, producing a tight blob of
# near-duplicate points that is detached from the real trajectory cloud. We look
# in each image corner for the densest small bin and flag it as an artifact only
# if it is both sizable AND highly concentrated (most of its neighborhood sits in
# that one bin) - which distinguishes a parked blob from a genuinely busy corner
# where the mouse spends real time.
ARTIFACT_CORNER_FRAC   = 0.14   # corner band = this fraction of the image diagonal
ARTIFACT_BIN_PX        = 5.0    # fine bin size used to find the spike
ARTIFACT_MIN_FRAC      = 0.001  # spike bin must hold >= this fraction of all points
ARTIFACT_CONCENTRATION = 0.30   # ...and >= this fraction of points within the radius
ARTIFACT_REMOVE_PX     = 90.0   # remove everything within this radius of the spike

PLOT_SUBSAMPLE = 40000        # max points drawn per panel (scatter is slow)
SEED = 0

# Final registration: after deprojecting at REGISTER_H, snap the cloud's edges to
# the elevation data's edges with a constant per-axis scale + translation, then
# export an aligned map (x, y, elevation, slope, + week/frame) for slope analysis.
REGISTER_H    = 1.915         # chosen camera height for the final map (meters)
REGISTER_PCTL = (0.5, 99.5)   # percentile taken as the cloud "edge" per axis
SLOPE_RES_M   = 0.01          # grid resolution for the elevation-gradient (slope) map
# --------------------------------------------------------------------------- #


def load_elevation(path, orient_rot180=True):
    """Return (X, Y, Z) arrays of arena column positions in the working frame."""
    arr = np.loadtxt(path, delimiter=",")
    X, Y, Z = arr[:, 0], arr[:, 1], arr[:, 2]
    if orient_rot180:
        # 180-deg rotation about the grid center -> "normal" coordinates.
        X = X.max() + X.min() - X
        Y = Y.max() + Y.min() - Y
    return X, Y, Z


def build_elevation_lookup(X, Y, Z):
    """Interpolator z(x, y): linear inside the hull, nearest outside (no NaNs)."""
    pts = np.column_stack([X, Y])
    lin = LinearNDInterpolator(pts, Z)
    nrst = NearestNDInterpolator(pts, Z)

    def z_at(x, y):
        z = lin(x, y)
        bad = np.isnan(z)
        if np.any(bad):
            z = np.where(bad, nrst(x, y), z)
        return z

    return z_at


def load_centroids(path):
    """Load pixel centroids, drop NaNs. Returns (px, py, meta) row-aligned.

    Columns are [week, aux1, aux2, frame, x_px, y_px]; meta keeps the first four
    so the exported map can be ordered in time for motion analysis.
    """
    df = pd.read_csv(path, header=None)
    px = df.iloc[:, 4].to_numpy(float)
    py = df.iloc[:, 5].to_numpy(float)
    ok = np.isfinite(px) & np.isfinite(py)
    meta = df.loc[ok, [0, 1, 2, 3]].copy()
    meta.columns = ["week", "aux1", "aux2", "frame"]
    return px[ok], py[ok], meta.reset_index(drop=True)


def remove_corner_artifacts(px, py):
    """Drop the tracker's parked "lost the mouse" corner blobs. Returns a mask."""
    n = len(px)
    xmin, xmax = px.min(), px.max()
    ymin, ymax = py.min(), py.max()
    diag = np.hypot(xmax - xmin, ymax - ymin)
    band = ARTIFACT_CORNER_FRAC * diag
    bs = ARTIFACT_BIN_PX

    keep = np.ones(n, dtype=bool)
    corners = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
    for cx, cy in corners:
        in_band = np.hypot(px - cx, py - cy) < band
        if in_band.sum() < ARTIFACT_MIN_FRAC * n:
            continue
        # densest fine bin within the band = candidate parked-fallback spike
        bx = np.floor(px[in_band] / bs).astype(np.int64)
        by = np.floor(py[in_band] / bs).astype(np.int64)
        keys, counts = np.unique(np.stack([bx, by], 1), axis=0, return_counts=True)
        k = counts.argmax()
        spikex = (keys[k, 0] + 0.5) * bs
        spikey = (keys[k, 1] + 0.5) * bs
        spike_ct = counts[k]

        near = np.hypot(px - spikex, py - spikey) < ARTIFACT_REMOVE_PX
        # parked blob: sizable AND most of its neighborhood is that one bin.
        if (spike_ct >= ARTIFACT_MIN_FRAC * n and
                spike_ct >= ARTIFACT_CONCENTRATION * near.sum()):
            keep &= ~near
    return keep


def calibrate(px, py, X, Y):
    """Return a px->meters mapping placing the cloud in the elevation frame."""
    plo, phi = CALIB_PCTL
    x0, x1 = np.percentile(px, [plo, phi])
    y0, y1 = np.percentile(py, [plo, phi])
    acx = 0.5 * (X.min() + X.max())          # arena center (camera axis)
    acy = 0.5 * (Y.min() + Y.max())

    if CALIB_MODE == "physical":
        # One isotropic scale (square nadir pixels) from the measured dimension.
        span_px = (y1 - y0) if FRONT_BACK_AXIS == "y" else (x1 - x0)
        s = ARENA_FRONT_BACK_M / span_px     # meters per pixel
        pcx, pcy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

        def to_m(qx, qy):
            return acx + (qx - pcx) * s, acy + (qy - pcy) * s

        return to_m

    # "bbox": anisotropic fit of the pixel box to the elevation extent.
    Xr, Yr = (X.min(), X.max()), (Y.min(), Y.max())

    def to_m(qx, qy):
        mx = Xr[0] + (qx - x0) / (x1 - x0) * (Xr[1] - Xr[0])
        my = Yr[0] + (qy - y0) / (y1 - y0) * (Yr[1] - Yr[0])
        return mx, my

    return to_m


def orient(mx, my, X, Y, mode):
    """Flip the metric cloud within the arena box to match the elevation frame."""
    cx = 0.5 * (X.min() + X.max())
    cy = 0.5 * (Y.min() + Y.max())
    if mode in ("flipx", "rot180"):
        mx = 2 * cx - mx
    if mode in ("flipy", "rot180"):
        my = 2 * cy - my
    return mx, my


def deproject(mx, my, z_at, center, h, iters=2):
    """Pull apparent metric points toward `center` by (1 - z/h); refine z once."""
    cx, cy = center
    tx, ty = mx.copy(), my.copy()
    for _ in range(iters):
        z = z_at(tx, ty)
        f = np.clip(1.0 - z / h, 1e-3, 1.0)   # h must exceed z
        tx = cx + (mx - cx) * f
        ty = cy + (my - cy) * f
    return tx, ty


def _subsample(*arrays, n=PLOT_SUBSAMPLE):
    m = len(arrays[0])
    if m <= n:
        return arrays
    rng = np.random.default_rng(SEED)
    idx = rng.choice(m, n, replace=False)
    return tuple(a[idx] for a in arrays)


def _terrain(ax, X, Y, Z):
    tc = ax.tricontourf(X, Y, Z, levels=14, cmap="terrain", alpha=0.75)
    ax.scatter(X, Y, s=4, c="k", alpha=0.15, linewidths=0)  # column markers
    ax.set_aspect("equal")
    ax.set_xlim(X.min() - 0.15, X.max() + 0.15)
    ax.set_ylim(Y.min() - 0.15, Y.max() + 0.15)
    return tc


def plot_h_sweep(mx, my, z_at, X, Y, Z, center, h_values, out_path):
    n = len(h_values) + 1
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.8 * nrow))
    axes = np.atleast_1d(axes).ravel()

    # reference panel: no deprojection (h = infinity)
    ax = axes[0]
    _terrain(ax, X, Y, Z)
    sx, sy = _subsample(mx, my)
    ax.scatter(sx, sy, s=2, c="crimson", alpha=0.06, linewidths=0)
    ax.set_title("apparent (no deprojection, h = INF)", fontsize=11)

    for i, h in enumerate(h_values, start=1):
        ax = axes[i]
        tc = _terrain(ax, X, Y, Z)
        tx, ty = deproject(mx, my, z_at, center, h)
        sx, sy = _subsample(tx, ty)
        ax.scatter(sx, sy, s=2, c="crimson", alpha=0.06, linewidths=0)
        ax.set_title(f"h = {h:g}", fontsize=13, fontweight="bold")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Deprojected centroids vs arena elevation  |  dataset {DATASET}  |  "
        f"orient={ORIENT}\n(red = mouse positions, terrain = column height; "
        "pick the h whose red cloud best fits the arena)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def plot_orientation_compare(px, py, to_m, z_at, X, Y, Z, center, h, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, mode in zip(axes.ravel(), ["none", "flipx", "flipy", "rot180"]):
        _terrain(ax, X, Y, Z)
        mx, my = to_m(px, py)
        mx, my = orient(mx, my, X, Y, mode)
        tx, ty = deproject(mx, my, z_at, center, h)
        sx, sy = _subsample(tx, ty)
        ax.scatter(sx, sy, s=2, c="crimson", alpha=0.06, linewidths=0)
        tag = "  <- current ORIENT" if mode == ORIENT else ""
        ax.set_title(f"orient = {mode}{tag}", fontsize=12)
    fig.suptitle(
        f"Orientation check (h = {h:g}) - set ORIENT to whichever lines up",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def build_slope_lookup(X, Y, Z, res):
    """Return slope_at(x, y) = |grad(elevation)| (dimensionless rise/run)."""
    from scipy.interpolate import griddata, RegularGridInterpolator

    gx = np.arange(X.min(), X.max() + res, res)
    gy = np.arange(Y.min(), Y.max() + res, res)
    GX, GY = np.meshgrid(gx, gy)                      # (ny, nx)
    GZ = griddata((X, Y), Z, (GX, GY), method="linear")
    nan = np.isnan(GZ)
    if nan.any():
        GZ[nan] = griddata((X, Y), Z, (GX[nan], GY[nan]), method="nearest")
    dzdy, dzdx = np.gradient(GZ, res, res)
    slope = np.hypot(dzdx, dzdy)
    interp = RegularGridInterpolator((gy, gx), slope,
                                     bounds_error=False, fill_value=None)

    def slope_at(x, y):
        xy = np.stack([np.clip(y, gy[0], gy[-1]), np.clip(x, gx[0], gx[-1])], -1)
        return interp(xy)

    return slope_at, (gx, gy, slope)


def register_to_edges(tx, ty, X, Y):
    """Constant per-axis scale+offset snapping the cloud edges to elevation edges.

    Returns (rx, ry, consts) where consts = dict(sx, ox, sy, oy) and
        rx = sx * tx + ox,   ry = sy * ty + oy.
    """
    plo, phi = REGISTER_PCTL
    dx0, dx1 = np.percentile(tx, [plo, phi])
    dy0, dy1 = np.percentile(ty, [plo, phi])
    sx = (X.max() - X.min()) / (dx1 - dx0)
    sy = (Y.max() - Y.min()) / (dy1 - dy0)
    ox = X.min() - sx * dx0
    oy = Y.min() - sy * dy0
    return sx * tx + ox, sy * ty + oy, dict(sx=sx, ox=ox, sy=sy, oy=oy)


def finalize_map(mx, my, meta, z_at, slope_at, X, Y, Z, center, h,
                 slope_grid, out_csv, out_path):
    """Deproject at h, register edges->edges, export the aligned map, and plot it."""
    tx, ty = deproject(mx, my, z_at, center, h)
    rx, ry, c = register_to_edges(tx, ty, X, Y)

    print("\nregistration (applied AFTER deprojection at h = %.4g):" % h)
    print(f"  x_registered = {c['sx']:.6f} * x_deproj + ({c['ox']:.6f})")
    print(f"  y_registered = {c['sy']:.6f} * y_deproj + ({c['oy']:.6f})")
    print(f"  elevation extent  x:[{X.min():.3f},{X.max():.3f}]  "
          f"y:[{Y.min():.3f},{Y.max():.3f}]")

    z = np.asarray(z_at(rx, ry), float)
    slope = np.asarray(slope_at(rx, ry), float)
    out = meta.copy()
    out["x_m"], out["y_m"], out["elev"], out["slope"] = rx, ry, z, slope
    out.to_csv(out_csv, index=False)
    print("wrote", out_csv, f"({len(out)} rows)")

    # figure: registered cloud over terrain (left) and over slope (right)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    gx, gy, sgrid = slope_grid

    ax = axes[0]
    _terrain(ax, X, Y, Z)
    px_s, py_s = _subsample(rx, ry)
    ax.scatter(px_s, py_s, s=2, c="crimson", alpha=0.06, linewidths=0)
    ax.add_patch(plt.Rectangle((X.min(), Y.min()), X.max() - X.min(),
                               Y.max() - Y.min(), fill=False, ec="k", lw=1.5))
    ax.set_title("registered centroids over elevation")

    ax = axes[1]
    im = ax.pcolormesh(gx, gy, sgrid, cmap="magma", shading="auto")
    ax.scatter(px_s, py_s, s=2, c="cyan", alpha=0.06, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlim(X.min() - 0.15, X.max() + 0.15)
    ax.set_ylim(Y.min() - 0.15, Y.max() + 0.15)
    fig.colorbar(im, ax=ax, label="slope |grad(elev)|", shrink=0.8)
    ax.set_title("registered centroids over slope")

    fig.suptitle(
        f"Final aligned map  |  dataset {DATASET}  |  h = {h:g}  |  orient={ORIENT}\n"
        f"x' = {c['sx']:.4f}x + {c['ox']:.4f}   "
        f"y' = {c['sy']:.4f}y + {c['oy']:.4f}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def main():
    os.makedirs(OUT, exist_ok=True)

    # Always undo the stated 180-deg storage rotation so elevation is in normal
    # coords; ORIENT then flips the *centroids* to match.
    X, Y, Z = load_elevation(ELEV_CSV, orient_rot180=True)
    z_at = build_elevation_lookup(X, Y, Z)
    slope_at, slope_grid = build_slope_lookup(X, Y, Z, SLOPE_RES_M)
    center = (0.5 * (X.min() + X.max()), 0.5 * (Y.min() + Y.max()))

    px, py, meta = load_centroids(CENTROID_CSV)
    n0 = len(px)
    keep = remove_corner_artifacts(px, py)
    px, py, meta = px[keep], py[keep], meta[keep].reset_index(drop=True)
    print(f"centroids: {n0} loaded, {n0 - keep.sum()} artifacts removed, "
          f"{len(px)} kept")

    to_m = calibrate(px, py, X, Y)
    mx, my = to_m(px, py)
    mx, my = orient(mx, my, X, Y, ORIENT)

    plot_h_sweep(mx, my, z_at, X, Y, Z, center, H_VALUES,
                 os.path.join(OUT, f"deproject_hsweep_{DATASET}.jpeg"))
    plot_orientation_compare(px, py, to_m, z_at, X, Y, Z, center,
                             h=H_VALUES[len(H_VALUES) // 2],
                             out_path=os.path.join(OUT, f"orientation_{DATASET}.jpeg"))

    finalize_map(mx, my, meta, z_at, slope_at, X, Y, Z, center, REGISTER_H,
                 slope_grid,
                 out_csv=os.path.join(OUT, f"aligned_map_{DATASET}.csv"),
                 out_path=os.path.join(OUT, f"aligned_map_{DATASET}.jpeg"))


if __name__ == "__main__":
    main()
