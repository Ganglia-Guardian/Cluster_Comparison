"""
Elevation-bin transition analysis.

For every mouse we build the aligned map (deproject at REGISTER_H, then snap edges
to the elevation extent -- exactly the pipeline in deproject_centroids.py), assign
each frame to its nearest elevation column ("elevation bin"), and record a
*transition* every time that nearest column changes.

IMPORTANT - week labels come from ROW POSITION, not from any column. The CSV is a
concatenation of per-week files (week8..week24) plus two LDOPA_week24 sets, in the
order/row-counts given by manifest.csv. Column 0 is an inner counter, NOT the week.
We assign each row its week by cumulative manifest row counts, and DISCARD the two
LDOPA sets (saline + ldopa).

IMPORTANT - ordering: the centroid CSV is ALREADY in chronological order (stitched
recordings). The (aux1, aux2, frame) counter is NOT globally monotonic -- several
recordings are concatenated per week -- so we must NOT sort by it (that interleaves
recordings and destroys the trajectory). We keep file order and start a new segment
at a week change or an odometer reset; transitions are only counted within a segment.

Boundary jitter: while the mouse sits on a column boundary, tracking noise flips the
nearest column back and forth (A-B-A-B). We debounce by requiring a column to be
occupied for at least MIN_DWELL consecutive samples before it counts as a real
occupancy; transitions are taken between consecutive stable occupancies.

Two per-transition metrics:
  slope = |z_to - z_from| / distance(column centers)   (units of |grad elev|, ~0..6)
  dz    = |z_to - z_from|                               (raw bin height difference, m)

Flat-corner transitions are discarded: both endpoints on the SAME flat plateau
(both ~0.41 m or both ~0.60 m) is flat-corner wandering and only inflates the 0 bin.

Outputs (elevation_analysis/output/), for each metric:
  <metric>_distribution.png  - per-mouse distribution over all weeks combined
  <metric>_weekly_trend.png  - weekly-mean per mouse, Spearman rho/p legend,
                               lc mice dashed, mp mice solid.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import deproject_centroids as dc

# --------------------------------------------------------------------------- #
ELEV_NAME   = "5-30-hi-complex-1-points.csv"   # elevation file expected under data-root
FPS         = 30             # recording rate (each week = 1800 s => ~54000 frames)
MIN_DWELL   = 3              # samples a column must be held to count (debounce jitter)
ZERO_DZ     = 0.005          # |dz| below this = a 0-slope (equal-height) transition
PLATEAUS    = (0.41, 0.60)   # flat-corner plateau heights (m) to discard
PLATEAU_TOL = 0.02           # a column is "on" a plateau if within this of its height

# group is inferred from the folder name: *lc = littermate control (dashed),
# *m(ito)?p = MitoPark (solid). Filled at runtime by discover_mice().
GROUP_PATTERNS = [("lc", re.compile(r"lc$", re.I)),
                  ("mp", re.compile(r"m(ito)?p$", re.I))]
COLORS = {}                  # label -> color, assigned in main() for the selected mice

METRICS = [
    ("slope", "transition slope  |dz|/dist"),
    ("dz",    "bin height difference  |dz|  (m)"),
]
# --------------------------------------------------------------------------- #


def group_of(name):
    """Infer group ('lc' or 'mp') from a folder/label name, else None."""
    for grp, pat in GROUP_PATTERNS:
        if pat.search(name):
            return grp
    return None


def discover_mice(data_root):
    """Find mouse subfolders under data_root that have the needed CSVs and whose
    name matches *lc or *m(ito)?p. Returns [(label, folder, group), ...] sorted."""
    mice = []
    for folder in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, folder)
        if not os.path.isdir(full):
            continue
        if not all(os.path.exists(os.path.join(full, f))
                   for f in ("all_weeks_centroid.csv", "manifest.csv")):
            continue
        label = folder.split("_")[-1]          # e.g. 042025_1mp -> 1mp
        grp = group_of(label)
        if grp:
            mice.append((label, folder, grp))
    return sorted(mice, key=lambda m: (m[2], m[0]))


def load_with_weeks(dataset, data_root):
    """Load centroids with the TRUE week per row (from manifest row positions).

    Drops the LDOPA saline/ldopa sets. Returns (px, py, meta) in file order, where
    meta has columns week, aux1, aux2, frame. Column 0 of the CSV is ignored.
    """
    path = os.path.join(data_root, dataset, "all_weeks_centroid.csv")
    df = pd.read_csv(path, header=None)
    mf = pd.read_csv(os.path.join(data_root, dataset, "manifest.csv"))

    sets = np.repeat(mf["set"].values, mf["rows"].values)   # file order == manifest order
    assert len(sets) == len(df), f"{dataset}: manifest rows != file rows"
    is_week = pd.Series(sets).str.startswith("week").to_numpy()   # False for LDOPA sets
    week = np.where(is_week,
                    pd.Series(sets).str.replace("week", "", regex=False).to_numpy(),
                    "-1").astype(int)

    px = df.iloc[:, 4].to_numpy(float)
    py = df.iloc[:, 5].to_numpy(float)
    ok = is_week & np.isfinite(px) & np.isfinite(py)
    meta = pd.DataFrame({
        "week": week[ok],
        "aux1": df.iloc[:, 1].to_numpy()[ok],
        "aux2": df.iloc[:, 2].to_numpy()[ok],
        "frame": df.iloc[:, 3].to_numpy()[ok],
    }).reset_index(drop=True)
    return px[ok], py[ok], meta


def aligned_xy(dataset, X, Y, z_at, center, data_root):
    """Deproject+register one mouse; return meta (FILE ORDER, unsorted) with x,y."""
    px, py, meta = load_with_weeks(dataset, data_root)
    keep = dc.remove_corner_artifacts(px, py)
    px, py, meta = px[keep], py[keep], meta[keep].reset_index(drop=True)

    to_m = dc.calibrate(px, py, X, Y)
    mx, my = to_m(px, py)
    mx, my = dc.orient(mx, my, X, Y, dc.ORIENT)
    tx, ty = dc.deproject(mx, my, z_at, center, dc.REGISTER_H)
    rx, ry, _ = dc.register_to_edges(tx, ty, X, Y)

    meta = meta.copy()
    meta["x"], meta["y"] = rx, ry
    return meta                       # NOTE: no sort -- file order is chronological


def transitions(df, tree, colxy, Zcol):
    """Debounced elevation-bin transitions -> DataFrame(week, slope, dz)."""
    b = tree.query(np.column_stack([df.x.values, df.y.values]))[1]  # nearest column
    week = df.week.values
    key = (df.aux1.values * 1_000_000 + df.aux2.values * 1000 + df.frame.values)
    # new segment at a week change or an odometer reset (concatenated recording)
    newseg = np.concatenate([[True], (week[1:] != week[:-1]) | (key[1:] <= key[:-1])])
    seg = np.cumsum(newseg)

    # run-length encode (bin constant within a segment)
    change = np.concatenate([[True], (b[1:] != b[:-1]) | (seg[1:] != seg[:-1])])
    starts = np.nonzero(change)[0]
    run_len = np.diff(np.append(starts, len(b)))
    run_bin, run_seg, run_wk = b[starts], seg[starts], week[starts]

    # keep only stable occupancies (>= MIN_DWELL samples), preserving order
    st = run_len >= MIN_DWELL
    sb, ss, sw = run_bin[st], run_seg[st], run_wk[st]

    # transitions between consecutive stable occupancies in the same segment
    i = np.nonzero((ss[1:] == ss[:-1]) & (sb[1:] != sb[:-1]))[0]
    b0, b1 = sb[i], sb[i + 1]
    z0, z1 = Zcol[b0], Zcol[b1]
    dz = np.abs(z1 - z0)
    dist = np.hypot(colxy[b1, 0] - colxy[b0, 0], colxy[b1, 1] - colxy[b0, 1])
    slope = dz / dist
    wk = sw[i + 1]

    # flag (don't drop): both endpoints on the SAME flat plateau (0.41 or 0.60)
    flat_corner = np.zeros(len(i), dtype=bool)
    for p in PLATEAUS:
        flat_corner |= (np.abs(z0 - p) < PLATEAU_TOL) & (np.abs(z1 - p) < PLATEAU_TOL)
    return pd.DataFrame({"week": wk, "slope": slope, "dz": dz, "flat_corner": flat_corner})


def plot_distributions(per_mouse, metric, label, out_path, xmax):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    bins = np.linspace(0, xmax, 60)

    for ax, (name, grp, tr) in zip(axes, per_mouse):
        ax.hist(tr[metric], bins=bins, density=True, color=COLORS[name], alpha=0.8)
        ax.set_title(f"{name} ({grp})   n={len(tr):,} transitions")
        ax.set_xlabel(label)
        ax.set_ylabel("density")

    ax = axes[5]
    for name, grp, tr in per_mouse:
        ls = "--" if grp == "lc" else "-"
        h, e = np.histogram(tr[metric], bins=bins, density=True)
        ax.plot(0.5 * (e[:-1] + e[1:]), h, ls, color=COLORS[name], label=f"{name} ({grp})")
    ax.set_title("all mice overlaid")
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.legend(fontsize=9)

    fig.suptitle(f"Elevation-bin {label} distribution  (all weeks combined; "
                 f"jitter-debounced, flat-corner transitions removed)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def plot_weekly_trend(per_mouse, metric, label, out_path):
    fig, ax = plt.subplots(figsize=(11, 7))
    for name, grp, tr in per_mouse:
        wk = tr.groupby("week")[metric].mean().sort_index()
        weeks, means = wk.index.values, wk.values
        rho, p = spearmanr(weeks, means) if len(weeks) > 2 else (np.nan, np.nan)
        ls = "--" if grp == "lc" else "-"
        ax.plot(weeks, means, ls, marker="o", color=COLORS[name],
                label=f"{name} ({grp})  rho={rho:+.2f}, p={p:.3f}")

    ax.set_xlabel("week")
    ax.set_ylabel("mean " + label)
    ax.set_title(f"Weekly-mean elevation-bin {label} per mouse\n"
                 "(lc dashed, mp solid; Spearman trend of weekly mean vs week)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def plot_weekly_count(per_mouse, frames_by, out_path, normalize, kind="transition"):
    """Weekly transition count (normalize=False) or transitions/minute (True)."""
    fig, ax = plt.subplots(figsize=(11, 7))
    for name, grp, tr in per_mouse:
        cnt = tr.groupby("week").size()
        weeks = np.sort(frames_by[name].index.values)   # all tracked weeks
        y = cnt.reindex(weeks, fill_value=0).values.astype(float)
        if normalize:
            minutes = frames_by[name].reindex(weeks).values / FPS / 60.0
            y = y / minutes
        rho, p = spearmanr(weeks, y) if len(weeks) > 2 else (np.nan, np.nan)
        ls = "--" if grp == "lc" else "-"
        ax.plot(weeks, y, ls, marker="o", color=COLORS[name],
                label=f"{name} ({grp})  rho={rho:+.2f}, p={p:.3f}")

    ylab = f"{kind} per minute" if normalize else f"number of {kind} transitions"
    ttl = "rate (per tracked minute)" if normalize else "count"
    ax.set_xlabel("week")
    ax.set_ylabel(ylab)
    ax.set_title(f"Weekly elevation-bin {kind} {ttl} per mouse\n"
                 "(lc dashed, mp solid; Spearman trend vs week)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Elevation-bin transition analysis. Discovers mice under "
                    "--data-root matching *lc (littermate control) or *m(ito)?p "
                    "(MitoPark) and writes weekly/distribution plots to --out-root.")
    p.add_argument("--data-root", default=dc.DATA,
                   help="folder with per-mouse subfolders and the elevation CSV "
                        f"(default: {dc.DATA})")
    p.add_argument("--out-root", default=dc.OUT,
                   help=f"output folder for figures (default: {dc.OUT})")
    p.add_argument("--elev", default=None,
                   help=f"elevation CSV path (default: <data-root>/{ELEV_NAME})")
    p.add_argument("--mice", nargs="+", metavar="NAME", default=None,
                   help="only these mice, by label (e.g. 1mp) or folder name; "
                        "default: all discovered")
    p.add_argument("--min-dwell", type=int, default=MIN_DWELL,
                   help=f"debounce dwell in samples (default: {MIN_DWELL})")
    p.add_argument("--open-field", action="store_true",
                   help="flat arena: assume every bin height = 0, so there is no "
                        "deprojection and no elevation/plateau/0-slope cleaning; "
                        "only the movement count/rate plots are produced. The --elev "
                        "CSV still supplies the bin (x,y) geometry; heights are ignored.")
    return p.parse_args()


def main():
    global MIN_DWELL
    args = parse_args()
    MIN_DWELL = args.min_dwell
    data_root, out = args.data_root, args.out_root
    elev = args.elev or os.path.join(data_root, ELEV_NAME)

    mice = discover_mice(data_root)
    if args.mice:
        sel = set(args.mice)
        mice = [m for m in mice if m[0] in sel or m[1] in sel]
    if not mice:
        raise SystemExit(f"no matching mice found under {data_root}")
    cmap = plt.get_cmap("tab10")
    COLORS.update({name: cmap(i % 10) for i, (name, _, _) in enumerate(mice)})
    print("mice:", ", ".join(f"{n}({g})" for n, _, g in mice))

    os.makedirs(out, exist_ok=True)
    if args.open_field:
        arr = np.loadtxt(elev, delimiter=",")     # only (x,y) used; heights forced to 0
        X = arr[:, 0].max() + arr[:, 0].min() - arr[:, 0]
        Y = arr[:, 1].max() + arr[:, 1].min() - arr[:, 1]
        Z = np.zeros(len(X))
        print("open-field mode: all bin heights = 0 (movement-only, no cleaning)")
    else:
        X, Y, Z = dc.load_elevation(elev, orient_rot180=True)
    z_at = dc.build_elevation_lookup(X, Y, Z)
    center = (0.5 * (X.min() + X.max()), 0.5 * (Y.min() + Y.max()))
    colxy = np.column_stack([X, Y])
    tree = cKDTree(colxy)

    per_mouse, per_mouse_zero, frames_by = [], [], {}
    for name, dataset, grp in mice:
        df = aligned_xy(dataset, X, Y, z_at, center, data_root)
        tr = transitions(df, tree, colxy, Z)
        frames_by[name] = df.groupby("week").size()
        nonflat = tr[~tr.flat_corner]                    # slope/dz/count plots (as before)
        zero = tr[tr.dz < ZERO_DZ]                        # ALL 0-slope, incl. plateaus
        print(f"{name}: weeks {df.week.min()}-{df.week.max()} "
              f"({df.week.nunique()} wk), {len(df):,} tracked frames -> "
              f"{len(tr):,} transitions ({len(zero):,} zero-slope incl. "
              f"{int(tr.flat_corner.sum()):,} on plateaus)")
        per_mouse.append((name, grp, nonflat))
        per_mouse_zero.append((name, grp, zero))

    # movement plots: meaningful in both modes
    plot_weekly_count(per_mouse, frames_by,
                      os.path.join(out, "transition_count_weekly.png"), normalize=False)
    plot_weekly_count(per_mouse, frames_by,
                      os.path.join(out, "transition_rate_weekly.png"), normalize=True)

    if args.open_field:
        return   # everything below is elevation-based; degenerate on a flat arena

    plot_weekly_count(per_mouse_zero, frames_by,
                      os.path.join(out, "zero_slope_count_weekly.png"),
                      normalize=False, kind="0-slope (equal-height, incl. plateaus)")
    plot_weekly_count(per_mouse_zero, frames_by,
                      os.path.join(out, "zero_slope_rate_weekly.png"),
                      normalize=True, kind="0-slope (equal-height, incl. plateaus)")

    for metric, label in METRICS:
        xmax = np.percentile(np.concatenate([t[metric].values for _, _, t in per_mouse]), 99)
        plot_distributions(per_mouse, metric, label,
                           os.path.join(out, f"{metric}_distribution.png"), xmax)
        plot_weekly_trend(per_mouse, metric, label,
                          os.path.join(out, f"{metric}_weekly_trend.png"))


if __name__ == "__main__":
    main()
