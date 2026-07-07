#!/usr/bin/env python3
"""
weekly_spatial_analysis.py
==========================
Drive ``cluster_spatial_map.py`` across every week of one mouse, animate the
per-week density maps, and produce position-occupancy summary plots.

For a mouse folder holding one ``Cluster_detail_results.csv`` and the per-week
``week{N}_centroid.csv`` files (as produced by ``collect_centroids.py``) this:

  1. Per week, subsets the detail CSV to that week's rows (by Folder_Name),
     NaN-cleans the week's centroid file, and runs ``cluster_spatial_map.py``
     into its own output folder:
         output/<mouse>/density_maps/<week>/cluster_density_map.png  (+ siblings)
  2. Stitches every week's density map, in chronological order, into an animated
     GIF at --fps frames/sec with the week label drawn in the top-right corner
     (kept alongside the summaries, not the per-week maps):
         output/<mouse>/summary_plots/weekly_density.gif
  3. Writes position-occupancy summaries from the (cleaned) centroid tracks:
         output/<mouse>/summary_plots/radius_distribution.png
         output/<mouse>/summary_plots/radial_entropy_by_week.png
         output/<mouse>/summary_plots/coordinate_entropy_by_week.png
         output/<mouse>/summary_plots/bins_visited_by_week.png
         output/<mouse>/summary_plots/entropy_by_week.csv

Radius is measured to a fixed arena center (default 700, 600; --center to change).

With --no-clusters the detail CSV is ignored entirely: step 1 skips the folder
subset and runs cluster_spatial_map.py in --centroid-only mode, so each density map
is drawn over the full centroid occupancy (every tracked sample colored by local
density) instead of the cluster events. The GIF and the summaries are unchanged --
they were always centroid-based -- so centroid results don't wait on clustering.

ENTROPY NOTES
  Shannon entropy is reported in bits over FIXED bins shared by every week
  (so weeks are comparable). Two supports are used:
    * radial     : |pos - center| binned into --radial-bins equal-width bins
    * coordinate : (x, y) binned into a --coord-bins x --coord-bins grid
  These live on different supports, so their bit values are not directly
  comparable to each other. Re-binning shifts entropy by a constant offset
  (~log2 of the bin-count ratio); the CSV also carries a normalized efficiency
  H / log2(n_bins) in [0, 1] that divides that offset out, so "the same shape at
  a different resolution" lands on the same normalized number.

USAGE
    uv run centroid_analysis/weekly_spatial_analysis.py --mouse 042025_1mp
    uv run centroid_analysis/weekly_spatial_analysis.py --mouse 042025_1mp \
        --center 700 600 --fps 2 --force-maps
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_SPATIAL_MAP = _HERE / "cluster_spatial_map.py"

# Fixed arena extent (px), matching cluster_spatial_map.py's ARENA_W/ARENA_H, so the
# radius range is the same for every mouse/week (max radius = center-to-farthest-corner).
ARENA_W, ARENA_H = 1400, 1200
_FONT_PATH = (
    Path(matplotlib.__file__).resolve().parent
    / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
)

# The density map file cluster_spatial_map.py writes on a data-derived background.
_DENSITY_MAP = "cluster_density_map.png"


# ----------------------------------------------------------------------------------
# Match each week{N}_centroid.csv (and the LDOPA files) to a Folder_Name value that
# actually occurs in this mouse's detail CSV. Folder naming is inconsistent between
# mice (e.g. 'week_24_ldop' vs 'w24_ldopa'), so we match by shape, not by literal.
# ----------------------------------------------------------------------------------
def detail_folder_index(detail_csv: Path) -> tuple[dict[int, str], str | None, str | None]:
    """Return (numeric_weeks, saline_folder, ldopa_folder) from the detail CSV's
    Folder_Name column. numeric_weeks maps week number -> the exact folder string."""
    with open(detail_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fmap = {n.strip().lower(): n for n in (reader.fieldnames or [])}
        key = fmap.get("folder_name")
        if key is None:
            raise SystemExit(f"ERROR: {detail_csv} has no Folder_Name column.")
        folders = {(row.get(key) or "").strip() for row in reader}

    numeric: dict[int, str] = {}
    saline = ldopa = None
    for f in folders:
        low = f.lower()
        if not low:
            continue
        if "salin" in low:
            saline = f
        elif "ldop" in low:
            ldopa = f
        else:
            m = re.match(r"w(?:eek)?_?(\d+)$", low)
            if m:
                numeric[int(m.group(1))] = f
    return numeric, saline, ldopa


def discover_weeks(mouse_dir: Path, detail_csv: Path | None) -> list[dict]:
    """Return an ordered list of week specs to process, each a dict with keys
    label, short, order, folder_subset, centroid (Path).

    With a detail CSV, only weeks that have BOTH a centroid file and a matching
    Folder_Name are kept, and folder_subset is that Folder_Name. When detail_csv is
    None (centroid-only mode), every week/LDOPA centroid file is taken as-is with
    folder_subset=None -- no cluster labeling is involved."""
    if detail_csv is not None:
        numeric, saline, ldopa = detail_folder_index(detail_csv)
    else:
        numeric, saline, ldopa = {}, None, None
    specs: list[dict] = []

    for cpath in sorted(mouse_dir.glob("week*_centroid.csv")):
        m = re.match(r"week(\d+)_centroid\.csv$", cpath.name.lower())
        if not m:
            continue
        n = int(m.group(1))
        if detail_csv is not None:
            folder = numeric.get(n)
            if folder is None:
                print(f"  ! week{n}: no matching Folder_Name in detail CSV; skipped",
                      file=sys.stderr)
                continue
        else:
            folder = None
        specs.append(dict(label=f"week{n}", short=str(n), order=float(n),
                          folder_subset=folder, centroid=cpath))

    # LDOPA week-24 challenges: matched by 'saline'/'ldopa' in the centroid name.
    for kind, folder, ordv in (("saline", saline, 900.0), ("ldopa", ldopa, 901.0)):
        hits = sorted(mouse_dir.glob(f"*{kind}*centroid.csv"))
        if not hits:
            continue
        if detail_csv is not None and folder is None:
            print(f"  ! {kind}: centroid present but no matching Folder_Name; skipped",
                  file=sys.stderr)
            continue
        specs.append(dict(label=f"week24_{kind}", short=f"24{kind[0]}",
                          order=ordv, folder_subset=folder, centroid=hits[0]))

    specs.sort(key=lambda s: s["order"])
    return specs


# ----------------------------------------------------------------------------------
# Clean a centroid file: keep only rows with finite x, y (cluster_spatial_map.py
# does not tolerate NaN positions -- W = cx.max() would poison the histogram).
# Rows are re-emitted verbatim so x/y precision is preserved.
# ----------------------------------------------------------------------------------
def _row_xy(row: list[str]) -> tuple[float, float] | None:
    if not row or len(row) < 6:
        return None
    try:
        int(row[0]); int(row[1]); int(row[2]); int(row[3])
        x, y = float(row[4]), float(row[5])
    except (ValueError, TypeError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def clean_centroid(src: Path, dst: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Write NaN-free rows of *src* to *dst*; return (x, y, n_dropped)."""
    kept_rows: list[list[str]] = []
    xs: list[float] = []
    ys: list[float] = []
    dropped = 0
    with open(src, newline="") as fh:
        for row in csv.reader(fh):
            xy = _row_xy(row)
            if xy is None:
                dropped += 1
                continue
            kept_rows.append(row)
            xs.append(xy[0]); ys.append(xy[1])
    with open(dst, "w", newline="") as fh:
        csv.writer(fh).writerows(kept_rows)
    return np.array(xs), np.array(ys), dropped


# ----------------------------------------------------------------------------------
# Run cluster_spatial_map.py for one week (as a subprocess: it is the home-base
# tool and stays the single source of truth for the density maps).
# ----------------------------------------------------------------------------------
def run_spatial_map(detail_csv: Path | None, folder_subset: str | None,
                    centroid: Path, outdir: Path, centroid_only: bool = False) -> bool:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_SPATIAL_MAP),
        "--centroid", str(centroid),
        "--outdir", str(outdir),
    ]
    if centroid_only:
        cmd.append("--centroid-only")
    else:
        cmd += ["--detail-csv", str(detail_csv), "--folder_subset", folder_subset]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    ! cluster_spatial_map failed (rc={res.returncode}):", file=sys.stderr)
        print("    " + (res.stderr.strip().replace("\n", "\n    ") or "<no stderr>"),
              file=sys.stderr)
        return False
    # surface the match-gap line so alignment quality is visible
    for line in res.stdout.splitlines():
        if line.startswith("match gap"):
            print(f"    {line.strip()}")
    return (outdir / _DENSITY_MAP).exists()


# ----------------------------------------------------------------------------------
# Animated GIF: pad every density map to a common canvas, label it top-right, and
# write the frames at the requested rate.
# ----------------------------------------------------------------------------------
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(_FONT_PATH), size)
    except Exception:
        return ImageFont.load_default()


def build_gif(frames: list[tuple[str, Path]], out_gif: Path, fps: float) -> None:
    """frames: (label, png_path) in play order. Pads to a shared size, draws the
    label top-right, saves an animated GIF at *fps* frames/sec."""
    if not frames:
        print("  ! no density maps to animate.", file=sys.stderr)
        return
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imgs = [(label, Image.open(p).convert("RGB")) for label, p in frames]
    canvas_w = max(im.width for _, im in imgs)
    canvas_h = max(im.height for _, im in imgs)

    out_frames = []
    for label, im in imgs:
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(im, ((canvas_w - im.width) // 2, (canvas_h - im.height) // 2))
        draw = ImageDraw.Draw(canvas)
        fsize = max(20, canvas_w // 28)
        font = _load_font(fsize)
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        pad = int(fsize * 0.35)
        x1 = canvas_w - tw - 3 * pad
        y1 = pad
        draw.rectangle([x1 - pad, y1 - pad, canvas_w - pad, y1 + th + pad],
                       fill=(0, 0, 0))
        draw.text((x1, y1 - box[1]), label, font=font, fill=(255, 255, 255))
        out_frames.append(canvas)

    duration_ms = int(round(1000.0 / fps))
    out_frames[0].save(
        out_gif, save_all=True, append_images=out_frames[1:],
        duration=duration_ms, loop=0, disposal=2,
    )
    print(f"  GIF: {len(out_frames)} frames @ {fps} fps -> {out_gif}")


# ----------------------------------------------------------------------------------
# Position-occupancy summaries
# ----------------------------------------------------------------------------------
def shannon_bits(counts: np.ndarray) -> float:
    """Shannon entropy (bits) of a count histogram."""
    c = counts[counts > 0].astype(float)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def radius_ridgeline(have: list[dict], center: tuple[float, float], out_path: Path,
                     mouse: str) -> tuple[float, float] | None:
    """Per-week ridgeline of radial distance to center, in the style of
    degeneracy_analysis/tba_over_weeks.py: one height-normalized KDE per week,
    earliest week at the top, viridis-colored by time, a black tick at the weekly
    median, and the weekly-median trend (Spearman rho) in the title. The x-range is
    fixed to the arena's max possible radius so every mouse compares directly."""
    cx, cy = center
    # max radius = center to the farthest corner of the fixed [0,W]x[0,H] arena
    corners = [(0, 0), (ARENA_W, 0), (0, ARENA_H), (ARENA_W, ARENA_H)]
    rmax = max(np.hypot(px - cx, py - cy) for px, py in corners)
    xlim = (0.0, rmax)
    grid = np.linspace(*xlim, 200)
    cmap = plt.get_cmap("viridis")
    n = len(have)

    fig, ax = plt.subplots(figsize=(9, 0.5 * n + 2))
    meds = []
    for i, w in enumerate(have):
        r = np.hypot(w["x"] - cx, w["y"] - cy)
        base = (n - 1 - i) * 1.0                     # earliest week at top
        color = cmap(i / max(1, n - 1))
        if len(r) >= 3 and np.std(r) > 0:
            dens = gaussian_kde(r)(grid)
            dens = dens / dens.max() * 0.9
            ax.fill_between(grid, base, base + dens, color=color, alpha=0.8,
                            lw=0.5, edgecolor="white")
        meds.append(float(np.median(r)))
        ax.plot([np.median(r)], [base], "|", color="black", ms=8, mew=1.2)
        ax.text(grid[0], base + 0.05, w["short"], fontsize=7, va="bottom")

    rho, p = spearmanr(np.arange(n), meds)           # weeks are already time-ordered
    grp = "control" if mouse.endswith("lc") else "MitoPark"
    ax.set_yticks([])
    ax.set_xlim(*xlim)
    ax.set_xlabel(f"radial distance to center ({cx:.0f}, {cy:.0f})  [px]")
    ax.set_title(f"{mouse} ({grp}): radial-distance distribution by week\n"
                 f"weekly-median trend rho={rho:+.2f}, p={p:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return float(rho), float(p)


def make_summaries(weeks: list[dict], center: tuple[float, float], outdir: Path,
                   radial_bins: int, coord_bins: int, mouse: str) -> None:
    """weeks: specs augmented with 'x','y' arrays. Writes the radius-distribution
    figure, the two entropy bar charts, the bins-visited bar chart, and the CSV."""
    cx, cy = center
    have = [w for w in weeks if w.get("x") is not None and len(w["x"])]
    if not have:
        print("  ! no centroid data for summaries.", file=sys.stderr)
        return
    outdir.mkdir(parents=True, exist_ok=True)

    # shared bin edges so every week is measured identically
    all_x = np.concatenate([w["x"] for w in have])
    all_y = np.concatenate([w["y"] for w in have])
    all_r = np.hypot(all_x - cx, all_y - cy)
    r_edges = np.linspace(0.0, all_r.max(), radial_bins + 1)
    x_edges = np.linspace(all_x.min(), all_x.max(), coord_bins + 1)
    y_edges = np.linspace(all_y.min(), all_y.max(), coord_bins + 1)

    # ---- 1) radius-to-center distribution: per-week ridgeline (earliest at top) ----
    radius_ridgeline(have, center, outdir / "radius_distribution.png", mouse)

    # ---- 2 & 3) per-week entropy on the shared bins ----
    rows = []
    for w in have:
        r = np.hypot(w["x"] - cx, w["y"] - cy)
        r_hist, _ = np.histogram(r, bins=r_edges)
        c_hist, _, _ = np.histogram2d(w["x"], w["y"], bins=[x_edges, y_edges])
        h_r = shannon_bits(r_hist)
        h_c = shannon_bits(c_hist.ravel())
        visited = int((c_hist > 0).sum())          # occupied cells of the shared grid
        rows.append(dict(
            label=w["label"], short=w["short"], n_points=int(len(w["x"])),
            radial_entropy_bits=h_r, coord_entropy_bits=h_c,
            radial_entropy_norm=h_r / np.log2(radial_bins),
            coord_entropy_norm=h_c / np.log2(coord_bins * coord_bins),
            bins_visited=visited,
            bins_visited_frac=visited / (coord_bins * coord_bins),
        ))

    shorts = [r["short"] for r in rows]

    def _bar(values, title, ylabel, fname, color):
        # weekly trend of entropy across the time-ordered week sequence
        rho, p = spearmanr(np.arange(len(rows)), values)
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(rows) + 3), 5))
        ax.bar(range(len(rows)), values, color=color, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(shorts)
        ax.set_xlabel("week")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nweekly trend rho={rho:+.2f}, p={p:.3f}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)

    _bar([r["radial_entropy_bits"] for r in rows],
         f"Radial-distance occupancy entropy ({radial_bins} bins)",
         "entropy [bits]", "radial_entropy_by_week.png", "#4C72B0")
    _bar([r["coord_entropy_bits"] for r in rows],
         f"Coordinate occupancy entropy ({coord_bins}x{coord_bins} grid)",
         "entropy [bits]", "coordinate_entropy_by_week.png", "#55A868")
    _bar([r["bins_visited"] for r in rows],
         f"Distinct grid cells visited ({coord_bins}x{coord_bins} grid, "
         f"{coord_bins * coord_bins} total)",
         "bins visited", "bins_visited_by_week.png", "#C44E52")

    with open(outdir / "entropy_by_week.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                             for k, v in r.items()})
    print(f"  summaries -> {outdir}")


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mouse", required=True, help="Mouse id, e.g. 042025_1mp")
    p.add_argument("--data-root", type=Path, default=_HERE / "data",
                   help="Folder holding <mouse>/ (default: centroid_analysis/data)")
    p.add_argument("--out-root", type=Path, default=_HERE / "output",
                   help="Output root (default: centroid_analysis/output)")
    p.add_argument("--detail-csv", type=Path, default=None,
                   help="Override the detail CSV path "
                        "(default: <data-root>/<mouse>/Cluster_detail_results.csv)")
    p.add_argument("--center", type=float, nargs=2, default=[700.0, 600.0],
                   metavar=("X", "Y"), help="Arena center for radius (default 700 600)")
    p.add_argument("--fps", type=float, default=2.0, help="GIF frames/sec (default 2)")
    p.add_argument("--radial-bins", type=int, default=40, help="radial entropy bins")
    p.add_argument("--coord-bins", type=int, default=40,
                   help="coordinate entropy grid is N x N")
    p.add_argument("--force-maps", action="store_true",
                   help="Re-run cluster_spatial_map even if a density map exists")
    p.add_argument("--skip-maps", action="store_true",
                   help="Skip map/GIF generation; rebuild summaries from existing "
                        "cleaned centroids")
    p.add_argument("--no-clusters", action="store_true",
                   help="Centroid-only: skip all cluster labeling and ignore "
                        "Cluster_detail_results.csv. Density maps are drawn from the "
                        "centroid occupancy alone, so results don't wait on clustering. "
                        "Summaries are identical (they were always centroid-based).")
    args = p.parse_args()

    mouse_dir = (args.data_root / args.mouse).resolve()
    if not mouse_dir.is_dir():
        raise SystemExit(f"ERROR: mouse folder not found: {mouse_dir}")
    if args.no_clusters:
        detail_csv = None
    else:
        detail_csv = args.detail_csv or (mouse_dir / "Cluster_detail_results.csv")
        if not detail_csv.exists():
            raise SystemExit(f"ERROR: detail CSV not found: {detail_csv}")

    out_mouse = (args.out_root / args.mouse).resolve()
    maps_dir = out_mouse / "density_maps"
    summ_dir = out_mouse / "summary_plots"

    print(f"Mouse : {args.mouse}")
    print(f"Detail: {detail_csv if detail_csv else '(none: --no-clusters, centroid-only)'}")
    print(f"Output: {out_mouse}")
    print(f"Center: ({args.center[0]:.0f}, {args.center[1]:.0f})\n")

    weeks = discover_weeks(mouse_dir, detail_csv)
    if not weeks:
        raise SystemExit("ERROR: no week centroid files found"
                         + ("." if args.no_clusters else
                            " with a matching Folder_Name in the detail CSV."))
    print(f"Weeks : {', '.join(w['label'] for w in weeks)}\n")

    frames: list[tuple[str, Path]] = []
    for w in weeks:
        week_out = maps_dir / w["label"]
        clean_path = week_out / f"{w['label']}_centroid_clean.csv"
        density_png = week_out / _DENSITY_MAP

        # NaN-clean once (also feeds the summaries); reuse if present under --skip-maps
        if clean_path.exists() and (args.skip_maps or not args.force_maps):
            x, y, _ = clean_centroid(w["centroid"], clean_path)  # cheap; keeps arrays
        else:
            week_out.mkdir(parents=True, exist_ok=True)
            x, y, dropped = clean_centroid(w["centroid"], clean_path)
            if dropped:
                print(f"[{w['label']}] cleaned {dropped} non-finite row(s)")
        w["x"], w["y"] = x, y

        if not args.skip_maps:
            if density_png.exists() and not args.force_maps:
                print(f"[{w['label']}] density map exists; reuse "
                      f"(use --force-maps to rebuild)")
            else:
                src = ("centroid-only" if args.no_clusters
                       else f"folder='{w['folder_subset']}'")
                print(f"[{w['label']}] running cluster_spatial_map "
                      f"({src}, {len(x)} centroid pts)")
                ok = run_spatial_map(detail_csv, w["folder_subset"], clean_path,
                                     week_out, centroid_only=args.no_clusters)
                if not ok:
                    print(f"  ! {w['label']}: no density map produced; skipping frame",
                          file=sys.stderr)
        if density_png.exists():
            frames.append((w["label"], density_png))

    if not args.skip_maps:
        print("\nBuilding animation ...")
        build_gif(frames, summ_dir / "weekly_density.gif", args.fps)

    print("\nBuilding summaries ...")
    make_summaries(weeks, (args.center[0], args.center[1]), summ_dir,
                   args.radial_bins, args.coord_bins, args.mouse)

    print("\nDone.")


if __name__ == "__main__":
    main()
