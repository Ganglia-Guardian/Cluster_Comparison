"""Collect the per-week mouse-centroid tracking files for a single mouse.

This is the centroid-only sibling of ``collect_mouse_recordings.py``. It walks the
same NAS cohort layout (mitopark weeks 8-24 plus the week-24 L-DOPA challenges, or
the wildtype ``{date}_{id}`` folders) with the same folder-matching strategy, but
the only file it cares about is the tracking output::

    mouse_centroid_{datetime}.csv        (h, m, s, ms, x, y per sample)

For each *week* it keeps only the leading ``--week-length`` hours of centroid data,
front-loaded across that week's recordings, and writes one cut-down CSV named for
the week::

    <out>/<mouse>/
        week8_centroid.csv
        week9_centroid.csv
        ...
        week24_centroid.csv
        LDOPA_week24_hightier_saline_centroid.csv
        LDOPA_week24_hightier_ldopa_centroid.csv
        manifest.csv

Each per-week file is in the exact raw format ``cluster_spatial_map.py`` expects
(no header; ``h,m,s,ms,x,y`` columns) so it can be fed straight to ``--centroid``.

As a final step (unless ``--no-stitch``) every per-week file is concatenated in
week order into a single ``all_weeks_centroid.csv`` -- again raw, with a
``stitch_manifest.csv`` recording the order, source and row/second counts so the
data file itself stays a clean drop-in.

Difference from ``collect_mouse_recordings.py``'s ``--partition-hours``: that budget
is a *total* split evenly across groups. Here ``--week-length`` is applied *per
week* -- every week independently keeps that many hours of centroid data.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Timestamp embedded in every recording filename, e.g. 2025-07-07T10_15_49.
_DT = r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})"
_DT_FMT = "%Y-%m-%dT%H_%M_%S"

# The one file we export, per recording. The tracker occasionally omits the
# timestamp (or stamps it differently); the loose fallback catches those.
_CENTROID = re.compile(rf"^mouse_centroid_{_DT}\.csv$", re.IGNORECASE)
_CENTROID_LOOSE = re.compile(r"^mouse_centroid.*\.csv$", re.IGNORECASE)

# A valid week folder is exactly "week<N>_<date>" (e.g. week8_062625). This
# excludes "week11_071425 - Copy", "week8-10-12-14-18Comp_video", etc.
_WEEK_DIR = re.compile(r"^week(\d+)_\d+$")

# Arena recording-folder matchers, by --arena choice (matched case-insensitively
# against the folder name). "arena_?h/l/m" catches "arenaH"/"arena_h"/"arenah";
# "restric.?ed" tolerates the "restriced" misspelling seen in the wildtype data.
_ARENA_MATCHERS = {
    "high": re.compile(r"arena_?h", re.IGNORECASE),        # mitopark high-tier
    "open": re.compile(r"open_?(arena|field)", re.IGNORECASE),
    "high_tier": re.compile(r"arena_?h", re.IGNORECASE),   # wildtype tiers
    "mid_tier": re.compile(r"arena_?m", re.IGNORECASE),
    "low_tier": re.compile(r"arena_?l", re.IGNORECASE),
    "restricted": re.compile(r"restric.?ed_?arena", re.IGNORECASE),
}

# Mitopark only: L-DOPA folders are ..._hightier_<cond> / ..._openarena_<cond>.
_LDOPA_TIER = {"high": "hightier", "open": "openarena"}

# Wildtype mouse folder: a "{date}_{id}" prefix, e.g. 011625_2 or 042025_1mp.
_MOUSE_DIR = re.compile(r"^\d{6}_\d+")

# Any path component containing one of these markers is not a real recording
# directory (clustering artefacts, dupes, multi-week comparison bundles). The
# centroid CSV is a tracking output that sits with the recording, so these are
# the same skips used for the raw recordings.
_EXCLUDE_MARKERS = (
    "combined_results",
    "clustering",
    "clutering",  # common misspelling seen in the data
    "cluster_output",
    "accprocesseddata",
    "videoanalysis",
    "results",
    "comp",
    " - copy",
    "all_arenas",
    "climbing",
    "video_clip",
    "partition",
    "partion",  # misspelling seen in the data ("Partion-evaluation")
    "evaluation",
    "viz",
)


@dataclass
class Centroid:
    datetime: str  # embedded timestamp, used to order recordings within a week
    path: Path


def _is_excluded(rel: Path) -> bool:
    """True if any directory component of *rel* is a non-recording marker."""
    for part in rel.parts:
        low = part.lower()
        if any(marker in low for marker in _EXCLUDE_MARKERS):
            return True
    return False


# ----------------------------------------------------------------------------------
# Discovery: which (set_name, search_root) pairs to collect from
# (identical strategy to collect_mouse_recordings.py so the two stay in lock-step)
# ----------------------------------------------------------------------------------
def _week_search_roots(week_dir: Path, arena: str) -> list[Path]:
    """The arena recording folder(s) of the chosen type inside a week directory."""
    matcher = _ARENA_MATCHERS[arena]
    return sorted(
        child
        for child in week_dir.iterdir()
        if child.is_dir() and matcher.search(child.name)
    )


def _ldopa_post_roots(ldopa_dir: Path) -> list[Path]:
    """The 'post' subfolder(s) inside an L-DOPA challenge directory."""
    return sorted(
        child
        for child in ldopa_dir.iterdir()
        if child.is_dir() and "post" in child.name.lower()
    )


def discover_sets(
    mouse_dir: Path, week_min: int, week_max: int, arena: str
) -> list[tuple[str, Path]]:
    """Return (set_name, search_root) pairs for a mitopark mouse, week-ascending."""
    sets: list[tuple[str, Path]] = []
    tier = _LDOPA_TIER.get(arena)

    week_children = []
    for child in mouse_dir.iterdir():
        if not child.is_dir():
            continue
        m = _WEEK_DIR.match(child.name)
        if not m:
            continue
        n = int(m.group(1))
        if week_min <= n <= week_max:
            week_children.append((n, child))

    for n, week_dir in sorted(week_children):
        roots = _week_search_roots(week_dir, arena)
        if not roots:
            print(f"  ! week{n}: no {arena}-arena folder under {week_dir.name}", file=sys.stderr)
            continue
        if len(roots) == 1:
            sets.append((f"week{n}", roots[0]))
        else:
            for root in roots:
                sets.append((f"week{n}__{root.name}", root))

    # Week-24 L-DOPA challenges (saline + ldopa) of the chosen arena, 'post' only.
    if tier:
        for condition in ("saline", "ldopa"):
            matches = sorted(mouse_dir.glob(f"L_DOPA_for_week24_*_{tier}_{condition}"))
            for ldopa_dir in matches:
                post_roots = _ldopa_post_roots(ldopa_dir)
                if not post_roots:
                    print(
                        f"  ! L-DOPA {condition}: no 'post' folder under {ldopa_dir.name}",
                        file=sys.stderr,
                    )
                    continue
                label = f"LDOPA_week24_{tier}_{condition}"
                if len(post_roots) == 1:
                    sets.append((label, post_roots[0]))
                else:
                    for root in post_roots:
                        sets.append((f"{label}__{root.name}", root))

    return sets


def discover_sets_wildtype(mouse_dirs: list[Path], arena: str) -> list[tuple[str, Path]]:
    """Wildtype layout: one (mouse_id, arena_folder) set per mouse."""
    matcher = _ARENA_MATCHERS[arena]
    sets: list[tuple[str, Path]] = []
    for mouse_dir in sorted(mouse_dirs):
        arena_folders = sorted(
            child
            for child in mouse_dir.iterdir()
            if child.is_dir()
            and matcher.search(child.name)
            and not _is_excluded(Path(child.name))
        )
        if not arena_folders:
            print(f"  ! {mouse_dir.name}: no {arena} arena folder", file=sys.stderr)
            continue
        if len(arena_folders) == 1:
            sets.append((mouse_dir.name, arena_folders[0]))
        else:
            for folder in arena_folders:
                sets.append((f"{mouse_dir.name}__{folder.name}", folder))
    return sets


# ----------------------------------------------------------------------------------
# Find the centroid CSV(s) under a search root, ordered by embedded timestamp
# ----------------------------------------------------------------------------------
def find_centroids(search_root: Path) -> list[Centroid]:
    """Every ``mouse_centroid_*.csv`` under *search_root* that is not inside a
    clustering/comparison folder, ordered by the timestamp in its name (files
    with no parseable timestamp sort last, by name)."""
    found: list[Centroid] = []
    for f in search_root.rglob("mouse_centroid*.csv"):
        if not f.is_file() or not _CENTROID_LOOSE.match(f.name):
            continue
        try:
            rel = f.parent.relative_to(search_root)
        except ValueError:
            rel = Path(f.parent.name)
        if _is_excluded(rel):
            continue
        m = _CENTROID.match(f.name) or re.search(_DT, f.name)
        dt = m.group(1) if m else ""
        found.append(Centroid(datetime=dt, path=f))
    # Timestamped files first (chronological); undated ones after, by name.
    found.sort(key=lambda c: (c.datetime == "", c.datetime, c.path.name))
    return found


# ----------------------------------------------------------------------------------
# Cut one centroid file to at most `remaining_s` seconds of leading data
# ----------------------------------------------------------------------------------
def _row_time(row: list[str]) -> float | None:
    """Seconds-of-day for a centroid row (h*3600 + m*60 + s + ms/1000), or None
    for a header/blank/short row -- matching cluster_spatial_map.load_centroid."""
    if not row or len(row) < 6:
        return None
    try:
        h, m, s, ms = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        float(row[4]); float(row[5])
    except (ValueError, TypeError):
        return None
    return h * 3600 + m * 60 + s + ms / 1000.0


def cut_centroid(path: Path, remaining_s: float) -> tuple[list[list[str]], float]:
    """Read *path* and return (kept_rows, seconds_kept): the leading data rows
    whose time-since-start is <= ``remaining_s``. Rows are returned verbatim (no
    reformatting, so x/y precision is preserved). ``seconds_kept`` is the span of
    the kept rows -- how much of the budget this recording consumed."""
    kept: list[list[str]] = []
    t0: float | None = None
    span = 0.0
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            t = _row_time(row)
            if t is None:
                continue  # skip header / malformed lines
            if t0 is None:
                t0 = t
            rel = t - t0
            if rel < 0:
                continue  # clock glitch / wrap; ignore out-of-order sample
            if rel > remaining_s:
                break  # budget spent for this recording
            kept.append(row)
            span = rel
    return kept, span


# ----------------------------------------------------------------------------------
# Collect: one cut-down centroid file per week
# ----------------------------------------------------------------------------------
def collect_centroids(
    sets: list[tuple[str, Path]],
    out_root: Path,
    week_length_h: float,
    dry_run: bool,
    overwrite: bool,
) -> list[dict[str, str]]:
    """For each set keep the leading ``week_length_h`` hours of centroid data,
    front-loaded across the set's recordings, and write one ``<set>_centroid.csv``.
    Returns a manifest (one row per set)."""
    manifest: list[dict[str, str]] = []
    if not sets:
        print("No matching arena folders / centroid files found.", file=sys.stderr)
        return manifest

    budget_s = week_length_h * 3600.0
    print(
        f"Week length: {week_length_h} h ({budget_s / 60:.0f} min) kept per set, "
        f"front-loaded across each set's recordings.{'  (dry run)' if dry_run else ''}\n"
    )

    for set_name, search_root in sets:
        out_file = out_root / f"{set_name}_centroid.csv"
        centroids = find_centroids(search_root)
        print(f"[{set_name}] {len(centroids)} centroid file(s) in {search_root.name}")
        if not centroids:
            print(f"  ! no mouse_centroid_*.csv under {search_root}", file=sys.stderr)
            manifest.append(_row(set_name, out_file, 0, 0, 0.0, "no centroid file"))
            continue

        if out_file.exists() and not overwrite and not dry_run:
            print(f"  - {out_file.name}: exists; skipping (use --overwrite to rebuild).")
            manifest.append(_row(set_name, out_file, 0, 0, 0.0, "exists; kept"))
            continue

        remaining = budget_s
        kept_rows: list[list[str]] = []
        used = 0
        for c in centroids:
            if remaining <= 0:
                print(f"  - {c.path.name}: skipped (week budget spent)")
                continue
            rows, span = cut_centroid(c.path, remaining)
            if not rows:
                print(f"  ! {c.path.name}: no usable rows", file=sys.stderr)
                continue
            kept_rows.extend(rows)
            remaining -= span
            used += 1
            print(f"  - {c.path.name}: {len(rows)} rows, {span / 60:.1f} min")

        kept_s = budget_s - remaining
        note = "" if used == len(centroids) else f"used {used}/{len(centroids)} recordings"
        if not kept_rows:
            print(f"  ! {set_name}: nothing kept", file=sys.stderr)
            manifest.append(_row(set_name, out_file, 0, 0, 0.0, note or "empty"))
            continue

        print(f"  => {out_file.name}: {len(kept_rows)} rows, {kept_s / 60:.1f} min")
        if not dry_run:
            out_root.mkdir(parents=True, exist_ok=True)
            with out_file.open("w", newline="") as fh:
                csv.writer(fh).writerows(kept_rows)
        manifest.append(_row(set_name, out_file, used, len(kept_rows), kept_s, note))

    verb = "Would write" if dry_run else "Wrote"
    n_files = sum(1 for m in manifest if int(m["rows"]) > 0)
    print(f"\n{verb} {n_files} per-week centroid file(s).")
    return manifest


def _row(set_name, out_file, used, rows, seconds, note) -> dict[str, str]:
    return {
        "set": set_name,
        "out_file": out_file.name,
        "recordings_used": str(used),
        "rows": str(rows),
        "seconds": f"{seconds:.1f}",
        "note": note,
    }


# ----------------------------------------------------------------------------------
# Stitch: concatenate every per-week file into one, in week order
# ----------------------------------------------------------------------------------
def stitch_weeks(
    sets: list[tuple[str, Path]], out_root: Path, dry_run: bool
) -> None:
    """Concatenate the per-week ``<set>_centroid.csv`` files (in the order the
    sets were discovered = week-ascending) into ``all_weeks_centroid.csv``. The
    data file stays raw; a ``stitch_manifest.csv`` records the stitch order and
    per-source row counts so provenance lives outside the data."""
    stitched = out_root / "all_weeks_centroid.csv"
    order: list[dict[str, str]] = []
    total = 0

    if not dry_run:
        out_fh = stitched.open("w", newline="")
        writer = csv.writer(out_fh)
    for i, (set_name, _root) in enumerate(sets, start=1):
        src = out_root / f"{set_name}_centroid.csv"
        if not src.exists():
            continue
        rows = 0
        if dry_run:
            with src.open(newline="") as fh:
                rows = sum(1 for _ in csv.reader(fh))
        else:
            with src.open(newline="") as fh:
                for row in csv.reader(fh):
                    writer.writerow(row)
                    rows += 1
        total += rows
        order.append({"order": str(len(order) + 1), "set": set_name,
                      "source_file": src.name, "rows": str(rows)})
        print(f"  {len(order):>2}. {src.name}: {rows} rows")
    if not dry_run:
        out_fh.close()

    if not order:
        print("  ! nothing to stitch (no per-week files present).", file=sys.stderr)
        return

    verb = "Would stitch" if dry_run else "Stitched"
    print(f"\n{verb} {len(order)} file(s) / {total} rows -> {stitched.name}")
    if not dry_run:
        man = out_root / "stitch_manifest.csv"
        with man.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["order", "set", "source_file", "rows"])
            w.writeheader()
            w.writerows(order)
        print(f"Wrote stitch manifest: {man}")


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a cut-down mouse_centroid_*.csv for each week of a "
        "mitopark or wildtype mouse (keeping --week-length hours per week), then "
        "stitch them into one centroid file in week order."
    )
    parser.add_argument(
        "--mouse",
        default=None,
        help="Mouse ID folder name, e.g. 042025_1mp (mitopark) or 011625_2 "
        "(wildtype). Required for mitopark; optional with --wildtype (omit to "
        "process every {date}_{id} mouse in the cohort).",
    )
    parser.add_argument(
        "--cohort-root",
        type=Path,
        default=Path(r"Y:\3darena_behavior\mitopark_042025"),
        help="Root of the cohort (default: Y:\\3darena_behavior\\mitopark_042025). "
        "For wildtype, point this at e.g. Y:\\3darena_behavior\\wildtype_062425.",
    )
    parser.add_argument(
        "--wildtype",
        action="store_true",
        help="Use the wildtype cohort layout: iterate {date}_{id} mouse folders and "
        "pull the --arena centroid from each.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./collected_centroids"),
        help="Parent output directory (a per-mouse/cohort subfolder is created inside).",
    )
    parser.add_argument("--week-min", type=int, default=8, help="Mitopark only.")
    parser.add_argument("--week-max", type=int, default=24, help="Mitopark only.")
    parser.add_argument(
        "--arena",
        choices=("high", "open", "high_tier", "mid_tier", "low_tier", "restricted"),
        default=None,
        help="Which arena's centroid to pull. Mitopark: 'high' (default) or 'open'. "
        "Wildtype: 'high_tier' (default), 'mid_tier', 'low_tier', 'open', 'restricted'.",
    )
    parser.add_argument(
        "--week-length",
        type=float,
        default=1.0,
        help="Hours of centroid data to keep from EACH week, front-loaded across "
        "that week's recordings (default: 1.0). Unlike collect_mouse_recordings' "
        "--partition-hours, this is per-week, not a total split across weeks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the plan without writing any files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild per-week files that already exist in the output.",
    )
    parser.add_argument(
        "--no-stitch",
        action="store_true",
        help="Skip the final all_weeks_centroid.csv stitch step.",
    )
    args = parser.parse_args()

    cohort_root = args.cohort_root.resolve()
    arena = args.arena or ("high_tier" if args.wildtype else "high")

    if args.wildtype:
        if args.mouse:
            mouse_dirs = [(cohort_root / args.mouse).resolve()]
            if not mouse_dirs[0].is_dir():
                raise FileNotFoundError(f"Mouse directory not found: {mouse_dirs[0]}")
            out_name, source_desc = args.mouse, mouse_dirs[0]
        else:
            mouse_dirs = [
                child
                for child in sorted(cohort_root.iterdir())
                if child.is_dir()
                and _MOUSE_DIR.match(child.name)
                and not _is_excluded(Path(child.name))
            ]
            out_name, source_desc = cohort_root.name, cohort_root
        sets = discover_sets_wildtype(mouse_dirs, arena)
    else:
        if not args.mouse:
            parser.error("--mouse is required unless --wildtype is set")
        mouse_dir = (cohort_root / args.mouse).resolve()
        if not mouse_dir.is_dir():
            raise FileNotFoundError(f"Mouse directory not found: {mouse_dir}")
        sets = discover_sets(mouse_dir, args.week_min, args.week_max, arena)
        out_name, source_desc = args.mouse, mouse_dir

    out_root = (args.out / out_name).resolve()
    print(f"Source: {source_desc}")
    print(f"Mode  : {'wildtype' if args.wildtype else 'mitopark'}  |  arena: {arena}")
    print(f"Output: {out_root}{'  (dry run)' if args.dry_run else ''}")
    print(
        "Note: each arena type reuses the same output file names; use a separate "
        "--out per arena to avoid mixing.",
        file=sys.stderr,
    )
    print()

    manifest = collect_centroids(
        sets=sets,
        out_root=out_root,
        week_length_h=args.week_length,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    if manifest and not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        manifest_path = out_root / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["set", "out_file", "recordings_used", "rows", "seconds", "note"],
            )
            writer.writeheader()
            writer.writerows(manifest)
        print(f"Wrote manifest: {manifest_path}")

    if not args.no_stitch:
        print(f"\nStitching per-week centroids in {out_root} (week order) ...")
        stitch_weeks(sets, out_root, args.dry_run)


if __name__ == "__main__":
    main()
