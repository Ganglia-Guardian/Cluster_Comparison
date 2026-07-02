#!/usr/bin/env python3
"""Run the arena_analysis pipeline end to end, in dependency order.

Each stage is a standalone script in this folder. Because the scripts import
each other by bare name and use relative data/output paths, every stage is run
as its own subprocess with the repo root as the working directory and both this
folder and the repo root on PYTHONPATH.

Dependency order:
    build_frame_table            -> frame_table.csv
    cluster_arena_exclusivity    -> output/exclusivity/cluster_verdicts.csv
    build_feature_table          -> frame_features.csv
    arena_occupancy_drift        (reads exclusivity + mat data)
    arena_transitions            -> output/transitions/{fanout,diversity}_by_arena.csv
    arena_tba_vulnerability      (reads frame_features + verdicts)
    arena_transition_tba         (reads frame_features + mat data)
    arena_transitions_by_clustertype -> output/transitions_by_clustertype/
    plot_feature_contraction_lines   (reads frame_features)
    plot_tba_combined_lines          (reads frame_features)
    plot_stitched_weeks              (reads transitions / by_clustertype CSVs)

Examples:
    uv run python arena_analysis/arena_analysis_main.py                # full run
    uv run python arena_analysis/arena_analysis_main.py --list         # list stages
    uv run python arena_analysis/arena_analysis_main.py --only arena_transitions --depth 30 --reps 200
    uv run python arena_analysis/arena_analysis_main.py --from arena_tba_vulnerability
    uv run python arena_analysis/arena_analysis_main.py --skip plot_stitched_weeks --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
REPO_ROOT = FOLDER.parent


class Stage:
    def __init__(self, name, script, accepts=(), needs=()):
        self.name = name
        self.script = script
        self.accepts = set(accepts)   # forwardable-flag dests this script's argparse takes
        self.needs = set(needs)       # dests that must be set for this stage to run


# Ordered pipeline. `accepts` uses argparse dest names (see FORWARD below).
STAGES = [
    Stage("build_frame_table", "build_frame_table.py"),
    Stage("cluster_arena_exclusivity", "cluster_arena_exclusivity.py",
          accepts=["excl", "shared"]),
    Stage("build_feature_table", "build_feature_table.py"),
    Stage("arena_occupancy_drift", "arena_occupancy_drift.py"),
    Stage("arena_transitions", "arena_transitions.py",
          accepts=["depth", "reps"]),
    Stage("arena_tba_vulnerability", "arena_tba_vulnerability.py"),
    Stage("arena_transition_tba", "arena_transition_tba.py"),
    Stage("arena_transitions_by_clustertype", "arena_transitions_by_clustertype.py",
          accepts=["membership", "split", "arena_mode", "link", "depth", "reps"]),
    Stage("plot_feature_contraction_lines", "plot_feature_contraction_lines.py",
          accepts=["feature", "n", "min_frames", "min_weeks", "complete", "use_abs"]),
    Stage("plot_tba_combined_lines", "plot_tba_combined_lines.py",
          accepts=["feature", "n", "min_frames", "min_weeks"]),
    Stage("plot_stitched_weeks", "plot_stitched_weeks.py",
          accepts=["view", "mode", "exclude"]),
]

# Forwardable flags: (cli, dest, kind). kind in {"value", "flag", "list"}.
# A flag is passed to a stage only when the user set it AND dest is in stage.accepts,
# so each script keeps its own defaults for anything left unspecified.
FORWARD = [
    ("--excl", "excl", "value"),
    ("--shared", "shared", "value"),
    ("--depth", "depth", "value"),
    ("--reps", "reps", "value"),
    ("--membership", "membership", "value"),
    ("--split", "split", "value"),
    ("--arena-mode", "arena_mode", "value"),
    ("--link", "link", "value"),
    ("--feature", "feature", "value"),
    ("--n", "n", "value"),
    ("--min-frames", "min_frames", "value"),
    ("--min-weeks", "min_weeks", "value"),
    ("--complete", "complete", "flag"),
    ("--abs", "use_abs", "flag"),
    ("--view", "view", "value"),
    ("--mode", "mode", "value"),
    ("--exclude", "exclude", "list"),
]


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_argument_group("pipeline control")
    g.add_argument("--list", action="store_true",
                   help="list the stages (with the flags each accepts) and exit")
    g.add_argument("--only", nargs="+", metavar="STAGE",
                   help="run only these stage(s)")
    g.add_argument("--skip", nargs="+", metavar="STAGE", default=[],
                   help="skip these stage(s)")
    g.add_argument("--from", dest="from_stage", metavar="STAGE",
                   help="start at this stage (inclusive)")
    g.add_argument("--to", dest="to_stage", metavar="STAGE",
                   help="stop after this stage (inclusive)")
    g.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    g.add_argument("--continue-on-error", action="store_true",
                   help="keep going if a stage fails (default: stop)")
    g.add_argument("--python", default=sys.executable,
                   help="interpreter used to run each stage (default: this one)")

    # cluster_arena_exclusivity
    e = ap.add_argument_group("cluster_arena_exclusivity")
    e.add_argument("--excl", type=float,
                   help="occ3d >= this -> 3D-exclusive (default 0.80)")
    e.add_argument("--shared", type=float,
                   help="half-width of the shared band around 0.5 (default 0.15)")

    # arena_transitions / arena_transitions_by_clustertype (rarefaction)
    t = ap.add_argument_group("transition rarefaction")
    t.add_argument("--depth", type=int,
                   help="rarefaction depth, out-transitions per cluster (default 20)")
    t.add_argument("--reps", type=int, help="rarefaction reps (default 100)")

    # arena_transitions_by_clustertype
    c = ap.add_argument_group("arena_transitions_by_clustertype")
    c.add_argument("--membership", choices=["split", "all"],
                   help="'split' groups by occ3d; 'all' forces home-arena mode (default split)")
    c.add_argument("--split", type=float,
                   help="occ3d < split -> 2D, >= split -> 3D (default 0.5)")
    c.add_argument("--arena-mode", dest="arena_mode", choices=["home", "all"],
                   help="home (default) or all")
    c.add_argument("--link", choices=["bridge", "adjacent"],
                   help="transition link type (default bridge)")

    # plot_feature_contraction_lines / plot_tba_combined_lines
    p = ap.add_argument_group("feature line plots")
    p.add_argument("--feature",
                   help="feature to plot ('all' + names for contraction; a single "
                        "name for tba_combined, default TotAccelBA)")
    p.add_argument("--n", type=int, help="top declining clusters to plot (default 5)")
    p.add_argument("--min-frames", dest="min_frames", type=int,
                   help="min frames for a cluster-week mean (default 25)")
    p.add_argument("--min-weeks", dest="min_weeks", type=int,
                   help="min well-sampled weeks for a cluster (default 4)")
    p.add_argument("--complete", action="store_true",
                   help="contraction only: clusters well-sampled in every week of both arenas")
    p.add_argument("--abs", dest="use_abs", action="store_true",
                   help="contraction only: rank/plot mean magnitude")

    # plot_stitched_weeks
    s = ap.add_argument_group("plot_stitched_weeks")
    s.add_argument("--view", choices=["clustertype", "arena"],
                   help="stitched-weeks view (default clustertype)")
    s.add_argument("--mode", choices=["home", "all", "all_bins"],
                   help="clustertype subset mode, ignored for --view arena (default home)")
    s.add_argument("--exclude", nargs="*",
                   help="weeks to drop as 'WEEK' or 'MOUSE:WEEK' (e.g. 2mp:20)")
    return ap


def build_argv(stage, args):
    argv = []
    for cli, dest, kind in FORWARD:
        if dest not in stage.accepts:
            continue
        val = getattr(args, dest)
        if kind == "flag":
            if val:
                argv.append(cli)
        elif kind == "list":
            if val is not None:
                argv.append(cli)
                argv += [str(x) for x in val]
        else:  # value
            if val is not None:
                argv += [cli, str(val)]
    return argv


def select_stages(args):
    names = [s.name for s in STAGES]

    def check(name):
        if name not in names:
            raise SystemExit(f"unknown stage: {name!r}\nvalid stages: {', '.join(names)}")

    if args.only:
        for n in args.only:
            check(n)
        want = set(args.only)
        return [s for s in STAGES if s.name in want]

    lo, hi = 0, len(STAGES)
    if args.from_stage:
        check(args.from_stage)
        lo = names.index(args.from_stage)
    if args.to_stage:
        check(args.to_stage)
        hi = names.index(args.to_stage) + 1
    for n in args.skip:
        check(n)
    skip = set(args.skip)
    return [s for s in STAGES[lo:hi] if s.name not in skip]


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list:
        print("Stages (in run order):\n")
        for i, s in enumerate(STAGES, 1):
            extra = f"  [flags: {', '.join(sorted(s.accepts))}]" if s.accepts else ""
            need = f"  (needs: {', '.join(sorted(s.needs))})" if s.needs else ""
            print(f"  {i:2d}. {s.name}{extra}{need}")
        return 0

    stages = select_stages(args)
    if not stages:
        print("no stages selected")
        return 0

    env = os.environ.copy()
    pypath = [str(FOLDER), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pypath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pypath)

    total = len(stages)
    for i, stage in enumerate(stages, 1):
        missing = {d for d in stage.needs if getattr(args, d, None) in (None, False)}
        if missing:
            print(f"[{i}/{total}] SKIP {stage.name} "
                  f"(needs --{'/--'.join(sorted(missing))})")
            continue

        cmd = [args.python, str(FOLDER / stage.script)] + build_argv(stage, args)
        print(f"[{i}/{total}] ==> {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            msg = f"stage {stage.name!r} exited with code {result.returncode}"
            if args.continue_on_error:
                print(f"  !! {msg} (continuing)")
            else:
                print(f"  !! {msg} (stopping; use --continue-on-error to override)")
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
