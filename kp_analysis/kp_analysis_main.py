#!/usr/bin/env python3
"""Run the kp_analysis pipeline end to end, in dependency order.

Each stage is a standalone script in this folder. The scripts import each other
by bare name and use relative paths ("kp_analysis/...", "data/..."), so every
stage is run as its own subprocess with the repo root as the working directory
and this folder + repo root on PYTHONPATH.

kp_features.py is a shared library / self-test (imported by the others) and is
not a pipeline stage, so it is not run here.

Dependency order:
    kp_cluster_compare          -> ami_results.csv, *_contingency.png
    kp_aligned_clusters         -> *_aligned_clusters.csv
    kp_conditional_pose         -> *_conditional_pose.csv
    kp_silhouette               -> *_silhouette.png
    kp_pose_grouping            -> *_pose_dendrogram.png, *_pose_corecruit_z.csv
    kp_pose_grouping_rarefaction-> *_grouping_rarefaction.csv, pose_grouping_rarefaction.png
    kp_pose_prior_imu           -> *_window_order.csv, *_reverse_contingency.csv,
                                   (*_ap_within_pose.csv needs --sim/--mat)
    kp_imu_pose_coherence       -> *_imu_pose_coherence.png  (needs --sim + --session)

The two IMU stages need an external IMU similarity matrix. kp_imu_pose_coherence
is skipped unless --sim and --session are given; kp_pose_prior_imu runs its
discrete part regardless and uses --sim/--mat/--session/--pref when provided.

Examples:
    uv run python kp_analysis/kp_analysis_main.py                 # full run (IMU coherence skipped)
    uv run python kp_analysis/kp_analysis_main.py --list
    uv run python kp_analysis/kp_analysis_main.py --sim kp_analysis/data/wk8lc/imu_sim.npy --session wk8lc
    uv run python kp_analysis/kp_analysis_main.py --only kp_imu_pose_coherence \
        --sim path/to/imu_sim.npy --session wk8mp --exclude-rest
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
REPO_ROOT = FOLDER.parent

SESSIONS = ["wk8lc", "wk8mp"]


class Stage:
    def __init__(self, name, script, accepts=(), needs=()):
        self.name = name
        self.script = script
        self.accepts = set(accepts)
        self.needs = set(needs)


STAGES = [
    Stage("kp_cluster_compare", "kp_cluster_compare.py"),
    Stage("kp_aligned_clusters", "kp_aligned_clusters.py"),
    Stage("kp_conditional_pose", "kp_conditional_pose.py"),
    Stage("kp_silhouette", "kp_silhouette.py"),
    Stage("kp_pose_grouping", "kp_pose_grouping.py"),
    Stage("kp_pose_grouping_rarefaction", "kp_pose_grouping_rarefaction.py"),
    Stage("kp_pose_prior_imu", "kp_pose_prior_imu.py",
          accepts=["sim", "mat", "session", "pref"]),
    Stage("kp_imu_pose_coherence", "kp_imu_pose_coherence.py",
          accepts=["sim", "session", "exclude_rest"], needs=["sim", "session"]),
]

# (cli, dest, kind) with kind in {"value", "flag", "list"}.
FORWARD = [
    ("--sim", "sim", "value"),
    ("--mat", "mat", "value"),
    ("--session", "session", "value"),
    ("--pref", "pref", "value"),
    ("--exclude-rest", "exclude_rest", "flag"),
]


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_argument_group("pipeline control")
    g.add_argument("--list", action="store_true",
                   help="list the stages (with the flags each accepts) and exit")
    g.add_argument("--only", nargs="+", metavar="STAGE", help="run only these stage(s)")
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

    # IMU stages: kp_pose_prior_imu / kp_imu_pose_coherence
    imu = ap.add_argument_group("IMU stages (kp_pose_prior_imu / kp_imu_pose_coherence)")
    imu.add_argument("--sim", help="path to an (N,N) IMU similarity .npy")
    imu.add_argument("--mat",
                     help="kp_pose_prior_imu: extract similarity from a session_*_out.mat first")
    imu.add_argument("--session", choices=SESSIONS, help="which session")
    imu.add_argument("--pref", help="kp_pose_prior_imu AP preference: min | median | float (default min)")
    imu.add_argument("--exclude-rest", dest="exclude_rest", action="store_true",
                     help="kp_imu_pose_coherence: exclude rest-dominated clusters")
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
        else:
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
