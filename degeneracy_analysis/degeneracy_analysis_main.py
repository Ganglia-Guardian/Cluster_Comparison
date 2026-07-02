#!/usr/bin/env python3
"""Run the degeneracy_analysis pipeline end to end, in dependency order.

Each stage is a standalone script in this folder. The scripts import each other
by bare name, read from a relative "data" dir, and write under
"degeneracy_analysis/out", so every stage is run as its own subprocess with the
repo root as the working directory and this folder + repo root on PYTHONPATH.

Dependency order:
    presence_similarity   -> out/{mouse}/presence.npz  (foundation; all mice)
    feature_similarity    -> out/{mouse}/feature.npz    (reads presence.npz)
    feature_time_map      -> out/{mouse}/feature_time_map.png
    joint_plane           -> out/{mouse}/{joint_plane.png, *_candidates.csv}
    temporal_classify     -> out/{mouse}/temporal_classes.{csv,png}

Only feature_similarity takes an argument (--mouse); every other stage iterates
all mice on its own. Passing --mouse limits which feature.npz is (re)built, so
the downstream all-mice stages still expect earlier outputs for every mouse.

Examples:
    uv run python degeneracy_analysis/degeneracy_analysis_main.py               # full run
    uv run python degeneracy_analysis/degeneracy_analysis_main.py --list
    uv run python degeneracy_analysis/degeneracy_analysis_main.py --only feature_similarity --mouse 1mp
    uv run python degeneracy_analysis/degeneracy_analysis_main.py --from joint_plane
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
REPO_ROOT = FOLDER.parent

def discover_mice():
    """Mice under data/ that have a Cluster_detail_results.csv (matches
    presence_similarity.discover_mice). Auto-picks up new data/<mouse>/ dirs."""
    data_dir = REPO_ROOT / "data"
    if not data_dir.is_dir():
        return []
    return sorted(p.name for p in data_dir.iterdir()
                  if (p / "Cluster_detail_results.csv").is_file())


MICE = discover_mice()


class Stage:
    def __init__(self, name, script, accepts=(), needs=()):
        self.name = name
        self.script = script
        self.accepts = set(accepts)
        self.needs = set(needs)


STAGES = [
    Stage("presence_similarity", "presence_similarity.py"),
    Stage("feature_similarity", "feature_similarity.py", accepts=["mouse"]),
    Stage("feature_time_map", "feature_time_map.py"),
    Stage("joint_plane", "joint_plane.py"),
    Stage("temporal_classify", "temporal_classify.py"),
]

# (cli, dest, kind) with kind in {"value", "flag", "list"}.
FORWARD = [
    ("--mouse", "mouse", "value"),
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

    # feature_similarity
    f = ap.add_argument_group("feature_similarity")
    f.add_argument("--mouse", choices=MICE,
                   help="build feature.npz for one mouse only (default: all)")
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
