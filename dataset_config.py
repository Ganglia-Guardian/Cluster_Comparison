"""Shared data-root / dataset-selection config for the cluster pipelines.

Every pipeline (degeneracy_analysis/, cluster_transition_*.py) reads per-mouse
folders -- each holding ``Cluster_detail_results.csv`` and ``session_1_out.mat``
-- from a *data root*. That root defaults to ``./data`` but can be pointed at
another cohort (e.g. ``early_analysis/data`` for the weeks 8-13 sessions) and
the folder list can be narrowed, three ways with this precedence:

  1. an explicit list of folder names   (``--datasets 1lc 2lc`` / CLUSTER_DATASETS)
  2. a glob filter over auto-discovery   (``--dataset-glob '*lc' '*mp'`` / CLUSTER_DATASET_GLOB)
  3. auto-discovery of every subfolder with a Cluster_detail_results.csv (default)

The data root itself is ``--data-root`` / CLUSTER_DATA_ROOT (default ``data``).

Two ways in, one resolver:
  * CLI helpers ``add_dataset_args`` / ``resolve_datasets`` for scripts that
    parse their own argv (the transition scripts, the degeneracy runner).
  * env vars, so the degeneracy runner can thread the *same* choice into each
    stage subprocess (which build their file paths at import time) without
    adding argparse to every stage -- see ``env_for``.

Glob note: the mouse cohorts are named ``<n>lc`` / ``<n>mp``; ``MOUSE_GLOB``
(``*lc``, ``*mp``) selects exactly those and excludes side folders such as
``1mp_open``. The default is no glob (discover everything with a CSV) so
existing runs are unchanged; pass ``--dataset-glob '*lc' '*mp'`` to opt in.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path

CSV_NAME = "Cluster_detail_results.csv"
DEFAULT_DATA_ROOT = "data"
LEGACY_DEGEN_OUT = "degeneracy_analysis/out"   # kept for the default ./data cohort
MOUSE_GLOB = ("*lc", "*mp")                    # convenience preset for --dataset-glob

# env var names (also the wire format the degeneracy runner uses per subprocess)
ENV_DATA_ROOT = "CLUSTER_DATA_ROOT"
ENV_DATASETS = "CLUSTER_DATASETS"
ENV_DATASET_GLOB = "CLUSTER_DATASET_GLOB"
ENV_DEGEN_OUT = "CLUSTER_DEGEN_OUT"


def _split_env(name):
    """Comma- or whitespace-separated env var -> list (or None if unset/empty)."""
    val = os.environ.get(name, "").strip()
    if not val:
        return None
    parts = [p.strip() for p in val.replace(",", " ").split()]
    return parts or None


def data_root(default=DEFAULT_DATA_ROOT):
    """The active data root: CLUSTER_DATA_ROOT if set, else `default`."""
    return Path(os.environ.get(ENV_DATA_ROOT) or default)


def discover_datasets(root=None, glob=None, datasets=None, require_csv=True):
    """Resolve the dataset-folder list under `root`.

    Precedence: explicit `datasets` (or CLUSTER_DATASETS) > `glob` filter (or
    CLUSTER_DATASET_GLOB) applied to auto-discovery > every subfolder. With
    `require_csv` (default) only folders that actually hold a
    Cluster_detail_results.csv are kept, so a bad name or a stray dir is dropped
    rather than blowing up downstream. Returns a sorted list of folder names.
    """
    root = Path(root) if root is not None else data_root()
    datasets = datasets or _split_env(ENV_DATASETS)
    glob = glob or _split_env(ENV_DATASET_GLOB)

    if datasets:
        names = list(datasets)
    elif root.is_dir():
        names = [d.name for d in root.iterdir() if d.is_dir()]
    else:
        names = []

    if glob:
        names = [n for n in names if any(fnmatch.fnmatch(n, g) for g in glob)]
    if require_csv:
        names = [n for n in names if (root / n / CSV_NAME).is_file()]
    return sorted(names)


def degen_out_root(root=None):
    """Where the degeneracy pipeline writes its out/<mouse>/ tree.

    CLUSTER_DEGEN_OUT wins if set. Otherwise the default ./data cohort keeps its
    historical location (degeneracy_analysis/out); any other cohort gets its own
    <data_root>/degeneracy_out so runs on different cohorts never clobber."""
    env = os.environ.get(ENV_DEGEN_OUT)
    if env:
        return env
    root = Path(root) if root is not None else data_root()
    if root == Path(DEFAULT_DATA_ROOT):
        return LEGACY_DEGEN_OUT
    return str(root / "degeneracy_out")


def add_dataset_args(parser, data_default=DEFAULT_DATA_ROOT):
    """Add the standard --data-root / --datasets / --dataset-glob options."""
    g = parser.add_argument_group("dataset selection")
    g.add_argument("--data-root", type=Path, default=None,
                   help=f"root folder holding the <mouse>/ dirs "
                        f"(default: ${ENV_DATA_ROOT} or {data_default})")
    g.add_argument("--datasets", nargs="+", default=None, metavar="NAME",
                   help="explicit dataset folder names (default: auto-discover all)")
    g.add_argument("--dataset-glob", nargs="+", default=None, metavar="GLOB",
                   help="glob(s) filtering discovered folders, e.g. '*lc' '*mp'")
    return parser


def resolve_datasets(args, require_csv=True):
    """(root, names) from parsed --data-root/--datasets/--dataset-glob.

    Falls back to the env vars / defaults for any option left unset, so a script
    invoked with no dataset flags behaves exactly as before."""
    root = args.data_root if getattr(args, "data_root", None) else data_root()
    names = discover_datasets(root,
                              glob=getattr(args, "dataset_glob", None),
                              datasets=getattr(args, "datasets", None),
                              require_csv=require_csv)
    return Path(root), names


def env_for(root=None, datasets=None, glob=None, degen_out=None):
    """Env-var overrides to thread the current selection into a subprocess.

    Returns a dict to merge into ``os.environ`` for the child; only the keys you
    pass are set, so the child's own defaults apply to the rest."""
    env = {}
    if root is not None:
        env[ENV_DATA_ROOT] = str(root)
    if datasets:
        env[ENV_DATASETS] = " ".join(map(str, datasets))
    if glob:
        env[ENV_DATASET_GLOB] = " ".join(map(str, glob))
    if degen_out is not None:
        env[ENV_DEGEN_OUT] = str(degen_out)
    return env
