import argparse
import colorsys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from scipy.io import loadmat
from sklearn.metrics import silhouette_samples, silhouette_score

# Cohort palette, in homage to UT Dallas: littermate controls (*lc) in shades of
# medium-bright orange, MitoPark (*mp) in shades of medium-dark green. Cohort
# reads off the hue, individual mouse off the lightness, which frees the line
# style -- every line stays solid.
UTD_GREEN = "#2E6F4E"          # medium-dark green   -> MitoPark
UTD_ORANGE = "#F08C1E"         # medium-bright orange -> controls


def load_mat(path):
    """Load a .mat file, falling back to mat73 for MATLAB v7.3 (HDF5) files."""
    try:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        # drop the metadata keys scipy injects
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except NotImplementedError:
        import mat73  # only needed for v7.3 files

        return mat73.loadmat(path)


def mat_to_csv(mat_path, csv_path=None):
    """Extract the array variables in a .mat file into a single .csv.

    Each top-level variable becomes one (or more) columns. 1-D arrays map to a
    single column; 2-D arrays expand to one column per matrix column.
    """
    mat_path = Path(mat_path)
    if csv_path is None:
        csv_path = mat_path.with_suffix(".csv")

    data = load_mat(mat_path)

    columns = {}
    for name, value in data.items():
        arr = np.atleast_1d(np.asarray(value))
        if arr.ndim == 1:
            columns[name] = arr
        elif arr.ndim == 2:
            for col in range(arr.shape[1]):
                columns[f"{name}_{col}"] = arr[:, col]
        else:
            print(f"Skipping '{name}': unsupported shape {arr.shape}")

    if not columns:
        raise ValueError(f"No 1-D or 2-D array variables found in {mat_path}")

    df = pd.DataFrame({k: pd.Series(v) for k, v in columns.items()})
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return csv_path


def cohort_of(name):
    """'lc' (littermate control) or 'mp' (MitoPark) from a dataset/mouse name."""
    return "lc" if "lc" in str(name).lower() else "mp"


def _shades(base, n, spread=0.34):
    """``n`` shades of ``base``, darkest first, by walking lightness around it.

    Hue and saturation are held fixed so every shade still reads as the same
    colour; only lightness separates the mice within a cohort.
    """
    hue, light, sat = colorsys.rgb_to_hls(*to_rgb(base))
    if n <= 1:
        levels = [light]
    else:
        lo = max(light - spread / 2, 0.18)
        hi = min(light + spread / 2, 0.78)
        levels = np.linspace(lo, hi, n)
    return [colorsys.hls_to_rgb(hue, lv, sat) for lv in levels]


def cohort_colors(names):
    """Map dataset/mouse names to their cohort colour: ``{name: rgb}``.

    Controls (*lc) get shades of ``UTD_ORANGE``, MitoPark (*mp) shades of
    ``UTD_GREEN``. Within a cohort the names are sorted and the shades spread
    darkest-to-lightest over however many mice are present, so the same set of
    mice always produces the same assignment.
    """
    out = {}
    for coh, base in (("lc", UTD_ORANGE), ("mp", UTD_GREEN)):
        members = sorted(n for n in names if cohort_of(n) == coh)
        out.update(zip(members, _shades(base, len(members))))
    return out


def apply_house_style(fig):
    """Force the house style onto every axes in ``fig``: white backgrounds, no
    grid, and only the left/bottom spines drawn.

    Applied at save time by ``save_figure`` so the style holds no matter how the
    figure was built (including under a seaborn theme). Colorbars keep their
    outline, and non-rectilinear axes (polar, 3-D) keep their spines, since for
    those the frame carries meaning rather than decoration.
    """
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        is_colorbar = ax.get_label() == "<colorbar>"
        if is_colorbar:
            continue
        ax.set_facecolor("white")
        ax.grid(False)
        if ax.name == "rectilinear":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)


def save_figure(fig, path, **savefig_kwargs):
    """Save a figure as a JPEG plus a companion SVG, returning both paths.

    The SVG is written alongside the JPEG under an underscore-prefixed name
    ("foo.jpeg" -> "_foo.svg") so the vector copy sorts first in a directory
    listing. Any suffix on ``path`` is replaced, so passing either name works.

    The house style (see ``apply_house_style``) is enforced before writing.
    """
    apply_house_style(fig)
    path = Path(path)
    jpeg_path = path.with_suffix(".jpeg")
    svg_path = path.with_name("_" + path.stem + ".svg")
    fig.savefig(jpeg_path, **savefig_kwargs)
    fig.savefig(svg_path, **savefig_kwargs)
    return jpeg_path, svg_path


def similarity_to_distance(sim):
    """Convert a (bin x bin) similarity matrix into a valid distance matrix.

    Assumes "higher = more similar" with the diagonal holding each item's
    self-similarity (its maximum). Works for unknown similarity types: we map
    distance = max(sim) - sim, then force the diagonal to exactly 0, symmetrize,
    and clip any tiny negatives. A warning is printed if the diagonal does not
    look like the per-row maximum, which would mean the input isn't a
    self-maximal similarity and the conversion may not be meaningful.
    """
    sim = np.asarray(sim, dtype=float)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"Similarity matrix must be square, got shape {sim.shape}")

    diag = np.diag(sim)
    row_max = sim.max(axis=1)
    if not np.allclose(diag, row_max, rtol=1e-3, atol=1e-6):
        print("Warning: matrix diagonal is not the per-row maximum; the "
              "similarity may not be self-maximal. Check the conversion.")

    dist = sim.max() - sim
    dist = (dist + dist.T) / 2.0       # enforce symmetry
    np.fill_diagonal(dist, 0.0)        # self-distance must be exactly 0
    dist[dist < 0] = 0.0               # clip floating-point negatives
    return dist


def silhouette_by_week(similarity, mapping, bin_col="bin", cluster_col="cluster",
                       week_col="week", return_samples=False):
    """Silhouette score per week from a bin x bin similarity matrix.

    Parameters
    ----------
    similarity : (N, N) array
        Pairwise similarity between bins; row/col i corresponds to bin i in the
        order given by ``mapping[bin_col]`` (i.e. mapping is aligned to the
        matrix). If your mapping is in a different order, sort/reindex it first.
    mapping : DataFrame or path to CSV
        One row per bin, with columns for the bin id, its cluster label, and its
        week. Must have exactly N rows matching the matrix.
    return_samples : bool
        If True, also return a per-bin silhouette value (NaN for weeks that
        could not be scored).

    Returns
    -------
    DataFrame with columns [week, n_bins, n_clusters, silhouette]. Weeks where
    silhouette is undefined (fewer than 2 clusters, or a cluster of size N) get
    NaN with the reason printed.
    """
    if not isinstance(mapping, pd.DataFrame):
        mapping = pd.read_csv(mapping)

    # Align mapping to the matrix: row/col i of the matrix is the i-th bin in
    # ascending bin order. Sorting makes that assumption explicit and robust to
    # an out-of-order mapping file.
    mapping = mapping.sort_values(bin_col).reset_index(drop=True)

    sim = np.asarray(similarity, dtype=float)
    if len(mapping) != sim.shape[0]:
        raise ValueError(f"Mapping has {len(mapping)} rows but matrix is "
                         f"{sim.shape[0]}x{sim.shape[1]}; they must align.")

    dist = similarity_to_distance(sim)

    rows = []
    samples = np.full(sim.shape[0], np.nan)
    for week, grp in mapping.groupby(week_col):
        idx = grp.index.to_numpy()
        labels = grp[cluster_col].to_numpy()
        n_clusters = len(np.unique(labels))

        score = np.nan
        if n_clusters < 2:
            print(f"Week {week}: only {n_clusters} cluster(s) present; "
                  "silhouette undefined, skipping.")
        elif n_clusters > len(idx) - 1:
            print(f"Week {week}: {n_clusters} clusters for {len(idx)} bins "
                  "(a cluster of size 1 spans every bin); skipping.")
        else:
            sub = dist[np.ix_(idx, idx)]
            score = silhouette_score(sub, labels, metric="precomputed")
            if return_samples:
                samples[idx] = silhouette_samples(sub, labels, metric="precomputed")

        rows.append({week_col: week, "n_bins": len(idx),
                     "n_clusters": n_clusters, "silhouette": score})

    result = pd.DataFrame(rows)
    if return_samples:
        return result, samples
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a .mat file to a .csv file.")
    parser.add_argument("mat_file", help="Path to the input .mat file")
    parser.add_argument("csv_file", nargs="?", help="Path to the output .csv file "
                        "(defaults to the input name with a .csv suffix)")
    args = parser.parse_args()
    mat_to_csv(args.mat_file, args.csv_file)
