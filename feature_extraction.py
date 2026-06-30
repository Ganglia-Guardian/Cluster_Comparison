"""Extract the per-sample functional features from a session_*_out.mat
(MATLAB v7.3 / HDF5) and bin them to one value per cluster window.

StructData/func holds 4 features, each at the raw sample rate (60 samples per
binned cluster window):
    0  anterior_posterior_x_accel
    1  dorsal_ventral_y_accel
    2  y_gyro
    3  TotAccelBA   -- total body acceleration, stored as log

Binning averages every `bin_size` samples. TotAccelBA is exponentiated before
averaging so the bin value is the mean *linear* acceleration; the other three
are plain means.
"""
import h5py
import numpy as np
import pandas as pd

FEATURE_NAMES = ["anterior_posterior_x_accel", "dorsal_ventral_y_accel",
                 "y_gyro", "TotAccelBA"]
LOG_FEATURE = "TotAccelBA"     # stored as log -> exp before averaging
BIN_SIZE = 60


def find_path_to_key(h5_obj, target_key, path=""):
    for key in h5_obj.keys():
        new_path = f"{path}/{key}"
        if key == target_key:
            return new_path
        if isinstance(h5_obj[key], h5py.Group):
            result = find_path_to_key(h5_obj[key], target_key, new_path)
            if result is not None:
                return result
    return None


def load_funct_features(filepath):
    """Return the 4 per-sample feature arrays stacked as (4, n_samples)."""
    with h5py.File(filepath, "r") as f:
        struct_path = find_path_to_key(f, "StructData")
        if struct_path is None:
            raise KeyError("Could not find 'StructData' in MAT file.")
        struct_group = f[struct_path]
        if "func" not in struct_group:
            raise KeyError("'func' not found inside StructData.")
        func = struct_group["func"]

        # MATLAB {1,1-4} -> Python func[0-3][0]
        feature_refs = [func[i][0] for i in range(len(FEATURE_NAMES))]
        feature_arrays = [np.array(f[ref][()]).squeeze() for ref in feature_refs]
        return np.vstack(feature_arrays)


def bin_features(features, bin_size=BIN_SIZE):
    """Average each per-sample feature into windows of `bin_size`, returning
    (4, n_bins). The log feature (TotAccelBA) is exponentiated before averaging
    so its bin value is the mean linear acceleration."""
    n_samples = features.shape[1]
    if n_samples % bin_size != 0:
        raise ValueError(f"Sample count ({n_samples}) is not divisible by "
                         f"bin_size ({bin_size}).")
    binned = features.reshape(features.shape[0], -1, bin_size).mean(axis=2)
    log_i = FEATURE_NAMES.index(LOG_FEATURE)
    binned[log_i] = np.exp(features[log_i]).reshape(-1, bin_size).mean(axis=1)
    return binned


def combine_results(mat_file, cb_matrix=None, output_file=None,
                    full_feature_file=None, bin_size=BIN_SIZE):
    """Bin the features in `mat_file` and (optionally) attach them to a cluster
    DataFrame.

    Parameters
    ----------
    mat_file : path to a session_*_out.mat
    cb_matrix : DataFrame, optional
        Per-window cluster table to append the binned features to. Must have one
        row per bin. If omitted, only the binned-feature table is returned.
    output_file : path, optional
        If given, write the (cluster + features) table here.
    full_feature_file : path, optional
        If given, also write the raw per-sample features (TotAccelBA stays as
        log here -- it is not exponentiated).
    bin_size : int
        Samples averaged per bin (default 60).

    Returns the resulting DataFrame.
    """
    features = load_funct_features(mat_file)
    binned = bin_features(features, bin_size)
    binned_cols = {name: binned[i] for i, name in enumerate(FEATURE_NAMES)}

    if cb_matrix is not None:
        if len(cb_matrix) != binned.shape[1]:
            raise ValueError(f"cb_matrix rows ({len(cb_matrix)}) do not match "
                             f"binned feature length ({binned.shape[1]}).")
        df = cb_matrix.reset_index(drop=True).copy()
        for name, col in binned_cols.items():
            df[name] = col
    else:
        df = pd.DataFrame(binned_cols)

    if output_file is not None:
        df.to_csv(output_file, index=False)
        print("Saved:", output_file)
    if full_feature_file is not None:
        full = pd.DataFrame({name: features[i] for i, name in enumerate(FEATURE_NAMES)})
        full.to_csv(full_feature_file, index=False)
        print("Saved:", full_feature_file)
    return df
