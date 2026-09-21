from pathlib import Path

import numpy as np

from config import CONFIG
from split import get_or_make_split, check_disjoint


def _read_any(path, array_key=None, label_key=None):
    """Read X, y and feature names from whichever file format was handed over."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npy":
        return np.load(p), None, None

    if suffix == ".npz":
        f = np.load(p, allow_pickle=True)
        X = f[array_key or "X"]
        y = f[label_key] if label_key else (f["y"] if "y" in f else None)
        names = list(f["feature_names"]) if "feature_names" in f else None
        return X, y, names

    if suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        label_col = label_key or "label"
        y = df.pop(label_col).values if label_col in df.columns else None
        return df.values, y, list(df.columns)

    if suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            X = f[array_key or "X"][:]
            y = f[label_key][:] if label_key and label_key in f else (
                f["y"][:] if "y" in f else None)
            names = None
            if "feature_names" in f:
                names = [n.decode("utf-8") if isinstance(n, bytes) else n
                         for n in f["feature_names"][:]]
        return X, y, names

    raise ValueError(f"unsupported file type '{suffix}'. Use .npz, .npy, .csv or .h5, "
                     "or pass arrays directly.")


def _encode_labels(y):
    """Map labels of any type to 0..n_classes-1, keeping the original names."""
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    return np.searchsorted(classes, y), [str(c) for c in classes]


def _finish(X_parts, y, split, feature_names, class_names, norm, extra=None):
    out = {"feature_names": feature_names, "class_names": class_names,
           "norm": norm, "split": split}
    for name in ("train", "val", "test"):
        idx = split[name]
        for key, arr in X_parts.items():
            out[f"{key}_{name}"] = arr[idx]
        out[f"y_{name}"] = y[idx]
    if extra:
        out.update(extra)
    return out


def load_tabular(source, y=None, feature_names=None, class_names=None,
                 array_key=None, label_key=None, cfg=None, split_path=None,
                 source_id="user_tabular", verbose=True):
    """Rows and columns. source is a path or an (n_rows, n_features) array."""
    cfg = cfg or CONFIG

    if isinstance(source, (str, Path)):
        X, y_file, names = _read_any(source, array_key, label_key)
        y = y if y is not None else y_file
        feature_names = feature_names or names
    else:
        X = np.asarray(source)

    if y is None:
        raise ValueError("no labels found. Pass y=..., or label_key=... naming the "
                         "label column or dataset inside the file.")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2d (rows, features) array, got shape {X.shape}. "
                         "For set-valued inputs use load_set_valued instead.")

    y_enc, found_classes = _encode_labels(y)
    class_names = class_names or found_classes
    feature_names = list(feature_names) if feature_names is not None else \
        [f"feature_{i}" for i in range(X.shape[1])]
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"{len(feature_names)} feature names for {X.shape[1]} columns")

    split = get_or_make_split(split_path or cfg.data.split_index_path,
                              n=len(y_enc), source_id=source_id,
                              val_fraction=cfg.data.val_fraction,
                              test_fraction=cfg.data.test_fraction,
                              seed=cfg.data.split_seed, y=y_enc)
    check_disjoint(split)

    mean = X[split["train"]].mean(axis=0)
    std = np.maximum(X[split["train"]].std(axis=0), 1e-6)
    out = _finish({"X": X}, y_enc, split, feature_names, class_names,
                  {"mean": mean, "std": std})
    for name in ("train", "val", "test"):
        out[f"X_{name}_sc"] = (out[f"X_{name}"] - mean) / std

    if verbose:
        print(f"tabular: {X.shape[0]} rows, {X.shape[1]} features, "
              f"{len(class_names)} classes, train {len(split['train'])}")
    return out


def load_set_valued(X, y, mask=None, coords=None, feature_names=None,
                    class_names=None, coord_indices=(0, 1), cfg=None,
                    split_path=None, source_id="user_set_valued", verbose=True):
    """Variable-length sets, padded. X is (examples, max_objects, features)."""
    cfg = cfg or CONFIG
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"expected (examples, objects, features), got shape {X.shape}")

    mask = np.asarray(mask, dtype=np.float32) if mask is not None else \
        (~np.all(X == 0.0, axis=-1)).astype(np.float32)
    coords = np.asarray(coords, dtype=np.float32) if coords is not None else \
        X[:, :, list(coord_indices)]

    y_enc, found_classes = _encode_labels(y)
    class_names = class_names or found_classes
    feature_names = list(feature_names) if feature_names is not None else \
        [f"feature_{i}" for i in range(X.shape[2])]

    keep = mask.sum(axis=1) > 0
    X, mask, coords, y_enc = X[keep], mask[keep], coords[keep], y_enc[keep]

    split = get_or_make_split(split_path or cfg.data.split_index_path,
                              n=len(y_enc), source_id=source_id,
                              val_fraction=cfg.data.val_fraction,
                              test_fraction=cfg.data.test_fraction,
                              seed=cfg.data.split_seed, y=y_enc)
    check_disjoint(split)

    train_mask = mask[split["train"]].astype(bool)
    flat = X[split["train"]][train_mask]
    norm = {"mean": flat.mean(axis=0), "std": np.maximum(flat.std(axis=0), 1e-6)}

    out = _finish({"X": X, "mask": mask, "coords": coords}, y_enc, split,
                  feature_names, class_names, norm)
    out["node_features"] = feature_names
    if verbose:
        print(f"set-valued: {X.shape[0]} examples, up to {X.shape[1]} objects, "
              f"{X.shape[2]} features, {mask.sum(axis=1).mean():.1f} real objects on average")
    return out
