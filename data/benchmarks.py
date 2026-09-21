from pathlib import Path

import numpy as np

from config import CONFIG
from split import check_disjoint, get_or_make_split

COVERTYPE_SOURCE_ID = "covertype_sklearn"
ADULT_SOURCE_ID = "adult_openml_v2"

# fetch_covtype puts the 10 quantitative columns first, then 44 binary indicators
COVERTYPE_CONTINUOUS = [
    "elevation", "aspect", "slope",
    "horizontal_distance_to_hydrology", "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways",
    "hillshade_9am", "hillshade_noon", "hillshade_3pm",
    "horizontal_distance_to_fire_points",
]
COVERTYPE_CLASSES = ["spruce_fir", "lodgepole_pine", "ponderosa_pine",
                     "cottonwood_willow", "aspen", "douglas_fir", "krummholz"]

ADULT_CONTINUOUS = ["age", "fnlwgt", "education-num", "capital-gain",
                    "capital-loss", "hours-per-week"]
ADULT_CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
                     "relationship", "race", "sex", "native-country"]


def _stratified_take(y, n, rng, exclude=None):
    """Exactly n row indices keeping the class balance of y, avoiding excluded rows."""
    y = np.asarray(y)
    available = np.ones(len(y), dtype=bool)
    if exclude is not None:
        available[exclude] = False
    n_available = int(available.sum())
    n = min(n, n_available)

    classes = np.unique(y)
    per_class = {c: rng.permutation(np.flatnonzero(available & (y == c))) for c in classes}
    exact = {c: n * len(per_class[c]) / n_available for c in classes}
    want = {c: min(int(np.floor(exact[c])), len(per_class[c])) for c in classes}

    # hand the rounded-off rows to the classes with the largest remainder
    order = sorted(classes, key=lambda c: exact[c] - np.floor(exact[c]), reverse=True)
    for c in order:
        if sum(want.values()) >= n:
            break
        if want[c] < len(per_class[c]):
            want[c] += 1

    picked = [per_class[c][:want[c]] for c in classes]
    leftover = np.concatenate([per_class[c][want[c]:] for c in classes]) \
        if sum(want.values()) < n else np.array([], dtype=int)
    if len(leftover):
        picked.append(rng.permutation(leftover)[:n - sum(want.values())])
    return np.sort(np.concatenate(picked).astype(int))


def _assemble(X, y, feature_names, class_names, source_id, cfg,
              n_pool=None, n_test=None, split_path=None, extra=None, verbose=True):
    """Held-out test block plus an inner train/validation split, as the jet loader does."""
    cfg = cfg or CONFIG
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(cfg.data.split_seed)

    n_test = n_test if n_test is not None else int(round(cfg.data.test_fraction * len(y)))
    n_test = min(n_test, len(y) - 2)
    test_idx = _stratified_take(y, n_test, rng)

    remaining = np.setdiff1d(np.arange(len(y)), test_idx)
    if len(remaining) < len(test_idx):
        raise ValueError(
            f"a test block of {len(test_idx)} rows leaves only {len(remaining)} for "
            f"training and validation out of {len(y)}. Pass a smaller n_test_rows.")
    if n_pool is None or n_pool >= len(remaining):
        pool_idx = remaining
    else:
        pool_idx = _stratified_take(y, n_pool, rng, exclude=test_idx)

    X_pool, y_pool = X[pool_idx], y[pool_idx]
    Xt, yt = X[test_idx], y[test_idx]

    # the pool holds train and val only, so the val share is renormalised
    val_share = cfg.data.val_fraction / (1.0 - cfg.data.test_fraction)
    split_path = split_path or f"artifacts/split_{source_id}_{len(y_pool)}.npz"
    inner = get_or_make_split(split_path, n=len(y_pool), source_id=source_id,
                              val_fraction=val_share, test_fraction=0.0,
                              seed=cfg.data.split_seed, y=y_pool)
    check_disjoint(inner)

    split = {"train": inner["train"], "val": inner["val"],
             "test": np.arange(len(yt)), "n": int(len(y_pool) + len(yt)),
             "seed": inner["seed"], "source_id": source_id,
             "val_fraction": cfg.data.val_fraction,
             "test_fraction": float(len(yt) / (len(y_pool) + len(yt))),
             "pool_rows": pool_idx, "test_rows": test_idx}

    out = {"feature_names": list(feature_names), "class_names": list(class_names),
           "split": split, "X_test": Xt, "y_test": yt}
    for name in ("train", "val"):
        idx = inner[name]
        out[f"X_{name}"], out[f"y_{name}"] = X_pool[idx], y_pool[idx]

    # standardisation for the DNN, fitted on training rows only. The BDT uses raw X.
    mean = out["X_train"].mean(axis=0)
    std = out["X_train"].std(axis=0)
    std[std < 1e-6] = 1e-6
    out["norm"] = {"mean": mean, "std": std}
    for name in ("train", "val", "test"):
        out[f"X_{name}_sc"] = (out[f"X_{name}"] - mean) / std

    if extra:
        out.update(extra)
    if verbose:
        counts = np.bincount(out["y_train"], minlength=len(class_names))
        shares = ", ".join(f"{c / counts.sum():.3f}" for c in counts)
        print(f"{source_id}: {len(out['y_train'])} train, {len(out['y_val'])} val, "
              f"{len(yt)} test, {X.shape[1]} features, {len(class_names)} classes")
        print(f"  train class shares: {shares}")
    return out


def load_covertype(cfg=None, n_rows=200000, n_test_rows=50000, features="continuous",
                   split_path=None, verbose=True):
    """Forest cover type, 7 classes."""
    from sklearn.datasets import fetch_covtype

    cfg = cfg or CONFIG
    Path(cfg.data.openml_cache).mkdir(parents=True, exist_ok=True)
    data = fetch_covtype(data_home=cfg.data.openml_cache)
    X, y = np.asarray(data["data"], dtype=np.float64), np.asarray(data["target"])

    if features == "continuous":
        X, feature_names = X[:, :10], list(COVERTYPE_CONTINUOUS)
    elif features == "all":
        feature_names = (list(COVERTYPE_CONTINUOUS)
                         + [f"wilderness_{i}" for i in range(4)]
                         + [f"soil_type_{i}" for i in range(40)])
        if X.shape[1] != len(feature_names):
            raise ValueError(f"expected 54 columns from fetch_covtype, got {X.shape[1]}")
    else:
        raise ValueError("features must be 'continuous' or 'all'")

    # fetch_covtype labels the classes 1..7
    y = y.astype(int) - 1
    source_id = f"{COVERTYPE_SOURCE_ID}_{features}"
    return _assemble(X, y, feature_names, COVERTYPE_CLASSES, source_id, cfg,
                     n_pool=n_rows, n_test=n_test_rows, split_path=split_path,
                     verbose=verbose)


def _encode_categoricals(df, columns, min_level_frac, verbose):
    """One column per level, with rare levels pooled so no column is nearly constant."""
    blocks, names, groups = [], [], []
    for col in columns:
        raw = df[col].astype("object").where(df[col].notna(), "missing").astype(str)
        shares = raw.value_counts(normalize=True)
        keep = set(shares[shares >= min_level_frac].index)
        levels = sorted(keep)
        pooled = raw.where(raw.isin(keep), "other")
        if (pooled == "other").any():
            levels = levels + ["other"]
        for level in levels:
            blocks.append((pooled == level).to_numpy(dtype=np.float64))
            names.append(f"{col}={level}")
            groups.append(col)
        if verbose and len(levels) < raw.nunique():
            print(f"  {col}: {raw.nunique()} levels, {len(levels)} kept "
                  f"at min share {min_level_frac}")
    return np.column_stack(blocks), names, groups


def load_adult(cfg=None, categorical="onehot", min_level_frac=0.01, n_rows=None,
               n_test_rows=None, split_path=None, verbose=True):
    """Adult income, 2 classes, 48,842 rows."""
    from sklearn.datasets import fetch_openml

    cfg = cfg or CONFIG
    Path(cfg.data.openml_cache).mkdir(parents=True, exist_ok=True)
    data = fetch_openml("adult", version=2, as_frame=True, data_home=cfg.data.openml_cache)
    df, y_raw = data["data"], data["target"]

    missing = [c for c in ADULT_CONTINUOUS + ADULT_CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"columns missing from the OpenML copy of adult: {missing}")

    X_cont = df[ADULT_CONTINUOUS].to_numpy(dtype=np.float64)

    if categorical == "onehot":
        X_cat, cat_names, cat_groups = _encode_categoricals(
            df, ADULT_CATEGORICAL, min_level_frac, verbose)
    elif categorical == "ordinal":
        cols = []
        for col in ADULT_CATEGORICAL:
            raw = df[col].astype("object").where(df[col].notna(), "missing").astype(str)
            levels = np.array(sorted(raw.unique()))
            cols.append(np.searchsorted(levels, raw.to_numpy()).astype(np.float64))
        X_cat = np.column_stack(cols)
        cat_names, cat_groups = list(ADULT_CATEGORICAL), list(ADULT_CATEGORICAL)
    else:
        raise ValueError("categorical must be 'onehot' or 'ordinal'")

    X = np.hstack([X_cont, X_cat])
    feature_names = list(ADULT_CONTINUOUS) + cat_names
    groups = list(ADULT_CONTINUOUS) + cat_groups

    class_names = sorted(str(c) for c in np.unique(np.asarray(y_raw).astype(str)))
    y = np.searchsorted(np.array(class_names), np.asarray(y_raw).astype(str))

    source_id = f"{ADULT_SOURCE_ID}_{categorical}"
    return _assemble(X, y, feature_names, class_names, source_id, cfg,
                     n_pool=n_rows, n_test=n_test_rows, split_path=split_path,
                     extra={"feature_groups": groups}, verbose=verbose)


LOADERS = {"covertype": load_covertype, "adult": load_adult}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="fetch a dataset and report its shape")
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="covertype")
    args = parser.parse_args()
    out = LOADERS[args.dataset]()
    print(f"X_train {out['X_train'].shape}, X_test {out['X_test'].shape}")
    shown = ", ".join(out["feature_names"][:12])
    print(f"features: {shown}{' ...' if len(out['feature_names']) > 12 else ''}")
