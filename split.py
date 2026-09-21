from pathlib import Path

import numpy as np

from exceptions import IncompatibleFeatureSpace


def make_split(n, val_fraction=0.2, test_fraction=0.2, seed=42, y=None, source_id="unknown"):
    """Index arrays for train, val and test. Stratified when y is given."""
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must leave rows for training")

    rng = np.random.default_rng(seed)
    n_val, n_test = int(round(val_fraction * n)), int(round(test_fraction * n))

    if y is None:
        order = rng.permutation(n)
        test_idx, val_idx = order[:n_test], order[n_test:n_test + n_val]
        train_idx = order[n_test + n_val:]
    else:
        y = np.asarray(y)
        test_idx, val_idx, train_idx = [], [], []
        # split within each class so the three sets keep the class balance
        for cls in np.unique(y):
            cls_idx = rng.permutation(np.flatnonzero(y == cls))
            n_c = cls_idx.size
            n_test_c, n_val_c = int(round(test_fraction * n_c)), int(round(val_fraction * n_c))
            test_idx.append(cls_idx[:n_test_c])
            val_idx.append(cls_idx[n_test_c:n_test_c + n_val_c])
            train_idx.append(cls_idx[n_test_c + n_val_c:])
        test_idx = rng.permutation(np.concatenate(test_idx))
        val_idx = rng.permutation(np.concatenate(val_idx))
        train_idx = rng.permutation(np.concatenate(train_idx))

    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx),
            "n": int(n), "seed": int(seed), "source_id": source_id,
            "val_fraction": float(val_fraction), "test_fraction": float(test_fraction)}


def save_split(split, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, train=split["train"], val=split["val"], test=split["test"],
             meta=np.array([split["n"], split["seed"]]),
             source_id=np.array(split["source_id"]),
             fractions=np.array([split["val_fraction"], split["test_fraction"]]))
    return p


def load_split(path, source_id=None, n=None):
    """Read a split back, checking it belongs to the dataset asking for it."""
    f = np.load(Path(path), allow_pickle=False)
    split = {"train": f["train"], "val": f["val"], "test": f["test"],
             "n": int(f["meta"][0]), "seed": int(f["meta"][1]),
             "source_id": str(f["source_id"]),
             "val_fraction": float(f["fractions"][0]),
             "test_fraction": float(f["fractions"][1])}

    if source_id is not None and split["source_id"] != source_id:
        raise IncompatibleFeatureSpace(
            f"split at {path} was built from '{split['source_id']}' but "
            f"'{source_id}' is asking for it. Row indices do not correspond "
            "between different source files, so the split cannot be reused."
        )
    if n is not None and split["n"] != n:
        raise ValueError(
            f"split at {path} covers {split['n']} rows but the dataset has {n}. "
            "Delete the split file to rebuild it, and retrain every model."
        )
    return split


def get_or_make_split(path, n, source_id, val_fraction=0.2, test_fraction=0.2,
                      seed=42, y=None):
    """Load the split if it exists, otherwise build and save it."""
    if Path(path).exists():
        return load_split(path, source_id=source_id, n=n)
    split = make_split(n, val_fraction, test_fraction, seed, y=y, source_id=source_id)
    save_split(split, path)
    return split


def check_disjoint(split):
    """Guard against a split that leaks rows between the three sets."""
    train, val, test = split["train"], split["val"], split["test"]
    total = train.size + val.size + test.size
    if total != split["n"]:
        raise ValueError(f"split covers {total} rows but claims {split['n']}")
    if np.intersect1d(train, val).size or np.intersect1d(train, test).size \
            or np.intersect1d(val, test).size:
        raise ValueError("split sets overlap")
    return True
