import hashlib
from pathlib import Path

import numpy as np

from config import CONFIG
from split import get_or_make_split, check_disjoint

JET_SOURCE_ID = "hls4ml_lhc_jets_hlf_openml"
PARTICLE_SOURCE_ID = "hls4ml_150p_h5"

EXPECTED_JET_FEATURES = [
    "zlogz", "c1_b0_mmdt", "c1_b1_mmdt", "c1_b2_mmdt", "c2_b1_mmdt",
    "c2_b2_mmdt", "d2_b1_mmdt", "d2_b2_mmdt", "d2_a1_b1_mmdt",
    "d2_a1_b2_mmdt", "m2_b1_mmdt", "m2_b2_mmdt", "n2_b1_mmdt",
    "n2_b2_mmdt", "mass_mmdt", "multiplicity",
]

P4_NAMES = ["j1_px", "j1_py", "j1_pz", "j1_e"]

# the same 16 observables inside the h5 files, where every jet column is prefixed
H5_JET_FEATURES = ["j_" + n for n in EXPECTED_JET_FEATURES]


def load_jet_level(cfg=None, split_path=None, verbose=True):
    """16 jet observables for the BDT and DNN, from the OpenML copy."""
    from sklearn.datasets import fetch_openml

    cfg = cfg or CONFIG
    Path(cfg.data.openml_cache).mkdir(parents=True, exist_ok=True)
    data = fetch_openml("hls4ml_lhc_jets_hlf", data_home=cfg.data.openml_cache)
    X_df, y_raw = data["data"], data["target"]

    feature_names = list(X_df.columns)
    if feature_names != EXPECTED_JET_FEATURES:
        mismatches = [f"position {i}: expected '{a}', got '{b}'"
                      for i, (a, b) in enumerate(zip(EXPECTED_JET_FEATURES, feature_names))
                      if a != b]
        raise ValueError(
            "Column order from fetch_openml does not match the expected order, so "
            "every importance vector would be mislabelled:\n" + "\n".join(mismatches))

    X = X_df.values.astype(np.float64)
    class_names = sorted(set(y_raw))
    y = np.searchsorted(np.array(class_names), np.asarray(y_raw))

    if cfg.data.n_jets is not None:
        X, y = X[:cfg.data.n_jets], y[:cfg.data.n_jets]

    split = get_or_make_split(split_path or cfg.data.split_index_path.replace(".npz", "_jet.npz"),
                              n=len(y), source_id=JET_SOURCE_ID,
                              val_fraction=cfg.data.val_fraction,
                              test_fraction=cfg.data.test_fraction,
                              seed=cfg.data.split_seed, y=y)
    check_disjoint(split)

    out = {"feature_names": feature_names, "class_names": list(class_names), "split": split}
    for name in ("train", "val", "test"):
        idx = split[name]
        out[f"X_{name}"], out[f"y_{name}"] = X[idx], y[idx]

    # standardisation for the DNN, fitted on train only. The BDT uses raw X.
    mean = out["X_train"].mean(axis=0)
    std = out["X_train"].std(axis=0)
    std[std < 1e-6] = 1e-6
    out["norm"] = {"mean": mean, "std": std}
    for name in ("train", "val", "test"):
        out[f"X_{name}_sc"] = (out[f"X_{name}"] - mean) / std

    if verbose:
        print(f"jet level: train {out['X_train'].shape}, val {out['X_val'].shape}, "
              f"test {out['X_test'].shape}")
    return out


def list_h5(directory):
    """Every .h5 in a directory, in a fixed order so runs are reproducible."""
    files = sorted(Path(directory).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 files in {directory}")
    return files


def estimate_memory_gb(n_jets, n_features, max_constituents=150):
    """Rough resident size of the arrays a load will produce, in GB."""
    per_jet = max_constituents * (n_features + 2 + 1) * 4     # X, coords, mask
    return n_jets * per_jet / 1024 ** 3


def read_h5_files(paths, cfg, n_jets=None, verbose=True):
    """Read constituents, coords, mask and labels from a list of files."""
    import h5py

    Xs, coords_l, masks, ys, taken = [], [], [], [], 0
    for path in paths:
        if n_jets is not None and taken >= n_jets:
            break
        with h5py.File(path, "r") as f:
            pf_names = [n.decode("utf-8") if isinstance(n, bytes) else n
                        for n in f["particleFeatureNames"][:]]
            jf_names = [n.decode("utf-8") if isinstance(n, bytes) else n
                        for n in f["jetFeatureNames"][:]]
            node_idx = [pf_names.index(n) for n in cfg.data.node_features]
            coord_idx = [pf_names.index(n) for n in cfg.data.coord_features]
            label_idx = [jf_names.index(n) for n in cfg.data.class_names]
            p4_idx = [pf_names.index(n) for n in P4_NAMES]

            want = None if n_jets is None else min(f["jets"].shape[0], n_jets - taken)
            sl = slice(None) if want is None else slice(0, want)
            constituents = f["jetConstituentList"][sl]
            jets = f["jets"][sl]

        mask = ~np.all(constituents[:, :, p4_idx] == 0.0, axis=-1)
        Xs.append(constituents[:, :, node_idx].astype(np.float32))
        coords_l.append(constituents[:, :, coord_idx].astype(np.float32))
        masks.append(mask.astype(np.float32))
        ys.append(np.argmax(jets[:, label_idx], axis=1))
        taken += len(ys[-1])

    X, coords = np.concatenate(Xs), np.concatenate(coords_l)
    mask, y = np.concatenate(masks), np.concatenate(ys)

    # dropping empty jets copies the whole array, so only do it if any are empty
    keep = mask.sum(axis=1) > 0
    if not keep.all():
        rows = np.flatnonzero(keep)
        X, coords = take_rows(X, rows), take_rows(coords, rows)
        mask, y = take_rows(mask, rows), y[rows]
    if verbose:
        print(f"  read {len(y)} jets from {len(paths)} file(s) "
              f"({estimate_memory_gb(len(y), X.shape[-1]):.2f} GB)")
    return X, coords, mask, y


def read_h5_jet_features(paths, cfg, n_jets=None, verbose=True):
    """The 16 jet observables and labels, from the constituents' own files."""
    import h5py

    Xs, ys, taken = [], [], 0
    for path in paths:
        if n_jets is not None and taken >= n_jets:
            break
        with h5py.File(path, "r") as f:
            pf_names = [n.decode("utf-8") if isinstance(n, bytes) else n
                        for n in f["particleFeatureNames"][:]]
            jf_names = [n.decode("utf-8") if isinstance(n, bytes) else n
                        for n in f["jetFeatureNames"][:]]
            hlf_idx = [jf_names.index(n) for n in H5_JET_FEATURES]
            label_idx = [jf_names.index(n) for n in cfg.data.class_names]
            p4_idx = [pf_names.index(n) for n in P4_NAMES]

            want = None if n_jets is None else min(f["jets"].shape[0], n_jets - taken)
            sl = slice(None) if want is None else slice(0, want)
            jets = f["jets"][sl]
            p4 = f["jetConstituentList"][sl][:, :, p4_idx]

        mask = ~np.all(p4 == 0.0, axis=-1)
        keep = mask.sum(axis=1) > 0
        Xs.append(jets[keep][:, hlf_idx].astype(np.float64))
        ys.append(np.argmax(jets[keep][:, label_idx], axis=1))
        taken += len(jets)

    X, y = np.concatenate(Xs), np.concatenate(ys)
    if verbose:
        print(f"  read {len(y)} jets from {len(paths)} file(s)")
    return X, y


def load_jet_level_h5(cfg=None, split_path=None, n_jets=None, n_test_jets=None,
                      verbose=True):
    """The BDT and DNN view read from the h5 files, aligned row for row with the particles."""
    cfg = cfg or CONFIG
    n_jets = n_jets if n_jets is not None else cfg.data.n_jets
    n_test_jets = n_test_jets if n_test_jets is not None else cfg.data.n_test_jets
    # the particle split file on purpose; sharing it is what keeps rows aligned
    split_path = split_path or cfg.data.split_index_path.replace(".npz", "_particle.npz")

    if cfg.data.particle_file or not cfg.data.use_official_partition:
        raise NotImplementedError(
            "load_jet_level_h5 follows the official partition only. The particle "
            "loader's three-way branch would need the same treatment before this "
            "can share a split with it.")

    train_files = list_h5(cfg.data.particle_train_dir)
    test_files = list_h5(cfg.data.particle_val_dir)
    if verbose:
        print("jet level from h5: official partition, train/ for train+val, val/ for test")
    X, y = read_h5_jet_features(train_files, cfg, n_jets, verbose)
    Xt, yt = read_h5_jet_features(test_files, cfg, n_test_jets, verbose)

    val_share = cfg.data.val_fraction / (1.0 - cfg.data.test_fraction)
    inner = get_or_make_split(split_path, n=len(y), source_id=PARTICLE_SOURCE_ID,
                              val_fraction=val_share, test_fraction=0.0,
                              seed=cfg.data.split_seed, y=y)
    check_disjoint(inner)

    split = {"train": inner["train"], "val": inner["val"],
             "test": np.arange(len(yt)), "n": int(len(y) + len(yt)),
             "seed": inner["seed"], "source_id": PARTICLE_SOURCE_ID,
             "val_fraction": cfg.data.val_fraction,
             "test_fraction": float(len(yt) / (len(y) + len(yt))),
             "test_from": str(cfg.data.particle_val_dir)}

    out = {"feature_names": list(EXPECTED_JET_FEATURES),
           "class_names": list(cfg.data.class_names), "split": split,
           "X_test": Xt, "y_test": yt}
    for name in ("train", "val"):
        idx = inner[name]
        out[f"X_{name}"], out[f"y_{name}"] = X[idx], y[idx]

    # standardisation for the DNN, fitted on train only. The BDT uses raw X.
    mean = out["X_train"].mean(axis=0)
    std = out["X_train"].std(axis=0)
    std[std < 1e-6] = 1e-6
    out["norm"] = {"mean": mean, "std": std}
    for name in ("train", "val", "test"):
        out[f"X_{name}_sc"] = (out[f"X_{name}"] - mean) / std

    if verbose:
        print(f"jet level: train {out['X_train'].shape}, val {out['X_val'].shape}, "
              f"test {out['X_test'].shape}")
    return out


def take_rows(X, idx, chunk=8192):
    """X[idx] into a preallocated output, filled in chunks."""
    idx = np.asarray(idx)
    out = np.empty((len(idx),) + X.shape[1:], dtype=X.dtype)
    for start in range(0, len(idx), chunk):
        np.take(X, idx[start:start + chunk], axis=0, out=out[start:start + chunk])
    return out


def _norm_from(X, mask):
    """Statistics over real constituents only."""
    flat = X[mask.astype(bool)]
    return {"mean": flat.mean(axis=0), "std": np.maximum(flat.std(axis=0), 1e-6)}


ARRAY_KEYS = [f"{k}_{s}" for s in ("train", "val", "test")
              for k in ("X", "coords", "mask", "y")]


def cache_key(cfg, n_jets, n_test_jets, single):
    """Identity of a preprocessed load: everything that changes the arrays."""
    parts = [str(single or cfg.data.particle_train_dir), str(cfg.data.particle_val_dir),
             f"official={cfg.data.use_official_partition}", f"n={n_jets}",
             f"ntest={n_test_jets}", f"seed={cfg.data.split_seed}",
             f"val={cfg.data.val_fraction}", f"test={cfg.data.test_fraction}",
             ",".join(cfg.data.node_features), ",".join(cfg.data.coord_features),
             ",".join(cfg.data.class_names)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def read_cache(directory, verbose=True):
    """Load a cached split, or None if it is missing or incomplete."""
    d = Path(directory)
    meta = d / "meta.npz"
    if not meta.exists() or not all((d / f"{k}.npy").exists() for k in ARRAY_KEYS):
        return None
    out = {k: np.load(d / f"{k}.npy") for k in ARRAY_KEYS}
    m = np.load(meta, allow_pickle=False)
    out["norm"] = {"mean": m["mean"], "std": m["std"]}
    out["split"] = {"train": m["train"], "val": m["val"], "test": m["test"],
                    "n": int(m["n"]), "seed": int(m["seed"]),
                    "source_id": PARTICLE_SOURCE_ID,
                    "val_fraction": float(m["fractions"][0]),
                    "test_fraction": float(m["fractions"][1])}
    if verbose:
        print(f"particle level: cache hit at {d}")
    return out


def write_cache(directory, out):
    """Write the split arrays so the next run skips reading and splitting."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for k in ARRAY_KEYS:
        np.save(d / f"{k}.npy", out[k])
    s = out["split"]
    np.savez(d / "meta.npz", mean=out["norm"]["mean"], std=out["norm"]["std"],
             train=s["train"], val=s["val"], test=s["test"],
             n=s["n"], seed=s["seed"],
             fractions=np.array([s["val_fraction"], s["test_fraction"]]))
    return d


def load_particle_level(cfg=None, h5_path=None, split_path=None, n_jets=None,
                        n_test_jets=None, use_cache=True, verbose=True):
    """Constituents and padding mask for the GNN, PFN and EFN."""
    cfg = cfg or CONFIG
    n_jets = n_jets if n_jets is not None else cfg.data.n_jets
    n_test_jets = n_test_jets if n_test_jets is not None else cfg.data.n_test_jets
    single = h5_path or cfg.data.particle_file
    split_path = split_path or cfg.data.split_index_path.replace(".npz", "_particle.npz")

    names = {"node_features": list(cfg.data.node_features),
             "coord_features": list(cfg.data.coord_features),
             "class_names": list(cfg.data.class_names)}

    cache = None
    if use_cache and cfg.data.cache_dir:
        cache = Path(cfg.data.cache_dir) / cache_key(cfg, n_jets, n_test_jets, single)
        hit = read_cache(cache, verbose)
        if hit is not None:
            hit.update(names)
            return hit

    if single or not cfg.data.use_official_partition:
        paths = [Path(single)] if single else list_h5(cfg.data.particle_train_dir)
        if verbose:
            print(f"particle level: three-way split of {len(paths)} file(s)")
        X, coords, mask, y = read_h5_files(paths, cfg, n_jets, verbose)

        split = get_or_make_split(split_path, n=len(y), source_id=PARTICLE_SOURCE_ID,
                                  val_fraction=cfg.data.val_fraction,
                                  test_fraction=cfg.data.test_fraction,
                                  seed=cfg.data.split_seed, y=y)
        check_disjoint(split)

        out = {"split": split}
        for name in ("train", "val", "test"):
            idx = split[name]
            out[f"X_{name}"] = take_rows(X, idx)
            out[f"coords_{name}"] = take_rows(coords, idx)
            out[f"mask_{name}"], out[f"y_{name}"] = take_rows(mask, idx), y[idx]
        del X, coords, mask
        out["norm"] = _norm_from(out["X_train"], out["mask_train"])
    else:
        # the dataset's own partition: train/ for fitting, val/ held out
        if verbose:
            print("particle level: official partition, train/ for train+val, val/ for test")
        train_files, test_files = list_h5(cfg.data.particle_train_dir), list_h5(cfg.data.particle_val_dir)
        X, coords, mask, y = read_h5_files(train_files, cfg, n_jets, verbose)
        Xt, coords_t, mask_t, yt = read_h5_files(test_files, cfg, n_test_jets, verbose)

        # only train/ is split, into train and val; test comes from val/
        val_share = cfg.data.val_fraction / (1.0 - cfg.data.test_fraction)
        inner = get_or_make_split(split_path, n=len(y), source_id=PARTICLE_SOURCE_ID,
                                  val_fraction=val_share, test_fraction=0.0,
                                  seed=cfg.data.split_seed, y=y)
        check_disjoint(inner)

        split = {"train": inner["train"], "val": inner["val"],
                 "test": np.arange(len(yt)), "n": int(len(y) + len(yt)),
                 "seed": inner["seed"], "source_id": PARTICLE_SOURCE_ID,
                 "val_fraction": cfg.data.val_fraction,
                 "test_fraction": float(len(yt) / (len(y) + len(yt))),
                 "test_from": str(cfg.data.particle_val_dir)}
        out = {"split": split,
               "X_test": Xt, "coords_test": coords_t, "mask_test": mask_t, "y_test": yt}
        for name in ("train", "val"):
            idx = inner[name]
            out[f"X_{name}"] = take_rows(X, idx)
            out[f"coords_{name}"] = take_rows(coords, idx)
            out[f"mask_{name}"], out[f"y_{name}"] = take_rows(mask, idx), y[idx]
        del X, coords, mask
        out["norm"] = _norm_from(out["X_train"], out["mask_train"])

    if cache is not None:
        write_cache(cache, out)
        if verbose:
            print(f"  cached to {cache}")
    out.update(names)

    if verbose:
        avg_n = out["mask_train"].sum(axis=1).mean()
        print(f"  train {len(out['y_train'])} / val {len(out['y_val'])} / "
              f"test {len(out['y_test'])}, {avg_n:.1f} constituents per jet on average")
    return out


def normalise(X, mask, norm):
    """Apply stored stats. Padded slots stay at zero."""
    Xn = (X - norm["mean"]) / norm["std"]
    return Xn * mask[..., None]
