import numpy as np
import pytest

from config import Config
from data import benchmarks, hls4ml, user
from split import (check_disjoint, get_or_make_split, load_split, make_split,
                        save_split)
from exceptions import IncompatibleFeatureSpace


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    return rng.integers(0, 5, size=400)


def test_take_rows_matches_fancy_indexing():
    rng = np.random.default_rng(0)
    X = rng.random((500, 12, 4)).astype(np.float32)
    idx = np.sort(rng.choice(500, 300, replace=False))
    assert np.array_equal(hls4ml.take_rows(X, idx, chunk=64), X[idx])


def test_take_rows_handles_a_chunk_larger_than_the_selection():
    X = np.arange(40, dtype=np.float32).reshape(10, 4)
    idx = np.array([1, 7])
    assert np.array_equal(hls4ml.take_rows(X, idx, chunk=8192), X[idx])


def test_particle_cache_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    out = {}
    for name, n in (("train", 30), ("val", 10), ("test", 12)):
        out[f"X_{name}"] = rng.random((n, 8, 3)).astype(np.float32)
        out[f"coords_{name}"] = rng.random((n, 8, 2)).astype(np.float32)
        out[f"mask_{name}"] = (rng.random((n, 8)) > 0.3).astype(np.float32)
        out[f"y_{name}"] = rng.integers(0, 5, size=n)
    out["norm"] = {"mean": np.zeros(3), "std": np.ones(3)}
    out["split"] = {"train": np.arange(30), "val": np.arange(10), "test": np.arange(12),
                    "n": 52, "seed": 42, "val_fraction": 0.2, "test_fraction": 0.2}

    hls4ml.write_cache(tmp_path, out)
    back = hls4ml.read_cache(tmp_path, verbose=False)
    for key in hls4ml.ARRAY_KEYS:
        assert np.array_equal(back[key], out[key])
    assert back["split"]["n"] == 52


def test_cache_is_a_miss_when_nothing_was_written(tmp_path):
    assert hls4ml.read_cache(tmp_path, verbose=False) is None


def test_cache_key_changes_with_the_feature_set():
    cfg = Config()
    a = hls4ml.cache_key(cfg, 1000, 500, None)
    cfg.data.node_features = cfg.data.node_features[:5]
    assert hls4ml.cache_key(cfg, 1000, 500, None) != a


def test_split_sets_are_disjoint_and_cover_everything(labels):
    split = make_split(len(labels), y=labels, source_id="test")
    assert check_disjoint(split)


def test_split_is_reproducible_from_the_seed(labels):
    a = make_split(len(labels), seed=7, y=labels, source_id="test")
    b = make_split(len(labels), seed=7, y=labels, source_id="test")
    assert np.array_equal(a["test"], b["test"])
    c = make_split(len(labels), seed=8, y=labels, source_id="test")
    assert not np.array_equal(a["test"], c["test"])


def test_stratified_split_keeps_the_class_balance(labels):
    split = make_split(len(labels), y=labels, source_id="test")
    full = np.bincount(labels, minlength=5) / len(labels)
    for name in ("train", "val", "test"):
        part = np.bincount(labels[split[name]], minlength=5) / len(split[name])
        assert np.allclose(part, full, atol=0.03)


def test_split_round_trips_through_disk(tmp_path, labels):
    split = make_split(len(labels), y=labels, source_id="mine")
    path = save_split(split, tmp_path / "split.npz")
    back = load_split(path, source_id="mine", n=len(labels))
    assert np.array_equal(split["train"], back["train"])
    assert back["seed"] == split["seed"]


def test_a_split_from_another_dataset_is_refused(tmp_path, labels):
    save_split(make_split(len(labels), y=labels, source_id="jet_level"), tmp_path / "s.npz")
    with pytest.raises(IncompatibleFeatureSpace):
        load_split(tmp_path / "s.npz", source_id="particle_level")


def test_a_split_of_the_wrong_size_is_refused(tmp_path, labels):
    save_split(make_split(len(labels), y=labels, source_id="x"), tmp_path / "s.npz")
    with pytest.raises(ValueError, match="covers 400 rows"):
        load_split(tmp_path / "s.npz", source_id="x", n=500)


def test_get_or_make_reuses_the_file_it_wrote(tmp_path, labels):
    path = tmp_path / "s.npz"
    first = get_or_make_split(path, len(labels), "x", y=labels)
    second = get_or_make_split(path, len(labels), "x", y=labels)
    assert np.array_equal(first["test"], second["test"])


def make_cfg(tmp_path):
    cfg = Config()
    cfg.data.split_index_path = str(tmp_path / "split.npz")
    return cfg


def test_load_tabular_from_arrays(tmp_path):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(300, 6)), rng.integers(0, 3, size=300)
    out = user.load_tabular(X, y=y, cfg=make_cfg(tmp_path), verbose=False)

    assert out["X_train"].shape[1] == 6
    assert len(out["class_names"]) == 3
    assert out["feature_names"][0] == "feature_0"
    total = sum(len(out[f"y_{s}"]) for s in ("train", "val", "test"))
    assert total == 300


def test_tabular_normalisation_uses_training_rows_only(tmp_path):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(300, 4)), rng.integers(0, 2, size=300)
    out = user.load_tabular(X, y=y, cfg=make_cfg(tmp_path), verbose=False)
    assert np.allclose(out["X_train_sc"].mean(axis=0), 0, atol=1e-8)
    assert np.allclose(out["norm"]["mean"], out["X_train"].mean(axis=0))


def test_tabular_from_csv(tmp_path):
    import pandas as pd
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    df["label"] = rng.integers(0, 2, size=200)
    df.to_csv(tmp_path / "d.csv", index=False)

    out = user.load_tabular(tmp_path / "d.csv", cfg=make_cfg(tmp_path), verbose=False)
    assert out["feature_names"] == ["a", "b", "c"]


def test_tabular_rejects_a_3d_array(tmp_path):
    with pytest.raises(ValueError, match="set-valued"):
        user.load_tabular(np.zeros((10, 5, 3)), y=np.zeros(10), cfg=make_cfg(tmp_path))


def test_tabular_without_labels_says_so(tmp_path):
    with pytest.raises(ValueError, match="no labels"):
        user.load_tabular(np.zeros((10, 3)), cfg=make_cfg(tmp_path))


def test_set_valued_infers_the_mask_from_padding(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 12, 4))
    X[:, 8:, :] = 0.0                      # last four slots are padding
    y = rng.integers(0, 3, size=200)

    out = user.load_set_valued(X, y, cfg=make_cfg(tmp_path), verbose=False)
    assert out["mask_train"].sum(axis=1).max() == 8
    assert out["coords_train"].shape[-1] == 2
    assert out["norm"]["mean"].shape == (4,)


def test_set_valued_normalisation_ignores_padded_slots(tmp_path):
    rng = np.random.default_rng(4)
    X = rng.normal(loc=5.0, size=(200, 10, 3))
    X[:, 5:, :] = 0.0
    y = rng.integers(0, 2, size=200)

    out = user.load_set_valued(X, y, cfg=make_cfg(tmp_path), verbose=False)
    # padding would drag the mean toward 2.5 if it were included
    assert np.all(out["norm"]["mean"] > 4.5)


def test_benchmark_split_holds_the_test_block_out_of_the_pool(tmp_path):
    rng = np.random.default_rng(5)
    X, y = rng.normal(size=(1000, 4)), rng.integers(0, 3, size=1000)

    out = benchmarks._assemble(X, y, ["a", "b", "c", "d"], ["0", "1", "2"],
                               "synthetic", make_cfg(tmp_path), n_pool=400,
                               n_test=200, split_path=str(tmp_path / "s.npz"),
                               verbose=False)

    assert len(out["y_test"]) == 200
    assert len(out["y_train"]) + len(out["y_val"]) == 400
    assert not np.intersect1d(out["split"]["pool_rows"], out["split"]["test_rows"]).size
    assert np.allclose(out["norm"]["mean"], out["X_train"].mean(axis=0))


def test_benchmark_split_keeps_the_class_balance(tmp_path):
    rng = np.random.default_rng(6)
    y = np.repeat([0, 1, 2], [700, 200, 100])
    X = rng.normal(size=(1000, 3))

    out = benchmarks._assemble(X, y, ["a", "b", "c"], ["0", "1", "2"], "synthetic",
                               make_cfg(tmp_path), n_pool=400, n_test=200,
                               split_path=str(tmp_path / "s.npz"), verbose=False)

    shares = np.bincount(out["y_train"], minlength=3) / len(out["y_train"])
    assert np.allclose(shares, [0.7, 0.2, 0.1], atol=0.02)


def test_rare_levels_are_pooled_into_one_column():
    import pandas as pd
    col = ["a"] * 90 + ["b"] * 9 + ["c"]
    df = pd.DataFrame({"x": col})

    X, names, groups = benchmarks._encode_categoricals(df, ["x"], 0.05, verbose=False)
    assert names == ["x=a", "x=b", "x=other"]
    assert groups == ["x", "x", "x"]
    assert X.sum(axis=1).min() == 1.0


def test_missing_categorical_values_get_their_own_level():
    import pandas as pd
    df = pd.DataFrame({"x": ["a"] * 50 + ["b"] * 30 + [None] * 20})

    _, names, _ = benchmarks._encode_categoricals(df, ["x"], 0.01, verbose=False)
    assert "x=missing" in names
