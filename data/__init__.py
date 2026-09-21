from config import DEFAULT_DATASET

from . import benchmarks, hls4ml, user

TABULAR_LOADERS = {
    "covertype": benchmarks.load_covertype,
    "adult": benchmarks.load_adult,
}


def has_set_valued_view(dataset):
    return dataset == DEFAULT_DATASET


def load_tabular_view(dataset, cfg=None, n_rows=None, n_test_rows=None,
                      jet_source="openml", verbose=True):
    """The rows-and-columns view of one dataset, in the layout every model expects."""
    if dataset == DEFAULT_DATASET:
        if jet_source == "h5":
            return hls4ml.load_jet_level_h5(cfg=cfg, n_jets=n_rows,
                                            n_test_jets=n_test_rows, verbose=verbose)
        return hls4ml.load_jet_level(cfg=cfg, verbose=verbose)

    if dataset not in TABULAR_LOADERS:
        raise ValueError(f"no tabular loader for '{dataset}'")
    # pass only what was asked for; n_test_rows=None would override a loader's default
    kwargs = {"cfg": cfg, "verbose": verbose}
    if n_rows is not None:
        kwargs["n_rows"] = n_rows
    if n_test_rows is not None:
        kwargs["n_test_rows"] = n_test_rows
    return TABULAR_LOADERS[dataset](**kwargs)
