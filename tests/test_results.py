import numpy as np
import pytest

import results
from config import Config
from exceptions import InsufficientCapability


def write_seed_stack(tmp_path, model, explainer, seeds, level="ceiling", n_rows=50, n_feats=16):
    rng = np.random.default_rng(0)
    for s in seeds:
        results.save(rng.normal(size=(n_rows, n_feats)), "attribution", artifact_dir=tmp_path,
                     model=model, explainer=explainer, level=level, seed=s,
                     n_rows=n_rows, epochs=40, output_space="probability")


def test_save_writes_payload_and_sidecar(tmp_path):
    path = results.save(np.zeros((5, 16)), "attribution", artifact_dir=tmp_path,
                        model="bdt", explainer="shap", level="ceiling", seed=0, n_rows=5)
    assert path.exists()
    assert path.with_suffix(".json").exists()
    arr, meta = results.load(path.stem, artifact_dir=tmp_path)
    assert arr.shape == (5, 16)
    assert meta["model"] == "bdt" and meta["n_features"] == 16


def test_scan_lists_what_has_been_run(tmp_path):
    write_seed_stack(tmp_path, "dnn", "shap", seeds=[0, 1, 2])
    df = results.scan(tmp_path)
    assert len(df) == 3
    assert results.seeds_available(df, "dnn", "shap", "ceiling") == [0, 1, 2]
    assert results.seeds_available(df, "gnn") == []


def test_scan_of_an_empty_directory_is_empty_not_an_error(tmp_path):
    assert results.scan(tmp_path).empty


def test_stack_seeds_returns_one_row_per_seed(tmp_path):
    write_seed_stack(tmp_path, "bdt", "lime", seeds=[0, 1, 2, 3])
    stack, seeds = results.stack_seeds("bdt", "lime", artifact_dir=tmp_path)
    assert stack.shape[0] == 4 and seeds == [0, 1, 2, 3]


def test_one_checkpoint_cannot_give_a_ceiling(tmp_path):
    write_seed_stack(tmp_path, "user_model", "shap", seeds=[0])
    with pytest.raises(InsufficientCapability):
        results.stack_seeds("user_model", "shap", artifact_dir=tmp_path)


def test_can_draw_reports_why_a_figure_is_skipped(tmp_path):
    write_seed_stack(tmp_path, "bdt", "shap", seeds=[0, 1])
    df = results.scan(tmp_path)
    ok, _ = results.can_draw(df, [{"model": "bdt", "explainer": "shap", "min_seeds": 2}])
    assert ok
    ok, reason = results.can_draw(df, [{"model": "bdt", "explainer": "shap", "min_seeds": 8}])
    assert not ok and "needs 8" in reason
    ok, reason = results.can_draw(df, [{"model": "gnn", "explainer": "shap"}])
    assert not ok and "nothing matching" in reason


def test_summarise_counts_seeds_per_model(tmp_path):
    write_seed_stack(tmp_path, "bdt", "shap", seeds=[0, 1, 2])
    write_seed_stack(tmp_path, "dnn", "shap", seeds=[0, 1])
    summary = results.summarise(tmp_path).set_index("model")
    assert summary.loc["bdt", "n_seeds"] == 3
    assert summary.loc["dnn", "n_seeds"] == 2


def test_config_dataclasses_instantiate():
    cfg = Config()
    assert len(cfg.data.jet_features) == 16
    assert cfg.explainer.output_space == "probability"
    assert len(cfg.protocol.model_seeds) == 8
