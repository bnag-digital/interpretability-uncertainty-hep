import matplotlib
import numpy as np
import pytest

import results
from plots import (attribution, concentration, interaction, mechanism, null,
                   performance, style, variance)

matplotlib.use("Agg")


def write_runs(tmp_path, model, level, seeds, n_rows, accuracies):
    rng = np.random.default_rng(0)
    for seed, acc in zip(seeds, accuracies):
        results.save(rng.uniform(0.8, 0.99, size=5), "performance", artifact_dir=tmp_path,
                     model=model, level=level, seed=seed, n_rows=n_rows,
                     accuracy=acc, epochs=20, n_features=16)


def test_accuracy_frame_reads_one_row_per_run(tmp_path):
    write_runs(tmp_path, "gnn", "n200000", [0, 1, 2], 150000, [0.80, 0.81, 0.82])
    frame = performance.accuracy_frame(artifact_dir=tmp_path)
    assert len(frame) == 3
    assert set(frame["model"]) == {"gnn"}
    assert frame["accuracy"].max() == pytest.approx(0.82)


def test_accuracy_frame_of_an_empty_directory_is_empty_not_an_error(tmp_path):
    frame = performance.accuracy_frame(artifact_dir=tmp_path)
    assert frame.empty
    assert "accuracy" in frame.columns


def test_spread_is_reported_in_accuracy_points(tmp_path):
    write_runs(tmp_path, "efn", "n200000", [0, 1], 150000, [0.7062, 0.7349])
    table = performance.spread_table(performance.accuracy_frame(artifact_dir=tmp_path))
    assert table["spread_pts"].iloc[0] == pytest.approx(2.87, abs=1e-2)


def test_size_curve_needs_more_than_one_size(tmp_path):
    write_runs(tmp_path, "gnn", "n200000", [0, 1], 150000, [0.80, 0.81])
    frame = performance.accuracy_frame(artifact_dir=tmp_path)
    fig = performance.plot_accuracy_vs_size(frame)
    assert fig.axes[0].get_lines() == []


def test_size_curve_draws_one_line_per_model(tmp_path):
    write_runs(tmp_path, "gnn", "n10000", [0, 1], 7500, [0.70, 0.72])
    write_runs(tmp_path, "gnn", "n200000", [0, 1], 150000, [0.80, 0.81])
    frame = performance.accuracy_frame(artifact_dir=tmp_path)
    fig = performance.plot_accuracy_vs_size(frame)
    assert fig.axes[0].get_xscale() == "log"
    assert len(fig.axes[0].get_lines()) >= 1


def test_accuracy_and_auc_panels_draw(tmp_path):
    write_runs(tmp_path, "bdt", "n200000", [0, 1], 150000, [0.7630, 0.7623])
    write_runs(tmp_path, "pfn", "n200000", [0, 1], 150000, [0.8165, 0.8162])
    frame = performance.accuracy_frame(artifact_dir=tmp_path)
    assert performance.plot_accuracy(frame).axes[0].get_ylabel() == "test accuracy"
    auc = performance.plot_per_class_auc(frame, artifact_dir=tmp_path)
    assert len(auc.axes[0].get_xticklabels()) == 5


def write_attributions(tmp_path, model, explainer, level, seeds, n_rows=40, n_feats=16,
                       scale=0.01):
    """Seed-to-seed jitter around one shared importance pattern."""
    key = sum(map(ord, explainer))
    base = np.random.default_rng(key).uniform(0.1, 1.0, size=n_feats)
    # the row factor depends on the explainer only, so scale=0 is identical across seeds
    rows = np.random.default_rng(key + 1).uniform(0.5, 1.5, size=(n_rows, n_feats))
    for seed in seeds:
        jitter = np.random.default_rng(seed).normal(scale=scale, size=n_feats)
        arr = np.abs(base + jitter) * rows
        results.save(arr, "attribution", artifact_dir=tmp_path, model=model,
                     explainer=explainer, level=level, seed=seed, n_rows=n_rows,
                     feature_space_id="testspace")


def test_collect_levels_reports_floor_and_ceiling(tmp_path):
    for explainer in ("shap", "lime"):
        write_attributions(tmp_path, "bdt", explainer, "ceiling", [0, 1, 2])
        write_attributions(tmp_path, "bdt", explainer, "floor", [0, 1, 2], scale=0.0)
    rows = variance.collect_levels({"bdt": ["shap", "lime"]}, artifact_dir=tmp_path)
    assert len(rows) == 2
    assert all(r["n_features"] == 16 for r in rows)
    # a floor written with no jitter is deterministic and has to be flagged as such
    assert all(r["deterministic"] for r in rows)


def test_collect_levels_survives_a_missing_floor(tmp_path):
    write_attributions(tmp_path, "gnn", "saliency", "ceiling", [0, 1, 2])
    rows = variance.collect_levels({"gnn": ["saliency"]}, artifact_dir=tmp_path)
    assert len(rows) == 1 and rows[0]["floor_tau"] is None


def test_verdicts_compare_cross_against_the_ceiling(tmp_path):
    for explainer in ("shap", "lime"):
        write_attributions(tmp_path, "dnn", explainer, "ceiling", [0, 1, 2])
    rows = variance.collect_verdicts({"dnn": ["shap", "lime"]}, artifact_dir=tmp_path)
    assert len(rows) == 1
    assert set(rows[0]) >= {"cross_tau", "ceiling_tau", "margin", "below_ceiling"}


def test_variance_panels_label_the_feature_count(tmp_path):
    for explainer in ("shap", "lime"):
        write_attributions(tmp_path, "bdt", explainer, "ceiling", [0, 1, 2])
    levels = variance.collect_levels({"bdt": ["shap", "lime"]}, artifact_dir=tmp_path)
    verdicts = variance.collect_verdicts({"bdt": ["shap", "lime"]}, artifact_dir=tmp_path)
    for fig in (variance.plot_levels(levels), variance.plot_verdicts(verdicts)):
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert all("(16f)" in text for text in labels)


def test_null_scatter_grows_as_features_fall():
    """The reason a tau on 3 features cannot be read against one on 16."""
    table = null.null_table(n_features=(3, 16), n_trials=200)
    sd = dict(zip(table["n_features"], table["tau_wtd_null_sd"]))
    assert sd[3] > sd[16]
    fig = null.plot_false_agreement(table)
    assert fig.axes[0].get_ylabel().startswith("chance")


def test_null_panel_places_observed_points(tmp_path):
    table = null.null_table(n_features=(3, 16), n_trials=100)
    observed = [{"label": "efn/ig", "model": "efn", "n_features": 3, "tau": 1.0},
                {"label": "gnn/ig", "model": "gnn", "n_features": 16, "tau": 0.94}]
    fig = null.plot_observed_against_null(table, observed)
    assert len(fig.axes[0].collections) >= 2


def test_interaction_square_needs_a_shared_feature_space(tmp_path):
    for model in ("bdt", "dnn"):
        for explainer in ("shap", "lime"):
            write_attributions(tmp_path, model, explainer, "ceiling", [0, 1, 2])
    result = interaction.collect_square(("bdt", "dnn", "shap", "lime"),
                                        artifact_dir=tmp_path)
    assert set(result["means"]) == {"cross_model_shap", "cross_model_lime",
                                    "cross_explainer_a", "cross_explainer_b"}
    assert interaction.plot_square(result).axes[0].get_xlabel() == "weighted Kendall tau"
    assert interaction.plot_paired_deltas(result) is not None


def test_interaction_refuses_two_different_feature_spaces(tmp_path):
    for explainer in ("saliency", "ig"):
        write_attributions(tmp_path, "gnn", explainer, "ceiling", [0, 1, 2], n_feats=16)
        write_attributions(tmp_path, "efn", explainer, "ceiling", [0, 1, 2], n_feats=3)
    with pytest.raises(Exception):
        interaction.collect_square(("gnn", "efn", "saliency", "ig"), artifact_dir=tmp_path)


def test_concentration_is_reported_as_a_fraction(tmp_path):
    write_attributions(tmp_path, "bdt", "shap", "ceiling", [0])
    rows = concentration.collect_concentration({"bdt": ["shap"]}, artifact_dir=tmp_path)
    assert len(rows) == 1
    assert 0.0 < rows[0]["eff_n_frac"] <= 1.0
    assert concentration.plot_concentration(rows) is not None
    assert concentration.plot_top1_share(rows) is not None


def test_attribution_panels_use_the_given_feature_names(tmp_path):
    write_attributions(tmp_path, "bdt", "shap", "ceiling", [0, 1])
    write_attributions(tmp_path, "bdt", "lime", "ceiling", [0, 1])
    a, _ = attribution.importance_of("bdt", "shap", artifact_dir=tmp_path)
    b, _ = attribution.importance_of("bdt", "lime", artifact_dir=tmp_path)
    names = [f"obs{i}" for i in range(len(a))]

    fig = attribution.plot_share_bars({"shap": a, "lime": b}, names)
    assert set(t.get_text() for t in fig.axes[0].get_xticklabels()) == set(names)
    assert attribution.plot_share_scatter(a, b, names, "shap", "lime") is not None


def test_mechanism_reads_predictions_written_on_the_explained_rows(tmp_path):
    write_attributions(tmp_path, "bdt", "shap", "ceiling", [0, 1, 2])
    rng = np.random.default_rng(0)
    for seed in (0, 1, 2):
        results.save(rng.integers(0, 5, size=40), "prediction", artifact_dir=tmp_path,
                     model="bdt", level="n200000", seed=seed, n_rows=40)
    rows = mechanism.collect_mechanism("bdt", ["shap"], artifact_dir=tmp_path)
    assert len(rows) == 1 and rows[0]["n_pairs"] == 3
    assert mechanism.plot_agreement_vs_tau(rows) is not None
    assert mechanism.plot_agreement_spread(rows) is not None


def test_mechanism_skips_an_explainer_with_no_predictions(tmp_path):
    write_attributions(tmp_path, "gnn", "ig", "ceiling", [0, 1, 2])
    assert mechanism.collect_mechanism("gnn", ["ig"], artifact_dir=tmp_path) == []


def test_summarise_keeps_predictions_apart_from_performance(tmp_path):
    """Both share a model and level, so grouping without kind merged them."""
    results.save(np.zeros(5), "performance", artifact_dir=tmp_path, model="bdt",
                 level="n200000", seed=0, n_features=16)
    results.save(np.zeros(500), "prediction", artifact_dir=tmp_path, model="bdt",
                 level="n200000", seed=0)
    table = results.summarise(tmp_path)
    assert len(table) == 2
    assert set(table["kind"]) == {"performance", "prediction"}


def test_models_are_drawn_in_a_fixed_order():
    assert style.order_models(["efn", "bdt", "gnn"]) == ["bdt", "gnn", "efn"]
    assert style.order_models(["zzz", "dnn"]) == ["dnn", "zzz"]


def test_save_writes_into_the_figure_directory(tmp_path):
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots()
    path = style.save(fig, "smoke", figure_dir=tmp_path)
    assert path.exists() and path.suffix == ".png"
