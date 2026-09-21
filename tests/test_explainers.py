import numpy as np
import pytest

torch = pytest.importorskip("torch")

from config import SUMMARY_NAMES, Config
from exceptions import LowSurrogateFidelity
from explainers import setvalued, surrogate
from models import efn, gnn, pfn

CLASS_NAMES = ["a", "b", "c"]
NODE_FEATURES = ["j1_etarel", "j1_phirel", "j1_ptrel", "j1_deltaR", "j1_costhetarel"]


def particle_data(tmp_path, n=60, n_obj=10, seed=0):
    """Set-valued data in the layout the loaders return, 7 of 10 slots filled."""
    from data import user

    rng = np.random.default_rng(seed)
    X = np.abs(rng.normal(size=(n, n_obj, 5))).astype(np.float32)
    X[:, 7:, :] = 0.0
    y = rng.integers(0, 3, size=n)

    cfg = Config()
    cfg.data.split_index_path = str(tmp_path / "psplit.npz")
    data = user.load_set_valued(X, y, cfg=cfg, class_names=CLASS_NAMES, verbose=False,
                                feature_names=list(NODE_FEATURES))
    data["coord_features"] = ["j1_etarel", "j1_phirel"]
    return data


def tiny_models():
    g = gnn.ParticleGNN(n_node_features=5, edgeconv_hidden=[8, 8], n_classes=3,
                        k=4, f_hidden=8).eval()
    p = pfn.PFN(5, 8, 3, 8, 8).eval()
    e = efn.EFN(2, 8, 3, 8, 8).eval()
    return {"gnn": g, "pfn": p, "efn": e}


@pytest.fixture
def setup(tmp_path):
    data = particle_data(tmp_path)
    return data, tiny_models(), torch.device("cpu")


def test_prepare_inputs_gives_each_model_its_own_feature_axis(setup):
    data, _, _ = setup
    for kind in ("gnn", "pfn"):
        inputs = setvalued.prepare_inputs(kind, data, "test")
        assert inputs["feature_names"] == NODE_FEATURES
        assert inputs["x"].shape[-1] == 5

    # Phi sees two angles, and the energy weight is carried as the last column
    inputs = setvalued.prepare_inputs("efn", data, "test")
    assert inputs["feature_names"] == ["j1_etarel", "j1_phirel", "j1_ptrel"]
    assert inputs["x"].shape[-1] == 3


def test_prepare_inputs_rejects_an_unknown_model(setup):
    data, _, _ = setup
    with pytest.raises(ValueError):
        setvalued.prepare_inputs("transformer", data, "test")


def test_prepare_inputs_selects_rows(setup):
    data, _, _ = setup
    inputs = setvalued.prepare_inputs("pfn", data, "test", rows=[0, 3, 5])
    assert inputs["x"].shape[0] == 3 and inputs["mask"].shape[0] == 3


@pytest.mark.parametrize("kind", ["gnn", "pfn", "efn"])
def test_saliency_and_ig_return_one_value_per_feature(setup, kind):
    data, models, device = setup
    cfg = Config()
    cfg.explainer.ig_steps = 3
    inputs = setvalued.prepare_inputs(kind, data, "test", rows=range(8))
    n_features = inputs["x"].shape[-1]

    sal = setvalued.saliency(models[kind], kind, inputs, device, batch_size=4,
                             k=4, verbose=False)
    ig = setvalued.integrated_gradients(models[kind], kind, inputs, device, cfg=cfg,
                                        batch_size=4, k=4, verbose=False)
    assert sal.shape == (8, n_features) and ig.shape == (8, n_features)
    assert np.all(np.isfinite(sal)) and np.all(np.isfinite(ig))
    # both are magnitudes averaged over real constituents
    assert np.all(sal >= 0) and np.all(ig >= 0)


def test_padded_constituents_do_not_enter_the_saliency_average(setup):
    """Changing a padded slot must not move the result."""
    data, models, device = setup
    inputs = setvalued.prepare_inputs("pfn", data, "test", rows=range(8))
    a = setvalued.saliency(models["pfn"], "pfn", inputs, device, batch_size=4, verbose=False)

    inputs["x"][:, 7:, :] = 99.0
    b = setvalued.saliency(models["pfn"], "pfn", inputs, device, batch_size=4, verbose=False)
    assert np.allclose(a, b, atol=1e-5)


def test_smoothgrad_is_on_the_same_axis_as_the_gradients(setup):
    data, models, device = setup
    cfg = Config()
    cfg.explainer.smoothgrad_samples = 4
    inputs = setvalued.prepare_inputs("pfn", data, "test", rows=range(8))
    out = setvalued.smoothgrad(models["pfn"], "pfn", inputs, device, cfg=cfg,
                               batch_size=4, verbose=False)
    assert out.shape == (8, inputs["x"].shape[-1])
    assert np.all(np.isfinite(out)) and np.all(out >= 0)


def test_smoothgrad_reproduces_from_the_seed_and_moves_between_seeds(setup):
    """The whole point of adding it: a set-valued explainer with a real floor."""
    data, models, device = setup
    cfg = Config()
    cfg.explainer.smoothgrad_samples = 4
    inputs = setvalued.prepare_inputs("pfn", data, "test", rows=range(8))

    def run(seed):
        return setvalued.smoothgrad(models["pfn"], "pfn", inputs, device, cfg=cfg,
                                    seed=seed, batch_size=4, verbose=False)

    assert np.array_equal(run(0), run(0))
    assert not np.allclose(run(0), run(1))


def test_smoothgrad_does_not_noise_padded_constituents(setup):
    data, models, device = setup
    cfg = Config()
    cfg.explainer.smoothgrad_samples = 4
    inputs = setvalued.prepare_inputs("pfn", data, "test", rows=range(8))
    a = setvalued.smoothgrad(models["pfn"], "pfn", inputs, device, cfg=cfg,
                             batch_size=4, verbose=False)

    inputs["x"][:, 7:, :] = 99.0
    b = setvalued.smoothgrad(models["pfn"], "pfn", inputs, device, cfg=cfg,
                             batch_size=4, verbose=False)
    assert np.allclose(a, b, atol=1e-5)


def test_feature_occlusion_is_on_the_same_axis_as_the_gradients(setup):
    data, models, device = setup
    inputs = setvalued.prepare_inputs("pfn", data, "test", rows=range(8))
    out = setvalued.feature_occlusion(models["pfn"], "pfn", inputs, device,
                                      batch_size=4, verbose=False)
    assert out.shape == (8, inputs["x"].shape[-1])
    assert np.all(np.abs(out) <= 1.0)


def test_constituent_occlusion_orders_by_energy_weight(setup):
    data, models, device = setup
    inputs = setvalued.prepare_inputs("pfn", data, "test", rows=range(8))
    out = setvalued.constituent_occlusion(models["pfn"], "pfn", inputs, device,
                                          n_constituents=5, batch_size=4, verbose=False)
    assert out["drop"].shape == (8, 5)
    assert np.all(np.diff(out["weight"], axis=1) <= 1e-6)
    # only real constituents should be picked, since padding has zero weight
    assert out["index"].max() < 7


def test_predict_logits_matches_a_direct_forward_pass(setup):
    data, models, device = setup
    inputs = setvalued.prepare_inputs("pfn", data, "test", rows=range(8))
    got = setvalued.predict_logits(models["pfn"], "pfn", inputs, device, batch_size=4)
    with torch.no_grad():
        want = models["pfn"](torch.tensor(inputs["x"]), torch.tensor(inputs["mask"])).numpy()
    assert np.allclose(got, want, atol=1e-5)


def test_summary_counts_real_constituents_only():
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(size=(5, 10, 5))).astype(np.float32)
    mask = np.zeros((5, 10), dtype=np.float32)
    mask[:, :6] = 1.0
    S = surrogate.compute_summary(X, mask, NODE_FEATURES)
    assert S.shape == (5, len(SUMMARY_NAMES))
    assert np.allclose(S[:, SUMMARY_NAMES.index("n_particles")], 6.0)


def test_girth_is_the_ptrel_weighted_delta_r():
    """One jet, two constituents, worked out by hand."""
    X = np.zeros((1, 2, 5), dtype=np.float32)
    X[0, :, NODE_FEATURES.index("j1_ptrel")] = [0.75, 0.25]
    X[0, :, NODE_FEATURES.index("j1_deltaR")] = [0.1, 0.5]
    mask = np.ones((1, 2), dtype=np.float32)

    S = surrogate.compute_summary(X, mask, NODE_FEATURES)
    expected = (0.75 * 0.1 + 0.25 * 0.5) / 1.0
    assert np.isclose(S[0, SUMMARY_NAMES.index("ptrel_weighted_deltaR")], expected, atol=1e-6)


def test_compute_summary_refuses_a_different_column_list():
    X = np.zeros((2, 3, 5), dtype=np.float32)
    mask = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        surrogate.compute_summary(X, mask, NODE_FEATURES, summary_names=["n_particles"])


def surrogate_fixture(n=400, seed=0):
    """Logits that really are a function of the summary features."""
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, len(SUMMARY_NAMES)))
    W = rng.normal(size=(len(SUMMARY_NAMES), 3))
    return S, S @ W


def test_a_surrogate_that_fits_reports_high_fidelity_and_passes_the_gate():
    S, Z = surrogate_fixture()
    predict_proba, fidelity = surrogate.fit_summary_surrogate(S, Z, CLASS_NAMES, seed=0)

    assert set(fidelity["r2_per_class"]) == set(CLASS_NAMES)
    assert fidelity["min_r2"] > 0.5
    ok, reason = surrogate.check_fidelity(fidelity, min_r2=0.5, min_argmax=0.5)
    assert ok and reason == ""

    probs = predict_proba(S[:10])
    assert probs.shape == (10, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_a_surrogate_that_does_not_track_the_model_is_refused():
    """Pure noise as the target: nothing to fit, so nothing may be reported."""
    rng = np.random.default_rng(1)
    S = rng.normal(size=(300, len(SUMMARY_NAMES)))
    Z = rng.normal(size=(300, 3))
    _, fidelity = surrogate.fit_summary_surrogate(S, Z, CLASS_NAMES, seed=0)

    assert fidelity["min_r2"] < 0.7
    with pytest.raises(LowSurrogateFidelity):
        surrogate.check_fidelity(fidelity)
    ok, reason = surrogate.check_fidelity(fidelity, raise_on_fail=False)
    assert not ok and "below threshold" in reason


def test_refitting_the_surrogate_under_a_new_seed_moves_the_fit():
    """The surrogate refit is its own level of variation, so the seed must matter."""
    S, Z = surrogate_fixture()
    a, _ = surrogate.fit_summary_surrogate(S, Z, CLASS_NAMES, seed=0)
    b, _ = surrogate.fit_summary_surrogate(S, Z, CLASS_NAMES, seed=1)
    assert not np.allclose(a(S[:20]), b(S[:20]))


def test_fidelity_row_carries_the_numbers_into_the_sidecar():
    S, Z = surrogate_fixture()
    _, fidelity = surrogate.fit_summary_surrogate(S, Z, CLASS_NAMES, seed=3)
    row = surrogate.fidelity_row(fidelity, model="gnn")
    assert row["surrogate_seed"] == 3
    assert row["n_features"] == len(SUMMARY_NAMES)
    assert all(f"r2_{cls}" in row for cls in CLASS_NAMES)


def test_lime_returns_the_shared_shape_convention():
    pytest.importorskip("lime")
    from explainers import tabular

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    names = [f"f{i}" for i in range(4)]

    def predict_proba(arr):
        z = np.asarray(arr) @ rng.normal(size=(4, 3))
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    explainer = tabular.build_lime_explainer(X, names, CLASS_NAMES, seed=0)
    out = tabular.explain_lime(explainer, X[:4], predict_proba, 4, 3, verbose=False)
    assert out.shape == (4, 4, 3)


def tiny_bdt():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = rng.integers(0, 3, size=200)
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3, objective="multi:softprob",
                              num_class=3, n_jobs=1)
    model.fit(X, y)
    return model, X, [f"f{i}" for i in range(4)]


def test_bdt_treeshap_returns_rows_features_classes():
    pytest.importorskip("shap")
    from explainers import tabular

    model, X, names = tiny_bdt()
    cfg = Config()
    cfg.explainer.output_space = "logodds"
    out = tabular.explain_bdt_shap(model, X[:10], names, 3, cfg=cfg, seed=0, verbose=False)
    assert out.shape == (10, 4, 3)


def test_multiclass_treeshap_refuses_probability_space_instead_of_guessing():
    """Softmax couples the class outputs, so there is no per-class link to invert."""
    pytest.importorskip("shap")
    from explainers import tabular

    model, X, names = tiny_bdt()
    cfg = Config()
    cfg.explainer.output_space = "probability"
    with pytest.raises(NotImplementedError, match="method='kernel'"):
        tabular.explain_bdt_shap(model, X[:5], names, 3, cfg=cfg, seed=0,
                                 method="tree", verbose=False)


def test_auto_picks_the_only_method_that_works_in_each_space():
    pytest.importorskip("shap")
    from explainers import tabular

    model, X, names = tiny_bdt()
    cfg = Config()
    cfg.explainer.background_size = 10
    cfg.explainer.shap_nsamples = 32

    cfg.explainer.output_space = "probability"
    assert tabular.explain_bdt_shap(model, X[:3], names, 3, cfg=cfg, seed=0,
                                    verbose=False).shape == (3, 4, 3)
    cfg.explainer.output_space = "logodds"
    assert tabular.explain_bdt_shap(model, X[:3], names, 3, cfg=cfg, seed=0,
                                    verbose=False).shape == (3, 4, 3)


def test_kernel_shap_gives_probability_space_values_for_the_bdt():
    pytest.importorskip("shap")
    from explainers import tabular

    model, X, names = tiny_bdt()
    cfg = Config()
    cfg.explainer.output_space = "probability"
    cfg.explainer.background_size = 10
    cfg.explainer.shap_nsamples = 32
    out = tabular.explain_bdt_shap(model, X[:3], names, 3, cfg=cfg, seed=0,
                                   method="kernel", verbose=False)
    assert out.shape == (3, 4, 3)


def test_kernel_shap_refuses_to_be_called_in_log_odds_space():
    pytest.importorskip("shap")
    from explainers import tabular

    model, X, names = tiny_bdt()
    cfg = Config()
    cfg.explainer.output_space = "logodds"
    with pytest.raises(ValueError, match="predict_proba"):
        tabular.explain_bdt_shap(model, X[:3], names, 3, cfg=cfg, method="kernel",
                                 verbose=False)


def tiny_mlp(n_features=4, n_classes=3, seed=0):
    """A small differentiable model in the shape the DNN explainers expect."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(n_features, 8), torch.nn.ReLU(),
                                torch.nn.Linear(8, n_classes)).eval()
    rng = np.random.default_rng(seed)
    return model, rng.normal(size=(12, n_features)).astype(np.float32)


@pytest.mark.parametrize("name", ["saliency", "ig", "smoothgrad"])
def test_dnn_gradient_explainers_return_rows_features_classes(name):
    from explainers import tabular

    model, X = tiny_mlp()
    fn = {"saliency": tabular.explain_dnn_saliency, "ig": tabular.explain_dnn_ig,
          "smoothgrad": tabular.explain_dnn_smoothgrad}[name]
    out = fn(model, torch.device("cpu"), X, [f"f{i}" for i in range(4)], 3,
             cfg=Config(), seed=0, verbose=False)
    assert out.shape == (12, 4, 3)
    assert np.all(np.isfinite(out))


def test_dnn_saliency_and_ig_are_deterministic():
    from explainers import tabular

    model, X = tiny_mlp()
    names, cfg = [f"f{i}" for i in range(4)], Config()
    for fn in (tabular.explain_dnn_saliency, tabular.explain_dnn_ig):
        a = fn(model, torch.device("cpu"), X, names, 3, cfg=cfg, seed=0, verbose=False)
        b = fn(model, torch.device("cpu"), X, names, 3, cfg=cfg, seed=7, verbose=False)
        assert np.array_equal(a, b)


def test_dnn_smoothgrad_reproduces_from_the_seed_and_moves_between_seeds():
    from explainers import tabular

    model, X = tiny_mlp()
    names, cfg = [f"f{i}" for i in range(4)], Config()
    a = tabular.explain_dnn_smoothgrad(model, torch.device("cpu"), X, names, 3,
                                       cfg=cfg, seed=0, verbose=False)
    same = tabular.explain_dnn_smoothgrad(model, torch.device("cpu"), X, names, 3,
                                          cfg=cfg, seed=0, verbose=False)
    other = tabular.explain_dnn_smoothgrad(model, torch.device("cpu"), X, names, 3,
                                           cfg=cfg, seed=1, verbose=False)
    assert np.array_equal(a, same)
    assert not np.array_equal(a, other)


def test_dnn_saliency_matches_an_autograd_gradient_of_the_probability():
    from explainers import tabular

    model, X = tiny_mlp()
    out = tabular.explain_dnn_saliency(model, torch.device("cpu"), X,
                                       [f"f{i}" for i in range(4)], 3,
                                       cfg=Config(), seed=0, verbose=False)
    x = torch.tensor(X[:1], requires_grad=True)
    torch.softmax(model(x), dim=1)[0, 2].backward()
    assert np.allclose(out[0, :, 2], x.grad.numpy()[0], atol=1e-6)


def test_ig_baseline_row_gets_almost_no_attribution():
    from explainers import tabular

    model, _ = tiny_mlp()
    # zero in standardised space is the baseline, so the path has no length
    zeros = np.zeros((3, 4), dtype=np.float32)
    out = tabular.explain_dnn_ig(model, torch.device("cpu"), zeros,
                                 [f"f{i}" for i in range(4)], 3, cfg=Config(),
                                 seed=0, verbose=False)
    assert np.allclose(out, 0.0, atol=1e-8)
