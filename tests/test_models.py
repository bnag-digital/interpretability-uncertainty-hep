import numpy as np
import pytest

torch = pytest.importorskip("torch")

from config import Config
from exceptions import InsufficientCapability
from models import efn, gnn, pfn
from models.dnn import (JetDNN, dnn_predict_proba_factory,
                        evaluate_dnn, load_dnn, train_dnn)
from models.pretrained import PretrainedModel, find_seed_checkpoints


CLASS_NAMES = ["a", "b", "c"]


def tiny_cfg(tmp_path, epochs=3):
    cfg = Config()
    cfg.checkpoint_dir = str(tmp_path)
    cfg.data.split_index_path = str(tmp_path / "split.npz")
    cfg.dnn.epochs, cfg.dnn.batch_size = epochs, 64
    cfg.gnn.epochs, cfg.gnn.batch_size, cfg.gnn.edgeconv_hidden = epochs, 32, [8, 8]
    cfg.pfn.epochs, cfg.pfn.batch_size, cfg.pfn.latent_dim = epochs, 32, 8
    cfg.efn.epochs, cfg.efn.batch_size, cfg.efn.latent_dim = epochs, 32, 8
    cfg.gnn.f_hidden = cfg.pfn.f_hidden = cfg.efn.f_hidden = 8
    cfg.pfn.phi_hidden = cfg.efn.phi_hidden = 8
    return cfg


def tabular_data(n=300, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = rng.integers(0, 3, size=n)
    return X, y


def particle_data(tmp_path, n=180, n_obj=10, seed=0):
    """Set-valued data in the layout the loaders return."""
    from data import user

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_obj, 5)).astype(np.float32)
    X[:, 7:, :] = 0.0
    y = rng.integers(0, 3, size=n)

    cfg = Config()
    cfg.data.split_index_path = str(tmp_path / "psplit.npz")
    data = user.load_set_valued(X, y, cfg=cfg, class_names=CLASS_NAMES, verbose=False,
                               feature_names=["j1_etarel", "j1_phirel", "j1_ptrel",
                                              "j1_deltaR", "j1_costhetarel"])
    data["coord_features"] = ["j1_etarel", "j1_phirel"]
    return data


def test_dnn_trains_and_checkpoint_carries_the_norm_stats(tmp_path):
    cfg = tiny_cfg(tmp_path)
    X, y = tabular_data()
    norm = {"mean": X[:200].mean(axis=0), "std": X[:200].std(axis=0)}

    model, device, info = train_dnn(X[:200], y[:200], X[200:], y[200:], cfg=cfg,
                                    n_classes=3, seed=1, norm=norm,
                                    feature_names=[f"f{i}" for i in range(6)],
                                    verbose=False)
    assert info["epochs"] >= 1 and info["model"] == "dnn"

    ckpt = torch.load(info["checkpoint"], weights_only=False)
    assert ckpt["norm"] is not None and ckpt["dropout"] == cfg.dnn.dropout
    assert len(ckpt["feature_names"]) == 6

    reloaded, _ = load_dnn(info["checkpoint"])
    assert isinstance(reloaded, JetDNN)


def test_dnn_predict_proba_matches_limes_interface(tmp_path):
    cfg = tiny_cfg(tmp_path)
    X, y = tabular_data()
    model, device, _ = train_dnn(X[:200], y[:200], X[200:], y[200:], cfg=cfg,
                                 n_classes=3, seed=1, verbose=False)
    probs = dnn_predict_proba_factory(model, device)(X[:20])
    assert probs.shape == (20, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_dnn_test_metrics_are_reported_per_class(tmp_path):
    cfg = tiny_cfg(tmp_path)
    X, y = tabular_data()
    model, device, _ = train_dnn(X[:200], y[:200], X[200:], y[200:], cfg=cfg,
                                 n_classes=3, seed=1, verbose=False)
    out = evaluate_dnn(model, device, X[200:], y[200:], CLASS_NAMES, verbose=False)
    assert set(out["per_class_auc"]) == set(CLASS_NAMES)
    assert 0.0 <= out["accuracy"] <= 1.0


def test_mc_dropout_actually_varies_the_output(tmp_path):
    cfg = tiny_cfg(tmp_path)
    X, y = tabular_data()
    model, device, _ = train_dnn(X[:200], y[:200], X[200:], y[200:], cfg=cfg,
                                 n_classes=3, seed=1, verbose=False)
    mc = dnn_predict_proba_factory(model, device, mc_dropout=True)
    assert not np.allclose(mc(X[:20]), mc(X[:20]))

    # the default path must stay deterministic
    plain = dnn_predict_proba_factory(model, device)
    assert np.allclose(plain(X[:20]), plain(X[:20]))


def test_knn_never_selects_a_padded_constituent():
    coords = torch.randn(4, 12, 2)
    mask = torch.zeros(4, 12)
    mask[:, :6] = 1.0
    idx = gnn.knn_indices(coords, mask, k=3)
    assert idx[:, :6, :].max().item() < 6


def test_knn_handles_fewer_constituents_than_k():
    coords = torch.randn(2, 4, 2)
    mask = torch.ones(2, 4)
    assert gnn.knn_indices(coords, mask, k=8).shape == (2, 4, 8)


def test_precomputed_graph_matches_the_one_built_in_the_forward_pass():
    coords, mask = torch.randn(6, 12, 2), torch.ones(6, 12)
    assert torch.equal(gnn.precompute_knn(coords, mask, k=4, batch_size=2),
                       gnn.knn_indices(coords, mask, k=4))


def test_gnn_trains_and_stores_norm_stats(tmp_path):
    cfg = tiny_cfg(tmp_path)
    data = particle_data(tmp_path)
    model, device, info = gnn.train_gnn(data, cfg=cfg, seed=2, verbose=False)
    assert info["model"] == "gnn" and info["epochs"] >= 1

    ckpt = torch.load(info["checkpoint"], weights_only=False)
    assert "norm" in ckpt and ckpt["k"] == cfg.gnn.k_neighbors

    loader = gnn.make_loader(data, "test", data["norm"], k=cfg.gnn.k_neighbors, batch_size=32)
    out = gnn.evaluate_gnn(*gnn.load_gnn(info["checkpoint"]), loader, CLASS_NAMES, verbose=False)
    assert 0.0 <= out["accuracy"] <= 1.0


def test_gnn_output_does_not_depend_on_constituent_order(tmp_path):
    cfg = tiny_cfg(tmp_path)
    model = gnn.ParticleGNN(n_node_features=5, edgeconv_hidden=[8, 8], n_classes=3,
                            k=4, f_hidden=8).eval()
    coords, x = torch.randn(2, 9, 2), torch.randn(2, 9, 5)
    mask = torch.ones(2, 9)
    perm = torch.randperm(9)
    with torch.no_grad():
        a = model(coords, x, mask)
        b = model(coords[:, perm], x[:, perm], mask[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_pfn_trains_and_is_permutation_invariant(tmp_path):
    cfg = tiny_cfg(tmp_path)
    data = particle_data(tmp_path)
    model, device, info = pfn.train_pfn(data, cfg=cfg, seed=3, verbose=False)
    assert info["model"] == "pfn"

    x, mask = torch.randn(2, 9, 5), torch.ones(2, 9)
    perm = torch.randperm(9)
    net = pfn.PFN(5, 8, 3, 8, 8).eval()
    with torch.no_grad():
        assert torch.allclose(net(x, mask), net(x[:, perm], mask[:, perm]), atol=1e-5)


def test_pfn_ignores_padded_constituents(tmp_path):
    net = pfn.PFN(5, 8, 3, 8, 8).eval()
    x, mask = torch.randn(2, 9, 5), torch.ones(2, 9)
    mask[:, 6:] = 0.0
    x_changed = x.clone()
    x_changed[:, 6:] = 99.0
    with torch.no_grad():
        assert torch.allclose(net(x, mask), net(x_changed, mask), atol=1e-5)


def test_efn_trains_and_phi_sees_angles_only(tmp_path):
    cfg = tiny_cfg(tmp_path)
    data = particle_data(tmp_path)
    model, device, info = efn.train_efn(data, cfg=cfg, seed=4, verbose=False)
    assert info["model"] == "efn"

    ckpt = torch.load(info["checkpoint"], weights_only=False)
    assert ckpt["angular_features"] == ["j1_etarel", "j1_phirel"]
    # two angular inputs, whatever the number of node features
    assert model.phi[0].in_features == 2


@pytest.mark.parametrize("name", ["pfn", "efn"])
def test_set_models_stop_early_instead_of_running_the_full_cap(tmp_path, name):
    """Without this the seed grids run every epoch of the cap, as the GNN never does."""
    cfg = tiny_cfg(tmp_path)
    data = particle_data(tmp_path)
    for sub in (cfg.pfn, cfg.efn):
        sub.epochs, sub.early_stop_patience = 40, 1

    train = pfn.train_pfn if name == "pfn" else efn.train_efn
    _, _, info = train(data, cfg=cfg, seed=7, verbose=False)
    assert info["epochs_run"] < 40
    assert info["epochs"] <= info["epochs_run"]


def test_efn_output_scales_linearly_in_the_energy_weight():
    net = efn.EFN(2, 8, 3, 8, 8).eval()
    x_ang, mask = torch.randn(2, 9, 2), torch.ones(2, 9)
    z = torch.rand(2, 9)
    with torch.no_grad():
        pooled_a = (net.phi(x_ang) * (z * mask).unsqueeze(-1)).sum(1)
        pooled_b = (net.phi(x_ang) * (2 * z * mask).unsqueeze(-1)).sum(1)
    assert torch.allclose(2 * pooled_a, pooled_b, atol=1e-5)


def test_weights_only_cannot_give_a_ceiling(tmp_path):
    wrapped = PretrainedModel(JetDNN(6, 3), name="user_model")
    caps = wrapped.capabilities()
    assert caps["floor"] and caps["cross"]
    assert not caps["ceiling"] and not caps["verdict"]
    assert "only one checkpoint" in caps["reason"]
    with pytest.raises(InsufficientCapability):
        wrapped.require_ceiling()


def test_a_training_entrypoint_unlocks_the_ceiling():
    wrapped = PretrainedModel(JetDNN(6, 3), train_fn=lambda seed: None)
    assert wrapped.capabilities()["ceiling"]
    assert wrapped.require_ceiling()


def test_several_seed_checkpoints_unlock_the_ceiling(tmp_path):
    for seed in range(3):
        torch.save({"model_state": {}}, tmp_path / f"m_seed{seed}.pt")
    found = find_seed_checkpoints(tmp_path)
    assert len(found) == 3
    assert PretrainedModel(JetDNN(6, 3), checkpoints=found).capabilities()["ceiling"]


def test_pretrained_predict_proba_returns_probabilities():
    wrapped = PretrainedModel(JetDNN(6, 3))
    probs = wrapped.predict_proba_factory()(np.random.default_rng(0).normal(size=(5, 6)))
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
