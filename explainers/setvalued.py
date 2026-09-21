import numpy as np
import torch

from config import CONFIG
from data.hls4ml import normalise
from models.gnn import knn_indices

KINDS = ("gnn", "pfn", "efn")


def prepare_inputs(kind, data, split="test", cfg=None, rows=None):
    """Assemble what one model differentiates with respect to."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    full_cfg = cfg or CONFIG
    norm = data["norm"]

    mask = np.asarray(data[f"mask_{split}"])
    coords = np.asarray(data[f"coords_{split}"])
    y = np.asarray(data[f"y_{split}"])

    if kind == "efn":
        from models.efn import _angular_and_energy

        x_ang, z, mask = _angular_and_energy(data, split, full_cfg.efn, norm)
        x = np.concatenate([x_ang, z[..., None]], axis=-1)
        names = list(full_cfg.efn.angular_features) + [full_cfg.efn.energy_feature]
        weight = z
    else:
        x = normalise(data[f"X_{split}"], mask, norm)
        names = list(data["node_features"])
        weight = np.asarray(data[f"X_{split}"])[:, :, names.index(full_cfg.efn.energy_feature)] * mask

    if rows is not None:
        rows = np.asarray(rows)
        x, mask, coords, y = x[rows], mask[rows], coords[rows], y[rows]
        weight = weight[rows]

    return {"x": np.asarray(x, dtype=np.float32),
            "mask": np.asarray(mask, dtype=np.float32),
            "coords": np.asarray(coords, dtype=np.float32),
            "weight": np.asarray(weight, dtype=np.float32),
            "y": y, "feature_names": names, "kind": kind}


def _logits(model, kind, x, mask, coords=None, knn_idx=None):
    """One forward pass, whatever signature the model happens to have."""
    if kind == "gnn":
        return model(coords, x, mask, knn_idx)
    if kind == "pfn":
        return model(x, mask)
    return model(x[..., :-1], x[..., -1], mask)


def _batches(n, size):
    for start in range(0, n, size):
        yield start, min(start + size, n)


def _to_device(inputs, sl, device):
    x = torch.tensor(inputs["x"][sl]).to(device)
    mask = torch.tensor(inputs["mask"][sl]).to(device)
    coords = torch.tensor(inputs["coords"][sl]).to(device)
    weight = torch.tensor(inputs["weight"][sl]).to(device)
    return x, mask, coords, weight


def _per_jet_mean(values, mask):
    """Collapse (batch, constituents, features) to (batch, features)."""
    m = mask[:, :, None]
    n_real = mask.sum(axis=1).clip(1)
    return (np.abs(values) * m).sum(axis=1) / n_real[:, None]


def predict_logits(model, kind, inputs, device, batch_size=256, k=None):
    """Logits for every row, in the same order as inputs."""
    model.eval()
    out = []
    with torch.no_grad():
        for start, end in _batches(len(inputs["x"]), batch_size):
            x, mask, coords, weight = _to_device(inputs, slice(start, end), device)
            knn_idx = knn_indices(coords, mask, k) if (kind == "gnn" and k) else None
            out.append(_logits(model, kind, x, mask, coords, knn_idx).cpu().numpy())
    return np.concatenate(out, axis=0)


def saliency(model, kind, inputs, device, batch_size=64, k=None, verbose=True):
    """Gradient of the predicted-class logit with respect to the node features."""
    model.eval()
    n = len(inputs["x"])
    out = np.zeros((n, inputs["x"].shape[-1]), dtype=np.float64)

    for start, end in _batches(n, batch_size):
        x, mask, coords, weight = _to_device(inputs, slice(start, end), device)
        knn_idx = knn_indices(coords, mask, k) if (kind == "gnn" and k) else None
        x.requires_grad_(True)
        logits = _logits(model, kind, x, mask, coords, knn_idx)
        pred = logits.argmax(dim=1)
        logits[torch.arange(len(pred)), pred].sum().backward()

        out[start:end] = _per_jet_mean(x.grad.detach().cpu().numpy(),
                                       mask.detach().cpu().numpy())
        if verbose:
            print(f"  saliency {end}/{n} jets", flush=True)
    return out


def smoothgrad(model, kind, inputs, device, cfg=None, seed=0, batch_size=32,
               k=None, verbose=True):
    """Saliency averaged over Gaussian-noised copies of each jet."""
    cfg = (cfg or CONFIG).explainer
    model.eval()
    n = len(inputs["x"])
    out = np.zeros((n, inputs["x"].shape[-1]), dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)

    for start, end in _batches(n, batch_size):
        x, mask, coords, weight = _to_device(inputs, slice(start, end), device)
        knn_idx = knn_indices(coords, mask, k) if (kind == "gnn" and k) else None
        real = mask.bool()

        # fix the class from the clean pass, or noise flips it near a boundary
        with torch.no_grad():
            pred = _logits(model, kind, x, mask, coords, knn_idx).argmax(dim=1)
        rows = torch.arange(len(pred), device=x.device)

        values = x[real]
        sigma = cfg.smoothgrad_noise * (values.max(dim=0).values - values.min(dim=0).values)
        sigma = sigma.clamp(min=1e-6)

        grads = torch.zeros_like(x)
        for _ in range(cfg.smoothgrad_samples):
            noise = torch.randn(x.shape, generator=generator).to(x.device) * sigma
            x_hat = (x + noise * mask[:, :, None]).requires_grad_(True)
            logits = _logits(model, kind, x_hat, mask, coords, knn_idx)
            logits[rows, pred].sum().backward()
            grads += x_hat.grad.detach()

        out[start:end] = _per_jet_mean((grads / cfg.smoothgrad_samples).cpu().numpy(),
                                       mask.detach().cpu().numpy())
        if verbose:
            print(f"  smoothgrad {end}/{n} jets", flush=True)
    return out


def integrated_gradients(model, kind, inputs, device, cfg=None, batch_size=32,
                         k=None, verbose=True):
    """Gradients accumulated along a straight path from an all-zero baseline."""
    cfg = (cfg or CONFIG).explainer
    model.eval()
    n = len(inputs["x"])
    out = np.zeros((n, inputs["x"].shape[-1]), dtype=np.float64)

    for start, end in _batches(n, batch_size):
        x, mask, coords, weight = _to_device(inputs, slice(start, end), device)
        knn_idx = knn_indices(coords, mask, k) if (kind == "gnn" and k) else None
        baseline = torch.zeros_like(x)
        grads = torch.zeros_like(x)

        for step in range(cfg.ig_steps):
            alpha = (step + 0.5) / cfg.ig_steps
            x_hat = (baseline + alpha * (x - baseline)).requires_grad_(True)
            logits = _logits(model, kind, x_hat, mask, coords, knn_idx)
            pred = logits.argmax(dim=1)
            logits[torch.arange(len(pred)), pred].sum().backward()
            grads += x_hat.grad.detach()

        ig = ((x - baseline) * grads / cfg.ig_steps).detach().cpu().numpy()
        out[start:end] = _per_jet_mean(ig, mask.detach().cpu().numpy())
        if verbose:
            print(f"  IG {end}/{n} jets", flush=True)
    return out


def feature_occlusion(model, kind, inputs, device, baseline=0.0, batch_size=128,
                      k=None, verbose=True):
    """Drop in the predicted class probability when one feature is blanked out."""
    model.eval()
    n, n_features = len(inputs["x"]), inputs["x"].shape[-1]
    out = np.zeros((n, n_features), dtype=np.float64)

    with torch.no_grad():
        for start, end in _batches(n, batch_size):
            x, mask, coords, weight = _to_device(inputs, slice(start, end), device)
            knn_idx = knn_indices(coords, mask, k) if (kind == "gnn" and k) else None

            probs = torch.softmax(_logits(model, kind, x, mask, coords, knn_idx), dim=1)
            pred = probs.argmax(dim=1)
            rows = torch.arange(len(pred), device=probs.device)
            p_full = probs[rows, pred]

            for j in range(n_features):
                x_occ = x.clone()
                x_occ[:, :, j] = baseline * mask
                p_occ = torch.softmax(
                    _logits(model, kind, x_occ, mask, coords, knn_idx), dim=1)[rows, pred]
                out[start:end, j] = (p_full - p_occ).cpu().numpy()
            if verbose:
                print(f"  feature occlusion {end}/{n} jets", flush=True)
    return out


def constituent_occlusion(model, kind, inputs, device, n_constituents=20,
                          batch_size=64, k=None, verbose=True):
    """Drop in the predicted class probability when one constituent is removed."""
    model.eval()
    n = len(inputs["x"])
    drops = np.zeros((n, n_constituents), dtype=np.float64)
    weights = np.zeros((n, n_constituents), dtype=np.float64)
    picked = np.zeros((n, n_constituents), dtype=np.int64)

    with torch.no_grad():
        for start, end in _batches(n, batch_size):
            x, mask, coords, weight = _to_device(inputs, slice(start, end), device)
            knn_idx = knn_indices(coords, mask, k) if (kind == "gnn" and k) else None

            probs = torch.softmax(_logits(model, kind, x, mask, coords, knn_idx), dim=1)
            pred = probs.argmax(dim=1)
            rows = torch.arange(len(pred), device=probs.device)
            p_full = probs[rows, pred]

            # push padding behind every real constituent, including zero-energy ones
            rank_by = torch.where(mask > 0, weight, torch.full_like(weight, -float("inf")))
            order = rank_by.argsort(dim=1, descending=True)[:, :n_constituents]

            for c in range(order.shape[1]):
                idx = order[:, c]
                mask_occ = mask.clone()
                mask_occ[rows, idx] = 0.0
                p_occ = torch.softmax(
                    _logits(model, kind, x, mask_occ, coords, None), dim=1)[rows, pred]
                drops[start:end, c] = (p_full - p_occ).cpu().numpy()
                weights[start:end, c] = weight[rows, idx].cpu().numpy()
                picked[start:end, c] = idx.cpu().numpy()
            if verbose:
                print(f"  constituent occlusion {end}/{n} jets", flush=True)

    return {"drop": drops, "weight": weights, "index": picked}
