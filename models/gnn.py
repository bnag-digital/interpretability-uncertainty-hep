from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG


def knn_indices(coords, mask, k):
    """k nearest real neighbours per constituent, padding pushed to the back."""
    B, M, _ = coords.shape
    dist = torch.cdist(coords, coords)
    dist = dist + (1.0 - mask).unsqueeze(1) * 1e9
    dist = dist.masked_fill(torch.eye(M, device=coords.device).bool().unsqueeze(0), float("inf"))
    k_eff = min(k, M - 1)
    idx = dist.topk(k_eff, dim=-1, largest=False).indices
    if k_eff < k:
        idx = torch.cat([idx, idx[:, :, -1:].expand(-1, -1, k - k_eff)], dim=-1)
    return idx


def precompute_knn(coords, mask, k, batch_size=256, dtype=torch.int16):
    """Build the neighbour graph once for a whole dataset."""
    out = []
    for start in range(0, len(coords), batch_size):
        c = torch.as_tensor(coords[start:start + batch_size], dtype=torch.float32)
        m = torch.as_tensor(mask[start:start + batch_size], dtype=torch.float32)
        out.append(knn_indices(c, m, k).to(dtype))
    return torch.cat(out)


class EdgeConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
        )

    def forward(self, x, knn_idx, mask):
        B, M, C = x.shape
        k = knn_idx.shape[-1]
        neighbors = torch.gather(x, 1, knn_idx.reshape(B, M * k).unsqueeze(-1).expand(-1, -1, C))
        neighbors = neighbors.reshape(B, M, k, C)
        x_center = x.unsqueeze(2).expand(-1, -1, k, -1)
        edge_feat = torch.cat([x_center, neighbors - x_center], dim=-1)
        return self.mlp(edge_feat).max(dim=2).values * mask.unsqueeze(-1)


class ParticleGNN(nn.Module):
    def __init__(self, n_node_features, edgeconv_hidden, n_classes, k=8, f_hidden=100):
        super().__init__()
        self.k = k
        dims = [n_node_features] + list(edgeconv_hidden)
        self.blocks = nn.ModuleList([EdgeConv(dims[i], dims[i + 1])
                                     for i in range(len(edgeconv_hidden))])
        self.f = nn.Sequential(
            nn.Linear(edgeconv_hidden[-1], f_hidden), nn.ReLU(),
            nn.Linear(f_hidden, f_hidden), nn.ReLU(),
            nn.Linear(f_hidden, n_classes),
        )

    def forward(self, coords, x, mask, knn_idx=None):
        if knn_idx is None:
            knn_idx = knn_indices(coords, mask, self.k)
        elif knn_idx.dtype != torch.long:
            knn_idx = knn_idx.long()
        for block in self.blocks:
            x = block(x, knn_idx, mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.f(pooled)


def make_loader(data, split, norm, k=None, batch_size=256, shuffle=False, knn_cache=None):
    """DataLoader over one split, with inputs normalised by the stored stats."""
    from data.hls4ml import normalise

    X = torch.tensor(normalise(data[f"X_{split}"], data[f"mask_{split}"], norm), dtype=torch.float32)
    coords = torch.tensor(data[f"coords_{split}"], dtype=torch.float32)
    mask = torch.tensor(data[f"mask_{split}"], dtype=torch.float32)
    y = torch.tensor(data[f"y_{split}"], dtype=torch.long)

    if knn_cache is not None:
        tensors = (coords, X, mask, y, knn_cache)
    elif k is not None:
        tensors = (coords, X, mask, y, precompute_knn(coords, mask, k, batch_size))
    else:
        tensors = (coords, X, mask, y)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            coords, x, mask, y = (t.to(device) for t in batch[:4])
            knn_idx = batch[4].to(device) if len(batch) > 4 else None
            logits = model(coords, x, mask, knn_idx)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def train_gnn(data, cfg=None, seed=42, checkpoint_path=None, verbose=True):
    """Train one seed. Selection is on val; the reported number comes from test."""
    full_cfg = cfg or CONFIG
    cfg = full_cfg.gnn
    n_classes = len(data["class_names"])
    checkpoint_path = Path(checkpoint_path or f"{full_cfg.checkpoint_dir}/gnn_seed{seed}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = data["norm"]
    loaders = {s: make_loader(data, s, norm, k=cfg.k_neighbors, batch_size=cfg.batch_size,
                              shuffle=(s == "train"))
               for s in ("train", "val")}

    model = ParticleGNN(n_node_features=len(data["node_features"]),
                        edgeconv_hidden=list(cfg.edgeconv_hidden),
                        n_classes=n_classes, k=cfg.k_neighbors,
                        f_hidden=cfg.f_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc, best_epoch, epochs_no_improve = 0.0, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], device)

        if val_acc > best_val_acc:
            epochs_no_improve = 0
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({"model_state": model.state_dict(), "seed": seed,
                        "k": cfg.k_neighbors, "edgeconv_hidden": list(cfg.edgeconv_hidden),
                        "f_hidden": cfg.f_hidden, "n_classes": n_classes,
                        "node_features": data["node_features"],
                        "coord_features": data["coord_features"],
                        "norm": {"mean": norm["mean"], "std": norm["std"]},
                        "epoch": epoch, "val_acc": val_acc},
                       checkpoint_path)
        else:
            epochs_no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train {train_loss:.4f}/{train_acc:.4f} | "
                  f"val {val_loss:.4f}/{val_acc:.4f}", flush=True)

        if epochs_no_improve >= cfg.early_stop_patience:
            if verbose:
                print(f"early stop at epoch {epoch}, best was {best_epoch}", flush=True)
            break

    model, device = load_gnn(checkpoint_path)
    info = {"model": "gnn", "seed": seed, "epochs": best_epoch,
            "epochs_run": epoch, "best_val_acc": float(best_val_acc),
            "checkpoint": str(checkpoint_path)}
    if verbose:
        print(f"GNN seed {seed}: best epoch {best_epoch}, val acc {best_val_acc:.4f} "
              "(selection number, not the one to report)")
    return model, device, info


def load_gnn(path, device=None):
    """Reload a checkpoint, including the normalisation stats it was trained with."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ParticleGNN(n_node_features=len(ckpt["node_features"]),
                        edgeconv_hidden=ckpt["edgeconv_hidden"],
                        n_classes=ckpt["n_classes"], k=ckpt["k"],
                        f_hidden=ckpt["f_hidden"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def evaluate_gnn(model, device, loader, class_names, verbose=True):
    """Held-out test metrics. The val number was used for selection, so it is optimistic."""
    model.eval()
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            coords, x, mask, y = (t.to(device) for t in batch[:4])
            knn_idx = batch[4].to(device) if len(batch) > 4 else None
            out = torch.softmax(model(coords, x, mask, knn_idx), dim=1)
            probs.append(out.cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
            labels.append(y.cpu().numpy())

    y_prob, y_pred, y_true = (np.concatenate(a) for a in (probs, preds, labels))
    acc = accuracy_score(y_true, y_pred)
    per_class_auc = {cls: float(roc_auc_score((y_true == i).astype(int), y_prob[:, i]))
                     for i, cls in enumerate(class_names)}
    if verbose:
        print(f"GNN test accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names))
        for cls, auc in per_class_auc.items():
            print(f"  {cls} tagger: AUC = {auc:.3f}")

    return {"accuracy": float(acc), "per_class_auc": per_class_auc,
            "y_pred": y_pred, "y_prob": y_prob}
