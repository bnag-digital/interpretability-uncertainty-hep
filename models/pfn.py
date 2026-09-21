from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG


class PFN(nn.Module):
    def __init__(self, n_particle_features, latent_dim, n_classes,
                 phi_hidden=100, f_hidden=100):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_particle_features, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, latent_dim), nn.ReLU(),
        )
        self.f = nn.Sequential(
            nn.Linear(latent_dim, f_hidden), nn.ReLU(),
            nn.Linear(f_hidden, f_hidden), nn.ReLU(),
            nn.Linear(f_hidden, n_classes),
        )

    def forward(self, x, mask):
        phi_out = self.phi(x) * mask.unsqueeze(-1)
        return self.f(phi_out.sum(dim=1))

    def particle_latent(self, x):
        """Phi(p_i) per particle, used by the gradient explainers."""
        return self.phi(x)


def make_loader(data, split, norm, batch_size=512, shuffle=False):
    from data.hls4ml import normalise

    X = torch.tensor(normalise(data[f"X_{split}"], data[f"mask_{split}"], norm), dtype=torch.float32)
    mask = torch.tensor(data[f"mask_{split}"], dtype=torch.float32)
    y = torch.tensor(data[f"y_{split}"], dtype=torch.long)
    return DataLoader(TensorDataset(X, mask, y), batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def train_pfn(data, cfg=None, seed=42, checkpoint_path=None, verbose=True):
    full_cfg = cfg or CONFIG
    cfg = full_cfg.pfn
    n_classes = len(data["class_names"])
    checkpoint_path = Path(checkpoint_path or f"{full_cfg.checkpoint_dir}/pfn_seed{seed}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = data["norm"]
    train_loader = make_loader(data, "train", norm, cfg.batch_size, shuffle=True)
    val_loader = make_loader(data, "val", norm, cfg.batch_size)

    model = PFN(n_particle_features=len(data["node_features"]), latent_dim=cfg.latent_dim,
                n_classes=n_classes, phi_hidden=cfg.phi_hidden, f_hidden=cfg.f_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc, best_epoch, epochs_no_improve = 0.0, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)

        if val_acc > best_val_acc:
            epochs_no_improve = 0
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({"model_state": model.state_dict(), "seed": seed,
                        "latent_dim": cfg.latent_dim, "phi_hidden": cfg.phi_hidden,
                        "f_hidden": cfg.f_hidden, "n_classes": n_classes,
                        "node_features": data["node_features"],
                        "norm": {"mean": norm["mean"], "std": norm["std"]},
                        "epoch": epoch, "val_acc": val_acc},
                       checkpoint_path)
        else:
            epochs_no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"epoch {epoch:3d} | train {train_loss:.4f}/{train_acc:.4f} | "
                  f"val {val_loss:.4f}/{val_acc:.4f}", flush=True)

        if epochs_no_improve >= cfg.early_stop_patience:
            if verbose:
                print(f"early stop at epoch {epoch}, best was {best_epoch}", flush=True)
            break

    model, device = load_pfn(checkpoint_path)
    info = {"model": "pfn", "seed": seed, "epochs": best_epoch, "epochs_run": epoch,
            "best_val_acc": float(best_val_acc), "checkpoint": str(checkpoint_path)}
    if verbose:
        print(f"PFN seed {seed}: best epoch {best_epoch}, val acc {best_val_acc:.4f}")
    return model, device, info


def load_pfn(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PFN(n_particle_features=len(ckpt["node_features"]), latent_dim=ckpt["latent_dim"],
                n_classes=ckpt["n_classes"], phi_hidden=ckpt["phi_hidden"],
                f_hidden=ckpt["f_hidden"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def evaluate_pfn(model, device, loader, class_names, verbose=True):
    model.eval()
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for x, mask, y in loader:
            out = torch.softmax(model(x.to(device), mask.to(device)), dim=1)
            probs.append(out.cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
            labels.append(y.numpy())

    y_prob, y_pred, y_true = (np.concatenate(a) for a in (probs, preds, labels))
    acc = accuracy_score(y_true, y_pred)
    per_class_auc = {cls: float(roc_auc_score((y_true == i).astype(int), y_prob[:, i]))
                     for i, cls in enumerate(class_names)}
    if verbose:
        print(f"PFN test accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names))
    return {"accuracy": float(acc), "per_class_auc": per_class_auc,
            "y_pred": y_pred, "y_prob": y_prob}
