from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG


class EFN(nn.Module):
    def __init__(self, n_angular_features, latent_dim, n_classes,
                 phi_hidden=100, f_hidden=100):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_angular_features, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, latent_dim), nn.ReLU(),
        )
        self.f = nn.Sequential(
            nn.Linear(latent_dim, f_hidden), nn.ReLU(),
            nn.Linear(f_hidden, f_hidden), nn.ReLU(),
            nn.Linear(f_hidden, n_classes),
        )

    def forward(self, x_ang, z, mask):
        weighted = self.phi(x_ang) * (z * mask).unsqueeze(-1)
        return self.f(weighted.sum(dim=1))

    def particle_latent(self, x_ang):
        """Phi(p_hat_i) per particle, an angular filter map."""
        return self.phi(x_ang)


def _angular_and_energy(data, split, cfg, norm):
    """Pull the angular columns and the energy weight out of the node features."""
    names = data["node_features"]
    ang_idx = [names.index(n) for n in cfg.angular_features]
    z_idx = names.index(cfg.energy_feature)

    X, mask = data[f"X_{split}"], data[f"mask_{split}"]
    x_ang = (X[:, :, ang_idx] - norm["mean"][ang_idx]) / norm["std"][ang_idx]
    x_ang = x_ang * mask[..., None]
    return x_ang, X[:, :, z_idx] * mask, mask


def make_loader(data, split, norm, cfg, batch_size=512, shuffle=False):
    x_ang, z, mask = _angular_and_energy(data, split, cfg, norm)
    tensors = (torch.tensor(x_ang, dtype=torch.float32),
               torch.tensor(z, dtype=torch.float32),
               torch.tensor(mask, dtype=torch.float32),
               torch.tensor(data[f"y_{split}"], dtype=torch.long))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for x_ang, z, mask, y in loader:
            x_ang, z, mask, y = (t.to(device) for t in (x_ang, z, mask, y))
            logits = model(x_ang, z, mask)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def train_efn(data, cfg=None, seed=42, checkpoint_path=None, verbose=True):
    full_cfg = cfg or CONFIG
    cfg = full_cfg.efn
    n_classes = len(data["class_names"])
    checkpoint_path = Path(checkpoint_path or f"{full_cfg.checkpoint_dir}/efn_seed{seed}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = data["norm"]
    train_loader = make_loader(data, "train", norm, cfg, cfg.batch_size, shuffle=True)
    val_loader = make_loader(data, "val", norm, cfg, cfg.batch_size)

    model = EFN(n_angular_features=len(cfg.angular_features), latent_dim=cfg.latent_dim,
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
                        "angular_features": list(cfg.angular_features),
                        "energy_feature": cfg.energy_feature,
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

    model, device = load_efn(checkpoint_path)
    info = {"model": "efn", "seed": seed, "epochs": best_epoch, "epochs_run": epoch,
            "best_val_acc": float(best_val_acc), "checkpoint": str(checkpoint_path)}
    if verbose:
        print(f"EFN seed {seed}: best epoch {best_epoch}, val acc {best_val_acc:.4f}")
    return model, device, info


def load_efn(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EFN(n_angular_features=len(ckpt["angular_features"]), latent_dim=ckpt["latent_dim"],
                n_classes=ckpt["n_classes"], phi_hidden=ckpt["phi_hidden"],
                f_hidden=ckpt["f_hidden"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def evaluate_efn(model, device, loader, class_names, verbose=True):
    model.eval()
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for x_ang, z, mask, y in loader:
            out = torch.softmax(model(x_ang.to(device), z.to(device), mask.to(device)), dim=1)
            probs.append(out.cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
            labels.append(y.numpy())

    y_prob, y_pred, y_true = (np.concatenate(a) for a in (probs, preds, labels))
    acc = accuracy_score(y_true, y_pred)
    per_class_auc = {cls: float(roc_auc_score((y_true == i).astype(int), y_prob[:, i]))
                     for i, cls in enumerate(class_names)}
    if verbose:
        print(f"EFN test accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names))
    return {"accuracy": float(acc), "per_class_auc": per_class_auc,
            "y_pred": y_pred, "y_prob": y_prob}
