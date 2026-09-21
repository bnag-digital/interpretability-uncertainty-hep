from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG


class JetDNN(nn.Module):
    def __init__(self, n_features=16, n_classes=5, hidden=(64, 32, 32), dropout=0.1):
        super().__init__()
        layers, in_dim = [], n_features
        for width in hidden:
            layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = width
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _to_tensors(X, y):
    return TensorDataset(torch.tensor(np.asarray(X), dtype=torch.float32),
                         torch.tensor(np.asarray(y), dtype=torch.long))


def _run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (out.argmax(1) == y_batch).sum().item()
            n += len(y_batch)
    return total_loss / n, correct / n


def train_dnn(X_train_sc, y_train, X_val_sc, y_val, cfg=None, n_classes=5, seed=42,
              checkpoint_path=None, norm=None, feature_names=None, verbose=True):
    """Adam with ReduceLROnPlateau and early stopping on validation loss."""
    full_cfg = cfg or CONFIG
    cfg = full_cfg.dnn
    checkpoint_path = Path(checkpoint_path or f"{full_cfg.checkpoint_dir}/dnn_seed{seed}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(_to_tensors(X_train_sc, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(_to_tensors(X_val_sc, y_val), batch_size=cfg.batch_size)

    # dropout is threaded through here and at reload, or MC-dropout looks deterministic
    model = JetDNN(n_features=X_train_sc.shape[1], n_classes=n_classes,
                   hidden=tuple(cfg.hidden), dropout=cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=10)

    best_val_loss, best_epoch, epochs_no_improve = float("inf"), 0, 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, epochs_no_improve = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "seed": seed,
                        "hidden": list(cfg.hidden), "dropout": cfg.dropout,
                        "n_features": X_train_sc.shape[1], "n_classes": n_classes,
                        "norm": norm, "feature_names": feature_names,
                        "epoch": epoch, "val_loss": val_loss},
                       checkpoint_path)
        else:
            epochs_no_improve += 1

        if verbose and epoch % 25 == 0:
            print(f"epoch {epoch:3d} | train {train_loss:.4f}/{train_acc:.4f} | "
                  f"val {val_loss:.4f}/{val_acc:.4f}")

        if epochs_no_improve >= cfg.early_stop_patience:
            break

    model, device = load_dnn(checkpoint_path)
    info = {"model": "dnn", "seed": seed, "epochs": best_epoch,
            "best_val_loss": float(best_val_loss),
            "checkpoint": str(checkpoint_path)}
    if verbose:
        print(f"DNN seed {seed}: best epoch {best_epoch}, val loss {best_val_loss:.4f}")
    return model, device, info


def load_dnn(path, device=None):
    """Reload with the architecture it was trained with, dropout included."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = JetDNN(n_features=ckpt["n_features"], n_classes=ckpt["n_classes"],
                   hidden=tuple(ckpt["hidden"]), dropout=ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def evaluate_dnn(model, device, X_test_sc, y_test, class_names, batch_size=1024, verbose=True):
    """Held-out test metrics."""
    loader = DataLoader(_to_tensors(X_test_sc, y_test), batch_size=batch_size)
    probs, preds, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = torch.softmax(model(X_batch.to(device)), dim=1)
            probs.append(out.cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
            labels.append(y_batch.numpy())

    y_prob, y_pred, y_true = (np.concatenate(a) for a in (probs, preds, labels))
    acc = accuracy_score(y_true, y_pred)
    per_class_auc = {cls: float(roc_auc_score((y_true == i).astype(int), y_prob[:, i]))
                     for i, cls in enumerate(class_names)}
    if verbose:
        print(f"DNN test accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names))
        for cls, auc in per_class_auc.items():
            print(f"  {cls} tagger: AUC = {auc:.3f}")

    return {"accuracy": float(acc), "per_class_auc": per_class_auc,
            "y_pred": y_pred, "y_prob": y_prob}


def dnn_predict_proba_factory(model, device, mc_dropout=False):
    """numpy in, probabilities out, the interface LIME expects."""
    def predict_proba(X_np):
        enable_mc_dropout(model) if mc_dropout else model.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X_np), dtype=torch.float32).to(device)
            return torch.softmax(model(X_t), dim=1).cpu().numpy()
    return predict_proba


def enable_mc_dropout(model):
    """Put dropout layers back in training mode for MC-dropout sampling."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model
