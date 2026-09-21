import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from config import CONFIG


def train_bdt(X_train, y_train, X_val, y_val, cfg=None, n_classes=5, seed=42,
              checkpoint_path=None, verbose=True):
    """Fit with early stopping on the validation split. Returns (model, info)."""
    import xgboost as xgb

    cfg = (cfg or CONFIG).bdt
    bdt = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        objective="multi:softprob",
        num_class=n_classes,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=cfg.early_stopping_rounds,
        eval_metric="mlogloss",
    )
    bdt.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50 if verbose else False)

    info = {"model": "bdt", "seed": seed,
            "epochs": int(bdt.best_iteration + 1),
            "best_val_mlogloss": float(bdt.best_score)}
    if verbose:
        print(f"BDT seed {seed}: stopped at {info['epochs']} trees "
              f"(best val mlogloss {info['best_val_mlogloss']:.4f})")

    if checkpoint_path:
        save_bdt(bdt, checkpoint_path, info)
    return bdt, info


def save_bdt(model, path, info):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(p))
    with open(p.with_suffix(".json.meta"), "w") as f:
        json.dump(info, f, indent=2)
    return p


def load_bdt(path):
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(str(path))
    meta_path = Path(path).with_suffix(".json.meta")
    info = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, info


def evaluate_bdt(model, X_test, y_test, class_names, verbose=True):
    """Held-out test metrics. Never report the validation number used for stopping."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    per_class_auc = {cls: float(roc_auc_score((y_test == i).astype(int), y_prob[:, i]))
                     for i, cls in enumerate(class_names)}
    if verbose:
        print(f"BDT test accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))
        for cls, auc in per_class_auc.items():
            print(f"  {cls} tagger: AUC = {auc:.3f}")

    return {"accuracy": float(acc), "per_class_auc": per_class_auc,
            "y_pred": y_pred, "y_prob": y_prob}


def bdt_predict_proba_factory(model):
    """numpy in, probabilities out, the interface LIME expects."""
    def predict_proba(X_np):
        return model.predict_proba(np.asarray(X_np))
    return predict_proba
