import numpy as np
import torch
import torch.nn as nn

from config import CONFIG


class _SoftmaxWrapper(nn.Module):
    """Turn a logit-output model into a probability-output one for DeepExplainer."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def _fix_multiclass_shape(arr, n_features, n_classes):
    """Coerce a SHAP return value to (n_rows, n_features, n_classes)."""
    if isinstance(arr, list):
        return np.stack(arr, axis=-1)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == n_classes and arr.shape[2] == n_features:
        arr = np.transpose(arr, (0, 2, 1))
    return arr


def _check_shape(arr, n_rows, n_features, n_classes, who):
    expected = (n_rows, n_features, n_classes)
    if arr.shape != expected:
        raise ValueError(f"{who} returned shape {arr.shape}, expected {expected}")
    return arr


def explain_bdt_shap(model, X_explain, feature_names, n_classes, cfg=None,
                     background=None, seed=42, batch_size=2000, method="auto",
                     verbose=True):
    """SHAP for an XGBClassifier, by TreeSHAP or KernelSHAP."""
    import pandas as pd
    import shap

    cfg = (cfg or CONFIG).explainer
    if cfg.output_space not in ("logodds", "probability"):
        raise ValueError("output_space must be 'logodds' or 'probability'")
    if method not in ("auto", "tree", "kernel"):
        raise ValueError("method must be 'auto', 'tree' or 'kernel'")
    if method == "auto":
        method = "kernel" if cfg.output_space == "probability" else "tree"

    X_explain = np.asarray(X_explain)
    rng = np.random.default_rng(seed)
    if background is None:
        n_bg = min(cfg.background_size, len(X_explain))
        background = X_explain[rng.choice(len(X_explain), n_bg, replace=False)]
    background = np.asarray(background)

    if method == "kernel":
        if cfg.output_space != "probability":
            raise ValueError("method='kernel' explains predict_proba, so "
                             "output_space must be 'probability'")
        explainer = shap.KernelExplainer(lambda a: model.predict_proba(np.asarray(a)),
                                         background)
        raw = explainer.shap_values(X_explain, nsamples=cfg.shap_nsamples,
                                    silent=not verbose)
        out = _fix_multiclass_shape(raw, len(feature_names), n_classes)
        return _check_shape(out, len(X_explain), len(feature_names), n_classes,
                            "BDT KernelSHAP")

    if cfg.output_space == "logodds":
        explainer = shap.TreeExplainer(model)
    elif n_classes > 2:
        raise NotImplementedError(
            "shap cannot put multiclass TreeSHAP in probability space: it has no "
            f"objective entry for multi:softprob (shap {shap.__version__}). Pass "
            "method='kernel' for probability-space SHAP, or keep output_space="
            "'logodds' and do not compare the result against LIME."
        )
    else:
        explainer = shap.TreeExplainer(
            model, data=pd.DataFrame(background, columns=feature_names),
            feature_perturbation="interventional", model_output="probability",
        )

    chunks = []
    for start in range(0, len(X_explain), batch_size):
        rows = pd.DataFrame(X_explain[start:start + batch_size], columns=feature_names)
        chunks.append(_fix_multiclass_shape(explainer(rows).values,
                                            len(feature_names), n_classes))
        if verbose:
            print(f"  BDT SHAP {min(start + batch_size, len(X_explain))}/{len(X_explain)} rows",
                  flush=True)

    out = np.concatenate(chunks, axis=0)
    return _check_shape(out, len(X_explain), len(feature_names), n_classes, "BDT SHAP")


def explain_dnn_shap(model, device, X_explain_sc, X_train_sc, feature_names, n_classes,
                     cfg=None, seed=42, batch_size=500, verbose=True):
    """SHAP over a torch model, DeepExplainer in log-odds and GradientExplainer in probability."""
    import shap

    cfg = (cfg or CONFIG).explainer
    if cfg.output_space not in ("logodds", "probability"):
        raise ValueError("output_space must be 'logodds' or 'probability'")

    model.eval()
    sampled = cfg.output_space == "probability"
    explained = _SoftmaxWrapper(model).to(device).eval() if sampled else model

    rng = np.random.default_rng(seed)
    n_bg = min(cfg.background_size, len(X_train_sc))
    background = torch.tensor(np.asarray(X_train_sc)[rng.choice(len(X_train_sc), n_bg, replace=False)],
                              dtype=torch.float32).to(device)
    explainer = (shap.GradientExplainer if sampled else shap.DeepExplainer)(explained, background)

    X_explain_sc = np.asarray(X_explain_sc)
    chunks = []
    for start in range(0, len(X_explain_sc), batch_size):
        batch = torch.tensor(X_explain_sc[start:start + batch_size],
                             dtype=torch.float32).to(device)
        values = (explainer.shap_values(batch, nsamples=cfg.shap_nsamples) if sampled
                  else explainer.shap_values(batch))
        chunks.append(_fix_multiclass_shape(values, len(feature_names), n_classes))
        if verbose:
            print(f"  DNN SHAP {min(start + batch_size, len(X_explain_sc))}/{len(X_explain_sc)} rows",
                  flush=True)

    out = np.concatenate(chunks, axis=0)
    return _check_shape(out, len(X_explain_sc), len(feature_names), n_classes, "DNN SHAP")


def _probability_grads(model, device, X_sc, n_classes, batch_size=256):
    """d p_c / d x for every class, as (n_rows, n_features, n_classes)."""
    model.eval()
    X_sc = np.asarray(X_sc, dtype=np.float32)
    out = np.zeros((len(X_sc), X_sc.shape[1], n_classes), dtype=np.float64)

    for start in range(0, len(X_sc), batch_size):
        batch = torch.tensor(X_sc[start:start + batch_size]).to(device)
        for cls in range(n_classes):
            x = batch.clone().requires_grad_(True)
            probs = torch.softmax(model(x), dim=1)
            probs[:, cls].sum().backward()
            out[start:start + len(batch), :, cls] = x.grad.detach().cpu().numpy()
    return out


def explain_dnn_saliency(model, device, X_explain_sc, feature_names, n_classes,
                         cfg=None, seed=42, verbose=True):
    """Gradient of each class probability with respect to the input features."""
    out = _probability_grads(model, device, X_explain_sc, n_classes)
    if verbose:
        print(f"  DNN saliency {len(out)} rows", flush=True)
    return _check_shape(out, len(out), len(feature_names), n_classes, "DNN saliency")


def explain_dnn_ig(model, device, X_explain_sc, feature_names, n_classes, cfg=None,
                   seed=42, batch_size=256, verbose=True):
    """Gradients accumulated along a straight path from an all-zero baseline."""
    cfg = (cfg or CONFIG).explainer
    model.eval()
    X_sc = np.asarray(X_explain_sc, dtype=np.float32)
    out = np.zeros((len(X_sc), X_sc.shape[1], n_classes), dtype=np.float64)

    for start in range(0, len(X_sc), batch_size):
        batch = torch.tensor(X_sc[start:start + batch_size]).to(device)
        baseline = torch.zeros_like(batch)
        for cls in range(n_classes):
            grads = torch.zeros_like(batch)
            for step in range(cfg.ig_steps):
                alpha = (step + 0.5) / cfg.ig_steps
                x = (baseline + alpha * (batch - baseline)).requires_grad_(True)
                torch.softmax(model(x), dim=1)[:, cls].sum().backward()
                grads += x.grad.detach()
            ig = (batch - baseline) * grads / cfg.ig_steps
            out[start:start + len(batch), :, cls] = ig.detach().cpu().numpy()
        if verbose:
            print(f"  DNN IG {min(start + batch_size, len(X_sc))}/{len(X_sc)} rows",
                  flush=True)
    return _check_shape(out, len(X_sc), len(feature_names), n_classes, "DNN IG")


def explain_dnn_smoothgrad(model, device, X_explain_sc, feature_names, n_classes,
                           cfg=None, seed=42, batch_size=256, verbose=True):
    """Saliency averaged over Gaussian-noised copies of each row."""
    cfg = (cfg or CONFIG).explainer
    model.eval()
    X_sc = np.asarray(X_explain_sc, dtype=np.float32)
    out = np.zeros((len(X_sc), X_sc.shape[1], n_classes), dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)

    spread = X_sc.max(axis=0) - X_sc.min(axis=0)
    sigma = torch.tensor(np.maximum(cfg.smoothgrad_noise * spread, 1e-6),
                         dtype=torch.float32)

    for start in range(0, len(X_sc), batch_size):
        batch = torch.tensor(X_sc[start:start + batch_size]).to(device)
        for cls in range(n_classes):
            grads = torch.zeros_like(batch)
            for _ in range(cfg.smoothgrad_samples):
                noise = (torch.randn(batch.shape, generator=generator) * sigma).to(device)
                x = (batch + noise).requires_grad_(True)
                torch.softmax(model(x), dim=1)[:, cls].sum().backward()
                grads += x.grad.detach()
            out[start:start + len(batch), :, cls] = \
                (grads / cfg.smoothgrad_samples).cpu().numpy()
        if verbose:
            print(f"  DNN smoothgrad {min(start + batch_size, len(X_sc))}/{len(X_sc)} rows",
                  flush=True)
    return _check_shape(out, len(X_sc), len(feature_names), n_classes, "DNN smoothgrad")


def build_lime_explainer(X_train, feature_names, class_names, seed=42,
                         discretize_continuous=False):
    """One LimeTabularExplainer on the scale the model's predict_proba expects."""
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=list(feature_names),
        class_names=list(class_names),
        mode="classification",
        random_state=seed,
        discretize_continuous=discretize_continuous,
    )


def explain_lime(explainer, X_explain, predict_proba, n_features, n_classes,
                 log_every=100, verbose=True):
    """Run LIME row by row. The seed lives in the explainer, not here."""
    X_explain = np.asarray(X_explain)
    out = np.zeros((len(X_explain), n_features, n_classes))

    for row in range(len(X_explain)):
        exp = explainer.explain_instance(
            X_explain[row], predict_proba,
            labels=list(range(n_classes)), num_features=n_features,
        )
        for cls_idx in range(n_classes):
            for feat_idx, coef in exp.local_exp[cls_idx]:
                out[row, feat_idx, cls_idx] = coef
        if verbose and (row + 1) % log_every == 0:
            print(f"  LIME {row + 1}/{len(X_explain)} rows", flush=True)

    return out
