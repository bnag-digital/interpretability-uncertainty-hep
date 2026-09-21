from pathlib import Path

import numpy as np
import torch

from exceptions import InsufficientCapability


class PretrainedModel:
    """A user checkpoint plus whatever is known about how to reproduce it."""

    def __init__(self, model, norm=None, feature_names=None, class_names=None,
                 train_fn=None, checkpoints=None, name="user_model", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # match the device the inputs are on, or a CPU-built model fails against a GPU
        self.model = model.to(self.device)
        self.norm = norm
        self.feature_names = feature_names
        self.class_names = class_names
        self.train_fn = train_fn
        self.checkpoints = [Path(p) for p in (checkpoints or [])]
        self.name = name

    @property
    def n_seeds(self):
        return max(len(self.checkpoints), 1)

    def capabilities(self):
        """What this model supports, and why anything missing is missing."""
        can_ceiling = self.train_fn is not None or len(self.checkpoints) >= 2
        if self.train_fn is not None:
            reason = "a training entrypoint was supplied, so seeds can be retrained"
        elif len(self.checkpoints) >= 2:
            reason = f"{len(self.checkpoints)} seed checkpoints were supplied"
        else:
            reason = ("only one checkpoint was supplied. The retraining ceiling needs "
                      "either several seeds or a way to retrain, and no resampling "
                      "recovers it from one set of weights.")
        return {"floor": True, "ceiling": can_ceiling, "cross": True,
                "verdict": can_ceiling, "reason": reason}

    def require_ceiling(self):
        caps = self.capabilities()
        if not caps["ceiling"]:
            raise InsufficientCapability(caps["reason"])
        return True

    def predict_proba_factory(self):
        """numpy in, probabilities out, the interface SHAP and LIME expect."""
        def predict_proba(X_np):
            self.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(np.asarray(X_np), dtype=torch.float32).to(self.device)
                out = self.model(X_t)
                if out.ndim == 1:
                    out = out.unsqueeze(0)
                return torch.softmax(out, dim=1).cpu().numpy()
        return predict_proba


def load_pretrained(path, model_class=None, model_kwargs=None, train_fn=None,
                    checkpoints=None, name="user_model", device=None):
    """Load a .pt checkpoint into a PretrainedModel."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(path, map_location=device, weights_only=False)

    norm, feature_names, class_names = None, None, None
    if isinstance(obj, dict):
        state = obj.get("model_state", obj.get("state_dict", obj))
        norm = obj.get("norm")
        feature_names = obj.get("feature_names") or obj.get("node_features")
        class_names = obj.get("class_names") or obj.get("label_names")
        if model_class is None:
            raise ValueError(
                f"{path} holds a state_dict, so model_class is needed to rebuild the "
                "architecture. Pass model_class=... and model_kwargs=...")
        model = model_class(**(model_kwargs or {})).to(device)
        model.load_state_dict(state)
    else:
        model = obj.to(device)

    model.eval()
    return PretrainedModel(model, norm=norm, feature_names=feature_names,
                           class_names=class_names, train_fn=train_fn,
                           checkpoints=checkpoints or [path], name=name, device=device)


def find_seed_checkpoints(directory, pattern="*seed*.pt"):
    """Every seed checkpoint in a directory, so a ceiling can be measured from them."""
    return sorted(Path(directory).glob(pattern))
