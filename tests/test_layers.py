import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

LAYERS = {
    "config": 0, "exceptions": 0,
    "metrics": 1, "results": 1, "split": 1,
    "data": 2,
    "models": 3,
    "explainers": 4,
    "protocols": 5, "perturbations": 5,
    "plots": 6,
    "scripts": 7,
}

SKIP_DIRS = {"references", "tests", "artifacts", "checkpoints", "figures",
             ".git", "__pycache__", ".ipynb_checkpoints", ".pixi"}


def project_files():
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def layer_of(path):
    rel = path.relative_to(ROOT)
    top = rel.parts[0] if len(rel.parts) > 1 else rel.stem
    return LAYERS.get(top)


def imported_modules(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


@pytest.mark.parametrize("path", list(project_files()), ids=lambda p: str(p.name))
def test_imports_only_from_lower_layers(path):
    own = layer_of(path)
    if own is None:
        pytest.skip(f"{path.name} is not in a known layer")
    for module in imported_modules(path):
        other = LAYERS.get(module)
        if other is None:
            continue
        assert other < own, (
            f"{path.relative_to(ROOT)} (L{own}) imports {module} (L{other}). "
            "Imports may only go to lower-numbered layers."
        )


def test_plots_never_import_the_training_stack():
    plots_dir = ROOT / "plots"
    if not plots_dir.exists():
        pytest.skip("plots/ does not exist yet")
    for path in plots_dir.rglob("*.py"):
        for module in imported_modules(path):
            assert module not in {"models", "data", "explainers", "torch", "xgboost"}, (
                f"{path.name} imports {module}. Plots must read artifacts only, so "
                "figures can be rebuilt with no GPU and no retraining."
            )
