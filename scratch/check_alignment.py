import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from data import hls4ml

N_JETS = 200000

cfg = CONFIG
cfg.data.n_jets = N_JETS
cfg.data.split_index_path = f"artifacts/split_gnn_{N_JETS}.npz"

particle = hls4ml.load_particle_level(cfg=cfg, n_jets=N_JETS, verbose=True)
jet = hls4ml.load_jet_level_h5(cfg=cfg, n_jets=N_JETS, verbose=True)

ok = True
for name in ("train", "val", "test"):
    a, b = np.asarray(particle[f"y_{name}"]), np.asarray(jet[f"y_{name}"])
    same_len = len(a) == len(b)
    same = same_len and bool(np.array_equal(a, b))
    ok = ok and same
    print(f"{name:6s} particle {len(a):7d}  jet {len(b):7d}  labels identical: {same}")
    if same_len and not same:
        print(f"       first mismatch at row {int(np.flatnonzero(a != b)[0])}")

print(f"\nclass balance train, particle: {np.bincount(particle['y_train'])}")
print(f"class balance train, jet:      {np.bincount(jet['y_train'])}")
print(f"\njet X_train {jet['X_train'].shape}, features {jet['feature_names'][:3]}...")
print("ALIGNED" if ok else "NOT ALIGNED: do not train on this")
