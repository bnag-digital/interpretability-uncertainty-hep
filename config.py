from dataclasses import dataclass, field
from pathlib import Path

CLASS_NAMES = ["j_g", "j_q", "j_w", "j_z", "j_t"]

# hls4ml artifacts carry no dataset prefix, so older results still load
DATASET_NAMES = ["hls4ml", "covertype", "adult"]
DEFAULT_DATASET = "hls4ml"

# BDT/DNN feature space. Order must match the loader, or importances are mislabelled.
JET_FEATURES = [
    "zlogz", "c1_b0_mmdt", "c1_b1_mmdt", "c1_b2_mmdt",
    "c2_b1_mmdt", "c2_b2_mmdt", "d2_b1_mmdt", "d2_b2_mmdt",
    "d2_a1_b1_mmdt", "d2_a1_b2_mmdt", "m2_b1_mmdt", "m2_b2_mmdt",
    "n2_b1_mmdt", "n2_b2_mmdt", "mass_mmdt", "multiplicity",
]

# GNN/PFN/EFN feature space. 16 columns, matching the 16 jet observables.
NODE_FEATURES = [
    "j1_px", "j1_py", "j1_pz", "j1_e", "j1_erel", "j1_pt", "j1_ptrel",
    "j1_eta", "j1_etarel", "j1_etarot", "j1_phi", "j1_phirel", "j1_phirot",
    "j1_deltaR", "j1_costheta", "j1_costhetarel",
]

# the 5 used by the reference implementation, kept for reproducing old results
NODE_FEATURES_5 = [
    "j1_etarel", "j1_phirel", "j1_ptrel", "j1_deltaR", "j1_costhetarel",
]
COORD_FEATURES = ["j1_etarel", "j1_phirel"]

# fixed-length summary basis the surrogate is fitted in
SUMMARY_NAMES = [
    "n_particles",
    "mean_ptrel", "max_ptrel", "std_ptrel",
    "mean_deltaR", "max_deltaR",
    "ptrel_weighted_deltaR",
    "mean_etarel", "mean_phirel",
    "mean_costhetarel",
]


@dataclass
class DataConfig:
    """Paths, split settings and caches for one dataset."""
    name: str = "hls4ml"
    jet_path: str = "/work/users/bnag/hls4ml_jets/train"
    # keep the OpenML cache off /home, where the quota is small
    openml_cache: str = "/work/users/bnag/.sklearn_cache"

    # the dataset's own partition; particle_file overrides it with a single .h5
    particle_train_dir: str = "/work/users/bnag/hls4ml_jets/train"
    particle_val_dir: str = "/work/users/bnag/hls4ml_jets/val"
    particle_file: str | None = None
    use_official_partition: bool = True

    n_jets: int | None = None            # None = use every jet available
    n_test_jets: int | None = None       # cap on jets read from val/
    max_constituents: int = 150
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    split_seed: int = 42
    split_index_path: str = "artifacts/split_index.npz"

    # cache of preprocessed split arrays, keyed by the settings that produced them
    cache_dir: str | None = "/work/users/bnag/.particle_cache"
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    jet_features: list[str] = field(default_factory=lambda: list(JET_FEATURES))
    node_features: list[str] = field(default_factory=lambda: list(NODE_FEATURES))
    coord_features: list[str] = field(default_factory=lambda: list(COORD_FEATURES))


@dataclass
class BDTConfig:
    n_estimators: int = 2000
    max_depth: int = 4
    learning_rate: float = 0.1
    # keep both below 1.0, or training is deterministic and the seed spread is zero
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 10


@dataclass
class DNNConfig:
    hidden: list[int] = field(default_factory=lambda: [64, 32, 32])
    dropout: float = 0.1
    lr: float = 1e-4
    batch_size: int = 1024
    epochs: int = 1000
    early_stop_patience: int = 10


@dataclass
class GNNConfig:
    k_neighbors: int = 8
    edgeconv_hidden: list[int] = field(default_factory=lambda: [64, 64, 128])
    f_hidden: int = 100
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 120                # cap, not a target
    early_stop_patience: int = 5


@dataclass
class PFNConfig:
    latent_dim: int = 128
    phi_hidden: int = 100
    f_hidden: int = 100
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 200                # cap, not a target
    early_stop_patience: int = 5     # same as the GNN, so the seed grids cost alike


@dataclass
class EFNConfig:
    latent_dim: int = 128
    phi_hidden: int = 100
    f_hidden: int = 100
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 200                # cap, not a target
    early_stop_patience: int = 5     # same as the GNN, so the seed grids cost alike
    # Phi sees angles only, which is what makes the EFN IRC-safe
    angular_features: list[str] = field(default_factory=lambda: ["j1_etarel", "j1_phirel"])
    energy_feature: str = "j1_ptrel"


@dataclass
class ExplainerConfig:
    """Settings shared by every explainer."""
    n_rows: int = 500
    background_size: int = 100
    output_space: str = "probability"
    shap_nsamples: int = 128
    lime_num_samples: int = 5000
    lime_num_features: int | None = None   # None = all features
    ig_steps: int = 50
    # the only set-valued explainer with a seed, so the only one with a floor
    smoothgrad_samples: int = 25
    smoothgrad_noise: float = 0.15


@dataclass
class ProtocolConfig:
    """How many times each level of variation is repeated."""
    model_seeds: list[int] = field(default_factory=lambda: list(range(8)))
    n_floor_repeats: int = 5        # level 1, explainer reruns at a fixed model
    n_split_repeats: int = 0        # level 4, 0 disables data resampling
    n_bootstrap: int = 500
    n_permutations: int = 5000
    ci: float = 0.90


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    bdt: BDTConfig = field(default_factory=BDTConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    pfn: PFNConfig = field(default_factory=PFNConfig)
    efn: EFNConfig = field(default_factory=EFNConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)

    models: list[str] = field(default_factory=lambda: ["bdt", "dnn", "gnn", "pfn", "efn"])
    explainers: list[str] = field(default_factory=lambda: ["shap", "lime"])
    artifact_dir: str = "artifacts"
    checkpoint_dir: str = "checkpoints"
    figure_dir: str = "figures"
    # job records rather than results, so they stay out of artifacts/
    log_dir: str = "logs"

    # a user model supplied as weights only; no training entrypoint means no ceiling
    pretrained_path: str | None = None

    def paths(self) -> dict[str, Path]:
        return {"artifacts": Path(self.artifact_dir),
                "checkpoints": Path(self.checkpoint_dir),
                "figures": Path(self.figure_dir),
                "logs": Path(self.log_dir)}

    def make_dirs(self) -> None:
        for p in self.paths().values():
            p.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
