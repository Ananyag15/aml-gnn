"""Central configuration (dataclasses) shared by every module."""
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    data_dir: str = "data/elliptic"   # folder holding the 3 Elliptic CSV files
    synthetic: bool = False            # use generated data (for CI / quick tests)
    train_end: int = 29                # time steps 1..train_end   -> train
    val_end: int = 34                  # train_end+1..val_end      -> validation
    to_undirected: bool = True         # message passing in both directions
    standardize: bool = True           # z-score features using TRAIN statistics only


@dataclass
class TrainConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    heads: int = 4                     # only used by GAT variants
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 30                 # early stopping on validation PR-AUC
    seed: int = 42
    device: str = "auto"               # "auto" | "cpu" | "cuda"


def as_dict(cfg) -> dict:
    return asdict(cfg)
