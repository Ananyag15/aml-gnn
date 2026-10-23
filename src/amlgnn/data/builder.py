"""One-call dataset construction used by every script."""
from torch_geometric.data import Data

from amlgnn.config import DataConfig
from .loader import load_elliptic
from .synthetic import make_synthetic
from .split import add_temporal_split, standardize_features, make_undirected


def build_dataset(cfg: DataConfig, **synthetic_kwargs) -> Data:
    data = make_synthetic(**synthetic_kwargs) if cfg.synthetic else load_elliptic(cfg.data_dir)
    data = add_temporal_split(data, cfg.train_end, cfg.val_end)
    if cfg.standardize:
        data = standardize_features(data)
    if cfg.to_undirected:
        data = make_undirected(data)
    return data
