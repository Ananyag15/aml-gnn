"""Temporal train/val/test split + leakage-safe feature standardisation."""
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def add_temporal_split(data: Data, train_end: int, val_end: int) -> Data:
    """Split by time step (never randomly -> no look-ahead leakage). Only labelled nodes enter masks."""
    labeled = data.y >= 0
    data.train_mask = labeled & (data.time <= train_end)
    data.val_mask = labeled & (data.time > train_end) & (data.time <= val_end)
    data.test_mask = labeled & (data.time > val_end)
    return data


def standardize_features(data: Data) -> Data:
    """z-score using statistics of TRAIN nodes only."""
    x = data.x[data.train_mask]
    mean = x.mean(0)
    std = x.std(0).clamp_min(1e-6)
    data.x = (data.x - mean) / std
    return data


def make_undirected(data: Data) -> Data:
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    return data
