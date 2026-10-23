"""Loader for the Elliptic Bitcoin dataset (Kaggle: ellipticco/elliptic-data-set).

Expected files inside ``data_dir``:
    elliptic_txs_features.csv   (no header: txId, time_step, 165 features)
    elliptic_txs_classes.csv    (txId, class in {"1" illicit, "2" licit, "unknown"})
    elliptic_txs_edgelist.csv   (txId1, txId2)

Returned ``torch_geometric.data.Data`` fields:
    x [N, F] float, edge_index [2, E] long, y [N] long (1 illicit, 0 licit, -1 unknown),
    time [N] long (1..49)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"


def load_elliptic(data_dir: str) -> Data:
    d = Path(data_dir)
    missing = [f for f in (FEATURES_FILE, CLASSES_FILE, EDGES_FILE) if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in '{d}'. Download the Elliptic dataset from Kaggle "
            "(ellipticco/elliptic-data-set) and put the CSV files there, "
            "or run with --synthetic."
        )

    feats = pd.read_csv(d / FEATURES_FILE, header=None)
    classes = pd.read_csv(d / CLASSES_FILE)
    edges = pd.read_csv(d / EDGES_FILE)

    tx_ids = feats.iloc[:, 0].to_numpy()
    id2idx = pd.Series(np.arange(len(tx_ids)), index=tx_ids)

    time = feats.iloc[:, 1].to_numpy().astype(np.int64)
    x = feats.iloc[:, 2:].to_numpy().astype(np.float32)

    cls = pd.Series(classes.iloc[:, 1].astype(str).to_numpy(), index=classes.iloc[:, 0].to_numpy())
    cls = cls.reindex(tx_ids).fillna("unknown")
    y = cls.map({"1": 1, "2": 0, "unknown": -1}).fillna(-1).astype(np.int64).to_numpy()

    src = id2idx.loc[edges.iloc[:, 0].to_numpy()].to_numpy()
    dst = id2idx.loc[edges.iloc[:, 1].to_numpy()].to_numpy()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.from_numpy(y),
        time=torch.from_numpy(time),
    )
