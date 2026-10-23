"""Synthetic stand-in for Elliptic with the SAME schema (used by CI and quick tests).

Illicit nodes have shifted features and preferentially connect to each other,
so a GNN has real signal to learn. Edges only connect nodes in the same time step,
exactly like the real dataset.
"""
import numpy as np
import torch
from torch_geometric.data import Data


def make_synthetic(n_per_step: int = 80, n_steps: int = 49, n_features: int = 32,
                   illicit_rate: float = 0.12, unknown_rate: float = 0.5,
                   degree: int = 3, homophily: float = 0.7, seed: int = 0) -> Data:
    rng = np.random.default_rng(seed)
    xs, ys, ts, src_all, dst_all = [], [], [], [], []
    offset = 0
    for t in range(1, n_steps + 1):
        n = n_per_step
        y_true = (rng.random(n) < illicit_rate).astype(np.int64)
        x = rng.normal(size=(n, n_features)) + 0.6 * y_true[:, None]
        ill = np.where(y_true == 1)[0]
        lic = np.where(y_true == 0)[0]
        for i in range(n):
            for _ in range(degree):
                same = rng.random() < homophily
                pool = ill if (y_true[i] == 1) == same else lic
                if len(pool) == 0:
                    continue
                j = int(rng.choice(pool))
                if j != i:
                    src_all.append(offset + i)
                    dst_all.append(offset + j)
        y = y_true.copy()
        y[rng.random(n) < unknown_rate] = -1       # hide some labels like the real data
        xs.append(x.astype(np.float32))
        ys.append(y)
        ts.append(np.full(n, t, dtype=np.int64))
        offset += n

    return Data(
        x=torch.from_numpy(np.concatenate(xs)),
        edge_index=torch.tensor([src_all, dst_all], dtype=torch.long),
        y=torch.from_numpy(np.concatenate(ys)),
        time=torch.from_numpy(np.concatenate(ts)),
    )
