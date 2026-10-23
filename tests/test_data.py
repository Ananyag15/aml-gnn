import torch

from amlgnn.config import DataConfig
from amlgnn.data import build_dataset, summarize


def _data():
    cfg = DataConfig(synthetic=True, train_end=6, val_end=8)
    return build_dataset(cfg, n_steps=12, n_per_step=40, n_features=8)


def test_schema():
    d = _data()
    assert d.x.dim() == 2 and d.edge_index.size(0) == 2
    assert d.y.shape[0] == d.x.shape[0] == d.time.shape[0]
    assert set(d.y.unique().tolist()) <= {-1, 0, 1}


def test_masks_disjoint_and_temporal():
    d = _data()
    assert not (d.train_mask & d.val_mask).any()
    assert not (d.train_mask & d.test_mask).any()
    assert not (d.val_mask & d.test_mask).any()
    assert d.time[d.train_mask].max() < d.time[d.val_mask].min()
    assert d.time[d.val_mask].max() < d.time[d.test_mask].min()
    for m in (d.train_mask, d.val_mask, d.test_mask):
        assert (d.y[m] >= 0).all()          # unknown labels never used


def test_standardize_uses_train_stats():
    d = _data()
    x = d.x[d.train_mask]
    assert torch.allclose(x.mean(0), torch.zeros(x.size(1)), atol=1e-4)


def test_summary():
    s = summarize(_data())
    assert s["illicit"] > 0 and s["licit"] > 0 and s["num_time_steps"] == 12
