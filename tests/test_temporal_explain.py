import torch

from amlgnn.config import DataConfig
from amlgnn.data import build_dataset, make_synthetic
from amlgnn.explain import (feature_saliency, illicit_components, neighbor_homophily,
                            structural_stats)
from amlgnn.models import build_model
from amlgnn.temporal import TemporalContextGNN


def test_temporal_forward_shape():
    d = make_synthetic(n_steps=6, n_per_step=20, n_features=8)
    m = TemporalContextGNN(8, hidden_dim=16, num_layers=2)
    out = m(d.x, d.edge_index, d.time)
    assert out.shape == (d.num_nodes, 2)
    out.sum().backward()


def test_temporal_is_causal():
    """Changing the LAST time step must not change predictions for earlier steps."""
    torch.manual_seed(0)
    d = make_synthetic(n_steps=6, n_per_step=20, n_features=8)
    m = TemporalContextGNN(8, hidden_dim=16).eval()
    x2 = d.x.clone()
    x2[d.time == 6] += 5.0
    with torch.no_grad():
        o1, o2 = m(d.x, d.edge_index, d.time), m(x2, d.edge_index, d.time)
    earlier = d.time < 6
    assert torch.allclose(o1[earlier], o2[earlier], atol=1e-5)


def test_explain_utils():
    d = build_dataset(DataConfig(synthetic=True, train_end=6, val_end=8, to_undirected=False),
                      n_steps=12, n_per_step=40, n_features=8)
    assert len(structural_stats(d)) == 2
    h = neighbor_homophily(d)
    assert h["P(next hop illicit | source illicit)"] > h["P(next hop illicit | source licit)"]
    g, sizes = illicit_components(d)
    assert sizes and sum(sizes) == int((d.y == 1).sum())
    model = build_model("gcn", 8, hidden_dim=16)
    sal = feature_saliency(model, d, d.test_mask & (d.y == 1), top_k=5)
    assert len(sal) == 5 and (sal["saliency"] >= 0).all()
