import pytest
import torch

from amlgnn.models import ALL_MODELS, build_model


@pytest.mark.parametrize("name", ALL_MODELS)
def test_forward_backward(name):
    torch.manual_seed(0)
    x = torch.randn(30, 16)
    ei = torch.randint(0, 30, (2, 90))
    m = build_model(name, in_dim=16, hidden_dim=32, num_layers=3, heads=4)
    out = m(x, ei)
    assert out.shape == (30, 2)
    assert m.embed(x, ei).shape == (30, m.embed_dim)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("nope", 8)


def test_dense_embed_dim():
    m = build_model("gcn_dense", 8, hidden_dim=16, num_layers=2)
    assert m.embed_dim == 16 * 3
