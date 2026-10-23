from amlgnn.baselines import run_baseline
from amlgnn.config import DataConfig, TrainConfig
from amlgnn.data import build_dataset
from amlgnn.models import build_model
from amlgnn.train import train_model


def _data():
    return build_dataset(DataConfig(synthetic=True, train_end=6, val_end=8),
                         n_steps=12, n_per_step=60, n_features=8)


def test_gnn_trains_and_beats_chance():
    d = _data()
    cfg = TrainConfig(hidden_dim=16, epochs=40, patience=40, device="cpu")
    res = train_model(build_model("gcn_res", 8, 16), d, cfg, verbose=False)
    assert res["test"]["pr_auc"] > 0.3          # base rate is ~0.12 on synthetic data
    assert len(res["history"]) > 0 and len(res["probs"]) == d.num_nodes


def test_baseline_runs():
    res = run_baseline("logreg", _data())
    assert set(res) >= {"val", "test", "threshold", "test_per_timestep"}
