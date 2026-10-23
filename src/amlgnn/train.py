"""Full-batch training loop with class-weighted loss and early stopping on validation PR-AUC."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from amlgnn.config import TrainConfig
from amlgnn.metrics import compute_metrics, evaluate_predictions
from amlgnn.utils import get_device, set_seed


def forward_pass(model, data):
    """Uniform call for plain GNNs and time-aware models (model.needs_time)."""
    if getattr(model, "needs_time", False):
        return model(data.x, data.edge_index, data.time)
    return model(data.x, data.edge_index)


def class_weights(y, mask) -> torch.Tensor:
    y = y[mask]
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    tot = n_pos + n_neg
    return torch.tensor([tot / (2 * n_neg), tot / (2 * n_pos)], dtype=torch.float)


@torch.no_grad()
def predict_proba(model, data) -> np.ndarray:
    """P(illicit) for every node (numpy, on CPU)."""
    model.eval()
    return F.softmax(forward_pass(model, data), dim=-1)[:, 1].cpu().numpy()


def train_model(model, data, cfg: TrainConfig = None, verbose: bool = True) -> dict:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    model = model.to(device)
    data = data.clone().to(device)

    weights = class_weights(data.y, data.train_mask).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y_np = data.y.cpu().numpy()
    val_np = data.val_mask.cpu().numpy()

    best_score, best_state, best_epoch, bad = -1.0, None, 0, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        out = forward_pass(model, data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weights)
        loss.backward()
        opt.step()

        prob = predict_proba(model, data)
        val = compute_metrics(y_np[val_np], prob[val_np])
        score = val["pr_auc"] if not np.isnan(val["pr_auc"]) else val["f1"]
        history.append({"epoch": epoch, "loss": float(loss), "val_pr_auc": val["pr_auc"], "val_f1": val["f1"]})

        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:4d} | loss {loss:.4f} | val PR-AUC {val['pr_auc']:.4f} | val F1 {val['f1']:.4f}")
        if bad >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best {best_epoch})")
            break

    model.load_state_dict(best_state)
    prob = predict_proba(model, data)
    result = evaluate_predictions(prob, data)
    result.update({"history": history, "best_epoch": best_epoch, "probs": prob})
    return result
