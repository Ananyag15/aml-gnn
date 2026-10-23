"""Classic non-graph baselines (the Elliptic paper reports Random Forest as a very strong baseline)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from amlgnn.metrics import evaluate_predictions

BASELINES = ["rf", "logreg"]


def run_baseline(name: str, data, seed: int = 42) -> dict:
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    tr = data.train_mask.cpu().numpy()
    if name == "rf":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     n_jobs=-1, random_state=seed)
    elif name == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        raise ValueError(f"Unknown baseline {name}")
    clf.fit(X[tr], y[tr])
    prob = clf.predict_proba(X)[:, 1]
    res = evaluate_predictions(prob, data)
    res.update({"history": [], "best_epoch": 0, "probs": prob})
    return res
