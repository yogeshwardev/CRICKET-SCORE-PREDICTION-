from __future__ import annotations
import numpy as np
from sklearn import metrics as m


def regression(y, prediction):
    errors = np.abs(np.asarray(y)-prediction)
    return {"mae": float(m.mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(m.mean_squared_error(y, prediction))),
            "r2": float(m.r2_score(y, prediction)), "median_ae": float(np.median(errors)),
            **{f"within_{n}_runs": float(np.mean(errors <= n)) for n in [1, 2, 3, 4]}}


def calibration_error(y, p, bins=10):
    result = 0.
    for a, b in zip(np.linspace(0, 1, bins+1)[:-1], np.linspace(0, 1, bins+1)[1:]):
        mask = (p >= a) & (p < b if b < 1 else p <= b)
        if mask.any():
            result += mask.mean() * abs(np.asarray(y)[mask].mean()-p[mask].mean())
    return float(result)


def classification(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = np.clip(probabilities, 1e-9, 1-1e-9)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    binary = p.shape[1] == 2
    result = {"accuracy": float(m.accuracy_score(y, pred)),
              "precision": float(m.precision_score(y, pred, average="binary" if binary else "macro", zero_division=0)),
              "recall": float(m.recall_score(y, pred, average="binary" if binary else "macro", zero_division=0)),
              "f1": float(m.f1_score(y, pred, average="binary" if binary else "macro", zero_division=0)),
              "weighted_f1": float(m.f1_score(y, pred, average="weighted", zero_division=0)),
              "log_loss": float(m.log_loss(y, p, labels=list(range(p.shape[1])))),
              "confusion_matrix": m.confusion_matrix(y, pred, labels=list(range(p.shape[1]))).tolist()}
    if binary:
        result.update(brier=float(m.brier_score_loss(y, p[:, 1])), ece=calibration_error(y, p[:, 1]),
                      roc_auc=float(m.roc_auc_score(y, p[:, 1])) if len(np.unique(y)) > 1 else None)
    else:
        onehot = np.eye(p.shape[1])[y]
        result.update(brier=float(np.mean(np.sum((p-onehot)**2, axis=1))),
                      ece=calibration_error(y == pred, p.max(axis=1)))
    result["reliability"] = []
    confidence = p[:, 1] if binary else p.max(axis=1)
    observed = y if binary else (pred == y)
    for lo in np.arange(0, 1, .1):
        mask = (confidence >= lo) & (confidence < lo+.1)
        if mask.any():
            result["reliability"].append({"predicted": float(confidence[mask].mean()),
                                           "observed": float(observed[mask].mean()), "n": int(mask.sum())})
    return result


def marginal_conformal_radius(y, predictions, alpha=.2):
    """Split conformal over single overs: the headline 80% interval for one forecast.

    Targets marginal coverage of the next over, which is the quantity a single-over
    forecast actually claims. Overs inside one match are correlated, so exchangeability
    holds only approximately; observed coverage is therefore always measured and reported
    rather than assumed from the construction.
    """
    scores = np.sort(np.abs(np.asarray(y)-predictions))
    rank = int(np.ceil((len(scores)+1)*(1-alpha)))
    if rank > len(scores):
        raise ValueError("Too few calibration overs for a finite conformal interval")
    return float(scores[rank-1])


def conformal_radius(y, predictions, match_ids, alpha=.2):
    """Block split conformal: max residual per match, finite-sample rank.

    Covers every over in a new exchangeable match simultaneously. Temporal drift
    violates exchangeability; report observed coverage and never guarantee it live.
    """
    import pandas as pd
    scores = pd.DataFrame({"error": np.abs(np.asarray(y)-predictions), "match": np.asarray(match_ids)}).groupby("match").error.max().to_numpy()
    rank = int(np.ceil((len(scores)+1)*(1-alpha)))
    if rank > len(scores):
        raise ValueError("Too few calibration matches for finite conformal interval")
    return float(np.sort(scores)[rank-1])
