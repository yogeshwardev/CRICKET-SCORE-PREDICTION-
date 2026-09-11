"""Evaluate the delivery simulator against the champion on identical held-out overs.

The hybrid weight and every operating threshold are fitted on the calibration season.
The test season is scored once, at the end, with everything already fixed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skm

from cricket_ai.delivery import BUCKET_NAMES, DeliverySimulator, bucket_probabilities
from cricket_ai.evaluation import calibration_error, regression
from cricket_ai.features import bucket as bucket_of
from cricket_ai.models import matrix, split, team_predict

EDGES = [5, 8, 11, 15, 20]


def simulate_frame(engine, frame: pd.DataFrame, draws: int, seed: int, label: str) -> pd.DataFrame:
    rows, started = [], time.perf_counter()
    for position, (_, row) in enumerate(frame.iterrows()):
        result = engine.simulate(row.to_dict(), draws=draws, seed=seed+position)
        rows.append(result)
        if position and position % 200 == 0:
            rate = (time.perf_counter()-started)/position
            print(f"  {label}: {position}/{len(frame)} ({rate:.2f}s/over, "
                  f"eta {rate*(len(frame)-position)/60:.1f} min)", flush=True)
    return pd.DataFrame(rows, index=frame.index)


def ranked_probability_score(probabilities: np.ndarray, actual_index: np.ndarray) -> float:
    """RPS rewards putting mass near the truth, not only on it."""
    cumulative = probabilities.cumsum(axis=1)
    onehot = np.eye(probabilities.shape[1])[actual_index].cumsum(axis=1)
    return float(np.mean(((cumulative-onehot)**2).sum(axis=1)/(probabilities.shape[1]-1)))


def bucket_block(probabilities: np.ndarray, actual_index: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    report = skm.classification_report(actual_index, predicted, labels=range(len(BUCKET_NAMES)),
                                       target_names=BUCKET_NAMES, output_dict=True, zero_division=0)
    onehot = np.eye(len(BUCKET_NAMES))[actual_index]
    return {"accuracy": float(skm.accuracy_score(actual_index, predicted)),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "per_class": {name: {"precision": report[name]["precision"], "recall": report[name]["recall"],
                                 "f1": report[name]["f1-score"], "support": int(report[name]["support"])}
                          for name in BUCKET_NAMES},
            "log_loss": float(skm.log_loss(actual_index, np.clip(probabilities, 1e-9, 1),
                                           labels=list(range(len(BUCKET_NAMES))))),
            "brier": float(np.mean(np.sum((probabilities-onehot)**2, axis=1))),
            "ranked_probability_score": ranked_probability_score(probabilities, actual_index),
            "confusion_matrix": skm.confusion_matrix(actual_index, predicted,
                                                     labels=list(range(len(BUCKET_NAMES)))).tolist()}


def tail_block(probability: np.ndarray, actual: np.ndarray, threshold: float | None) -> dict:
    block = {"base_rate": float(actual.mean()), "mean_probability": float(probability.mean()),
             "roc_auc": float(skm.roc_auc_score(actual, probability)) if actual.any() and not actual.all() else None,
             "pr_auc": float(skm.average_precision_score(actual, probability)) if actual.any() else None,
             "brier": float(skm.brier_score_loss(actual, probability)),
             "ece": calibration_error(actual, probability)}
    if threshold is not None:
        flagged = probability >= threshold
        block.update(threshold=float(threshold), flagged=int(flagged.sum()),
                     precision=float(skm.precision_score(actual, flagged, zero_division=0)),
                     recall=float(skm.recall_score(actual, flagged, zero_division=0)),
                     f1=float(skm.f1_score(actual, flagged, zero_division=0)))
    return block


def choose_threshold(probability: np.ndarray, actual: np.ndarray) -> float | None:
    """Best-F1 operating point, chosen on the calibration season only."""
    if not actual.any():
        return None
    precision, recall, thresholds = skm.precision_recall_curve(actual, probability)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(precision), where=(precision+recall) > 0)
    return float(thresholds[max(0, min(int(np.argmax(f1)), len(thresholds)-1))])


def interval_from_quantiles(simulated: pd.DataFrame, low: str, high: str, truth: np.ndarray, target: float) -> dict:
    lower = np.floor(simulated[low].to_numpy())
    upper = np.ceil(simulated[high].to_numpy())
    covered = (truth >= lower) & (truth <= upper)
    return {"target_coverage": target, "observed_coverage": float(covered.mean()),
            "coverage_error": float(covered.mean()-target),
            "mean_width": float(np.mean(upper-lower)), "median_width": float(np.median(upper-lower))}


def quantile_columns(simulated: pd.DataFrame) -> pd.DataFrame:
    q = pd.DataFrame(list(simulated["quantiles"]), index=simulated.index)
    b = pd.DataFrame(list(simulated["bucket_probabilities"]), index=simulated.index)
    return pd.concat([simulated.drop(columns=["quantiles", "bucket_probabilities"]), q, b], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--sample", type=int, default=0, help="evaluate a fixed subsample of test overs")
    arguments = parser.parse_args()
    root = arguments.root.resolve()

    overs = pd.read_parquet(root / "data/processed/overs.parquet")
    report = json.loads((root / "reports/latest.json").read_text())
    champion_report = json.loads((root / "models/20260907T150806188678Z/report.json").read_text())
    champion = joblib.load(root / "models/20260907T150806188678Z/bundle.joblib")
    heads = joblib.load(root / "models/delivery_simulator_challenger/heads.joblib")
    engine = DeliverySimulator(heads["models"], heads["columns"])

    parts = split(overs)
    calibration = pd.concat([parts["probability_calibration"], parts["interval_calibration"]])
    test = parts["test"]
    if arguments.sample:
        test = test.sample(n=min(arguments.sample, len(test)), random_state=20260907).sort_index()

    print(f"calibration overs {len(calibration)}, test overs {len(test)}, draws {arguments.draws}", flush=True)
    simulated_calibration = quantile_columns(simulate_frame(engine, calibration, arguments.draws, 11, "calib"))
    simulated_test = quantile_columns(simulate_frame(engine, test, arguments.draws, 101, "test"))

    truth_calibration = calibration.target_next_over_runs.to_numpy(dtype=float)
    truth = test.target_next_over_runs.to_numpy(dtype=float)
    champion_calibration = np.asarray(team_predict(champion, matrix(calibration, champion["features"])), dtype=float)
    champion_test = np.asarray(team_predict(champion, matrix(test, champion["features"])), dtype=float)
    simulator_test = simulated_test["expected_runs"].to_numpy()

    # Hybrid weight fitted on the calibration season only.
    weights = np.linspace(0, 1, 21)
    scores = [np.abs(truth_calibration-((1-w)*champion_calibration
                                        + w*simulated_calibration["expected_runs"].to_numpy())).mean()
              for w in weights]
    best_weight = float(weights[int(np.argmin(scores))])
    hybrid_test = (1-best_weight)*champion_test + best_weight*simulator_test

    actual_index = np.array([bucket_of(v) for v in truth])
    bucket_probability = simulated_test[BUCKET_NAMES].to_numpy()
    champion_bucket = champion["classifiers"]["run_bucket"].predict_proba(matrix(test, champion["features"]))

    calibration_index = np.array([bucket_of(v) for v in truth_calibration])
    threshold15 = choose_threshold(simulated_calibration["probability_15_plus"].to_numpy(),
                                   (truth_calibration >= 15).astype(int))
    threshold20 = choose_threshold(simulated_calibration["probability_20_plus"].to_numpy(),
                                   (truth_calibration >= 20).astype(int))

    result = {
        "draws": arguments.draws, "test_overs": int(len(test)),
        "test_is_subsample": bool(arguments.sample),
        "hybrid_weight_on_simulator": best_weight,
        "hybrid_weight_fitted_on": "2025 calibration blocks",
        "point_prediction": {
            "champion": regression(truth, champion_test),
            "simulator": regression(truth, simulator_test),
            "hybrid": regression(truth, hybrid_test),
        },
        "run_buckets": {
            "champion": bucket_block(champion_bucket, actual_index),
            "simulator": bucket_block(bucket_probability, actual_index),
        },
        "tails": {
            "simulator_15_plus": tail_block(simulated_test["probability_15_plus"].to_numpy(),
                                            (truth >= 15).astype(int), threshold15),
            "simulator_20_plus": tail_block(simulated_test["probability_20_plus"].to_numpy(),
                                            (truth >= 20).astype(int), threshold20),
            "champion_15_plus": tail_block(champion_bucket[:, 4]+champion_bucket[:, 5],
                                           (truth >= 15).astype(int), None),
            "champion_20_plus": tail_block(champion_bucket[:, 5], (truth >= 20).astype(int), None),
        },
        "intervals": {
            "simulator_80": interval_from_quantiles(simulated_test, "p10", "p90", truth, .80),
            "simulator_95": interval_from_quantiles(simulated_test, "p10", "p95", truth, .95),
            "champion_80": {"target_coverage": .80,
                            "observed_coverage": float(((truth >= np.maximum(0, np.floor(champion_test-champion["radius"])))
                                                        & (truth <= np.ceil(champion_test+champion["radius"]))).mean()),
                            "mean_width": float(np.mean(np.ceil(champion_test+champion["radius"])
                                                        - np.maximum(0, np.floor(champion_test-champion["radius"]))))},
        },
        "explosive_segments": {},
    }
    for name, mask in [("actual_15_to_19", (truth >= 15) & (truth < 20)), ("actual_20_plus", truth >= 20)]:
        mask = np.asarray(mask)
        if mask.sum():
            result["explosive_segments"][name] = {
                "n": int(mask.sum()),
                "champion_mae": float(np.abs(truth[mask]-champion_test[mask]).mean()),
                "simulator_mae": float(np.abs(truth[mask]-simulator_test[mask]).mean()),
                "hybrid_mae": float(np.abs(truth[mask]-hybrid_test[mask]).mean()),
                "champion_mean_prediction": float(champion_test[mask].mean()),
                "simulator_mean_prediction": float(simulator_test[mask].mean()),
                "mean_actual": float(truth[mask].mean()),
                "mean_probability_15_plus": float(simulated_test["probability_15_plus"].to_numpy()[mask].mean()),
                "mean_probability_20_plus": float(simulated_test["probability_20_plus"].to_numpy()[mask].mean()),
            }

    out = root / "reports/experiments"
    out.mkdir(parents=True, exist_ok=True)
    simulated_test.assign(actual=truth, champion=champion_test, hybrid=hybrid_test).to_csv(
        out / "simulator_test_predictions.csv", index=False)
    (out / "simulator_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["point_prediction", "tails", "intervals",
                                             "explosive_segments", "hybrid_weight_on_simulator"]}, indent=2))
    print("written:", out / "simulator_eval.json")


if __name__ == "__main__":
    main()
