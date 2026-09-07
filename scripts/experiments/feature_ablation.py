"""Phase 17 experiment: permutation importance and a feature-removal ablation.

Importance is measured on the validation season against the promoted ensemble, so
it reflects the model that is actually served. The reduced-feature retrain is judged
on the same validation season. The final test season is never read here, and no
feature set is adopted unless validation improves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cricket_ai.evaluation import regression
from cricket_ai.features import CATEGORICAL, feature_columns
from cricket_ai.models import SEED, cat_reg, matrix, split, team_predict


def paired_improvement(y, baseline, challenger, sigmas=2.0):
    """Is the challenger better by more than the noise of the comparison itself?

    Both models score the same rows, so the right quantity is the per-row difference in
    absolute error. Its standard error says how much of any gap is sampling noise. A gap
    inside `sigmas` standard errors is not evidence, however tempting the raw number is.
    """
    y = np.asarray(y, dtype=float)
    difference = np.abs(y-np.asarray(baseline, dtype=float)) - np.abs(y-np.asarray(challenger, dtype=float))
    mean = float(difference.mean())
    standard_error = float(difference.std(ddof=1)/np.sqrt(len(difference))) if len(difference) > 1 else float("inf")
    return {"mae_improvement": mean, "standard_error": standard_error,
            "z": float(mean/standard_error) if standard_error > 0 else 0.0,
            "threshold_sigmas": sigmas, "significant": bool(mean > sigmas*standard_error)}


def permutation_importance(bundle, frame, columns, repeats, seed):
    rng = np.random.default_rng(seed)
    x = matrix(frame, columns)
    y = frame.target_next_over_runs.to_numpy()
    reference = regression(y, team_predict(bundle, x))["mae"]
    rows = []
    for name in columns:
        deltas = []
        original = x[name].to_numpy(copy=True)
        for _ in range(repeats):
            x[name] = rng.permutation(original)
            deltas.append(regression(y, team_predict(bundle, x))["mae"] - reference)
        x[name] = original
        rows.append({"feature": name, "mae_increase": float(np.mean(deltas)), "std": float(np.std(deltas))})
        print(f"{name:<44} {rows[-1]['mae_increase']:+.5f}", flush=True)
    return reference, pd.DataFrame(rows).sort_values("mae_increase", ascending=False)


def run(root: Path, repeats: int, threshold: float) -> dict:
    overs = pd.read_parquet(root / "data/processed/overs.parquet")
    report = json.loads((root / "reports/latest.json").read_text())
    bundle = joblib.load(root / "models" / report["version"] / "bundle.joblib")
    parts = split(overs)
    columns = feature_columns(overs)
    if columns != bundle["features"]:
        raise ValueError("Processed dataset schema no longer matches the trained bundle")

    print("Permutation importance on the validation season", flush=True)
    reference, importance = permutation_importance(bundle, parts["validation"], columns, repeats, SEED)
    destination = root / "reports/experiments"
    destination.mkdir(parents=True, exist_ok=True)
    importance.to_csv(destination / "permutation_importance.csv", index=False)

    dropped = importance[importance.mae_increase <= threshold].feature.tolist()
    kept = [c for c in columns if c not in dropped]
    parameters = dict(report["catboost_parameters"])
    y_train = parts["train"].target_next_over_runs.to_numpy()
    y_validation = parts["validation"].target_next_over_runs.to_numpy()

    def catboost_on(subset):
        model = cat_reg(**parameters)
        categorical = [c for c in CATEGORICAL if c in subset]
        model.fit(matrix(parts["train"], subset), y_train, cat_features=categorical,
                  eval_set=(matrix(parts["validation"], subset), y_validation), early_stopping_rounds=50)
        prediction = model.predict(matrix(parts["validation"], subset))
        return regression(y_validation, prediction), prediction

    print(f"Retraining CatBoost on {len(kept)} of {len(columns)} features", flush=True)
    full, full_prediction = catboost_on(columns)
    reduced, reduced_prediction = catboost_on(kept) if dropped else (full, full_prediction)
    significance = paired_improvement(y_validation, full_prediction, reduced_prediction)
    improvement = significance["mae_improvement"]
    result = {"model_version": report["version"], "repeats": repeats, "drop_threshold_mae_increase": threshold,
              "validation_mae_of_served_ensemble": reference,
              "features_total": len(columns), "features_dropped": len(dropped), "dropped": dropped,
              "catboost_all_features_validation": full, "catboost_reduced_features_validation": reduced,
              "validation_mae_improvement": float(improvement), "paired_test": significance,
              "top_20": importance.head(20).to_dict("records"),
              "adopt_reduced_feature_set": bool(dropped and significance["significant"])}
    result["decision"] = (
        f"Removing {len(dropped)} features improves validation MAE by {improvement:.4f} runs "
        f"({significance['z']:.1f} standard errors); the reduced set is adopted."
        if result["adopt_reduced_feature_set"] else
        f"Removing {len(dropped)} features moves validation MAE by {improvement:.4f} runs, "
        f"{significance['z']:.1f} standard errors of the paired difference and inside sampling noise. "
        f"The full feature set is retained; the ranking is still useful for knowing which features carry the signal.")
    (destination / "feature_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["features_total", "features_dropped",
                                             "catboost_all_features_validation", "catboost_reduced_features_validation",
                                             "decision"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Drop features whose mean validation MAE increase is at or below this value")
    arguments = parser.parse_args()
    run(arguments.root.resolve(), arguments.repeats, arguments.threshold)
