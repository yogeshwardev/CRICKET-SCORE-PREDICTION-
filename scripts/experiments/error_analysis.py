"""Phase 36: automated failure-pattern analysis of the largest held-out errors.

This is post-hoc diagnosis of an already-frozen model, not tuning. Anything learned
here must be validated on the validation season and claimed only on a later,
genuinely unseen season; the current test season is spent once it is read this way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cricket_ai.models import matrix, split, team_predict

# Each condition is observable in the recorded outcome or the pre-over state.
CONDITIONS = {
    "wicket_fell": lambda f: f.target_wicket == 1,
    "two_or_more_boundaries": lambda f: f.target_next_over_runs.notna() & (f.target_six + f.target_boundary >= 2),
    "extras_of_two_or_more": lambda f: f.target_extras >= 2,
    "death_overs": lambda f: f.over_number >= 16,
    "powerplay": lambda f: f.over_number <= 6,
    "new_batter_under_6_balls": lambda f: f.batter_innings_balls < 6,
    "sparse_batter_history_under_120_balls": lambda f: f.batter_career_balls < 120,
    "sparse_bowler_history_under_120_balls": lambda f: f.bowler_career_balls < 120,
    "unseen_batter_bowler_matchup": lambda f: f.batter_matchup_balls == 0,
    "required_rate_at_least_12": lambda f: f.required_run_rate >= 12,
    "second_innings_chase": lambda f: f.innings == 2,
}


def run(root: Path, top: int) -> dict:
    overs = pd.read_parquet(root / "data/processed/overs.parquet")
    report = json.loads((root / "reports/latest.json").read_text())
    bundle = joblib.load(root / "models" / report["version"] / "bundle.joblib")
    test = split(overs)["test"].copy()
    test["predicted"] = team_predict(bundle, matrix(test, bundle["features"]))
    test["error"] = test.predicted - test.target_next_over_runs
    test["absolute_error"] = test.error.abs()
    worst = test.nlargest(top, "absolute_error")

    rows = []
    for name, condition in CONDITIONS.items():
        everywhere, inside = condition(test), condition(worst)
        rows.append({"condition": name,
                     "share_of_test_overs": float(everywhere.mean()),
                     f"share_of_worst_{top}": float(inside.mean()),
                     "lift": float(inside.mean() / everywhere.mean()) if everywhere.mean() else None,
                     "mae_when_true": float(test.absolute_error[everywhere].mean()) if everywhere.any() else None,
                     "mae_when_false": float(test.absolute_error[~everywhere].mean()) if (~everywhere).any() else None,
                     "n": int(everywhere.sum())})
    frequency = pd.DataFrame(rows).sort_values("lift", ascending=False)

    destination = root / "reports/experiments"
    destination.mkdir(parents=True, exist_ok=True)
    frequency.to_csv(destination / "error_conditions.csv", index=False)
    keep = ["match_id", "date", "innings", "over_number", "phase", "batter", "bowler",
            "current_score", "current_wickets", "required_run_rate", "batter_career_balls",
            "batter_matchup_balls", "target_next_over_runs", "target_wicket", "target_extras",
            "predicted", "error", "absolute_error"]
    worst[keep].to_csv(destination / f"worst_{top}_diagnosed.csv", index=False)

    under = worst[worst.error < 0]
    result = {
        "model_version": report["version"],
        "test_overs": int(len(test)),
        "test_mae": float(test.absolute_error.mean()),
        "worst_n": top,
        "worst_mae": float(worst.absolute_error.mean()),
        "worst_error_share_of_total_absolute_error": float(worst.absolute_error.sum() / test.absolute_error.sum()),
        "under_prediction_share_of_worst": float(len(under) / len(worst)),
        "mean_actual_runs_in_worst": float(worst.target_next_over_runs.mean()),
        "mean_actual_runs_overall": float(test.target_next_over_runs.mean()),
        "conditions": frequency.to_dict("records"),
        "interpretation": [
            "Lift above one means the condition is over-represented among the largest errors.",
            "The model regresses toward the conditional mean, so extreme overs dominate the error tail by construction.",
            "Sparse-history and unseen-matchup lifts indicate where more prior data, not more model capacity, is the constraint.",
        ],
    }
    (destination / "error_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(frequency.to_string(index=False))
    print(json.dumps({k: result[k] for k in ["test_mae", "worst_mae", "under_prediction_share_of_worst",
                                             "mean_actual_runs_in_worst", "mean_actual_runs_overall"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--top", type=int, default=100)
    arguments = parser.parse_args()
    run(arguments.root.resolve(), arguments.top)
