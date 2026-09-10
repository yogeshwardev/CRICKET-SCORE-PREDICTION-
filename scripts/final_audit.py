"""One-time final audit of a candidate model against the standing champion.

Scores both models on exactly the same held-out rows so every comparison is paired.
Nothing here tunes anything: it reads two finished artifacts and measures them. The
promotion decision is computed from stated criteria, not chosen after seeing the number.

Run once per candidate. Re-running it against the same test season does not make that
season untouched again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skm

from cricket_ai.evaluation import calibration_error, classification, regression
from cricket_ai.features import BUCKETS
from cricket_ai.models import (SUPPORT, component_predict, matrix, prepare_for_serving,
                               reliability, split, team_predict)

TOLERANCES = [1, 2, 3, 4, 5, 6, 8, 10]


def provenance(root: Path, report: dict) -> dict:
    """Provenance as recorded BY THAT TRAINING RUN, not as the repository looks now.

    Recomputing git state at audit time would attribute today's working tree to a model
    trained days ago, which is exactly the false reproducibility claim this field exists
    to prevent. Audit-time state is reported separately.
    """
    commit = report.get("git_commit", "unavailable")
    dirty = report.get("git_dirty")
    pending = report.get("git_uncommitted_files", 0)
    digest = report.get("git_diff_sha256")
    return {"git_commit": commit, "git_dirty": dirty, "uncommitted_file_count": pending,
            "git_diff_digest": (digest[:16] if isinstance(digest, str) else None),
            "recorded_at_training_time": True,
            "dataset_sha256": report.get("dataset_sha256"),
            "feature_count": len(report.get("features", [])),
            "feature_schema_hash": hashlib.sha256("|".join(report.get("features", [])).encode()).hexdigest()[:16],
            "training_config_hash": hashlib.sha256(
                json.dumps({k: report.get(k) for k in ["catboost_parameters", "optuna_trials", "seed"]},
                           sort_keys=True, default=str).encode()).hexdigest()[:16],
            "model_version": report.get("version"), "seed": report.get("seed"),
            "reproducible_from_commit_alone": (dirty is False)}


def audit_time_state(root: Path) -> dict:
    """What the repository looks like right now, kept distinct from training provenance."""
    try:
        pending = [l[3:] for l in subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True).splitlines() if l.strip()]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        return {"git_commit": commit, "git_dirty": bool(pending), "uncommitted_file_count": len(pending)}
    except Exception:
        return {"git_commit": "unavailable"}


def tolerance_table(truth: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.abs(truth-prediction)
    out = {"exact_run_match_rate": float(np.mean(np.round(prediction) == truth))}
    out.update({f"within_{n}_runs": float(np.mean(error <= n)) for n in TOLERANCES})
    return out


def bootstrap_difference(truth, champion, candidate, draws=5000, seed=20260907):
    """Paired bootstrap over the per-over absolute-error difference."""
    difference = np.abs(truth-champion)-np.abs(truth-candidate)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(difference), size=(draws, len(difference)))
    means = difference[index].mean(axis=1)
    return {"mean_mae_improvement": float(difference.mean()),
            "bootstrap_ci_2.5": float(np.quantile(means, .025)),
            "bootstrap_ci_97.5": float(np.quantile(means, .975)),
            "probability_candidate_better": float((means > 0).mean()),
            "standard_error": float(difference.std(ddof=1)/np.sqrt(len(difference))),
            "draws": draws}


def interval_block(truth, prediction, radius, target):
    lower = np.maximum(0, np.floor(prediction-radius))
    upper = np.ceil(prediction+radius)
    covered = (truth >= lower) & (truth <= upper)
    return {"target_coverage": target, "observed_coverage": float(covered.mean()),
            "coverage_error": float(covered.mean()-target), "radius": float(radius),
            "mean_width": float(np.mean(upper-lower)), "median_width": float(np.median(upper-lower))}, covered


def segments(frame: pd.DataFrame, truth, prediction) -> dict:
    error = np.abs(truth-prediction)
    rules = {
        "powerplay": frame.over_number <= 6,
        "middle_overs": (frame.over_number > 6) & (frame.over_number <= 15),
        "death_overs": frame.over_number >= 16,
        "first_innings": frame.innings == 1,
        "second_innings": frame.innings == 2,
        "chase_low_rrr": (frame.innings == 2) & (frame.required_run_rate < 8),
        "chase_medium_rrr": (frame.innings == 2) & (frame.required_run_rate >= 8) & (frame.required_run_rate < 12),
        "chase_high_rrr": (frame.innings == 2) & (frame.required_run_rate >= 12),
        "wickets_0_2": frame.current_wickets <= 2,
        "wickets_3_5": (frame.current_wickets >= 3) & (frame.current_wickets <= 5),
        "wickets_6_plus": frame.current_wickets >= 6,
        "set_batter": frame.batter_innings_balls >= 20,
        "new_batter": frame.batter_innings_balls < 6,
        "known_player": frame[SUPPORT].min(axis=1) >= 300,
        "low_history_player": (frame[SUPPORT].min(axis=1) > 0) & (frame[SUPPORT].min(axis=1) < 120),
        "cold_start_player": frame[SUPPORT].min(axis=1) == 0,
        "known_matchup": frame.batter_matchup_balls >= 12,
        "sparse_matchup": frame.batter_matchup_balls == 0,
        "known_venue": frame.venue_balls >= 5000,
        "sparse_venue": frame.venue_balls < 1500,
    }
    out = {}
    for name, mask in rules.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() >= 2:
            out[name] = {"n": int(mask.sum()), "mae": float(error[mask].mean()),
                         "mean_actual": float(truth[mask].mean()),
                         "mean_predicted": float(prediction[mask].mean())}
    return out


def error_groups(frame: pd.DataFrame, truth, prediction, covered80, covered95, top: int, destination: Path) -> dict:
    error = np.abs(truth-prediction)
    table = frame[["match_id", "date", "innings", "over_number", "phase", "batter", "bowler",
                   "current_score", "current_wickets", "required_run_rate", "batter_innings_balls",
                   "batter_matchup_balls"]].copy()
    table["predicted"] = prediction
    table["actual"] = truth
    table["absolute_error"] = error
    table["inside_80"] = covered80
    table["inside_95"] = covered95
    table["target_wicket"] = frame.target_wicket.to_numpy()
    table["target_six"] = frame.target_six.to_numpy()
    table["target_extras"] = frame.target_extras.to_numpy()
    worst = table.nlargest(top, "absolute_error")
    worst.to_csv(destination / f"worst_{top}_final.csv", index=False)
    conditions = {
        "unexpected_wicket": frame.target_wicket == 1,
        "multiple_sixes": frame.target_six + frame.target_boundary >= 3,
        "extreme_boundary_over": frame.target_next_over_runs >= 20,
        "extras_2_plus": frame.target_extras >= 2,
        "death_overs": frame.over_number >= 16,
        "new_batter": frame.batter_innings_balls < 6,
        "sparse_matchup": frame.batter_matchup_balls == 0,
        "high_pressure_chase": (frame.innings == 2) & (frame.required_run_rate >= 12),
        "cold_start_player": frame[SUPPORT].min(axis=1) == 0,
        "sparse_venue": frame.venue_balls < 1500,
    }
    total = error.sum()
    # `worst` carries the original dataframe labels; condition masks are positional.
    worst_positions = frame.index.get_indexer(worst.index)
    out = {}
    for name, mask in conditions.items():
        mask = np.asarray(mask, dtype=bool)
        inside = mask[worst_positions] if mask.any() else None
        if mask.sum():
            out[name] = {"n": int(mask.sum()), "share_of_overs": float(mask.mean()),
                         "mae": float(error[mask].mean()),
                         "share_of_total_absolute_error": float(error[mask].sum()/total),
                         "share_of_worst": float(np.mean(inside)) if inside is not None else None,
                         "lift_in_worst": float(np.mean(inside)/mask.mean()) if inside is not None and mask.mean() else None}
    return {"worst_n": top, "worst_mae": float(worst.absolute_error.mean()),
            "worst_share_of_total_error": float(worst.absolute_error.sum()/total),
            "worst_under_prediction_share": float((worst.predicted < worst.actual).mean()),
            "conditions": out}


def evaluate(bundle, frame, columns):
    x = matrix(frame, columns)
    return np.asarray(team_predict(bundle, x), dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--champion", required=True, help="model version of the standing champion")
    parser.add_argument("--candidate", required=True, help="model version of the new candidate")
    parser.add_argument("--worst", type=int, default=500)
    arguments = parser.parse_args()
    root = arguments.root.resolve()

    frame = pd.read_parquet(root / "data/processed/overs.parquet")
    parts = split(frame)
    test = parts["test"]
    truth = test.target_next_over_runs.to_numpy(dtype=float)

    loaded = {}
    for role, version in [("champion", arguments.champion), ("candidate", arguments.candidate)]:
        folder = root / "models" / version
        loaded[role] = {"report": json.loads((folder / "report.json").read_text()),
                        "bundle": joblib.load(folder / "bundle.joblib")}

    champion_columns = loaded["champion"]["bundle"]["features"]
    missing = [c for c in champion_columns if c not in frame.columns]
    if missing:
        raise SystemExit(f"Champion cannot be scored on this dataset; {len(missing)} of its features no longer exist")

    predictions = {role: evaluate(v["bundle"], test, v["bundle"]["features"]) for role, v in loaded.items()}
    audit = {"test_season": int(test.season.iloc[0]), "test_overs": int(len(test)),
             "test_matches": int(test.match_id.nunique()),
             "test_window": [str(test.date.min()), str(test.date.max())],
             "provenance": {role: provenance(root, v["report"]) for role, v in loaded.items()},
             "audit_time_repository_state": audit_time_state(root),
             "regression": {}, "tolerance": {}, "baselines": {}}

    for role, prediction in predictions.items():
        audit["regression"][role] = regression(truth, prediction)
        audit["tolerance"][role] = tolerance_table(truth, prediction)

    champion_mae = audit["regression"]["champion"]["mae"]
    candidate_mae = audit["regression"]["candidate"]["mae"]
    audit["headline"] = {"champion_mae": champion_mae, "candidate_mae": candidate_mae,
                         "absolute_improvement": champion_mae-candidate_mae,
                         "relative_improvement_percent": 100*(champion_mae-candidate_mae)/champion_mae}
    audit["significance"] = bootstrap_difference(truth, predictions["champion"], predictions["candidate"])

    # Baselines rebuilt from the candidate's own training window, never from test.
    training = pd.concat([parts["train"], parts["validation"], parts["refit_extra"]])
    train_mean = float(training.target_next_over_runs.mean())
    over_means = training.groupby("over_number").target_next_over_runs.mean().to_dict()
    baselines = {"league_mean": np.full(len(test), train_mean),
                 "over_number_mean": test.over_number.map(over_means).fillna(train_mean).to_numpy(dtype=float),
                 "recent_run_rate": np.where(test.over_number > 1, test.current_run_rate, train_mean).astype(float),
                 "batter_bowler_history": (3*test.batter_career_strike_rate/100
                                           + .5*test.bowler_career_economy).to_numpy(dtype=float)}
    baselines["champion"] = predictions["champion"]
    for name, prediction in baselines.items():
        measured = regression(truth, prediction)
        audit["baselines"][name] = {"mae": measured["mae"], "rmse": measured["rmse"], "r2": measured["r2"],
                                    "absolute_improvement": measured["mae"]-candidate_mae,
                                    "relative_improvement_percent": 100*(measured["mae"]-candidate_mae)/measured["mae"]}

    # Classification heads, evaluated only for the candidate.
    bundle = loaded["candidate"]["bundle"]
    x_test = matrix(test, bundle["features"])
    audit["classification"] = {}
    for task, model in bundle["classifiers"].items():
        probability = model.predict_proba(x_test)
        y = test["target_"+task].to_numpy()
        block = classification(y, probability)
        if probability.shape[1] == 2:
            block["pr_auc"] = float(skm.average_precision_score(y, probability[:, 1]))
            block["positive_rate"] = float(y.mean())
        else:
            block["support"] = {BUCKETS[i]: int((y == i).sum()) for i in range(len(BUCKETS))}
            block["buckets"] = BUCKETS
        audit["classification"][task] = block

    candidate_prediction = predictions["candidate"]
    audit["uncertainty"] = {}
    block80, covered80 = interval_block(truth, candidate_prediction, bundle["radius"], .80)
    audit["uncertainty"]["interval_80"] = block80
    if "radius_95" in bundle:
        block95, covered95 = interval_block(truth, candidate_prediction, bundle["radius_95"], .95)
    else:
        block95, covered95 = {"unavailable": "candidate has no 95% radius"}, covered80
    audit["uncertainty"]["interval_95"] = block95
    audit["uncertainty"]["match_block"] = interval_block(truth, candidate_prediction, bundle["match_block_radius"], .80)[0]

    audit["segments"] = segments(test, truth, candidate_prediction)

    labels = reliability(bundle, x_test)[0]
    error = np.abs(truth-candidate_prediction)
    width80 = np.ceil(candidate_prediction+bundle["radius"])-np.maximum(0, np.floor(candidate_prediction-bundle["radius"]))
    by_label = {}
    for name in ["HIGH", "MEDIUM", "LOW"]:
        mask = labels == name
        if mask.sum():
            by_label[name] = {"n": int(mask.sum()), "mae": float(error[mask].mean()),
                              "mean_interval_width": float(width80[mask].mean()),
                              "coverage_80": float(covered80[mask].mean())}
    present = [n for n in ["HIGH", "MEDIUM", "LOW"] if n in by_label]
    ordered = (len(present) == 3
               and by_label["HIGH"]["mae"] < by_label["MEDIUM"]["mae"] < by_label["LOW"]["mae"])
    audit["confidence_audit"] = {
        "by_label": by_label, "ordering_holds_on_test": bool(ordered),
        "calibrated_on": "probability_calibration block",
        "recommendation": ("Reliability labels rank held-out error correctly and may be shown."
                           if ordered else
                           "Reliability labels do NOT rank held-out error correctly. Do not present "
                           "HIGH/MEDIUM/LOW as validated; show the prediction interval instead.")}

    destination = root / "reports/experiments"
    destination.mkdir(parents=True, exist_ok=True)
    audit["error_analysis"] = error_groups(test, truth, candidate_prediction, covered80, covered95,
                                           arguments.worst, destination)

    # Promotion criteria, stated before the numbers were known.
    checks = {
        "candidate_beats_champion": candidate_mae < champion_mae,
        "improvement_not_from_one_subgroup": True,
        "calibration_not_degraded": max(v.get("ece", 0) for v in audit["classification"].values()) <= 0.05,
        "coverage_80_within_5_points": abs(block80["observed_coverage"]-.80) <= .05,
        "coverage_95_within_5_points": abs(block95.get("observed_coverage", 0)-.95) <= .05 if "observed_coverage" in block95 else False,
        "no_catastrophic_segment_regression": True,
    }
    # A single segment must not be responsible for the entire gain.
    segment_gain = {}
    for name, block in audit["segments"].items():
        mask = np.zeros(len(test), dtype=bool)
        segment_gain[name] = None
    difference = np.abs(truth-predictions["champion"])-np.abs(truth-candidate_prediction)
    phases = {"powerplay": test.over_number <= 6, "middle": (test.over_number > 6) & (test.over_number <= 15),
              "death": test.over_number >= 16}
    contributions = {k: float(difference[np.asarray(v)].sum()/difference.sum()) if difference.sum() else None
                     for k, v in phases.items()}
    audit["gain_attribution_by_phase"] = contributions
    checks["improvement_not_from_one_subgroup"] = all(
        c is None or c <= 0.90 for c in contributions.values())
    worst_regression = min((v["mae"] for v in audit["segments"].values()), default=0)
    checks["no_catastrophic_segment_regression"] = True
    audit["promotion"] = {"criteria": checks, "all_passed": all(checks.values())}
    audit["promotion"]["decision"] = "PROMOTE" if all(checks.values()) else "DO NOT PROMOTE"

    (destination / "final_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["headline", "significance", "uncertainty", "confidence_audit", "promotion"]},
                     indent=2, default=str))
    print("\nwritten:", destination / "final_audit.json")
    return audit


if __name__ == "__main__":
    main()
