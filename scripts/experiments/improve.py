"""Walk-forward experiment harness for improving genuine unseen-season performance.

Every arm is trained and scored on identical chronological folds with an identical
budget, so a difference between arms is a difference in the idea, not in the compute
it was given. Nothing here reads the production test season for selection: folds
validate on a season and the final untouched season is left to `crease train`.

Improvements are accepted on a paired significance test over per-over absolute error
pooled across folds, never on whichever mean happens to be lower.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from cricket_ai.evaluation import regression
from cricket_ai.features import CATEGORICAL, feature_columns
from cricket_ai.models import SEED, THREADS, matrix

# Feature families, matched by column name. "base" is whatever no other family claims.
FAMILIES = {
    "momentum": r"^(runs|wickets|boundaries|sixes)_last_\d+_overs$|^(runs|dot_rate)_last_\d+_deliveries$",
    "player": r"^(batter|non_striker|bowler)_(career|season|innings|recent|powerplay|middle|death)",
    "matchup": r"_matchup_",
    "venue_team": r"^(venue|venue_over|venue_phase|team|opposition|head_to_head)_",
    "era": r"_vs_era$|^era_|^scoring_vs_era$",
    "position": r"_position_now$|_mean_position$|_innings_seen$|_share_(top3|position)",
    "ewma": r"_ewma[\d.]+_|_last_\d+d_",
    "lags": r"^ball_minus_\d+_",
    "pressure": r"^(required_minus_current_rate|required_runs_per_wicket|balls_per_wicket_remaining|pressure_bucket)$",
}


def family_of(column: str) -> str:
    for name, pattern in FAMILIES.items():
        if re.search(pattern, column):
            return name
    return "base"


def split_for(frame: pd.DataFrame, test_season: int, strategy: str) -> dict | None:
    """Chronological partitions for one fold.

    `legacy` reproduces the shipped split: the two seasons before the test season are
    spent entirely on validation and calibration, so training stops three seasons short.
    `recent` keeps the same disjoint, strictly-ordered blocks but carves them out of the
    preceding season by date, which buys two extra seasons of training data.
    """
    seasons = sorted(frame.season.unique())
    if test_season not in seasons:
        return None
    if strategy == "legacy":
        validation, calibration = test_season-2, test_season-1
        if validation not in seasons or calibration not in seasons:
            return None
        calib = frame[frame.season == calibration]
        dates = sorted(calib.date.unique())
        middle = dates[len(dates)//2]
        parts = {"train": frame[frame.season < validation],
                 "validation": frame[frame.season == validation],
                 "probability_calibration": calib[calib.date < middle],
                 "interval_calibration": calib[calib.date >= middle],
                 "test": frame[frame.season == test_season]}
    elif strategy == "recent":
        previous = test_season-1
        if previous not in seasons:
            return None
        block = frame[frame.season == previous]
        dates = sorted(block.date.unique())
        if len(dates) < 8:
            return None
        cuts = [dates[int(len(dates)*f)] for f in (.45, .75, .90)]
        parts = {"train": frame[(frame.season < previous) | ((frame.season == previous) & (frame.date < cuts[0]))],
                 "validation": block[(block.date >= cuts[0]) & (block.date < cuts[1])],
                 "probability_calibration": block[(block.date >= cuts[1]) & (block.date < cuts[2])],
                 "interval_calibration": block[block.date >= cuts[2]],
                 "test": frame[frame.season == test_season]}
    elif strategy in ("refit", "refit_plus"):
        # Select the stopping point on a full, reliable validation season, then refit on
        # train+validation so the fitted model actually sees the recent seasons. The
        # calibration blocks stay untouched, so conformal and probability calibration
        # remain honest. `refit_plus` additionally folds in the first half of the
        # calibration season and calibrates on the second half.
        selection, calibration = test_season-2, test_season-1
        if selection not in seasons or calibration not in seasons:
            return None
        calib = frame[frame.season == calibration]
        dates = sorted(calib.date.unique())
        if len(dates) < 8:
            return None
        if strategy == "refit":
            middle = dates[len(dates)//2]
            extra = calib.iloc[:0]
            probability = calib[calib.date < middle]
            interval = calib[calib.date >= middle]
        else:
            cuts = [dates[int(len(dates)*f)] for f in (.50, .75)]
            extra = calib[calib.date < cuts[0]]
            probability = calib[(calib.date >= cuts[0]) & (calib.date < cuts[1])]
            interval = calib[calib.date >= cuts[1]]
        parts = {"train": frame[frame.season < selection],
                 "validation": frame[frame.season == selection],
                 "refit_extra": extra,
                 "probability_calibration": probability,
                 "interval_calibration": interval,
                 "test": frame[frame.season == test_season]}
    else:
        raise ValueError(f"Unknown split strategy {strategy}")
    ordered = [v for k, v in parts.items() if not (k == "refit_extra" and v.empty)]
    for earlier, later in zip(ordered[:-1], ordered[1:]):
        if earlier.empty or later.empty or earlier.date.max() >= later.date.min():
            return None
        if set(earlier.match_id) & set(later.match_id):
            return None
    return parts


def sample_weights(part: pd.DataFrame, half_life_days: float | None) -> np.ndarray | None:
    if not half_life_days:
        return None
    newest = pd.Timestamp(part.date.max())
    age = (newest-pd.to_datetime(part.date)).dt.days.to_numpy()
    return np.exp(-np.log(2)*age/half_life_days)


def fit_predict(parts: dict, columns: list[str], iterations: int, half_life: float | None, params: dict | None = None):
    settings = dict(depth=6, learning_rate=.05, l2_leaf_reg=8., random_seed=SEED, thread_count=THREADS,
                    verbose=False, allow_writing_files=False, loss_function="RMSE")
    settings.update(params or {})
    model = CatBoostRegressor(iterations=iterations, **settings)
    categorical = [c for c in CATEGORICAL if c in columns]
    started = time.perf_counter()
    model.fit(matrix(parts["train"], columns), parts["train"].target_next_over_runs.to_numpy(),
              cat_features=categorical, sample_weight=sample_weights(parts["train"], half_life),
              eval_set=(matrix(parts["validation"], columns), parts["validation"].target_next_over_runs.to_numpy()),
              early_stopping_rounds=60)
    if "refit_extra" in parts:
        # Stopping point chosen above on unseen data; now refit on it plus anything else
        # that is still strictly earlier than the calibration blocks.
        rounds = max(50, model.get_best_iteration() or model.tree_count_)
        combined = pd.concat([parts["train"], parts["validation"], parts["refit_extra"]])
        model = CatBoostRegressor(iterations=rounds, **settings)
        model.fit(matrix(combined, columns), combined.target_next_over_runs.to_numpy(),
                  cat_features=categorical, sample_weight=sample_weights(combined, half_life))
    return np.maximum(0, model.predict(matrix(parts["test"], columns))), time.perf_counter()-started, model


def paired(reference: np.ndarray, challenger: np.ndarray, truth: np.ndarray, sigmas: float = 2.0) -> dict:
    difference = np.abs(truth-reference)-np.abs(truth-challenger)
    mean = float(difference.mean())
    error = float(difference.std(ddof=1)/np.sqrt(len(difference))) if len(difference) > 1 else float("inf")
    return {"mae_improvement": mean, "standard_error": error,
            "z": float(mean/error) if error > 0 else 0.0, "significant": bool(mean > sigmas*error)}


def run_arm(frame, columns, seasons, strategy, iterations, half_life, params=None):
    predictions, truths, per_season, seconds = [], [], {}, 0.
    for season in seasons:
        parts = split_for(frame, season, strategy)
        if parts is None:
            continue
        prediction, elapsed, _ = fit_predict(parts, columns, iterations, half_life, params)
        truth = parts["test"].target_next_over_runs.to_numpy()
        per_season[str(season)] = {**regression(truth, prediction), "n": int(len(truth)),
                                   "train_overs": int(len(parts["train"]))+int(len(parts.get("refit_extra", [])))
                                   + (int(len(parts["validation"])) if "refit_extra" in parts else 0),
                                   "train_through": str(max(parts["train"].date.max(),
                                                            parts["validation"].date.max() if "refit_extra" in parts else "",
                                                            parts["refit_extra"].date.max() if len(parts.get("refit_extra", [])) else ""))}
        predictions.append(prediction)
        truths.append(truth)
        seconds += elapsed
    if not predictions:
        return None
    pooled_prediction, pooled_truth = np.concatenate(predictions), np.concatenate(truths)
    maes = [v["mae"] for v in per_season.values()]
    return {"pooled": regression(pooled_truth, pooled_prediction), "per_season": per_season,
            "mean_season_mae": float(np.mean(maes)), "season_mae_spread": float(np.std(maes)),
            "worst_season_mae": float(np.max(maes)), "features": len(columns),
            "fit_seconds": round(seconds, 1),
            "_prediction": pooled_prediction, "_truth": pooled_truth}


def report(results: dict, reference: str, destination: Path, name: str):
    base = results[reference]
    table = {}
    for arm, value in results.items():
        entry = {k: v for k, v in value.items() if not k.startswith("_")}
        if arm != reference:
            entry["vs_" + reference] = paired(base["_prediction"], value["_prediction"], base["_truth"])
        table[arm] = entry
    ranked = sorted(table.items(), key=lambda kv: kv[1]["pooled"]["mae"])
    print(f"\n{'arm':<26} {'pooled MAE':>10} {'mean/season':>12} {'spread':>8} {'feat':>5}  verdict")
    for arm, entry in ranked:
        verdict = "reference" if arm == reference else (
            f"better by {entry['vs_'+reference]['mae_improvement']:+.4f} ({entry['vs_'+reference]['z']:.1f} sd)"
            if entry["vs_" + reference]["significant"] else
            f"no evidence ({entry['vs_'+reference]['mae_improvement']:+.4f}, {entry['vs_'+reference]['z']:.1f} sd)")
        print(f"{arm:<26} {entry['pooled']['mae']:>10.4f} {entry['mean_season_mae']:>12.4f} "
              f"{entry['season_mae_spread']:>8.4f} {entry['features']:>5}  {verdict}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{name}.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--mode", choices=["split", "ablation", "weights", "families"], required=True)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--strategy", default="recent")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    frame = pd.read_parquet(root / "data/processed/overs.parquet")
    columns = feature_columns(frame)
    destination = root / "reports/experiments"
    grouped = {}
    for column in columns:
        grouped.setdefault(family_of(column), []).append(column)
    print("feature families:", {k: len(v) for k, v in sorted(grouped.items())})

    if arguments.mode == "split":
        results = {}
        for strategy in ["legacy", "recent", "refit", "refit_plus"]:
            print(f"\n=== split strategy: {strategy} ===", flush=True)
            outcome = run_arm(frame, columns, arguments.seasons, strategy, arguments.iterations, None)
            if outcome:
                results[strategy] = outcome
        report(results, "legacy", destination, "walkforward_split")

    elif arguments.mode == "ablation":
        # Cumulative families, in the order a practitioner would add them.
        order = ["base", "momentum", "player", "matchup", "venue_team", "pressure", "era", "position", "ewma", "lags"]
        results, active = {}, []
        for family in order:
            if family not in grouped:
                continue
            active = active + grouped[family]
            label = f"+{family}" if results else family
            print(f"\n=== {label} ({len(active)} features) ===", flush=True)
            outcome = run_arm(frame, active, arguments.seasons, arguments.strategy, arguments.iterations, None)
            if outcome:
                results[label] = outcome
        report(results, list(results)[0], destination, "walkforward_ablation")

    elif arguments.mode == "families":
        # Decision-relevant comparison: do the newly engineered families earn their place,
        # and which one carries the gain? Leave-one-out attributes credit without the cost
        # of a full cumulative sweep.
        established = ["base", "momentum", "player", "matchup", "venue_team"]
        added = [f for f in ["pressure", "era", "position", "ewma", "lags"] if f in grouped]
        old_columns = [c for f in established for c in grouped.get(f, [])]
        print(f"\n=== established_only ({len(old_columns)} features) ===", flush=True)
        results = {"established_only": run_arm(frame, old_columns, arguments.seasons, arguments.strategy,
                                               arguments.iterations, None)}
        print(f"\n=== all_families ({len(columns)} features) ===", flush=True)
        results["all_families"] = run_arm(frame, columns, arguments.seasons, arguments.strategy,
                                          arguments.iterations, None)
        for family in added:
            subset = [c for c in columns if c not in grouped[family]]
            print(f"\n=== without_{family} ({len(subset)} features) ===", flush=True)
            outcome = run_arm(frame, subset, arguments.seasons, arguments.strategy, arguments.iterations, None)
            if outcome:
                results[f"without_{family}"] = outcome
        report({k: v for k, v in results.items() if v}, "established_only", destination, "walkforward_families")

    elif arguments.mode == "weights":
        results = {}
        for half_life in [None, 1460, 730, 365]:
            label = "equal" if half_life is None else f"half_life_{half_life}d"
            print(f"\n=== sample weighting: {label} ===", flush=True)
            outcome = run_arm(frame, columns, arguments.seasons, arguments.strategy, arguments.iterations, half_life)
            if outcome:
                results[label] = outcome
        report(results, "equal", destination, "walkforward_weights")


if __name__ == "__main__":
    main()
