"""Training, selection, calibration, immutable model artifacts, and inference."""
from __future__ import annotations
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import optuna

from .features import CATEGORICAL, BUCKETS, feature_columns
from .evaluation import regression, classification, conformal_radius

SEED = 20260907
COMPONENTS = ["batter_runs", "non_striker_runs", "extras", "other_batters", "bowler_conceded"]
TASKS = ["wicket", "boundary", "six", "ten_plus", "run_bucket"]


def split(frame):
    years = sorted(frame.season.unique())
    if len(years) < 5:
        raise ValueError("At least five seasons required for independent temporal partitions")
    test_year, calibration_year, validation_year = years[-1], years[-2], years[-3]
    calibration = frame[frame.season == calibration_year]
    dates = sorted(calibration.date.unique())
    midpoint = dates[len(dates)//2]
    parts = {"train": frame[frame.season < validation_year],
             "validation": frame[frame.season == validation_year],
             "probability_calibration": calibration[calibration.date < midpoint],
             "interval_calibration": calibration[calibration.date >= midpoint],
             "test": frame[frame.season == test_year]}
    for a, b in zip(list(parts.values())[:-1], list(parts.values())[1:]):
        if a.empty or b.empty or a.date.max() >= b.date.min() or set(a.match_id) & set(b.match_id):
            raise ValueError("Invalid chronological partitions")
    return parts


def cat_reg(**kwargs):
    return CatBoostRegressor(loss_function="RMSE", random_seed=SEED, thread_count=4,
                             verbose=False, allow_writing_files=False, **kwargs)


def matrix(frame, columns):
    x = frame[columns].copy()
    for c in CATEGORICAL:
        x[c] = x[c].fillna("unknown").astype(str)
    return x


class EncodedRegressor:
    def __init__(self, model):
        self.model = model
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def transform(self, x):
        numeric = x.drop(columns=CATEGORICAL).to_numpy(dtype=float)
        return np.column_stack([numeric, self.encoder.transform(x[CATEGORICAL])])

    def fit(self, x, y):
        self.encoder.fit(x[CATEGORICAL])
        self.model.fit(self.transform(x), y)
        return self

    def predict(self, x):
        return self.model.predict(self.transform(x))


class CalibratedTask:
    def __init__(self, model, calibrator):
        self.model, self.calibrator = model, calibrator

    def predict_proba(self, x):
        p = np.clip(self.model.predict_proba(x), 1e-9, 1)
        return self.calibrator.predict_proba(np.log(p))


def team_predict(bundle, x):
    return np.maximum(0, sum(w * bundle["candidates"][name].predict(x)
                             for name, w in bundle["weights"].items() if w > 1e-8))


def component_predict(bundle, x):
    values = np.maximum(0, bundle["components"].predict(x))
    if bundle.get("batter_mode") == "two_stage":
        balls = np.maximum(0, bundle["balls"].predict(x))
        rates = np.maximum(0, bundle["rates"].predict(x))
        values[:, :2] = balls*rates
    # Explicit replacement-batter component prevents losing runs after a wicket.
    total = team_predict(bundle, x)
    denom = values[:, :4].sum(axis=1)
    values[:, :4] *= (total/np.maximum(denom, 1e-9))[:, None]
    values[denom <= 1e-9, 3] = total[denom <= 1e-9]
    values[:, 4] = np.minimum(values[:, 4], total)
    return values


def predict(bundle, features: dict, explain=True):
    started = time.perf_counter()
    x = matrix(pd.DataFrame([features]), bundle["features"])
    total = float(team_predict(bundle, x)[0])
    components = component_predict(bundle, x)[0]
    probabilities = {t: model.predict_proba(x)[0] for t, model in bundle["classifiers"].items()}
    p = probabilities["run_bucket"]
    radius = bundle["radius"]
    support = min(features["batter_career_balls"], features["non_striker_career_balls"], features["bowler_career_balls"])
    response = {"expected_runs": total, "lower_80": max(0, int(np.floor(total-radius))),
                "upper_80": int(np.ceil(total+radius)), "run_bucket": BUCKETS[int(p.argmax())],
                "run_bucket_probability": float(p.max()), "distribution": dict(zip(BUCKETS, p.tolist())),
                "batter_expected_runs": float(components[0]), "non_striker_expected_runs": float(components[1]),
                "extras_expected": float(components[2]), "other_batters_expected_runs": float(components[3]),
                "bowler_expected_conceded": float(components[4]),
                **{t+"_probability": float(probabilities[t][1]) for t in TASKS[:-1]},
                "confidence": "LOW" if support < 24 else "MEDIUM",
                "reliability_reason": "Sparse player history" if support < 24 else "Historical support; future coverage is not guaranteed",
                "interval_method": "80% match-block split conformal; exchangeability required",
                "observed_test_coverage": bundle["test_coverage"], "model_version": bundle["version"],
                "explanations": []}
    if explain:
        # CatBoost's native exact TreeSHAP requires no third-party SHAP runtime.
        shap = bundle["candidates"]["catboost"].get_feature_importance(Pool(x, cat_features=CATEGORICAL), type="ShapValues")[0]
        response["explanation_scope"] = "CatBoost component before ensemble/clipping; contributions sum with base value"
        response["explanation_base_value"] = float(shap[-1])
        response["explanation_prediction"] = float(bundle["candidates"]["catboost"].predict(x)[0])
        response["explanations"] = [{"feature": bundle["features"][i], "contribution": float(shap[i])}
                                    for i in np.argsort(np.abs(shap[:-1]))[::-1][:10]]
    response["latency_ms"] = (time.perf_counter()-started)*1000
    return response


def train(root: Path, trials=6, iterations=400):
    frame = pd.read_parquet(root / "data/processed/overs.parquet")
    parts = split(frame)
    columns = feature_columns(frame)
    xs = {k: matrix(v, columns) for k, v in parts.items()}
    ys = {k: v.target_next_over_runs.to_numpy() for k, v in parts.items()}
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / "models" / version
    destination.mkdir(parents=True, exist_ok=False)
    best_models = {}
    def objective(trial):
        params = {"depth": trial.suggest_int("depth", 4, 7),
                  "learning_rate": trial.suggest_float("learning_rate", .025, .12, log=True),
                  "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3, 30, log=True)}
        model = cat_reg(iterations=iterations, **params)
        model.fit(xs["train"], ys["train"], cat_features=CATEGORICAL,
                  eval_set=(xs["validation"], ys["validation"]), early_stopping_rounds=50)
        best_models[trial.number] = model
        return regression(ys["validation"], model.predict(xs["validation"]))["mae"]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    candidates = {"catboost": best_models[study.best_trial.number]}
    alternatives = {
        "random_forest": RandomForestRegressor(n_estimators=160, max_depth=14, min_samples_leaf=25, n_jobs=4, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15, l2_regularization=15,
                                                                 early_stopping=False, random_state=SEED)}
    for name, model in alternatives.items():
        print(f"Training {name}", flush=True)
        candidates[name] = EncodedRegressor(model).fit(xs["train"], ys["train"])
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    for name, model in {
        "lightgbm": LGBMRegressor(n_estimators=250, num_leaves=15, learning_rate=.035, reg_lambda=15, verbosity=-1, n_jobs=4, random_state=SEED),
        "xgboost": XGBRegressor(n_estimators=250, max_depth=4, learning_rate=.035, reg_lambda=15, n_jobs=4, random_state=SEED)
    }.items():
        print(f"Training {name}", flush=True)
        candidates[name] = EncodedRegressor(model).fit(xs["train"], ys["train"])
    predictions = np.column_stack([m.predict(xs["validation"]) for m in candidates.values()])
    fitted = minimize(lambda w: np.mean(np.abs(ys["validation"] - predictions @ w)),
                      np.ones(len(candidates))/len(candidates), method="SLSQP",
                      bounds=[(0, 1)]*len(candidates), constraints={"type": "eq", "fun": lambda w: w.sum()-1})
    val = {name: regression(ys["validation"], predictions[:, i]) for i, name in enumerate(candidates)}
    single = min(val, key=lambda k: val[k]["mae"])
    weights = {k: float(k == single) for k in candidates}
    if fitted.success and regression(ys["validation"], predictions @ fitted.x)["mae"] < val[single]["mae"]:
        weights = dict(zip(candidates, fitted.x.tolist()))
    print("Training batter, extras and bowler models", flush=True)
    params = dict(study.best_params, iterations=max(100, candidates["catboost"].tree_count_))
    components = CatBoostRegressor(loss_function="MultiRMSE", random_seed=SEED, thread_count=4,
                                   verbose=False, allow_writing_files=False, **params)
    component_y = lambda key: parts[key][["target_"+c for c in COMPONENTS]]
    components.fit(xs["train"], component_y("train"), cat_features=CATEGORICAL)
    balls = CatBoostRegressor(loss_function="MultiRMSE", random_seed=SEED, thread_count=4, verbose=False, allow_writing_files=False, **params)
    balls_y = parts["train"][["target_batter_balls", "target_non_striker_balls"]].to_numpy()
    balls.fit(xs["train"], balls_y, cat_features=CATEGORICAL)
    rates = CatBoostRegressor(loss_function="MultiRMSE", random_seed=SEED, thread_count=4, verbose=False, allow_writing_files=False, **params)
    rates.fit(xs["train"], component_y("train").to_numpy()[:, :2]/np.maximum(balls_y, 1), cat_features=CATEGORICAL)
    # Compare reconciled final values, not an unrelated intermediate objective.
    bundle = dict(version=version, features=columns, candidates=candidates, weights=weights,
                  components=components, balls=balls, rates=rates, batter_mode="direct")
    direct_mae = np.mean(np.abs(component_y("validation").to_numpy()[:, :2]-component_predict(bundle, xs["validation"])[:, :2]))
    bundle["batter_mode"] = "two_stage"
    staged_mae = np.mean(np.abs(component_y("validation").to_numpy()[:, :2]-component_predict(bundle, xs["validation"])[:, :2]))
    bundle["batter_mode"] = "two_stage" if staged_mae < direct_mae else "direct"
    classifiers = {}
    for task in TASKS:
        print(f"Training calibrated {task} classifier", flush=True)
        model = CatBoostClassifier(loss_function="MultiClass" if task == "run_bucket" else "Logloss",
                                    random_seed=SEED, thread_count=4, verbose=False,
                                    allow_writing_files=False, **params)
        model.fit(xs["train"], parts["train"]["target_"+task], cat_features=CATEGORICAL,
                  eval_set=(xs["validation"], parts["validation"]["target_"+task]), early_stopping_rounds=40)
        probabilities = model.predict_proba(xs["probability_calibration"])
        calibrator = LogisticRegression(C=1., max_iter=2000, random_state=SEED)
        calibrator.fit(np.log(np.clip(probabilities, 1e-9, 1)), parts["probability_calibration"]["target_"+task])
        if list(calibrator.classes_) != list(range(6 if task == "run_bucket" else 2)):
            raise ValueError("Missing calibration class; need more calibration matches")
        classifiers[task] = CalibratedTask(model, calibrator)
    bundle["classifiers"] = classifiers
    interval_predictions = team_predict(bundle, xs["interval_calibration"])
    bundle["radius"] = conformal_radius(ys["interval_calibration"], interval_predictions, parts["interval_calibration"].match_id)
    # All choices frozen above. First and only final-test prediction follows.
    print("Choices frozen. Evaluating untouched final season.", flush=True)
    test_prediction = team_predict(bundle, xs["test"])
    lower = np.maximum(0, np.floor(test_prediction-bundle["radius"]))
    upper = np.ceil(test_prediction+bundle["radius"])
    covered = (ys["test"] >= lower) & (ys["test"] <= upper)
    bundle["test_coverage"] = float(covered.mean())
    bundle["trained_through"] = str(parts["train"].date.max())
    bundle["evaluated_through"] = str(parts["test"].date.max())
    train_mean = float(ys["train"].mean())
    over_means = parts["train"].groupby("over_number").target_next_over_runs.mean().to_dict()
    def baselines(part):
        return {"league_mean": np.full(len(part), train_mean),
                "over_number_mean": part.over_number.map(over_means).fillna(train_mean).to_numpy(),
                "recent_run_rate": np.where(part.over_number > 1, part.current_run_rate, train_mean),
                "batter_bowler_history": (3*part.batter_career_strike_rate/100 + .5*part.bowler_career_economy).to_numpy()}
    baseline_report = {key: {name: regression(ys[key], p) for name, p in baselines(parts[key]).items()} for key in ["validation", "test"]}
    final_val = regression(ys["validation"], team_predict(bundle, xs["validation"]))
    beats_baselines = all(final_val["mae"] < m["mae"] for m in baseline_report["validation"].values())
    bundle["promotion_eligible"] = beats_baselines
    report = {"version": version, "seed": SEED, "dataset_sha256": hashlib.sha256((root / "data/processed/overs.parquet").read_bytes()).hexdigest(),
              "matches": int(frame.match_id.nunique()), "overs": len(frame),
              "partitions": {k: {"start": str(v.date.min()), "end": str(v.date.max()), "matches": int(v.match_id.nunique()), "overs": len(v)} for k, v in parts.items()},
              "features": columns, "catboost_parameters": params, "optuna_trials": trials,
              "candidate_validation": val, "selected_validation": final_val, "ensemble_weights": weights,
              "batter_comparison_validation_mae": {"direct": float(direct_mae), "two_stage": float(staged_mae)},
              "batter_mode": bundle["batter_mode"], "baseline_comparison": baseline_report,
              "test_regression": regression(ys["test"], test_prediction),
              "candidate_test": {name: regression(ys["test"], model.predict(xs["test"])) for name, model in candidates.items()},
              "classification": {t: classification(parts["test"]["target_"+t], m.predict_proba(xs["test"])) for t, m in classifiers.items()},
              "uncertainty": {"method": "match-block split conformal", "nominal_simultaneous_match_coverage": .8,
                              "observed_over_coverage": float(covered.mean()), "mean_width": float(np.mean(upper-lower)),
                              "observed_simultaneous_match_coverage": float(pd.DataFrame({"match": parts["test"].match_id.to_numpy(), "covered": covered}).groupby("match").covered.all().mean()),
                              "radius": bundle["radius"], "caveat": "Finite-sample validity assumes exchangeable matches. Temporal distribution shift can invalidate this assumption."},
              "promotion_eligible": beats_baselines, "segments": {}}
    for col in ["phase", "innings", "venue"]:
        for value in parts["test"][col].unique():
            mask = (parts["test"][col] == value).to_numpy()
            report["segments"][f"{col}={value}"] = {"n": int(mask.sum()), **regression(ys["test"][mask], test_prediction[mask])}
    for name, mask in {"new_batter": parts["test"].batter_innings_balls < 6,
                       "set_batter": parts["test"].batter_innings_balls >= 20,
                       "unseen_matchup": parts["test"].batter_matchup_balls == 0,
                       "known_matchup": parts["test"].batter_matchup_balls > 0,
                       "high_required_rate": parts["test"].required_run_rate >= 12,
                       "low_required_rate_chase": (parts["test"].required_run_rate < 8) & (parts["test"].innings == 2)}.items():
        mask = mask.to_numpy()
        if mask.sum() > 1:
            report["segments"][name] = {"n": int(mask.sum()), **regression(ys["test"][mask], test_prediction[mask])}
    comp = component_predict(bundle, xs["test"])
    report["component_test"] = {c: regression(parts["test"]["target_"+c], comp[:, i]) for i, c in enumerate(COMPONENTS)}
    report["limitations"] = ["Pace/spin, handedness and batting role require verified, dated metadata; not inferred from names.",
                              "Historical retrospective forecasts condition on the opening pair and nominated bowler; provider must supply those before first delivery.",
                              "Same-day matches never contribute to history. Prior completed test matches do contribute to later test-date history (rolling-origin evaluation).",
                              "Rain-revised, shortened, no-result and ambiguous innings are excluded; these states are unsupported live.",
                              "Sequential neural experiment is optional and not part of this evaluated artifact.",
                              "Two-stage product is an approximation; model choice uses validation after reconciliation."]
    try:
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        report["git_commit"] = "uncommitted"
    sample = xs["test"].iloc[0].to_dict()
    timings = [predict(bundle, sample)["latency_ms"] for _ in range(20)]
    report["latency_ms"] = {"p50": float(np.median(timings)), "p95": float(np.quantile(timings, .95)), "n": len(timings), "includes": "all heads and TreeSHAP; excludes network/database"}
    worst = parts["test"][["match_id", "date", "innings", "over_number", "phase", "target_next_over_runs", "target_wicket", "target_extras", "batter_career_balls", "batter_matchup_balls"]].copy()
    worst["predicted"] = test_prediction
    worst["absolute_error"] = np.abs(ys["test"]-test_prediction)
    worst.sort_values("absolute_error", ascending=False).head(100).to_csv(destination / "worst_100.csv", index=False)
    importance = pd.DataFrame({"feature": columns, "importance": candidates["catboost"].feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(destination / "feature_importance.csv", index=False)
    study.trials_dataframe().to_csv(destination / "optuna_trials.csv", index=False)
    joblib.dump(bundle, destination / "bundle.joblib")
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Local MLflow captures immutable artifacts; promotion remains explicit.
    import mlflow
    mlflow.set_tracking_uri((root / "mlruns").as_uri())
    mlflow.set_experiment("crease-next-over")
    with mlflow.start_run(run_name=version):
        mlflow.log_params({**params, "seed": SEED, "dataset_sha256": report["dataset_sha256"]})
        mlflow.log_metrics({"test_"+k: v for k, v in report["test_regression"].items()})
        mlflow.log_metrics({"test_interval_coverage": bundle["test_coverage"]})
        mlflow.log_artifacts(str(destination))
    (root / "reports/latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
