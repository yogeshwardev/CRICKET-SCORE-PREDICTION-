"""Training, selection, calibration, immutable model artifacts, and inference."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.base import clone
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import optuna

from .features import CATEGORICAL, BUCKETS, feature_columns
from .evaluation import regression, classification, conformal_radius, marginal_conformal_radius

SEED = 20260907
# Fit parallelism only affects runtime; results stay seed-deterministic.
THREADS = max(1, int(os.getenv("CREASE_THREADS", os.cpu_count() or 4)))
COMPONENTS = ["batter_runs", "non_striker_runs", "extras", "other_batters", "bowler_conceded"]
TASKS = ["wicket", "boundary", "six", "ten_plus", "run_bucket"]


def split(frame):
    years = sorted(frame.season.unique())
    if len(years) < 5:
        raise ValueError("At least five seasons required for independent temporal partitions")
    test_year, calibration_year, validation_year = years[-1], years[-2], years[-3]
    calibration = frame[frame.season == calibration_year]
    dates = sorted(calibration.date.unique())
    # Model selection happens on a full validation season, then every head is refit on
    # train + validation + the first half of the calibration season. Walk-forward folds
    # showed that selecting on a full season and refitting beats both the old split
    # (which wasted two seasons) and training straight through with a small validation
    # block (which made early stopping unreliable). Calibration blocks stay unseen.
    first, second = dates[int(len(dates)*.50)], dates[int(len(dates)*.75)]
    parts = {"train": frame[frame.season < validation_year],
             "validation": frame[frame.season == validation_year],
             "refit_extra": calibration[calibration.date < first],
             "probability_calibration": calibration[(calibration.date >= first) & (calibration.date < second)],
             "interval_calibration": calibration[calibration.date >= second],
             "test": frame[frame.season == test_year]}
    for a, b in zip(list(parts.values())[:-1], list(parts.values())[1:]):
        if a.empty or b.empty or a.date.max() >= b.date.min() or set(a.match_id) & set(b.match_id):
            raise ValueError("Invalid chronological partitions")
    return parts


def cat_reg(**kwargs):
    return CatBoostRegressor(loss_function="RMSE", random_seed=SEED, thread_count=THREADS,
                             verbose=False, allow_writing_files=False, **kwargs)


def matrix(frame, columns):
    # Feature-selection experiments may drop a categorical entirely, so only convert
    # the ones actually requested rather than assuming the full schema is present.
    x = frame[columns].copy()
    for c in CATEGORICAL:
        if c in x.columns:
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


SUPPORT = ["batter_career_balls", "non_striker_career_balls", "bowler_career_balls"]
# Reliability cut points are validation quantiles, never hand-picked numbers.
RELIABILITY_QUANTILES = {"support_low": .25, "support_high": .60, "spread_low": .40, "spread_high": .80}


def prepare_for_serving(bundle):
    """Serve single rows on one thread.

    Fitting wants every core, but a one-row prediction spends more time dispatching to a
    thread pool than doing the work: the random forest alone drops from ~170ms to ~32ms.
    Applied when a bundle is loaded for inference, never during training.
    """
    for wrapper in bundle["candidates"].values():
        inner = getattr(wrapper, "model", wrapper)
        if hasattr(inner, "n_jobs"):
            inner.n_jobs = 1
    bundle["serving_threads"] = 1
    return bundle


def candidate_predictions(bundle, x):
    """Every candidate's prediction, computed once so one request never repeats a model."""
    return {name: model.predict(x) for name, model in bundle["candidates"].items()}


def team_predict(bundle, x, candidates=None):
    if candidates is None:
        candidates = {name: bundle["candidates"][name].predict(x)
                      for name, w in bundle["weights"].items() if w > 1e-8}
    return np.maximum(0, sum(w * candidates[name] for name, w in bundle["weights"].items() if w > 1e-8))


def dispersion(bundle, x, candidates=None):
    """Disagreement across every trained candidate, whatever its ensemble weight."""
    candidates = candidate_predictions(bundle, x) if candidates is None else candidates
    return np.column_stack(list(candidates.values())).std(axis=1)


def reliability(bundle, x, candidates=None):
    """Phase 27 label from model agreement, player history and matchup evidence.

    Cut points come from the validation season. HIGH is only ever issued when the
    label ordering was verified monotone in validation error; otherwise the rule
    degrades to MEDIUM/LOW rather than asserting unearned confidence.
    """
    rule = bundle["reliability"]
    spread = dispersion(bundle, x, candidates)
    support = x[SUPPORT].to_numpy(dtype=float).min(axis=1)
    matchup = x["batter_matchup_balls"].to_numpy(dtype=float)
    label = np.full(len(x), "MEDIUM", dtype=object)
    if rule["high_enabled"]:
        label[(support >= rule["support_high"]) & (spread <= rule["spread_low"]) & (matchup > 0)] = "HIGH"
    label[(support < rule["support_low"]) | (spread > rule["spread_high"])] = "LOW"
    return label, spread, support, matchup


def reliability_reason(bundle, label, spread, support, matchup):
    rule = bundle["reliability"]
    if label == "LOW":
        causes = ([f"only {support:.0f} balls of prior history for the least-seen player"] if support < rule["support_low"] else []) + \
                 ([f"candidate models disagree by {spread:.2f} runs"] if spread > rule["spread_high"] else [])
        return "Wider than usual uncertainty: " + " and ".join(causes) + "."
    if label == "HIGH":
        return (f"Candidate models agree within {spread:.2f} runs, every player has at least {support:.0f} balls of history, "
                f"and this batter has faced this bowler for {matchup:.0f} balls.")
    reasons = ["candidate models disagree by %.2f runs" % spread] if spread > rule["spread_low"] else []
    reasons += ["this batter-bowler pair has no shared history"] if matchup == 0 else []
    reasons += ["player history is thinner than the validated high-support threshold"] if support < rule["support_high"] else []
    detail = "; ".join(reasons) if reasons else "the high-support ordering was not verified for this model"
    return "Ordinary historical support: " + detail + "."


def component_predict(bundle, x, total=None):
    values = np.maximum(0, bundle["components"].predict(x))
    if bundle.get("batter_mode") == "two_stage":
        balls = np.maximum(0, bundle["balls"].predict(x))
        rates = np.maximum(0, bundle["rates"].predict(x))
        values[:, :2] = balls*rates
    # Explicit replacement-batter component prevents losing runs after a wicket.
    total = team_predict(bundle, x) if total is None else total
    denom = values[:, :4].sum(axis=1)
    values[:, :4] *= (total/np.maximum(denom, 1e-9))[:, None]
    values[denom <= 1e-9, 3] = total[denom <= 1e-9]
    values[:, 4] = np.minimum(values[:, 4], total)
    return values


def predict(bundle, features: dict, explain=True):
    started = time.perf_counter()
    x = matrix(pd.DataFrame([features]), bundle["features"])
    # One pass over the candidates feeds the ensemble, the components and the
    # reliability spread, instead of each of them re-running the same models.
    cached = candidate_predictions(bundle, x)
    expected = team_predict(bundle, x, cached)
    total = float(expected[0])
    components = component_predict(bundle, x, total=expected)[0]
    probabilities = {t: model.predict_proba(x)[0] for t, model in bundle["classifiers"].items()}
    p = probabilities["run_bucket"]
    radius = bundle["radius"]
    label, spread, support, matchup = (v[0] for v in reliability(bundle, x, cached))
    response = {"expected_runs": total, "lower_80": max(0, int(np.floor(total-radius))),
                "upper_80": int(np.ceil(total+radius)),
                "lower_80_match_block": max(0, int(np.floor(total-bundle["match_block_radius"]))),
                "upper_80_match_block": int(np.ceil(total+bundle["match_block_radius"])),
                "run_bucket": BUCKETS[int(p.argmax())],
                "run_bucket_probability": float(p.max()), "distribution": dict(zip(BUCKETS, p.tolist())),
                "batter_expected_runs": float(components[0]), "non_striker_expected_runs": float(components[1]),
                "extras_expected": float(components[2]), "other_batters_expected_runs": float(components[3]),
                "bowler_expected_conceded": float(components[4]),
                **{t+"_probability": float(probabilities[t][1]) for t in TASKS[:-1]},
                "confidence": label,
                "reliability_reason": reliability_reason(bundle, label, spread, support, matchup),
                "reliability_inputs": {"model_disagreement_runs": float(spread), "min_player_history_balls": float(support),
                                       "batter_bowler_matchup_balls": float(matchup),
                                       "validation_mae_by_label": bundle["reliability"]["validation_mae_by_label"]},
                "interval_method": "80% marginal split conformal for one over; the match-block band additionally covers every over of a match at once",
                "observed_test_coverage": bundle["test_coverage"],
                "observed_test_match_block_coverage": bundle["test_match_block_coverage"], "model_version": bundle["version"],
                "explanations": []}
    if explain:
        # CatBoost's native exact TreeSHAP requires no third-party SHAP runtime.
        shap = bundle["candidates"]["catboost"].get_feature_importance(Pool(x, cat_features=CATEGORICAL), type="ShapValues")[0]
        response["explanation_scope"] = "CatBoost component before ensemble/clipping; contributions sum with base value"
        response["explanation_base_value"] = float(shap[-1])
        response["explanation_prediction"] = float(bundle["candidates"]["catboost"].predict(x)[0])
        response["explanations"] = [{"feature": bundle["features"][i], "contribution": float(shap[i])}
                                    for i in np.argsort(np.abs(shap[:-1]))[::-1][:10]]
    # Older promoted artifacts predate the 95% band; serving must not break on them.
    if bundle.get("radius_95") is not None:
        response["lower_95"] = max(0, int(np.floor(total-bundle["radius_95"])))
        response["upper_95"] = int(np.ceil(total+bundle["radius_95"]))
        response["observed_test_coverage_95"] = bundle.get("test_coverage_95")
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
        # Bernoulli bootstrap and column sampling are searched too; both are supported
        # by every loss function reused below, so one parameter set serves all heads.
        params = {"depth": trial.suggest_int("depth", 4, 8),
                  "learning_rate": trial.suggest_float("learning_rate", .02, .15, log=True),
                  "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2, 40, log=True),
                  "random_strength": trial.suggest_float("random_strength", .5, 4),
                  "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 120, log=True),
                  "subsample": trial.suggest_float("subsample", .6, 1.),
                  "rsm": trial.suggest_float("rsm", .5, 1.)}
        model = cat_reg(iterations=iterations, bootstrap_type="Bernoulli", **params)
        model.fit(xs["train"], ys["train"], cat_features=CATEGORICAL,
                  eval_set=(xs["validation"], ys["validation"]), early_stopping_rounds=50)
        best_models[trial.number] = model
        return regression(ys["validation"], model.predict(xs["validation"]))["mae"]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    candidates = {"catboost": best_models[study.best_trial.number]}
    alternatives = {
        "random_forest": RandomForestRegressor(n_estimators=160, max_depth=14, min_samples_leaf=25, max_features=.33, n_jobs=THREADS, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15, l2_regularization=15,
                                                                 early_stopping=False, random_state=SEED)}
    for name, model in alternatives.items():
        print(f"Training {name}", flush=True)
        candidates[name] = EncodedRegressor(model).fit(xs["train"], ys["train"])
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    for name, model in {
        "lightgbm": LGBMRegressor(n_estimators=250, num_leaves=15, learning_rate=.035, reg_lambda=15, verbosity=-1, n_jobs=THREADS, random_state=SEED),
        "xgboost": XGBRegressor(n_estimators=250, max_depth=4, learning_rate=.035, reg_lambda=15, n_jobs=THREADS, random_state=SEED)
    }.items():
        print(f"Training {name}", flush=True)
        candidates[name] = EncodedRegressor(model).fit(xs["train"], ys["train"])
    # Weights and stopping points are chosen from models that have never seen validation.
    predictions = np.column_stack([m.predict(xs["validation"]) for m in candidates.values()])
    fitted = minimize(lambda w: np.mean(np.abs(ys["validation"] - predictions @ w)),
                      np.ones(len(candidates))/len(candidates), method="SLSQP",
                      bounds=[(0, 1)]*len(candidates), constraints={"type": "eq", "fun": lambda w: w.sum()-1})
    val = {name: regression(ys["validation"], predictions[:, i]) for i, name in enumerate(candidates)}
    single = min(val, key=lambda k: val[k]["mae"])
    weights = {k: float(k == single) for k in candidates}
    if fitted.success and regression(ys["validation"], predictions @ fitted.x)["mae"] < val[single]["mae"]:
        weights = dict(zip(candidates, fitted.x.tolist()))
    params = dict(study.best_params, bootstrap_type="Bernoulli", iterations=max(100, candidates["catboost"].tree_count_))
    # Everything above is frozen. Refit on the recent data the selection blocks occupied,
    # so the served model is not three seasons out of date. Calibration and test remain unseen.
    refit_frame = pd.concat([parts["train"], parts["validation"], parts["refit_extra"]])
    xs["refit"], ys["refit"] = matrix(refit_frame, columns), refit_frame.target_next_over_runs.to_numpy()
    parts["refit"] = refit_frame
    print(f"Refitting candidates on {len(refit_frame):,} overs through {refit_frame.date.max()}", flush=True)
    for name, model in list(candidates.items()):
        if name == "catboost":
            candidates[name] = cat_reg(**params).fit(xs["refit"], ys["refit"], cat_features=CATEGORICAL)
        else:
            candidates[name] = EncodedRegressor(clone(model.model)).fit(xs["refit"], ys["refit"])
    print("Training batter, extras and bowler models", flush=True)
    component_y = lambda key: parts[key][["target_"+c for c in COMPONENTS]]
    balls_columns = ["target_batter_balls", "target_non_striker_balls"]
    balls_y = {k: parts[k][balls_columns].to_numpy() for k in ["refit", "validation"]}
    rates_y = {k: component_y(k).to_numpy()[:, :2]/np.maximum(balls_y[k], 1) for k in ["refit", "validation"]}

    def multi_head(train_y, validation_y):
        # Every head stops on the validation season instead of running the tuned
        # iteration count blind, which bounds both overfitting and training time.
        model = CatBoostRegressor(loss_function="MultiRMSE", random_seed=SEED, thread_count=THREADS,
                                  verbose=False, allow_writing_files=False, **params)
        model.fit(xs["refit"], train_y, cat_features=CATEGORICAL,
                  eval_set=(xs["validation"], validation_y), early_stopping_rounds=50)
        return model

    components = multi_head(component_y("refit"), component_y("validation"))
    balls = multi_head(balls_y["refit"], balls_y["validation"])
    rates = multi_head(rates_y["refit"], rates_y["validation"])
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
                                    random_seed=SEED, thread_count=THREADS, verbose=False,
                                    allow_writing_files=False, **params)
        model.fit(xs["refit"], parts["refit"]["target_"+task], cat_features=CATEGORICAL,
                  eval_set=(xs["validation"], parts["validation"]["target_"+task]), early_stopping_rounds=40)
        probabilities = model.predict_proba(xs["probability_calibration"])
        calibrator = LogisticRegression(C=1., max_iter=2000, random_state=SEED)
        calibrator.fit(np.log(np.clip(probabilities, 1e-9, 1)), parts["probability_calibration"]["target_"+task])
        if list(calibrator.classes_) != list(range(6 if task == "run_bucket" else 2)):
            raise ValueError("Missing calibration class; need more calibration matches")
        classifiers[task] = CalibratedTask(model, calibrator)
    bundle["classifiers"] = classifiers
    print("Calibrating reliability labels on validation", flush=True)
    # The refit models have now seen validation, so cut points and the ordering check
    # use the probability-calibration block, which they have not.
    reference = "probability_calibration"
    spread_validation = dispersion(bundle, xs[reference])
    support_validation = xs[reference][SUPPORT].to_numpy(dtype=float).min(axis=1)
    bundle["reliability"] = {
        "support_low": float(np.quantile(support_validation, RELIABILITY_QUANTILES["support_low"])),
        "support_high": float(np.quantile(support_validation, RELIABILITY_QUANTILES["support_high"])),
        "spread_low": float(np.quantile(spread_validation, RELIABILITY_QUANTILES["spread_low"])),
        "spread_high": float(np.quantile(spread_validation, RELIABILITY_QUANTILES["spread_high"])),
        "quantiles": RELIABILITY_QUANTILES, "high_enabled": True, "validation_mae_by_label": {}}
    validation_labels = reliability(bundle, xs[reference])[0]
    validation_error = np.abs(ys[reference] - team_predict(bundle, xs[reference]))
    by_label = {name: {"n": int((validation_labels == name).sum()),
                       "mae": float(validation_error[validation_labels == name].mean())}
                for name in ["HIGH", "MEDIUM", "LOW"] if (validation_labels == name).any()}
    # HIGH is only offered if it genuinely ranked ahead of MEDIUM and LOW in validation error.
    monotone = (len(by_label) == 3 and by_label["HIGH"]["mae"] < by_label["MEDIUM"]["mae"] < by_label["LOW"]["mae"])
    bundle["reliability"].update(high_enabled=bool(monotone), monotone_on_validation=bool(monotone),
                                 validation_mae_by_label=by_label)
    interval_predictions = team_predict(bundle, xs["interval_calibration"])
    bundle["radius"] = marginal_conformal_radius(ys["interval_calibration"], interval_predictions)
    bundle["radius_95"] = marginal_conformal_radius(ys["interval_calibration"], interval_predictions, alpha=.05)
    bundle["match_block_radius"] = conformal_radius(ys["interval_calibration"], interval_predictions,
                                                    parts["interval_calibration"].match_id)
    # All choices frozen above. First and only final-test prediction follows.
    print("Choices frozen. Evaluating untouched final season.", flush=True)
    test_prediction = team_predict(bundle, xs["test"])
    # Coverage is measured on the integer interval that is actually displayed, not on the
    # real-valued one, because outward rounding widens it and would otherwise flatter us.
    lower = np.maximum(0, np.floor(test_prediction-bundle["radius"]))
    upper = np.ceil(test_prediction+bundle["radius"])
    covered = (ys["test"] >= lower) & (ys["test"] <= upper)
    block_lower = np.maximum(0, np.floor(test_prediction-bundle["match_block_radius"]))
    block_upper = np.ceil(test_prediction+bundle["match_block_radius"])
    block_covered = (ys["test"] >= block_lower) & (ys["test"] <= block_upper)
    wide_lower = np.maximum(0, np.floor(test_prediction-bundle["radius_95"]))
    wide_upper = np.ceil(test_prediction+bundle["radius_95"])
    wide_covered = (ys["test"] >= wide_lower) & (ys["test"] <= wide_upper)
    bundle["test_coverage_95"] = float(wide_covered.mean())
    bundle["test_coverage"] = float(covered.mean())
    bundle["test_match_block_coverage"] = float(block_covered.mean())
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
              "uncertainty": {"method": "marginal split conformal over single overs, reported alongside a conservative match-block band",
                              "nominal_over_coverage": .8, "radius": bundle["radius"],
                              "observed_over_coverage": float(covered.mean()),
                              "mean_width": float(np.mean(upper-lower)),
                              "median_width": float(np.median(upper-lower)),
                              "observed_simultaneous_match_coverage": float(pd.DataFrame({"match": parts["test"].match_id.to_numpy(), "covered": covered}).groupby("match").covered.all().mean()),
                              "match_block": {"method": "match-block split conformal", "nominal_simultaneous_match_coverage": .8,
                                              "radius": bundle["match_block_radius"],
                                              "observed_over_coverage": float(block_covered.mean()),
                                              "mean_width": float(np.mean(block_upper-block_lower)),
                                              "observed_simultaneous_match_coverage": float(pd.DataFrame({"match": parts["test"].match_id.to_numpy(), "covered": block_covered}).groupby("match").covered.all().mean())},
                              "caveat": "The headline interval targets marginal coverage of one over; the match-block band targets every over of a match at once and is much wider by construction. Overs within a match are correlated and seasons shift, so exchangeability is approximate: coverage is measured, not guaranteed.",
                              "interval_95": {"nominal_over_coverage": .95, "radius": bundle["radius_95"],
                                              "observed_over_coverage": float(wide_covered.mean()),
                                              "mean_width": float(np.mean(wide_upper-wide_lower))},
                              "rounding": "Displayed bounds are rounded outward to integers and coverage is measured on those displayed bounds, so observed coverage sits slightly above nominal."},
              "promotion_eligible": beats_baselines, "segments": {}}
    test_labels = reliability(bundle, xs["test"])[0]
    report["reliability"] = {**{k: v for k, v in bundle["reliability"].items() if k != "validation_mae_by_label"},
                             "validation_mae_by_label": bundle["reliability"]["validation_mae_by_label"],
                             "test_by_label": {name: {"n": int((test_labels == name).sum()),
                                                      "mae": float(np.mean(np.abs(ys["test"][test_labels == name]-test_prediction[test_labels == name]))),
                                                      "interval_coverage": float(covered[test_labels == name].mean()),
                                                      "mean_interval_width": float(np.mean((upper-lower)[test_labels == name]))}
                                               for name in ["HIGH", "MEDIUM", "LOW"] if (test_labels == name).any()},
                             "note": "Qualitative ranking of expected error, not a probability. HIGH is withheld unless the ordering held on validation."}
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
    report["limitations"] = ["Only CatBoost is Optuna-tuned; the four alternatives use fixed regularized settings, so the candidate table understates their achievable performance.",
                              "Model weights are fitted only through the training seasons, so the final test season is forecast across a multi-season recency gap. Refitting on later seasons would improve recency but would place the conformal and probability calibration blocks inside training data, so it is deliberately not done.",
                              "Pace/spin, handedness and batting role require verified, dated metadata; not inferred from names.",
                              "Historical retrospective forecasts condition on the opening pair and nominated bowler; provider must supply those before first delivery.",
                              "Same-day matches never contribute to history. Prior completed test matches do contribute to later test-date history (rolling-origin evaluation).",
                              "Rain-revised, shortened, no-result and ambiguous innings are excluded; these states are unsupported live.",
                              "Sequential neural experiment is optional and not part of this evaluated artifact.",
                              "Two-stage product is an approximation; model choice uses validation after reconciliation."]
    # A bare commit hash is a false provenance claim when the tree that produced the model
    # differs from it. Record the dirty state and a digest of the diff so a run can never
    # silently appear reproducible from a commit that does not contain its code.
    try:
        def git(*arguments):
            return subprocess.check_output(["git", *arguments], cwd=root, stderr=subprocess.DEVNULL, text=True)
        commit = git("rev-parse", "HEAD").strip()
        pending = [line[3:] for line in git("status", "--porcelain").splitlines() if line.strip()]
        report["git_commit"] = commit
        report["git_dirty"] = bool(pending)
        report["git_uncommitted_files"] = len(pending)
        if pending:
            report["git_diff_sha256"] = hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()
            report["git_provenance_warning"] = (
                f"{len(pending)} files were uncommitted when this model was trained, so commit {commit[:12]} "
                "does not describe the code that produced it. Commit before a run whose provenance must hold.")
    except Exception:
        report["git_commit"] = "unavailable"
        report["git_dirty"] = None
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
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports/latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Local MLflow captures immutable artifacts; promotion remains explicit. MLflow 3 rejects
    # a filesystem tracking store, so the default is the recommended SQLite backend.
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///"+(root / "mlflow.db").as_posix()))
    if mlflow.get_experiment_by_name("crease-next-over") is None:
        mlflow.create_experiment("crease-next-over", artifact_location=(root / "mlartifacts").as_uri())
    mlflow.set_experiment("crease-next-over")
    with mlflow.start_run(run_name=version):
        mlflow.log_params({**params, "seed": SEED, "dataset_sha256": report["dataset_sha256"]})
        mlflow.log_metrics({"test_"+k: v for k, v in report["test_regression"].items()})
        mlflow.log_metrics({"test_interval_coverage": bundle["test_coverage"],
                            "validation_mae": report["selected_validation"]["mae"]})
        mlflow.set_tags({"git_commit": report["git_commit"], "promotion_eligible": report["promotion_eligible"],
                         "batter_mode": bundle["batter_mode"]})
        mlflow.log_artifacts(str(destination))
    return report
