"""Train the delivery-level heads and evaluate the Monte Carlo simulator as a challenger.

Partitions are taken from the same chronological helper the champion uses, so the two
models are trained on identical data and compared on identical rows. The hybrid weight
is fitted on the calibration season, never on the final test season.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn import metrics as skm

from cricket_ai.delivery import (CATEGORICAL, EXTRA_CLASSES, HEADS, LEGALITY_CLASSES,
                                 RUN_CLASSES)
from cricket_ai.models import SEED, THREADS

TARGET = {"runs": "target_runs", "extras": "target_extras",
          "legality": "target_legality", "wicket": "target_wicket"}


def provenance(root: Path, dataset: Path, columns: list[str], config: dict) -> dict:
    """Everything needed to say which code and data produced these heads.

    A commit hash alone is a false claim when the tree was dirty, so the dirty flag and
    a digest of the diff travel with it.
    """
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True)
    try:
        pending = [l[3:] for l in git("status", "--porcelain").splitlines() if l.strip()]
        commit = git("rev-parse", "HEAD").strip()
        digest = hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()[:16] if pending else None
    except Exception:
        commit, pending, digest = "unavailable", [], None
    return {"git_commit": commit, "git_dirty": bool(pending), "uncommitted_file_count": len(pending),
            "git_diff_digest": digest,
            "dataset_path": dataset.name,
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "feature_schema_hash": hashlib.sha256("|".join(columns).encode()).hexdigest()[:16],
            "feature_count": len(columns),
            "training_config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
            "training_config": config,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "reproducible_from_commit_alone": not pending}


def partitions(frame: pd.DataFrame, report: dict) -> dict:
    """Reuse the champion's exact date boundaries so nothing is compared across splits."""
    windows = report["partitions"]
    def block(name):
        start, end = windows[name]["start"], windows[name]["end"]
        return frame[(frame.date >= start) & (frame.date <= end)]
    return {"train": frame[frame.date <= windows["train"]["end"]],
            "validation": block("validation"),
            "refit_extra": block("refit_extra"),
            "probability_calibration": block("probability_calibration"),
            "interval_calibration": block("interval_calibration"),
            "test": block("test")}


def prepare(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = frame[columns].copy()
    for column in CATEGORICAL:
        if column in x:
            x[column] = x[column].fillna("unknown").astype(str)
    return x


def train_heads(parts: dict, columns: list[str], iterations: int) -> dict:
    categorical = [c for c in CATEGORICAL if c in columns]
    refit = pd.concat([parts["train"], parts["validation"], parts["refit_extra"]])
    models = {}
    for head in HEADS:
        target = TARGET[head]
        classes = sorted(refit[target].unique())
        print(f"  {head:<9} classes={len(classes)} rows={len(refit):,}", flush=True)
        loss = "MultiClass" if len(classes) > 2 else "Logloss"
        selector = CatBoostClassifier(loss_function=loss, iterations=iterations, depth=6,
                                      learning_rate=.08, l2_leaf_reg=8., random_seed=SEED,
                                      thread_count=THREADS, verbose=False, allow_writing_files=False)
        selector.fit(prepare(parts["train"], columns), parts["train"][target],
                     cat_features=categorical,
                     eval_set=(prepare(parts["validation"], columns), parts["validation"][target]),
                     early_stopping_rounds=40)
        rounds = max(50, selector.get_best_iteration() or selector.tree_count_)
        model = CatBoostClassifier(loss_function=loss, iterations=rounds, depth=6,
                                   learning_rate=.08, l2_leaf_reg=8., random_seed=SEED,
                                   thread_count=THREADS, verbose=False, allow_writing_files=False)
        model.fit(prepare(refit, columns), refit[target], cat_features=categorical)
        models[head] = model
    return models


def delivery_metrics(models: dict, part: pd.DataFrame, columns: list[str]) -> dict:
    x = prepare(part, columns)
    out = {}
    for head in HEADS:
        target = TARGET[head]
        y = part[target].to_numpy()
        p = models[head].predict_proba(x)
        labels = list(range(p.shape[1]))
        block = {"log_loss": float(skm.log_loss(y, p, labels=labels)),
                 "n": int(len(y)), "classes": p.shape[1]}
        if p.shape[1] == 2:
            block.update(roc_auc=float(skm.roc_auc_score(y, p[:, 1])),
                         pr_auc=float(skm.average_precision_score(y, p[:, 1])),
                         brier=float(skm.brier_score_loss(y, p[:, 1])),
                         base_rate=float(y.mean()))
        else:
            onehot = np.eye(p.shape[1])[y]
            block["brier"] = float(np.mean(np.sum((p-onehot)**2, axis=1)))
        out[head] = block
    # Event-level reliability for the outcomes a user actually sees.
    runs_probability = models["runs"].predict_proba(x)
    six_index, four_index = 5, 4
    for name, column, actual in [("six", six_index, part.target_runs == six_index),
                                 ("four", four_index, part.target_runs == four_index),
                                 ("dot", 0, part.target_runs == 0)]:
        probability = runs_probability[:, column]
        out[f"event_{name}"] = {"roc_auc": float(skm.roc_auc_score(actual, probability)),
                                "pr_auc": float(skm.average_precision_score(actual, probability)),
                                "brier": float(skm.brier_score_loss(actual, probability)),
                                "base_rate": float(np.mean(actual))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--iterations", type=int, default=400)
    arguments = parser.parse_args()
    root = arguments.root.resolve()

    frame = pd.read_parquet(root / "data/processed/deliveries_model.parquet")
    report = json.loads((root / "reports/latest.json").read_text())
    parts = partitions(frame, report)
    columns = [c for c in frame.columns if not c.startswith("target_")
               and c not in ("match_id", "date", "season")]
    print("delivery rows:", {k: len(v) for k, v in parts.items()})
    print("features:", len(columns))

    started = time.time()
    print("training heads", flush=True)
    models = train_heads(parts, columns, arguments.iterations)
    elapsed = time.time()-started

    destination = root / "models/delivery_simulator_challenger"
    destination.mkdir(parents=True, exist_ok=True)
    config = {"iterations": arguments.iterations, "depth": 6, "learning_rate": .08,
              "l2_leaf_reg": 8., "seed": SEED, "early_stopping_rounds": 40,
              "refit": "train+validation+refit_extra at the selected iteration count"}
    meta = provenance(root, root / "data/processed/deliveries_model.parquet", columns, config)
    meta["model_type"] = "CatBoostClassifier per head"
    meta["class_mapping"] = {
        "runs": {str(i): ("5_or_other" if v == -1 else v) for i, v in enumerate(RUN_CLASSES)},
        "extras": {str(i): ("4_or_more" if v == 4 else v) for i, v in enumerate(EXTRA_CLASSES)},
        "legality": {str(i): v for i, v in enumerate(LEGALITY_CLASSES)},
        "wicket": {"0": "no_wicket", "1": "wicket"}}
    meta["calibration"] = "raw CatBoost probabilities; isotonic/sigmoid evaluated separately on the calibration block"
    meta["partitions"] = {k: [str(v.date.min()), str(v.date.max()), int(len(v))]
                          for k, v in parts.items() if len(v)}
    joblib.dump({"models": models, "columns": columns, "trained_seconds": elapsed,
                 "provenance": meta}, destination / "heads.joblib")
    (destination / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    result = {"provenance": meta, "train_seconds": elapsed, "features": len(columns),
              "rows": {k: int(len(v)) for k, v in parts.items()},
              "calibration_block": delivery_metrics(models, parts["probability_calibration"], columns),
              "test_block": delivery_metrics(models, parts["test"], columns)}
    out = root / "reports/experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "delivery_heads.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["test_block"], indent=2))
    print("trained in", round(elapsed, 1), "s ->", destination)


if __name__ == "__main__":
    main()
