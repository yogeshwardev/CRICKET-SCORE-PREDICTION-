"""Re-measure serving latency for the active model and record it in the report.

Latency is a property of the inference code and the machine, not of the trained model,
so it can be re-measured without retraining. Everything else in the report stays frozen.
The measurement notes when it was taken so a stale figure cannot be mistaken for the
one produced by the training run itself.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cricket_ai.models import predict, prepare_for_serving


def run(root: Path, repeats: int) -> dict:
    report_path = root / "reports/latest.json"
    report = json.loads(report_path.read_text())
    bundle = prepare_for_serving(joblib.load(root / "models" / report["version"] / "bundle.joblib"))
    frame = pd.read_parquet(root / "data/processed/overs.parquet")
    rows = frame[frame.date >= report["partitions"]["test"]["start"]]
    sample = [rows.iloc[i].to_dict() for i in range(min(repeats, len(rows)))]

    for row in sample[:5]:  # Warm the model caches before timing.
        predict(bundle, row)
    # Interleaved, because running one mode fully before the other lets cache warmth
    # and machine load masquerade as a difference between them.
    with_shap, without_shap = [], []
    for row in sample:
        with_shap.append(predict(bundle, row)["latency_ms"])
        without_shap.append(predict(bundle, row, explain=False)["latency_ms"])
    measured = {"p50": float(np.median(with_shap)), "p95": float(np.quantile(with_shap, .95)),
                "n": len(with_shap), "p50_without_explanation": float(np.median(without_shap)),
                "p95_without_explanation": float(np.quantile(without_shap, .95)),
                "includes": "all heads and TreeSHAP; excludes network/database",
                "method": "interleaved modes over distinct held-out states on a shared developer laptop",
                "measured_at": datetime.now(timezone.utc).isoformat(),
                "note": "Re-measured on the serving path after training; model and metrics unchanged."}
    report["latency_ms"] = measured
    for path in [report_path, root / "models" / report["version"] / "report.json"]:
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return measured


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeats", type=int, default=60)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.root.resolve(), arguments.repeats), indent=2))
