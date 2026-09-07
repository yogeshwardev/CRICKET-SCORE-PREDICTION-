"""Integration checks against the real trained artifact; skipped before training."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from cricket_ai.models import predict
from cricket_ai.service import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained():
    report_path = ROOT / "reports/latest.json"
    if not report_path.exists():
        pytest.skip("Run real historical training before artifact integration checks")
    report = json.loads(report_path.read_text())
    bundle = joblib.load(ROOT / "models" / report["version"] / "bundle.joblib")
    frame = pd.read_parquet(ROOT / "data/processed/overs.parquet")
    row = frame[frame.date >= report["partitions"]["test"]["start"]].iloc[0].to_dict()
    return report, bundle, row


def test_real_forecast_is_coherent(trained):
    report, bundle, row = trained
    forecast = predict(bundle, row)
    assert sum(forecast["distribution"].values()) == pytest.approx(1)
    assert forecast["lower_80"] <= forecast["expected_runs"] <= forecast["upper_80"]
    assert forecast["expected_runs"] == pytest.approx(sum(forecast[k] for k in ["batter_expected_runs", "non_striker_expected_runs", "extras_expected", "other_batters_expected_runs"]))
    assert forecast["bowler_expected_conceded"] <= forecast["expected_runs"]
    for task in ["wicket", "boundary", "six", "ten_plus"]:
        assert 0 <= forecast[task+"_probability"] <= 1
    assert forecast["model_version"] == report["version"]
    assert len(forecast["explanations"]) == 10
    # Target values must not affect inference.
    mutated = {k: 9999 if k.startswith("target_") else v for k, v in row.items()}
    assert predict(bundle, mutated)["expected_runs"] == forecast["expected_runs"]


def test_unseen_players_and_venue(trained):
    _, bundle, row = trained
    for field in ["batter", "non_striker", "bowler", "venue", "team_batting", "team_bowling"]:
        row = dict(row, **{field: "unseen-test-"+field})
    forecast = predict(bundle, row)
    assert np.isfinite(forecast["expected_runs"])


def test_promoted_replay_api_and_auth(trained, tmp_path, monkeypatch):
    report, _, _ = trained
    pointer = ROOT / "models/active.json"
    if not pointer.exists():
        pytest.skip("Requires explicit promoted model")
    monkeypatch.setenv("CREASE_API_KEY", "integration-test-secret")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///"+str(tmp_path / "integration.db"))
    headers = {"Authorization": "Bearer integration-test-secret"}
    with TestClient(create_app(ROOT)) as client:
        assert client.get("/health").json()["model_loaded"]
        assert client.get("/model").status_code == 401
        assert client.get("/model", headers=headers).json()["version"] == report["version"]
        states = client.get("/replay/states", headers=headers).json()
        result = client.get(f"/replay/{states[0]['id']}", headers=headers)
        assert result.status_code == 200
        assert result.json()["mode"] == "historical_replay"
        assert client.get("/monitoring", headers=headers).json()["scored"] == 0
