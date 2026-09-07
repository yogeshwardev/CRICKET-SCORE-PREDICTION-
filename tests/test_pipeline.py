import copy
import numpy as np
import pandas as pd
import pytest
from cricket_ai.data import normalize_match
from cricket_ai.features import History, build_features, targets, bucket, feature_columns
from cricket_ai.evaluation import conformal_radius, classification
from cricket_ai.models import split
from cricket_ai.service import MatchState, Delivery, create_app
from fastapi.testclient import TestClient


def document():
    delivery = {"batter": "A", "non_striker": "B", "bowler": "C", "runs": {"batter": 1, "extras": 0, "total": 1}}
    return {"info": {"dates": ["2020-01-01"], "match_type": "T20", "venue": "V", "teams": ["T", "U"],
                      "registry": {"people": {"A": "a", "B": "b", "C": "c", "D": "d"}}},
            "innings": [{"team": "T", "overs": [{"over": 0, "deliveries": [copy.deepcopy(delivery) for _ in range(6)]},
                                                     {"over": 1, "deliveries": [copy.deepcopy(delivery) for _ in range(6)]}]}]}


def rows():
    return normalize_match("m", document())[0]


def test_extras_and_wickets():
    doc = document()
    ds = doc["innings"][0]["overs"][0]["deliveries"]
    ds[0].update(extras={"wides": 2}, runs={"batter": 0, "extras": 2, "total": 2})
    ds[1].update(extras={"noballs": 1}, runs={"batter": 4, "extras": 1, "total": 5})
    ds[2].update(extras={"legbyes": 4}, runs={"batter": 0, "extras": 4, "total": 4})
    ds[3]["wickets"] = [{"player_out": "B", "kind": "run out"}]
    ds[4]["wickets"] = [{"player_out": "A", "kind": "retired hurt"}]
    parsed, reason, _ = normalize_match("m", doc)
    assert reason is None
    assert parsed[0]["legal"] == parsed[0]["batter_ball"] == 0
    assert parsed[1]["legal"] == 0 and parsed[1]["batter_ball"] == 1
    assert parsed[1]["bowler_conceded"] == 5
    assert parsed[2]["bowler_conceded"] == parsed[2]["boundary"] == 0
    assert parsed[3]["wicket"] == 1 and parsed[3]["bowler_wicket"] == 0
    assert parsed[4]["wicket"] == 0


def test_state_before_over_and_target_independence():
    ds = rows()
    h = History()
    original = build_features(ds[6], ds[:6], h)
    ds[7]["runs_total"] = 30
    ds[6]["runs_batter"] = 6
    assert build_features(ds[6], ds[:6], h) == original
    assert original["current_score"] == 6
    assert not any(c.startswith("target_") for c in feature_columns(pd.DataFrame([original])))


def test_history_future_and_same_day_rejected():
    ds = rows()
    h = History()
    h.update(pd.DataFrame(ds))
    with pytest.raises(ValueError, match="precede"):
        build_features(ds[0], [], h)
    state = dict(ds[0], date="2019-01-01")
    with pytest.raises(ValueError, match="precede"):
        build_features(state, [], h)


def test_historical_matches_commit_only_after_date(tmp_path):
    from cricket_ai.features import make_samples
    ds = rows()
    frames = []
    # Bootstrap deliberately needs 1000 balls; duplicate fixture MATCHES only here.
    for day in range(1, 5):
        for match in range(100):
            frames.extend([dict(d, date=f"2020-01-0{day}", match_id=f"{day}-{match}") for d in ds])
    frame = pd.DataFrame(frames)
    before = make_samples(frame, tmp_path)
    changed = frame.copy()
    future_last = (changed.date == "2020-01-04") & (changed.over_number == 2)
    changed.loc[future_last, "runs_batter"] += 1
    changed.loc[future_last, "runs_total"] += 1
    # First over of the future date is allowed to change TARGETS, never history.
    later = make_samples(changed, tmp_path)
    pd.testing.assert_frame_equal(before[before.date < "2020-01-04"].reset_index(drop=True),
                                  later[later.date < "2020-01-04"].reset_index(drop=True))
    assert (before.history_through < before.date).all()


def test_partial_over_and_replacement_conservation():
    ds = rows()[:3]
    ds[1]["batter"] = "d"
    t = targets(ds, ds[0])
    assert t["target_next_over_runs"] == 3
    assert t["target_other_batters"] == 1
    assert t["target_next_over_runs"] == sum(t[k] for k in ["target_batter_runs", "target_non_striker_runs", "target_extras", "target_other_batters"])


def test_mapping_strict_and_super_over_excluded():
    doc = document()
    doc["innings"].append(dict(doc["innings"][0], super_over=True))
    assert len(normalize_match("m", doc)[0]) == 12
    del doc["info"]["registry"]["people"]["A"]
    with pytest.raises(ValueError, match="Unregistered"):
        normalize_match("m", doc)


@pytest.mark.parametrize("runs,expected", [(0, 0), (4, 0), (5, 1), (7, 1), (8, 2), (10, 2), (11, 3), (14, 3), (15, 4), (19, 4), (20, 5), (36, 5)])
def test_buckets(runs, expected):
    assert bucket(runs) == expected


def test_split_whole_matches():
    frame = pd.DataFrame([dict(season=y, date=f"{y}-05-{d:02}", match_id=f"{y}-{d}") for y in range(2018, 2025) for d in range(1, 29)])
    parts = split(frame)
    assert parts["test"].season.unique().tolist() == [2024]
    assert parts["train"].season.max() == 2021
    assert not set(parts["interval_calibration"].match_id) & set(parts["probability_calibration"].match_id)


def test_conformal_match_block_rank():
    y = np.arange(10)
    assert conformal_radius(y, np.zeros(10), np.arange(10), .2) == 8
    # A duplicate over in the same match doesn't change sample size/quantile.
    assert conformal_radius(np.r_[y, 9], np.zeros(11), np.r_[np.arange(10), 9], .2) == 8
    with pytest.raises(ValueError):
        conformal_radius([1], np.array([0]), ["a"], .2)


def state_payload():
    return dict(match_id="live", date="2027-04-01", innings=1, over=1, score=0, wickets=0,
                striker="a", non_striker="b", bowler="c", venue="V", team_batting="T", team_bowling="U",
                bowler_confirmed=True, previous_deliveries=[])


def test_live_features_match_training():
    state = MatchState(**state_payload())
    ds = rows()
    training = dict(ds[0], date=state.date, season=2027)
    assert state.features(History()) == build_features(training, [], History())


def test_live_rejects_target_over_deliveries():
    payload = state_payload()
    d = rows()[0]
    payload["previous_deliveries"] = [{k: d[k] for k in Delivery.model_fields}]
    with pytest.raises(ValueError, match="leakage"):
        MatchState(**payload)


def test_api_fails_closed_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CREASE_API_KEY", "test-token")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///"+str(tmp_path / "test.db"))
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post("/predict/next-over", json=state_payload()).status_code == 401
        assert client.post("/predict/next-over", json=state_payload(), headers={"Authorization": "Bearer test-token"}).status_code == 503


def test_rain_and_missing_data_exclusions():
    for field, value in [("outcome", {"method": "D/L"}), ("missing", ["powerplays"]), ("overs", 10)]:
        doc = document()
        doc["info"][field] = value
        parsed, reason, _ = normalize_match("m", doc)
        assert not parsed and reason


def test_single_class_metrics_are_defined():
    report = classification(np.zeros(10, dtype=int), np.tile([.8, .2], (10, 1)))
    assert report["roc_auc"] is None
    assert report["brier"] == pytest.approx(.04)
