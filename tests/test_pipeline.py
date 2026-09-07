import copy
import pathlib
import numpy as np
import pandas as pd
import pytest
from cricket_ai.data import normalize_match
from cricket_ai.features import History, build_features, targets, bucket, feature_columns
from cricket_ai.evaluation import conformal_radius, marginal_conformal_radius, classification
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


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), float(self.value))


def reliability_bundle(high_enabled=True, values=(9., 9.5, 10.)):
    from cricket_ai.models import RELIABILITY_QUANTILES
    return {"candidates": {f"m{i}": Constant(v) for i, v in enumerate(values)},
            "weights": {f"m{i}": 1/len(values) for i in range(len(values))},
            "reliability": {"support_low": 50., "support_high": 400., "spread_low": .5, "spread_high": 2.,
                            "quantiles": RELIABILITY_QUANTILES, "high_enabled": high_enabled,
                            "monotone_on_validation": high_enabled, "validation_mae_by_label": {}}}


def reliability_frame(support, matchup):
    return pd.DataFrame([{"batter_career_balls": support, "non_striker_career_balls": support+10,
                          "bowler_career_balls": support+20, "batter_matchup_balls": matchup}])


@pytest.mark.parametrize("support,matchup,spread_values,expected", [
    (900, 30, (10., 10., 10.), "HIGH"),
    (900, 0, (10., 10., 10.), "MEDIUM"),      # No shared matchup history.
    (900, 30, (6., 10., 14.), "LOW"),         # Candidates disagree beyond the calibrated cut.
    (10, 30, (10., 10., 10.), "LOW"),         # Least-seen player below the support floor.
    (200, 30, (10., 10., 10.), "MEDIUM"),
])
def test_reliability_label_rules(support, matchup, spread_values, expected):
    from cricket_ai.models import reliability
    bundle = reliability_bundle(values=spread_values)
    label, spread, seen, faced = reliability(bundle, reliability_frame(support, matchup))
    assert label[0] == expected
    assert seen[0] == support and faced[0] == matchup
    assert spread[0] == pytest.approx(np.std(spread_values))


def test_high_is_withheld_when_validation_ordering_failed():
    from cricket_ai.models import reliability, reliability_reason
    bundle = reliability_bundle(high_enabled=False)
    label, spread, seen, faced = reliability(bundle, reliability_frame(900, 30))
    assert label[0] == "MEDIUM"
    assert "not verified" in reliability_reason(bundle, label[0], spread[0], seen[0], faced[0])


def test_reliability_reason_names_its_cause():
    from cricket_ai.models import reliability_reason
    bundle = reliability_bundle()
    assert "12 balls" in reliability_reason(bundle, "LOW", .1, 12., 5.)
    assert "faced this bowler" in reliability_reason(bundle, "HIGH", .1, 900., 30.)


def test_venue_variants_collapse_to_one_ground():
    from cricket_ai.data import canonical_venue, venue_aliases
    aliases = venue_aliases()
    for variants, expected in [
        (["Wankhede Stadium", "Wankhede Stadium, Mumbai"], "Wankhede Stadium"),
        (["M Chinnaswamy Stadium", "M.Chinnaswamy Stadium", "M Chinnaswamy Stadium, Bengaluru"], "M Chinnaswamy Stadium"),
        (["MA Chidambaram Stadium", "MA Chidambaram Stadium, Chepauk", "MA Chidambaram Stadium, Chepauk, Chennai"], "MA Chidambaram Stadium"),
        (["Feroz Shah Kotla", "Arun Jaitley Stadium, Delhi"], "Arun Jaitley Stadium"),
        (["Punjab Cricket Association Stadium, Mohali", "Punjab Cricket Association IS Bindra Stadium"],
         "Punjab Cricket Association IS Bindra Stadium"),
    ]:
        assert {canonical_venue(v, aliases) for v in variants} == {expected}
    # Distinct grounds must never be merged.
    distinct = ["Eden Gardens", "Wankhede Stadium", "Brabourne Stadium", "Dr DY Patil Sports Academy"]
    assert len({canonical_venue(v, aliases) for v in distinct}) == len(distinct)


def test_venue_alias_file_is_reviewed_and_acyclic():
    import pandas as pd
    from cricket_ai.data import VENUE_ALIASES_FILE, canonical_venue, venue_aliases
    table = pd.read_csv(VENUE_ALIASES_FILE)
    assert list(table.columns) == ["alias", "canonical", "reason"]
    assert table.reason.str.len().min() > 10
    aliases = venue_aliases()
    # Every canonical target must itself be stable under the mapping.
    for target in table.canonical:
        assert canonical_venue(target, aliases) == target


def test_normalized_match_uses_canonical_venue():
    doc = document()
    doc["info"]["venue"] = "M.Chinnaswamy Stadium, Bengaluru"
    parsed, reason, _ = normalize_match("m", doc)
    assert reason is None
    assert {row["venue"] for row in parsed} == {"M Chinnaswamy Stadium"}


def test_marginal_conformal_is_tighter_than_match_block():
    # Ten matches of ten overs; exactly one over per match is a large miss, so the
    # misses are 10% of overs but appear in 100% of matches.
    y, prediction, matches = [], [], []
    for match in range(10):
        y += [2] * 9 + [20]
        prediction += [0.] * 10
        matches += [f"m{match}"] * 10
    marginal = marginal_conformal_radius(y, np.array(prediction), .2)
    block = conformal_radius(y, np.array(prediction), matches, .2)
    # Per-over coverage only needs the 80th percentile of over residuals; covering every
    # over of a match at once needs the worst over of the match, so it is much wider.
    assert marginal == 2 and block == 20


def test_marginal_conformal_rank_and_minimum_sample():
    y = np.arange(10)
    assert marginal_conformal_radius(y, np.zeros(10), .2) == 8
    with pytest.raises(ValueError, match="Too few calibration overs"):
        marginal_conformal_radius([1], np.array([0]), .2)


def test_marginal_conformal_attains_nominal_coverage():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=4000)
    radius = marginal_conformal_radius(truth, np.zeros(4000), .2)
    fresh = rng.normal(size=4000)
    assert .77 < np.mean(np.abs(fresh) <= radius) < .83


def test_matrix_tolerates_a_dropped_categorical():
    from cricket_ai.models import matrix
    from cricket_ai.features import CATEGORICAL
    frame = pd.DataFrame([{**{c: None for c in CATEGORICAL}, "over_number": 5}])
    subset = [c for c in CATEGORICAL if c != "competition"] + ["over_number"]
    built = matrix(frame, subset)
    assert "competition" not in built.columns
    # Remaining categoricals are still filled and stringified for CatBoost.
    assert built["venue"].tolist() == ["unknown"]
    assert built["batter"].map(type).tolist() == [str]
    assert matrix(frame, list(CATEGORICAL))["competition"].tolist() == ["unknown"]


def test_report_provenance_flags_a_dirty_tree(tmp_path, monkeypatch):
    """A commit hash alone must never imply the code that produced a model."""
    import json
    import subprocess
    report = json.loads((pathlib.Path(__file__).resolve().parents[1] / "reports/latest.json").read_text()) \
        if (pathlib.Path(__file__).resolve().parents[1] / "reports/latest.json").exists() else None
    if report is None:
        pytest.skip("Run training before provenance checks")
    assert "git_dirty" in report, "reports must state whether the tree was clean"
    if report["git_dirty"]:
        assert report["git_uncommitted_files"] > 0
        assert "does not describe the code that produced it" in report["git_provenance_warning"]
    else:
        assert report.get("git_uncommitted_files", 0) == 0
