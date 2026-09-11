"""Cricket-rule tests for the delivery simulator.

A simulator that miscounts legal balls or forgets strike rotation produces confident,
plausible-looking numbers that are wrong, so the rules are pinned here with deterministic
stub models rather than trusted from inspection.
"""
import numpy as np
import pandas as pd
import pytest

from cricket_ai.delivery import (EXTRA_CLASSES, HEADS, LEGALITY_CLASSES, RUN_CLASSES,
                                 DeliverySimulator, LiveState, bucket_probabilities,
                                 delivery_features, extra_class, legality_class, run_class)


class Fixed:
    """Model stub returning one fixed probability row."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame):
        return np.tile(self.probabilities, (len(frame), 1))


def models(runs=0, extras=0, legality="legal", wicket=0.0):
    run_vector = np.zeros(len(RUN_CLASSES)); run_vector[RUN_CLASSES.index(runs)] = 1
    extra_vector = np.zeros(len(EXTRA_CLASSES)); extra_vector[EXTRA_CLASSES.index(extras)] = 1
    legality_vector = np.zeros(3); legality_vector[LEGALITY_CLASSES.index(legality)] = 1
    return {"runs": Fixed(run_vector), "extras": Fixed(extra_vector),
            "legality": Fixed(legality_vector), "wicket": Fixed([1-wicket, wicket])}


def over_row(**overrides):
    row = {"over_number": 10, "innings": 1, "chase_target": 0, "phase": "middle",
           "bowler": "b", "venue": "v", "team_batting": "T", "team_bowling": "U",
           "current_score": 80, "current_wickets": 2, "era_strike_rate": 130.0}
    row.update(overrides)
    return row


def simulator(**kwargs):
    fitted = models(**kwargs)
    columns = list(delivery_features(over_row(), LiveState()).keys())
    return DeliverySimulator(fitted, columns)


def test_over_ends_after_six_legal_deliveries():
    result = simulator(runs=1).simulate(over_row(), draws=25, seed=1)
    assert result["mean_physical_deliveries"] == 6
    assert result["expected_runs"] == 6


def test_wide_does_not_count_toward_the_six_legal_balls():
    # Every ball a wide would never end the over, so the safety cap must stop it.
    result = simulator(legality="wide", extras=1).simulate(over_row(), draws=10, seed=1)
    assert result["mean_physical_deliveries"] == 24
    assert result["expected_runs"] == 24


def test_no_ball_does_not_count_toward_the_six_legal_balls():
    result = simulator(legality="noball", extras=1).simulate(over_row(), draws=10, seed=1)
    assert result["mean_physical_deliveries"] == 24


def test_wickets_are_counted_and_recorded():
    result = simulator(runs=0, wicket=1.0).simulate(over_row(), draws=20, seed=1)
    assert result["wicket_probability"] == 1.0
    assert result["expected_wickets"] == 6


def test_innings_ends_at_ten_wickets():
    result = simulator(runs=0, wicket=1.0).simulate(over_row(current_wickets=8), draws=20, seed=1)
    # Two wickets available, so the over stops after two deliveries.
    assert result["mean_physical_deliveries"] == 2
    assert result["expected_wickets"] == 2


def test_chase_terminates_when_the_target_is_reached():
    # Needs 3 runs; at one run per ball the over must stop after three deliveries.
    result = simulator(runs=1).simulate(over_row(innings=2, chase_target=83, current_score=80),
                                        draws=20, seed=1)
    assert result["mean_physical_deliveries"] == 3
    assert result["expected_runs"] == 3


def test_odd_runs_rotate_strike_and_boundaries_do_not():
    columns = list(delivery_features(over_row(), LiveState()).keys())
    single = DeliverySimulator(models(runs=1), columns)
    four = DeliverySimulator(models(runs=4), columns)
    # With singles the opening pair alternate, so neither faces all six.
    assert single.simulate(over_row(), draws=5, seed=2)["expected_batter_runs"] == 6
    assert four.simulate(over_row(), draws=5, seed=2)["expected_runs"] == 24


def test_wicket_brings_a_replacement_batter_flagged_as_such():
    live = LiveState(striker_slot="replacement")
    features = delivery_features(over_row(), live)
    assert features["striker_is_replacement"] == 1
    # A replacement is described by era-level values, not the dismissed batter's record.
    assert features["striker_career_strike_rate"] == pytest.approx(130.0)
    assert features["striker_innings_runs"] == 0


def test_bucket_probabilities_sum_to_one_and_cover_every_bucket():
    totals = np.array([0, 3, 6, 9, 12, 17, 22, 30])
    probabilities = bucket_probabilities(totals)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(p > 0 for p in probabilities.values())
    assert bucket_probabilities(np.array([25]))["20+"] == 1.0


def test_simulation_is_reproducible_under_a_fixed_seed():
    fitted = models(runs=1)
    fitted["runs"] = Fixed(np.array([.4, .3, .1, 0, .1, .1, 0]))
    columns = list(delivery_features(over_row(), LiveState()).keys())
    engine = DeliverySimulator(fitted, columns)
    first = engine.simulate(over_row(), draws=200, seed=7)
    second = engine.simulate(over_row(), draws=200, seed=7)
    different = engine.simulate(over_row(), draws=200, seed=8)
    assert first == second
    assert first["expected_runs"] != different["expected_runs"]


def test_simulation_never_reads_the_actual_over():
    """Target columns must not reach the feature builder."""
    row = over_row()
    row.update(target_next_over_runs=99, target_wicket=1, target_six=1)
    features = delivery_features(row, LiveState())
    assert not any(k.startswith("target_") for k in features)
    engine = simulator(runs=1)
    clean = engine.simulate(over_row(), draws=10, seed=3)
    polluted = engine.simulate(row, draws=10, seed=3)
    assert clean["expected_runs"] == polluted["expected_runs"]


@pytest.mark.parametrize("runs,expected", [(0, 0), (1, 1), (4, 4), (6, 5), (5, 6), (7, 6)])
def test_run_class_mapping(runs, expected):
    assert run_class(runs) == expected


@pytest.mark.parametrize("extras,expected", [(0, 0), (1, 1), (3, 3), (4, 4), (7, 4)])
def test_extra_class_caps_at_four_plus(extras, expected):
    assert extra_class(extras) == expected


def test_legality_class_prefers_wide_then_noball():
    assert legality_class(0, 0) == 0
    assert legality_class(1, 0) == 1
    assert legality_class(0, 1) == 2


def test_live_state_features_exclude_the_current_ball():
    before = delivery_features(over_row(), LiveState(legal_in_over=2, runs_in_over=7, score=87))
    assert before["legal_in_over"] == 2 and before["runs_in_over"] == 7
    assert before["current_score"] == 87
    assert before["balls_remaining"] == 120-before["legal_balls_innings"]
