"""Leakage and correctness checks for the validation-only experiment scripts."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "scripts/experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sequence_module():
    pytest.importorskip("torch", reason="sequence experiment needs the optional torch extra")
    return load("sequence_model")


def deliveries_frame():
    rows = []
    for over in range(1, 11):
        for ball in range(1, 7):
            rows.append({"match_id": "m1", "innings": 1, "over_number": over, "ball_number": ball,
                         "runs_total": over, "runs_batter": over, "runs_extras": 0, "wicket": 0,
                         "boundary": int(over == 4), "six": 0, "legal": 1, "batter_ball": 1,
                         "bowler": f"b{over % 2}", "batter": "striker"})
    return pd.DataFrame(rows)


def test_sequence_window_excludes_the_predicted_over(sequence_module):
    deliveries = deliveries_frame()
    overs = pd.DataFrame([{"match_id": "m1", "innings": 1, "over_number": 8, "bowler": "b0", "batter": "striker"}])
    window = sequence_module.sequences(overs, deliveries)
    assert window.shape == (1, sequence_module.WINDOW, len(sequence_module.CHANNELS))
    gap = window[0, :, sequence_module.CHANNELS.index("over_gap")]
    used = window[0].any(axis=1)
    # Every non-padded delivery must come from a strictly earlier over.
    assert (gap[used] > 0).all()
    # Overs 1-7 supply 42 deliveries, so the 24-slot window is full and holds overs 4-7.
    assert used.sum() == sequence_module.WINDOW
    assert window[0, used, 0].sum() == 6 * (4 + 5 + 6 + 7)


def test_sequence_pads_the_first_over_and_orders_oldest_first(sequence_module):
    deliveries = deliveries_frame()
    overs = pd.DataFrame([{"match_id": "m1", "innings": 1, "over_number": 1, "bowler": "b0", "batter": "striker"},
                          {"match_id": "m1", "innings": 1, "over_number": 3, "bowler": "b0", "batter": "striker"}])
    window = sequence_module.sequences(overs, deliveries)
    assert not window[0].any(), "nothing precedes the first over"
    used = window[1].any(axis=1)
    assert used.sum() == 12
    # Padding sits at the front and the newest delivery is last.
    assert not used[:-12].any() and used[-12:].all()
    runs = window[1, -12:, 0]
    assert list(runs[:6]) == [1] * 6 and list(runs[6:]) == [2] * 6


def test_sequence_never_crosses_match_or_innings(sequence_module):
    deliveries = pd.concat([deliveries_frame(),
                            deliveries_frame().assign(match_id="m2"),
                            deliveries_frame().assign(innings=2)], ignore_index=True)
    overs = pd.DataFrame([{"match_id": "m2", "innings": 1, "over_number": 2, "bowler": "b0", "batter": "striker"}])
    window = sequence_module.sequences(overs, deliveries)
    used = window[0].any(axis=1)
    # Only over 1 of that match and innings qualifies, despite identical rows elsewhere.
    assert used.sum() == 6
    assert window[0, used, 0].sum() == 6


def test_error_analysis_conditions_are_observable_at_scoring_time():
    module = load("error_analysis")
    frame = pd.DataFrame([{"target_wicket": 1, "target_six": 1, "target_boundary": 1, "target_extras": 3,
                           "target_next_over_runs": 22, "over_number": 18, "batter_innings_balls": 2,
                           "batter_career_balls": 10, "bowler_career_balls": 10, "batter_matchup_balls": 0,
                           "required_run_rate": 15.0, "innings": 2}])
    for name, condition in module.CONDITIONS.items():
        value = condition(frame)
        assert value.dtype == bool and len(value) == 1, name
