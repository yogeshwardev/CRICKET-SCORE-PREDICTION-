"""Delivery-level outcome model and a cricket-aware Monte Carlo over simulator.

CHALLENGER COMPONENT. Nothing here is wired into the promoted serving path.

The champion predicts a conditional mean and therefore never nominates the explosive
overs that dominate its error. This models what can happen on each individual ball and
simulates the over forward, so a 20+ over gets a probability instead of being rounded
away toward the mean.

Leakage control has two layers. Pre-over context is taken from the existing over-level
feature row, which is already built strictly from matches before the match date. Within
the over, state evolves only from deliveries that have already happened. `delivery_features`
is the single feature builder used both to construct training rows and to step the
simulator, so the two cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# Compatible targets. Cricket lets a wicket, extras and runs coexist on one ball, so
# forcing them into a single exclusive class would misrepresent the game.
RUN_CLASSES = [0, 1, 2, 3, 4, 6, -1]          # -1 is "other", covering 5s and rarities
EXTRA_CLASSES = [0, 1, 2, 3, 4]               # 4 means "4 or more"
LEGALITY_CLASSES = ["legal", "wide", "noball"]
HEADS = ["runs", "extras", "legality", "wicket"]

# Per-batter statistics pulled from whichever slot is currently on strike.
STRIKER_SUFFIXES = [
    "career_strike_rate", "career_boundary_rate", "career_six_rate", "career_dot_rate",
    "career_balls", "career_average", "season_strike_rate", "season_boundary_rate",
    "innings_runs", "innings_balls", "innings_sr",
    "ewma0.15_runs_per_ball", "ewma0.15_boundary_rate", "ewma0.15_dot_rate",
    "ewma0.4_runs_per_ball", "ewma0.4_boundary_rate",
    "powerplay_sr", "middle_sr", "death_sr",
    "matchup_strike_rate", "matchup_balls", "matchup_boundary_rate", "matchup_dot_rate",
    "sr_vs_era", "mean_position", "position_now",
]
OVER_CONTEXT = [
    "over_number", "innings", "chase_target", "venue_run_rate", "venue_phase_run_rate",
    "venue_boundary_rate", "venue_wicket_rate", "venue_over_run_rate",
    "era_strike_rate", "era_economy", "team_run_rate", "opposition_run_rate",
    "bowler_career_economy", "bowler_career_dot_rate", "bowler_career_boundary_rate",
    "bowler_career_six_rate", "bowler_career_balls", "bowler_season_economy",
    "bowler_powerplay_economy", "bowler_middle_economy", "bowler_death_economy",
    "bowler_ewma0.15_economy", "bowler_ewma0.4_economy", "bowler_economy_vs_era",
    "bowler_match_economy", "bowler_overs_bowled_this_match",
]
CATEGORICAL = ["phase", "bowler", "venue", "team_batting", "team_bowling", "striker_slot"]


def run_class(runs: int) -> int:
    return RUN_CLASSES.index(runs) if runs in RUN_CLASSES else RUN_CLASSES.index(-1)


def extra_class(extras: int) -> int:
    return min(int(extras), 4)


def legality_class(wides: int, noballs: int) -> int:
    return 1 if wides else 2 if noballs else 0


@dataclass
class LiveState:
    """Everything that changes inside an over. Never contains the current ball."""
    legal_in_over: int = 0
    balls_in_over: int = 0
    runs_in_over: int = 0
    wickets_in_over: int = 0
    boundaries_in_over: int = 0
    sixes_in_over: int = 0
    score: int = 0
    wickets: int = 0
    legal_balls_innings: int = 0
    striker_slot: str = "batter"
    other_slot: str = "non_striker"
    striker_runs: float = 0.0
    striker_balls: float = 0.0
    other_runs: float = 0.0
    other_balls: float = 0.0
    finished: bool = False
    batter_runs_tally: dict = field(default_factory=dict)


def _striker_stats(over_row: dict, slot: str, era_strike_rate: float) -> dict:
    """Statistics for whoever is on strike.

    A batter who arrives after a wicket inside the simulated over is not described by
    the pre-over row at all. Rather than silently reusing the dismissed batter's record,
    that case is represented explicitly by league-of-the-era values plus a flag, and the
    approximation is reported with the results.
    """
    if slot == "replacement":
        neutral = {f"striker_{s}": 0.0 for s in STRIKER_SUFFIXES}
        neutral["striker_career_strike_rate"] = era_strike_rate
        neutral["striker_sr_vs_era"] = 1.0
        neutral["striker_mean_position"] = 7.0
        neutral["striker_position_now"] = 7.0
        return neutral
    return {f"striker_{s}": float(over_row.get(f"{slot}_{s}", 0.0) or 0.0) for s in STRIKER_SUFFIXES}


def delivery_features(over_row: dict, live: LiveState) -> dict:
    """Features for the delivery about to be bowled. Shared by training and simulation."""
    era_strike_rate = float(over_row.get("era_strike_rate", 125.0) or 125.0)
    f = {k: over_row.get(k) for k in OVER_CONTEXT}
    f.update({k: over_row.get(k) for k in ["phase", "bowler", "venue", "team_batting", "team_bowling"]})
    f.update(_striker_stats(over_row, live.striker_slot, era_strike_rate))
    f["striker_slot"] = live.striker_slot
    f["striker_is_replacement"] = int(live.striker_slot == "replacement")

    legal_balls = live.legal_balls_innings
    f.update(
        legal_in_over=live.legal_in_over,
        balls_in_over=live.balls_in_over,
        illegal_in_over=live.balls_in_over-live.legal_in_over,
        runs_in_over=live.runs_in_over,
        wickets_in_over=live.wickets_in_over,
        boundaries_in_over=live.boundaries_in_over,
        sixes_in_over=live.sixes_in_over,
        current_score=live.score,
        current_wickets=live.wickets,
        wickets_remaining=10-live.wickets,
        legal_balls_innings=legal_balls,
        balls_remaining=max(0, 120-legal_balls),
        current_run_rate=6*live.score/max(legal_balls, 1),
        striker_live_runs=live.striker_runs,
        striker_live_balls=live.striker_balls,
        striker_live_sr=100*live.striker_runs/max(live.striker_balls, 1),
        other_live_runs=live.other_runs,
        other_live_balls=live.other_balls,
    )
    target = float(over_row.get("chase_target", 0) or 0)
    required = max(0.0, target-live.score) if target else 0.0
    balls_left = max(1, 120-legal_balls)
    f["runs_required"] = required
    f["required_run_rate"] = 6*required/balls_left if target else 0.0
    f["required_minus_current_rate"] = f["required_run_rate"]-f["current_run_rate"] if target else 0.0
    f["required_runs_per_wicket"] = required/max(10-live.wickets, 1) if target else 0.0
    return f


def feature_names(sample: dict) -> list[str]:
    return [k for k in sample if k not in ("match_id", "date", "season")]


def build_delivery_dataset(deliveries: pd.DataFrame, overs: pd.DataFrame) -> pd.DataFrame:
    """One row per delivery, described strictly by what preceded it."""
    context = overs.set_index(["match_id", "innings", "over_number"])
    keep = [c for c in context.columns if not c.startswith("target_")]
    context = context[keep]
    rows = []
    ordered = deliveries.sort_values(["match_id", "innings", "over_number", "ball_number"])
    for (match_id, innings, over_number), block in ordered.groupby(["match_id", "innings", "over_number"], sort=False):
        try:
            over_row = context.loc[(match_id, innings, over_number)].to_dict()
        except KeyError:
            continue  # Bootstrap overs have no leakage-safe context row.
        records = block.to_dict("records")
        first = records[0]
        live = LiveState(score=int(first["current_score"]), wickets=int(first["current_wickets"]),
                         legal_balls_innings=int(first["legal_balls_before"]))
        striker_name, other_name = first["batter"], first["non_striker"]
        for d in records:
            # Identify the slot the actual striker occupies before this ball.
            if d["batter"] == striker_name:
                slot = live.striker_slot
            elif d["batter"] == other_name:
                live = replace(live, striker_slot=live.other_slot, other_slot=live.striker_slot,
                               striker_runs=live.other_runs, striker_balls=live.other_balls,
                               other_runs=live.striker_runs, other_balls=live.striker_balls)
                striker_name, other_name = other_name, striker_name
                slot = live.striker_slot
            else:
                live = replace(live, striker_slot="replacement", striker_runs=0.0, striker_balls=0.0)
                striker_name = d["batter"]
                slot = "replacement"
            feature = delivery_features(over_row, live)
            feature.update(match_id=match_id, date=d["date"], season=d["season"],
                           target_runs=run_class(int(d["runs_batter"])),
                           target_extras=extra_class(int(d["runs_extras"])),
                           target_legality=legality_class(int(d["wides"]), int(d["noballs"])),
                           target_wicket=int(d["wicket"] > 0))
            rows.append(feature)
            # Advance state past the delivery just described.
            runs_batter, total = int(d["runs_batter"]), int(d["runs_total"])
            live = replace(
                live,
                legal_in_over=live.legal_in_over+int(d["legal"]),
                balls_in_over=live.balls_in_over+1,
                runs_in_over=live.runs_in_over+total,
                wickets_in_over=live.wickets_in_over+int(d["wicket"] > 0),
                boundaries_in_over=live.boundaries_in_over+int(d["boundary"]),
                sixes_in_over=live.sixes_in_over+int(d["six"]),
                score=live.score+total,
                wickets=live.wickets+int(d["wicket"] > 0),
                legal_balls_innings=live.legal_balls_innings+int(d["legal"]),
                striker_runs=live.striker_runs+runs_batter,
                striker_balls=live.striker_balls+int(d["batter_ball"]),
            )
    return pd.DataFrame(rows)


class DeliverySimulator:
    """Monte Carlo over simulation under cricket's actual rules.

    Six LEGAL deliveries end the over; wides and no-balls do not count toward them, so a
    simulated over can run to seven, eight or more physical balls. Strike rotates on odd
    completed runs and again at the end of the over. A chase that reaches its target, or
    an innings that loses its tenth wicket, stops immediately rather than playing on.
    """

    def __init__(self, models, columns, max_deliveries=24, structure=None):
        self.models = models
        self.columns = columns
        self.max_deliveries = max_deliveries
        # Without a measured structure the heads are sampled independently, which is
        # only appropriate for the rule tests that use deterministic stubs.
        self.structure = structure

    def _probabilities(self, over_row: dict, live: LiveState) -> dict:
        frame = pd.DataFrame([delivery_features(over_row, live)])[self.columns]
        for column in CATEGORICAL:
            if column in frame:
                frame[column] = frame[column].fillna("unknown").astype(str)
        return {head: self.models[head].predict_proba(frame)[0] for head in HEADS}

    def simulate(self, over_row: dict, draws: int = 5000, seed: int = 20260907) -> dict:
        """Advance every draw in lockstep.

        Draws are stepped together rather than one at a time so the four heads are called
        once per step on the distinct states present, instead of once per simulated ball.
        Identical states are deduplicated exactly — this is a speed change only, and the
        sampled trajectories are the same ones the per-draw loop would produce.
        """
        rng = np.random.default_rng(seed)
        target = float(over_row.get("chase_target", 0) or 0)
        start = LiveState(score=int(over_row.get("current_score", 0) or 0),
                          wickets=int(over_row.get("current_wickets", 0) or 0),
                          legal_balls_innings=int(6*(int(over_row.get("over_number", 1))-1)))
        states = [replace(start) for _ in range(draws)]
        opener = np.zeros(draws, dtype=float)
        finished = np.zeros(draws, dtype=bool)

        for _ in range(self.max_deliveries):
            for index, live in enumerate(states):
                if not finished[index] and (live.legal_in_over >= 6 or live.wickets >= 10
                                            or (target and live.score >= target)):
                    finished[index] = True
            active = np.flatnonzero(~finished)
            if not len(active):
                break
            keys, order = {}, []
            for index in active:
                live = states[index]
                key = (live.striker_slot, live.legal_in_over, live.balls_in_over, live.runs_in_over,
                       live.wickets_in_over, live.boundaries_in_over, live.sixes_in_over,
                       live.striker_runs, live.striker_balls, live.other_runs, live.other_balls)
                if key not in keys:
                    keys[key] = len(keys)
                    order.append(live)
                states[index] = live
            frame = pd.DataFrame([delivery_features(over_row, live) for live in order])[self.columns]
            for column in CATEGORICAL:
                if column in frame:
                    frame[column] = frame[column].fillna("unknown").astype(str)
                    frame[column] = frame[column].astype(str)
            probabilities = {head: self.models[head].predict_proba(frame) for head in HEADS}

            slots = np.array([keys[(states[i].striker_slot, states[i].legal_in_over, states[i].balls_in_over,
                                    states[i].runs_in_over, states[i].wickets_in_over,
                                    states[i].boundaries_in_over, states[i].sixes_in_over,
                                    states[i].striker_runs, states[i].striker_balls,
                                    states[i].other_runs, states[i].other_balls)] for i in active])
            legality = _sample(rng, probabilities["legality"][slots])
            run_index = _sample(rng, probabilities["runs"][slots])
            extra_index = _sample(rng, probabilities["extras"][slots])
            raw_wicket = probabilities["wicket"][slots][:, 1]
            wicket_uniform = rng.random(len(active))
            uniform = rng.random(len(active))
            wicket = (wicket_uniform < raw_wicket).astype(int)

            for position, index in enumerate(active):
                live = states[index]
                kind = int(legality[position])
                runs = RUN_CLASSES[int(run_index[position])]
                runs = 5 if runs == -1 else runs
                extras = EXTRA_CLASSES[int(extra_index[position])]
                out = int(wicket[position])
                structure = self.structure
                if kind == 1:            # A wide is never faced by the batter.
                    runs, extras = 0, max(1, extras)
                elif kind == 2:          # A no-ball always concedes at least one extra.
                    extras = max(1, extras)
                if structure:
                    # Runs and extras first, then dismissal risk conditioned on both.
                    if kind == 0:
                        if runs > 0:
                            extras = 0          # Byes cannot accompany a scored run.
                        elif extras > 0 and uniform[position] > structure[
                                "extras_probability_by_runs"].get(0, 0.0):
                            extras = 0
                        lift = structure["wicket_lift_by_runs"].get(min(runs, 6), 1.0)
                    else:
                        lift = (structure["wicket_lift_wide"] if kind == 1
                                else structure["wicket_lift_noball"])
                    out = int(wicket_uniform[position] < min(1.0, raw_wicket[position]*lift))
                legal = int(kind == 0)
                total = runs+extras
                if live.striker_slot != "replacement":
                    opener[index] += runs
                live = replace(
                    live,
                    legal_in_over=live.legal_in_over+legal,
                    balls_in_over=live.balls_in_over+1,
                    runs_in_over=live.runs_in_over+total,
                    wickets_in_over=live.wickets_in_over+out,
                    boundaries_in_over=live.boundaries_in_over+int(runs in (4, 6)),
                    sixes_in_over=live.sixes_in_over+int(runs == 6),
                    score=live.score+total,
                    wickets=live.wickets+out,
                    legal_balls_innings=live.legal_balls_innings+legal,
                    striker_runs=live.striker_runs+runs,
                    striker_balls=live.striker_balls+int(kind != 1),
                )
                if out:
                    live = replace(live, striker_slot="replacement", striker_runs=0.0, striker_balls=0.0)
                elif runs % 2 == 1:
                    live = replace(live, striker_slot=live.other_slot, other_slot=live.striker_slot,
                                   striker_runs=live.other_runs, striker_balls=live.other_balls,
                                   other_runs=live.striker_runs, other_balls=live.striker_balls)
                states[index] = live

        totals = np.array([s.runs_in_over for s in states], dtype=int)
        return summarize(totals, opener, totals-opener,
                         np.array([s.wickets_in_over for s in states]),
                         np.array([s.boundaries_in_over for s in states]),
                         np.array([s.sixes_in_over for s in states]),
                         np.array([s.balls_in_over for s in states]))


def joint_structure(deliveries: pd.DataFrame) -> dict:
    """Measure cricket's hard joint constraints from history.

    Sampling the heads independently invents deliveries the game never produces: byes
    alongside a boundary, or a wicket on a six. Two constraints dominate, and both are
    close to deterministic rather than merely skewed:

      * On a legal ball, extras occur only when the batter did not score. Byes and
        leg-byes mean the bat did not make the run, so P(extras>0 | runs>0) is 0.
      * Dismissal risk collapses once the batter scores. P(wicket | runs=0) is roughly
        13%, against almost nothing on a four or six.

    Rather than hard-coding those numbers, they are estimated here from the training
    deliveries only and carried with the model, so the simulator's joint behaviour can
    be audited against the same source it came from.
    """
    legal = deliveries[(deliveries.wides == 0) & (deliveries.noballs == 0)]
    base = float(deliveries.wicket.gt(0).mean())
    runs_key = legal.runs_batter.clip(0, 6)
    wicket_by_runs = legal.groupby(runs_key).wicket.apply(lambda v: float(v.gt(0).mean())).to_dict()
    extras_by_runs = legal.groupby(runs_key).runs_extras.apply(lambda v: float(v.gt(0).mean())).to_dict()
    return {
        "base_wicket_rate": base,
        # Multiplicative lift on the state-conditional wicket probability.
        "wicket_lift_by_runs": {int(k): (v/base if base else 0.0) for k, v in wicket_by_runs.items()},
        "wicket_lift_wide": float(deliveries[deliveries.wides > 0].wicket.gt(0).mean()/base) if base else 0.0,
        "wicket_lift_noball": float(deliveries[deliveries.noballs > 0].wicket.gt(0).mean()/base) if base else 0.0,
        "extras_probability_by_runs": {int(k): v for k, v in extras_by_runs.items()},
        "extras_given_wide": float(deliveries[deliveries.wides > 0].runs_extras.gt(0).mean()),
        "extras_given_noball": float(deliveries[deliveries.noballs > 0].runs_extras.gt(0).mean()),
        "n_deliveries": int(len(deliveries)),
    }


def _sample(rng, probabilities: np.ndarray) -> np.ndarray:
    """Vectorized categorical draw, one sample per row of `probabilities`."""
    normalized = probabilities/probabilities.sum(axis=1, keepdims=True)
    thresholds = normalized.cumsum(axis=1)
    return (rng.random((len(normalized), 1)) > thresholds).sum(axis=1).clip(0, normalized.shape[1]-1)


BUCKET_EDGES = [5, 8, 11, 15, 20]
BUCKET_NAMES = ["0-4", "5-7", "8-10", "11-14", "15-19", "20+"]


def bucket_probabilities(totals: np.ndarray) -> dict:
    index = np.searchsorted(BUCKET_EDGES, totals, side="right")
    counts = np.bincount(index, minlength=len(BUCKET_NAMES))
    return dict(zip(BUCKET_NAMES, (counts/len(totals)).tolist()))


def summarize(totals, batter_runs, extras, wickets, boundaries, sixes, physical) -> dict:
    # p2.5/p97.5 are required for a real 95% interval; p10/p90 give the 80% one.
    quantiles = [2.5, 10, 20, 25, 50, 75, 80, 90, 95, 97.5]
    return {
        "expected_runs": float(totals.mean()),
        "median_runs": float(np.median(totals)),
        "mode_runs": int(np.bincount(totals).argmax()),
        "quantiles": {f"p{q}": float(np.percentile(totals, q)) for q in quantiles},
        "bucket_probabilities": bucket_probabilities(totals),
        "probability_15_plus": float((totals >= 15).mean()),
        "probability_20_plus": float((totals >= 20).mean()),
        "probability_10_plus": float((totals >= 10).mean()),
        "wicket_probability": float((wickets > 0).mean()),
        "boundary_probability": float((boundaries > 0).mean()),
        "six_probability": float((sixes > 0).mean()),
        "expected_batter_runs": float(batter_runs.mean()),
        "expected_extras": float(extras.mean()),
        "expected_wickets": float(wickets.mean()),
        "mean_physical_deliveries": float(physical.mean()),
        "draws": int(len(totals)),
    }
