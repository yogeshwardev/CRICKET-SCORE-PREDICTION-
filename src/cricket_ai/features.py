"""One feature function shared by historical sample generation and live serving.

History is committed only after ALL matches on a date have been featurized.
This conservatively excludes same-day games when start times are unavailable.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BUCKETS = ["0-4", "5-7", "8-10", "11-14", "15-19", "20+"]
CATEGORICAL = ["competition", "venue", "team_batting", "team_bowling", "batter", "non_striker", "bowler", "phase"]
META = {"match_id", "date", "season", "history_through"}


def bucket(runs):
    return int(np.searchsorted([5, 8, 11, 15, 20], runs, side="right"))


def phase(over):
    return "powerplay" if over <= 6 else "middle" if over <= 15 else "death"


def fresh():
    return np.zeros(8, dtype=float)  # runs, balls, outs, boundaries, sixes, dots, conceded, legal


@dataclass
class History:
    aggregates: dict = field(default_factory=dict)
    form: dict = field(default_factory=dict)
    through: str = "0000-00-00"

    def get(self, key):
        return self.aggregates.get(key, fresh())

    def update(self, deliveries: pd.DataFrame):
        for match_id, match in deliveries.groupby("match_id", sort=False):
            bat_form, bowl_form = defaultdict(fresh), defaultdict(fresh)
            for d in match.to_dict("records"):
                a = np.array([d["runs_batter"], d["batter_ball"], 0, d["boundary"], d["six"],
                              int(d["runs_total"] == 0), d["bowler_conceded"], d["legal"]], float)
                keys = [("bat", d["batter"]), ("bat_season", d["season"], d["batter"]),
                        ("bat_phase", d["batter"], phase(d["over_number"])),
                        ("bowl", d["bowler"]), ("bowl_season", d["season"], d["bowler"]),
                        ("bowl_phase", d["bowler"], phase(d["over_number"])),
                        ("matchup", d["batter"], d["bowler"]), ("league",)]
                for key in keys:
                    if key not in self.aggregates:
                        self.aggregates[key] = fresh()
                    self.aggregates[key] += a
                for kind in ["bowl", "bowl_season", "bowl_phase", "matchup", "league"]:
                    for key in keys:
                        if key[0] == kind:
                            self.aggregates[key][2] += d["bowler_wicket"]
                for dismissed in filter(None, d["dismissed_player"].split("|")):
                    for key in [("bat", dismissed), ("bat_season", d["season"], dismissed)]:
                        if key not in self.aggregates:
                            self.aggregates[key] = fresh()
                        self.aggregates[key][2] += 1
                bat_form[d["batter"]] += a
                bowl_form[d["bowler"]] += a
                team_a = a.copy()
                team_a[0] = d["runs_total"]
                team_a[2] = d["wicket"]
                for key in [("venue", d["venue"]), ("venue_over", d["venue"], d["over_number"]),
                            ("venue_phase", d["venue"], phase(d["over_number"])),
                            ("team", d["team_batting"]), ("opposition", d["team_bowling"]),
                            ("head_to_head", d["team_batting"], d["team_bowling"]),
                            ("team_phase", d["team_batting"], phase(d["over_number"]))]:
                    if key not in self.aggregates:
                        self.aggregates[key] = fresh()
                    self.aggregates[key] += team_a
            for kind, values in [("bat", bat_form), ("bowl", bowl_form)]:
                for player, stats in values.items():
                    self.form.setdefault((kind, player), deque(maxlen=10)).append(stats)
        self.through = max(self.through, str(deliveries.date.max()))

    def statistics(self, key, prefix):
        a = self.get(key)
        prior = self.get(("league",))
        # Shrink sparse rates to observed prior league rates. Zero-history bootstrap
        # rows are excluded from training instead of inventing cricket averages.
        pball = max(prior[1], 1)
        smoothed = lambda index: (a[index] + 24 * prior[index] / pball) / (a[1] + 24)
        return {prefix + "_runs": a[0], prefix + "_balls": a[1], prefix + "_outs": a[2],
                prefix + "_average": a[0] / max(a[2], 1),
                prefix + "_strike_rate": 100 * smoothed(0),
                prefix + "_boundary_rate": smoothed(3), prefix + "_six_rate": smoothed(4),
                prefix + "_dot_rate": smoothed(5),
                prefix + "_economy": 6 * (a[6] + 24 * prior[6] / max(prior[7], 1)) / (a[7] + 24)}


def build_features(state: dict, previous: list[dict], history: History) -> dict:
    if history.through >= state["date"]:
        raise ValueError("Historical store must precede match date")
    over = state["over_number"]
    legal = sum(d["legal"] for d in previous)
    score = sum(d["runs_total"] for d in previous)
    wickets = sum(d["wicket"] for d in previous)
    if legal != (over - 1) * 6:
        raise ValueError("Prediction requires a complete, contiguous over-boundary history")
    if score != state["current_score"] or wickets != state["current_wickets"]:
        raise ValueError("Score/wickets disagree with delivery history")
    if wickets >= 10 or over > 20 or (state["target"] and score >= state["target"]):
        raise ValueError("Innings already complete")
    f = {k: state[k] for k in CATEGORICAL if k != "phase"}
    f.update(phase=phase(over), innings=state["innings"], over_number=over, balls_remaining=120-legal,
             current_score=score, current_wickets=wickets, wickets_remaining=10-wickets,
             current_run_rate=6*score/max(legal, 1), chase_target=state["target"],
             runs_required=max(0, state["target"]-score) if state["target"] else 0,
             required_run_rate=6*max(0, state["target"]-score)/max(120-legal, 1) if state["target"] else 0)
    for n in [1, 2, 3, 5]:
        recent = [d for d in previous if d["over_number"] >= over-n]
        for name, col in [("runs", "runs_total"), ("wickets", "wicket"), ("boundaries", "boundary"), ("sixes", "six")]:
            f[f"{name}_last_{n}_overs"] = sum(d[col] for d in recent)
    # Delivery windows count attempted deliveries, including wides/no-balls.
    for n in [6, 12, 18, 24]:
        recent = previous[-n:]
        f[f"runs_last_{n}_deliveries"] = sum(d["runs_total"] for d in recent)
        f[f"dot_rate_last_{n}_deliveries"] = sum(d["runs_total"] == 0 for d in recent)/max(len(recent), 1)
    for role, player in [("batter", state["batter"]), ("non_striker", state["non_striker"])]:
        f.update(history.statistics(("bat", player), role+"_career"))
        f.update(history.statistics(("bat_season", state["season"], player), role+"_season"))
        f.update(history.statistics(("matchup", player, state["bowler"]), role+"_matchup"))
        cur = [d for d in previous if d["batter"] == player]
        f[role+"_innings_runs"] = sum(d["runs_batter"] for d in cur)
        f[role+"_innings_balls"] = sum(d["batter_ball"] for d in cur)
        f[role+"_innings_sr"] = 100*f[role+"_innings_runs"]/max(f[role+"_innings_balls"], 1)
        for n in [5, 10]:
            form = list(history.form.get(("bat", player), []))[-n:]
            a = np.sum(form, axis=0) if form else fresh()
            f[f"{role}_recent_{n}_mean_runs"] = a[0]/max(len(form), 1)
            f[f"{role}_recent_{n}_sr"] = 100*a[0]/max(a[1], 1)
        for p in ["powerplay", "middle", "death"]:
            f[f"{role}_{p}_sr"] = history.statistics(("bat_phase", player, p), "x")["x_strike_rate"]
    bowler = state["bowler"]
    f.update(history.statistics(("bowl", bowler), "bowler_career"))
    f.update(history.statistics(("bowl_season", state["season"], bowler), "bowler_season"))
    cur = [d for d in previous if d["bowler"] == bowler]
    f["bowler_match_legal_balls"] = sum(d["legal"] for d in cur)
    f["bowler_match_conceded"] = sum(d["bowler_conceded"] for d in cur)
    f["bowler_match_wickets"] = sum(d["bowler_wicket"] for d in cur)
    f["bowler_match_economy"] = 6*f["bowler_match_conceded"]/max(f["bowler_match_legal_balls"], 1)
    for n in [5, 10]:
        form = list(history.form.get(("bowl", bowler), []))[-n:]
        a = np.sum(form, axis=0) if form else fresh()
        f[f"bowler_recent_{n}_economy"] = 6*a[6]/max(a[7], 1)
    for p in ["powerplay", "middle", "death"]:
        f[f"bowler_{p}_economy"] = history.statistics(("bowl_phase", bowler, p), "x")["x_economy"]
    for key, prefix in [(("venue", state["venue"]), "venue"),
                        (("venue_over", state["venue"], over), "venue_over"),
                        (("venue_phase", state["venue"], phase(over)), "venue_phase"),
                        (("team", state["team_batting"]), "team"),
                        (("opposition", state["team_bowling"]), "opposition"),
                        (("head_to_head", state["team_batting"], state["team_bowling"]), "head_to_head")]:
        a = history.get(key)
        f[prefix+"_run_rate"] = 6*a[0]/max(a[7], 1)
        f[prefix+"_balls"] = a[7]
        f[prefix+"_boundary_rate"] = a[3]/max(a[1], 1)
        f[prefix+"_wicket_rate"] = a[2]/max(a[1], 1)
    return f


def targets(over: list[dict], state: dict):
    total = sum(d["runs_total"] for d in over)
    out = {"target_next_over_runs": total, "target_run_bucket": bucket(total),
           "target_wicket": int(any(d["wicket"] for d in over)),
           "target_boundary": int(any(d["boundary"] for d in over)),
           "target_six": int(any(d["six"] for d in over)), "target_ten_plus": int(total >= 10),
           "target_extras": sum(d["runs_extras"] for d in over),
           "target_bowler_conceded": sum(d["bowler_conceded"] for d in over),
           "target_other_batters": sum(d["runs_batter"] for d in over if d["batter"] not in (state["batter"], state["non_striker"]))}
    for role in ["batter", "non_striker"]:
        ds = [d for d in over if d["batter"] == state[role]]
        out[f"target_{role}_runs"] = sum(d["runs_batter"] for d in ds)
        out[f"target_{role}_balls"] = sum(d["batter_ball"] for d in ds)
    return out


def make_samples(deliveries: pd.DataFrame, root: Path) -> pd.DataFrame:
    history, rows = History(), []
    for date, day in deliveries.groupby("date", sort=True):
        for (_, _), innings in day.groupby(["match_id", "innings"], sort=False):
            previous = []
            for number, over in innings.groupby("over_number", sort=True):
                records = over.to_dict("records")
                state = records[0]
                # Opening pair and nominated bowler are assumed known before first ball.
                if history.get(("league",))[1] >= 1000:
                    row = build_features(state, previous, history)
                    row.update({k: state[k] for k in ["match_id", "date", "season"]})
                    row["history_through"] = history.through
                    row.update(targets(records, state))
                    rows.append(row)
                previous.extend(records)
        history.update(day)
    result = pd.DataFrame(rows)
    if (result.history_through >= result.date).any():
        raise AssertionError("Historical leakage")
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    result.to_parquet(root / "data/processed/overs.parquet", index=False)
    joblib.dump(history, root / "data/processed/history.joblib")
    return result


def feature_columns(frame):
    return [c for c in frame if c not in META and not c.startswith("target_")]
