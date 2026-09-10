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


FORM_WINDOW = 25          # innings retained per player for weighted recent form
DECAYS = [0.15, 0.4]      # per-innings exponential decay factors, compared on validation
DAY_WINDOWS = [90, 365]   # calendar form windows in days
LAGS = 12                 # individually encoded preceding deliveries


def days_between(later: str, earlier: str) -> float:
    return (pd.Timestamp(later)-pd.Timestamp(earlier)).days


@dataclass
class History:
    aggregates: dict = field(default_factory=dict)
    form: dict = field(default_factory=dict)
    positions: dict = field(default_factory=dict)
    through: str = "0000-00-00"

    def get(self, key):
        return self.aggregates.get(key, fresh())

    def position(self, player):
        # sum_position, innings, top3, positions 4-6, positions 7+
        return self.positions.get(player, np.zeros(5, dtype=float))

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
                        ("matchup", d["batter"], d["bowler"]), ("league",),
                        ("league_season", d["season"])]
                for key in keys:
                    if key not in self.aggregates:
                        self.aggregates[key] = fresh()
                    self.aggregates[key] += a
                for kind in ["bowl", "bowl_season", "bowl_phase", "matchup", "league", "league_season"]:
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
            match_date = str(match.date.iloc[0])
            for kind, values in [("bat", bat_form), ("bowl", bowl_form)]:
                for player, stats in values.items():
                    self.form.setdefault((kind, player), deque(maxlen=FORM_WINDOW)).append((match_date, stats))
            # Batting position is the order in which players first face a delivery.
            for _, innings in match.groupby("innings", sort=False):
                order, seen = [], set()
                for batter in innings.batter.tolist():
                    if batter not in seen:
                        seen.add(batter)
                        order.append(batter)
                for index, player in enumerate(order, 1):
                    if player not in self.positions:
                        self.positions[player] = np.zeros(5, dtype=float)
                    self.positions[player] += [index, 1, index <= 3, 4 <= index <= 6, index >= 7]
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


def weighted_form(history: History, kind: str, player: str, date: str) -> dict:
    """Recent form under exponential decay by innings and by calendar time.

    A fixed last-N average treats an innings from three seasons ago exactly like
    yesterday's. Decaying by both innings count and elapsed days lets the model use
    whichever notion of recency actually carries signal; both are offered and the
    ablation decides. Only innings strictly before `date` are ever visible.
    """
    entries = [(d, stats) for d, stats in history.form.get((kind, player), []) if d < date]
    out = {}
    for decay in DECAYS:
        weights = np.array([np.exp(-decay*i) for i in range(len(entries))][::-1])
        if len(entries):
            stacked = np.vstack([stats for _, stats in entries])
            total = weights @ stacked
            out[f"{kind}_ewma{decay}_runs_per_ball"] = total[0]/max(total[1], 1e-9)
            out[f"{kind}_ewma{decay}_boundary_rate"] = total[3]/max(total[1], 1e-9)
            out[f"{kind}_ewma{decay}_dot_rate"] = total[5]/max(total[1], 1e-9)
            out[f"{kind}_ewma{decay}_economy"] = 6*total[6]/max(total[7], 1e-9)
        else:
            for suffix in ["runs_per_ball", "boundary_rate", "dot_rate", "economy"]:
                out[f"{kind}_ewma{decay}_{suffix}"] = 0.
    for window in DAY_WINDOWS:
        recent = [stats for d, stats in entries if days_between(date, d) <= window]
        total = np.sum(recent, axis=0) if recent else fresh()
        out[f"{kind}_last_{window}d_innings"] = float(len(recent))
        out[f"{kind}_last_{window}d_runs_per_ball"] = total[0]/max(total[1], 1e-9)
        out[f"{kind}_last_{window}d_economy"] = 6*total[6]/max(total[7], 1e-9)
    return out


def batting_position(previous: list[dict], player: str, fallback: int) -> int:
    """Position in the current innings: order of first appearance as striker."""
    order, seen = [], set()
    for d in previous:
        if d["batter"] not in seen:
            seen.add(d["batter"])
            order.append(d["batter"])
    if player in order:
        return order.index(player)+1
    return max(len(order)+1, fallback)


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
    era = history.get(("league_season", state["season"]))
    lifetime = history.get(("league",))
    # Before a season has any completed matches its own baseline is empty, so fall back
    # to the all-time league rate rather than dividing by nothing.
    era_strike_rate = 100*era[0]/era[1] if era[1] >= 600 else (100*lifetime[0]/max(lifetime[1], 1e-9))
    era_boundary_rate = era[3]/era[1] if era[1] >= 600 else (lifetime[3]/max(lifetime[1], 1e-9))
    era_economy = 6*era[6]/era[7] if era[7] >= 600 else (6*lifetime[6]/max(lifetime[7], 1e-9))
    f["era_strike_rate"] = era_strike_rate
    f["era_economy"] = era_economy
    f["era_balls_observed"] = era[1]
    for role, player in [("batter", state["batter"]), ("non_striker", state["non_striker"])]:
        f.update(history.statistics(("bat", player), role+"_career"))
        f.update(history.statistics(("bat_season", state["season"], player), role+"_season"))
        f.update(history.statistics(("matchup", player, state["bowler"]), role+"_matchup"))
        cur = [d for d in previous if d["batter"] == player]
        f[role+"_innings_runs"] = sum(d["runs_batter"] for d in cur)
        f[role+"_innings_balls"] = sum(d["batter_ball"] for d in cur)
        f[role+"_innings_sr"] = 100*f[role+"_innings_runs"]/max(f[role+"_innings_balls"], 1)
        for n in [5, 10]:
            form = [stats for d, stats in history.form.get(("bat", player), []) if d < state["date"]][-n:]
            a = np.sum(form, axis=0) if form else fresh()
            f[f"{role}_recent_{n}_mean_runs"] = a[0]/max(len(form), 1)
            f[f"{role}_recent_{n}_sr"] = 100*a[0]/max(a[1], 1)
        for p in ["powerplay", "middle", "death"]:
            f[f"{role}_{p}_sr"] = history.statistics(("bat_phase", player, p), "x")["x_strike_rate"]
        f.update({role+k[3:]: v for k, v in weighted_form(history, "bat", player, state["date"]).items()})
        # Scoring inflates across seasons, so raw rates mean different things in
        # different eras. These express a player against the league of their own time.
        f[role+"_sr_vs_era"] = f[role+"_career_strike_rate"]/max(era_strike_rate, 1e-9)
        f[role+"_boundary_rate_vs_era"] = f[role+"_career_boundary_rate"]/max(era_boundary_rate, 1e-9)
        seen = history.position(player)
        f[role+"_position_now"] = batting_position(previous, player, 1 if role == "batter" else 2)
        f[role+"_mean_position"] = seen[0]/max(seen[1], 1)
        f[role+"_innings_seen"] = seen[1]
        f[role+"_share_top3"] = seen[2]/max(seen[1], 1)
        f[role+"_share_position_4_6"] = seen[3]/max(seen[1], 1)
        f[role+"_share_position_7_plus"] = seen[4]/max(seen[1], 1)
    bowler = state["bowler"]
    f.update(history.statistics(("bowl", bowler), "bowler_career"))
    f.update(history.statistics(("bowl_season", state["season"], bowler), "bowler_season"))
    cur = [d for d in previous if d["bowler"] == bowler]
    f["bowler_match_legal_balls"] = sum(d["legal"] for d in cur)
    f["bowler_match_conceded"] = sum(d["bowler_conceded"] for d in cur)
    f["bowler_match_wickets"] = sum(d["bowler_wicket"] for d in cur)
    f["bowler_match_economy"] = 6*f["bowler_match_conceded"]/max(f["bowler_match_legal_balls"], 1)
    for n in [5, 10]:
        form = [stats for d, stats in history.form.get(("bowl", bowler), []) if d < state["date"]][-n:]
        a = np.sum(form, axis=0) if form else fresh()
        f[f"bowler_recent_{n}_economy"] = 6*a[6]/max(a[7], 1)
    for p in ["powerplay", "middle", "death"]:
        f[f"bowler_{p}_economy"] = history.statistics(("bowl_phase", bowler, p), "x")["x_economy"]
    f.update({"bowler"+k[4:]: v for k, v in weighted_form(history, "bowl", bowler, state["date"]).items()})
    f["bowler_economy_vs_era"] = f["bowler_career_economy"]/max(era_economy, 1e-9)
    f["bowler_overs_bowled_this_match"] = f["bowler_match_legal_balls"]/6
    f["bowler_overs_remaining_quota"] = max(0., 4-f["bowler_match_legal_balls"]/6)
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
    # Individually encoded recent deliveries. Aggregates over a window cannot express
    # ordering; these can. Whether that ordering carries signal is settled by ablation.
    for lag in range(1, LAGS+1):
        d = previous[-lag] if len(previous) >= lag else None
        f[f"ball_minus_{lag}_runs"] = d["runs_total"] if d else -1
        f[f"ball_minus_{lag}_wicket"] = d["wicket"] if d else -1
        f[f"ball_minus_{lag}_boundary"] = d["boundary"] if d else -1
    # Chase pressure beyond the raw required rate.
    required = f["runs_required"]
    balls_left = f["balls_remaining"]
    f["required_minus_current_rate"] = f["required_run_rate"]-f["current_run_rate"] if state["target"] else 0.
    f["required_runs_per_wicket"] = required/max(f["wickets_remaining"], 1) if state["target"] else 0.
    f["balls_per_wicket_remaining"] = balls_left/max(f["wickets_remaining"], 1)
    f["pressure_bucket"] = (0 if not state["target"] else
                            int(np.searchsorted([6, 8, 10, 12], f["required_run_rate"], side="right"))+1)
    f["scoring_vs_era"] = f["current_run_rate"]/max(era_strike_rate*6/100, 1e-9)
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
    last_year = None
    for date, day in deliveries.groupby("date", sort=True):
        if str(date)[:4] != last_year:
            last_year = str(date)[:4]
            print(f"Featurizing {last_year}: {len(rows):,} prior over samples", flush=True)
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


# Per-delivery lag columns are still computed and stored so the ablation stays
# reproducible from the dataset, but they are excluded from the model. Two independent
# experiments agree that delivery ordering carries no usable signal here: a GRU over the
# last 24 deliveries failed to beat the tabular ensemble, and removing these flattened
# lags improved walk-forward MAE by 0.022 runs (4.8 standard errors) against the
# established feature set. Aggregates over the same window already capture what matters.
EXCLUDED_PREFIXES = ("ball_minus_",)


def feature_columns(frame):
    return [c for c in frame if c not in META and not c.startswith("target_")
            and not c.startswith(EXCLUDED_PREFIXES)]
