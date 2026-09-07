"""Phase 24: materialize the as-of historical store into normalized SQL tables.

The served model reads the joblib history artifact; this exports the same aggregates
so they can be inspected, joined and audited. Every row carries the `as_of` date the
store was built through, because these statistics are only valid for later matches.
Serving is not changed by this script, and no row is derived from a later match.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine, text

from cricket_ai.features import History

# Aggregate slots, in the order History.fresh() packs them.
SLOTS = ["runs", "balls", "outs", "boundaries", "sixes", "dots", "conceded", "legal"]


def unpack(history: History, kind: str, names: list[str]) -> pd.DataFrame:
    rows = []
    for key, values in history.aggregates.items():
        if key[0] != kind or len(key) != len(names) + 1:
            continue
        rows.append({**dict(zip(names, key[1:])), **dict(zip(SLOTS, values.tolist()))})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["as_of"] = history.through
    return frame


def derived(frame: pd.DataFrame, batting: bool) -> pd.DataFrame:
    if frame.empty:
        return frame
    if batting:
        frame["average"] = frame.runs / frame.outs.clip(lower=1)
        frame["strike_rate"] = 100 * frame.runs / frame.balls.clip(lower=1)
        frame["boundary_rate"] = frame.boundaries / frame.balls.clip(lower=1)
        frame["six_rate"] = frame.sixes / frame.balls.clip(lower=1)
    else:
        frame["economy"] = 6 * frame.conceded / frame.legal.clip(lower=1)
        frame["bowling_average"] = frame.conceded / frame.outs.clip(lower=1)
        frame["balls_per_wicket"] = frame.legal / frame.outs.clip(lower=1)
    frame["dot_rate"] = frame.dots / frame.balls.clip(lower=1)
    return frame


def build(root: Path) -> dict[str, pd.DataFrame]:
    history: History = joblib.load(root / "data/processed/history.joblib")
    deliveries = pd.read_parquet(root / "data/interim/deliveries.parquet")
    aliases = pd.read_csv(root / "data/player_mapping.csv")

    tables = {
        "players": aliases.groupby("player_id").alias.agg(["first", "count"]).reset_index()
                          .rename(columns={"first": "display_name", "count": "alias_count"}),
        "player_aliases": aliases,
        "player_batting_stats": derived(unpack(history, "bat", ["player_id"]), True),
        "player_batting_stats_by_season": derived(unpack(history, "bat_season", ["season", "player_id"]), True),
        "player_batting_stats_by_phase": derived(unpack(history, "bat_phase", ["player_id", "phase"]), True),
        "player_bowling_stats": derived(unpack(history, "bowl", ["player_id"]), False),
        "player_bowling_stats_by_season": derived(unpack(history, "bowl_season", ["season", "player_id"]), False),
        "player_bowling_stats_by_phase": derived(unpack(history, "bowl_phase", ["player_id", "phase"]), False),
        "batter_bowler_matchups": derived(unpack(history, "matchup", ["batter_id", "bowler_id"]), True),
        "venues": unpack(history, "venue", ["venue"]),
        "venue_by_over": unpack(history, "venue_over", ["venue", "over_number"]),
        "venue_by_phase": unpack(history, "venue_phase", ["venue", "phase"]),
        "teams": unpack(history, "team", ["team"]),
        "team_head_to_head": unpack(history, "head_to_head", ["team_batting", "team_bowling"]),
        "league_totals": unpack(history, "league", []),
        "matches": deliveries.groupby("match_id").agg(
            date=("date", "first"), season=("season", "first"), competition=("competition", "first"),
            venue=("venue", "first"), city=("city", "first"), deliveries=("runs_total", "size"),
            runs=("runs_total", "sum"), wickets=("wicket", "sum")).reset_index(),
    }
    # Player-form recency deques, flattened so recent-form joins do not need the artifact.
    form = []
    for (kind, player), window in history.form.items():
        for position, values in enumerate(window, 1):
            form.append({"role": kind, "player_id": player, "match_index_from_oldest": position,
                         **dict(zip(SLOTS, values.tolist())), "as_of": history.through})
    tables["player_form"] = pd.DataFrame(form)
    return {name: frame for name, frame in tables.items() if not frame.empty}


def run(root: Path, url: str) -> dict[str, int]:
    tables = build(root)
    engine = create_engine(url)
    written = {}
    with engine.begin() as connection:
        for name, frame in tables.items():
            # Replace wholesale: an as-of store is rebuilt, never incrementally patched.
            connection.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
            frame.to_sql(name, connection, index=False)
            written[name] = len(frame)
    engine.dispose()
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--database-url", default=os.getenv("FEATURE_STORE_URL"))
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    url = arguments.database_url or "sqlite:///" + str(root / "feature_store.db")
    for name, count in run(root, url).items():
        print(f"{name:<34} {count:>8,} rows")
    print("Written to", url.rsplit("@", 1)[-1])
