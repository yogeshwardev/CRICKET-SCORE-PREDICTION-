"""Monte Carlo convergence and latency for the delivery simulator.

Picks the draw count from evidence rather than habit: the smallest count whose tail
probabilities are stable across seeds, since those are the outputs the experiment exists
to produce. Representative states are chosen by match situation, not by outcome.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cricket_ai.delivery import DeliverySimulator
from cricket_ai.models import split

DRAWS = [1000, 2500, 5000, 10000, 25000]
SEEDS = [1, 2, 3]
WATCHED = ["expected_runs", "probability_15_plus", "probability_20_plus",
           "wicket_probability", "p10", "p90", "p2.5", "p97.5"]


def representative(test: pd.DataFrame) -> dict:
    """One state per match situation, taken deterministically by position."""
    def first(mask, name):
        block = test[mask]
        return (name, block.iloc[len(block)//2].to_dict()) if len(block) else None
    picks = [
        first(test.over_number <= 6, "powerplay"),
        first((test.over_number > 6) & (test.over_number <= 15), "middle"),
        first(test.over_number >= 16, "death"),
        first((test.innings == 2) & (test.required_run_rate < 8), "low_pressure_chase"),
        first((test.innings == 2) & (test.required_run_rate >= 12), "high_pressure_chase"),
        first(test.current_run_rate >= 11, "high_scoring_state"),
        first(test.current_run_rate <= 6, "low_scoring_state"),
    ]
    return {name: row for name, row in filter(None, picks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    arguments = parser.parse_args()
    root = arguments.root.resolve()

    bundle = joblib.load(root / "models/delivery_simulator_challenger/heads.joblib")
    engine = DeliverySimulator(bundle["models"], bundle["columns"], structure=bundle["structure"])
    test = split(pd.read_parquet(root / "data/processed/overs.parquet"))["test"]
    states = representative(test)
    print("states:", list(states))

    records = []
    for draws in DRAWS:
        for name, row in states.items():
            for seed in SEEDS:
                started = time.perf_counter()
                result = engine.simulate(row, draws=draws, seed=seed)
                elapsed = (time.perf_counter()-started)*1000
                flat = {**{k: result[k] for k in WATCHED if k in result},
                        **{k: v for k, v in result["quantiles"].items()}}
                records.append({"draws": draws, "state": name, "seed": seed,
                                "latency_ms": elapsed, **flat})
        print(f"  {draws} draws done", flush=True)
    frame = pd.DataFrame(records)

    # Seed-to-seed spread at each draw count: how much of the answer is noise.
    stability = {}
    for draws, block in frame.groupby("draws"):
        spread = block.groupby("state")[[c for c in WATCHED if c in block]].std()
        stability[int(draws)] = {column: float(spread[column].mean()) for column in spread}
    latency = {int(d): {"mean_ms": float(b.latency_ms.mean()), "p50_ms": float(b.latency_ms.median()),
                        "p95_ms": float(b.latency_ms.quantile(.95)), "p99_ms": float(b.latency_ms.quantile(.99))}
               for d, b in frame.groupby("draws")}

    # Smallest count whose tail probabilities vary by under half a point across seeds.
    tolerance = 0.005
    adequate = [d for d in DRAWS
                if stability[d]["probability_15_plus"] <= tolerance
                and stability[d]["probability_20_plus"] <= tolerance]
    chosen = adequate[0] if adequate else DRAWS[-1]

    result = {"draw_counts": DRAWS, "seeds": SEEDS, "states": list(states),
              "seed_standard_deviation": stability, "latency": latency,
              "tail_tolerance": tolerance, "chosen_draws": chosen,
              "rationale": (f"{chosen} draws is the smallest count whose P(15+) and P(20+) vary by "
                            f"under {tolerance:.3f} across seeds; larger counts cost latency for "
                            "no meaningful change.")}
    out = root / "reports/experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "simulator_convergence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    frame.to_csv(out / "simulator_convergence_raw.csv", index=False)

    print("\nseed-to-seed standard deviation")
    print(f"{'draws':>7} {'E[runs]':>9} {'P(15+)':>9} {'P(20+)':>9} {'P(wkt)':>9} {'p50 ms':>9} {'p95 ms':>9}")
    for draws in DRAWS:
        s, l = stability[draws], latency[draws]
        print(f"{draws:>7} {s['expected_runs']:>9.4f} {s['probability_15_plus']:>9.4f} "
              f"{s['probability_20_plus']:>9.4f} {s['wicket_probability']:>9.4f} "
              f"{l['p50_ms']:>9.1f} {l['p95_ms']:>9.1f}")
    print("\n" + result["rationale"])


if __name__ == "__main__":
    main()
