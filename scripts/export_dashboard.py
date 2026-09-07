"""Package genuine evaluated forecasts for a private dashboard without a Python host."""
import json
from pathlib import Path
import joblib
import pandas as pd
from cricket_ai.models import predict

root = Path(__file__).resolve().parents[1]
report = json.loads((root / "reports/latest.json").read_text())
bundle = joblib.load(root / "models" / report["version"] / "bundle.joblib")
frame = pd.read_parquet(root / "data/processed/overs.parquet")
test = frame[frame.date >= report["partitions"]["test"]["start"]]
# Final two matches, deterministic and not selected by error quality.
ids = test.match_id.drop_duplicates().tail(2)
chosen = test[test.match_id.isin(ids)]
names = pd.read_csv(root / "data/player_mapping.csv").drop_duplicates("player_id").set_index("player_id").alias.to_dict()
records = []
for index, row in chosen.iterrows():
    state = row.to_dict()
    forecast = predict(bundle, state)
    public_state = {k: state[k] for k in ["match_id", "date", "innings", "over_number", "current_score", "current_wickets", "batter", "non_striker", "bowler", "venue", "team_batting", "team_bowling"]}
    for key in ["batter", "non_striker", "bowler"]:
        public_state[key] = names.get(state[key], state[key])
    forecast.update(mode="historical_replay", state=public_state, actual_runs=int(state["target_next_over_runs"]))
    records.append({"id": int(index), "label": f"{state['date']} · {state['team_batting']} · over {state['over_number']}", "forecast": forecast})
public = root / "frontend/public"
(public / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(public / "replays.json").write_text(json.dumps(records), encoding="utf-8")
print(f"Exported {len(records)} genuine historical forecasts")
