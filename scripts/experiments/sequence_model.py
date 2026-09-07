"""Phase 10 experiment: does a delivery-sequence encoder add anything to the tabular model?

Kept out of the production package deliberately. The GRU is trained on the same
chronological training seasons and judged on the same validation season used for
tabular model selection. The untouched final test season is never read here.
Integration into the served bundle is only justified if validation MAE improves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from cricket_ai.evaluation import regression
from cricket_ai.features import CATEGORICAL, feature_columns
from cricket_ai.models import SEED, matrix, split, team_predict

def paired_improvement(y, baseline, challenger, sigmas=2.0):
    """Is the challenger better by more than the noise of the comparison itself?

    Both models score the same rows, so the right quantity is the per-row difference in
    absolute error. Its standard error says how much of any gap is sampling noise. A gap
    inside `sigmas` standard errors is not evidence, however tempting the raw number is.
    """
    y = np.asarray(y, dtype=float)
    difference = np.abs(y-np.asarray(baseline, dtype=float)) - np.abs(y-np.asarray(challenger, dtype=float))
    mean = float(difference.mean())
    standard_error = float(difference.std(ddof=1)/np.sqrt(len(difference))) if len(difference) > 1 else float("inf")
    return {"mae_improvement": mean, "standard_error": standard_error,
            "z": float(mean/standard_error) if standard_error > 0 else 0.0,
            "threshold_sigmas": sigmas, "significant": bool(mean > sigmas*standard_error)}


WINDOW = 24
# Per-delivery channels available strictly before the predicted over.
CHANNELS = ["runs_total", "runs_batter", "wicket", "boundary", "six", "legal",
            "runs_extras", "batter_ball", "same_bowler", "same_striker", "ball_number", "over_gap"]


def sequences(overs: pd.DataFrame, deliveries: pd.DataFrame) -> np.ndarray:
    """Last WINDOW deliveries of the innings, oldest first, zero-padded at the front."""
    grouped = {k: g for k, g in deliveries.groupby(["match_id", "innings"], sort=False)}
    out = np.zeros((len(overs), WINDOW, len(CHANNELS)), dtype=np.float32)
    for row_index, row in enumerate(overs.itertuples(index=False)):
        frame = grouped.get((row.match_id, row.innings))
        if frame is None:
            continue
        prior = frame[frame.over_number < row.over_number]
        if prior.empty:
            continue
        prior = prior.tail(WINDOW)
        block = np.column_stack([
            prior.runs_total.to_numpy(dtype=np.float32),
            prior.runs_batter.to_numpy(dtype=np.float32),
            prior.wicket.to_numpy(dtype=np.float32),
            prior.boundary.to_numpy(dtype=np.float32),
            prior.six.to_numpy(dtype=np.float32),
            prior.legal.to_numpy(dtype=np.float32),
            prior.runs_extras.to_numpy(dtype=np.float32),
            prior.batter_ball.to_numpy(dtype=np.float32),
            (prior.bowler.to_numpy() == row.bowler).astype(np.float32),
            (prior.batter.to_numpy() == row.batter).astype(np.float32),
            prior.ball_number.to_numpy(dtype=np.float32) / 6.0,
            (row.over_number - prior.over_number.to_numpy(dtype=np.float32)) / 20.0,
        ])
        out[row_index, WINDOW - len(block):] = block
    return out


class SequenceForecaster(nn.Module):
    def __init__(self, static_width: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.GRU(len(CHANNELS), hidden, batch_first=True)
        self.static = nn.Sequential(nn.Linear(static_width, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, sequence, static):
        _, last = self.encoder(sequence)
        return self.head(torch.cat([last[-1], self.static(static)], dim=1)).squeeze(1)


def run(root: Path, epochs: int, hidden: int) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    overs = pd.read_parquet(root / "data/processed/overs.parquet")
    deliveries = pd.read_parquet(root / "data/interim/deliveries.parquet")
    parts = split(overs)
    columns = feature_columns(overs)
    numeric = [c for c in columns if c not in CATEGORICAL]

    # Standardization statistics come from training rows only.
    train_numeric = parts["train"][numeric].to_numpy(dtype=np.float32)
    centre = train_numeric.mean(axis=0)
    spread = train_numeric.std(axis=0)
    scale = np.where(spread < 1e-6, 1.0, spread)

    def tensors(part):
        static = ((part[numeric].to_numpy(dtype=np.float32) - centre) / scale).clip(-8, 8)
        return (torch.from_numpy(sequences(part, deliveries)), torch.from_numpy(static.astype(np.float32)),
                torch.from_numpy(part.target_next_over_runs.to_numpy(dtype=np.float32)))

    print("Building delivery sequences", flush=True)
    train_data, validation_data = tensors(parts["train"]), tensors(parts["validation"])
    model = SequenceForecaster(len(numeric), hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*train_data), batch_size=512,
                                         shuffle=True, generator=torch.Generator().manual_seed(SEED))
    history, best_state, best_mae = [], None, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for sequence, static, y in loader:
            optimizer.zero_grad()
            nn.functional.mse_loss(model(sequence, static), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(validation_data[0], validation_data[1]).clamp(min=0).numpy()
        measured = regression(validation_data[2].numpy(), prediction)
        history.append({"epoch": epoch, **measured})
        print(f"epoch {epoch:02d} validation MAE {measured['mae']:.4f}", flush=True)
        if measured["mae"] < best_mae:
            best_mae, best_state = measured["mae"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        sequence_validation = model(validation_data[0], validation_data[1]).clamp(min=0).numpy()

    import joblib
    report = json.loads((root / "reports/latest.json").read_text())
    bundle = joblib.load(root / "models" / report["version"] / "bundle.joblib")
    tabular_validation = team_predict(bundle, matrix(parts["validation"], bundle["features"]))
    truth = validation_data[2].numpy()

    # The blend weight is searched on validation only, exactly as the tabular ensemble is.
    weights = np.linspace(0, 1, 21)
    blended = [regression(truth, (1 - w) * tabular_validation + w * sequence_validation)["mae"] for w in weights]
    best_weight = float(weights[int(np.argmin(blended))])
    result = {"window": WINDOW, "channels": CHANNELS, "epochs": epochs, "hidden": hidden, "seed": SEED,
              "training_curve": history,
              "sequence_validation": regression(truth, sequence_validation),
              "tabular_validation": regression(truth, tabular_validation),
              "best_blend_weight_on_sequence": best_weight,
              "blended_validation_mae": float(min(blended)),
              "tabular_version": report["version"]}
    blended_validation = (1-best_weight)*tabular_validation + best_weight*sequence_validation
    test = paired_improvement(truth, tabular_validation, blended_validation)
    result["paired_test"] = test
    result["validation_mae_improvement"] = test["mae_improvement"]
    result["improves_validation"] = bool(best_weight > 0 and test["significant"])
    result["decision"] = (
        f"Blending at weight {best_weight:.2f} improves validation MAE by {test['mae_improvement']:.4f} runs "
        f"({test['z']:.1f} standard errors); that clears the noise threshold, so integration is justified."
        if result["improves_validation"] else
        f"Blending at weight {best_weight:.2f} moves validation MAE by only {test['mae_improvement']:.4f} runs, "
        f"{test['z']:.1f} standard errors of the paired difference and inside sampling noise. The blend weight was "
        f"also chosen on this same season, which flatters it further. The sequence model is not integrated.")
    destination = root / "reports/experiments"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sequence_model.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(best_state, destination / "sequence_model.pt")
    print(json.dumps({k: result[k] for k in ["sequence_validation", "tabular_validation",
                                             "blended_validation_mae", "best_blend_weight_on_sequence", "decision"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=64)
    arguments = parser.parse_args()
    run(arguments.root.resolve(), arguments.epochs, arguments.hidden)
