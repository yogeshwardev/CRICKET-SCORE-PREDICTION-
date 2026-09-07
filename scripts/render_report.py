"""Render machine-generated results into a readable, reproducible report."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
r = json.loads((root / "reports/latest.json").read_text())
out = root / "reports"
deliveries = pd.read_parquet(root / "data/interim/deliveries.parquet")
overs = pd.read_parquet(root / "data/processed/overs.parquet")
excluded = pd.read_csv(root / "data/interim/exclusions.csv")
lines = ["# Evaluation report", "", f"Model: `{r['version']}`. Seed: `{r['seed']}`.", "",
         f"Real Cricsheet IPL data: **{r['matches']:,} modeled matches**, **{r['overs']:,} overs**, **{len(deliveries):,} normalized deliveries**. "
         f"Excluded match files: {len(excluded)}. Initial history-bootstrap matches have deliveries but no training rows.", "",
         "## Temporal partitions", "", "| Partition | Start | End | Matches | Overs |", "|---|---|---|---:|---:|"]
for name, p in r["partitions"].items():
    lines.append(f"| {name} | {p['start']} | {p['end']} | {p['matches']} | {p['overs']} |")
lines += ["", "Model selection uses validation only. Probability calibration and conformal calibration use disjoint date blocks. "
          "Test-period historical statistics update after prior dates complete; no model weights are trained on test outcomes.", "",
          "## Models and baselines", "", "| Model | Validation MAE | Test MAE | Test RMSE | Test R² |", "|---|---:|---:|---:|---:|"]
for name, t in r["candidate_test"].items():
    lines.append(f"| {name} | {r['candidate_validation'][name]['mae']:.3f} | {t['mae']:.3f} | {t['rmse']:.3f} | {t['r2']:.3f} |")
for name, t in r["baseline_comparison"]["test"].items():
    lines.append(f"| baseline: {name} | {r['baseline_comparison']['validation'][name]['mae']:.3f} | {t['mae']:.3f} | {t['rmse']:.3f} | {t['r2']:.3f} |")
t = r["test_regression"]
lines += [f"| Selected forecast | {r['selected_validation']['mae']:.3f} | {t['mae']:.3f} | {t['rmse']:.3f} | {t['r2']:.3f} |", "",
          f"Median absolute error: {t['median_ae']:.3f} runs.", ""]
for n in [1, 2, 3, 4]:
    lines.append(f"- {t[f'within_{n}_runs']:.1%} of predictions fall within ±{n} runs. This is a tolerance rate, not exact accuracy.")
lines += ["", "## Classification", "", "| Task | Accuracy | Precision | Recall | F1 | ROC-AUC | Log loss | Brier | ECE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for task, m in r["classification"].items():
    auc = f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "n/a"
    lines.append(f"| {task} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {auc} | {m['log_loss']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} |")
u = r["uncertainty"]
lines += ["", "Binary threshold: 0.5. Run-bucket F1 is macro; weighted F1 and confusion matrices are in latest.json. "
          "Binary Brier is mean squared event-probability error; multiclass Brier sums squared class errors. ECE uses ten equal-width bins.", "",
          "## Uncertainty", "", f"Target simultaneous match coverage: {u['nominal_simultaneous_match_coverage']:.0%}. "
          f"Observed simultaneous match coverage: **{u['observed_simultaneous_match_coverage']:.1%}**. "
          f"Observed marginal over coverage: **{u['observed_over_coverage']:.1%}**. Average interval width: **{u['mean_width']:.2f} runs**.", "",
          "These are match-block split-conformal intervals: calibrate the maximum absolute over residual in each calibration match. "
          "Use the ceil((n+1)×0.8)-th ordered match score, truncate the lower bound at zero and round outwards. "
          "The finite-sample coverage result assumes exchangeable matches. Chronological cricket data can shift; "
          "there is no unconditional future coverage guarantee. The deliberately conservative simultaneous guarantee differs from an 80% marginal over interval.", "",
          "## Batter and bowler forecasts", "", f"Selected batter approach: `{r['batter_mode']}`. Validation comparison: `{r['batter_comparison_validation_mae']}`.", "",
          "The direct model has explicit striker, non-striker, extras and replacement-batter components. These are reconciled to the team forecast. "
          "Bowler-conceded runs are separately modeled and capped at team runs. Runs from replacement batters are not incorrectly attributed to the opening pair.", "",
          "## Reproducibility and latency", "", f"CatBoost settings: `{r['catboost_parameters']}`. Optuna trials: {r['optuna_trials']}.", "",
          f"Ensemble weights fitted on validation: `{r['ensemble_weights']}`.", "",
          f"Prediction latency: median {r['latency_ms']['p50']:.1f} ms; p95 {r['latency_ms']['p95']:.1f} ms over {r['latency_ms']['n']} calls, including TreeSHAP and all heads, excluding HTTP/database overhead.", "",
          f"Dataset SHA-256: `{r['dataset_sha256']}`. Git revision: `{r['git_commit']}`.", "",
          f"Validation baseline promotion gate: **{'passed' if r['promotion_eligible'] else 'failed'}**.", "",
          "## Exclusions", ""]
for reason, count in excluded.reason.value_counts().items():
    lines.append(f"- {reason}: {count}")
lines += ["", "## Scope and limitations", ""] + ["- "+x for x in r["limitations"]]
lines += ["", "The latest test report is now inspected. Future feature/model experiments must use validation and reserve a later genuinely unseen season for a fresh final performance claim.", "",
          "## Feature schema", "", ", ".join(f"`{c}`" for c in r["features"]), "",
          "## Detailed artifacts", "", "See latest.json for all slice metrics, component metrics, confusion matrices, reliability bins and parameters. "
          f"See ../models/{r['version']}/worst_100.csv and feature_importance.csv for error analysis and native importance.", "",
          "## Sources", "", "Historical deliveries: [Cricsheet IPL downloads](https://cricsheet.org/downloads/). "
          "Field semantics and identities: [Cricsheet JSON specification](https://cricsheet.org/format/json/)."]
(out / "EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
overs.target_next_over_runs.hist(bins=range(0, 41), ax=axes[0], color="#0a8f79")
axes[0].set(title="Observed next-over runs", xlabel="Runs", ylabel="Overs")
overs.groupby("over_number").target_next_over_runs.mean().plot(ax=axes[1], color="#0a8f79")
axes[1].set(title="Descriptive mean by over (all data)", xlabel="Over", ylabel="Runs")
for name in ["wicket", "boundary", "six"]:
    bins = r["classification"][name]["reliability"]
    axes[2].plot([b["predicted"] for b in bins], [b["observed"] for b in bins], marker="o", label=name)
axes[2].plot([0, 1], [0, 1], "k--", alpha=.3)
axes[2].set(title="Final-test reliability", xlabel="Predicted probability", ylabel="Observed frequency")
axes[2].legend()
fig.tight_layout()
fig.savefig(out / "diagnostics.png", dpi=160)
print(out / "EVALUATION.md")
