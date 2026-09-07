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
          "## Uncertainty", "",
          f"**Headline interval.** Target marginal coverage for one over: {u['nominal_over_coverage']:.0%}. "
          f"Observed test coverage: **{u['observed_over_coverage']:.1%}**. Average width: **{u['mean_width']:.2f} runs** "
          f"(median {u['median_width']:.2f}). Radius: {u['radius']:.2f} runs.", "",
          f"**Conservative match-block band.** Target simultaneous coverage of every over in a match: "
          f"{u['match_block']['nominal_simultaneous_match_coverage']:.0%}. Observed simultaneous match coverage: "
          f"**{u['match_block']['observed_simultaneous_match_coverage']:.1%}**; observed per-over coverage "
          f"{u['match_block']['observed_over_coverage']:.1%}; average width {u['match_block']['mean_width']:.2f} runs.", "",
          "Both are split conformal on a calibration block that no model was fitted on. The headline interval ranks absolute "
          "over residuals and takes the ceil((n+1)x0.8)-th; the match-block band ranks the maximum residual per match instead, "
          "so it is far wider by construction and answers a different question. Lower bounds are truncated at zero and both "
          "bounds are rounded outward to integers; coverage above is measured on those displayed integer bounds, which is why "
          "it sits slightly above nominal. " + u["caveat"], "",
          "## Batter and bowler forecasts", "", f"Selected batter approach: `{r['batter_mode']}`. Validation comparison: `{r['batter_comparison_validation_mae']}`.", "",
          "The direct model has explicit striker, non-striker, extras and replacement-batter components. These are reconciled to the team forecast. "
          "Bowler-conceded runs are separately modeled and capped at team runs. Runs from replacement batters are not incorrectly attributed to the opening pair.", "",
          "## Reproducibility and latency", "", f"CatBoost settings: `{r['catboost_parameters']}`. Optuna trials: {r['optuna_trials']}.", "",
          f"Ensemble weights fitted on validation: `{r['ensemble_weights']}`.", "",
          f"Prediction latency: median {r['latency_ms']['p50']:.1f} ms; p95 {r['latency_ms']['p95']:.1f} ms over {r['latency_ms']['n']} calls, including TreeSHAP and all heads, excluding HTTP/database overhead.", "",
          f"Dataset SHA-256: `{r['dataset_sha256']}`. Git revision: `{r['git_commit']}`."
          + (f" **{r.get('git_provenance_warning', '')}**" if r.get("git_dirty") else " Working tree was clean at training time."), "",
          f"Validation baseline promotion gate: **{'passed' if r['promotion_eligible'] else 'failed'}**.", "",
          "## Exclusions", ""]
for reason, count in excluded.reason.value_counts().items():
    lines.append(f"- {reason}: {count}")
rl = r["reliability"]
lines += ["", "## Reliability labels", "",
          f"Cut points are validation quantiles `{rl['quantiles']}`: support_low={rl['support_low']:.0f} balls, "
          f"support_high={rl['support_high']:.0f} balls, spread_low={rl['spread_low']:.3f} runs, spread_high={rl['spread_high']:.3f} runs. "
          f"Validation label ordering monotone in MAE: **{rl['monotone_on_validation']}**, so HIGH is "
          f"{'issued' if rl['high_enabled'] else 'withheld entirely'}.", "",
          "| Label | Test overs | Test MAE | Interval coverage | Mean width |", "|---|---:|---:|---:|---:|"]
for name, m in rl["test_by_label"].items():
    lines.append(f"| {name} | {m['n']} | {m['mae']:.3f} | {m['interval_coverage']:.1%} | {m['mean_interval_width']:.2f} |")
lines += ["", f"Validation MAE by label: `{rl['validation_mae_by_label']}`. {rl['note']}"]

experiments = out / "experiments"
sequence, ablation, errors = (experiments / n for n in ["sequence_model.json", "feature_ablation.json", "error_analysis.json"])
if any(path.exists() for path in [sequence, ablation, errors]):
    lines += ["", "## Experiments", "",
              "Each experiment is judged on the validation season and never on the final test season."]
if sequence.exists():
    e = json.loads(sequence.read_text())
    lines += ["", f"**Sequence model (Phase 10).** A GRU over the last {e['window']} deliveries alongside the standardized static "
              f"features, trained on the same training seasons. Validation MAE {e['sequence_validation']['mae']:.3f} against the "
              f"tabular ensemble's {e['tabular_validation']['mae']:.3f}. Best validation blend weight on the sequence model: "
              f"{e['best_blend_weight_on_sequence']:.2f}, giving {e['blended_validation_mae']:.3f}. {e['decision']}"]
if ablation.exists():
    e = json.loads(ablation.read_text())
    lines += ["", f"**Feature selection (Phase 17).** Permutation importance over the served ensemble on validation. "
              f"{e['features_dropped']} of {e['features_total']} features had a mean validation MAE increase at or below "
              f"{e['drop_threshold_mae_increase']}. Retrained CatBoost validation MAE: "
              f"{e['catboost_all_features_validation']['mae']:.3f} with all features against "
              f"{e['catboost_reduced_features_validation']['mae']:.3f} reduced. {e['decision']} "
              f"Full ranking in experiments/permutation_importance.csv."]
if errors.exists():
    e = json.loads(errors.read_text())
    share_key = f"share_of_worst_{e['worst_n']}"
    top = sorted([c for c in e["conditions"] if c["lift"]], key=lambda c: -c["lift"])[:5]
    lines += ["", f"**Error analysis (Phase 36).** The worst {e['worst_n']} overs carry "
              f"{e['worst_error_share_of_total_absolute_error']:.1%} of total test absolute error and average "
              f"{e['mean_actual_runs_in_worst']:.1f} actual runs against {e['mean_actual_runs_overall']:.1f} overall; "
              f"{e['under_prediction_share_of_worst']:.0%} of them are under-predictions. "
              "Conditions most over-represented in that tail:", ""]
    lines += [f"- {c['condition']}: {c[share_key]:.0%} of the worst overs against {c['share_of_test_overs']:.0%} of all test "
              f"overs (lift {c['lift']:.2f}); MAE {c['mae_when_true']:.2f} when true against {c['mae_when_false']:.2f} when false"
              for c in top]

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
