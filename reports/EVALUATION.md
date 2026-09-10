# Evaluation report

Model: `20260910T124741567173Z`. Seed: `20260907`.

Real Cricsheet IPL data: **1,189 modeled matches**, **46,242 overs**, **287,376 normalized deliveries**. Excluded match files: 49. Initial history-bootstrap matches have deliveries but no training rows.

## Temporal partitions

| Partition | Start | End | Matches | Overs |
|---|---|---|---:|---:|
| train | 2008-04-22 | 2023-05-26 | 979 | 38142 |
| validation | 2024-03-22 | 2024-05-26 | 70 | 2706 |
| refit_extra | 2025-03-22 | 2025-04-19 | 35 | 1342 |
| probability_calibration | 2025-04-20 | 2025-05-04 | 17 | 660 |
| interval_calibration | 2025-05-07 | 2025-06-03 | 17 | 655 |
| test | 2026-03-28 | 2026-05-31 | 71 | 2737 |
| refit | 2008-04-22 | 2025-04-19 | 1084 | 42190 |

Model selection uses validation only. Probability calibration and conformal calibration use disjoint date blocks. Test-period historical statistics update after prior dates complete; no model weights are trained on test outcomes.

## Models and baselines

| Model | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| catboost | 3.798 | 3.891 | 4.872 | 0.100 |
| random_forest | 3.831 | 3.881 | 4.893 | 0.092 |
| hist_gradient_boosting | 3.823 | 3.880 | 4.903 | 0.089 |
| lightgbm | 3.822 | 3.869 | 4.869 | 0.101 |
| xgboost | 3.812 | 3.866 | 4.874 | 0.099 |
| baseline: league_mean | 4.121 | 4.145 | 5.387 | -0.100 |
| baseline: over_number_mean | 4.064 | 4.160 | 5.390 | -0.102 |
| baseline: recent_run_rate | 4.240 | 4.273 | 5.424 | -0.115 |
| baseline: batter_bowler_history | 4.051 | 4.048 | 5.184 | -0.019 |
| Selected forecast | 3.627 | 3.877 | 4.865 | 0.102 |

Median absolute error: 3.359 runs.

- 16.7% of predictions fall within ±1 runs. This is a tolerance rate, not exact accuracy.
- 31.3% of predictions fall within ±2 runs. This is a tolerance rate, not exact accuracy.
- 45.0% of predictions fall within ±3 runs. This is a tolerance rate, not exact accuracy.
- 58.3% of predictions fall within ±4 runs. This is a tolerance rate, not exact accuracy.

## Classification

| Task | Accuracy | Precision | Recall | F1 | ROC-AUC | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wicket | 0.721 | 0.487 | 0.072 | 0.126 | 0.602 | 0.578 | 0.195 | 0.030 |
| boundary | 0.756 | 0.756 | 0.994 | 0.858 | 0.660 | 0.536 | 0.177 | 0.021 |
| six | 0.628 | 0.533 | 0.146 | 0.230 | 0.634 | 0.639 | 0.224 | 0.025 |
| ten_plus | 0.613 | 0.590 | 0.592 | 0.591 | 0.650 | 0.655 | 0.232 | 0.023 |
| run_bucket | 0.260 | 0.184 | 0.206 | 0.189 | n/a | 1.655 | 0.796 | 0.027 |

Binary threshold: 0.5. Run-bucket F1 is macro; weighted F1 and confusion matrices are in latest.json. Binary Brier is mean squared event-probability error; multiclass Brier sums squared class errors. ECE uses ten equal-width bins.

## Uncertainty

**Headline interval.** Target marginal coverage for one over: 80%. Observed test coverage: **87.3%**. Average width: **13.35 runs** (median 13.00). Radius: 6.17 runs.

**Conservative match-block band.** Target simultaneous coverage of every over in a match: 80%. Observed simultaneous match coverage: **98.6%**; observed per-over coverage 100.0%; average width 27.22 runs.

Both are split conformal on a calibration block that no model was fitted on. The headline interval ranks absolute over residuals and takes the ceil((n+1)x0.8)-th; the match-block band ranks the maximum residual per match instead, so it is far wider by construction and answers a different question. Lower bounds are truncated at zero and both bounds are rounded outward to integers; coverage above is measured on those displayed integer bounds, which is why it sits slightly above nominal. The headline interval targets marginal coverage of one over; the match-block band targets every over of a match at once and is much wider by construction. Overs within a match are correlated and seasons shift, so exchangeability is approximate: coverage is measured, not guaranteed.

## Batter and bowler forecasts

Selected batter approach: `two_stage`. Validation comparison: `{'direct': 3.1486697482954535, 'two_stage': 3.099660711898518}`.

The direct model has explicit striker, non-striker, extras and replacement-batter components. These are reconciled to the team forecast. Bowler-conceded runs are separately modeled and capped at team runs. Runs from replacement batters are not incorrectly attributed to the opening pair.

## Reproducibility and latency

CatBoost settings: `{'depth': 4, 'learning_rate': 0.02249207955351036, 'l2_leaf_reg': 18.618686697348792, 'random_strength': 2.7379541306637893, 'min_data_in_leaf': 5, 'subsample': 0.7765396025769856, 'rsm': 0.9338627065232251, 'bootstrap_type': 'Bernoulli', 'iterations': 1112}`. Optuna trials: 20.

Ensemble weights fitted on validation: `{'catboost': 0.703078290878586, 'random_forest': 2.350240774097333e-18, 'hist_gradient_boosting': 0.1576545497192322, 'lightgbm': 0.0, 'xgboost': 0.1392671594021818}`.

Prediction latency: median 328.2 ms; p95 373.9 ms over 20 calls, including TreeSHAP and all heads, excluding HTTP/database overhead.

Dataset SHA-256: `0b3d7b2e336b1271a379513de4af42902e542888569046825c7bb2af06bab79c`. Git revision: `d3f31023ccfbd226a3930ab841d5296039cda829`. **6 files were uncommitted when this model was trained, so commit d3f31023ccfb does not describe the code that produced it. Commit before a run whose provenance must hold.**

Validation baseline promotion gate: **passed**.

## Exclusions

- revised_target_or_rain_method: 23
- shortened_or_ambiguous_innings: 17
- no_result: 9

## Reliability labels

Cut points are validation quantiles `{'support_low': 0.25, 'support_high': 0.6, 'spread_low': 0.4, 'spread_high': 0.8}`: support_low=85 balls, support_high=394 balls, spread_low=0.266 runs, spread_high=0.435 runs. Validation label ordering monotone in MAE: **True**, so HIGH is issued.

| Label | Test overs | Test MAE | Interval coverage | Mean width |
|---|---:|---:|---:|---:|
| HIGH | 285 | 3.838 | 86.3% | 13.39 |
| MEDIUM | 1109 | 3.671 | 89.7% | 13.35 |
| LOW | 1343 | 4.056 | 85.6% | 13.35 |

Validation MAE by label: `{'HIGH': {'n': 110, 'mae': 3.465434278359499}, 'MEDIUM': {'n': 288, 'mae': 3.4842780936265836}, 'LOW': {'n': 262, 'mae': 4.1308727859371706}}`. Qualitative ranking of expected error, not a probability. HIGH is withheld unless the ordering held on validation.

## Experiments

Each experiment is judged on the validation season and never on the final test season.

**Sequence model (Phase 10).** A GRU over the last 24 deliveries alongside the standardized static features, trained on the same training seasons. Validation MAE 3.833 against the tabular ensemble's 3.798. Best validation blend weight on the sequence model: 0.25, giving 3.795. Blending at weight 0.25 moves validation MAE by only 0.0031 runs, 0.7 standard errors of the paired difference and inside sampling noise. The blend weight was also chosen on this same season, which flatters it further. The sequence model is not integrated.

**Feature selection (Phase 17).** Permutation importance over the served ensemble on validation. 57 of 167 features had a mean validation MAE increase at or below 0.0. Retrained CatBoost validation MAE: 3.808 with all features against 3.800 reduced. Removing 57 features moves validation MAE by 0.0079 runs, 1.0 standard errors of the paired difference and inside sampling noise. The full feature set is retained; the ranking is still useful for knowing which features carry the signal. Full ranking in experiments/permutation_importance.csv.

**Error analysis (Phase 36).** The worst 100 overs carry 11.9% of total test absolute error and average 21.0 actual runs against 9.8 overall; 90% of them are under-predictions. Conditions most over-represented in that tail:

- extras_of_two_or_more: 24% of the worst overs against 9% of all test overs (lift 2.74); MAE 4.30 when true against 3.84 when false
- two_or_more_boundaries: 84% of the worst overs against 38% of all test overs (lift 2.21); MAE 4.42 when true against 3.54 when false
- powerplay: 45% of the worst overs against 31% of all test overs (lift 1.45); MAE 4.07 when true against 3.79 when false
- sparse_bowler_history_under_120_balls: 19% of the worst overs against 16% of all test overs (lift 1.19); MAE 4.10 when true against 3.83 when false
- unseen_batter_bowler_matchup: 43% of the worst overs against 46% of all test overs (lift 0.94); MAE 3.88 when true against 3.88 when false

## Scope and limitations

- Only CatBoost is Optuna-tuned; the four alternatives use fixed regularized settings, so the candidate table understates their achievable performance.
- Model weights are fitted only through the training seasons, so the final test season is forecast across a multi-season recency gap. Refitting on later seasons would improve recency but would place the conformal and probability calibration blocks inside training data, so it is deliberately not done.
- Pace/spin, handedness and batting role require verified, dated metadata; not inferred from names.
- Historical retrospective forecasts condition on the opening pair and nominated bowler; provider must supply those before first delivery.
- Same-day matches never contribute to history. Prior completed test matches do contribute to later test-date history (rolling-origin evaluation).
- Rain-revised, shortened, no-result and ambiguous innings are excluded; these states are unsupported live.
- Sequential neural experiment is optional and not part of this evaluated artifact.
- Two-stage product is an approximation; model choice uses validation after reconciliation.

The latest test report is now inspected. Future feature/model experiments must use validation and reserve a later genuinely unseen season for a fresh final performance claim.

## Feature schema

`competition`, `venue`, `team_batting`, `team_bowling`, `batter`, `non_striker`, `bowler`, `phase`, `innings`, `over_number`, `balls_remaining`, `current_score`, `current_wickets`, `wickets_remaining`, `current_run_rate`, `chase_target`, `runs_required`, `required_run_rate`, `runs_last_1_overs`, `wickets_last_1_overs`, `boundaries_last_1_overs`, `sixes_last_1_overs`, `runs_last_2_overs`, `wickets_last_2_overs`, `boundaries_last_2_overs`, `sixes_last_2_overs`, `runs_last_3_overs`, `wickets_last_3_overs`, `boundaries_last_3_overs`, `sixes_last_3_overs`, `runs_last_5_overs`, `wickets_last_5_overs`, `boundaries_last_5_overs`, `sixes_last_5_overs`, `runs_last_6_deliveries`, `dot_rate_last_6_deliveries`, `runs_last_12_deliveries`, `dot_rate_last_12_deliveries`, `runs_last_18_deliveries`, `dot_rate_last_18_deliveries`, `runs_last_24_deliveries`, `dot_rate_last_24_deliveries`, `era_strike_rate`, `era_economy`, `era_balls_observed`, `batter_career_runs`, `batter_career_balls`, `batter_career_outs`, `batter_career_average`, `batter_career_strike_rate`, `batter_career_boundary_rate`, `batter_career_six_rate`, `batter_career_dot_rate`, `batter_career_economy`, `batter_season_runs`, `batter_season_balls`, `batter_season_outs`, `batter_season_average`, `batter_season_strike_rate`, `batter_season_boundary_rate`, `batter_season_six_rate`, `batter_season_dot_rate`, `batter_season_economy`, `batter_matchup_runs`, `batter_matchup_balls`, `batter_matchup_outs`, `batter_matchup_average`, `batter_matchup_strike_rate`, `batter_matchup_boundary_rate`, `batter_matchup_six_rate`, `batter_matchup_dot_rate`, `batter_matchup_economy`, `batter_innings_runs`, `batter_innings_balls`, `batter_innings_sr`, `batter_recent_5_mean_runs`, `batter_recent_5_sr`, `batter_recent_10_mean_runs`, `batter_recent_10_sr`, `batter_powerplay_sr`, `batter_middle_sr`, `batter_death_sr`, `batter_ewma0.15_runs_per_ball`, `batter_ewma0.15_boundary_rate`, `batter_ewma0.15_dot_rate`, `batter_ewma0.15_economy`, `batter_ewma0.4_runs_per_ball`, `batter_ewma0.4_boundary_rate`, `batter_ewma0.4_dot_rate`, `batter_ewma0.4_economy`, `batter_last_90d_innings`, `batter_last_90d_runs_per_ball`, `batter_last_90d_economy`, `batter_last_365d_innings`, `batter_last_365d_runs_per_ball`, `batter_last_365d_economy`, `batter_sr_vs_era`, `batter_boundary_rate_vs_era`, `batter_position_now`, `batter_mean_position`, `batter_innings_seen`, `batter_share_top3`, `batter_share_position_4_6`, `batter_share_position_7_plus`, `non_striker_career_runs`, `non_striker_career_balls`, `non_striker_career_outs`, `non_striker_career_average`, `non_striker_career_strike_rate`, `non_striker_career_boundary_rate`, `non_striker_career_six_rate`, `non_striker_career_dot_rate`, `non_striker_career_economy`, `non_striker_season_runs`, `non_striker_season_balls`, `non_striker_season_outs`, `non_striker_season_average`, `non_striker_season_strike_rate`, `non_striker_season_boundary_rate`, `non_striker_season_six_rate`, `non_striker_season_dot_rate`, `non_striker_season_economy`, `non_striker_matchup_runs`, `non_striker_matchup_balls`, `non_striker_matchup_outs`, `non_striker_matchup_average`, `non_striker_matchup_strike_rate`, `non_striker_matchup_boundary_rate`, `non_striker_matchup_six_rate`, `non_striker_matchup_dot_rate`, `non_striker_matchup_economy`, `non_striker_innings_runs`, `non_striker_innings_balls`, `non_striker_innings_sr`, `non_striker_recent_5_mean_runs`, `non_striker_recent_5_sr`, `non_striker_recent_10_mean_runs`, `non_striker_recent_10_sr`, `non_striker_powerplay_sr`, `non_striker_middle_sr`, `non_striker_death_sr`, `non_striker_ewma0.15_runs_per_ball`, `non_striker_ewma0.15_boundary_rate`, `non_striker_ewma0.15_dot_rate`, `non_striker_ewma0.15_economy`, `non_striker_ewma0.4_runs_per_ball`, `non_striker_ewma0.4_boundary_rate`, `non_striker_ewma0.4_dot_rate`, `non_striker_ewma0.4_economy`, `non_striker_last_90d_innings`, `non_striker_last_90d_runs_per_ball`, `non_striker_last_90d_economy`, `non_striker_last_365d_innings`, `non_striker_last_365d_runs_per_ball`, `non_striker_last_365d_economy`, `non_striker_sr_vs_era`, `non_striker_boundary_rate_vs_era`, `non_striker_position_now`, `non_striker_mean_position`, `non_striker_innings_seen`, `non_striker_share_top3`, `non_striker_share_position_4_6`, `non_striker_share_position_7_plus`, `bowler_career_runs`, `bowler_career_balls`, `bowler_career_outs`, `bowler_career_average`, `bowler_career_strike_rate`, `bowler_career_boundary_rate`, `bowler_career_six_rate`, `bowler_career_dot_rate`, `bowler_career_economy`, `bowler_season_runs`, `bowler_season_balls`, `bowler_season_outs`, `bowler_season_average`, `bowler_season_strike_rate`, `bowler_season_boundary_rate`, `bowler_season_six_rate`, `bowler_season_dot_rate`, `bowler_season_economy`, `bowler_match_legal_balls`, `bowler_match_conceded`, `bowler_match_wickets`, `bowler_match_economy`, `bowler_recent_5_economy`, `bowler_recent_10_economy`, `bowler_powerplay_economy`, `bowler_middle_economy`, `bowler_death_economy`, `bowler_ewma0.15_runs_per_ball`, `bowler_ewma0.15_boundary_rate`, `bowler_ewma0.15_dot_rate`, `bowler_ewma0.15_economy`, `bowler_ewma0.4_runs_per_ball`, `bowler_ewma0.4_boundary_rate`, `bowler_ewma0.4_dot_rate`, `bowler_ewma0.4_economy`, `bowler_last_90d_innings`, `bowler_last_90d_runs_per_ball`, `bowler_last_90d_economy`, `bowler_last_365d_innings`, `bowler_last_365d_runs_per_ball`, `bowler_last_365d_economy`, `bowler_economy_vs_era`, `bowler_overs_bowled_this_match`, `bowler_overs_remaining_quota`, `venue_run_rate`, `venue_balls`, `venue_boundary_rate`, `venue_wicket_rate`, `venue_over_run_rate`, `venue_over_balls`, `venue_over_boundary_rate`, `venue_over_wicket_rate`, `venue_phase_run_rate`, `venue_phase_balls`, `venue_phase_boundary_rate`, `venue_phase_wicket_rate`, `team_run_rate`, `team_balls`, `team_boundary_rate`, `team_wicket_rate`, `opposition_run_rate`, `opposition_balls`, `opposition_boundary_rate`, `opposition_wicket_rate`, `head_to_head_run_rate`, `head_to_head_balls`, `head_to_head_boundary_rate`, `head_to_head_wicket_rate`, `required_minus_current_rate`, `required_runs_per_wicket`, `balls_per_wicket_remaining`, `pressure_bucket`, `scoring_vs_era`

## Detailed artifacts

See latest.json for all slice metrics, component metrics, confusion matrices, reliability bins and parameters. See ../models/20260910T124741567173Z/worst_100.csv and feature_importance.csv for error analysis and native importance.

## Sources

Historical deliveries: [Cricsheet IPL downloads](https://cricsheet.org/downloads/). Field semantics and identities: [Cricsheet JSON specification](https://cricsheet.org/format/json/).