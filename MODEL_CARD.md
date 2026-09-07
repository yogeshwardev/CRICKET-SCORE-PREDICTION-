# Model card — Crease

## Intended use

Decision-support forecasts at the beginning of regulation IPL T20 overs when the opening striker, non-striker and nominated bowler are known. Predictions are conditional expectations, not certain scores. Forecasts are not a guarantee of betting profitability.

## Data and target

Source: [Cricsheet IPL JSON](https://cricsheet.org/downloads/) and its embedded canonical player registry. Runs include all deliveries in the next recorded over, including illegal deliveries and match-ending partial overs. Batter targets refer to the two players present at over start. A separate replacement-batter component accounts for runs after dismissals. Wicket target includes run-outs and excludes retired hurt/not out. Bowler-conceded runs exclude byes, leg byes and penalty runs. Boundary flags respect Cricsheet's non-boundary marker.

Excluded: rain/DLS revisions, shortened scheduled innings, no-results, missing-data flags, miscounted overs, unattributable innings penalties, noncontiguous overs and target discrepancies. Super overs are excluded. These exclusions mean evaluation does not describe all live T20 states, and may select for uninterrupted matches. The service rejects known unsupported states.

## Historical availability

Every feature row records a history cutoff strictly earlier than its match date. All same-day matches are held out of one another's history. Within-innings momentum uses strictly prior overs. Identities at the first recorded delivery are assumed nominated before the over; actual nomination timestamps are unavailable in this dataset. This assumption must be satisfied by the live provider. Player replacements can invalidate it and need a refreshed pre-ball state.

Missing player-role/pace/spin/handedness metadata is not guessed. Cold starts use categorical unknown handling and rate shrinkage toward prior league observations. Role/cluster fallback is not implemented without verified role data. Team name aliases are explicit; venue-name variants are currently separate categories, so venue samples can be fragmented.

## Validation

See reports/EVALUATION.md and latest.json for measured results. No accuracy claims are fabricated. Match-grouped chronological partitions isolate model selection, probability calibration, conformal calibration and final test. Baseline promotion checks validation MAE only. That is a development gate, not a production safety certificate or assurance that every candidate beats baselines.

## Uncertainty and reliability

80% match-block conformal intervals aim at simultaneous over coverage within a new exchangeable match. They are intentionally wider than marginal-over intervals. Chronological distribution shift can break exchangeability. Reported test coverage is empirical and applies to the supported population only. Reliability labels are qualitative support indicators, not probabilities: LOW for sparse player history, otherwise MEDIUM. HIGH is not issued without independent evidence supporting that claim.

## Explanation

CatBoost native TreeSHAP contributions explain the CatBoost component before any ensemble weighting, clipping or component reconciliation. The API states this scope and returns its base value and prediction. The ten largest contributions are shown; omitted contributions mean the displayed subset alone need not sum to the prediction. They are associations, not causal effects.

## Current release boundaries

Implemented: genuine historical training; five tabular comparisons; Optuna; four baselines; direct versus two-stage batter models; five calibrated classification heads; conformal intervals; TreeSHAP; model artifacts and MLflow; authenticated API; provider-neutral ingestion; replay dashboard; durable feedback and per-version monitoring; unit/integration tests and container configuration.

Not yet validated or implemented: sequence/transformer comparison; verified role/pace/spin metadata and related slices; learned player clusters; feature-removal/permutation ablation; production load testing; commercial live-provider translation; automatic history-store event ingestion; automated retraining scheduling; alert delivery; full normalized PostgreSQL feature-store tables; model registry lifecycle in a remote MLflow server; formal database migrations and restore testing. Local aggregate history is an as-of artifact, not a distributed online feature store.

The PostgreSQL container configuration must be exercised on the target deployment. Local tests use SQLite. Do not label this release fully production-grade until the operational gaps above are addressed.
