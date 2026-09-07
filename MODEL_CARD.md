# Model card — Crease

## Intended use

Decision-support forecasts at the beginning of regulation IPL T20 overs when the opening striker, non-striker and nominated bowler are known. Predictions are conditional expectations, not certain scores. Forecasts are not a guarantee of betting profitability.

## Data and target

Source: [Cricsheet IPL JSON](https://cricsheet.org/downloads/) and its embedded canonical player registry. Runs include all deliveries in the next recorded over, including illegal deliveries and match-ending partial overs. Batter targets refer to the two players present at over start. A separate replacement-batter component accounts for runs after dismissals. Wicket target includes run-outs and excludes retired hurt/not out. Bowler-conceded runs exclude byes, leg byes and penalty runs. Boundary flags respect Cricsheet's non-boundary marker.

Excluded: rain/DLS revisions, shortened scheduled innings, no-results, missing-data flags, miscounted overs, unattributable innings penalties, noncontiguous overs and target discrepancies. Super overs are excluded. These exclusions mean evaluation does not describe all live T20 states, and may select for uninterrupted matches. The service rejects known unsupported states.

## Historical availability

Every feature row records a history cutoff strictly earlier than its match date. All same-day matches are held out of one another's history. Within-innings momentum uses strictly prior overs. Identities at the first recorded delivery are assumed nominated before the over; actual nomination timestamps are unavailable in this dataset. This assumption must be satisfied by the live provider. Player replacements can invalidate it and need a refreshed pre-ball state.

Missing player-role/pace/spin/handedness metadata is not guessed. Cold starts use categorical unknown handling and rate shrinkage toward prior league observations. Role/cluster fallback is not implemented without verified role data. Team name aliases are explicit. Venue names are canonicalized: locality suffixes and punctuation variants are collapsed onto the leading ground name, and genuine renames come from the reviewed `configs/venue_aliases.csv`, each with a stated reason. This maps 60 recorded venue strings onto 36 grounds, so venue history is no longer split across spellings of the same ground. Merging renamed grounds assumes scoring behaviour is continuous across the rename; that is a modelling choice, not a fact, and it is least safe for Narendra Modi Stadium, which was rebuilt rather than only renamed.

## Validation

See reports/EVALUATION.md and latest.json for measured results. No accuracy claims are fabricated. Match-grouped chronological partitions isolate model selection, probability calibration, conformal calibration and final test. Baseline promotion checks validation MAE only. That is a development gate, not a production safety certificate or assurance that every candidate beats baselines.

## Uncertainty and reliability

Two conformal intervals are produced from a calibration block no model was fitted on. The headline 80% interval targets marginal coverage of the single next over, which is the quantity a next-over forecast actually claims. A second, much wider match-block band targets simultaneous coverage of every over in a match; it answers a different question and should not be read as the forecast's uncertainty. An earlier version of this system reported only the match-block band, which over-covered at roughly 99% against an 80% target and was too wide to be decision-useful.

Lower bounds are truncated at zero and both bounds are rounded outward to integers. Coverage is measured on those displayed integer bounds, so observed coverage sits slightly above nominal; that is a property of the display, not a modelling claim. Overs within a match are correlated and seasons shift, so exchangeability holds only approximately and the finite-sample guarantee is weaker than it looks. Reported coverage is empirical, applies to the supported population only, and is not a promise about future matches.

Reliability labels are a qualitative ranking of expected error, never a probability. They combine three measured quantities: disagreement between the trained candidate models, the prior-ball history of the least-seen of the three named players, and the batter-bowler matchup sample. Cut points are quantiles of the validation season, not hand-picked numbers. HIGH is offered only when validation error actually ranked HIGH below MEDIUM below LOW; when that ordering fails, the rule degrades to MEDIUM/LOW and HIGH is withheld entirely. The evaluation report states which case applied and gives held-out MAE, interval coverage and interval width for each label, so the label can be checked rather than trusted.

## Explanation

CatBoost native TreeSHAP contributions explain the CatBoost component before any ensemble weighting, clipping or component reconciliation. The API states this scope and returns its base value and prediction. The ten largest contributions are shown; omitted contributions mean the displayed subset alone need not sum to the prediction. They are associations, not causal effects.

## Current release boundaries

Implemented: genuine historical training; five tabular comparisons; Optuna; four baselines; direct versus two-stage batter models; five calibrated classification heads; conformal intervals; TreeSHAP; model artifacts and MLflow; authenticated API; provider-neutral ingestion; replay dashboard; durable feedback and per-version monitoring; unit/integration tests and container configuration.

Also implemented, outside the served package: a GRU sequence comparison, permutation importance with a feature-removal ablation, automated failure-pattern analysis of the largest held-out errors, and a normalized SQL export of the as-of historical store. Each is decided on the validation season by a paired significance test on per-row absolute errors, not by whichever number happens to look smaller.

Both model experiments returned negative results and neither was adopted. A GRU over the last 24 deliveries scored 3.833 validation MAE against the tabular ensemble's 3.798; the best blend moved MAE by 0.0031 runs, 0.7 standard errors, and the blend weight was chosen on that same season. Dropping the 57 features whose permutation cost was non-positive moved MAE by 0.0079 runs, 1.0 standard errors. Sequence order and the low-importance features are therefore not shown to add anything here; that is a measurement about this dataset and these settings, not proof that they cannot help.

Error analysis is blunter. The 100 worst held-out overs carry 11.9% of all test absolute error, average 21.0 actual runs against 9.8 overall, and 90% of them are under-predictions. The model regresses toward the conditional mean and misses explosive overs; overs with two or more boundaries are over-represented in that tail by a factor of about two. This is the central weakness of the current system.

Not yet validated or implemented: verified role/pace/spin metadata and the pace-versus-spin and top-order-versus-tail slices that depend on it; learned player clusters; tuning of the four alternative models, which use fixed regularized settings while only CatBoost is searched; production load testing; commercial live-provider translation; automatic history-store event ingestion; automated retraining scheduling; alert delivery; model registry lifecycle in a remote MLflow server; formal database migrations and restore testing. The SQL feature store is an as-of export for inspection and audit; serving still reads the local history artifact, and neither is a distributed online feature store.

The PostgreSQL container configuration must be exercised on the target deployment. Local tests use SQLite. Do not label this release fully production-grade until the operational gaps above are addressed.
