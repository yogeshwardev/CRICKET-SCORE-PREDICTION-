# Delivery simulator — work in progress

**Status: INCOMPLETE. Stopped deliberately mid-experiment. Production is unaffected.**

Last clean commit: `2b9e566`. Working tree clean at the time of stopping.

## Production is untouched and healthy

The promoted champion `20260907T150806188678Z` is still serving, unchanged:
MAE 3.8776, R² 0.095, ECE ≤ 0.030, 80%/95% conformal intervals. Nothing in this
experiment was wired into the serving path. 79 tests pass.

## What is finished

| Item | State |
|---|---|
| `src/cricket_ai/delivery.py` | Complete. Multi-head targets, shared feature builder, vectorized Monte Carlo simulator. |
| `tests/test_delivery_simulator.py` | Complete. 24 cricket-rule tests passing. |
| `data/processed/deliveries_model.parquet` | Built. 286,198 rows, 81 features. Verified: 9,545 wides, 1,170 no-balls, 4.95% wicket rate, 22,488 post-wicket replacement deliveries. |
| `scripts/experiments/delivery_simulator.py` | Complete, with full provenance recording. |
| `scripts/experiments/simulator_eval.py` | Written but NOT yet run. Contains one known bug — see below. |

Chronological partitions confirmed to match the champion's exact date windows:
train 235,625 / validation 16,902 / refit_extra 8,376 / probability_calibration 4,109 /
interval_calibration 4,070 / test 17,116 deliveries.

## What is NOT finished

`models/delivery_simulator_challenger/` is **empty**. The four heads were training when
the run was stopped; the multiclass `runs` head (7 classes, 260,903 rows) had been
fitting for ~25 minutes and the script had not reached its save step. No artifact was
written, so nothing is half-saved or corrupt.

Everything downstream of that is therefore also outstanding: head validation,
calibration, joint-event realism check, convergence study, latency benchmark,
simulation of the test overs, champion/hybrid comparison, and the final decision.

## Known bug to fix before running the evaluation

`simulator_eval.py` builds its 95% interval from `p10`–`p95`, which is really about an
85% nominal interval. A 95% interval needs `p2.5`–`p97.5`, and those quantiles are not
currently produced. Fix `summarize()` in `src/cricket_ai/delivery.py` to include
`2.5` and `97.5` in its quantile list, then use them in `interval_from_quantiles`.

This was spotted by reading the code, not by running it, so it has never produced a
wrong published number.

## To resume

```bash
git status                      # confirm clean before training
# 1. fix the p2.5/p97.5 quantile bug above, commit it
CREASE_THREADS=10 python scripts/experiments/delivery_simulator.py --iterations 400
python scripts/experiments/simulator_eval.py --draws 2000 --sample 800
```

The `--sample` flag exists because simulating all 2,737 test overs is slow; a fixed
subsample is statistically adequate for the tail questions and the sample size is
recorded in the output. Budget roughly 40 minutes for the heads and comparable time
for the simulation, depending on draw count.

## The research question, still unanswered

*Does modelling delivery-level outcome distributions and simulating the over provide
useful information about explosive 15+ and 20+ overs that the production
conditional-mean model currently misses?*

No evidence has been gathered yet either way. The motivating measurement stands:
125 overs of 20+ runs are 4.6% of the test season but carry 13.2% of all absolute
error, and the champion's run-bucket classifier never predicts the 15-19 or 20+
buckets at all.

A caution for whoever resumes: the champion's event heads run at AUC 0.60-0.66, so the
realistic prize here is **calibrated P(15+) and P(20+)**, not a better MAE. Outcome
"AUXILIARY TAIL MODEL" is at least as likely as "PROMOTE", and both are acceptable
results.
