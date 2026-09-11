# Cricket AI — delivery simulation experiment

**Decision: USE AS AUXILIARY TAIL MODEL** — and the primary hypothesis largely failed.
The champion remains promoted and unchanged.

## 1. Hypothesis

The champion optimizes a conditional mean, so it was thought blind to explosive overs:
125 overs of 20+ runs are 4.6% of the test season but carry 13.2% of its absolute error,
and its run-bucket argmax never selects the 15–19 or 20+ classes. The hypothesis was that
modelling each delivery's outcome distribution and simulating the over forward would
recover tail information the champion misses.

**A correction to that framing.** The champion's run-bucket classifier does emit tail
probabilities; only its *argmax* never lands on a high bucket. Those probabilities already
rank the tail at AUC 0.648 for 15+ and 0.698 for 20+. The earlier statement that the model
"never predicts explosive overs" was about argmax, and it overstated the gap the simulator
had to close.

## 2. Dataset

286,198 delivery rows from the same Cricsheet archive, `deliveries_model.parquet`,
SHA-256 `a1df0dfd46f8f4f7…`. 81 features. Verified against source: 9,545 wides,
1,170 no-balls, 4.95% wicket rate, 22,488 post-wicket replacement deliveries.

Partitions are the champion's exact date windows, so both models train on identical data
and are scored on identical rows: train 235,625 / validation 16,902 / refit_extra 8,376 /
probability_calibration 4,109 / interval_calibration 4,070 / test 17,116 deliveries.

## 3. Delivery target design

Four compatible heads rather than one exclusive class, because a wicket, extras and runs
can all occur on the same ball:

| head | classes |
|---|---|
| runs | 0, 1, 2, 3, 4, 6, other |
| extras | 0, 1, 2, 3, 4+ |
| legality | legal, wide, no-ball |
| wicket | binary |

## 4. Leakage controls

Pre-over context comes from the existing over-level row, already built strictly from
matches before the match date. Within-over state evolves only from deliveries that have
already happened. `delivery_features` is the single builder used for both training rows
and simulator steps, so the two cannot drift. A test asserts that target columns placed on
the input row cannot change a simulation.

## 5. Models

CatBoost per head, `max_ctr_complexity=1`, selected on the validation season then refit on
train+validation+refit_extra. 3,348 seconds total. Test-block discrimination is weak but
above base rate — wicket PR-AUC 0.078 against a 0.050 base, six 0.130 against 0.081,
four 0.178 against 0.133.

## 6. Joint event generator — independent heads were unrealistic

Checked before simulating, and the check changed the design. Measured on training
deliveries only:

| constraint | historical |
|---|---|
| extras on a legal ball | only when batter runs = 0 (4.8%); exactly 0.0000 at 1, 2, 3, 4, 6 |
| wicket lift by runs | 2.66 on a dot, 0.076 on a single, 0.00 on a four, 0.002 on a six |
| wicket lift, wide / no-ball | 0.114 / 0.169 |

Sampling the heads independently invents byes alongside boundaries and wickets on sixes.
The simulator now samples runs and extras first, then conditions dismissal on both.
On a sample over this moved simulated wickets per over from 0.250 to 0.306 against a
historical 0.312.

**Remaining defect:** simulated extras per over are 0.95 against a historical 0.46. The
extras head over-fires. This is reported rather than tuned away, and it inflates simulated
totals slightly.

## 7. Simulation

Six legal deliveries end the over, so wides and no-balls extend it; strike rotates on odd
runs and at the end of the over; a chase stops when the target is reached; an innings stops
at the tenth wicket. 28 tests pin these rules using deterministic stub models, including
seed reproducibility.

A batter arriving after a mid-over wicket is not described by the pre-over row. He is
represented by era-level values plus an explicit flag — a documented approximation, not a
silent reuse of the dismissed batter's record.

## 8. Convergence

Seed-to-seed standard deviation across 7 representative match states, 3 seeds:

| draws | E[runs] | P(15+) | P(20+) | p50 ms | p95 ms |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.0764 | 0.0071 | 0.0033 | 620 | 708 |
| **2,500** | **0.0307** | **0.0049** | **0.0010** | **858** | **1,005** |
| 5,000 | 0.0388 | 0.0026 | 0.0008 | 1,228 | 1,433 |
| 10,000 | 0.0319 | 0.0039 | 0.0006 | 2,139 | 2,683 |
| 25,000 | 0.0206 | 0.0010 | 0.0006 | 4,120 | 4,545 |

2,500 draws chosen: the smallest count whose tail probabilities vary by under 0.005 across
seeds. 25,000 costs five times the latency for no decision-relevant change.

## 9. Champion comparison — 2,737 held-out 2026 overs

| | Champion | Simulator | Hybrid |
|---|---:|---:|---:|
| MAE | 3.8776 | **3.8597** | 3.8597 |
| RMSE | **4.8854** | 4.8997 | 4.8997 |
| R² | **0.0950** | 0.0896 | 0.0896 |
| Median AE | 3.323 | **3.192** | 3.192 |
| within ±1 | 16.6% | 16.7% | 16.7% |
| within ±3 | 45.5% | **47.3%** | 47.3% |
| Bucket accuracy | 26.7% | **27.6%** | — |
| Bucket macro-F1 | 0.2023 | **0.2072** | — |
| 15–19 recall | **0.042** | 0.023 | — |
| 20+ recall | 0.000 | 0.000 | — |
| 15+ PR-AUC | 0.2652 | **0.2854** | — |
| 20+ PR-AUC | 0.1013 | **0.1041** | — |
| Bucket log loss | 1.6521 | **1.6392** | — |
| Bucket Brier | 0.7932 | **0.7886** | — |
| Ranked probability score | 0.1489 | **0.1485** | — |
| 80% coverage / width | 0.8663 / 13.15 | **0.8444 / 12.40** | — |
| 95% coverage / width | not produced | 0.9646 / 18.19 | — |
| p50 latency | **328 ms** | 858 ms | — |

The hybrid weight, fitted on the 2025 calibration blocks, came out at **1.0** — all weight
on the simulator. The hybrid therefore collapses to the simulator and is not a distinct
option.

## 10. Explosive overs — the primary objective, and it failed

| actual bucket | n | champion MAE | simulator MAE | champion pred | simulator pred | mean P(15+) | mean P(20+) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15–19 | 355 | **6.050** | 6.488 | 10.43 | 9.99 | 0.193 | 0.052 |
| 20+ | 125 | **11.603** | 12.051 | 11.00 | 10.55 | 0.228 | 0.068 |

**The simulator is worse on explosive overs, not better.** It predicts *lower* than the
champion exactly where the actual totals are highest, and its 15–19 bucket recall is worse
(0.023 against 0.042). Neither model ever selects the 20+ bucket by argmax — that is
arithmetic, not a fixable defect: a distribution whose mode sits at 8–10 will not have its
argmax land on 20+, however the probabilities were produced.

## 11. Statistical testing

Paired bootstrap, 5,000 resamples, identical overs:

| comparison | mean | 95% CI | P(simulator better) |
|---|---:|---|---:|
| MAE improvement | +0.0180 | [−0.0145, +0.0514] | 0.860 |
| AUC(15+) | +0.0146 | **[+0.0006, +0.0285]** | 0.980 |
| AUC(20+) | +0.0184 | [−0.0063, +0.0418] | 0.931 |

Only the 15+ ranking gain clears zero, and it barely does. The MAE gain and the 20+ gain
are both inside noise. A 0.018-run MAE difference is also practically invisible.

## 12. Limitations

- Extras are over-generated (0.95 per over against 0.46), inflating simulated totals.
- Incoming batters after a mid-over wicket use era-level values, not a real lineup.
- No-ball dismissals are modelled through the lift term only; free-hit rules are not encoded.
- The heads use one fixed hyperparameter setting; no Optuna search was run.
- 2,500 draws leaves P(15+) with a seed standard deviation of 0.005, so two runs can differ
  in the third decimal.
- 2026 was also a fold in the earlier walk-forward work, so it is not pristine for this
  comparison either. That contamination would flatter the challenger, and it still lost on
  the primary objective.

## 13. Decision

**USE AS AUXILIARY TAIL MODEL.**

Not PROMOTE: the MAE gain is not significant, explosive-over accuracy is worse, RMSE and R²
regress, calibration is worse (ECE 0.0068 against 0.0010 on 20+), and latency is 2.6×.

Not HYBRID: the fitted weight was 1.0, so blending produces nothing distinct.

Not REJECT: two contributions are real. The **prediction intervals are genuinely better** —
84.4% coverage at width 12.40 against 86.6% at 13.15, simultaneously closer to the 80%
target *and* narrower, which is the one unambiguous win here. And every proper scoring rule
on the full distribution favours the simulator slightly (log loss 1.6392 vs 1.6521, Brier
0.7886 vs 0.7932, RPS 0.1485 vs 0.1489), which is the right lens for a probabilistic
forecast.

So the simulator earns a place supplying the **distribution and intervals**, while the
champion keeps expected runs. It has not earned a place as the tail detector it was built
to be.

## 14. Answer to the research question

*Does modelling delivery-level outcomes and simulating the over provide useful information
about explosive 15+ and 20+ overs that the production model misses?*

**Largely no.** The champion already carried tail signal in its bucket probabilities. The
simulator improves 15+ ranking by a marginal +0.015 AUC and 20+ by a non-significant
+0.018, does not improve explosive-over point accuracy (it degrades it), and does not
change 20+ argmax recall, which stays at zero for both. The useful result is a better
calibrated, narrower prediction interval — a genuine but different benefit from the one
the experiment set out to find.
