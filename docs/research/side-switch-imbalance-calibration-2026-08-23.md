# Side-switch imbalance and recording-calibration experiment — 2026-08-23

## Decision

Changing the rare-event loss prior is modestly useful; label-free recording score
calibration is not. Nested leave-one-recording-out (LOO) variant selection improves the
full-union ranker's ±4-second result from 27 TP/29 FP/23 FN and 50.94% F1 to
**27 TP/27 FP/23 FN and 51.92% F1**. Strict F1 also moves from 39.62% to 40.38%.

Retain square-root class balancing as a candidate objective for the next model, but do
not promote this artifact or replace the current research-winner pointer. Five variants
win at least one outer fold, and the approximately one-point nested gain is too small
for the opened 11-recording scope. Reject robust-logit and percentile recording
calibration for threshold cleanup.

## Scope and controlled variables

This loop reuses the exact 704-candidate `FULL-UNION-V5-STATE42` artifact and its 46
one-to-one positive candidate labels. It decodes against all 50 exhaustive markers.
The source videos, candidate generator, features, and matching contract are unchanged.

To isolate objective and calibration effects, the experiment fixes the settings that
were stable in Loop 4:

- L2 `0.1`, selected in all 11 prior outer folds;
- adjacent-candidate local suppression;
- six free predictions per recording;
- a 0.5 logit penalty after the sixth prediction;
- no cadence, re-anchoring, or hard count cap.

Each outer held recording remains excluded from variant and threshold selection. Inner
grouped LOO probabilities on the other ten recordings select among ten variants: two
feature views crossed with natural, square-root-balanced, and fully balanced raw-score
objectives, plus robust-logit and percentile calibration for the fully balanced head.

## Implemented objectives

Let `e` be the class-balance exponent. Per-class sample weights are proportional to
`(N / (2 N_class))^e`, normalized to mean one:

- `e=0`: natural empirical prevalence;
- `e=0.5`: square-root balancing;
- `e=1`: equal total positive and negative weight, matching the prior ranker.

All variants still produce the same small linear logistic head at inference time.

Two label-free recording transforms were also tested:

- robust logit: within-recording median centering and MAD/std scaling, then sigmoid;
- percentile: tied empirical rank within the recording's complete candidate list.

These transforms preserve within-recording order but try to make one global threshold
less sensitive to recording score shift.

## Primary results

| Variant | Proposals | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Loop 4 nested control | 56 | 27 | 29 | 23 | 48.21% | 54.00% | 50.94% |
| Loop 5 nested variant selection | 54 | 27 | 27 | 23 | 50.00% | 54.00% | **51.92%** |
| Best fixed F1: natural/raw/union34 | 60 | 29 | 31 | 21 | 48.33% | 58.00% | **52.73%** |
| Square-root/raw/union34 | 53 | 27 | 26 | 23 | 50.94% | 54.00% | 52.43% |
| Square-root/raw/primary32 | 50 | 26 | 24 | 24 | 52.00% | 52.00% | 52.00% |
| Balanced/percentile/primary32 | 66 | 30 | 36 | 20 | 45.45% | 60.00% | 51.72% |
| Balanced/raw/primary32 | 56 | 27 | 29 | 23 | 48.21% | 54.00% | 50.94% |
| Balanced/robust-logit/primary32 | 65 | 29 | 36 | 21 | 44.62% | 58.00% | 50.43% |

The best fixed natural-loss variant recovers two additional events but adds two false
positives relative to the nested selector. Square-root balancing supplies the better
precision/recall compromise and wins 8/11 outer folds across its two feature views.

Percentile calibration has the highest candidate AP (about 47.1%) and reaches 60%
event recall, but it makes the threshold too permissive and adds 7–9 false positives
relative to the raw-score control. Robust-logit normalization also adds false
positives. Recording shift is therefore not solved by discarding absolute score scale.

The nested selector's strict result is 21 TP/33 FP/29 FN, 38.89% precision, 42% recall,
and 40.38% F1. Timing remains a separate limitation.

## Does predicting the opposite help?

Not by itself for this binary linear formulation. The experiment fits a second head to
`no-switch = 1 - switch` with the same features, regularization, and symmetric class
weight formula. Its switch-equivalent score differs from the original switch head by
at most `2.22e-16`:

```text
P(switch | x) = 1 - P(no-switch | x)
```

This is a useful control, not a reason to abandon negative modeling. A separate
negative task can help only when it adds information or asymmetry—for example explicit
same-side continuity targets, distinct hard-negative sampling, abstention, or a
one-class representation. Re-labeling the same rows and fitting the same binary loss
cannot add evidence.

## Stability and disposition

- Square-root raw `primary32` wins 4 outer folds.
- Square-root raw `union34` wins 4.
- Balanced raw `primary32`, balanced percentile `primary32`, and natural raw `union34`
  each win one.
- The all-development selection is square-root/raw/primary32 at threshold
  `0.44626951586737873`; its same-scope selected F1 is 54.95% and is not the primary
  held-out claim.
- Recording `193307688` still has zero recall; objective/calibration changes do not fix
  its representation failure.

The loop therefore keeps the Loop 4 candidate/features/decoder architecture and carries
square-root balancing forward as a candidate. It rejects recording calibration and the
redundant opposite head.

## On-device implications

Objective changes affect training only. The selected inference model remains a 32-input
logistic head and needs no calibration pass. Percentile and robust-logit alternatives
would require the complete recording candidate score distribution, but neither is
retained. No browser or Android port was implemented.

## Artifacts and reproduction

```bash
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-full-union-calibration.py
PYTHONPATH=. .venv/bin/python -m unittest discover -s analysis/tests -p 'test_side_switch*.py'
```

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-full-union-calibrated-v1/model.json` | `894b4c553ea94889d0c6257b4c09f8a6e6960ef75fd204f54ccbdc7593cb93aa` |
| `reports/side-switch/side-switch-full-union-calibrated-v1-evaluation.json` | `fd8ca4a8695e289cc53ec58c2042819c466bf2bbf13f0b573ac87e891dd6343f` |

The run completed 84 side-switch unit tests. Exact objective/calibration implementation
is in
[`side_switch_full_union_ranker.py`](../../analysis/side_switch_full_union_ranker.py),
and the nested experiment is
[`train-side-switch-full-union-calibration.py`](../../scripts/train-side-switch-full-union-calibration.py).

## Sources

- [Rare-event improvement plan and literature sources](./side-switch-rare-event-improvement-plan-2026-08-23.md)
- [Full-union feature/ranker experiment](./side-switch-full-union-ranker-2026-08-23.md)
- [Full-trace candidate-union decision](./side-switch-candidate-union-2026-08-23.md)
- [Same-side continuity-verifier decision](./side-switch-continuity-verifier-2026-08-23.md)

