# Side-switch internal-peak penalty experiment — 2026-08-23

## Decision

A soft candidate-type penalty produces a small nested improvement, but the mechanism is
not stable enough to retain as a general rule. Against a matched zero-penalty control,
the nested ±4-second result moves from 27 TP/24 FP/23 FN and 53.47% F1 to **28 TP/24
FP/22 FN and 54.90% F1**. Strict F1 moves from 41.58% to 43.14%.

Reject the internal-peak penalty for promotion. Six of 11 held-out recordings select
zero penalty, and the penalized result suppresses the control's only correctly matched
internal proposal. The small aggregate gain comes from variant/threshold selection on
boundary candidates, not from better recovery of switches inside overlong production
ranges. Keep square-root class balancing, the full candidate union, and the fixed
local-peak plus soft-count decoder as the research path. The current user-designated
winner and production inference remain unchanged.

## Question and controlled comparison

The full-union ranker selected seven internal candidates in its first nested study, only
one of which matched a marker. This loop asks whether candidate kind can act as a soft
negative prior without becoming a hard production gate.

The experiment fixes:

- the immutable 704-row `FULL-UNION-V5-STATE42` feature artifact;
- square-root class balancing and L2 `0.1`;
- the 32-input primary and 34-input union-native feature views;
- adjacent-candidate local suppression;
- six free predictions per recording and a 0.5-logit penalty thereafter;
- all 50 exhaustive markers and ±4-second one-to-one matching.

It varies only an internal-candidate penalty of `0`, `0.5`, `1`, `1.5`, `2`, or `3`
logits. If `p` is the fitted switch probability, an internal peak receives

```text
sigmoid(logit(p) - penalty)
```

while a boundary score is unchanged. This is deliberately soft: no candidate is made
impossible, and there is no cadence, re-anchoring, hard count cap, suppression score, or
production hard gate.

Each outer fold excludes one recording from feature-view, penalty, and threshold
selection. A matched control independently selects the best zero-penalty feature view
and threshold within the same inner folds. This control isolates the penalty search
from the fixed square-root objective and decoder changes already established in prior
loops.

## Results

| Nested result | Proposals | TP | FP | FN | Precision | Recall | F1 | Internal TP / proposals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched zero-penalty control, ±4 s | 51 | 27 | 24 | 23 | 52.94% | 54.00% | 53.47% | 1 / 5 |
| Penalty selection, ±4 s | 52 | 28 | 24 | 22 | 53.85% | 56.00% | **54.90%** | 0 / 4 |
| Matched zero-penalty control, strict | 51 | 21 | 30 | 29 | 41.18% | 42.00% | 41.58% | 0 / 5 |
| Penalty selection, strict | 52 | 22 | 30 | 28 | 42.31% | 44.00% | **43.14%** | 0 / 4 |

Nested row average precision is 42.72%. The primary ±4-second macro per-video precision
and recall are 53.68% and 55.91%, respectively.

Penalty selection is split:

| Penalty logits | Outer folds selecting it |
| ---: | ---: |
| 0.0 | 6 |
| 0.5 | 2 |
| 1.0 | 2 |
| 1.5 | 1 |
| 2.0 or 3.0 | 0 |

The all-opened-development fit selects the 32-input view, a 1.0-logit penalty, and
threshold `0.4246529678370791`. Its same-data 55.91% F1 is not a held-out claim and is
not used for promotion. It emits no internal candidates, which reinforces the central
failure: only three of the 46 positive candidate labels are internal peaks, and the
current appearance/state representation does not reliably distinguish those three
from the 77 negative internal candidates.

## Interpretation

The aggregate nested delta is real under this protocol but does not validate the
intended hypothesis. The penalized decoder gains two boundary matches while losing the
control's one internal match. Candidate type is therefore acting as a weak selector
regularizer, not as evidence that internal dead-state peaks are intrinsically false.

A global type penalty is also risky for future data. Internal candidates were added to
recover real switches hidden inside overlong production ranges; systematically lowering
them recreates part of the upstream coverage failure. The next useful experiment should
improve the internal candidate representation or localization rather than encode the
current model's weakness as a permanent prior.

## On-device implications

The operation would cost one conditional logit subtraction per candidate and is fully
on-device compatible. Runtime cost is not the rejection reason. It is rejected because
the evidence is sparse, fold selection is unstable, and the causal control shows that
the only correct internal proposal is removed.

No browser, Android, review-UI, current-winner pointer, or production model files were
changed.

## Artifacts and reproduction

```bash
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-internal-peak-penalty.py
PYTHONPATH=. .venv/bin/python -m unittest discover -s analysis/tests -p 'test_side_switch*.py'
```

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-internal-peak-penalty-v1/model.json` | `1719ca433cb91c2be134ef43109fd0a41b63c0177e181eb744a80852e289e021` |
| `reports/side-switch/side-switch-internal-peak-penalty-v1-evaluation.json` | `4e0cc9a4050a7526817f48c5c1a773725c5e32a8f4d7f03db9088aa9e52aee08` |

The run completed 85 side-switch unit tests. Implementation is in
[`side_switch_full_union_ranker.py`](../../analysis/side_switch_full_union_ranker.py),
and the nested runner is
[`train-side-switch-internal-peak-penalty.py`](../../scripts/train-side-switch-internal-peak-penalty.py).

## Sources

- [Rare-event improvement plan and literature sources](./side-switch-rare-event-improvement-plan-2026-08-23.md)
- [Imbalance and recording-calibration experiment](./side-switch-imbalance-calibration-2026-08-23.md)
- [Full-union feature/ranker experiment](./side-switch-full-union-ranker-2026-08-23.md)
- [Full-trace candidate-union decision](./side-switch-candidate-union-2026-08-23.md)
