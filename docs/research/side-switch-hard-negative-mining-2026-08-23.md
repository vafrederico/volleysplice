# Side-switch recording-balanced hard-negative mining — 2026-08-23

## Decision

Recording-balanced hard-negative mining is a useful training change and should be
retained for the next side-switch model. The nested variant selector removes one false
proposal at unchanged recall relative to its matched no-mining control: ±4-second F1
moves from 53.47% to **54.00%** (27 TP/23 FP/23 FN). Strict F1 moves from 41.58% to
42.00%.

More importantly, the simplest fixed mining variant—34 inputs, the top two fitted
negatives per recording, and 2× negative loss—reaches **29 TP/23 FP/21 FN, 55.77%
precision, 58% recall, and 56.86% F1**. Its strict F1 is 45.10%. This fixed result is a
development selection across the declared variant grid, not untouched-test evidence.

Retain `union34-top2-x2` as the candidate objective for the next loop, but do not change
the user-designated current-winner pointer or production inference. The exact mining
strength varies across folds, and new reviewed sets are still required before any
threshold or runtime promotion.

## Method

The experiment returns to the retained 704-candidate union and its 46 one-to-one
positive candidate labels. It fixes:

- square-root class balancing;
- L2 `0.1`;
- the adjacent-candidate local-peak decoder;
- six free outputs followed by a 0.5-logit soft count penalty;
- raw, uncalibrated scores; and
- the 50 exhaustive markers with ±4-second one-to-one event matching.

For each fit scope:

1. fit the ordinary square-root-weighted logistic head;
2. score that same fit scope;
3. rank labeled negative candidates separately inside each recording;
4. multiply the loss of the top `K` negatives; and
5. refit the same linear head from scratch.

The grid compares `K = 0`, top 2 at 2×, top 4 at 2×, top 4 at 4×, and top 8 at 2×,
crossed with the 32-input and 34-input feature views. Mining is recording-balanced so a
longer game cannot contribute all hard negatives. Candidate selection, feature view,
mining setting, and threshold never use the outer held recording's labels.

The initial mining score is in-fit rather than cross-fitted. This is intentional hard
example mining, but it can favor model-specific outliers. The outer recording holdout
measures whether the refitted decision transfers.

## Results

| Result | Proposals | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Loop 5 nested variant selection | 54 | 27 | 27 | 23 | 50.00% | 54.00% | 51.92% |
| Matched no-mining control | 51 | 27 | 24 | 23 | 52.94% | 54.00% | 53.47% |
| Nested mining-variant selector | 50 | 27 | 23 | 23 | 54.00% | 54.00% | **54.00%** |
| Fixed `union34-top2-x2` | 52 | 29 | 23 | 21 | 55.77% | 58.00% | **56.86%** |
| Fixed `union34-top4-x4` | 54 | 29 | 25 | 21 | 53.70% | 58.00% | 55.77% |
| Fixed `union34-top4-x2` | 56 | 29 | 27 | 21 | 51.79% | 58.00% | 54.72% |

For strict matching, the matched control is 21 TP/30 FP/29 FN and 41.58% F1; nested
variant selection is 21 TP/29 FP/29 FN and 42.00% F1. Fixed `union34-top2-x2` reaches
23 TP/29 FP/27 FN and 45.10% F1.

The fixed top-2 variant's row average precision is 44.50%, compared with about 42.5%
for its no-mining feature-view controls. The mixed nested selector's AP is 43.37%.

Mining-setting selection across the 11 outer folds is:

| Setting | Folds |
| --- | ---: |
| No mining | 2 |
| Top 2 at 2× | 4 |
| Top 4 at 2× | 2 |
| Top 4 at 4× | 1 |
| Top 8 at 2× | 2 |

Nine folds therefore select some mining, but no single setting dominates a majority.
The 34-input view wins eight folds and the 32-input view wins three. This explains why
the adaptive nested selector gains only one FP while the post-grid fixed top-2 result is
stronger: mining is directionally stable, but exact per-fold hyperparameter selection
is noisy on ten fit recordings.

The all-opened-development artifact selects `union34-top2-x2`, threshold
`0.39884973953581804`, and the fixed local-peak plus soft-count decoder. Its same-data
57.14% F1 is not a held-out claim.

## On-device implications

Mining affects training only. The exported model remains one 34-input logistic head;
there is no additional candidate, calibration pass, second model, or runtime state.
This is therefore one of the lowest-cost improvements tested so far.

No browser or Android port was implemented.

## Artifacts and reproduction

```bash
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-hard-negative-mining.py
PYTHONPATH=. .venv/bin/python -m unittest discover -s analysis/tests -p 'test_side_switch*.py'
```

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-hard-negative-mining-v1/model.json` | `c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3` |
| `reports/side-switch/side-switch-hard-negative-mining-v1-evaluation.json` | `e67088b36d177d68c24587efcb186eab4201b5294532be1fdf978ef13aaacc4b` |

The run completed all 86 side-switch unit tests.

Implementation is in
[`side_switch_full_union_ranker.py`](../../analysis/side_switch_full_union_ranker.py),
and the nested runner is
[`train-side-switch-hard-negative-mining.py`](../../scripts/train-side-switch-hard-negative-mining.py).

## Sources

- [Rare-event improvement plan and literature sources](./side-switch-rare-event-improvement-plan-2026-08-23.md)
- [Imbalance and recording-calibration experiment](./side-switch-imbalance-calibration-2026-08-23.md)
- [Full-union feature/ranker experiment](./side-switch-full-union-ranker-2026-08-23.md)
- [Expanded internal-candidate rejection](./side-switch-expanded-internal-candidates-2026-08-23.md)
