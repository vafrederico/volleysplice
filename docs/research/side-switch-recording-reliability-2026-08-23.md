# Side-switch recording-reliability head — 2026-08-23

## Decision

Reject the recording-level reliability/abstention head. The nested selector chooses
the unmodified promoted winner in all 11 outer folds. The best fixed reliability
setting gains one TP but adds six FP, moving ±4-second F1 from 56.86% to 55.05%.
Stronger offsets add still more proposals and fall to about 52% F1.

The current `union34-top2-x2` research winner remains unchanged. No browser, Android,
or production port was implemented.

## Method

The experiment keeps the promoted 34-input classifier, recording-balanced top-2/2×
hard-negative mining, threshold selection, local-peak suppression, and soft count cost
fixed. It adds a 13-input ridge head that predicts a recording-specific threshold-logit
offset from label-free summaries:

- camera-shift P90 and alignment-response P10;
- player-side-separation and proposal-coverage P10;
- palette instability and global appearance-change P90;
- serve-anchor-error P90 and serve-confidence P10;
- candidate-score mean, standard deviation, P90, and fraction above the base threshold;
- candidates per minute.

The retained union artifact does not contain a direct blur scalar, so camera,
alignment, palette, and appearance measurements are only quality proxies.

Inside each outer fit scope, base scores are cross-fitted by recording. A per-recording
target is the logit difference between that video's optimal event threshold and the
fit-global threshold, clipped to ±2. Reliability predictions for variant selection are
themselves recording-LOO ridge predictions. The held recording supplies neither its
target nor event labels to the head.

The grid compares 0.5× and 1× offset strength with ridge 1 and 4. Strength zero is the
exact promoted control.

## Outer-held results

| Variant | Row AP | Proposals | TP | FP | FN | Precision | Recall | ±4 F1 | Strict F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Promoted control** | **44.50%** | **52** | 29 | **23** | 21 | **55.77%** | 58.00% | **56.86%** | **45.10%** |
| 0.5× offset, ridge 1 | 42.48% | 59 | **30** | 29 | **20** | 50.85% | **60.00%** | 55.05% | 44.04% |
| 1× offset, ridge 1 | 38.93% | 64 | 30 | 34 | 20 | 46.88% | 60.00% | 52.63% | 42.11% |
| 0.5× offset, ridge 4 | 43.25% | 59 | 29 | 30 | 21 | 49.15% | 58.00% | 53.21% | 42.20% |
| 1× offset, ridge 4 | 41.38% | 65 | 30 | 35 | 20 | 46.15% | 60.00% | 52.17% | 41.74% |

The learned offsets are often negative, lowering the effective threshold to chase a
per-video recall optimum. With only ten training recordings in each outer fold, the
quality summaries do not reliably distinguish when that extra proposal load is safe.
The one additional TP is therefore outweighed by six additional FP.

Nested selection returns the promoted control in 11/11 folds and reproduces 52
proposals, 29 TP, 23 FP, 21 FN, and 56.86% F1 exactly. Full-development selection also
returns strength zero. The exported experiment classifier, threshold, and decoder are
byte-value equivalent to the promoted model; the artifact is an experiment record, not
a replacement.

## Artifacts and reproduction

```bash
PYTHONPATH=. /home/developer/volleycut/.venv/bin/python \
  scripts/train-side-switch-recording-reliability.py
PYTHONPATH=. /home/developer/volleycut/.venv/bin/python \
  -m unittest discover -s analysis/tests -p 'test_side_switch*.py'
```

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-recording-reliability-v1/model.json` | `076bc7081a89cca0ab30fa054061e578aa7f44899b6026475107b423615ed5ff` |
| `reports/side-switch/side-switch-recording-reliability-v1-evaluation.json` | `9b6ca0eb34601236710e95a1b03bfb10b7bea563f93ef8a3a191689466705469` |

## Sources

- [Rare-event improvement plan](./side-switch-rare-event-improvement-plan-2026-08-23.md)
- [Hard-negative winner promotion](./side-switch-hard-negative-winner-promotion-2026-08-23.md)
- [Pairwise-ranking rejection](./side-switch-pairwise-ranking-2026-08-23.md)
