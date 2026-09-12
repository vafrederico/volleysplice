# Side-switch within-recording pairwise ranking — 2026-08-23

## Decision

Reject the within-recording pairwise logistic objective for the current side-switch
ranker. The weakest pairwise weight raises row average precision only from 44.50% to
44.73%, but emits seven additional false proposals without matching another human
switch. Its ±4-second event F1 falls from 56.86% to 53.21%. Larger pairwise weights and
the tested 0.5-logit margins regress further.

Nested variant selection chooses a pairwise objective in 5/11 outer folds, but the
combined outer-held result keeps the same 29 TP while adding four FP: 56.86%→54.72%
F1. The promoted `union34-top2-x2` pointwise hard-negative model remains the current
research winner. No production, browser, or Android model changes.

## Method

The experiment changes only the loss used for the final 34-input linear head. It holds
fixed:

- the retained 704-candidate full union and 46 candidate labels;
- square-root class balancing and L2 `0.1`;
- recording-balanced top-2/2× hard-negative mining;
- adjacent-candidate local peak suppression;
- six free outputs followed by a 0.5-logit soft count cost; and
- the exhaustive 50 markers with strict and ±4-second one-to-one matching.

For each fit scope, the new loss forms every positive/negative candidate pair within
the same recording. Each eligible recording receives equal total pair weight regardless
of its candidate count. The convex objective is:

```text
pointwise weighted logistic loss
  + lambda * mean_recording mean_pair softplus(margin - positiveLogit + negativeLogit)
  + L2
```

This is a pairwise logistic AUC surrogate, not a reproduction of the PESG deep-AUC
optimizer cited in the research plan. The grid compares pairwise strengths 0.25, 0.5,
1, and 2 at zero margin, plus strengths 0.5 and 1 at a 0.5-logit margin. Strength zero
is the promoted pointwise control. Pairwise variant and threshold selection remain
inside each outer recording holdout.

## Fixed-variant outer-held results

| Variant | Row AP | Proposals | TP | FP | FN | Precision | Recall | ±4 F1 | Strict F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pointwise hard-negative control** | 44.50% | 52 | 29 | 23 | 21 | 55.77% | 58.00% | **56.86%** | **45.10%** |
| Pairwise λ=0.25 | **44.73%** | 59 | 29 | 30 | 21 | 49.15% | 58.00% | 53.21% | 42.20% |
| Pairwise λ=0.5 | 44.30% | 56 | 27 | 29 | 23 | 48.21% | 54.00% | 50.94% | 41.51% |
| Pairwise λ=1 | 44.19% | 57 | 26 | 31 | 24 | 45.61% | 52.00% | 48.60% | 37.38% |
| Pairwise λ=2 | 43.78% | 60 | 27 | 33 | 23 | 45.00% | 54.00% | 49.09% | 38.18% |
| Pairwise λ=0.5, margin=0.5 | 44.49% | 57 | 28 | 29 | 22 | 49.12% | 56.00% | 52.34% | 41.12% |
| Pairwise λ=1, margin=0.5 | 44.07% | 56 | 25 | 31 | 25 | 44.64% | 50.00% | 47.17% | 37.74% |

The small AP improvement at λ=0.25 does not reach the sparse event operating point.
It preserves every matched event but admits additional candidates in seven videos.
Optimizing all within-recording orderings is therefore misaligned with the post-decoder
precision cost on this sample.

## Weakest pairwise variant by video

| Video | Control proposals / TP / FP / FN | Pairwise λ=0.25 proposals / TP / FP / FN |
| --- | ---: | ---: |
| `160023210` | 3 / 2 / 1 / 2 | 5 / 2 / 3 / 2 |
| `161923155` | 7 / 5 / 2 / 0 | 7 / 5 / 2 / 0 |
| `164327879` | 7 / 4 / 3 / 0 | 7 / 4 / 3 / 0 |
| `171720964` | 6 / 4 / 2 / 1 | 7 / 4 / 3 / 1 |
| `180646590` | 4 / 4 / 0 / 1 | 6 / 4 / 2 / 1 |
| `183701800` | 6 / 3 / 3 / 1 | 6 / 3 / 3 / 1 |
| `190429172` | 6 / 2 / 4 / 3 | 7 / 2 / 5 / 3 |
| `193307688` | 3 / 0 / 3 / 4 | 3 / 0 / 3 / 4 |
| `203801418` | 1 / 1 / 0 / 4 | 2 / 1 / 1 / 4 |
| `210449857` | 3 / 2 / 1 / 2 | 3 / 2 / 1 / 2 |
| `212717581` | 6 / 2 / 4 / 3 | 6 / 2 / 4 / 3 |
| **Total** | **52 / 29 / 23 / 21** | **59 / 29 / 30 / 21** |

The pairwise loss does not repair the zero-TP `193307688` video or any of the four
markers outside the candidate union. It primarily lowers proposal selectivity.

## Selection stability and exported artifact

The inner selector chooses the pointwise control in six outer folds, λ=0.25 in three,
λ=0.5 in one, and λ=1 with margin 0.5 in one. Its outer-held aggregate is 56 proposals,
29 TP, 27 FP, 21 FN, 51.79% precision, 58.00% recall, and 54.72% F1. Strict F1 is
43.40%.

Full-development selection chooses the pointwise control at the exact promoted
threshold `0.39884973953581804`. The exported experiment classifier, threshold, and
decoder are exactly equal to the promoted hard-negative artifact. This duplicate is
retained only as an immutable experiment record; it is not a new winner.

## Artifacts and reproduction

```bash
PYTHONPATH=. .venv/bin/python \
  scripts/train-side-switch-pairwise-ranking.py
PYTHONPATH=. .venv/bin/python \
  -m unittest discover -s analysis/tests -p 'test_side_switch*.py'
```

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-pairwise-ranking-v1/model.json` | `3fa593b9abed880078ec33e888eb62a689aef1e45fe48dcab7112a1786ca3e9b` |
| `reports/side-switch/side-switch-pairwise-ranking-v1-evaluation.json` | `e95c529ba676824ef06a4b1578b0ccca8178a9b5827e557ef0b67dcb215f4dee` |

The pointwise control reproduces the promoted model's strict and ±4 event counts
exactly. Pair construction, loss validation, and zero-strength exact parity are covered
by the side-switch unit suite.

## Sources

- [Rare-event improvement plan and literature](./side-switch-rare-event-improvement-plan-2026-08-23.md)
- [Hard-negative winner promotion](./side-switch-hard-negative-winner-promotion-2026-08-23.md)
- [Hard-negative mining experiment](./side-switch-hard-negative-mining-2026-08-23.md)
