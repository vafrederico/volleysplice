# Side-switch uncertain score/cadence prior — 2026-08-23

## Decision

Reject the latent score/cadence prior over detected rally ordinals. Exact seven-rally
cadence is badly misaligned: only 6 of the 46 positive union candidates have an
opportunity ordinal divisible by seven. Its lightest weight emits 118 proposals and
falls to 23.81% ±4-second F1.

Allowing redo and missing-point uncertainty avoids that collapse but still does not
beat the promoted no-cadence winner. The best soft prior preserves 29 TP while adding
five FP, reducing F1 from 56.86% to 54.21%. Nested selection chooses no prior in 8/11
folds and reaches 54.90% F1. Keep the current winner unchanged.

## Method

The decoder-only experiment keeps the promoted classifier, hard-negative mining,
local-peak suppression, and soft count cost fixed. It starts every one-set recording at
latent total zero. For every detected rally observation, a forward distribution allows:

- `+0` for a redo/non-point;
- `+1` for a normal point; and
- `+2` for a missed point between observations.

A Gaussian kernel measures posterior proximity to positive multiples of seven. The
per-recording centered hazard logit is added softly to the classifier logit. Predictions
never alter the latent distribution, so there is no re-anchoring or cascade from a
selected switch.

The grid compares:

- exact `0/1/0` transitions with 0.75-point kernel width;
- symmetric uncertain `0.05/0.90/0.05` transitions;
- redo-heavy `0.10/0.85/0.05` transitions; and
- prior weights 0.25, 0.5, or 1 where declared.

Variant and threshold selection stay inside each outer recording holdout. Strength zero
reproduces the promoted event counts exactly.

## Fixed-variant outer-held results

| Prior | Row AP | Proposals | TP | FP | FN | Precision | Recall | ±4 F1 | Strict F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **No-prior control** | **44.50%** | 52 | **29** | **23** | **21** | **55.77%** | **58.00%** | **56.86%** | **45.10%** |
| Exact, weight 0.25 | 20.47% | 118 | 20 | 98 | 30 | 16.95% | 40.00% | 23.81% | 17.86% |
| Exact, weight 0.5 | 17.69% | 89 | 11 | 78 | 39 | 12.36% | 22.00% | 15.83% | 11.51% |
| Uncertain, weight 0.25 | 43.75% | **48** | 24 | 24 | 26 | 50.00% | 48.00% | 48.98% | 42.86% |
| Uncertain, weight 0.5 | 42.23% | 47 | 23 | 24 | 27 | 48.94% | 46.00% | 47.42% | 41.24% |
| Uncertain, weight 1 | 36.05% | 47 | 23 | 24 | 27 | 48.94% | 46.00% | 47.42% | 41.24% |
| Redo-heavy, weight 0.25 | 43.02% | 57 | 29 | 28 | 21 | 50.88% | 58.00% | 54.21% | 41.12% |
| Redo-heavy, weight 0.5 | 41.08% | 55 | 26 | 29 | 24 | 47.27% | 52.00% | 49.52% | 38.10% |

The positive candidate modulo-seven distribution is broad: residues 0–6 contain
`6, 5, 6, 12, 4, 5, 8` positives respectively. Detected production rallies therefore
cannot be treated as a point counter. Redos, missed detections, incorrect rally ranges,
and non-point activity have already accumulated before the first switch.

The uncertain prior spreads mass enough to become a weak ranking signal, but it either
removes true proposals or adds false ones. Nested selection uses the no-prior control in
eight folds, uncertain weight 0.25 in two, and redo-heavy weight 0.25 in one. The
combined result is 52 proposals, 28 TP, 24 FP, 22 FN, and 54.90% F1. Strict F1 remains
45.10% but does not offset the primary recall loss.

Full-development selection returns no prior, the exact promoted threshold
`0.39884973953581804`, and the unchanged classifier/decoder. A future score prior needs
actual point-result observations or a learned point/outcome state, not rally ordinal
alone.

## Artifacts and reproduction

```bash
PYTHONPATH=. .venv/bin/python \
  scripts/train-side-switch-soft-score-prior.py
PYTHONPATH=. .venv/bin/python \
  -m unittest discover -s analysis/tests -p 'test_side_switch*.py'
```

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-soft-score-prior-v1/model.json` | `293edc871580d12f0063587a47007fece67a3571fd4a808b0de4fb202b1cb3c7` |
| `reports/side-switch/side-switch-soft-score-prior-v1-evaluation.json` | `2fd5f6719352b81d2ed583d7b6a453901597d227d56104ff10a507241dbe2fcc` |

## Sources

- [Rare-event improvement plan](./side-switch-rare-event-improvement-plan-2026-08-23.md)
- [No-cadence experiment](./side-switch-v5-no-cadence-2026-08-20.md)
- [Hard-negative winner promotion](./side-switch-hard-negative-winner-promotion-2026-08-23.md)
