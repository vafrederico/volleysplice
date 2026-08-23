# Side-switch focal and effective-number losses — 2026-08-23

## Decision

Reject focal loss and effective-number class weighting for the retained side-switch
head. No fixed variant beats the promoted square-root-balanced BCE objective. Focal
gamma 1 slightly raises row AP from 44.50% to 44.66%, but loses one TP and adds three
FP, lowering ±4-second F1 from 56.86% to 53.85%. Effective-number beta 0.99 is the best
alternative event result at 54.90% F1.

Nested objective selection is split across five settings and regresses to 27 TP/24
FP/23 FN and 53.47% F1. Full-development selection returns the promoted control. The
current research-winner pointer and all runtimes remain unchanged.

## Method

The experiment holds union34 features, L2 `0.1`, recording-balanced top-2/2× hard
negative mining, local-peak suppression, soft count cost, markers, and outer recording
holdouts fixed. Only the final linear-head loss changes.

Focal variants use exact binary focal loss with gamma 1 or 2, retaining square-root
class weights. A deterministic BFGS optimizer starts from the matched BCE solution.
Effective-number variants replace square-root class weighting with
`(1-beta)/(1-beta^classCount)` at beta 0.9, 0.99, or 0.999. Hard-negative multipliers
remain identical for every objective.

Each outer fold selects objective and threshold using recording-cross-fitted scores on
the other ten videos. The zero-change control reproduces the promoted strict and ±4
event counts exactly.

## Fixed-variant outer-held results

| Objective | Row AP | Proposals | TP | FP | FN | Precision | Recall | ±4 F1 | Strict F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Square-root BCE control** | 44.50% | 52 | **29** | **23** | **21** | **55.77%** | **58.00%** | **56.86%** | **45.10%** |
| Focal gamma 1 | **44.66%** | 54 | 28 | 26 | 22 | 51.85% | 56.00% | 53.85% | 44.23% |
| Focal gamma 2 | 44.56% | 54 | 28 | 26 | 22 | 51.85% | 56.00% | 53.85% | 44.23% |
| Effective beta 0.9 | 39.34% | 52 | 27 | 25 | 23 | 51.92% | 54.00% | 52.94% | 39.22% |
| Effective beta 0.99 | 43.00% | 52 | 28 | 24 | 22 | 53.85% | 56.00% | 54.90% | 43.14% |
| Effective beta 0.999 | 43.22% | 51 | 25 | 26 | 25 | 49.02% | 50.00% | 49.50% | 37.62% |

The focal AP gain does not survive decoding at the required precision/recall operating
point. Effective-number weighting is sensitive to beta and none improves AP or event
F1. This reinforces the prior result that recording-balanced hard-example selection is
more useful here than broad rare-class reweighting.

Outer selection chooses the control in four folds, focal gamma 1 in three, effective
beta 0.99 in two, and focal gamma 2/effective beta 0.9 once each. That mixed selection
produces 51 proposals, 27 TP, 24 FP, 23 FN, 52.94% precision, 54.00% recall, and 53.47%
F1. Strict F1 is 41.58%.

Full-development selection returns the exact promoted classifier, threshold
`0.39884973953581804`, and decoder. The exported model is an experiment record rather
than a new candidate.

## Artifacts and reproduction

```bash
PYTHONPATH=. /home/developer/volleycut/.venv/bin/python \
  scripts/train-side-switch-rare-event-losses.py
PYTHONPATH=. /home/developer/volleycut/.venv/bin/python \
  -m unittest discover -s analysis/tests -p 'test_side_switch*.py'
```

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-rare-event-losses-v1/model.json` | `bca760af4f8fda98f7a8665439a62e61141660d3a7ec477e8a592d216fc6b645` |
| `reports/side-switch/side-switch-rare-event-losses-v1-evaluation.json` | `cfd89de98636899af9b30bfa62de460392b0d737b1518b4d003fe435a65d8462` |

## Sources

- [Rare-event improvement plan and focal/effective-number sources](./side-switch-rare-event-improvement-plan-2026-08-23.md)
- [Hard-negative winner promotion](./side-switch-hard-negative-winner-promotion-2026-08-23.md)
- [Recording-reliability rejection](./side-switch-recording-reliability-2026-08-23.md)
