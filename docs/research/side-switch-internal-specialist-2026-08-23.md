# Side-switch internal-candidate specialist — 2026-08-23

## Decision

Reject the learned internal-candidate specialist. None of the outer-held specialist
branches matches an internal true event; the best internal row AP is only 6.87% over
eight positives and 220 negatives. Adding a selected internal output to the retained
boundary branch adds one FP and no TP.

Retain **boundary-only** as a promising opened-development cleanup candidate. Removing
the promoted model's five internal outputs drops one TP and four FP, moving the fixed
outer result from 29 TP/23 FP/21 FN and 56.86% F1 to 28 TP/19 FP/22 FN and **57.73%
F1**. Strict F1 improves from 45.10% to 47.42%.

This 0.87-point primary gain is small, was discovered on opened development, and does
not recover the four formerly out-of-union events. Do not replace the explicitly
promoted research winner or port the branch automatically.

## Method

The experiment keeps the promoted 704-candidate base head and its outer-selected
threshold exactly. Its selected adjacent-rally boundaries become a frozen branch. The
five base internal proposals can either remain unchanged, be removed entirely, or be
replaced by a separate head over the expanded 228 internal candidates.

The expanded generator has 100% candidate coverage and eight internal positive labels,
concentrated in only four recordings. The internal representation adds nine range/peak
features to the extracted V5/state evidence:

- range duration, time since start, time to end, and normalized position;
- time since serve anchor and distance to the nearest boundary;
- peak count and inverse dead-state-score rank within the source range; and
- before/after palette-instability difference.

Three internal views compare 12 transition/appearance values, 12 geometry/state
values, and their 24-value union. Each uses square-root balancing, recording-balanced
top-2/2× hard-negative mining, L2 0.1 or 1, independent threshold selection, ten-second
internal NMS, and four-second duplicate suppression around retained boundaries.

Branch, view, L2, and internal threshold remain inside every outer recording holdout.
The master evaluation union has 853 IDs: the 852 expanded candidates plus the one old
internal peak replaced by tighter expanded clustering.

## Fixed-variant outer-held results

| Branch | Internal AP | Proposals | TP | FP | FN | Precision | Recall | ±4 F1 | Strict F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Promoted control | — | 52 | **29** | 23 | **21** | 55.77% | **58.00%** | 56.86% | 45.10% |
| **Boundary-only** | — | **47** | 28 | **19** | 22 | **59.57%** | 56.00% | **57.73%** | **47.42%** |
| Transition 12, L2 0.1 | 6.69% | 48 | 28 | 20 | 22 | 58.33% | 56.00% | 57.14% | 46.94% |
| Transition 12, L2 1 | 4.51% | 48 | 28 | 20 | 22 | 58.33% | 56.00% | 57.14% | 46.94% |
| Geometry 12, L2 0.1 | 3.05% | 48 | 28 | 20 | 22 | 58.33% | 56.00% | 57.14% | 46.94% |
| Geometry 12, L2 1 | 2.39% | 47 | 28 | 19 | 22 | 59.57% | 56.00% | 57.73% | 47.42% |
| Combined 24, L2 0.1 | **6.87%** | 48 | 28 | 20 | 22 | 58.33% | 56.00% | 57.14% | 46.94% |
| Combined 24, L2 1 | 4.64% | 48 | 28 | 20 | 22 | 58.33% | 56.00% | 57.14% | 46.94% |

The boundary-only changes are localized:

| Video | Promoted proposals / TP / FP | Boundary-only proposals / TP / FP |
| --- | ---: | ---: |
| `171720964` | 6 / 4 / 2 | 5 / 4 / 1 |
| `183701800` | 6 / 3 / 3 | 4 / 3 / 1 |
| `190429172` | 6 / 2 / 4 | 4 / 1 / 3 |
| All other videos | unchanged | unchanged |

The removed internal TP is the 759.75-second proposal matching the `190429172`
763.772-second marker. The other four promoted internal proposals are false. This is a
precision improvement, not better recovery.

Nested branch selection chooses a learned specialist in 10/11 folds, but its combined
outer result is 48 proposals, 28 TP, 20 FP, 22 FN, and 57.14% F1—one FP worse than the
fixed boundary-only policy. That gap is direct evidence that internal threshold/view
selection overfits the tiny four-recording positive support.

Full-development selection returns the promoted control. The exported experiment model
therefore contains the exact base classifier and no internal specialist. Boundary-only
remains a candidate for validation on new exhaustive sets; the next recovery attempt
requires new continuous motion/side-state evidence rather than more combinations of
the existing six-frame flank summaries.

## Artifacts and reproduction

```bash
PYTHONPATH=. /home/developer/volleycut/.venv/bin/python \
  scripts/train-side-switch-internal-specialist.py
PYTHONPATH=. /home/developer/volleycut/.venv/bin/python \
  -m unittest discover -s analysis/tests -p 'test_side_switch*.py'
```

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-internal-specialist-v1/model.json` | `cbbbc0966c2828e9de953f8d49e649abf997e2f1a47793f86f97f4347a56935a` |
| `reports/side-switch/side-switch-internal-specialist-v1-evaluation.json` | `004bd4e5754c1c3f000d5305717c5725b3ea5aea373e8f738e02d5c59a8624cb` |

## Sources

- [Expanded internal-candidate experiment](./side-switch-expanded-internal-candidates-2026-08-23.md)
- [Hard-negative winner promotion](./side-switch-hard-negative-winner-promotion-2026-08-23.md)
- [Rare-event improvement plan](./side-switch-rare-event-improvement-plan-2026-08-23.md)
