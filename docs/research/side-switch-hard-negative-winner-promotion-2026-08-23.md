# Side-switch hard-negative winner promotion — 2026-08-23

## Decision

Promote the fixed `union34-top2-x2` hard-negative model to the current
**research-only** side-switch winner. It improves the exhaustive 50-event,
11-video development audit from 25 to 29 correct matches while emitting ten fewer
proposals. Under ±4-second proposal-boundary matching, pooled F1 rises from 44.64% to
56.86%.

This changes the research-winner pointer only. It does not install the model in the
web editor, Android app, or production inference. The variant was selected after the
development scope was opened, so an unseen reviewed set is still required for an
independent promotion decision.

## Overall comparison

| Research model | Proposals | Correct matches (TP) | Wrong proposals (FP) | Missed (FN) | Pooled precision | Pooled recall | Pooled F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Former `local-peak-soft-count` | 62 | 25 | 37 | 25 | 40.32% | 50.00% | 44.64% |
| **Promoted `union34-top2-x2`** | **52** | **29** | **23** | **21** | **55.77%** | **58.00%** | **56.86%** |
| Change | -10 | +4 | -14 | -4 | +15.45 pp | +8.00 pp | +12.22 pp |

Strict, zero-margin matching for the promoted model is 23 TP, 29 FP, 27 FN,
44.23% precision, 46.00% recall, and 45.10% F1.

## Average per video

Precision and recall are macro averages of the 11 per-video values. Proposal, correct,
wrong, and missed counts are totals divided by 11.

| Research model | Avg precision | Avg recall | Avg proposals | Avg correct matches | Avg wrong proposals | Avg missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Former `local-peak-soft-count` | 41.19% | 50.91% | 5.64 | 2.27 | 3.36 | 2.27 |
| **Promoted `union34-top2-x2`** | **58.66%** | **57.73%** | **4.73** | **2.64** | **2.09** | **1.91** |

## Promoted result by video

Each row is the fixed variant's outer-held prediction. The short video ID is the
timestamp suffix of the raw-phone recording ID.

| Video | Human switches | Proposals | Correct matches (TP) | Wrong proposals (FP) | Missed (FN) | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `160023210` | 4 | 3 | 2 | 1 | 2 | 66.67% | 50.00% |
| `161923155` | 5 | 7 | 5 | 2 | 0 | 71.43% | 100.00% |
| `164327879` | 4 | 7 | 4 | 3 | 0 | 57.14% | 100.00% |
| `171720964` | 5 | 6 | 4 | 2 | 1 | 66.67% | 80.00% |
| `180646590` | 5 | 4 | 4 | 0 | 1 | 100.00% | 80.00% |
| `183701800` | 4 | 6 | 3 | 3 | 1 | 50.00% | 75.00% |
| `190429172` | 5 | 6 | 2 | 4 | 3 | 33.33% | 40.00% |
| `193307688` | 4 | 3 | 0 | 3 | 4 | 0.00% | 0.00% |
| `203801418` | 5 | 1 | 1 | 0 | 4 | 100.00% | 20.00% |
| `210449857` | 4 | 3 | 2 | 1 | 2 | 66.67% | 50.00% |
| `212717581` | 5 | 6 | 2 | 4 | 3 | 33.33% | 40.00% |
| **Total / macro average** | **50** | **52** | **29** | **23** | **21** | **58.66%** | **57.73%** |

The largest remaining failure is recording `193307688`, where all four switches are
missed. Recordings `190429172`, `203801418`, and `212717581` also retain substantial
recall debt. Four of the 50 human events fall outside the retained 704-candidate union;
ranking alone cannot recover those events.

## Promotion and reproducibility

The live pointer is
[`data/side-switch-current-research-winner-v1.json`](../../data/side-switch-current-research-winner-v1.json),
SHA-256 `163c7844267bc48410e89f86bd5cbf0a58b3145c1ec691932e4e3374dae7286c`.
It records the fixed outer-held result, per-video arithmetic, final development
threshold `0.39884973953581804`, and the 34-input/top-2/2× training objective.

The former winner is preserved byte-for-byte as
[`data/side-switch-research-winner-2026-08-21-local-peak-soft-count.json`](../../data/side-switch-research-winner-2026-08-21-local-peak-soft-count.json),
SHA-256 `ea1423a3dd7f96812b401dbe7a8cdb2fe8435d4e7f2a290356eb2c0cfba4fb8f`.
Historical continuity experiments bind that archive instead of the mutable live
pointer.

The source model is
`models/side-switch-hard-negative-mining-v1/model.json`, SHA-256
`c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3`.
The source evaluation is
`reports/side-switch/side-switch-hard-negative-mining-v1-evaluation.json`, SHA-256
`e67088b36d177d68c24587efcb186eab4201b5294532be1fdf978ef13aaacc4b`.
Both live on the canonical labeling NAS root.

## Sources

- [Hard-negative mining experiment](./side-switch-hard-negative-mining-2026-08-23.md)
- [Full-video marker audit](./side-switch-full-video-marker-audit-2026-08-21.md)
- [Rare-event improvement plan and literature](./side-switch-rare-event-improvement-plan-2026-08-23.md)
