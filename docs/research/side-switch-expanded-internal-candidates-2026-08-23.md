# Side-switch expanded internal-candidate experiment — 2026-08-23

## Decision

Expanding the dead-state candidate union reaches 100% marker coverage but makes the
ranker materially worse. The 704-candidate control's nested ±4-second result is 27
TP/29 FP/23 FN and 50.94% F1. The 852-candidate variant produces **25 TP/42 FP/25 FN
and 42.74% F1**. Strict F1 falls from 39.62% to 32.48%, and nested row average precision
falls from 41.82% to 30.89%.

Reject the expanded union and retain the 704-candidate full union for ranking research.
The new candidates prove that all four formerly uncovered markers are reachable from
the existing 4 Hz production trace, but only one survives nested decoding. The next
step must improve the internal before/after representation or use a separate internal
head; lowering the dead-state threshold alone adds too many confusing windows.

The current user-designated winner, production inference, and review UI are unchanged.

## Candidate change

The prior selected generator keeps all 624 adjacent production boundaries and internal
dead-state peaks with score at least 0.98, separated by 14 seconds inside each decoded
range. Its 80 internal peaks cover 46/50 markers under the ±4-second one-to-one
contract.

Auditing the four misses showed usable but lower dead-state peaks near every event:

| Recording | Marker | Useful local dead-state peak |
| --- | ---: | ---: |
| `180646590` | 452.620 | 449.750, score 0.919 |
| `203801418` | 280.119 | 284.000, score 0.961 |
| `203801418` | 460.978 | 460.750, score 0.887 |
| `212717581` | 542.833 | 541.000, score 0.986 |

The expanded configuration is therefore the already enumerated v1-grid setting:

```text
signal = deadState
threshold = 0.80
minimum peak separation = 10 seconds
range-edge exclusion = 4 seconds
proposal half-width = 1 second
```

It is fixed after inspecting the opened-development grid; its 100% coverage is an
optimistic development diagnostic, not a source-held-out generalization claim.
The artifact's `selectionScope` calls it preregistered only in the narrow sense that it
was frozen before feature extraction and ranker training—not before marker inspection.

| Candidate universe | Boundaries | Internal peaks | Total | Covered markers | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected full-union control | 624 | 80 | 704 | 46/50 | 92% |
| Expanded threshold/separation | 624 | 228 | 852 | 50/50 | **100%** |

Candidate load rises from 64.0 to 77.45 windows per video. The expanded set shares 703
exact event IDs with the prior union; tighter score-ranked clustering replaces one old
internal peak and adds 149 new ones.

## Feature extraction

The feature contract is unchanged: 42 V5 appearance and production-state values, with
the same 32-input primary view and 34-input kind/score view used by the ranker. The
extractor now supports hash-bound feature reuse:

- 703 shared candidate rows are reused exactly;
- 149 new candidates require 298 comparison sequences and 2,086 decoded frames;
- all 352 legacy rows still match every one of their 42 values exactly;
- maximum legacy feature difference remains `0.0` at tolerance `1e-8`.

The resulting profile is named `FULL-UNION-EXPANDED-V5-STATE42`. Reuse is an extraction
optimization only; every recording still passes source identity, production probability,
and decoded-range parity checks.

## Nested ranker protocol

The ranker uses the original full-union nested protocol. Each outer fold holds out one
recording from:

- 32-input versus 34-input feature selection;
- L2 selection from `0.1`, `1`, and `10`;
- threshold selection;
- adjacent-index/time local suppression; and
- zero versus 0.5 post-six soft count penalty.

There is no cadence, re-anchoring, hard count cap, production gate, or suppression
input. The candidate-generator setting itself was selected on the opened scope as
noted above; only ranker and decoder selection are outer-recording-held-out.

## Results

| Result | Proposals | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current user winner, ±4 s | 62 | 25 | 37 | 25 | 40.32% | 50.00% | 44.64% |
| 704-candidate nested ranker, ±4 s | 56 | 27 | 29 | 23 | 48.21% | 54.00% | **50.94%** |
| Expanded 852-candidate ranker, ±4 s | 67 | 25 | 42 | 25 | 37.31% | 50.00% | **42.74%** |
| Expanded + continuity veto, ±4 s | 63 | 23 | 40 | 27 | 36.51% | 46.00% | 40.71% |
| Frozen current winner translated to expanded union | 81 | 26 | 55 | 24 | 32.10% | 52.00% | 39.69% |
| 704-candidate nested ranker, strict | 56 | 21 | 35 | 29 | 37.50% | 42.00% | 39.62% |
| Expanded 852-candidate ranker, strict | 67 | 19 | 48 | 31 | 28.36% | 38.00% | 32.48% |

Relative to the 704-candidate nested ranker, expansion adds 11 emitted proposals and
13 FP while losing two TP. The decoder emits 13 internal candidates: one TP and 12 FP.
That one TP is the newly covered `180646590` event at 452.62 seconds. Neither new
`203801418` event nor the new `212717581` event survives the held-out ranker.

The expanded label assignment contains eight internal positives among 50 candidate
labels, versus four among 46 for the prior union. The extra supervision is still too
sparse relative to 220 internal negatives and is not separable with the current flank
features. The continuity veto is also rejected because it removes two TP for only two
FP.

Eight of 11 outer folds select L2 `0.1`; six select the 32-input view and five the
34-input view. Eight choose the 0.5 soft count penalty and three choose no count
penalty. The all-opened-development fit reaches 50.36% F1 at 89 proposals; this is a
same-data selection result and is not the primary claim.

## On-device implications

The candidate scan still reuses existing 4 Hz rally/dead-state traces and is linear in
trace length. The material cost is downstream appearance extraction: 148 net additional
candidates per set, or about 21 additional internal windows per video instead of 7.
Although the linear head remains tiny, the added frame sampling and lower ranking
quality make this variant unattractive for on-device inference.

No browser or Android port was implemented.

## Artifacts and reproduction

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-side-switch-candidate-union.py \
  --fixed-config-id deadState-threshold-0.8-separation-10 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-candidate-union-expanded-v1-evaluation.json
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-full-union-features.py \
  --candidates /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-candidate-union-expanded-v1-evaluation.json \
  --reuse-features /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-full-union-v5-state-features-v1.json \
  --expected-candidate-sha256 5936163e459544208e252069da651055161f073c8ebbf2b4e3552f6b1dcf09f8 \
  --expected-reuse-sha256 9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551 \
  --expected-candidate-count 852 \
  --profile-name FULL-UNION-EXPANDED-V5-STATE42 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-full-union-expanded-v5-state-features-v1.json
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-full-union-ranker.py \
  --features /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-full-union-expanded-v5-state-features-v1.json \
  --expected-features-sha256 fe6563d083e49911c1d1c6f79a8c6dae4326774bf40a0041e157fcb2610fc919 \
  --expected-positive-candidates 50 \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-full-union-expanded-ranker-v1/model.json \
  --evaluation /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-full-union-expanded-ranker-v1-evaluation.json
```

The exact expected source hashes used in the completed commands are also recorded in
each artifact. The run completed all 85 side-switch unit tests.

| Artifact | SHA-256 |
| --- | --- |
| `reports/side-switch/side-switch-candidate-union-expanded-v1-evaluation.json` | `5936163e459544208e252069da651055161f073c8ebbf2b4e3552f6b1dcf09f8` |
| `reports/side-switch/side-switch-full-union-expanded-v5-state-features-v1.json` | `fe6563d083e49911c1d1c6f79a8c6dae4326774bf40a0041e157fcb2610fc919` |
| `models/side-switch-full-union-expanded-ranker-v1/model.json` | `1e1a33bfc84ec2c50ff35f31b844fc5b62aa32cd2652ae5e5c66a0bfad2092b1` |
| `reports/side-switch/side-switch-full-union-expanded-ranker-v1-evaluation.json` | `85e2b2e3998a18085a54874741fff793173d2a17862fbaf48325fa8e89dbab3f` |

## Sources

- [Rare-event improvement plan and literature sources](./side-switch-rare-event-improvement-plan-2026-08-23.md)
- [Full-trace candidate-union decision](./side-switch-candidate-union-2026-08-23.md)
- [Full-union feature/ranker experiment](./side-switch-full-union-ranker-2026-08-23.md)
- [Internal-peak penalty experiment](./side-switch-internal-peak-penalty-2026-08-23.md)
