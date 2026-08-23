# Side-switch full-union feature and rare-event ranker experiment — 2026-08-23

## Decision

The full 704-candidate union is now scoreable with an exact V5-state-compatible
feature pipeline. All 352 legacy rows reproduce all 42 values with maximum absolute
difference `0.0`. A nested leave-one-recording-out (LOO) class-balanced ranker improves
the current research winner's end-to-end ±4-second pooled F1 from **44.64% to 50.94%**,
while reducing proposals from 62 to 56 and improving both precision and recall.

This is a **research contender, not a production promotion and not an automatic
replacement for the user-designated current winner**. The 11 recordings and their 50
markers are opened development scope. The feature view is unstable across outer folds,
one recording receives no true proposal, and the new internal-peak candidates remain
poorly discriminated. A newly refit continuity veto is rejected on this distribution.

## Question

The preceding candidate-union experiment raised the maximum ±4-second event coverage
from 33/50 to 46/50, but only 352 of its 704 candidates had frozen V5 features. This
loop asks:

1. Can every union candidate receive the same V5-state signature without changing old
   rows?
2. Does a rare-event classifier trained with recording isolation use that added
   candidate coverage to improve final event precision and recall?
3. Does the same-side continuity veto still remove false positives after the candidate
   distribution changes?

The hypothesis follows the staged rare-event plan: improve upstream coverage first,
then apply a class-balanced discriminative ranker and local/soft-count decoding. It
does not use cadence or selection re-anchoring.

## Scope and leakage policy

- Scope: 11 raw-phone one-set recordings, all starting at score zero.
- Truth: 50 exhaustive user-reviewed point markers.
- Candidate universe: 624 adjacent production-range boundaries plus 80 internal
  `deadState >= 0.98` peaks.
- Primary event matching: one-to-one monotonic interval matching with a ±4-second
  boundary allowance.
- Sensitivity: strict point-inside-interval matching with no allowance.
- Candidate labels: within each recording, maximum-cardinality ±4-second matching,
  followed by minimum total anchor distance, assigns one candidate to each covered
  marker. This produces 46 positives and 658 negatives; the four uncovered markers
  remain false negatives in event evaluation.
- The 11 recordings are **opened development only**, not an untouched test set.
- Every outer held recording is excluded from feature-group, L2, threshold, decoder,
  and continuity-cutoff selection.

## Implemented feature contract

### Adjacent boundaries

Each production range retains the frozen V4/V5 whole-rally contract: seven 256×144
ROI frames sampled from the interior of the decoded range. Neighboring boundaries
reuse the same range summary.

### Internal peaks

For an internal candidate centered at `t`, the extractor compares two fixed stable
flanks:

```text
before: [t - 4, t - 1]
candidate: [t - 1, t + 1]
after:  [t + 1, t + 4]
```

Seven 256×144 frames summarize each three-second flank. The candidate generator's
four-second range-edge exclusion guarantees both flanks remain in the same decoded
range. The window rule is fixed and label-independent.

The complete artifact stores:

- 22 `PLAYER-ORIENTATION22` appearance/player features;
- 20 `PRODUCTION-STATE20` features from replaying the two shipped production bundles;
- the exact comparison windows and production evidence used for each candidate.

The frozen primary head consumes the first 22 plus the ten state-gate inputs. Internal
peaks do not have two distinct decoded-rally serve anchors, so their production-context
head input is declared ineligible rather than pretending that the same containing
range is two separate rallies.

### Parity and extraction cost

The extractor decoded 5,565 low-resolution frames across 11 videos. Both production
bundles were replayed over the existing cached on-device F104 matrix. The replay stayed
within the pre-existing `5e-6` probability tolerance and reproduced production range
boundaries within `1e-6` seconds.

Most importantly, all 352 exact legacy windows match every old V5-state feature at the
stored eight-decimal precision:

| Check | Result |
| --- | ---: |
| Full-union rows | 704 |
| Legacy rows expected/matched | 352 / 352 |
| Feature columns checked per legacy row | 42 |
| Maximum absolute feature difference | 0.0 |
| Frame extraction errors | 0 |

## Ranker and decoder protocol

The ranker is a small class-balanced logistic head suitable for on-device execution.
Two views are compared:

- `frozen-primary32`: the frozen 22 V5 inputs plus ten production state-gate inputs;
- `union-native34`: those 32 inputs plus internal-candidate kind and generator score.

L2 values `{0.1, 1, 10}` are compared. Each outer LOO fold uses grouped inner LOO
probabilities from the other ten recordings to select the view, L2, threshold, and
decoder. Thresholds use 65 score quantiles. The decoder grid compares independent
thresholding, adjacent-candidate suppression, 8/14/20-second suppression, and a soft
post-six logit penalty of zero or 0.5. There is no hard count cap.

The continuity follow-up recording-normalizes `playerSwapMargin`, chooses its cutoff
only on the outer-fit recordings, and must retain at least 90% of their candidate true
positives.

## Results

### Primary ±4-second end-to-end results

| Variant | Proposals | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current user-designated winner | 62 | 25 | 37 | 25 | 40.32% | 50.00% | 44.64% |
| Frozen winner scored on expanded union | 73 | 26 | 47 | 24 | 35.62% | 52.00% | 42.28% |
| Nested-LOO full-union ranker | 56 | 27 | 29 | 23 | 48.21% | 54.00% | **50.94%** |
| Nested ranker + nested continuity veto | 55 | 26 | 29 | 24 | 47.27% | 52.00% | 49.52% |

The nested ranker adds two true positives, removes eight false positives, and emits six
fewer proposals than the current winner. Its macro per-video precision/recall are
48.05%/54.09%, versus 41.19%/50.91% for the current winner.

The frozen head cannot use the added union safely without retraining: recall rises only
two points while 11 false proposals are added. Candidate coverage and candidate ranking
must therefore be treated as separate stages.

### Strict sensitivity

| Variant | Proposals | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current winner | 62 | 23 | 39 | 27 | 37.10% | 46.00% | 41.07% |
| Frozen winner on expanded union | 73 | 23 | 50 | 27 | 31.51% | 46.00% | 37.40% |
| Nested-LOO full-union ranker | 56 | 21 | 35 | 29 | 37.50% | 42.00% | 39.62% |
| Nested ranker + continuity veto | 55 | 20 | 35 | 30 | 36.36% | 40.00% | 38.10% |

The new ranker's advantage is specific to the declared ±4-second boundary-margin
contract; it does not improve strict timing. This supports improving proposal-time
localization separately from switch/non-switch discrimination.

### Per-video nested-LOO primary result

| Recording suffix | Proposals | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `160023210` | 2 | 2 | 0 | 2 | 100.00% | 50.00% |
| `161923155` | 6 | 4 | 2 | 1 | 66.67% | 80.00% |
| `164327879` | 6 | 4 | 2 | 0 | 66.67% | 100.00% |
| `171720964` | 7 | 4 | 3 | 1 | 57.14% | 80.00% |
| `180646590` | 6 | 4 | 2 | 1 | 66.67% | 80.00% |
| `183701800` | 7 | 3 | 4 | 1 | 42.86% | 75.00% |
| `190429172` | 7 | 2 | 5 | 3 | 28.57% | 40.00% |
| `193307688` | 3 | 0 | 3 | 4 | 0.00% | 0.00% |
| `203801418` | 3 | 1 | 2 | 4 | 33.33% | 20.00% |
| `210449857` | 6 | 2 | 4 | 2 | 33.33% | 50.00% |
| `212717581` | 3 | 1 | 2 | 4 | 33.33% | 20.00% |

### Stability and failure analysis

- L2 `0.1` is selected in all 11 outer folds.
- Adjacent-candidate local suppression is selected in all folds.
- A 0.5 soft-count penalty is selected in 10/11 folds.
- `frozen-primary32` is selected in 6 folds and `union-native34` in 5. Candidate-kind
  features therefore do not have stable recording-level transfer.
- Outer thresholds range from about 0.48 to 0.73, another sign of recording shift.
- Seven internal peaks survive the nested decoder, but only one matches a switch
  (`190429172` near 763.772 seconds). The internal candidate problem is not solved.
- The nested continuity veto removes one proposal, but it is a true positive. It removes
  no false positives and is rejected for this distribution.
- The full-development selection reaches 60% precision / 48% recall / 53.33% F1 with
  40 proposals, but this is selection-scope performance and is not used as the primary
  claim.

## On-device implications

The learned head is only a 34-input logistic classifier, and its selected decoder uses
adjacent local suppression plus a soft count penalty. Production rally/serve/dead-state
heads already exist on device. The added visual work is seven 256×144 frames per decoded
range and two seven-frame flanks per internal candidate; summaries are shared across
neighboring boundaries.

No browser or Android runtime port was implemented in this loop. Runtime parity,
incremental decode cost, memory, and battery measurements remain required before any
automatic use.

## Disposition and next step

Retain `side-switch-full-union-ranker-v1` as the strongest nested-development contender.
Do not replace the checked-in current-winner pointer, do not add it to production, and
reject the expanded-union continuity veto.

The next loop should target imbalance and calibration without changing the candidate
universe: compare recording-normalized calibration and explicitly negative/continuity
objectives against this exact nested protocol. The `193307688` zero-recall fold and the
three unreliably ranked internal recovered events are predeclared failure slices.

## Reproduction and artifacts

```bash
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-full-union-features.py
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-full-union-ranker.py
PYTHONPATH=. .venv/bin/python -m unittest discover -s analysis/tests -p 'test_side_switch*.py'
```

| Artifact | SHA-256 |
| --- | --- |
| `reports/side-switch/side-switch-full-union-v5-state-features-v1.json` | `9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551` |
| `models/side-switch-full-union-ranker-v1/model.json` | `a405b88ec4a76dc925fb46a16b0dafd4e208bf3297ad69b36118e77b77121b46` |
| `reports/side-switch/side-switch-full-union-ranker-v1-evaluation.json` | `e332b03b20387ed0d401c3fab37de31cdaafef76420f9045923c1c4c53919ccf` |

The run completed 82 side-switch unit tests. Exact implementation is in
[`side_switch_full_union_features.py`](../../analysis/side_switch_full_union_features.py),
[`side_switch_production_replay.py`](../../analysis/side_switch_production_replay.py),
[`side_switch_full_union_ranker.py`](../../analysis/side_switch_full_union_ranker.py),
[`extract-side-switch-full-union-features.py`](../../scripts/extract-side-switch-full-union-features.py),
and
[`train-side-switch-full-union-ranker.py`](../../scripts/train-side-switch-full-union-ranker.py).

## Sources

- [Rare-event improvement plan and literature sources](./side-switch-rare-event-improvement-plan-2026-08-23.md)
- [Full-trace candidate-union decision](./side-switch-candidate-union-2026-08-23.md)
- [Same-side continuity-verifier decision](./side-switch-continuity-verifier-2026-08-23.md)
- [Exhaustive full-video marker audit](./side-switch-full-video-marker-audit-2026-08-21.md)
- [Production-state and serve-grounding experiment](./side-switch-production-state-experiment-2026-08-20.md)
- [V5 peak/count/context cleanup](./side-switch-v5-peak-cleanup-2026-08-20.md)
