# Side-switch full-trace candidate union — 2026-08-23

## Decision

Retain `deadState-threshold-0.98-separation-14` as the candidate generator for the
next feature-extraction/ranking loop. It passes the predeclared 90% candidate-recall
target, selects identically in all 11 leave-one-recording-out (LOO) folds, and adds no
new model inference or video decode.

Do not expose its 704 internal candidates as editor proposals, and do not replace the
current `local-peak-soft-count` winner. The frozen V5 stack can score only half of this
expanded universe; final output therefore remains unchanged until new appearance/state
features are extracted for the other 352 candidates.

## Why the old universe misses switches

The prior exhaustive audit found only 33/50 markers inside any of the 352 V5 candidate
gaps at the primary ±4-second allowance. Loop 1 showed that every production rally
boundary—not only the historically reviewed subset—raises coverage to 43/50 across 624
boundaries.

All remaining seven markers fall **inside** production ranges that were classified as
rallies. They are not ordinary missing gaps:

- `180646590`: 452.620;
- `190429172`: 763.772;
- `203801418`: 280.119, 460.978, and 656.290; and
- `212717581`: 542.833 and 912.828.

A production-rally hard gate would make these events unrecoverable. The union instead
searches strong dead-state evidence inside those ranges.

## Candidate generator

The input is already present in each on-device model-feedback trace:

- decoded production rally ranges; and
- 4 Hz `rally` and `deadState` probabilities from `model-1ca43e38eefc`.

Every adjacent decoded-range boundary becomes a candidate. Inside each range, the
selected internal path:

1. excludes four seconds at each edge because the adjacent-boundary path already owns
   that region;
2. keeps samples with `deadState >= 0.98`;
3. applies score-ranked 14-second non-maximum suppression within the range; and
4. emits a two-second interval centered on each surviving peak.

The candidate selector compares 30 fixed configurations: dead-state versus
`max(deadState, 1-rally)`, thresholds 0.70/0.80/0.90/0.95/0.98, and 6/10/14-second
separation. It chooses the smallest average candidate load reaching 90% recall on the
fit recordings, breaking ties by higher recall and stable configuration ID.

This is candidate generation, not event classification. Candidate hit rate below must
not be reported as model precision.

## Results

### Candidate coverage

| Universe | Candidates | Per video | Covered at ±4 | Recall | Candidate hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior V5 gaps | 352 | 32.00 | 33/50 | 66% | 9.38% |
| All adjacent rally boundaries | 624 | 56.73 | 43/50 | 86% | 6.89% |
| Selected boundary + internal-peak union | 704 | 64.00 | 46/50 | 92% | 6.53% |

The selected union consists of 624 boundaries and 80 internal peaks, or 7.27 internal
peaks per video. Internal peaks recover one additional marker each in `190429172`,
`203801418`, and `212717581`. Four remain uncovered at ±4 seconds: 452.620 in
`180646590`; 280.119 and 460.978 in `203801418`; and 542.833 in `212717581`.

Strict interval containment is 36/50 (72%). The selected primary contract remains ±4
seconds because markers identify an instant within a physical transition and the peak
proposal is only two seconds wide.

### Recording-held-out stability

Every LOO fold independently selects
`deadState-threshold-0.98-separation-14`. Applying each fold's selection to its held-out
recording reproduces the same 704 candidates and 46/50 (92%) aggregate recall. This is
strong within-scope transfer, though the recordings and truth were already opened.

### Downstream compatibility

| Stage | Count/result |
| --- | ---: |
| Selected union candidates | 704 |
| Exact windows with frozen V5 features | 352 |
| New windows needing appearance/state extraction | 352 |
| Compatible current-winner + LOO continuity output | 57 proposals; 25 TP / 32 FP / 25 FN; 46.73% F1 |

The 272 additional rally boundaries and 80 internal peaks have no frozen
`PLAYER-ORIENTATION22+STATE10` row. Assigning them a fake score or copying a neighboring
gap would violate feature provenance. This loop therefore stops at candidate coverage
and passes only exact legacy windows through the frozen scorer/verifier. The next loop
must extract the same comparison features around all selected windows, then fit or
calibrate a ranker without changing this candidate generator.

## On-device impact

The generator adds no video decode, feature extraction, or neural inference. It scans
existing 4 Hz dead-state probabilities once and runs short per-range NMS. Its memory is
the small candidate list plus the already-produced probability trace. The 64/video
figure is internal ranking load, not intended UI proposal volume.

## Provenance note

One of the 11 mutable model-feedback JSON files no longer matches the whole-file hash
stored in the earlier corpus manifest. The evaluator does not silently trust that
mutable file identity: it validates the production model/ensemble IDs, verifies that
decoded ranges equal the frozen manifest ranges, and stores a canonical SHA-256 of each
exact `initialInference` block. Whole-file current/manifest hashes and their match flag
are also retained in the artifact.

## Implementation and artifact

- Candidate construction:
  [`side_switch_candidate_union.py`](../../analysis/side_switch_candidate_union.py)
- Immutable grid/LOO evaluator:
  [`evaluate-side-switch-candidate-union.py`](../../scripts/evaluate-side-switch-candidate-union.py)
- Focused tests:
  [`test_side_switch_candidate_union.py`](../../analysis/tests/test_side_switch_candidate_union.py)
- NAS artifact:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-candidate-union-v1-evaluation.json`
- Artifact SHA-256:
  `c4717a6e056b659fc7c63541a2eac13451518b0720dd7362dfb49643688edbc2`

Run from the repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-side-switch-candidate-union.py
PYTHONPATH=. .venv/bin/python -m unittest analysis.tests.test_side_switch_candidate_union
```
