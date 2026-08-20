# Side-switch specialist v3 — 2026-08-20

## Decision

V3 is implemented, trained, and retrospectively evaluated. Do not promote it to browser
or Android inference. The detector-free low-resolution visual head showed a small
exact-gap validation lift over cadence alone, but failed on the unseen raw-phone source:
the selected decoder reached 6.82% precision, 8.57% recall, and 7.59% exact-gap F1.

The useful outcome is the executable cadence/margin framework and a measured negative
result. The next iteration must improve point-like rally counting or obtain score/redo
state; widening the candidate margin cannot repair an overcounted rally sequence.

## Frozen data and label policy

The split is source-group separated:

| Role | Source groups | Recordings | Rows | Switch gaps |
| --- | --- | ---: | ---: | ---: |
| Train | `yang-showmatch-20260703`, `kb-kob-20250614` | 6 | 234 | 29 |
| Validation | `shoreline-kob-20250616` | 4 | 143 | 17 |
| Retrospective evaluation | `volleycut-raw-no-backup` | 11 | 352 | 35 |

`beach-source-02` is absent from every role. The full review's 734 source events
collapse to 729 unique rally gaps: one duplicate exact marker is collapsed, four unclear
gaps are excluded, and two exact markers are invalidated by the completed visual review.
Thus exact markers supply timing provenance, while the completed review determines
whether the gap visibly contains a switch. No source event is double-counted.

The evaluation source was not opened during v3 selection, but it was used by historical
v1/v2 research. Results are therefore retrospective confirmation, not a pristine
promotion test.

## On-device feature profile

The extractor reads three frames from the first 2.4 seconds of each rally adjacent to a
reviewed gap, applies the recording ROI, and resizes to 192×108. It uses no HOG, person
detector, neural network, or full-resolution crop. Adjacent gaps reuse each rally summary.

The fixed 12-input signature is:

- same-side assignment cost, swapped assignment cost, and their margin;
- near/far orientation flip evidence;
- minimum and changed motion coverage;
- within-rally frame change before and after the gap;
- global HSV appearance change;
- minimum log Laplacian blur;
- luma change; and
- edge-density change.

All 729 rows have finite features and all 21 recordings have zero frame errors. Training
selects L2 from `0.01, 0.1, 1, 10` by recording-held-out average precision. L2 1.0 won
with OOF AP 0.1964. Validation row AP is 0.2766; evaluation row AP is 0.1841 against a
9.94% positive prevalence.

## Cadence decoder

Each recording is one set starting at point total zero, with an opportunity every seven
points. Rally order is only a point-count proxy. The decoder:

- evaluates candidate margins ±1, ±2, ±3, and the requested ±4;
- uses a monotonic one-to-one assignment when ±4 windows overlap;
- permits no selection for weak evidence;
- re-anchors the next seven-rally window after a selected switch; and
- caps a set at six opportunities, using the product prior supplied before evaluation.

Validation selected margin ±1, distance penalty 0.5, and classifier threshold
0.4199503141. The selected visual decoder achieved 80.00% exact-gap F1 on validation,
versus 77.78% for cadence-only ±1. At ±1 rally evaluation tolerance, cadence-only was
better: 94.44% versus 91.43%.

## Retrospective evaluation

The table reports event F1 at exact, ±1-rally, and ±2-rally matching tolerance. Margin is
the decoder's candidate-search window; tolerance changes only scoring.

| Decoder | Candidate margin | Predicted | Exact F1 | ±1 F1 | ±2 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cadence only | ±1 | 51 | 9.30% | 23.26% | 39.53% |
| Cadence only | ±2 | 63 | 10.20% | 26.53% | 46.94% |
| Cadence only | ±3 | 64 | 8.08% | 24.24% | 40.40% |
| Cadence only | ±4 | 64 | 12.12% | 28.28% | 42.42% |
| Cadence + visual | selected ±1 | 44 | 7.59% | 20.25% | 32.91% |
| Cadence + visual | forced ±4 sensitivity | 44 | 7.59% | 20.25% | 32.91% |

For the requested ±4 cadence-only case, exact precision/recall are 9.38%/17.14%; at
±2 scoring tolerance they are 32.81%/60.00%. The visual prediction set is unchanged
across margins because the frozen threshold and distance penalty reject all additional
outer candidates. It does not add useful margin-placement evidence on this source.

The selected visual decoder predicts the exact switch count in 2 of 11 recordings
(18.18%). Cadence-only ±4 predicts it in 0 of 11. The dominant failure is not compute or
blur: the raw rally candidate sequence often contains substantially more intervals than
scored points, so fixed seven-rally opportunities drift away from seven-point score
milestones. Re-anchoring cannot recover until the first visual placement is reliable.

## Immutable artifacts

- Features:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v3-features.json`
  (`e50bad8020d0e1992b65617bd2a8d2ce7f9a5aa407fb0834c19e0921e35c05fa`)
- Selected re-anchored/capped model:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v3-reanchored-capped6/model.json`
  (`d274aefd2582dac0967f6d0e622fa58a038e477d10c76eb858b50d2eaa142a34`)
- Development dataset:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v3-reanchored-capped6/dataset-development.json`
  (`6b65610d3668dc17c453f58213541101fd425a17e4f2393d8c7a0a8a354ffe58`)
- Evaluation:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-specialist-v3-reanchored-capped6-evaluation.json`
  (`c9cfa932718295abafd273b844bba71fdc833d31360bb24fd3a30dc04280ac7c`)
- Provenance:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v3-provenance.json`
  (`9bcc8d33f3a268f991d311a719205b5657544c23943096629c18e3ca70ba598e`)

The deployable model/decoder fingerprint is
`54a31e37b0b89883582a4070d54fd72e76424ce3852b6a8662ef83ae8ab29621`.
The provenance manifest binds implementation revision
`ecadbdd1cbfb4c0d88455a0bd329a495b17ce2d3`, all 21 source videos by full-file
SHA-256, the source manifest and decision map, and every selected artifact above.

The pre-correction fixed-grid diagnostic remains immutable at
`models/side-switch-specialist-v3/` and
`reports/side-switch/side-switch-specialist-v3-evaluation.json`; it is not the selected
v3 result. Its model/dataset/evaluation hashes are respectively `d0509b50…`,
`fe880c77…`, and `adaa7e35…`. It omitted the preregistered re-anchor behavior and the
six-opportunity set cap, so it is retained only to make that correction auditable.

## Disposition and next gate

No TypeScript/Java port or production bundle is created because the source-held-out
result fails. The cheapest meaningful next experiment is to distinguish scored points
from rally candidates—using score markers when available, explicit redo labels, or a
high-precision accepted-point head—then rerun the frozen visual ranker without changing
the raw evaluation result. New untouched recordings are still required for promotion.
