# Side-switch specialist v2 — 2026-08-20

## Decision

Retain v2 as a stronger research/review-ranking specialist, but do not use it for
automatic score tracking. On the four frozen confirmation recordings it improves
same-scope v1 F1 from 19.05% to 32.73%, ROC AUC from 0.590 to 0.796, and average
precision from 0.124 to 0.373. Its absolute precision is still only 20.45%, its
validation ranking was at chance, and the selected temporal decoder is an exact no-op.

The result is candidate-conditioned: it classifies generated inter-rally gaps and says
nothing about real switches that the rally candidate generator failed to expose.

## Frozen recording split

The split was committed before v2 frame extraction. All recordings share source group
`volleycut-raw-no-backup`, so this is recording-held-out confirmation rather than
source-group-independent or pristine test evaluation.

| Role | Recordings | Reviewed rows | Switch | No-switch | Unclear |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fit | 5 | 155 | 20 | 134 | 1 |
| Threshold/decoder selection | 2 | 70 | 4 | 66 | 0 |
| Confirmation evaluation | 4 | 129 | 11 | 117 | 1 |

Unclear decisions are excluded. Feature extraction, geometry calibration, orientation
binding, and recording normalization use no decisions. The freeze phase materializes
only fit and validation rows; confirmation decisions are first materialized by the
separate evaluation command after the feature family, L2, threshold, and decoder are
frozen.

The decision mapping is unchanged from the handoff. A later UI autosave changed only
`savedAt`, so the full-file SHA-256 changed to
`d21d38990b0cb9d3da2942442a9e7a140185affaf5a3dae39367b7a653d1e0eb`.
The canonical mapping SHA-256, which excludes mutable save metadata, is
`5a59727692470ab4984a29d0d48138525e1b5f28b56b3fab630c77070ce1dbd7`.
Reconstructing the current mapping with the handoff timestamp reproduces the handoff's
exact full-file hash.

## V2 feature generation

Each candidate uses four frames before and four after its midpoint, within an 8-second
flank and with a 0.75-second edge margin. The OpenCV HOG proposal input is limited to
1,280 pixels wide. Central-court proposals are filtered in normalized coordinates.

Camera distance and framing are not represented by one fixed pixel boundary. For every
recording, unlabeled weighted two-means over detected player foot positions estimates
the near/far divider from the complete candidate-frame sequence. Each event then fits a
local divider and shrinks it toward the recording estimate; sparse or degenerate local
geometry falls back to the recording value. All 11 recording-wide calibrations
succeeded. Event dividers ranged from 0.6792 to 0.7828 normalized frame height; 240 of
354 events used local refinement and 114 used the recording fallback.

The 17 scalar base features are:

- seven derived existing signals: equal/area palette mean, minimum, disagreement,
  player-minus-background evidence, geometry stability, and two palette-by-stability
  interactions;
- ten side-conditioned signals: median near/far same-assignment cost, median swapped
  cost, median and lower-quartile swap margin, fraction of pairs supporting a swap,
  margin variance, usable pair count, minimum before/after coverage, median continuous
  color-moment flip evidence, and swap-margin-by-stability.

Every scalar has an explicit missingness indicator. The selected `SIDE34-V2` signature
therefore has 34 ordered inputs. Raw gap duration is retained only as audit context and
is absent from every learned feature family.

The artifact also binds unlabeled per-recording median/MAD normalized variants over the
complete candidate sequence and a first-principal-component orientation coordinate for
the decoder. Model selection preferred the raw combined family. Detector coverage is a
material limitation: 112 events had zero usable near/far frame pairs, only 34 had all
four, and mean minimum-side coverage was 43.15%.

## Model and development selection

The classifier is deterministic class-balanced logistic regression. Feature family and
L2 selection use leave-one-recording-out predictions across the five fit recordings.
The search covers four feature families and L2 values `0.01`, `0.1`, `1.0`, and `10.0`.

The winner is raw `combined`, L2 `0.01`. Its grouped OOF operating point produced 11 TP,
23 FP, 9 FN, 32.35% precision, 55.00% recall, 40.74% F1, ROC AUC 0.672, and AP 0.296.

Validation selected threshold `0.4587729277`. It produced 3 TP, 28 FP, 1 FN, 9.68%
precision, 75.00% recall, 17.14% F1, ROC AUC 0.500, and AP 0.073. This large instability
is a promotion warning despite the later confirmation improvement.

The separate Viterbi decoder searched rally-order spacing penalties, parity/orientation
weights, and extra switch penalties on validation only. Validation selected every
setting as zero. The decoder is therefore reported independently but makes the same
binary decisions as the static classifier; no temporal complexity was earned.

## Frozen confirmation result

| Recording | TP | FP | FN | Precision | Recall | F1 | ROC AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `PXL_20260816_160023210` | 2 | 4 | 2 | 33.33% | 50.00% | 40.00% | 0.604 | 0.373 |
| `PXL_20260816_161923155` | 3 | 3 | 0 | 50.00% | 100.00% | 66.67% | 1.000 | 1.000 |
| `PXL_20260816_203801418` | 3 | 17 | 0 | 15.00% | 100.00% | 26.09% | 0.944 | 0.792 |
| `PXL_20260816_212717581` | 1 | 11 | 0 | 8.33% | 100.00% | 15.38% | 0.921 | 0.250 |
| **Pooled static** | **9** | **35** | **2** | **20.45%** | **81.82%** | **32.73%** | **0.796** | **0.373** |
| **Pooled decoder** | **9** | **35** | **2** | **20.45%** | **81.82%** | **32.73%** | **0.796** | **0.373** |

On these same 128 reviewed rows, v1 produced 4 TP, 27 FP, 7 FN, 12.90% precision,
36.36% recall, 19.05% F1, ROC AUC 0.590, and AP 0.124. V2 is a meaningful ranking
improvement, but 35 false positives for 9 true positives is not an acceptable automatic
scoring operating point. Indoor remains fixed to no-switch and outside all specialist
metrics.

## Immutable artifacts and provenance

Implementation revision: `9382fb932fdd873f7b6e0907645a97add2e5fa05`.

- Decision-free feature artifact:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v2-features.json`
  (`8cad75f8800ca84544751980420ccd6bdf7ddac0ab61f0450eb6c5dcf1a6f321`)
- Frozen model:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v2/model.json`
  (`376425c2d70d3e2418de9592e376e40a6f2a6c5b398ac9f8f65278c0839b3216`)
- Development-only selected-feature dataset:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v2/dataset-development.json`
  (`2bd711e612a64cdfef421204cf97334a531972756cd32bddc9e7edb2db92fdc5`)
- Confirmation evaluation and per-marker predictions:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-specialist-v2-evaluation.json`
  (`d06c2a77c2f7014148cf4f4dcbd6d70a1cae30384ab71af8f805292d1426f433`)
- Full source/artifact provenance, including all 11 full-file video hashes:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v2-provenance.json`
  (`3b97b66745b9cddf236448022dc49f33565809fb28a53af6d1cc3452b712caa4`)

The stable deployable model/decoder fingerprint is
`b24265fdac554eece6d8e1e446290be1797e9f3669005fa4f1c392fb72c1671f`.
All writers refuse to overwrite their destinations.
