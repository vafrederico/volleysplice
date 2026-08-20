# Side-switch specialist v1 — 2026-08-20

## Decision

The first side-switch marker specialist is trained and evaluated, but it is not
accurate enough for automatic score tracking. Keep it as a review-ranking baseline.
On the source-group-independent raw recordings it reaches 11.50% precision, 37.14%
recall, and 17.57% F1, with ranking performance close to chance.

## Frozen review data

The full-NAS appearance report has 1,096 generated markers. The saved NAS review has
one decision for every marker: 88 `switch`, 1,004 `no-switch`, and 4 `unclear`.
Unclear decisions are excluded from fitting, threshold selection, and evaluation.

The specialist is candidate-conditioned. Its metrics answer whether a generated
marker is a switch; they do not measure switches that the marker generator failed to
propose.

## Leakage-safe split

| Role | Inclusion | Rows | Switch | Recordings |
|---|---|---:|---:|---:|
| Train | Declared `train`, beach or grass | 151 | 20 | 4 |
| Threshold validation | Declared `validation`, non-indoor | 74 | 9 | 2 |
| Source-group-independent evaluation | Raw `non-training`, non-indoor | 352 | 35 | 11 |
| Repeated-group challenge diagnostic | `challenge`, non-indoor | 190 | 24 | 5 |
| Fixed indoor evaluation | Same held-out splits, indoor | 253 | 0 | 6 |

Feature-family and L2 selection use leave-one-recording-out predictions across the
four training recordings. The selected family is then refit on all training rows.
The probability threshold is selected once on validation. Only then is the frozen
model scored on challenge/raw/test rows.

Recording IDs are disjoint. All raw evaluation rows come from the unseen
`volleycut-raw-no-backup` source group. Challenge recordings repeat development
groups: `kb-kob-20250614` was used in training and `shoreline-kob-20250616` was used
for validation. Challenge is therefore a diagnostic, not the primary held-out result.
There is also no held-out beach recording; beach generalization remains unmeasured.

Indoor is outside the specialist's scope because reviewed indoor recordings have no
side-switch markers. Product policy therefore returns `no-switch` for indoor footage.
Those 253 easy negatives are reported separately and never included in the specialist
precision/recall headline.

## Model and selection

The implementation is a deterministic, class-balanced logistic model using the
already-extracted marker features. It compares three existing-label variants:

- palette-area distance only;
- the seven appearance/detection-change scores;
- those seven scores plus before/after detection count, box area, box height,
  detector score, usable-frame context, and inter-rally gap length.

Every numeric input has a paired missingness indicator; finite values use
training-only median imputation and standardization. The search evaluates L2 values
`0.01`, `0.1`, `1.0`, and `10.0`.

The selected variant is `appearance-context`, L2 `0.1`. Its grouped training OOF
result was 51.61% precision, 80.00% recall, and 62.75% F1. Validation selected
threshold `0.5011574986`, producing 44.44% precision, 44.44% recall, and 44.44% F1
on 9 positive and 65 negative validation markers. The large train-to-validation drop
is an early generalization warning.

## Untouched evaluation result

| Scope | TP | FP | FN | Precision | Recall | F1 | ROC AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw non-training (primary) | 13 | 100 | 22 | 11.50% | 37.14% | 17.57% | 0.538 | 0.110 |
| Challenge grass (repeated-group diagnostic) | 14 | 42 | 10 | 25.00% | 58.33% | 35.00% | 0.736 | 0.263 |
| All non-indoor rows (diagnostic) | 27 | 142 | 32 | 15.98% | 45.76% | 23.68% | 0.621 | 0.163 |

The fixed product policy adds 253 correctly gated indoor negatives, but its switch
precision, recall, and F1 remain exactly the specialist values above. Overall policy
accuracy rises to 78.11%; that number is not suitable for model selection because it
is dominated by no-switch markers.

The result supports the next-model direction already identified during review:
side-conditioned team appearance representations, a detector with adequate beach
player coverage, and temporal consistency across adjacent rallies. Threshold tuning
alone cannot repair the raw-recording domain shift.

## Immutable NAS artifacts

- Model: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v1/model.json`
  (`b24f0fa1e23bbb5c257887a51079f2805cd56f325dcfc9255f6538da19089477`)
- Reproducible selected-feature dataset:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v1/dataset.json`
  (`3c2f78a81ca6400f919cb83b8209551dfe0781c6c2b207339a0679a9b2f95df2`)
- Evaluation and per-marker predictions:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-specialist-v1-evaluation.json`
  (`163312c99dd05399d08fca8639c597601c96e6791bbbe5e010dd81e9fe8d742b`)

The stable model-parameter fingerprint is
`5475e13addbe22940ae4a9d84e999d9e6db98a49ae77380ddd65505afbad4b7b`.
The trainer refuses to overwrite any of these destinations.
