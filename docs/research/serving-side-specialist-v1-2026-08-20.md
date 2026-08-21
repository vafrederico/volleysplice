# Serving-side specialist v1 — 2026-08-20

## Decision

Retain the first serving-side specialist as a strong research baseline, but do not
promote it to automatic score tracking yet. It reaches 90.41% balanced accuracy and
90.45% macro-F1 on the 389 reviewed raw non-training rallies, while the single
protected-test indoor recording reaches only 62.30% balanced accuracy. The split
between strong raw/grass behavior and weaker indoor behavior requires more independent
indoor development coverage before product integration.

## Frozen review data

The source report contains 1,424 unique rally candidates with a complete NAS-backed
review: 568 `near`, 561 `far`, and 295 `unclear`. Unclear/Ignore rows are excluded from
fitting, model selection, and metrics, leaving 1,129 binary near/far examples.

The task is candidate-conditioned. These metrics classify serving side for generated
rally rows; they do not measure missed rallies, rally-boundary correctness, or serve
anchor recall.

## Leakage-safe protocol

| Role | Rows | Near | Far | Recordings | Use |
|---|---:|---:|---:|---:|---|
| Train | 217 | 106 | 111 | 6 | Recording-grouped OOF feature-family/L2 selection and final weight fitting |
| Validation | 76 | 32 | 44 | 2 | One operating-threshold selection |
| Challenge | 408 | 203 | 205 | 10 | Held-out diagnostic; some source groups repeat development groups |
| Raw non-training | 389 | 206 | 183 | 11 | Held-out source-group-independent raw-camera result |
| Protected test | 39 | 21 | 18 | 1 | Opened once after model and threshold freeze |

Recording IDs are disjoint across train, validation, and held-out roles. Model-family
and L2 selection maximize recording-grouped training OOF balanced accuracy, with
macro-F1 and accuracy tie-breaks. The selected family is refit on training only. The
threshold then maximizes validation balanced accuracy with the same tie-breaks. No
held-out label participates in either decision.

## Model and existing-feature bank

The implementation is a deterministic, class-balanced logistic model requiring only
NumPy. Five predeclared feature families compare whole-half motion/palette margins,
baseline-band margins, their combination, the seven existing signed margins, and all
19 existing serving-side scalars. Each input has a paired missingness indicator.
Training-only median imputation and standardization are stored in the model.

The selected model uses `full-existing-bank`: 19 whole-half, baseline-band, and HOG
change values plus 19 missingness indicators, L2 `10.0`, and near-probability threshold
`0.5116070408`. Its stable model-parameter fingerprint is
`364dbd9e961e0d15146c4bf403e62b00345f3ebdf9e2df5a923b5f0e120d94cf`.

## Results

| Scope | Rows | Balanced accuracy | Macro-F1 | Accuracy | Near recall | Far recall |
|---|---:|---:|---:|---:|---:|---:|
| Training recording-grouped OOF | 217 | 82.67% | 82.62% | 82.95% | 70.75% | 94.59% |
| Validation | 76 | 96.88% | 97.28% | 97.37% | 93.75% | 100.00% |
| Raw non-training | 389 | 90.41% | 90.45% | 90.49% | 91.75% | 89.07% |
| Source-group-independent held-out | 654 | 81.07% | 80.83% | 80.89% | 73.43% | 88.71% |
| Challenge | 408 | 75.64% | 74.79% | 75.74% | 56.65% | 94.63% |
| All held-out | 836 | 82.19% | 81.86% | 81.94% | 73.26% | 91.13% |
| Protected test | 39 | 62.30% | 61.44% | 61.54% | 52.38% | 72.22% |

On raw non-training recordings, the strongest fixed signed-score baseline is baseline
motion plus palette at 85.22% balanced accuracy. The learned existing-feature baseline
adds 5.19 percentage points there. Challenge grass reaches 86.32% balanced accuracy,
but challenge indoor reaches 66.64%; the broader independent indoor scope reaches
66.21%. This indoor gap is the main reason not to promote v1.

## Immutable NAS artifacts

- Model:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serving-side-specialist-v1/model.json`
  (`97356fe4ad382774451e19b82a67292eafe7ddf12114c64f6e7d6d70bad3862a`)
- Reproducible selected-feature dataset:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serving-side-specialist-v1/dataset.json`
  (`7d28ee5ad23a3d5f2427de54aadf673f60332c56ced80aceed4899e99f828881`)
- Evaluation, heuristic comparisons, and per-rally predictions:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-specialist-v1-evaluation.json`
  (`59e9ffa15f148742e07cb6b2fd6e9e89ce53122d8bfbeace224a8e2fbf8fc7e1`)

The trainer binds to the frozen report and decision SHA-256 values, records exact
implementation-file hashes, and refuses to overwrite any artifact destination. Because
held-out results are now known, any new feature family or calibration strategy is a v2
hypothesis and must use a newly declared development protocol rather than reselecting
against these v1 held-out metrics.
