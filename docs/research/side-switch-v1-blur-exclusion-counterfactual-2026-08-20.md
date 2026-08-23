# Side-switch v1 blur-exclusion counterfactual — 2026-08-20

## Question and decision

Would the historical side-switch v1 specialist improve if
`beach-source-02`, whose image becomes blurry roughly halfway through, had not
participated in training?

No. The counterfactual is retained as a reproducible diagnostic, but it does not replace
the historical v1 artifact and is not promoted. The primary source-group-independent F1
decreased from 17.57% to 17.02%; ROC AUC decreased from 0.538 to 0.524 and average
precision decreased from 0.110 to 0.106.

The exclusion remains mandatory for v3 and later fitting because it is a known input
quality defect. This result says only that removing it does not fix v1's weak feature
representation and domain shift.

## Controlled refit

The original appearance report, saved review decisions, feature families, L2 grid,
grouped training selection, validation recordings, evaluation recordings, optimizer,
and seed behavior were held fixed. The only change was removing
`beach-source-02` before grouped model-family selection and fitting.

Training changed from 151 rows, 20 switches, and four recordings to 114 rows, 14
switches, and three recordings. The selected feature family remained
`appearance-context`; L2 changed from 0.1 to 1.0 and the validation-selected threshold
changed from 0.5011574986 to 0.4324883545.

The run was created with:

```bash
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-specialist.py \
  --exclude-fit-recording beach-source-02 \
  --model-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v1-no-blurry-beach-2026-08-20 \
  --evaluation-output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-specialist-v1-no-blurry-beach-2026-08-20-evaluation.json
```

## Results

| Scope | Variant | TP | FP | FN | Precision | Recall | F1 | ROC AUC | AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw non-training, 11 recordings | Original v1 | 13 | 100 | 22 | 11.50% | 37.14% | 17.57% | 0.538 | 0.110 |
| Raw non-training, 11 recordings | No-blur refit | 12 | 94 | 23 | 11.32% | 34.29% | 17.02% | 0.524 | 0.106 |
| Repeated-group challenge | Original v1 | 14 | 42 | 10 | 25.00% | 58.33% | 35.00% | 0.736 | 0.263 |
| Repeated-group challenge | No-blur refit | 16 | 54 | 8 | 22.86% | 66.67% | 34.04% | 0.740 | 0.257 |
| All non-indoor diagnostic | Original v1 | 27 | 142 | 32 | 15.98% | 45.76% | 23.68% | 0.621 | 0.163 |
| All non-indoor diagnostic | No-blur refit | 28 | 148 | 31 | 15.91% | 47.46% | 23.83% | 0.622 | 0.166 |
| V2 four-recording confirmation | Original v1 | 4 | 27 | 7 | 12.90% | 36.36% | 19.05% | 0.590 | 0.124 |
| V2 four-recording confirmation | No-blur refit | 5 | 28 | 6 | 15.15% | 45.45% | 22.73% | 0.570 | 0.123 |

The small all-non-indoor and four-recording F1 increases conflict with the primary
11-recording decline and with lower confirmation ROC AUC/AP. They are not a promotion
signal. V2 itself requires no counterfactual refit: its frozen five-recording training
split contains only `raw-no-backup-*` recordings and never included the blurry beach
video.

## Immutable artifacts

- Model:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v1-no-blurry-beach-2026-08-20/model.json`
  (`14223b8f09034f87639e6b13c4478ba0840b526cac8387d7e01786bef24f9d52`)
- Selected-feature dataset:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v1-no-blurry-beach-2026-08-20/dataset.json`
  (`b10e682b25c37c8f71a370ca7af193affa5988fa95119b632b053026f6931c73`)
- Evaluation:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-specialist-v1-no-blurry-beach-2026-08-20-evaluation.json`
  (`a1618da7f38a9169bf6ae47775da1e5def3e214a01ec85c9cd26b834c4147fae`)

The stable model-parameter fingerprint is
`cf761aaaea6f87ec9c5ad0be2320145c728d08d4b85cc71eeef9d89bd61393ae`.
The model artifact records the exclusion and both pre-exclusion and post-exclusion
training counts. Existing v1 and v2 immutable artifacts were not modified.
