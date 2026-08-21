# Serving-side review freeze — 2026-08-20

## Decision

The full-NAS serving-side review is complete and frozen as the input contract for
future serving-side model work. The review contains one decision for every generated
rally candidate: 568 `near`, 561 `far`, and 295 `unclear` decisions across 1,424
unique rallies and 30 recordings.

`near` and `far` are fixed camera-space labels. The review UI presents `unclear` as
**Ignore**; those rows must be excluded from fitting, threshold selection, and model
evaluation.

## Frozen inputs

| Artifact | Path | SHA-256 |
|---|---|---|
| Full-NAS serving-side report | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-existing-label-variants-full-nas-v2.json` | `611d698961534740b3b07624a2ade1d574de4e06b8bf5ce701b641b4690fc35b` |
| Completed review decisions | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-review-decisions-full-nas-v1.json` | `1066edcf9579d3c92157a8504dbe5d798023ebb3ff4605e0dad9e451c97080b4` |

The report identity is
`volleycut-serving-side-existing-label-variants-v2`, created at
`2026-08-20T03:10:33.547792+00:00`. The decision file is bound to that exact kind and
timestamp and was last saved at `2026-08-20T08:22:36.256Z`.

Any later label correction must create a newly named revision and a new recorded
hash. It must not silently replace these frozen inputs.

## Completeness verification

- Report rows: 1,424
- Unique report rally IDs: 1,424
- Saved decisions: 1,424
- Missing decisions: 0
- Unknown decision IDs: 0
- Report identity match: yes

| Declared split | Rows | Near | Far | Ignore | Recordings |
|---|---:|---:|---:|---:|---:|
| Train | 230 | 106 | 111 | 13 | 6 |
| Validation | 76 | 32 | 44 | 0 | 2 |
| Challenge | 444 | 203 | 205 | 36 | 10 |
| Non-training | 635 | 206 | 183 | 246 | 11 |
| Test | 39 | 21 | 18 | 0 | 1 |

## Modeling guardrails

The labels are candidate-conditioned: they classify the serving side for generated
rally rows and do not measure missed rallies or serve-anchor recall. Preserve the
recording-level split boundaries. Model and feature-family selection belongs only on
the declared development scope, threshold selection belongs on validation, and the
protected test split must not be opened for iteration selection.

Candidate-only rows are now human reviewed, but that does not retroactively make the
underlying rally boundaries gold. Serving-side metrics and rally-detection metrics
must remain separate.
