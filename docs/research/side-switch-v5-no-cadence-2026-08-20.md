# Side-switch V5 no-cadence experiment — 2026-08-20

## Decision

The hard seven-point cadence decoder is suppressing real side-switch evidence and can
propagate an early mistake through later opportunities. Keep this no-cadence V5-state
variant as a research/review result, but do not promote it directly to automatic use:
it raises retrospective exact recall from 31.43% to 80.00% and exact F1 from 30.14%
to 44.44%, while increasing proposals from 38 to 91 and false positives from 27 to 63.

The next decoder should therefore replace hard re-anchored cadence with independent
appearance scoring followed by validation-selected local peak/cluster suppression and,
if useful, a soft whole-set count prior. It should not make the next opportunity depend
on whether the preceding candidate was selected.

## Controlled question

The experiment asks one narrow question: does the seven-point opportunity path cause
errors beyond the limitations of the learned V5 appearance/state head?

It reuses the exact frozen V5 production-state feature rows and the same
6-train/4-validation/11-retrospective recording split. Both selectable heads are refit:

- `original:base`, the 22-input `PLAYER-ORIENTATION22` head; and
- `original:state-gate`, the same 22 V5 inputs plus ten production-state/gap inputs.

The blurry `beach-source-02` recording remains excluded from fitting and every
selection stage. Suppression remains quarantined because its target contains side
switches and its fitting recordings overlap this experiment.

The only decoder change is:

| Property | Frozen V5-state cadence control | No-cadence variant |
| --- | --- | --- |
| Candidate opportunity centers | Seven-point path | None |
| Candidate margin | ±1 rally | None; score every reviewed gap |
| Re-anchor after a selection | Yes | No |
| Maximum selections | Six | None |
| Minimum spacing | Cadence path | Zero |
| Decision | Dynamic-program path | Independent `score >= threshold` |

L2 is selected by recording-held-out training average precision. Each independent
threshold and the winning feature view are selected only on the four validation
recordings. Retrospective labels are opened once after that selection is locked.

This is a true decoder ablation, not a newly favorable classifier fit. For both feature
views, the refitted imputation, normalization, weights, bias, and L2 match the cadence
model exactly; the maximum absolute parameter difference is `0.0`. Threshold is
intentionally reselected because its meaning changes when every gap can be emitted.

## Validation selection

Validation selects `original:state-gate`, L2 `1.0`, and independent threshold
`0.4470970867570025`.

| No-cadence candidate | P | R | Exact F1 | ±1 F1 | ±2 F1 | Proposals | Row AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V5 base | **50.00%** | 52.94% | 51.43% | 57.14% | **68.57%** | 18 | 53.46% |
| **V5 + state gate (selected)** | 48.15% | **76.47%** | **59.09%** | **63.64%** | 68.18% | 27 | **65.91%** |

The selected validation output already shows why an independent threshold is not the
finished product decoder: its 27 proposals form 21 consecutive-gap clusters, include
six adjacent selected pairs, and contain a run of four consecutive gaps.

## Retrospective result

All values below use the validation-selected V5-state feature view and threshold on the
same 11 raw-phone recordings used by the cadence control.

| Decoder | TP / FP / FN | P | R | F1 | Proposals |
| --- | --- | ---: | ---: | ---: | ---: |
| Seven-point cadence, exact | 11 / 27 / 24 | 28.95% | 31.43% | 30.14% | 38 |
| **No cadence, exact** | **28 / 63 / 7** | **30.77%** | **80.00%** | **44.44%** | 91 |
| Seven-point cadence, ±1 rally | 15 / 23 / 20 | 39.47% | 42.86% | 41.10% | 38 |
| **No cadence, ±1 rally** | **30 / 61 / 5** | 32.97% | **85.71%** | **47.62%** | 91 |
| Seven-point cadence, ±2 rallies | 17 / 21 / 18 | **44.74%** | 48.57% | 46.58% | 38 |
| **No cadence, ±2 rallies** | **30 / 61 / 5** | 32.97% | **85.71%** | **47.62%** | 91 |

Exact no-cadence deltas are +53 proposals, +17 true positives, +36 false positives,
-17 false negatives, +1.82 percentage points of precision, +48.57 points of recall,
and +14.31 points of F1. Row AP is unchanged at 44.30% because cadence does not alter
row scores.

The no-cadence base head is not selected because the state head won validation. As a
post-lock retrospective diagnostic, base produces 88 proposals with 31.82% precision,
80.00% recall, 45.53% exact F1, and 48.78% ±1/±2 F1. Its slightly better retrospective
result cannot replace the validation winner.

## Does cadence lose the correct switches?

Yes. Across the 35 positive reviewed gaps:

| Exact positive outcome | Count |
| --- | ---: |
| Selected by both decoders | 11 |
| Selected only by cadence | 0 |
| Selected only without cadence | 17 |
| Selected by neither | 7 |

The independent variant retains every exact true positive selected by cadence and
recovers 17 additional exact positives. Representative recordings make the persistence
problem visible:

| Recording suffix | Labeled switch gaps | Cadence exact hits | No-cadence exact hits |
| --- | --- | --- | --- |
| `160023210` | 8, 17, 27, 34 | 34 | 8, 17, 34 |
| `171720964` | 13, 24, 39, 49, 59 | 13 | 13, 24, 39, 49, 59 |
| `180646590` | 10, 22, 34, 55 | 34 | 10, 22, 34, 55 |
| `190429172` | 8, 25, 45 | 8 | 8, 25, 45 |

## Direct cascade evidence

Cadence is not merely filtering independent probabilities. Its dynamic-program path
can accept a locally poor selection when that selection makes later re-anchored choices
more valuable. On `raw-no-backup-PXL_20260816_164327879`, it selects negative gap 12
at score `0.2661506538796391`, below the cadence classifier threshold
`0.37157755318599694`.

The selected sequences also contain repeated cadence tracks after false choices:

- On `raw-no-backup-PXL_20260816_190429172`, truth is at gaps 8, 25, and 45. Cadence
  selects 8 correctly, then selects 15, 22, 29, 36, and 43—all false and exactly seven
  gaps apart.
- On `raw-no-backup-PXL_20260816_193307688`, truth is at gap 29. Cadence selects gaps
  15, 21, 28, and 35, all false.
- On `raw-no-backup-PXL_20260816_160023210`, truth is at 8, 17, 27, and 34. Cadence
  selects false gaps 7 and 28 before its only exact hit at 34.

These examples support the original concern: a mistaken cadence opportunity can alter
where the decoder looks next and keep later high-scoring true gaps off the path.

## Output structure and next decoder

Pure independent thresholding supplies no count control. Its 91 retrospective proposals
form 69 consecutive-gap clusters, include 22 adjacent selected pairs, and have a maximum
run of five consecutive gaps. Exact per-recording count accuracy remains 0/11.

The appropriate follow-up is a decoder selected only on development data with this
order of operations:

1. independently score every reviewed inter-rally gap;
2. merge adjacent above-threshold gaps or retain one local peak per short cluster;
3. apply a minimum spacing justified by switch duration, not seven-point score cadence;
4. optionally add a soft count/rate prior that cannot re-anchor later search windows;
5. retain the cadence result as a feature or diagnostic only, never as a hard candidate
   generator.

That design preserves the recovered appearance evidence while addressing the 91-output
precision cost. It also remains on-device friendly: the no-cadence ablation adds no
video decode, neural network, or feature computation and simplifies sequence inference.
No browser or Android runtime is added by this research artifact.

## Scope limitations

- The retrospective recordings and labels were opened during earlier V1–V6 research;
  this is confirmation on a fixed historical scope, not a pristine protected test.
- Metrics cover the frozen reviewed rally-gap inventory, not exhaustive full-video
  truth. They must not be interpreted as deployed full-video accuracy.
- Labels determine evaluation only. Consecutive-cluster diagnostics are label-free and
  are reported so later suppression choices cannot silently use retrospective truth.
- The validation-selected state variant stays selected even though base is slightly
  stronger retrospectively.

## Reproduction

The feature extraction is already frozen. Refit and evaluate with:

```bash
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v5-no-cadence.py freeze
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v5-no-cadence.py evaluate
PYTHONPATH=. .venv/bin/python scripts/build-side-switch-v5-no-cadence-provenance.py
```

Every destination refuses overwrite. The provenance builder verifies the inherited
production-state feature/source identities and binds the implementation revision.

## Immutable artifacts

All paths are under `/mnt/freenas/volleycut/labeling-v1-2026-08-09/`.

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-v5-no-cadence-v1/model.json` | `5c89d60c865812f3c42ab1ffa4265d19d1ff9c3de82ea8ac7633850e7a5e9d17` |
| `models/side-switch-v5-no-cadence-v1/dataset-development.json` | `a57f355f7ce56cbd7bee0c4bce24b737045f8c8b24d283a14148de0f1790b309` |
| `reports/side-switch/side-switch-v5-no-cadence-v1-evaluation.json` | `c249f17b9b77471609c651656b7ca3cfc66d18168380a7ba5b450008e9bdcfde` |

- selected fingerprint:
  `c8e6014892b0012971d288fd60eadb81150a9646ec0b4c5ce6c77e76eeca76ab`;
- implementation revision:
  `b7c732d61b2d242ee50f28af1686f9fab73d6a21`; and
- provenance:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v5-no-cadence-v1-provenance.json`,
  SHA-256 `b0a6d21df15c4c5f7f968efa600836bd1b13c6d72791ce746c0c89c7725d64e2`.
