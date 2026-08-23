# Side-switch same-side continuity verifier — 2026-08-23

## Decision

Retain `side-switch-continuity-verifier-v1/player-veto` as a promising research layer
for the next candidate-union experiment. Do not replace the user-selected
`local-peak-soft-count` control, publish the verifier to the review UI, or port it to
browser/Android inference yet.

The conservative leave-one-recording-out (LOO) veto removes five false proposals and
no true proposals. This is the useful version of “predict the opposite”: explicit,
strong same-side continuity can veto a switch. A generic hard requirement that swapped
assignment merely be cheaper is decisively rejected because it destroys recall.

## Experiment

This is Loop 2 from the
[rare-event improvement plan](./side-switch-rare-event-improvement-plan-2026-08-23.md).
It uses the exhaustive 50-marker truth and the frozen 352 V5 candidate rows. Every gap
receives a full-marker label through monotonic one-to-one interval matching with the
primary ±4-second boundary allowance; 33 gaps match a marker and 319 do not.

The local player signal is the existing V5 assignment margin:

```text
playerSwapMargin = sameCost - swapCost
```

A lower value is stronger evidence that the same team-to-side mapping continues. The
signal is normalized within each recording by subtracting its median and dividing by
`max(1.4826 × MAD, 0.25 × standard deviation, 1e-6)`. This normalization uses no
labels. The primary verifier considers only the current winner's 62 proposals and
vetoes proposals below a fitted strong-continuity cutoff.

For every held-out recording, the other ten choose the cutoff that maximizes full-event
F1 while retaining at least 90% of their control true positives. The held-out video is
then evaluated once. A one-value forward research artifact is fit on all opened
development after LOO evaluation; its threshold is not reported as independent test
performance.

Predeclared sensitivities include the V4 court-band margin, the mean of recording-
normalized V4/player margins, and quality abstention. The verifier is also evaluated
alone over all 352 gaps and as an add-only union with the control.

## Results

### Primary veto

| Decoder | Proposals | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current local peak + soft count | 62 | 25 | 37 | 25 | 40.32% | 50.00% | 44.64% |
| LOO player-continuity veto | 57 | 25 | 32 | 25 | 43.86% | 50.00% | 46.73% |

The cutoff is stable: 10/11 folds select `-0.836035` and one selects `-0.864245`.
False proposals are removed from five recordings: one each in `161923155`,
`164327879`, and `190429172`, and two in `180646590`. No matched proposal is removed
from any video.

The final all-development forward cutoff is `-0.8360349704653693`. Its same-data fit
retains all 25 TP and removes six FP, but that resubstitution count is provenance, not
the selection result. Future evaluation must use the frozen cutoff without refitting.

### Why a hard opposite gate fails

| Rule | Proposals | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retain only when raw `sameCost > swapCost` | 16 | 7 | 9 | 43 | 43.75% | 14.00% | 21.21% |

Most true switches do not make the noisy motion-component swap cost absolutely cheaper.
The safe information is only the extreme continuity tail. This agrees with Loop 1:
the V5 coordinate shifts with state but is poorly calibrated around zero.

### Sensitivities and candidate use

| LOO variant/mode | Proposals | TP | FP | FN | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Player veto | 57 | 25 | 32 | 25 | 46.73% |
| V4 veto | 53 | 24 | 29 | 26 | 46.60% |
| Player + quality veto | 53 | 22 | 31 | 28 | 42.72% |
| Player verifier alone | 92 | 22 | 70 | 28 | 30.99% |
| Player add-only union | 118 | 29 | 89 | 21 | 34.52% |

The V4 veto removes more false positives but also one true positive. Player/V4
agreement and the support/separation/alignment quality abstention variants regress.
The current quality values therefore do not identify when continuity is trustworthy.

Verifier-alone and add-only results show that local swap evidence is not a candidate
generator: add-only recovers four events but adds 52 false proposals over the control.
Candidate expansion should instead come from the stable-state/rally-boundary path, with
continuity used only after proposal generation.

## Limitations

- All 11 recordings and markers are opened development evidence. LOO tests
  recording-to-recording transfer inside this scope but is not untouched confirmation.
- `playerSwapMargin` is already an input to the V5 control head. The new information is
  the separately constrained low-tail veto, not an independent visual source.
- Normalization currently uses the complete recording's candidate rows. That is valid
  for recorded-video editor inference but would need a causal calibration policy for
  live streaming.
- The verifier cannot recover the control's 25 missed events and has no UI/runtime
  implementation.

## Implementation and artifacts

- Verifier normalization and threshold selection:
  [`side_switch_continuity.py`](../../analysis/side_switch_continuity.py)
- Immutable trainer/evaluator:
  [`train-side-switch-continuity-verifier.py`](../../scripts/train-side-switch-continuity-verifier.py)
- Focused tests:
  [`test_side_switch_continuity.py`](../../analysis/tests/test_side_switch_continuity.py)

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-continuity-verifier-v1/model.json` | `9eb71d0403f638d4674acce20b7f42eaf0e0cdd9048dbdec790f017ec82534ce` |
| `reports/side-switch/side-switch-continuity-verifier-v1-evaluation.json` | `19f1379971db2fbd80cdaf09c8f16d784607ed4892579fb9156a742d6ac0a11d` |

Both paths are relative to
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/`. The artifacts bind the V5 feature
matrix, exhaustive full-marker audit, and checked-in current-winner manifest by SHA-256.

Run from the repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-continuity-verifier.py
PYTHONPATH=. .venv/bin/python -m unittest analysis.tests.test_side_switch_continuity
```
