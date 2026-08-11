# Full-video model inference and crop-padding sensitivity — 2026-08-11

## Outcome

All three saved model versions were run over all nine continuously labeled full videos. The
result is **27 validated, immutable `analysis.json` outputs** under
`/mnt/freenas/volleycut/analyses/model-<version>--<recording-id>/`. Every version has exactly
nine outputs; all files have matching IDs, valid model hashes, unique ordered rallies, and
timestamps inside the corresponding video duration. No partial model-output directories remain.

The main review UI discovers these results automatically and places them on the same video
clock as the human gold labels, blind Sol candidates, and both no-model heuristic versions.

## Padding evaluation method

The full-video predictions were compared directly with the frozen `full-gold-v1` labels.
For each predicted core interval, the evaluator added the same amount before and after,
clipped it to the video bounds, and merged crops that touched or overlapped. The model
predictions themselves were not changed. Settings were 0, 1, 2, and 3 seconds on each side.

The complete machine-readable report, including every recording, split, analysis SHA-256,
and model-artifact SHA-256, is:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/all-models-padding-v1.json`

The held-out column below is the one-video indoor test split. Validation is tuning evidence;
the six training videos and all-nine aggregate are descriptive only.

## Full model: held-out crop tradeoff

| Padding on each side | Live-play recall | Live-time precision | Time IoU | Event F1 | Missed live time | Dead time retained | Video retained |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0s | 92.68% | 70.83% | 67.07% | 62.92% | 23.60s | 123.08s | 38.15% |
| 1s | **96.77%** | 59.78% | 58.62% | 58.43% | **10.40s** | 209.88s | 47.20% |
| 2s | 97.34% | 51.19% | 50.48% | 47.50% | 8.57s | 299.30s | 55.45% |
| 3s | 97.34% | 45.35% | 44.79% | 42.67% | 8.57s | 378.28s | 62.59% |

One second on each side is the practical knee. Relative to unpadded core intervals, it
recovers 13.20 seconds of missed play and adds 4.09 percentage points of live-play recall,
at the cost of 86.80 seconds more dead time and 100 seconds more retained footage. Moving
from 1s to 2s recovers only another 1.84 seconds, while retaining another 91.25 seconds.
Moving from 2s to 3s recovers no additional live time on this test video.

Across all nine videos, 1s padding moves live-play recall from 86.72% to 93.48% and reduces
missed live time by 176.68 seconds, while increasing retained footage from 38.50% to 47.79%.
This all-video result includes training and tuning footage and must not be presented as a
generalization estimate.

## Model-version comparison on the held-out video

| Model | Padding | Event F1 | Live-play recall | Live-time precision | Video retained |
|---|---:|---:|---:|---:|---:|
| `full-percentile-v1` | 0s | **62.92%** | 92.68% | **70.83%** | 38.15% |
| `full-percentile-v1` | 1s | **58.43%** | **96.77%** | 59.78% | 47.20% |
| `pilot-percentile-v1` | 0s | 52.38% | 71.52% | 69.47% | 30.02% |
| `pilot-percentile-v1` | 1s | 52.50% | 79.67% | 61.32% | 37.88% |
| `pilot-baseline-v0` | 0s | 3.13% | 99.89% | 34.59% | 84.20% |
| `pilot-baseline-v0` | 1s | 3.13% | 100.00% | 32.90% | 88.63% |

The raw pilot baseline's near-total recall is not useful performance: it retains most of
the video and collapses many points into long intervals. The full-corpus percentile model
is the strongest of the three and remains the default review-assist model.

## Decision

Use **1 second before and 1 second after** as the initial default crop padding for model
review/export previews. Keep 2 seconds available as an explicit conservative option. The
observed 3-second setting has no held-out coverage gain over 2 seconds and retains materially
more dead time.

Padding metrics answer a different question from core detector metrics. Event F1 and time
IoU can fall after padding because longer crops have lower interval IoU and neighboring crops
may merge, even while practical live-play coverage improves. Therefore use live-play recall,
missed seconds, dead seconds retained, and retained-video rate to choose crop padding; use the
unpadded event F1, boundary errors, and time IoU to compare detector quality.

This is not a final external benchmark. The test split contains only one indoor recording,
and its 90-second pilot excerpt was previously inspected. A future acceptance result needs
new source groups with untouched indoor, grass, and beach test recordings.
