# Completed-label training result — 2026-08-09

## Outcome

The nine completed pilot clips materially improved the validation-selected CPU baseline, but the result is not ready for unattended cutting. Tied within-recording percentile normalization reduced camera/court scale shift, and a more event-focused decoder search reduced validation over-segmentation. Validation event F1 increased from **0.320 to 0.706**. The one held-out indoor recording still scored **0 event F1**, so the model remains a review-assist baseline.

No held-out result was used to select features, weights, thresholds, or decoder durations.

## Frozen data

The user-confirmed drafts were preserved unchanged under:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/labels/pilot`

A completed snapshot was frozen under:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/completed/pilot-v1`

All nine documents passed video hash, duration, interval, ROI, capture, and annotation-policy validation. The snapshot contains 30 rallies across 810 seconds of footage. Every document records continuous-video review. Drafts without an annotator name were normalized to `Vini`, matching the named first draft, after the user confirmed the batch was complete.

The immutable training manifest is:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/pilot-gold-v1.json`

The split is source-group disjoint:

- Train: six clips from three source matches; two each from beach, grass, and indoor.
- Validation: two grass clips from a fourth source match.
- Test: one indoor clip from a fifth source match.

The test clip remains weak evidence by itself. It is only 90 seconds, contains three annotated rallies, uses four players per team, and its proxy is below the preferred 720p floor.

## Model change

The feature extractor still samples a manually selected court rectangle at 4 fps and uses appearance, frame-difference, regional-motion, and optical-flow channels with centered temporal context. Two changes were selected using training and validation only:

1. Each feature channel is mapped to tied percentile ranks within its recording before temporal context is assembled. Equal feature values receive the same average rank; a constant channel maps to 0.5. This reduces raw motion-scale differences among courts and cameras.
2. Decoder selection now searches smoothing, hysteresis thresholds, minimum rally duration, and gap bridging. Its objective prioritizes interval event F1, then temporal IoU and live-time recall. The selected decoder uses 0.5 seconds smoothing, 0.70/0.65 enter/exit thresholds, a 3-second minimum duration, and 0.5-second gap bridging.

Saved models created before this change continue to load with raw sequence values. The model artifact records which normalization it requires.

## Results

| Split | Model | True / predicted / matched rallies | Event F1 | Time IoU | Live precision | Live recall |
|---|---|---:|---:|---:|---:|---:|
| Validation | `pilot-baseline-v0` | 8 / 17 / 4 | 0.320 | 0.395 | 0.415 | 0.892 |
| Validation | `pilot-percentile-v1` | 8 / 9 / 6 | **0.706** | **0.523** | **0.680** | 0.693 |
| Held-out test | `pilot-baseline-v0` | 3 / 1 / 0 | 0.000 | **0.617** | 0.617 | **1.000** |
| Held-out test | `pilot-percentile-v1` | 3 / 5 / 0 | 0.000 | 0.379 | **0.851** | 0.407 |

On validation, the new normalization and decoder substantially reduce false rally fragments while retaining six of eight matchable intervals. On test, the baseline merged essentially the entire clip into one rally. The new model rejects more dead time, but recognizes only active portions of the long rallies and splits them into five intervals. None overlap a complete ground-truth interval by the required 0.5 IoU.

The candidate artifact and immutable reports are:

- Model: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/pilot-percentile-v1`
- Validation: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/pilot-percentile-v1-validation.json`
- Held-out test: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/pilot-percentile-v1-test.json`

## Decision

Use `pilot-percentile-v1` for local inference experiments and human review. Do not use it for automatic exports or represent its confidence as a calibrated probability. Keep the held-out report as the acceptance baseline; further choices must not be tuned to that single clip.

The most useful next model change is the learned visual sequence classifier already identified in the research decision: a pretrained image backbone with short and long temporal branches, followed by the existing interval decoder and evaluation contract. The current labels can train and regression-test that implementation, but 13.5 minutes is not enough to establish generalization.

For the next data increment, prioritize source-match-disjoint continuous clips, especially long indoor rallies and relatively static play that motion features miss. Add enough new source groups to reserve multiple matches per environment for testing. The annotation contract remains only rally start, rally end, ignored/unresolvable time, continuous-review confirmation, capture geometry, and game context; individual touches, ball tracks, scores, and player identities are still unnecessary for the cutter.
