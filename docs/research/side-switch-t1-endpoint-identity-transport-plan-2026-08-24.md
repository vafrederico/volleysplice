# Side-switch T1 endpoint-identity transport — 2026-08-24

## Decision and data boundary

T1 follows the rejected M1 dense-gap-motion experiment. It asks whether detected
player appearance moves from near-before to far-after and far-before to near-after,
rather than asking whether arbitrary foreground pixels move during the dead gap.

The engineering pass is boundary-only and label-independent:

- source universe: the immutable 704-row full-union artifact;
- eligible rows: all 624 adjacent-rally boundaries;
- ineligible rows: all 80 internal dead-state peaks, retained without T1 inputs;
- endpoint cache: 635 unique `(recording, start, end)` live windows;
- frame requests: exactly 1,905, three per unique endpoint window;
- detector calls: exactly 7,620 four-tile calls;
- candidate generation, existing feature values, row IDs, and row order: unchanged;
- current 50 side-switch markers, audit artifact, feedback labels, and all row labels:
  prohibited extraction inputs; and
- T1 may produce observability and feature-novelty diagnostics, but no model-selection
  result on the repeatedly opened 50-marker scope.

No genuinely new side-switch gold exists in the workspace. The 26 original curated
markers, 15 reviewed-draft challenge markers, and 50 PXL feedback markers have all
been used or inspected in prior side-switch work. Consequently, the T0/T1 outcome
comparison remains gated on a newly collected recording-held scope. This is a data
gate, not permission to reuse the opened labels as an implicit validation set.

## Frozen endpoint sampling and detector

For every unique before/after comparison window `[w0,w1]`, sample frames at 15%, 50%,
and 85% of its duration. Crop the frozen manifest ROI but preserve its native cropped
resolution for player localization. Use the already installed pinned detector:

- model: `opencv-zoo-mediapipe-person-int8bq-2023mar`;
- ONNX SHA-256: `c5ed8c00c028b98e5d2c55b920a6e975af6c4cd538cfeea7c054f4fbbd8b9075`;
- input: four overlapping ownership tiles per frame;
- score threshold: 0.20;
- cross-tile NMS threshold: 0.30; and
- maximum detections: six per frame before court/side filtering.

Map detector hip coordinates through the existing piecewise court normalization using
the frozen per-recording net ratio. Retain detections only inside court x `[0.06,0.94]`
and canonical y `[0.18,0.98]`. Assign near/far by the existing sigmoid around canonical
y `0.56`, then retain at most two detections per side per frame, ordered by detector
confidence times square-root torso-box area.

## Frozen appearance descriptor and endpoint tracklets

Every retained torso crop becomes a 64-value non-negative descriptor:

1. a 52-value HSV descriptor: a `12 x 4` hue/saturation histogram plus four value
   bins, using an elliptical torso mask weighted by `0.75 + 0.25 * saturation`;
2. twelve Lab marginals: four bins independently for L, a, and b under the same mask;
3. normalize the HSV block and each Lab marginal to unit mass; and
4. concatenate and normalize the complete vector to unit mass.

Within each endpoint and court side, link detections across the three ordered frames.
The Hungarian link cost is:

```text
0.75 * Hellinger(descriptor) +
0.25 * min(spatial_distance / 0.50, 1)
```

Accept links at cost at most `0.55`. Unlinked detections start new tracklets. Pool a
tracklet descriptor by detector-confidence weighting; its reliability weight is mean
confidence times `(0.5 + 0.5 * observed_frame_fraction)`. Retain at most three
tracklets per side by reliability. These are anonymous appearance tracklets, not
supervised identities or team labels.

## Frozen transport reductions

For two side tracklet sets, calculate the Hellinger cost matrix and a Hungarian
assignment. Unmatched reliability mass has fixed cost `0.70`. Report:

- assignment cost: matched reliability-weighted cost plus unmatched penalty, divided
  by the larger endpoint reliability mass;
- matched identity mass: matched geometric-mean reliability times `(1 - cost)`,
  divided by the larger endpoint reliability mass; and
- coverage: the smaller matched-reliability fraction of the two endpoint sets.

Empty-side input produces cost one, matched mass zero, and coverage zero.

The shared extraction batch emits six values:

| Feature | Frozen definition | First-head use |
| --- | --- | --- |
| `appearanceTransportSwapMargin` | Mean same-side cost minus mean cross-side cost; positive means swapped endpoints fit better. | T1 core |
| `bidirectionalMatchedIdentityMinimum` | Minimum matched identity mass for near-before→far-after and far-before→near-after. | T1 core |
| `transportCoverageMinimum` | Minimum coverage of those two cross-side assignments. | T1 core |
| `sameSideIdentityRetentionPenalty` | Mean matched identity mass for the two same-side assignments. | Diagnostic/reliability arm only |
| `endpointTeamSeparationMinimum` | Minimum near-versus-far assignment cost within the before and after endpoints. | Diagnostic/reliability arm only |
| `sceneContinuityConfidence` | Global before/after matched identity mass after pooling both sides. | Diagnostic/reliability arm only |

The first future learned profile appends only the three T1-core values to the matched
34-input boundary control. The three reliability values must not enter that head or be
used to rescue it. If T1 passes on new data, leave-one-core-feature-out pruning comes
before a separately preregistered reliability arm.

## Immutable artifact and extraction gates

Inputs are pinned to:

- full-union features SHA-256 `9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551`;
- Visual Summary V2 SHA-256 `6ce23b43018d04045ba783510ad86ef3c6d8767fdc435c7684481fca89ba4871`;
- manifest SHA-256 `c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24`;
- detector model, metadata, and license hashes; and
- extractor and transport-module hashes embedded at write time.

The artifact must prove:

- 704 exact source IDs/order and exact preservation of every old feature;
- 624 eligible and 80 explicitly ineligible rows;
- 635 endpoint summaries, 1,905 frame requests, and 7,620 tile calls;
- no frame or detector errors;
- finite six-value bundles on every eligible row;
- labels not loaded or read;
- wall time no more than 60 minutes and peak RSS no more than 768 MiB; and
- detector observability reported per recording, endpoint, and court side.

Output path:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t1-endpoint-identity-transport-features-v1.json`.

## Label-free acceptance and stop rules

This pass succeeds as an engineering artifact only if:

- at least 90% of endpoint windows contain at least one retained player tracklet;
- at least 60% contain at least one retained tracklet on both court sides;
- at least 50% of boundary rows have nonzero `transportCoverageMinimum`;
- no T1 core feature is constant;
- no T1 core feature has absolute Spearman correlation above 0.98 with another core
  feature or an existing 34-input feature; and
- all extraction/parity/resource gates pass.

Failing observability stops the representation before any label evaluation. Passing
it makes the artifact validation-ready but does not promote a model.

## Future evaluation contract

Do not run this gate until new labels exist. Freeze at least ten new recordings with
roughly 40 switches, grouped by source recording. Then:

1. T0 exactly reproduces the 624-row boundary-control machinery on the new scope's
   analogous adjacent-boundary universe.
2. T1 appends only the three core features.
3. Fit preprocessing, hard negatives, classifier, inner threshold, and decoder inside
   each outer recording fold.
4. Require at least +2.0 pp pooled outer-held ±4-second F1, no more than 2.0 pp loss
   of precision or recall, no more than 1.0 pp strict-F1 loss, recovery of at least
   one covered control miss, reduction of a preregistered low-transport false slice,
   and the existing recording-robustness rule.
5. Keep candidate expansion separate; T1 cannot recover a no-candidate marker.

No browser or Android port begins from the label-free engineering artifact.

## Execution ledger

### Preregistration — frozen 2026-08-24

No T1 feature artifact or T1 label result existed when this contract was written.

### Extraction — pending

### New-data evaluation — waiting for eligible labels

