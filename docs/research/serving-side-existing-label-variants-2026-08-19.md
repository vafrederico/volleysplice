# Serving-side variants from existing labels — 2026-08-19

## Status and purpose

This is the first serving-side experiment after the side-switch appearance study. It asks how
far the current `completed/full-v1` labels can take us without adding a new structured label.
The output is an interpretable evidence report, not a promoted serving-side model.

The intended downstream state remains:

```text
existing serve/rally event -> physical serving side (near/far) -> team/side map -> score tracker
```

The existing rally start is used as the serve-contact anchor. A serving side is expressed in
camera-space terms: `near` is the lower/foreground half of the recording ROI and `far` is the
upper/background half. This is a fixed-view proxy, not a court-geometry solution.

## What the current labels provide

The frozen `completed/full-v1` corpus contains 9 recordings and 345 labeled rallies. It has no
structured `servingCourtSide`, no populated player tracklets, and no completed court geometry in
the reviewed files. Rally notes do, however, contain occasional near/far serving descriptions.

For this experiment, only strict phrases tied directly to serving were promoted to weak targets:

- near-side serve/server/contact/toss/action;
- near baseline or camera-side baseline;
- foreground serve/server/toss/contact; and
- far-side serve/server/contact/toss/onset, far baseline, distant serve/contact/onset, or an
  explicit far-side serving phrase.

Notes mentioning a far player, a foreground ball, an unresolved view, or a conflicting
far-side/outside view remain unknown. The weak target parser is in
[`analysis/serving_side.py`](../../analysis/serving_side.py); it never changes the gold label
files.

The strict pass produced 120 weakly targetable rallies: 69 near and 51 far. Their note-strength
breakdown was 53 strong, 19 medium, and 48 weak. The remaining 225 rallies were retained in the
report as unlabeled rows, but excluded from target metrics.

| Recording | Environment | Rallies | Weak targets | Near | Far |
| --- | --- | ---: | ---: | ---: | ---: |
| `beach-source-02` | beach | 39 | 3 | 0 | 3 |
| `beach-source-01` | beach | 32 | 10 | 4 | 6 |
| `grass-source-04` | grass | 36 | 32 | 11 | 21 |
| `grass-source-01` | grass | 45 | 4 | 0 | 4 |
| `grass-source-09` | grass | 40 | 6 | 0 | 6 |
| `grass-source-10` | grass | 40 | 21 | 12 | 9 |
| `indoor-source-01` | indoor | 38 | 12 | 12 | 0 |
| `indoor-source-07` | indoor | 36 | 28 | 28 | 0 |
| `indoor-source-05` | indoor | 39 | 4 | 2 | 2 |
| **Total** |  | **345** | **120** | **69** | **51** |

This is useful weak supervision for feature discovery, not a reliable gold target. The indoor
rows are especially one-sided and cannot establish near/far discrimination.

## Variants that can run now

The implementation is [`scripts/evaluate-serving-side-existing-labels.py`](../../scripts/evaluate-serving-side-existing-labels.py).
It samples three pre-serve frames at offsets `-1.25`, `-0.75`, and `-0.35` seconds, plus three
immediate action frames at `+0.05`, `+0.30`, and `+0.55` seconds relative to `rallies[].start`.
The recorded ROI is applied before the vertical split. Each score is signed near-minus-far;
positive means the near half has more of that evidence.

| Variant | Evidence | Why it is useful |
| --- | --- | --- |
| `pixelMotion` | Mean grayscale change over each full vertical half | Fast receiver/server activity baseline |
| `paletteChange` | Hellinger distance between HSV palette vectors before and after | Tests the color-presence idea without a learned embedding |
| `hogAreaChange` | Absolute OpenCV HOG person-proposal area change by half | Tests basic person-box evidence; proposals are diagnostic only |
| `motionPalette` | 0.65 pixel motion + 0.35 palette change | Reduces dependence on one raw cue |
| `motionPaletteHog` | 0.55 motion + 0.25 palette + 0.20 HOG area | Fixed three-channel bundle |
| `baselineMotion` | Pixel motion only in the outer 35% baseline bands | Avoids central receiver motion dominating the signal |
| `baselinePalette` | Palette change in the outer baseline bands | Color-only baseline-zone control |
| `baselineHogArea` | HOG area change in the outer baseline bands | Sparse server-proposal control |
| `baselineMotionPalette` | 0.65 baseline motion + 0.35 baseline palette | Baseline-focused fixed bundle |

The HOG detector is the existing default OpenCV people detector, resized to a maximum width of
960 pixels for this experiment. It does not establish person identity, team identity, or a
server track. A fixed decision margin of `0.05` reports both raw directional accuracy and the
coverage/accuracy after abstaining on low-margin rows. No threshold was tuned for production.

## All-video run

The complete nine-recording report was generated at:

```text
/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-existing-label-variants-all-v1.json
```

Command:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-serving-side-existing-labels.py \
  --labels-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/completed/full-v1 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-existing-label-variants-all-v1.json \
  --environments beach grass indoor
```

All 345 rallies decoded successfully. The report was created at `2026-08-20T00:52:08Z`.
Pooled metrics on the 120 weakly targetable rows were:

| Variant | Directional accuracy | Balanced accuracy | Decision coverage | Decision accuracy |
| --- | ---: | ---: | ---: | ---: |
| `pixelMotion` | 0.767 | 0.766 | 0.908 | 0.817 |
| `paletteChange` | 0.608 | 0.570 | 0.817 | 0.622 |
| `hogAreaChange` | 0.575 | 0.500 | 0.058 | 0.857 |
| `motionPalette` | 0.825 | 0.812 | 0.842 | 0.851 |
| `motionPaletteHog` | 0.833 | 0.822 | 0.800 | 0.844 |
| `baselineMotion` | 0.775 | 0.758 | 0.917 | 0.800 |
| `baselinePalette` | 0.725 | 0.725 | 0.925 | 0.757 |
| `baselineHogArea` | 0.575 | 0.500 | 0.192 | 0.870 |
| `baselineMotionPalette` | 0.817 | 0.810 | 0.942 | 0.823 |

These pooled numbers are not a model-selection result. Indoor has 44 weak targets, 42 of which
are near, and the test recording contributes only four weak targets. The environment breakdown
is more informative:

- beach has only 13 weak targets; whole-half motion is poor at 0.308 directional accuracy, while
  baseline motion reaches 0.538;
- grass has 63 weak targets; `motionPaletteHog` reaches 0.857 directional accuracy and
  `baselinePalette` reaches 0.810; and
- indoor has 44 weak targets with a strongly one-sided near distribution, so its high raw
  accuracy is not evidence of general near/far discrimination.

On the existing validation split (53 weak targets), `motionPaletteHog` and
`baselineMotionPalette` both reach 0.830 directional accuracy; their balanced accuracies are
0.814 and 0.820 respectively. This is still a small, note-derived, recording-level result and
should not be treated as a protected model-selection decision. The four weak targets in the
existing test split are insufficient for a meaningful conclusion.

## Interpretation

The first useful result is structural: a whole-half activity score is easily contaminated by
the receiving team, while an outer-baseline band is a better place to look for serving action.
The color-presence feature is more useful as a bundle component than as a standalone classifier.
HOG area change alone is sparse and mostly acts as a diagnostic gate; it should not be the first
serving-side model.

The current evidence is not enough to choose a production variant. The next implementation step
should be a reviewable serving-side report/UI and a small structured annotation pass, not a hidden
threshold or automatic score-map update.

## Label changes needed for a real serving-side model

The smallest high-value addition is a per-rally field beside `start`/`end`:

```json
{
  "servingCourtSide": "near|far|unknown",
  "servingConfidence": 0.0,
  "servingObservability": "visible|partially-visible|offscreen|ambiguous",
  "serverTrackId": "optional-anonymous-track-id"
}
```

For an initial pass, only label rallies where the serving side is visible or confidently
inferable. Keep `unknown` for offscreen/occluded serves rather than forcing a side. The existing
`playerTracklets` structure can carry `serverTrackId`, but it needs observations around the
serve window and a physical `courtSide`; current completed labels do not populate it.

For a specialist that can generalize across recordings, add these alongside a subset of serve
labels:

- normalized player boxes/footpoints at pre-serve, toss/contact, and first-receive frames;
- anonymous `team-a`/`team-b` plus physical `near`/`far` side for those observations;
- court corners, net anchors, and near/far service-zone anchors;
- whether the server is outside the ROI, occluded, or too small to inspect;
- serve fault/ace/immediate-dead-ball status, so a receiver reaction is not mistaken for server
  movement; and
- a recording-level camera/view orientation field if near/far semantics can change between
  recordings.

For score tracking, the product still needs an initial team-to-side mapping and initial score,
plus a recovery path when a serve-side decision is unknown. Side-switch markers alone can signal
a map flip but cannot initialize the map.

## Files and reproducibility

- `analysis/serving_side.py` — weak-note parser, ROI/side split, pixel/palette/HOG evidence helpers.
- `scripts/evaluate-serving-side-existing-labels.py` — all-video report runner and metrics.
- `analysis/tests/test_serving_side.py` — parser, signed-margin, pixel-motion, and occupancy tests.
- External report: `serving-side-existing-label-variants-all-v1.json` at the NAS path above.

The runner refuses to overwrite an existing report. It records label-file SHA-256 values, sampling
offsets, ROI/split assumptions, variant definitions, per-rally features, weak-target provenance,
and both labeled and unlabeled row counts.
