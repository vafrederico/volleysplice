# Side-switch appearance experiment — 2026-08-19

## Purpose

This note records the current design decisions and the first experiments for detecting
team court-side switches in beach and grass footage. It is a research hypothesis and
diagnostic report, not a production decision.

The existing serve detector supplies rally/serve events. The proposed downstream state is:

```text
serve event -> physical serving side (near/far) -> side-map flip events -> score tracker
```

The serving-side head should predict physical court side, not anonymous Team A/Team B.
The score tracker can maintain an arbitrary initial Team A/Team B mapping and flip that
mapping when a side switch is accepted.

## Current decisions

### Side-switch target

For the first experiment, detect only:

```text
switch / no-switch / unknown
```

The detector does not need to identify which team is which. It only needs evidence that the
two teams exchanged court halves. This is compatible with a side-blind appearance signal:
the detector can compare the player-appearance distribution before and after a dead-time
window without explicitly assigning a color cluster to the near or far team.

### Appearance representation

Start with interpretable, CPU-friendly player color features rather than a learned embedding:

- HSV/Lab color histograms from torso-focused person crops;
- dominant-color and chroma proportions;
- equal-player-weighted palette summaries;
- pixel-area/box-size-weighted color-mass summaries; and
- detector confidence, crop size, blur, and visibility weights.

The area-weighted representation is intentional. With a fixed end-line camera, a team moving
from the near half to the far half may contribute fewer pixels, while the other team contributes
more. That perspective-dependent color-mass change may be a useful switch signal even when the
unweighted set of team colors is unchanged.

Do not use an unmasked full-frame embedding as the primary signal. Court, sand, grass, shadows,
and exposure changes can dominate it. A full-frame color change is retained as a scene-change
control only.

### Temporal sampling

Use labeled side-switch markers and rally boundaries to define dead-time windows. Sample multiple
stable frames before and after each candidate transition rather than relying on one frame. Stable
frames should be preferred over frames with large motion, blur, or a camera shift.

The first implementation uses OpenCV HOG person proposals because they require no new model
artifact. These proposals are diagnostic only; they are not treated as ground-truth people
detections. A later version may use a lightweight person detector or pose model.

### Evidence combination

Appearance change is one channel, not the entire decision. The next feature families are:

- player color-mass change;
- player box-size and count changes;
- approximate net-crossing and synchronized movement;
- post-transition formation stability;
- camera-motion rejection; and
- timing constraints that restrict switches to dead time between rallies.

A switch should be accepted conservatively. An `unknown` result is preferable to a false mapping
flip because one false flip can corrupt every later score update.

## What the current labels support

The frozen `completed/full-v1` corpus currently contains:

- 9 recordings: 2 beach, 4 grass, and 3 indoor;
- 345 labeled rallies;
- 26 side-switch markers, all in beach/grass footage (8 beach and 18 grass); and
- complete continuous-review labels with rally start/end boundaries.

The existing labels do not currently contain:

- annotated player boxes or footpoints;
- populated player tracklets or team assignments;
- completed court corner, net, or service-zone geometry;
- a per-rally/per-serve physical serving-side field;
- an initial mapping from physical side to anonymous team; or
- an initial score.

Therefore the current labels are sufficient for an event-level heuristic study. They are not
sufficient for supervised person-detector evaluation, learned team-identity clustering, or a
physical serving-side classifier.

The current `sideSwitches` markers can provide positive transition examples. Unmarked inter-rally
dead-time windows in continuously reviewed beach/grass recordings can provide no-switch controls,
with care around short or otherwise ambiguous gaps.

## Variants that can run now

The first runner evaluates these variants without changing the labels:

| Variant | Inputs | Purpose |
| --- | --- | --- |
| `full-frame-control` | Broad-frame HSV/Lab histogram distance | Measures ordinary scene/exposure change |
| `player-palette-equal` | HOG person torso palettes, equal player weight | Tests whether the visible color set changes |
| `player-palette-area` | Same palettes weighted by crop/box area | Tests perspective-dependent color mass |
| `player-palette-area-plus-geometry` | Area palette plus box size/count and simple frame-change controls | Tests a slightly richer side-blind switch score |

These are diagnostic distances and feature rows, not tuned production thresholds. The report
must expose per-event values and detection coverage so a zero-detection result is not mistaken
for a negative appearance finding.

## Initial diagnostic run

The first run used the six beach/grass recordings from `completed/full-v1`, two frames per
side of each transition, an eight-second flank, a 0.75-second edge margin, and every eligible
unmarked inter-rally gap of at least eight seconds as a control. The command was:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-side-switch-appearance.py \
  --labels-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/completed/full-v1 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/appearance-diagnostic-v1.json \
  --samples-per-side 2 \
  --minimum-gap-seconds 8
```

It analyzed 225 transition windows: 26 labeled switches and 199 no-switch controls. One
window was too short for the requested before/after samples. HOG-person palette features were
usable for 186/225 windows (82.7% overall), but coverage was only 35/68 windows (51.5%) in
beach and 151/157 (96.2%) in grass. The full-frame control had 99.6% coverage.

Pooled raw ranking diagnostics were:

| Variant | Usable events | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: |
| `full-frame-control` | 224 | 0.559 | 0.193 |
| `player-palette-equal` | 186 | 0.629 | 0.162 |
| `player-palette-area` | 186 | 0.658 | 0.168 |
| `player-palette-area-plus-geometry` | 186 | 0.618 | 0.142 |

The area-weighted player palette is the best pooled raw feature, but the effect is weak and
heterogeneous. Grass-only area-weighted AUC was 0.653. Beach-only equal-weighted AUC was 0.909,
but only two positive switch windows had usable HOG detections, so that number is not reliable.
The fixed area/count/height bundle was worse than area palette alone, so its weights should not
be carried forward as a model decision.

The first conclusion is feasibility, not promotion: there is some evidence that player color
mass carries switch information, particularly in grass, but the current person proposal path
misses too many beach players and the switch count is too small for a generalization claim.
The next no-label experiment should improve person proposals or compare a second detector before
adding a learned classifier. The report remains an external diagnostic artifact; no production
threshold was selected.

## All completed recordings run

The same label-only protocol was then run across all nine completed recordings, adding the three
indoor recordings as no-switch controls. This produced 335 windows: 26 labeled switches, 309
unmarked controls, and one insufficient window. The command was:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-side-switch-appearance.py \
  --labels-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/completed/full-v1 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/appearance-diagnostic-all-v1.json \
  --environments beach grass indoor \
  --samples-per-side 2 \
  --minimum-gap-seconds 8
```

The expanded pooled diagnostics were:

| Variant | Usable events | Coverage | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: |
| `full-frame-control` | 334 | 99.7% | 0.582 | 0.168 |
| `player-palette-equal` | 296 | 88.4% | 0.736 | 0.149 |
| `player-palette-area` | 296 | 88.4% | 0.744 | 0.149 |
| `player-palette-area-plus-geometry` | 296 | 88.4% | 0.714 | 0.126 |

The indoor set has no positive switch labels, so it supplies false-positive context rather than
an AUC. It contributed 110 controls. Beach HOG coverage remains the main weakness: only 35/68
beach windows had usable player palettes, while grass coverage was 151/157 and indoor coverage
was complete. The area-weighted palette remains the best pooled raw discriminator, but average
precision is still low because the control set is much larger than the switch set. These results
support a review and labeling pass, not a production side-map flip threshold.

The development app now exposes this report at `/side-switch-review`. It provides recording and
environment filters, a scrollable event queue, ranged video playback through the prepared proxy,
full-recording and transition-window timelines, per-event feature values and pooled metrics, and
`switch`/`no-switch`/`unclear` review decisions. Decisions are automatically debounced to the
NAS-backed `appearance-review-decisions-v1.json` file through a development API, with a manual
retry button; they remain separate from the research report and completed label documents.

The first serving-side variants built from the existing rally starts and weak note cues are
documented in [`serving-side-existing-label-variants-2026-08-19.md`](serving-side-existing-label-variants-2026-08-19.md).

## Label changes needed for later versions

### Minimal additions for a stronger switch detector

The current event markers are enough to begin. A small follow-up annotation pass should add:

- confidence or observability on each side-switch marker;
- a coarse switch interval when the exact crossing frame is unclear;
- a few no-switch hard cases: huddle, ball retrieval, timeout, celebration, and long setup;
- court corners and net anchors for a subset of recordings; and
- sparse person boxes/footpoints around switch and no-switch windows.

### Additions for appearance/team association

If the diagnostic color signal is promising, annotate a subset of person observations with:

- anonymous team (`team-a`/`team-b`);
- physical court side (`near`/`far`);
- torso-visible/occluded state; and
- a short within-window track ID.

This would test whether the color clusters actually represent teams rather than lighting,
background, or individual-player differences.

### Additions for score tracking

The score tracker needs product initialization and recovery fields, either manually or from
separate OCR:

- initial near/far team mapping;
- initial score;
- scoring rule and format;
- replay/service-fault indicators when visible; and
- optional scoreboard anchors.

The later serve-side model also needs a per-serve label:

```json
{
  "servingCourtSide": "near|far|unknown",
  "servingConfidence": 0.0,
  "serverTrackId": "optional"
}
```

## Evaluation protocol

For the diagnostic experiment, report:

- positive/negative sample counts;
- person-detection coverage and average usable crops;
- feature distributions for switch versus no-switch windows;
- ROC-AUC/average precision when both classes support it;
- threshold sweeps with precision, recall, and false switches per recording;
- results separately for beach and grass; and
- the per-recording event table.

Do not select a production threshold on the repeatedly inspected indoor footage. The current
switch set is small, so results should be described as feasibility evidence. Future model
selection should hold out recordings/source groups and include hard negative categories.

## Next steps

1. Run the current-label appearance variants and inspect coverage.
2. Keep the best diagnostic feature family only if it separates switch markers from comparable
   no-switch dead time without excessive per-recording false flips.
3. Add geometry and sparse player labels around a stratified switch/no-switch pilot.
4. Add the physical serving-side labels separately; do not use switch markers as a substitute.
5. Build the conservative side-map and score state machine after the two event channels have
   measurable reliability.
