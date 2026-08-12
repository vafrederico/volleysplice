# Minimum ball-presence pilot — 2026-08-11

## Outcome

The minimum viable pilot is complete as an isolated, development-only
feasibility experiment. It does not change the production rally feature
extractor or the saved audiovisual model. The pilot now has:

- exact-frame, six-stratum sampling from all eight development recordings;
- a checksum-pinned CPU detector and immutable proposal artifacts;
- a native frame/box annotation schema and a local review UI;
- separate human, detector, and detector-blind Sol layers;
- a source-group-held-out detector evaluator; and
- eight confidence/availability-gated signals that could later be appended to
  the 4 fps rally model without percentile-ranking missing values; and
- a completed full-frame-versus-tiled detector comparison against the same
  human-verified frames.

All 2,160 human labels and all 2,160 detector-blind Sol labels are complete.
The generic detector nevertheless failed the predeclared source-group-held-out
quality gate, and the registered tiled mode made the important presence metrics
worse. No threshold was frozen, the detector was not promoted, and the ball
signals were intentionally not trained into the rally model. This is the
negative result the feasibility gate was designed to catch.

## Frozen Stage A data

- Root: `/mnt/freenas/volleycut/ball-presence-v1/round-01`
- Pilot index SHA-256:
  `c6461d34d1c3165ce93d501bd3bbc99fcbd1cc46d3075524800856478715ed3c`
- Source manifest SHA-256:
  `b662c5078d37858fd979a0c68759c0033152a3aefdce4886037351b4afb6b6fe`
- Coverage: 8 development recordings, 4 independent source groups, 48
  non-overlapping 3-second windows, and 2,160 lossless 960x540 frames at exact
  15 fps.
- Environments: 2 beach recordings, 4 grass recordings, and 2 indoor
  recordings.
- Strata per recording: serve window, two different mid-rally windows, rally
  end/dead transition, ordinary dead time, and timeout or highest-motion dead
  time.
- Protected exclusion: `indoor-source-05`, the sole test recording, was
  neither sampled nor sent through the detector.

Every image, task, source proxy, manifest, sampling rule, frame index, and
timestamp is hash- or derivation-checked. The task pack occupies about 1.5 GiB.

Human review is complete for all 2,160 frames. Of these, 193 were reviewed
without either proposal source, 1,965 were human-verified with Sol assistance,
and 2 were exposed to both Sol and detector proposals. Detector calibration and
quality evaluation therefore use 2,158 detector-independent frames. The 193
Sol-independent frames contain no human-positive ball frames, so they cannot
support a meaningful independent Sol recall or localization claim; the
assisted-inclusive Sol comparison is descriptive and circular by construction.

The browser had normalized some integral immutable JSON numbers while saving
reviews. The original review files remain untouched. A non-destructive repair
reconstructed their immutable sections from the SHA-pinned pristine tasks and
wrote validated copies to `round-01/reviews-canonical`. The canonicalization
receipt SHA-256 is
`739409f3481187126ed0eafd6f86e4924bbcdaaf7d8cd84cb0fca837dd4c5f0c`.

## Bootstrap detector and proposals

The bootstrap is OpenCV Zoo YOLOX-s FP32 executed by OpenCV DNN on CPU. No
PyTorch or Ultralytics dependency was added.

- Model ID: `opencv-zoo-yolox-s-2022nov`
- ONNX SHA-256:
  `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`
- Model metadata SHA-256:
  `7e851ebf95e6048f96f8f7acdf2c3510cc97dd87d4bf29eb22a8fba05c019fab`
- Model-specific Apache-2.0 license SHA-256:
  `0ec3668d3274bcf29e8a29e9576d5a2cd96fc78d3c5bec4387355a796e5d9088`
- Proposal index SHA-256:
  `a142cd0ca46b9a97abe25003dfa83ad25182a06b2180c1e78747f5a03e689cab`

At the deliberately permissive 0.01 extraction floor, the detector emitted
1,830 boxes and at least one candidate on 1,268 of 2,160 frames (58.70%). That
number is **not precision, recall, or prevalence**. The generic COCO detector
can confuse logos, background objects, and non-primary-court balls with the
target, while distant live balls can be only a few pixels wide. Thresholds are
therefore selected only after independent labels, never from these proposal
counts.

Per-task timings in the proposal ledger are not a controlled benchmark because
the full-frame and tiled jobs ran under different host loads. The recorded
means were 632 ms per full-frame pilot image and 1,395 ms per five-view tiled
pilot image. These non-controlled figures establish that dense CPU extraction
is expensive; they are not a fair tiled/full speed ratio or a throughput
acceptance result.

### Registered tiled-recall mode

The proposal runner also has an opt-in `full-plus-overlap-2x2-v1` mode for a
controlled small-ball recall experiment. The default remains the original
single `full-frame-v1` inference. On each 960×540 pilot image, the opt-in mode
runs the full image plus four 534×300 edge-anchored crops starting at x=0/426
and y=0/240. This is the deterministic integer realization of 20% overlap from
`ceil(axis / (2 - 0.20))`.

Tile boxes are projected and clipped into full-frame coordinates. A tile box is
retained only when its center belongs to that tile's half-open image quadrant;
the full-frame view remains unrestricted. Stable frame-global NMS then merges
all retained view proposals at IoU 0.5 and caps the result at 20 boxes. The
registered extraction floor remains 0.01. Proposal provenance stores the mode,
ordered pixel crops, overlap, ownership rule, global NMS settings, and five-view
count. Per-frame diagnostics separately store each forward-pass duration, their
sum, total detector wall time, and forward-pass count, so the roughly 5× compute
tradeoff cannot be mistaken for the original detector timing.

Use a new non-overwriting suggestion directory when testing the mode:

```bash
npm run detect:ball-pilot -- infer-task \
  --task /path/to/round-01/tasks/RECORDING.ball-presence.json \
  --model /path/to/models/opencv-zoo-yolox-s-2022nov \
  --detector-mode full-plus-overlap-2x2-v1 \
  --output /path/to/round-01/suggestions-tiled/RECORDING.ball-presence.json
```

The controlled development run is now complete. Its proposal index SHA-256 is
`48155d547484637286b75541c7e602a52713500944dc643bb6dc7cfbcb93255f`.
It was evaluated against exactly the same detector-independent human truth as
the full-frame bootstrap; no test recording was opened.

## Annotation and comparison contract

The positive target is every real volleyball associated with the primary
camera court, including a held, retrieved, or grounded ball during dead time.
Adjacent-court balls are real objects with role `other-court`; broadcast marks,
logos, and overlays are negatives. Each reviewed frame records one of:

- `localizable`, with exactly one primary-court box;
- `fully_occluded`;
- `out_of_frame`; or
- `indeterminate`.

Additional volleyball boxes may be marked `other-court` or `unknown`, with
clear, motion-blurred, or partly occluded visibility.

Human reviews, detector proposals, and Sol reviews are different files with
different provenance. Sol review copies can only be prepared from an original
task whose detector suggestions are empty, and their source SHA is retained.
Human review supports blind and Sol-assisted modes. A proposal layer can be
shown after the current human decision without contaminating that saved
decision. Revealing Sol or detector output before a decision irreversibly marks
that frame `shown_before_label_finalized` and records the exact proposal source.
Assisted frames remain in inclusive human-verified workflow metrics;
detector-exposed frames are excluded only from detector threshold selection,
and Sol-exposed frames are excluded only from the separate Sol-independent
quality subset. Completed inputs without an exposure audit are rejected. A
post-decision reveal is bound to the saved human-annotation hash, so changing
that decision afterward also marks the frame assisted.

The local `/label/ball` route supports frame playback/stepping, zoom, SVG box
editing, roles and visibility, copy-forward, progress, and toggleable Human,
Sol, and Detector overlays. Sol boxes can be accepted individually as correct
or replaced with a better human box; proposal boxes are never silently copied
into human truth.

## Detector evaluation gate

Completed human labels are merged with detector proposals only after their
independent source files, immutable task identity, image hashes, and exposure
audit are revalidated. Threshold selection is leave-one-`sourceGroup`-out.
Reports include primary- and any-ball presence, IoU 0.25/0.50 box metrics,
center matching, ranked AP, Brier score, false detections, and source,
environment, stratum, visibility, and object-size slices. Adjacent frames are
also summarized at window, recording, and source-group level.

The current freeze gate requires pooled out-of-fold primary precision at least
0.85, recall at least 0.60, every source group recall at least 0.35, and every
environment recall at least 0.45. A diagnostic fallback threshold cannot be
frozen. Even a passing threshold remains downstream-ineligible until false
track and uncontended throughput gates are measured.

### Completed detector result

Neither registered detector mode passes that gate:

| Development metric | Full frame | Full + four tiles |
|---|---:|---:|
| All-development candidate threshold | 0.0356 | 0.3447 |
| All-development presence precision | 0.850 | 0.851 |
| All-development presence recall | 0.473 | 0.366 |
| All-development presence F1 | 0.608 | 0.512 |
| Source-group OOF presence precision | 0.790 | 0.677 |
| Source-group OOF presence recall | 0.390 | 0.318 |
| Source-group OOF presence F1 | 0.522 | 0.433 |
| OOF box F1 at IoU 0.25 | 0.260 | 0.281 |
| OOF box F1 at IoU 0.50 | 0.157 | 0.151 |
| OOF center-match recall | 0.224 | 0.234 |
| OOF false-positive frames / 1,000 ball-free frames | 263 | 432 |
| 8–15 px center-match recall, all-development threshold | 0.094 | 0.113 |

The tiles recover a few very small balls, but that narrow gain does not survive
as useful presence quality. Tiling reduces pooled OOF precision and recall,
increases the false-positive-frame rate by 64%, worsens medium- and large-ball
localization, and still yields zero held-out beach recall because thresholds
learned on the other source groups do not transfer. The result points to both
small-object localization and severe score/domain calibration problems; this
simple 2×2 scaling scheme was insufficient.

The full-frame report is
`/mnt/freenas/volleycut/ball-presence-v1/reports/ball-presence-development.json`
(SHA-256
`941c056b16661b0a55b6e00d251c2fdbbe8dbdd8018ede71815c2f54623c28a2`).
The tiled report is
`/mnt/freenas/volleycut/ball-presence-v1/reports/ball-presence-development-yolox-s-full-plus-2x2-v1.json`
(SHA-256
`2aa6b7fc78e1a71ef98986fd29562379de5406df82fd83e15ef0e0c056b3494e`).

Sol has no calibrated per-box confidence, so its comparison is one fixed
operating point: presence and role-aware box/center precision, recall, and F1.
It does not receive AP, calibration, or threshold-sweep claims.

## Candidate rally-model signals

The high-rate sidecar is aggregated to the existing 4 fps model grid as:

- detector availability;
- observation fraction;
- observability quality;
- maximum presence probability;
- detected fraction at a frozen threshold;
- normalized candidate count;
- seconds since a detection; and
- quality-gated presence.

The 8 raw signals become 40 contextual columns at the existing five temporal
offsets. They deliberately bypass within-recording percentile ranking so an
unavailable detector stays zero instead of becoming 0.5. A missing sidecar is
a hard error; an explicitly unavailable detector and a detector that ran but
saw no ball are distinct states.

Only after a detector gate passes should complete high-rate development
sidecars be generated. The downstream comparison would then be `full
audiovisual` versus `full audiovisual + ball`, with candidate-specific nested
leave-one-source-group-out fitting and decoding. Model selection uses unpadded
predictions; 0/1/2/3-second padding remains a reported operational tradeoff.
Ball features should be promoted only if event F1 and time IoU each improve by
at least 0.02, the objective improves in at least three of four groups, live
recall drops by at most one percentage point, and short-event slices do not
regress.

## Reproduction outline

Prepare a new, non-overwriting development sample:

```bash
npm run prepare:ball-pilot -- \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --output /path/to/new/round-01 \
  --round 1 \
  --expected-recordings 8
```

Install and verify the pinned detector, then create proposals for each pristine
task in a sibling `suggestions` directory:

```bash
npm run detect:ball-pilot -- install-model \
  --destination /path/to/models/opencv-zoo-yolox-s-2022nov

npm run detect:ball-pilot -- infer-task \
  --task /path/to/round-01/tasks/RECORDING.ball-presence.json \
  --model /path/to/models/opencv-zoo-yolox-s-2022nov \
  --output /path/to/round-01/suggestions/RECORDING.ball-presence.json
```

Prepare a detector-blind Sol copy from the pristine task, never from the
proposal artifact:

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare-ball-presence-sol-review.py \
  --source-task /path/to/round-01/tasks/RECORDING.ball-presence.json \
  --pilot-index /path/to/round-01/index.json \
  --output /path/to/round-01/sol-labels/RECORDING.ball-presence.json \
  --agent-id SOL_RUNNER \
  --model-id MODEL_ID \
  --run-id UNIQUE_RUN_ID
```

After human review, merge the complete human and detector files and run the
development-only evaluator. It will refuse test/challenge tasks, incomplete or
assisted-only truth, missing blind provenance, task substitution, and an
unbound Sol layer.

```bash
npm run merge:ball-pilot -- \
  --reviewed-labels /path/to/round-01/reviews/RECORDING.ball-presence.json \
  --detector-proposals /path/to/round-01/suggestions/RECORDING.ball-presence.json \
  --output /path/to/round-01/merged/RECORDING.ball-presence.json

npm run evaluate:ball-pilot -- \
  --task /path/to/round-01/merged \
  --sol-task /path/to/round-01/sol-labels \
  --pilot-index /path/to/round-01/index.json \
  --output /path/to/reports/ball-presence-development.json
```

## Current decision

Keep the completed labels, canonical reviews, proposal layers, and both reports
as the reusable detector-development corpus. Do not spend several additional
CPU hours generating dense development sidecars, and do not train or report
ball-feature padding or ablation results from either failed detector mode.

The next materially different experiment would be a volleyball-specific
detector or fine-tune evaluated with source-group-held-out weights and
thresholds. The boxes may supervise that detector, but other-court balls must
remain real-ball positives rather than being turned into clean negatives;
`indeterminate` and fully occluded frames require the target-policy handling
already encoded in the annotation schema. That is a larger training project,
not a threshold or image-scaling tweak to this minimum pilot.
