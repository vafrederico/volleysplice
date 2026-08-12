# Minimum ball-presence pilot — 2026-08-11

## Outcome

The minimum viable pilot is implemented as an isolated, development-only
experiment. It does not change the production rally feature extractor or the
saved audiovisual model. The pilot now has:

- exact-frame, six-stratum sampling from all eight development recordings;
- a checksum-pinned CPU detector and immutable proposal artifacts;
- a native frame/box annotation schema and a local review UI;
- separate human, detector, and detector-blind Sol layers;
- a source-group-held-out detector evaluator; and
- eight confidence/availability-gated signals that can later be appended to the
  4 fps rally model without percentile-ranking missing values.

The generic detector has **not** been promoted and the ball signals have **not**
been trained into the rally model yet. Human truth is required first. Treating
the detector's low-threshold candidates as labels would make both detector
quality and the downstream feature result circular.

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

Per-task timings in the proposal ledger are not a benchmark because several
tasks ran while unrelated CPU-heavy jobs occupied the host. An earlier
uncontended development-only smoke test measured about 220 ms per FP32 forward
pass, or roughly 4.5 sampled frames per second.

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

Only after the detector gate passes should complete high-rate development
sidecars be generated. The downstream comparison is then `full audiovisual`
versus `full audiovisual + ball`, with candidate-specific nested
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

Proceed with independent Sol and human labeling. Do not spend the several CPU
hours required for full-corpus high-rate sidecars, and do not train or report
ball-feature padding/ablation results, until the reviewed detector gate says
the signal is real. This is a deliberate feasibility checkpoint rather than a
partial success claim.
