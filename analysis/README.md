# Rally detector

This package is a CPU-first feasibility baseline for continuous, fixed-camera volleyball video. It genuinely trains a model, runs inference, and evaluates rally intervals; it is not a claim of production accuracy before representative videos arrive.

The learned task is binary `live` versus `dead` at 4 samples per second. The audiovisual v2 extractor contains 90 base signals: the original low-resolution appearance, frame-difference, and optical-flow channels; camera motion, focus/blur, visibility, and occlusion proxies; camera-compensated court-motion and stand-down/formation-change proxies; and audio level, transient, onset-cadence, cadence-collapse, and time-since-transient signals. Five centered temporal samples (`-2, -1, 0, +1, +2` seconds) produce 450 model inputs for a class-weighted logistic classifier.

Most channels are converted to tied within-recording percentile ranks to reduce camera/court scale shift. Absolute availability and quality gates retain their original scale. A validation-selected hysteresis decoder jointly tunes smoothing, thresholds, minimum rally duration, gap bridging, and an optional high-confidence short-event exception. User-facing pre-roll and post-roll remain separate edit-list settings.

This deliberately mirrors the reusable ideas in the beach-volleyball thesis—fixed view, temporal frame clusters, a learned classifier, smoothing, minimum-duration filtering—without depending on its unavailable code, model, or data. The extractor/classifier boundary allows a later EfficientNetV2 or STES-derived model to reuse the same manifests, splits, decoder, metrics, and `analysis.json` output.

## Set up

From the repository root:

```bash
npm run analysis:setup
.venv/bin/python -m analysis doctor
.venv/bin/python -m analysis smoke
```

`ffmpeg` and `ffprobe` are required when audio features are enabled (the default) and for media normalization. Source recordings remain unchanged. Normalize phone footage to a constant-frame-rate analysis master before annotating it:

```bash
.venv/bin/python -m analysis normalize \
  --video data/videos/original/match.mov \
  --output data/videos/indoor/match-001-set-01.mp4
```

The command also creates a provenance sidecar with SHA-256 digests. All annotation timestamps refer to the normalized file. This is important because OpenCV timestamps on variable-frame-rate phone recordings are not a safe editing clock.

For a time-bounded feasibility excerpt, add `--start <seconds> --duration <seconds>`. The sidecar records the requested source range while still hashing the complete immutable source file. Excerpts derived from the same match must retain the same `sourceGroup` and split. `--preset fast --threads 2` is useful for bounded evaluation proxies on memory-constrained hosts; retain the default `medium` preset when throughput is less important.

## Annotate and split

Create a starter manifest:

```bash
.venv/bin/python -m analysis init-manifest --output data/videos/manifest.json
```

See [`examples/dataset.example.json`](examples/dataset.example.json) for the full schema. The objective label contract is:

- `start`: serve-ball contact.
- `end`: the first instant live play has ended.
- Intervals are half-open `[start, end)` seconds on the normalized video.
- Celebration, reset, timeouts, warmups, neighboring courts, and replay-like behavior stay negative unless the image is genuinely unusable.

Each recording has a `sourceGroup`. Every set, excerpt, proxy, or re-encode derived from the same match must keep the same group and split. Validation rejects a group crossing splits. Choose splits before training; thresholds are selected on validation, and test is untouched until final evaluation. Training/validation rows also require explicit `consent.train: true`.

Record game context in the optional `game` object. `playersPerTeam` accepts integers from 1 through 6; `targetPoints` accepts 1 through 100 or `null`. Use `null` when the clip does not establish the target—filenames and conventional scoring rules are not sufficient evidence. Evaluation reports stratify results by both fields, including an `unknown` target-points group.

Use `ignoredIntervals` for partial/censored rallies, camera gaps, or genuinely unresolvable spans. Those samples are removed from model fitting, decoder selection, and evaluation rather than silently becoming dead-time labels. Completed workstation exports can be checked with `validate-labels` and combined with `build-manifest`; see [`../docs/labeling-guide.md`](../docs/labeling-guide.md).

After a batch has been continuously reviewed, freeze it before training. This preserves the editable drafts, refuses to overwrite an existing snapshot, marks the copies complete with one review timestamp, validates them, rewrites relative video paths for the snapshot location, emits a SHA-256 ledger, and makes the snapshot files read-only. Completed snapshots reject exactly touching rallies because the binary live/dead target cannot represent two events without dead time between them:

```bash
.venv/bin/python -m analysis freeze-labels \
  --labels-dir data/labels/full \
  --output-dir data/completed/full-v1 \
  --annotator "Reviewer name"
```

If a pre-freeze audit identifies split-shortcut remnants, `--drop-touching-duplicate-tails` removes only a zero-gap second interval whose tags and note exactly duplicate the preceding rally. Every removal is embedded verbatim in `snapshot.json`; other touching intervals still fail validation.

The optional normalized ROI is `{x,y,width,height}` in fractions of the source frame. Start with the full playing zone plus both service areas and a small margin. Consistent manual ROIs are safer than premature automatic court detection.

Validate before extracting hours of video:

```bash
.venv/bin/python -m analysis validate --manifest data/videos/manifest.json
```

## Train, infer, and evaluate

```bash
.venv/bin/python -m analysis train \
  --manifest data/videos/manifest.json \
  --model data/models/rally-v0

.venv/bin/python -m analysis infer \
  --model data/models/rally-v0 \
  --video data/videos/indoor/unseen-set.mp4 \
  --roi 0.05,0.12,0.90,0.86 \
  --output data/analyses/unseen-set-v0

.venv/bin/python -m analysis evaluate \
  --manifest data/videos/manifest.json \
  --model data/models/rally-v0 \
  --split test \
  --output data/reports/rally-v0-test.json
```

To infer every recording in an immutable manifest with one saved model, use the repository
batch wrapper. It writes `model-<version>--<recording-id>` analysis directories, safely skips
completed runs, and refuses incomplete or ambiguous destinations:

```bash
npm run infer:model-dataset -- \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-percentile-v1
```

After the batch is complete, measure how symmetric export padding changes coverage and
footage cost without changing the model's core predictions:

```bash
npm run evaluate:model-padding -- \
  --model-version full-percentile-v1 \
  --padding-seconds 0 1 2 3 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-model-padding-v1.json
```

The report merges overlapping padded crops and reports each split separately so training,
validation/tuning, and held-out evaluation results are not conflated.

New training runs default to tied within-recording percentile normalization. Use `--sequence-normalization none` only for an explicit raw-feature ablation. The normalization mode is stored in the model artifact, and older saved models without that field retain their original raw-feature behavior.

Use `--no-audio` or `--no-advanced-visual` only for explicit extractor ablations. Audio is decoded to 16 kHz mono by default; change it with `--audio-sample-rate`. Videos without a decodable audio stream receive zero-valued audio channels plus `audio_available=0` rather than a fabricated percentile signal.

Feature caches are keyed by source-content SHA-256, ROI, extractor version, and configuration. Model artifacts contain human-readable metadata plus NumPy weights and never use pickle. Existing model, analysis, normalization, and evaluation artifacts are not overwritten.

Evaluation reports product-relevant interval precision/recall/F1 at IoU 0.5, temporal IoU, live-time recall, dead time retained, exact rally-count rate, and boundary errors. Frame accuracy is intentionally not the primary metric because long dead periods can make it look good while rallies are missed.

## Grouped feature study

The feature-study runner keeps the fixed test split unopened while it performs nested leave-one-`sourceGroup`-out development evaluation. It prepares the audiovisual superset once, then fits full, legacy-only, added-only, and full-minus-family candidates. It also reports grouped circular-shift importance for every family and every base signal, standardized coefficient profiles, the short-event decoder ablation, outcome slices, and 0/1/2/3-second padding sensitivity:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-model-features.py development \
  --manifest data/manifests/full-gold-v1.json \
  --cache-dir data/features/audiovisual-v2 \
  --padding-seconds 0 1 2 3 \
  --output data/reports/audiovisual-v2-development.json
```

Only after that report is frozen, explicitly open the fixed regression-test split:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-model-features.py final-test \
  --manifest data/manifests/full-gold-v1.json \
  --cache-dir data/features/audiovisual-v2 \
  --development-report data/reports/audiovisual-v2-development.json \
  --model data/models/audiovisual-v2-final \
  --output data/reports/audiovisual-v2-final-test.json \
  --open-test
```

The final-test gate verifies the manifest, recording snapshots, feature version/signature, and experiment-code hashes. Importance labels require directionally consistent source-group effects plus compatible mean and median magnitude. They are exploratory evidence, not significance tests.

## Ball-presence feasibility pilot

Ball presence is isolated from the production extractor until a detector is
validated. The pilot samples exact 15 fps development frames, keeps Human, Sol,
and detector layers provenance-distinct, selects thresholds with held-out
source groups, and can aggregate a validated high-rate detector sidecar into
eight raw 4 fps signals. It rejects test/challenge tasks and excludes any human
frame that saw proposals before its label was finalized.

See the [ball-presence labeling guide](../docs/ball-presence-labeling-guide.md)
for the UI workflow and the [pilot decision record](../docs/research/minimum-ball-presence-pilot-2026-08-11.md)
for frozen artifacts, quality gates, and the reason downstream model training
is deferred until independent labels exist.

## What the current model does not do

- It does not detect whether the whole court is visible; capture geometry must be confirmed by a person.
- It does not identify players, poses, receiving formations, ball trajectories, aces, or service faults as semantic classes. Motion/formation/occlusion channels are deliberately named proxies.
- Reliable ball tracking is not part of the current model. A separate high-resolution ball-presence pilot is collecting and validating the required labels before any promotion.
- The short-event decoder is a generic high-confidence duration exception; ace and service-fault tags are used for evaluation slices, not outcome-aware inference.
- Weighted logistic output is a ranking confidence, not a calibrated probability.
- It is an offline centered-context model, not low-latency live detection.
- It should establish a reproducible baseline and expose data problems, not substitute for benchmarking on unseen indoor, grass, and beach matches.
