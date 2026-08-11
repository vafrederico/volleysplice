# Rally detector v0

This package is a CPU-first feasibility baseline for continuous, fixed-camera volleyball video. It genuinely trains a model, runs inference, and evaluates rally intervals; it is not a claim of production accuracy before representative videos arrive.

The learned task is binary `live` versus `dead` at 4 samples per second. Each sample contains low-resolution court appearance, frame difference, regional motion, and optical-flow features. Feature channels are converted to tied within-recording percentile ranks to reduce camera/court scale shift. Five centered temporal samples (`-2, -1, 0, +1, +2` seconds) are passed to a class-weighted logistic classifier. A validation-selected hysteresis decoder jointly tunes smoothing, thresholds, minimum rally duration, and gap bridging to convert probabilities into core rally intervals. User-facing pre-roll and post-roll remain separate edit-list settings.

This deliberately mirrors the reusable ideas in the beach-volleyball thesis—fixed view, temporal frame clusters, a learned classifier, smoothing, minimum-duration filtering—without depending on its unavailable code, model, or data. The extractor/classifier boundary allows a later EfficientNetV2 or STES-derived model to reuse the same manifests, splits, decoder, metrics, and `analysis.json` output.

## Set up

From the repository root:

```bash
npm run analysis:setup
.venv/bin/python -m analysis doctor
.venv/bin/python -m analysis smoke
```

`ffmpeg` and `ffprobe` should be installed on the host. Source recordings remain unchanged. Normalize phone footage to a constant-frame-rate analysis master before annotating it:

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

New training runs default to tied within-recording percentile normalization. Use `--sequence-normalization none` only for an explicit raw-feature ablation. The normalization mode is stored in the model artifact, and older saved models without that field retain their original raw-feature behavior.

Feature caches are keyed by source-content SHA-256, ROI, extractor version, and configuration. Model artifacts contain human-readable metadata plus NumPy weights and never use pickle. Existing model, analysis, normalization, and evaluation artifacts are not overwritten.

Evaluation reports product-relevant interval precision/recall/F1 at IoU 0.5, temporal IoU, live-time recall, dead time retained, exact rally-count rate, and boundary errors. Frame accuracy is intentionally not the primary metric because long dead periods can make it look good while rallies are missed.

## What v0 does not do

- It does not detect whether the whole court is visible; capture geometry must be confirmed by a person.
- It does not yet use audio, ball tracking, player detection, or VNL action labels.
- Weighted logistic output is a ranking confidence, not a calibrated probability.
- It is an offline centered-context model, not low-latency live detection.
- It should establish a reproducible baseline and expose data problems, not substitute for benchmarking on unseen indoor, grass, and beach matches.
