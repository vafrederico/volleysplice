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

For ranking rally-model iterations, the primary ordering metric is **Padded P/Core R F1**
(`F1_padP_coreR`), not event F1. It combines precision of the padded model export against
equally padded human labels with recall of that same model export against core human labels.
Rank the pooled development/validation result descending under identical padding, labels,
and recording scope; keep the protected test split closed during iteration. The complete
definition and aggregation rules are in
[`../docs/model-ranking-metric.md`](../docs/model-ranking-metric.md).
After padding, join positive gaps strictly shorter than the configured short-gap threshold
(3 seconds by default) in both model and padded-human exports. Count the retained gap as
export time, report the threshold in the artifact, and never rejoin across ignored time.
Every iteration evaluation must report the complete symmetric padding sensitivity sweep:
`(before, after) = (0, 0), (1, 1), (2, 2), (3, 3)` seconds. Calculate the metric and
its pooled components independently for all four cases. Use only the predeclared target
padding for ranking; do not choose a different best-padding case per model.

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

## Multiscale transition and interaction study

The transition-feature runner reuses the frozen 90-signal audiovisual cache and appends a
predeclared bank of 0.5/1/2/4/8-second summaries and seven cross-modal interactions exactly once
per timestamp. Development selection is nested leave-one-`sourceGroup`-out, reports each family
and individual interaction ablations, and never prepares the protected test split:

```bash
npm run evaluate:transition-features -- development \
  --manifest data/manifests/full-gold-v1.json \
  --cache-dir data/features/audiovisual-v2 \
  --output data/reports/multiscale-interactions-v1-development.json
```

The derived bank uses centered windows and is therefore an offline cutting experiment, not a
streaming model. Only after the development report is frozen can the selected candidate be fit on
all development rows and checked once against the retrospective regression split:

```bash
npm run evaluate:transition-features -- retrospective-test \
  --manifest data/manifests/full-gold-v1.json \
  --cache-dir data/features/audiovisual-v2 \
  --development-report data/reports/multiscale-interactions-v1-development.json \
  --output data/reports/multiscale-interactions-v1-retrospective-test.json \
  --open-test
```

## Court-relative feature study

The court-relative runner uses the exact frozen winner from the transition study as its control.
It derives reflection-invariant, signed fixed-endline, and combined directional summaries from the
existing rectangle ROI and cached 3×3 motion grids. No environment, player-count, side-switch, or
annotation fields enter the model:

```bash
npm run evaluate:court-relative-features -- development \
  --manifest data/manifests/full-gold-v1.json \
  --cache-dir data/features/audiovisual-v2 \
  --step1-development-report data/reports/multiscale-interactions-v1-development.json \
  --output data/reports/court-relative-v1-development.json
```

After development selection is frozen, the explicit retrospective regression check is:

```bash
npm run evaluate:court-relative-features -- retrospective-test \
  --manifest data/manifests/full-gold-v1.json \
  --cache-dir data/features/audiovisual-v2 \
  --development-report data/reports/court-relative-v1-development.json \
  --output data/reports/court-relative-v1-retrospective-test.json \
  --open-test
```

## Serve-anchored multistate study

The multistate runner reconstructs the frozen court-study winner and compares two architectures
with exactly the same feature rows: an ordinary binary logistic control and four one-vs-rest state
heads decoded through the legal `DEAD -> SETUP -> SERVE -> LIVE -> DEAD` graph. State-duration
priors and transition bonuses are fit inside each training fold:

```bash
npm run evaluate:multistate-features -- development \
  --manifest data/manifests/full-gold-v1.json \
  --cache-dir data/features/audiovisual-v2 \
  --upstream-report data/reports/court-relative-v1-development.json \
  --output data/reports/multistate-v1-development.json
```

The retrospective command rejects reduced-inner-fold reports and stays closed unless multistate
passes every frozen objective, short/fault, ordinary-long, and live-recall development guardrail:

```bash
npm run evaluate:multistate-features -- retrospective-test \
  --manifest data/manifests/full-gold-v1.json \
  --cache-dir data/features/audiovisual-v2 \
  --development-report data/reports/multistate-v1-development.json \
  --output data/reports/multistate-v1-retrospective-test.json \
  --open-test
```

The follow-up diagnostic command exactly refits each frozen outer-fold state and binary model,
checks its fingerprint and metrics, and writes reusable development-only OOF scores. It reports
per-state precision, recall, Brier score, log loss, calibration error, decoded durations, and true
serves rejected by the constrained path. Its start/end boundary replacements use development
labels only and are explicitly non-promotable upper bounds:

```bash
npm run evaluate:multistate-followup -- diagnostics \
  --manifest data/manifests/full-gold-v1.json \
  --feature-cache-dir data/features/audiovisual-v2 \
  --multistate-report data/reports/multistate-v1-development.json \
  --oof-cache-output data/models/multistate-followup-oof-v1 \
  --output data/reports/multistate-followup-diagnostics-v1.json
```

The command never prepares protected rows. Downstream hybrid and calibration experiments must
validate the cache index and its per-recording hashes before using these scores.

The conservative hybrid is a separate full nested study. Its four predeclared modes are binary
no-op, boundary snapping, strict isolated short rescue, and snapping plus rescue. Every mode retains
all binary proposals; long proposals also have structural overlap/IoU floors. Mode selection uses
only the three inner OOF source groups inside each outer fold, and the no-op must exactly reproduce
the frozen binary control:

```bash
npm run evaluate:multistate-followup -- hybrid-development \
  --manifest data/manifests/full-gold-v1.json \
  --feature-cache-dir data/features/audiovisual-v2 \
  --multistate-report data/reports/multistate-v1-development.json \
  --output data/reports/multistate-conservative-hybrid-v1-development.json
```

The protected retrospective stays closed unless paired objective evidence and every live-recall,
ordinary-long, precision, dead-time, boundary, and short-rescue development guardrail pass.

The first existing-label ablation is another full nested study with a fixed four-candidate Cartesian
product. It compares the v1 normalized balanced OVR emissions with a fold-prior correction, and the
v1 geometric durations with an empirical SETUP/two-geometric LIVE prior. No numeric hyperparameter
is searched, and every state head must reproduce its frozen v1 fingerprint:

```bash
npm run evaluate:multistate-followup -- existing-label-ablation \
  --manifest data/manifests/full-gold-v1.json \
  --feature-cache-dir data/features/audiovisual-v2 \
  --multistate-report data/reports/multistate-v1-development.json \
  --output data/reports/multistate-existing-label-ablation-v1-development.json
```

Class prevalence and duration mixtures are estimated only from the corresponding training fold.
The reference row must exactly reproduce the frozen v1 multistate metrics, and test remains closed
unless the candidate both improves v1 and passes the original operational-binary promotion gate.
Inner selection additionally constrains source-macro objective, live recall, precision, retained
dead time, event F1, and short/fault/ordinary outcome slices.

The next existing-label study treats current ace/service-fault tags as a weak immediate-result
proxy. It compares v1 with a private ordinary-LIVE/result-LIVE duration graph, first using only the
training-fold tag prevalence and then a fixed 17-feature cross-fitted proxy. Tags are targets only,
the graph never permits a direct SERVE-to-DEAD event, and every candidate interval contains LIVE:

```bash
npm run evaluate:multistate-followup -- immediate-result-development \
  --manifest data/manifests/full-gold-v1.json \
  --feature-cache-dir data/features/audiovisual-v2 \
  --multistate-report data/reports/multistate-v1-development.json \
  --existing-label-report data/reports/multistate-existing-label-ablation-v1-development.json \
  --output data/reports/multistate-immediate-result-proxy-v1-development.json
```

This report never opens the reused protected test. A passing development gate authorizes only a
fresh independent source-group validation; the binary control remains operational meanwhile.

The joint-emission study then holds the v1 graph, geometric duration priors, and fold-frozen
transition bonus fixed while replacing four independently balanced OVR heads with one normalized
four-state linear softmax. The primary candidate uses ordinary categorical cross-entropy so the
training-fold class prior is learned natively; a separately named balanced/prior-corrected arm is
reported as a secondary diagnostic:

```bash
npm run evaluate:multistate-followup -- joint-emissions-development \
  --manifest data/manifests/full-gold-v1.json \
  --feature-cache-dir data/features/audiovisual-v2 \
  --multistate-report data/reports/multistate-v1-development.json \
  --existing-label-report data/reports/multistate-existing-label-ablation-v1-development.json \
  --immediate-result-report data/reports/multistate-immediate-result-proxy-v1-development.json \
  --output data/reports/multistate-joint-emissions-v1-development.json
```

Softmax epoch caps are chosen inside each outer fold from its three inner fits. State confusion,
multiclass log loss, Brier score, and calibration are held-source diagnostics and do not select a
candidate. Advancement requires unanimous outer selection plus every v1 and operational-binary
guardrail, and even then authorizes only a fresh independent source-group assessment.

The edge-evidence study keeps the prior-corrected OVR/geometric decoder fixed and evaluates the
complete four-arm family: no evidence, serve-edge only, terminal-edge only, and both. Independent
balanced specialists map their probabilities to fixed bounded evidence `2p-1` at coefficient one;
there is no threshold or strength search. Serve evidence affects only `SETUP -> SERVE`, terminal
evidence affects only `LIVE -> DEAD`, and the combined arm is eligible only when both individual
arms pass their mechanism gates:

```bash
npm run evaluate:multistate-followup -- edge-evidence-development \
  --manifest data/manifests/full-gold-v1.json \
  --feature-cache-dir data/features/audiovisual-v2 \
  --multistate-report data/reports/multistate-v1-development.json \
  --existing-label-report data/reports/multistate-existing-label-ablation-v1-development.json \
  --immediate-result-report data/reports/multistate-immediate-result-proxy-v1-development.json \
  --joint-emissions-report data/reports/multistate-joint-emissions-v1-development.json \
  --output data/reports/multistate-edge-evidence-v1-development.json
```

The serve target is the established fixed one-second pulse around current rally starts. The
terminal target is the established two-second live/dead window around rally ends plus a one-second
pre-serve already-dead negative. Specialists and their epoch caps remain fold-local. Promotion
also requires start/serve-anchor or endpoint mechanism gains, the common recall/precision/outcome
guardrails, unanimous outer selection, and the original operational-binary gate.

## Transition-label experiment gate

The transition-cue pilot has a separate read-only gate that distinguishes the eight development
recordings from the protected test recording. It validates each draft, content-hashes the label
files, reports the five-rally-per-recording cue debt and three-hard-negative-per-recording debt,
and registers the exact future serve-edge, terminal-edge, verified-result, and hard-negative
studies without preparing video or model features:

```bash
npm run report:transition-label-gate -- \
  --labels-dir data/labeling-v1-2026-08-09/labels/full \
  --output data/reports/transition-label-experiment-gate-v2.json
```

Add `--require-development-ready` in an automated workflow to fail closed until the 40 development
pilot rallies are fully cued. The five protected pilot rallies are reported as sealed debt but are
never required or consumed by a development runner. A future experiment must rebuild and freeze a
new manifest from the completed train/validation snapshots; it must not attach mutable draft fields
to an earlier frozen report lineage.

After labels are ready, freeze **only** the eight train/validation documents into a new immutable
snapshot, rebuild a development-only manifest from that snapshot, then run the candidate-specific
preflight before any feature preparation:

```bash
npm run preflight:transition-pilot -- \
  --candidate reaction-supervised-serve-edge \
  --baseline-manifest data/manifests/full-gold-v1.json \
  --manifest data/manifests/transition-pilot-development-v1.json \
  --snapshot-ledger data/labels/transition-pilot-development-v1/snapshot.json \
  --transition-gate data/reports/transition-label-experiment-gate-v2.json \
  --output data/reports/transition-pilot-reaction-preflight-v1.json
```

The preflight requires exact development IDs from the baseline, non-null fields for the selected
candidate, immutable draft and snapshot hashes, byte-matching snapshot/manifest target payloads,
and at least two development source groups. It rejects test/challenge rows and does not open the
protected label document. The same command accepts the other three registered candidate names.

## Out-of-fold component-selector study

The component-selector study regenerates v4/v5 rally and serve candidates with fold-specific
models. Template artifacts donate configuration only; their fitted weights are never used for OOF
predictions. The cache excludes protected rows and records generator provenance for every overlap
component:

```bash
npm run evaluate:component-selector -- prepare-oof \
  --manifest data/manifests/full-gold-v1.json \
  --output-dir data/models/component-selector-oof-v1

npm run evaluate:component-selector -- development \
  --manifest data/manifests/full-gold-v1.json \
  --oof-cache data/models/component-selector-oof-v1 \
  --output data/reports/component-selector-v1-development.json
```

The trained selector is assessed on a generator-held grass validation source against v4, v5, and
the exact frozen v4/v5 intersection. This is leakage-safe tuning evidence, not a full nested LOGO
promotion study; the report predeclares the frozen intersection as the retained policy and does not
implement a retrospective-test path.

## Frozen high-resolution embedding study

The high-resolution extractor uses an explicitly supplied, SHA-pinned official OpenCV Zoo
MobileNetV2 ONNX graph. It samples four role-neutral rectangle-ROI crops at one fps, extracts the
1,280-value penultimate global-average-pool representation, and applies a deterministic 64-value
projection. It never downloads a model or overwrites a cache:

```bash
npm run extract:highres-embeddings -- \
  --manifest data/manifests/full-gold-v1.json \
  --backbone data/models/image_classification_mobilenetv2_2022apr.onnx \
  --backbone-sha256 <pinned-sha256> \
  --cache-dir data/features/highres-mobilenetv2 \
  --output data/reports/highres-development-extraction.json

npm run evaluate:highres-embeddings -- development \
  --manifest data/manifests/full-gold-v1.json \
  --warm-cache-dir data/features/audiovisual-v2 \
  --highres-cache-dir data/features/highres-mobilenetv2 \
  --upstream-report data/reports/court-relative-v1-development.json \
  --backbone data/models/image_classification_mobilenetv2_2022apr.onnx \
  --backbone-sha256 <pinned-sha256> \
  --output data/reports/highres-mobilenetv2-v1-development.json
```

Centered 2/8-second summaries make this an offline-cutter experiment. Protected caches are not
extracted during development. Only if the candidate passes the full nested paired gate may test be
extracted with `--splits test --allow-protected-splits`; the retrospective evaluator then requires
that separate extraction index and `--open-test`. Reports pin the manifest, upstream config,
backbone, projection, cache hashes, implementation hashes, and OpenCV CPU runtime identity.

## Experimental serve specialist

The optional serve path trains a second logistic head on narrow windows around rally starts while
reusing the rally model's cached features. Its validation search freezes a contact threshold,
temporal non-maximum suppression, and an anchored short-rally composition. Artifacts record their
`predictionTask`; legacy artifacts default to `rally-live`, and incompatible model pairs are
rejected.

```bash
.venv/bin/python -m analysis train-serve \
  --manifest data/videos/manifest.json \
  --rally-model data/models/rally-v0 \
  --model data/models/serve-v0 \
  --output data/reports/serve-v0-validation.json

.venv/bin/python -m analysis evaluate-serve \
  --manifest data/videos/manifest.json \
  --rally-model data/models/rally-v0 \
  --serve-model data/models/serve-v0 \
  --split test \
  --output data/reports/serve-v0-test.json

.venv/bin/python -m analysis infer \
  --model data/models/rally-v0 \
  --serve-model data/models/serve-v0 \
  --video data/videos/indoor/unseen-set.mp4 \
  --output data/analyses/unseen-set-paired
```

This path is opt-in because its first real-data experiment improved short-rally coverage but did
not improve strict held-out event recall. See the
[serve-specialist experiment](../docs/research/serve-specialist-experiment-2026-08-11.md) and the
[audiovisual follow-up](../docs/research/serve-specialist-audiovisual-experiment-2026-08-11.md).

The research-only stacking path appends the frozen specialist's continuous score to the rally
inputs. Training rows use leave-one-`sourceGroup`-out specialist scores; validation and evaluation
use the exact frozen specialist bound into the artifact. It also trains a matched base-feature
control and reports frozen-decoder, validation-selected, and score-clamped variants:

```bash
.venv/bin/python -m analysis train-rally-with-serve \
  --manifest data/videos/manifest.json \
  --baseline-rally-model data/models/rally-v0 \
  --serve-model data/models/serve-v0 \
  --control-model data/models/rally-stack-control-v0 \
  --model data/models/rally-stack-v0 \
  --output data/reports/rally-stack-v0-validation.json

.venv/bin/python -m analysis evaluate-rally-with-serve \
  --manifest data/videos/manifest.json \
  --baseline-rally-model data/models/rally-v0 \
  --serve-model data/models/serve-v0 \
  --control-model data/models/rally-stack-control-v0 \
  --model data/models/rally-stack-v0 \
  --split test \
  --output data/reports/rally-stack-v0-test.json
```

The first real-data stack did not improve the retrospective test and is not wired into ordinary
inference. See the
[stacking experiment](../docs/research/rally-with-serve-feature-experiment-2026-08-11.md).

The follow-up evidence experiment compares a zero-fallback serve gate with a rally head that
consumes four decoded peak-window features. Training peaks are source-group cross-fitted; the
fixed two-second radius and frozen serve decoder are recorded in the artifact:

```bash
.venv/bin/python -m analysis train-serve-evidence \
  --manifest data/videos/manifest.json \
  --baseline-rally-model data/models/rally-v0 \
  --serve-model data/models/serve-v0 \
  --control-model data/models/rally-stack-control-v0 \
  --model data/models/rally-peak-window-v0 \
  --output data/reports/serve-evidence-v0-validation.json

.venv/bin/python -m analysis evaluate-serve-evidence \
  --manifest data/videos/manifest.json \
  --baseline-rally-model data/models/rally-v0 \
  --serve-model data/models/serve-v0 \
  --control-model data/models/rally-stack-control-v0 \
  --model data/models/rally-peak-window-v0 \
  --split test \
  --output data/reports/serve-evidence-v0-test.json
```

The first run did not pass validation promotion guardrails. Its no-fallback gate was cleaner on the
retrospective test, while the learned head recovered faults at the cost of ordinary rallies. See
the [serve-evidence follow-up](../docs/research/serve-evidence-gate-and-peak-window-2026-08-11.md).

The dual-pair fusion evaluator compares an older rally+serve pair with a newer pair. Validation
decides whether to enable add-only fusion after auditing score gates (including two-head serve
agreement and live-score uplift), and selects a conservative boundary selector over unambiguous
overlap components. Non-validation evaluation requires the immutable validation report so the
chosen thresholds cannot change after test access:

```bash
.venv/bin/python -m analysis evaluate-dual-serve-fusion \
  --manifest data/videos/manifest.json \
  --v4-rally-model data/models/rally-v4 \
  --v4-serve-model data/models/serve-v4 \
  --v4-cache-dir data/features/v4 \
  --v5-rally-model data/models/rally-v5 \
  --v5-serve-model data/models/serve-v5 \
  --v5-cache-dir data/features/v5 \
  --split validation \
  --output data/reports/dual-fusion-validation.json

.venv/bin/python -m analysis evaluate-dual-serve-fusion \
  --manifest data/videos/manifest.json \
  --v4-rally-model data/models/rally-v4 \
  --v4-serve-model data/models/serve-v4 \
  --v4-cache-dir data/features/v4 \
  --v5-rally-model data/models/rally-v5 \
  --v5-serve-model data/models/serve-v5 \
  --v5-cache-dir data/features/v5 \
  --split test --retrospective \
  --decision data/reports/dual-fusion-validation.json \
  --output data/reports/dual-fusion-test.json
```

The first real-data run disabled uncovered additions because all eight validation candidates were
false. Its selected boundary rule improved validation event F1 and retrospective boundary quality,
but not retrospective strict event recall. See the
[v4/v5 fusion experiment](../docs/research/dual-serve-v4-v5-fusion-experiment-2026-08-11.md).

To materialize the frozen v4+v5 decision and every persisted specialist iteration as immutable
review-dashboard timelines, use the cache-backed dataset wrappers:

```bash
npm run infer:dual-serve-fusion-dataset
npm run infer:specialist-model-dataset
```

The specialist wrapper validates every stacked, endpoint, and dead-state artifact against its
recorded rally/serve dependencies and manifest before prediction. It uses only persisted
validation-selected decoders and refinements. Pure serve-contact and end-contact heads are shown
through their defined rally composition rather than decoded as meaningless standalone intervals.
Long artifact names use a short SHA-based analysis directory ID while retaining the full model
version and lineage inside `analysis.json`. Existing complete outputs are verified and skipped;
partial or differently bound destinations are rejected.

The end-boundary experiments support two additional specialist targets. `train-dead-ball` fits a
narrow pulse around annotated rally ends. `train-dead-state` fits either a local live-to-dead
transition (`end-transition`, the default) or the all-frame inverse-rally control (`global-dead`).
Both commands can ablate legacy and normalized-band audio while preserving the shared full feature
signature:

```bash
.venv/bin/python -m analysis train-dead-state \
  --manifest data/videos/manifest.json \
  --rally-model data/models/rally-v5 \
  --serve-model data/models/serve-v5 \
  --model data/models/dead-state-v0 \
  --dead-state-input-profile visual-plus-legacy-audio \
  --output data/reports/dead-state-v0-validation.json

.venv/bin/python -m analysis evaluate-dead-state \
  --manifest data/videos/manifest.json \
  --rally-model data/models/rally-v5 \
  --serve-model data/models/serve-v5 \
  --model data/models/dead-state-v0 \
  --split test --retrospective \
  --output data/reports/dead-state-v0-test.json
```

These paths remain research-only. The first audio ablation favored the legacy-audio local
transition head and did not support promoting the new band features. See the
[end-boundary/dead-state experiment](../docs/research/end-and-dead-state-audio-experiment-2026-08-12.md).

## Side-switch specialist v2

The v2 marker pipeline is separate from rally feature extraction. It first creates one
immutable, decision-free feature artifact over the preregistered raw recording split,
then freezes model family/L2, static threshold, and decoder settings before the separate
evaluation command can materialize confirmation decisions:

```bash
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-v2.py
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v2.py freeze
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v2.py evaluate
PYTHONPATH=. .venv/bin/python scripts/build-side-switch-v2-provenance.py
```

Every command refuses to overwrite its durable NAS destination. The provenance command
verifies full-file SHA-256 for all 11 source videos and binds the feature, model,
development-dataset, and evaluation artifacts. The retained result is review-ranking
research, not a production automatic scorer. See the
[v2 decision record](../docs/research/side-switch-specialist-v2-2026-08-20.md).

Side-switch v3 uses the frozen policy in
[`side_switch_training_policy.py`](side_switch_training_policy.py): every recording is
one set beginning at score zero, cadence is seven points, and candidate margins ±1
through ±4 are tested rather than asserting an event at an exact rally index. The v3
decoder re-anchors after selection, permits no switch, uses one-to-one assignment in
overlapping ±4 windows, and caps a set at six opportunities. Every new trainer must call
`validate_side_switch_fit_recordings` before fitting so the blurry
`beach-source-02` recording cannot influence preprocessing or model selection.
The historical v1 trainer also accepts repeatable `--exclude-fit-recording` arguments for
controlled counterfactuals.

V3 rebuild commands are:

```bash
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-v3.py
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v3.py freeze
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v3.py evaluate
PYTHONPATH=. .venv/bin/python scripts/build-side-switch-v3-provenance.py
```

All destinations are immutable. V3 failed its source-held-out promotion gate, so the
Python reference was not ported into production TypeScript/Java. See the
[v3 decision record](../docs/research/side-switch-specialist-v3-2026-08-20.md),
[v3 on-device plan](../docs/research/side-switch-v3-on-device-plan-2026-08-20.md), and
[v1 blur-exclusion result](../docs/research/side-switch-v1-blur-exclusion-counterfactual-2026-08-20.md).

Side-switch v4 changes only the visual representation requested after v3. It samples
seven frames throughout each rally, calibrates foreground-net height from the first seven
rallies, applies court-relative vertical normalization and camera-translation
compensation, and pools broad/tight near/far side palettes. Labels, split, cadence,
candidate margins, re-anchoring, and maximum opportunity count remain unchanged.

V4 rebuild commands are:

```bash
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-v4.py
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v4.py freeze
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v4.py evaluate
PYTHONPATH=. .venv/bin/python scripts/build-side-switch-v4-provenance.py
```

All destinations are immutable. V4 improves the retrospective raw-phone result over v3
but remains below automatic-use requirements, so it also has no production port. See the
[v4 decision record](../docs/research/side-switch-specialist-v4-2026-08-20.md).

Side-switch v5 keeps v4 geometry and extracts player-like motion components from every
rally in the set. The classifier uses player-isolated near/far palettes and proposal
quality. The optional decoder anchors team sides from the first three score-zero rallies
and carries parity across switches; validation selected orientation weight zero.

V5 rebuild commands are:

```bash
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-v5.py
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v5.py freeze
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v5.py evaluate
PYTHONPATH=. .venv/bin/python scripts/build-side-switch-v5-provenance.py
```

All destinations are immutable. V5 is the best visual side-switch result but remains
below automatic-use precision, so it has no production port. See the
[v5 decision record](../docs/research/side-switch-specialist-v5-2026-08-20.md).

Side-switch v6 replaces v5's motion components with the pinned 3.48 MB block-int8
OpenCV Zoo MediaPipe person localizer. It samples three frames per rally, runs four
overlapping ownership tiles, pools landmark-defined torso palettes, and keeps at most
two detections per canonical court side. Confidence-gated online team prototypes add
three adaptive orientation inputs. A separately fitted 26-input ablation holds the
score-zero prototypes fixed.

V6 rebuild commands are:

```bash
PYTHONPATH=. .venv/bin/python scripts/install-side-switch-player-detector.py
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-v6.py
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v6.py freeze
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v6.py evaluate
PYTHONPATH=. .venv/bin/python scripts/build-side-switch-v6-provenance.py
```

Every destination is immutable, and the trainer pins the extracted feature SHA before
the first fit. V6 improves raw-phone row AP but regresses exact and tolerant event F1,
so it is not promoted and has no production port. V5/V6 event metrics are conditioned
on reviewed rally-gap candidates rather than exhaustive full-video truth; their 59-gap
proposal union is available in the
[review UI](../docs/research/side-switch-v5-v6-review-ui-2026-08-20.md). See the
[v6 decision record](../docs/research/side-switch-specialist-v6-2026-08-20.md).

The production-state follow-up replays both shipped production bundles over the
available F104 matrices, derives soft rally/serve/dead-state context, and creates
optional serve-anchored V5/V6 appearance views. Suppression is retained only as a
quarantined diagnostic because its targets and fitting recordings overlap this task.

Rebuild commands are:

```bash
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-production-state.py prepare
# Run the documented V5/V6 grounded extraction and four augment commands.
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py freeze --family v5
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py evaluate --family v5
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py attribute --family v5
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py freeze --family v6
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py evaluate --family v6
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py attribute --family v6
PYTHONPATH=. .venv/bin/python scripts/build-side-switch-production-state-provenance.py
```

V5 original appearance plus the ten state/gating inputs is a positive exploratory
review-ranking result; V6 and serve-grounded appearance are not promoted. See the
[production-state decision record](../docs/research/side-switch-production-state-experiment-2026-08-20.md)
for exact commands, hashes, attribution, and guardrails.

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
