# VolleyCut production model architecture

This document summarizes audiovisual feature extraction and specifies learned-model
inference, temporal model decoding, ensemble composition, and the suppression safety
policy. The normative feature formulas, version profiles, and production/research split
are in [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md). Trained artifacts, fitting sources,
and predecessor changes are registered in [`MODELS.md`](MODELS.md). This document does
not describe the broader editor, project-storage, playback, video-decoding, or
export-encoding architecture.

VolleyCut's production architecture is a lightweight, on-device audiovisual signal-
processing system. It does not explicitly detect the volleyball, players, net, score,
or named volleyball actions. Instead, it learns statistical patterns that distinguish
live rallies, serve contact, end-of-play transitions, common false positives, and—in
the production browser—the physical camera-space serving side.

The currently deployed rally/export system contains seven logistic classifiers:

- two independent three-head rally-model bundles, used as a recall-safety ensemble; and
- one false-positive suppression head, used as an optional, review-first veto.

The production browser adds one class-balanced logistic serving-side classifier. Its
hybrid serve gate is deterministic and reuses the two existing serve heads and their
decoded rally intervals; it is not a ninth learned classifier. The browser therefore
runs eight learned classifiers in total.

The seven rally/suppression classifiers run locally in both the production browser and
native Android applications. Those clients share the same F104/520 model artifacts,
temporal decoders, ensemble rules, and suppression-policy contract. Serving-side fixed-
flight v3 and its hybrid gate are additionally deployed in the production browser with a
checked-in runtime artifact and browser tests. Android does not yet implement or package
that eighth classifier; the cross-platform port is specified in
[`docs/serving-side-score-tracking-android-spec.md`](docs/serving-side-score-tracking-android-spec.md).

## What the models look for

Every quarter-second, VolleyCut measures low-resolution video and audio signals including:

- motion magnitude and the fraction of the court that is active;
- whether motion is distributed across several court regions;
- motion onset, motion collapse, and synchronized stand-down;
- changes in the location and spread of player-relative motion;
- camera motion versus motion remaining after camera compensation;
- frame difference, optical flow, brightness, sharpness, obstruction, and visibility;
- audio transients that resemble contact;
- repeated audio-onset cadence during live play;
- cadence collapse after play; and
- frequency-band energy relative to the recording's background noise.

The serving-side target separately measures court-end residual flow and concentrated
post-contact motion around a candidate anchor. It uses changes across launch, early-flight,
and late-flight phases to infer `near` versus `far`, including cases where the server is
partially visible or offscreen. It is a motion proxy rather than a semantic ball detector.

These are learned associations rather than semantic detections. For example, a strong
transient does not necessarily mean ball contact, and court-wide movement does not
necessarily mean a rally. Walking, retrieving a ball, changing sides, and setting up for
the next serve are important confusing cases represented in training and evaluation.

## Feature pipeline

The production feature path is:

```text
Video sampled at 4 fps and reduced to 192x108
                |
                |-- 73 appearance and motion features
                |--  4 temporal-motion features
                `-- 27 audio features
                          |
                     104 base features
                          |
          whole-game-window percentile normalization
                          |
           context at -2s, -1s, 0s, +1s, and +2s
                          |
                     520 model inputs
```

For a prediction at time `t`, the model therefore sees evidence around the prediction:

```text
t - 2s    t - 1s    t    t + 1s    t + 2s
```

This context can represent a sequence such as:

```text
quiet setup -> serve transient -> distributed motion -> motion/audio collapse
```

Most feature columns are converted to percentile ranks within the selected game window.
This reduces sensitivity to recording-specific exposure, camera distance, background
noise, and absolute motion magnitude. Features whose absolute values are meaningful, such
as visibility and camera-response quality, retain their absolute values.

The canonical feature list is defined in
[`prod/src/lib/on-device/feature-schema.ts`](prod/src/lib/on-device/feature-schema.ts),
and contextual normalization is implemented in
[`prod/src/lib/on-device/feature-math.ts`](prod/src/lib/on-device/feature-math.ts).

The production-browser serving-side path is a separate candidate-conditioned branch:

```text
Saved rally/serve anchor and ROI
             |
             |-- 8-frame court-flow window -> 82 values
             `-- 9-frame 192x108 affine-compensated 4x6 flight window -> 155 values
                                      |
                  tied percentile ranks within the recording
                                      |
                         237 ordered visual inputs
                                      |
                  fixed-flight v3 logistic side score
```

The 82-value bank describes near/far residual flow in pre/contact/post phases and their
phase/side differences. The 155-value bank describes grid energy, vertical flow,
centroid/spread, entropy, connected-component concentration, direction/divergence, and
phase-to-phase trajectory deltas. It does not use audio or human visibility annotations.
The exact offsets, formulas, names, ranking rule, and feature-count expansion are normative
in [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md). All 237 visual columns and their within-
recording ranks are separate from F104/520, while serve-head scores and ensemble rally
agreement are reused gate evidence rather than side-model inputs.

## Three-head rally-model bundles

Each of the two production rally-model bundles contains three independent, class-weighted
logistic-regression heads. Every head has a bias and 520 learned feature weights:

```text
score = sigmoid(bias + sum(standardized_feature[i] * weight[i]))
```

There are no hidden layers, convolutions, recurrent layers, attention layers, or
transformers. The small linear architecture is intentional: video and audio decoding and
feature extraction dominate runtime, while the classifiers themselves are inexpensive on
mobile CPUs.

The three heads have different training targets and responsibilities:

| Head | Learned target | Product responsibility |
| --- | --- | --- |
| Rally | Whether the sample belongs to live play | Produces primary and permissive candidate intervals |
| Serve | Whether the sample is near labeled serve-ball contact | Anchors starts and helps rescue short rallies |
| Dead state | Whether the sample represents the transition out of live play | Refines proposed rally ends |

Raw sigmoid values are useful evidence but are not guaranteed to be calibrated
probabilities. The temporal decoder and resulting interval confidence must not be
interpreted as exact semantic certainty.

## Temporal decoding

The classifier heads emit one score per 250-millisecond sample. The production decoder
turns those scores into candidate intervals:

1. Smooth rally evidence over 2 seconds.
2. Enter live play at a `0.50` rally score and remain live until it falls below `0.45`.
3. Bridge internal gaps of up to 1.5 seconds.
4. Require ordinary live runs to last at least 3 seconds.
5. Retain very short runs when they are at least 0.25 seconds long and peak at `0.90`.
6. Decode serve peaks at `0.85`, separated by at least 10 seconds.
7. Use serve evidence to anchor or rescue an associated permissive rally candidate.
8. Use high-confidence dead-state evidence to refine the proposed end within the configured
   end window; if no qualifying transition exists, leave the end unchanged.

The exact cross-platform implementation is in
[`prod/src/lib/on-device/model.ts`](prod/src/lib/on-device/model.ts) and
[`android/app/src/main/java/com/volleycut/nativeanalysis/ModelRunner.java`](android/app/src/main/java/com/volleycut/nativeanalysis/ModelRunner.java).

## Production recall-safety ensemble

Production runs both three-head bundles over the same extracted feature matrix:

| Model | Role |
| --- | --- |
| `model-1ca43e38eefc` | All-labels v2; refit on six grass and five indoor recordings, including newer walking and ball-retrieval hard negatives |
| `model-9c92b8e9333f` | Previous production; retained because it still finds some rallies missed by the newer fit |

The two models have the same architecture, inputs, and decoder structure, but different
learned weights. Production unions their overlap-connected candidate ranges:

- a component containing predictions from both models is marked `both-models`;
- a component found only by all-labels v2 is marked `all-labels-v2-only`;
- a component found only by previous production is marked `previous-production-only`; and
- one-model-only detections remain included, but their displayed confidence is reduced and
  capped at `0.49` so the editor surfaces them for review.

Disagreement therefore does not automatically remove footage. This design favors recall
and uses model disagreement as review information. The merge contract is implemented in
[`prod/src/lib/on-device/ensemble.ts`](prod/src/lib/on-device/ensemble.ts).

## False-positive suppression specialist

Production also runs the single-head
`suppression-overlap-exclusion-retrained` classifier over the same 520 inputs. Its positive
class is valid non-rally time selected by the production ensemble, together with explicitly
reviewed false positives, hard negatives, and side-switch samples. Human rally samples form
its negative class.

The specialist learns to suggest vetoes for patterns such as:

- ball retrieval or tossing during setup;
- players walking or transitioning between points;
- side switches; and
- other reviewed dead-time patterns mistakenly exported as play.

Its evidence is smoothed over 1 second and decoded with `0.75`/`0.65` enter/exit
thresholds, a 0.5-second minimum duration, and a 0.5-second bridge.

Suppression is protected by a one-model-only eligibility gate. It may suggest cuts only
inside agreement components supported by exactly one of the two rally models. Components
supported by both models are protected in full. Removing this gate improved aggregate
precision in experiments but deleted too much real rally coverage.

The editor exposes four policies:

| Product policy | Eligibility construction | Effect |
| --- | --- | --- |
| No suppression | Suppression is not applied | Default and safest |
| Conservative | Build agreement components with 2.0-second padding and joins below 0.5 seconds | Protects the most nearby cross-model support |
| Balanced | Build agreement components with 1.5-second padding and joins below 0.5 seconds | Intermediate option |
| Aggressive | Use raw overlap-connected agreement components | Makes the most one-model-only time eligible |

Suggestions remain reviewable and individually overridable. The default scope for an
accepted suggestion is the whole affected rally, with a veto-region option available.
Suppression is therefore an optional review-assisted veto, not an unconditional fourth
rally vote.

The specialist runner and policy are implemented in
[`prod/src/lib/on-device/suppression-model.ts`](prod/src/lib/on-device/suppression-model.ts),
[`prod/src/lib/on-device/suppression-policy.ts`](prod/src/lib/on-device/suppression-policy.ts),
[`android/app/src/main/java/com/volleycut/nativeanalysis/SuppressionModelRunner.java`](android/app/src/main/java/com/volleycut/nativeanalysis/SuppressionModelRunner.java),
and
[`android/app/src/main/java/com/volleycut/nativeanalysis/SuppressionPolicyEngine.java`](android/app/src/main/java/com/volleycut/nativeanalysis/SuppressionPolicyEngine.java).

## Production-browser serving-side model and hybrid gate

`serving-side-fixed-flight-v3` replaces the earlier 38-input v1 research baseline in the
production browser. It is candidate-conditioned: at a known source-aligned anchor,
it maps the 237 tied within-recording ranks to a near-side score:

```text
filled[i] = feature[i], or the fitted median when missing
z[i]      = (filled[i] - fitted_mean[i]) / fitted_scale[i]
score     = sigmoid(clamp(bias + sum(z[i] * weight[i]), -30, 30))
side      = near when score >= 0.4783744762021848, otherwise far
```

The model has 237 weights, one bias, class-balanced fitting, and L2 `0.1`; it has no
hidden layers, tree stages, recurrence, attention, or direct player/ball detection. Its
fingerprint is
`85bc3325fbd43abba6ba3726ac091dc6a3a1eafc68d63d979cb2a7450eb49e06`.
Identity calibration is retained. Independently of the side decision threshold, scores
from `0.3121748736511044` through—but not including—`0.5028396703865513` are routed to
human review; lower scores are automatic far and higher scores are automatic near.

The side classifier always produces `near` or `far`. `not-serve` comes from the separate
`serving-side-hybrid-serve-gate-v2` composition:

```text
either production serve head >= 0.85 inside anchor +/- 1 second
    -> expose fixed-flight side; decision source = serve-head

otherwise, anchor contained in a production rally marked both-models
    -> expose fixed-flight side; decision source = production-rally-recovery;
       mandatory review

otherwise
    -> not-serve; decision source = none
```

The production rally intervals are the same overlap-connected union already decoded from
`model-1ca43e38eefc` and `model-9c92b8e9333f`. The fallback requires support from both
models; a one-model-only interval is insufficient. The gate does not alter intervals,
retrain a serve head, or use the side score to decide whether a serve occurred. Side-score
uncertainty and rally recovery are independent review reasons.

The selected development fit contains 1,027 correction-clean near/far rows across 28
recordings and nine source groups. Development leave-one-source-group-out performance is
94.12% source-group macro balanced accuracy and 94.05% pooled balanced accuracy. On the
current 1,114-row corrected-label results universe, the hybrid gate has 99.63% serve
precision and 98.01% serve recall, reducing missed serves from 56 to 22 while adding one
false serve. Those gate numbers are assisted post-hoc diagnostics, not a clean held-out
model-selection estimate.

The model, thresholds, source lineage, immutable NAS paths/hashes, and deployment status
are registered in [`MODELS.md`](MODELS.md). The browser runtime implementation is in
[`prod/src/lib/on-device/serving-side-model.ts`](prod/src/lib/on-device/serving-side-model.ts)
and [`prod/src/lib/on-device/serving-side.ts`](prod/src/lib/on-device/serving-side.ts).
The training/reference implementation is in
[`analysis/serving_side_v2.py`](analysis/serving_side_v2.py),
[`analysis/serving_side_flight.py`](analysis/serving_side_flight.py),
[`analysis/production_serve_gate.py`](analysis/production_serve_gate.py), and the matching
extraction/inference scripts. The historical v1 non-promotion remains documented in
[`serving-side-specialist-v1-2026-08-20.md`](docs/research/serving-side-specialist-v1-2026-08-20.md).

## Production-browser inference, scoring, and export flow

The complete selected production path is:

```text
Decode video and audio once
           |
Extract and cache 104 base features at 4 fps
           |
Normalize and gather 520 contextual inputs
           |
           |-- run all-labels-v2 rally/serve/dead-state heads
           `-- run previous-production rally/serve/dead-state heads
                           |
              union the two rally outputs
                           |
             mark agreement and disagreement
                           |
       run suppression head and build gated suggestions
                           |
             for each saved candidate anchor
                   |                    |
                   |                    `-- extract/rank 237 serving-side inputs
                   |                                   |
                   |                          fixed-flight near/far score
                   |
                   `-- inspect both serve heads and production-rally agreement
                                      |
                       hybrid gate: serve-head / rally recovery / not-serve
                                      |
                      add side-score and recovery review reasons
                           |
             seed editable source-time serve markers
                           |
       derive point winners from each following server
                           |
            human review and manual corrections
                           |
        add export padding and join short positive gaps
                           |
             preview or export MP4, optionally with score
```

New edit drafts default to:

- 2 seconds of padding before and after each retained core rally;
- merging overlapping or touching padded ranges;
- joining positive gaps shorter than 3 seconds; and
- suppression disabled.

Manual include/exclude decisions, boundary edits, ignored intervals, and selected
suppression overrides are materialized before preview and export.

Serving-side candidates are generated after the browser's rally/suppression analysis
result for a new project and before the editor is shown. Every included merged production
interval start is an anchor. The raw 237-column matrix, verdict evidence, and model
identity are stored with the project; tied ranks are recomputed over the complete
candidate set during inference. Editor ignored ranges and disabled/suppressed rallies
filter markers after inference and never cause video reprocessing.

If serving-side extraction alone fails, the browser reports the error and opens the
editor with the already-complete rally/suppression result; a connected source plus the
two retained serve outputs allows the user to rerun that stage. A failure is not stored
as a valid serving-side cache.

The score reducer treats the first visible serve as the initial server and awards no
point. Every later serve awards the preceding rally to the team that serves next, unless
that marker is `review` or is marked replay/ignored. Team 1 begins near and Team 2 begins
far; a side-switch marker flips that physical mapping for serve markers at or after the
switch. The final rally cannot be awarded without a later serve marker, so the user may
add one manually. This is a product inference convention, not a learned score model.

The production browser orchestration is in
[`prod/src/lib/on-device/production-inference.ts`](prod/src/lib/on-device/production-inference.ts)
and [`prod/src/App.tsx`](prod/src/App.tsx). Score reduction is in
[`prod/src/lib/score-tracking.ts`](prod/src/lib/score-tracking.ts), and optional MP4
composition is in [`prod/src/lib/score-overlay.ts`](prod/src/lib/score-overlay.ts) and
[`prod/src/lib/on-device/export.ts`](prod/src/lib/on-device/export.ts). The native Android
rally/suppression equivalent begins in
[`android/app/src/main/java/com/volleycut/nativeanalysis/AnalysisEngine.java`](android/app/src/main/java/com/volleycut/nativeanalysis/AnalysisEngine.java).
The Android serving-side, score editor, timeline markers, persistence, feedback, preview,
and export-overlay work remains to be implemented against the dedicated Android spec.

## Design philosophy and limitations

The production strategy is:

```text
two rally models maximize recall
              +
disagreement highlights uncertain material
              +
fixed-flight v3 estimates camera-space serving side
              +
the hybrid gate recovers dual-head serve misses only with both-model rally support
              +
optional gated suppression improves precision
              +
human review and padding protect the final export
```

The system understands audiovisual dynamics, not volleyball semantics. It does not reason
directly about the ball trajectory, legal contacts, possession, score, outcome, or rules.
Its predictions should remain editable suggestions, and production changes must continue
to be evaluated with source-separated data, recall guardrails, ignored-interval handling,
and the canonical [`F1_padP_coreR`](docs/model-ranking-metric.md) ranking contract.
