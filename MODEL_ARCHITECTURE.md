# VolleyCut production model architecture

This document summarizes audiovisual feature extraction and specifies learned-model
inference, temporal model decoding, ensemble composition, and the suppression safety
policy. The normative feature formulas, version profiles, and production/research split
are in [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md). Trained artifacts, fitting sources,
and predecessor changes are registered in [`MODELS.md`](MODELS.md). This document does
not describe the broader editor, project-storage, playback, video-decoding, or
export-encoding architecture.

VolleyCut's production model is a lightweight, on-device audiovisual signal-processing
system. It does not explicitly detect the volleyball, players, net, score, or named
volleyball actions. Instead, it learns statistical patterns that distinguish live rallies,
serve contact, end-of-play transitions, and common false positives.

The production system contains seven logistic classifiers:

- two independent three-head rally-model bundles, used as a recall-safety ensemble; and
- one false-positive suppression head, used as an optional, review-first veto.

All inference runs locally in the production browser and native Android applications. The
two clients share the same model artifacts, feature signature, temporal decoders, ensemble
rules, and suppression-policy contract.

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

## Research-only side-switch specialist

Side-switch classification is a separate candidate-marker pipeline and is not part of
the production rally ensemble. `side-switch-specialist-v2` applies a class-balanced
logistic head to 34 inputs: 17 adaptive near/far and derived appearance scalars, each
paired with a missingness indicator. The extractor calibrates court depth without team
labels, first across the full candidate sequence and then locally with shrinkage, so
moderate zoom and camera-distance changes do not rely on one fixed pixel divider.

The optional temporal stage is a Viterbi decoder over rally order. Its state carries
near/far orientation parity and the previous switch position; candidate settings can
penalize close switches, reward orientation-consistent toggles, or add a switch prior.
Validation selected all of these settings as zero, so the frozen v2 decoder is an exact
no-op over static threshold decisions. It is retained in the artifact and reported
separately to preserve the negative decoder result.

V2 improves same-scope v1 confirmation ranking, but its 20.45% precision is not suitable
for automatic score tracking. It remains a research/review-ranking artifact. Indoor is
fixed to no-switch and excluded from specialist selection and metrics. Exact features,
lineage, and metrics are in [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md),
[`MODELS.md`](MODELS.md), and
[`side-switch-specialist-v2-2026-08-20.md`](docs/research/side-switch-specialist-v2-2026-08-20.md).

V3 implements local opportunity ranking with a detector-free 12-input linear head over
192×108 rally frames. Each recording is one set beginning at score zero; the decoder
starts from a seven-point opportunity, searches margins ±1 through ±4, uses monotonic
one-to-one assignment for overlapping ±4 windows, re-anchors after a selected switch,
permits no selection, and caps the set at six opportunities. It failed source-held-out
evaluation: selected exact-gap F1 is 7.59%, and the ±4 visual sensitivity is identical.
The implementation and artifacts remain research-only; no TypeScript/Java port or
production component was created. See
[`side-switch-specialist-v3-2026-08-20.md`](docs/research/side-switch-specialist-v3-2026-08-20.md).

V4 keeps that cadence decoder fixed and replaces only visual representation. It samples
seven 256×144 frames across each adjacent rally, calibrates net height from the first
seven rallies, normalizes the net to a stable vertical coordinate, compensates camera
translation, and compares broad/tight multi-frame side palettes. This improves
raw-phone exact-gap F1 from 7.59% to 16.67%, but remains far below automatic-use
requirements. It is research-only and has no TypeScript/Java port. See
[`side-switch-specialist-v4-2026-08-20.md`](docs/research/side-switch-specialist-v4-2026-08-20.md).

V5 isolates up to six player-like motion components per frame and forms near/far team
palettes from proposal-foot position. A whole-set decoder can compare every rally with
team-side anchors pooled from the first three score-zero rallies and carry orientation
parity across selected switches. Validation selected orientation weight zero, making the
state path a no-op, while player isolation improved raw-phone exact F1 to 27.85% and row
AP to 43.40%. Exact precision is still only 25.00%, so v5 remains research-only with no
production port. See
[`side-switch-specialist-v5-2026-08-20.md`](docs/research/side-switch-specialist-v5-2026-08-20.md).

V6 replaces those motion proposals with a pinned 3.48 MB block-int8 MediaPipe person
localizer. Three frames per rally run through four overlapping ownership tiles; pose
landmarks define torso palettes, and v4 net geometry maps hip positions to canonical
near/far sides. Team palettes start from the first three score-zero rallies and update
online only when localization, assignment, and side-separation quality agree. The full
29-input classifier and a separately fitted 26-input fixed-prototype ablation share the
same cadence decoder search. Validation again selects orientation weight zero. Although
raw-phone row AP improves from 43.40% to 45.47%, candidate-window exact F1 falls to
24.10% and ±2-tolerant F1 falls to 40.96%; both v6 variants select the same 48 events.
V6 is not promoted, has no production port, and v5 remains the best event-level research
artifact. These event metrics are candidate-window agreement rather than exhaustive
full-video accuracy; the 61-gap V5/V5-state/V6 proposal union is attached to stable
event IDs and rendered as three aligned recording timelines in the development review
UI. See
[`side-switch-specialist-v6-2026-08-20.md`](docs/research/side-switch-specialist-v6-2026-08-20.md).

The production-state follow-up reuses both shipped bundles' rally, serve, dead-state,
decoded-range, and agreement outputs after ordinary on-device analysis. A 20-scalar
bank can be appended to V5/V6, and an alternate appearance path samples around the
production serve anchor. Validation selects original V5 appearance plus ten soft
state/gating inputs, improving retrospective exact F1 to 30.14% while remaining far
below automatic-use quality. Hard agreement/serve gates and every V6 variant are
rejected. The suppression head is quarantined because it was trained with explicit
side-switch positives and overlaps all experiment roles. The selected V5-state outputs
are reviewable, but no shipped inference graph is changed. See
[`side-switch-production-state-experiment-2026-08-20.md`](docs/research/side-switch-production-state-experiment-2026-08-20.md).

The no-cadence follow-up holds those frozen V5/V5-state rows fixed, refits both linear
heads with exact learned-parameter parity, and replaces the re-anchored seven-point
path with independent thresholding of every reviewed gap. The validation-selected
V5-state variant retains all 11 cadence exact true positives and recovers 17 more,
raising retrospective exact recall/F1 from 31.43%/30.14% to 80.00%/44.44%. It also
raises proposals from 38 to 91 because it deliberately has no spacing, cluster
suppression, or count cap. This isolates a real cadence failure but is not a production
decoder; no shipped graph changes. See
[`side-switch-v5-no-cadence-2026-08-20.md`](docs/research/side-switch-v5-no-cadence-2026-08-20.md).

## Production inference and export flow

The complete production path is:

```text
Decode video and audio once
           |
Extract and cache 104 base features at 4 fps
           |
Normalize and gather 520 contextual inputs
           |
           |-- run all-labels-v2 rally/serve/dead-state heads
           |-- run previous-production rally/serve/dead-state heads
           `-- run the suppression head
                           |
              union the two rally outputs
                           |
             mark agreement and disagreement
                           |
          create optional gated suppression suggestions
                           |
            human review and manual corrections
                           |
        add export padding and join short positive gaps
                           |
                    preview or export MP4
```

New edit drafts default to:

- 2 seconds of padding before and after each retained core rally;
- merging overlapping or touching padded ranges;
- joining positive gaps shorter than 3 seconds; and
- suppression disabled.

Manual include/exclude decisions, boundary edits, ignored intervals, and selected
suppression overrides are materialized before preview and export.

The production browser orchestration is in
[`prod/src/lib/on-device/production-inference.ts`](prod/src/lib/on-device/production-inference.ts)
and [`prod/src/lib/on-device/pipeline.ts`](prod/src/lib/on-device/pipeline.ts). The native
Android equivalent begins in
[`android/app/src/main/java/com/volleycut/nativeanalysis/AnalysisEngine.java`](android/app/src/main/java/com/volleycut/nativeanalysis/AnalysisEngine.java).

## Design philosophy and limitations

The production strategy is:

```text
two rally models maximize recall
              +
disagreement highlights uncertain material
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
