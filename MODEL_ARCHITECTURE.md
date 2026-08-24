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
the production browser—the physical camera-space serving side and team side changes.

The currently deployed rally/export system contains seven logistic classifiers:

- two independent three-head rally-model bundles, used as a recall-safety ensemble; and
- one false-positive suppression head, used as an optional, review-first veto.

The production browser always adds one class-balanced logistic serving-side classifier.
Its hybrid serve gate is deterministic and reuses the two existing serve heads and their
decoded rally intervals. An optional, default-off score-tracking branch adds the
`union34-top2-x2` side-switch logistic classifier. The hybrid gate and side-switch
post-decoder are deterministic composition, not learned heads. The browser therefore
runs eight learned classifiers normally and nine when side-switch inference is enabled.

The seven rally/suppression classifiers run locally in both the production browser and
native Android applications. Those clients share the same F104/520 model artifacts,
temporal decoders, ensemble rules, and suppression-policy contract. Serving-side fixed-
flight v3 and its hybrid gate are additionally deployed in the production browser and
implemented on Android with physical-device release gates still pending. The side-switch
classifier and its checked-in runtime are browser-only. The cross-platform serving-side
contract is specified in
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

The optional production-browser side-switch path is another candidate-conditioned branch:

```text
Production range boundaries + retained internal dead-state peaks
                              |
       two seven-frame 256x144 comparison windows -> V5 visual22
                              |
            both production bundles -> production state10
                              |
                 candidate kind + generator score2
                              |
                    34 ordered runtime inputs
                              |
        union34 logistic score -> adjacent NMS -> post-six cost
```

Serving-side and side-switch timestamp schedules are unioned and decoded in one full
sequential specialist pass. Requested source samples are rendered independently to each
model's frozen pixel format, then feature extraction, ranking, classification, and
post-decoding proceed separately. This sharing changes execution cost only; neither
feature signature, model artifact, cache identity, nor inference output contract changes.

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

## Side-switch specialist lineage and production beta

Side-switch classification is a separate candidate-marker pipeline and is not part of
the production rally ensemble itself. The historical `side-switch-specialist-v2`
applies a class-balanced
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
V6 is not promoted, has no production port, and v5 remains the strongest base
appearance specialist; the later V5-state-based peak+soft-count cleanup is the current
decoder winner. These event metrics are candidate-window agreement rather than exhaustive
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

The cleanup follow-up keeps that frozen probability stream and compares eight
cadence-free post-decoders. Score-ranked adjacent/time NMS removes local duplicates;
a soft count prior adds increasing logit cost only after six outputs and never imposes
a hard cap. A separate 19-input head summarizes production rally/dead/serve context,
excluding raw gap duration, and contributes soft log odds rather than eligibility.
Local peaks transfer to raw-phone data, while the production-context validation gain
does not: the historically selected peak+context variant cuts proposals 91→60 but
leaves exact F1 flat at 44.21%. After the later exhaustive review, the user designated
the locked peak+soft-count mechanism as the research winner at that time. It uses
adjacent gap suppression plus a `0.25` post-six logit penalty, has no production-context
weight, and reaches 44.64% end-to-end pooled F1 with 62 proposals. It was superseded by the
full-union hard-negative winner on 2026-08-23; its historical artifact remains frozen.
See
[`side-switch-v5-peak-cleanup-2026-08-20.md`](docs/research/side-switch-v5-peak-cleanup-2026-08-20.md).

The later continuous full-video review exposes a larger architectural bottleneck. Only
33 of 50 confirmed raw-phone switches lie inside any modern candidate gap even after a
four-second boundary allowance. No decoder over the existing gap stream can exceed 66%
end-to-end recall on this scope. The no-cadence head recovers 28 events; its remaining
22 misses split into 17 upstream candidate misses and five decoder misses inside the
available universe. Local peak plus soft count was the baseline for that successor,
but candidate generation still needs to become independent of the production
rally intervals; decoder cleanup alone cannot recover those 17 events. See
[`side-switch-full-video-marker-audit-2026-08-21.md`](docs/research/side-switch-full-video-marker-audit-2026-08-21.md).

The full-trace successor broadens the internal universe to every adjacent production
range boundary plus strong dead-state peaks inside overlong ranges. Its selected fixed
generator yields 624 boundaries and 80 internal peaks, covering 46/50 markers at the
declared four-second allowance. Boundaries reuse whole-rally V5 summaries. An internal
candidate at `t` compares fixed three-second flanks `[t-4,t-1]` and `[t+1,t+4]` inside
the same range. Replaying the two shipped bundles adds the existing state inputs; all
42 stored values reproduce exactly on the 352 legacy rows.

`side-switch-full-union-ranker-v1` compares class-balanced 32-input V5+STATE10 and
34-input union-native logistic heads. Nested recording LOO selects L2, threshold,
local suppression, and soft count without using the outer video. The pooled held-out
result reaches 50.94% F1 at the four-second allowance with 27 TP, 29 FP, 23 FN, and 56
proposals, improving the current winner while remaining opened-development evidence.
Its decoder uses adjacent-candidate suppression and usually a 0.5 post-six logit
penalty; it has no cadence, re-anchoring, or hard cap. A refitted continuity veto
regresses to 49.52% F1 and is rejected. No production graph or client runtime changes.
See
[`side-switch-full-union-ranker-2026-08-23.md`](docs/research/side-switch-full-union-ranker-2026-08-23.md).

The imbalance follow-up holds that candidate/features/decoder path fixed and varies
only the training prior and label-free recording score normalization. Square-root class
balancing is the useful direction: nested selection removes two false positives at
unchanged true positives, moving F1 from 50.94% to 51.92%. Robust-logit and percentile
recording transforms are rejected. A symmetric head trained to predict no-switch is
numerically just `1 - P(switch)` and adds no independent evidence. The architecture
therefore remains one binary head; future negative modeling needs a genuinely distinct
continuity/hard-negative target. See
[`side-switch-imbalance-calibration-2026-08-23.md`](docs/research/side-switch-imbalance-calibration-2026-08-23.md).

The next diagnostic keeps square-root balancing and the same fixed decoder, then applies
a selected soft logit penalty only to internal dead-state-peak candidates. Nested ±4
F1 rises from a matched 53.47% zero-penalty control to 54.90%, but six of eleven folds
select no penalty and the penalized path suppresses the control's only correct internal
proposal. The offset is therefore rejected as a general architectural rule. Candidate
kind remains available as a feature; no hard type gate or runtime branch is added. See
[`side-switch-internal-peak-penalty-2026-08-23.md`](docs/research/side-switch-internal-peak-penalty-2026-08-23.md).

The expanded-candidate diagnostic lowers the internal dead-state threshold from 0.98
to 0.80 and peak separation from 14 to 10 seconds. This raises the internal universe
from 80 to 228 and opened-scope candidate recall from 92% to 100%, but the unchanged
32/34-input nested ranker falls to 42.74% ±4 F1. It selects 13 internal proposals with
one TP. The 704-candidate architecture is therefore retained; the next internal path
needs a distinct representation or head, not a lower global candidate threshold. See
[`side-switch-expanded-internal-candidates-2026-08-23.md`](docs/research/side-switch-expanded-internal-candidates-2026-08-23.md).

The hard-negative follow-up leaves the retained 704-candidate inference graph
unchanged. During training it fits an initial square-root-weighted head, upweights the
highest-scoring labeled negatives independently per recording, and refits. Nested
variant selection reaches 54.00% ±4 F1; the fixed union34/top-2/2× variant reaches
56.86% and is now the explicit research winner. The exported artifact is still one
34-input linear head, so mining adds no on-device operation. That selected head is now
ported to the production browser as a score-tracking beta; its training and selection
evidence remain research artifacts. See
[`side-switch-hard-negative-winner-promotion-2026-08-23.md`](docs/research/side-switch-hard-negative-winner-promotion-2026-08-23.md).

### Selected side-switch research-winner execution contract

The selected graph is fixed independently of later rejected experiments:

```text
existing production range union + 4 Hz rally/dead-state traces
                              |
       624 adjacent boundaries + deadState>=0.98 internal peaks
                              |
        seven 256x144 frames in each of two candidate windows
                              |
          V5 visual22 + production state10 + candidate metadata2
                              |
        stored impute/mean/scale -> 34-input logistic classifier
                              |
 threshold 0.3988497395 -> candidate-index NMS -> post-six logit cost
                              |
                   team-side switch proposals
```

Adjacent boundaries compare the entire decoded range before and after the gap. Internal
peaks compare `[t-4,t-1]` with `[t+1,t+4]` inside the containing range. The visual path
uses recording-level net calibration, court normalization, frame translation alignment,
motion-weighted HSV side palettes, and motion-component player proposals. The ten state
inputs reduce both production bundles' range support plus rally/dead scores over the
candidate gap. Candidate kind and generator score complete the ordered vector.

The decoder sorts by score, keeps candidate ordinals at least two apart, gives six
outputs no count cost, then subtracts `0.5` logits per additional selected output. It
has no cadence, no re-anchoring, no time-distance NMS, and no hard cap. Serve-anchor
features, suppression, recording reliability, the expanded union, internal specialist,
and boundary-only ablation are not part of this winner.

The exact browser implementation contract is
[`side-switch-current-research-winner-production-port-v1.json`](data/side-switch-current-research-winner-production-port-v1.json).
The production browser runs it after rally inference and seeds editable switch markers
when the default-off project option is enabled. Its frame schedule shares the one full
sequential specialist decode with serving-side inference; the 34-input extractor and
decoder remain independent after frame routing. Representative-device profiling and
independent exhaustive validation remain required before the score-tracking beta label is
removed.

The pairwise follow-up preserves that entire inference graph and adds only a training
loss over positive-minus-negative logits within each fit recording. Equal total pair
weight per recording prevents long games from dominating. The best row-AP variant adds
seven false event proposals at unchanged recall, and nested selection also regresses,
so the pairwise term is rejected. See
[`side-switch-pairwise-ranking-2026-08-23.md`](docs/research/side-switch-pairwise-ranking-2026-08-23.md).

The recording-reliability follow-up adds a ridge-predicted threshold-logit offset from
13 per-video quality and score-distribution summaries. It is nested by recording, but
only ten targets train each outer head and the available artifact lacks direct blur.
Every outer selector keeps zero offset; fixed nonzero heads trade one TP for at least
six FP. The layer is rejected. See
[`side-switch-recording-reliability-2026-08-23.md`](docs/research/side-switch-recording-reliability-2026-08-23.md).

The next training-only comparison swaps square-root BCE for exact focal loss or
effective-number class weights while preserving hard-negative mining. Neither improves
outer-held event F1, and the nested objective selector also regresses. The promoted
linear head remains unchanged. See
[`side-switch-rare-event-losses-2026-08-23.md`](docs/research/side-switch-rare-event-losses-2026-08-23.md).

The soft cadence follow-up propagates a latent count from set start with `+0/+1/+2`
redo/point/missed-point transitions and never re-anchors on predictions. Detected rally
ordinal is not a usable point counter, and both exact and uncertain hazards regress.
The architecture remains cadence-free. See
[`side-switch-soft-score-prior-2026-08-23.md`](docs/research/side-switch-soft-score-prior-2026-08-23.md).

The internal-specialist follow-up splits retained boundaries from 228 expanded internal
peaks and gives the latter a distinct transition/range/serve/peak head. Sparse support
prevents transfer: no held-out internal TP is added. Dropping the internal branch
entirely removes four FP and one TP and is retained as a small, unpromoted precision
candidate. See
[`side-switch-internal-specialist-2026-08-23.md`](docs/research/side-switch-internal-specialist-2026-08-23.md).

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
Decode main video at 4 fps and decode audio
                     |
       extract and cache 104 base features
                     |
      normalize and gather 520 contextual inputs
                     |
        |-- all-labels-v2 rally/serve/dead-state heads
        `-- previous-production rally/serve/dead-state heads
                               |
          union rally outputs and mark agreement
                               |
        suppression head and gated suggestions
                               |
       build required serving timestamps and, when enabled,
                 optional side-switch timestamps
                               |
          one full sequential specialist video pass
                    over the timestamp union
                 /                             \
  192x108 grayscale serving frames       256x144 BGR switch frames
                 |                             |
 extract/rank SERVSIDE237-FLIGHT      extract SIDE-SWITCH-UNION34-V1
                 |                             |
 fixed-flight near/far + hybrid gate    union34 score + post-decoder
                 |                             |
       editable serve markers             editable switch markers
                 \                             /
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

Score-specialist candidates are generated after the browser's rally/suppression analysis
result for a new project and before the editor is shown. Every included merged production
interval start is a serving-side anchor. Side-switch candidates are generated only when
the project's default-off side-switch option is enabled. The two requested timestamp sets
share one sequential decode, while their feature matrices and model outputs remain
separate. The raw matrices, verdict/proposal evidence, and model identities are stored
with the project. Editor ignored ranges and disabled/suppressed rallies filter markers
after inference and never cause video reprocessing.

If the shared specialist stage fails, the browser reports the error and opens the editor
with the already-complete rally/suppression result. A connected source plus the retained
production outputs allows the user to rerun only the missing specialist work. A failure
is not stored as a valid serving-side or side-switch cache.

The score reducer treats the first visible serve as the initial server and awards no
point. Every later serve awards the preceding rally to the team that serves next, unless
that marker is `review` or is marked replay/ignored. Team 1 begins near and Team 2 begins
far; a side-switch marker flips that physical mapping for serve markers at or after the
switch. The final rally cannot be awarded without a later serve marker, so the user may
add one manually. This is a product inference convention, not a learned score model.

The production browser orchestration is in
[`prod/src/lib/on-device/production-inference.ts`](prod/src/lib/on-device/production-inference.ts)
and [`prod/src/App.tsx`](prod/src/App.tsx). Shared specialist planning and sampling are in
[`prod/src/lib/on-device/score-specialists.ts`](prod/src/lib/on-device/score-specialists.ts)
and
[`prod/src/lib/on-device/specialist-frame-sampling.ts`](prod/src/lib/on-device/specialist-frame-sampling.ts).
Side-switch inference is in
[`prod/src/lib/on-device/side-switch-model.ts`](prod/src/lib/on-device/side-switch-model.ts)
and [`prod/src/lib/on-device/side-switch.ts`](prod/src/lib/on-device/side-switch.ts).
Score reduction is in
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
