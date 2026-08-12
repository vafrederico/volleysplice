# Future feature experiment backlog — 2026-08-12

## Purpose

This note records feature ideas for future VolleyCut rally-detection experiments. It is a
prioritized hypothesis backlog, not evidence that the features work and not a production decision.
Every candidate still requires source-group-held-out ablation.

The strongest near-term direction is a structured transition model built from multiscale,
court-aware player-motion features. This is more promising than adding another flat sensor channel:
formation geometry and residual player motion are the most consistently useful recent feature
families, while raw serve-score stacking, generic short-event thresholds, and additional
frequency-band audio have already produced mixed or negative downstream results.

Ball presence remains a separate experiment in progress. If that detector becomes reliable, its
highest-value extension will be trajectory and interaction features rather than presence alone.

## Current substrate

The frozen corpus contains 345 rallies, including 39 aces, 48 service faults, and 73 rallies at most
three seconds long. The current audiovisual model extracts 90 base signals at 4 fps and 192x108,
then samples them at `-2`, `-1`, `0`, `+1`, and `+2` seconds to produce 450 inputs for a weighted
linear classifier. The feature cache retains aligned timestamps, raw values, feature names, and
video metadata, so many derived-feature experiments can reuse warm caches without decoding the
videos again.

The current feature families are:

- appearance, frame difference, and optical flow;
- camera and visibility quality;
- camera-compensated residual player motion;
- coarse formation geometry;
- audio level, onset, transient, and cadence cues; and
- optional noise-normalized and frequency-band audio.

The raw footage is substantially richer than the current representation: every recording has
stereo audio and high-frame-rate full-court video, with seven 4K and two 1080p sources. Existing
manifest metadata includes a rectangular court ROI, stationary end-line capture information,
environment, players per team, game format, and side-switch markers.

The main observed failure is no longer simply finding serve contact. For most remaining missed
short rallies, the serve is found but the resulting interval closes too late. In the prior miss
audit, six of eight failed short events had a serve contact within one second and four covered the
entire truth interval but were too long to reach strict IoU 0.5. This makes post-serve state and
end-boundary evidence the highest-value targets.

## Priority 0: experiments using existing cached data

### 1. Multiscale transition features

Replace isolated temporal samples with summaries over approximately 0.5, 1, 2, 4, and 8 seconds.
For player motion, formation geometry, frame difference, optical flow, audio cadence, and quality,
derive:

- rolling mean, maximum, variance, and quantiles;
- signed slope and acceleration;
- pre-window versus post-window differences;
- thresholded run length and persistence;
- time since motion onset or collapse; and
- short-versus-long-window disagreement.

Start-boundary hypotheses include a setup lull followed by a rapid receiving-side reaction.
End-boundary hypotheses include sustained live motion or contact cadence followed by persistent
stand-down, walking, or silence. Persistence matters because a momentarily quiet rally should not
look like a completed point.

### 2. Explicit cross-modal interactions

The current linear classifier cannot learn conditional relationships unless their products or gates
are supplied explicitly. Test a small, predeclared interaction bank:

- serve peak x receiving-side motion onset;
- audio transient x coherent court-directed optical flow;
- terminal transient x motion collapse;
- cadence collapse x formation contraction;
- visibility quality x residual player motion;
- camera shift x frame difference; and
- recent serve evidence x elapsed-live duration x dead-state evidence.

Test the bank as a feature family and ablate individual interactions. Avoid an unconstrained product
search, which would overfit the four available development source groups.

### 3. Court-relative directional motion

Collapse the existing generic grids into volleyball-specific regions and asymmetries:

- near court versus far court;
- serving zone versus receiving half;
- court interior versus margins and off-court space;
- motion along versus across the court axis;
- near-side versus far-side reaction onset and lag;
- side-specific motion centroid, spread, and contraction;
- migration into setup positions or away from active play; and
- huddle, retrieval, and walking patterns after a point.

A rough version can use the existing ROI and fixed end-line orientation. A later homography-backed
version should use manually identified court geometry.

### 4. Serve-anchored multistate short-event model

Test an explicit sequence rather than another flat live/dead score:

```text
DEAD -> SETUP -> SERVE -> LIVE -> DEAD
                  `-----------> DEAD  (immediate result)
```

Candidate inputs are source-group-cross-fitted serve scores, multiscale motion and formation
features, local end-transition evidence, elapsed time since serve, and persistent post-event cues.
Compare a small semi-Markov model, temporal convolutional network, or GRU against a same-feature
linear control.

This experiment must be able to open a missing short event as well as close it. That distinguishes it
from the existing local endpoint refiner, which can only move the end of an interval already found by
the upstream rally/serve pair.

Use duration priors only when estimated inside each training fold. Gold duration, outcome, or the
next annotated boundary must never become an inference feature.

### 5. Outcome auxiliary head

Use the existing `ace` and `service-fault` tags as supervision rather than model inputs. A compact
post-serve auxiliary target could be:

- ordinary continuation;
- immediate result; or
- uncertain/unobservable.

The decoder could use that output to choose between the short-event and ordinary-live paths. This
may teach the representation that a 1-2 second event is legitimate without requiring complete touch
annotations or forcing the main head to distinguish ace from service fault.

### 6. Out-of-fold component selector

Construct overlap components from the existing heuristic, rally, serve, normalized-audio, and local
end-refiner candidates. Candidate selector features include:

- overlap topology and component cardinality;
- each model's separate raw rally and serve scores;
- start and end disagreement;
- candidate duration and relative containment;
- transition evidence before and after each proposed boundary;
- visibility, audio availability, and camera quality; and
- provenance of every candidate boundary.

Train only on source-group-out-of-fold component predictions. Compare against the frozen
conservative v4/v5 intersection rule, which already improved retrospective end MAE from 1.424 to
1.124 seconds without increasing prediction count.

## Priority 1: richer representations from existing footage

### 7. Frozen higher-resolution visual embeddings

Extract frozen pretrained embeddings from higher-resolution crops of:

- the complete court;
- serving zone;
- receiving half;
- net region; and
- optional player-centered regions.

Use separate short 1-2 second and long 6-10 second temporal branches. First feed frozen embeddings
to a small classifier or temporal head alongside the hand-engineered features. Do not fine-tune a
large visual backbone until the dataset contains more independent source groups.

The hypothesis is semantic: higher-resolution features may distinguish active defensive posture
from walking, retrieval, timeout, and celebration, which coarse optical flow cannot reliably do.

### 8. Stereo and cadence-sequence audio

Before collecting new microphones, retain the existing stereo channels and derive:

- left/right level difference and inter-channel coherence;
- compact log-mel transient embeddings;
- the last several inter-transient intervals;
- cadence slope, regularity, and collapse persistence;
- whistle-like tonal likelihood; and
- agreement between audio events and localized court motion.

The goal is supporting evidence and adjacent-court rejection, not audio-only rally detection. Prior
audio-only and additional frequency-band experiments did not improve downstream intervals.

### 9. Side-switch-aware spatial normalization

Use the 26 existing side-switch markers to canonicalize near/far team orientation when directional
features are tested. Treat the marker only as recording context available at inference if the product
can obtain it; otherwise compare with an orientation-invariant representation.

Do not treat environment, game format, or players per team as predictive inputs on the current
corpus. Environment and player count are confounded: every beach/grass recording is 2v2 and every
indoor recording is 4v4. Keep them as reporting slices until new crossed source groups exist.

## High-value data to add

### 10. Court geometry and semantic zones

Collect four court corners plus the net and service-zone anchors once per recording. Four to eight
clicks per video would unlock:

- a playable-court polygon and exclusion mask;
- court homography and normalized court coordinates;
- near/far and left/right semantic zones;
- direction normalized for camera perspective;
- server-zone occupancy;
- better adjacent-court and spectator rejection; and
- geometry-aware player and ball trajectories.

This is likely the highest return per annotation minute. Compare rectangle-only, polygon-only, and
homography-derived feature families.

### 11. Explicit dead-time and hard-negative states

The frozen full-video labels contain only two hard-negative intervals, both timeouts, despite the
schema already supporting:

- setup between points;
- walking or ball retrieval;
- celebration or huddle;
- timeout;
- warmup;
- adjacent-court play;
- foreground crossing; and
- camera motion or disruption.

Label several examples per recording, including random dead-time controls as well as model false
positives. Use them for an auxiliary state head, targeted weighting, and a dedicated failure suite.
Do not evaluate an error-mined feature on the same sources used to choose those errors.

### 12. Minimal transition annotations

On a stratified subset of ordinary rallies, aces, service faults, and uncertain endpoints, add:

- first coordinated receiver reaction;
- first collective stand-down;
- terminal cue: ball down/out, whistle/stoppage, no recovery, or unobservable;
- end observability and confidence; and
- verified `immediate-result` status.

These labels directly supervise the weakest boundaries without the cost of annotating every touch.
Keep start and end confidence separate.

### 13. Sparse anonymous player tracklets

Around serve and rally-end windows, generate person or pose proposals and have a reviewer correct:

- anonymous player boxes or foot points;
- team/court side;
- short within-window track identity; and
- coarse state such as ready, playing, jumping, stand-down, or walking.

This would enable player count, pairwise distance, formation symmetry, reaction synchrony, side
occupancy, pose relaxation, and movement toward or away from the court. Evaluate the player detector
against independent truth before using generated sidecars downstream.

### 14. Capture metadata

For new recordings, retain:

- camera height and approximate distance behind the end line;
- device and lens/zoom mode;
- resolution, frame rate, and stabilization mode;
- venue, court, session, and lighting;
- whether framing changed; and
- whether the complete service areas and margins remained visible.

Initially use these as domain and failure-analysis slices. Promote them to normalization or model
conditioning only after enough independent groups exist to prevent memorizing a venue or device.

### 15. Optional synchronized scorekeeper event log

A simple tap interface or scoring-app feed could record approximate serve, point-end, and outcome
timestamps. This would be a powerful optional anchor, but it must be evaluated as a separate product
mode with missing-feed and clock-offset tests. It cannot be used to claim video-only performance.
Scoreboard OCR is lower priority unless the scoreboard is consistently visible and stable.

### 16. Lower-priority additional sensors

- Phone gyro/IMU could identify camera bumps, but the stationary capture contract and existing
  camera compensation make it less urgent.
- A directional or isolated court microphone could reduce adjacent-court contamination, but existing
  stereo features should be exhausted first.
- Multiple cameras could improve 3D ball and player tracking, but would materially change the product
  and capture contract and are outside the current cutter's near-term scope.

## Ball-dependent follow-up features

The round-01 ball pack contains 2,160 reviewed frames across 48 three-second windows at 15 fps. It
records ball state, normalized boxes, primary/other-court role, visibility, truncation, and
track-like IDs within a window. This is a deliberately phase-balanced 144-second sample, not
continuous full-video coverage or an unbiased prevalence sample.

The original generic detector did not pass its development gate: out-of-fold primary precision was
0.790, recall was 0.390, one source group had zero recall, and beach recall was zero. Its threshold is
diagnostic rather than frozen. The five-view tiled small-ball mode is a current attempt to improve
recall and has no quality claim until evaluated on the same independent truth.

If a detector passes the frame-quality, false-track, and throughput gates, derive more structured
features than the already implemented presence aggregates:

- track duration, observation fraction, and gap pattern;
- velocity, acceleration, curvature, and direction changes;
- trajectory smoothness and ballistic consistency;
- court-half and net-crossing transitions;
- flying versus held, grounded, or retrieved probability;
- disappearance followed by coordinated receiver reaction;
- distance and approach rate to anonymous player tracks;
- terminal trajectory followed by collective stand-down;
- role-consistent primary-court track confidence; and
- disagreement between visual tracking and audio contact cadence.

Before full-video inference, run an oracle experiment on human boxes to answer whether trajectory
features separate serve, live, end-transition, and dead-time strata at all. This tests feature value
independently of detector failure.

## Suggested experiment order

1. Add multiscale transitions and a small interaction bank from warm feature caches.
2. Add rough court-relative directional summaries using the existing ROI.
3. Test a serve-anchored multistate short-event model against a same-feature linear control.
4. Train an out-of-fold component selector and compare it with the frozen intersection rule.
5. Annotate court geometry, hard-negative states, and minimal transition cues.
6. Test frozen higher-resolution embeddings and sparse player tracklets.
7. If the ball detector passes its gates, run oracle and detected trajectory-feature experiments.

## Evaluation and leakage guardrails

- Select features and decoders with nested leave-one-`sourceGroup`-out development evaluation.
- Cross-fit every generated input, including serve, end, player, ball, and component-model scores.
- Report event F1 at multiple IoUs, time IoU, live precision/recall, dead seconds retained,
  start/end error, exact count, and short/ace/fault/ordinary-long slices.
- For the multistate short-event experiment, preserve ordinary-long strict recall, improve short and
  service-fault recall, and allow no more than a one-percentage-point live-recall loss.
- Treat classifier sigmoid outputs as uncalibrated scores unless calibration is fit out of fold.
- Keep gold outcome, duration, side switches, annotator confidence, AI provenance, and later human
  corrections out of inference inputs unless the same signal is genuinely available in production.
- Document centered or future-looking features as offline and non-causal.
- Aggregate evidence by recording and source group; adjacent frames are not independent samples.
- Freeze hypotheses before evaluating fresh sources. The existing indoor test has been inspected
  repeatedly and is a regression source, not an unbiased final test.
- Require untouched indoor, grass, and beach source groups before any new promotion decision.

There are currently only five independent source groups: four development groups and one repeatedly
inspected indoor test group. Beach has no held-out source, validation is grass-only, and environment
is entangled with player count. This data limitation is more important than small metric differences
between feature variants.

## Related evidence

- [Audiovisual feature ablation](./audiovisual-feature-ablation-2026-08-11.md)
- [Rally model with serve-probability feature](./rally-with-serve-feature-experiment-2026-08-11.md)
- [Serve evidence gate and peak-window experiment](./serve-evidence-gate-and-peak-window-2026-08-11.md)
- [Noise-normalized audio experiment](./noise-normalized-audio-serve-experiment-2026-08-11.md)
- [Dual serve-model fusion](./dual-serve-v4-v5-fusion-experiment-2026-08-11.md)
- [End-boundary and dead-state experiment](./end-and-dead-state-audio-experiment-2026-08-12.md)
- [Minimum ball-presence pilot](./minimum-ball-presence-pilot-2026-08-11.md)
- [Full-corpus training](./full-corpus-training-2026-08-10.md)

