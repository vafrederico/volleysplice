# Side-switch v3 on-device plan — 2026-08-20

## Corrected dataset contract

Every dataset recording contains exactly one game/set, begins before the first point at
total point count 0, and uses a side-switch cadence of seven total points for both beach
and grass. There is therefore no unknown phase and no set-boundary detection problem.
Expected switch opportunities are deterministically after total points
`7, 14, 21, 28, 35, ...` until the recording ends.

Those are score milestones, not exact rally-marker indexes. A re-do does not increment
the score, an automated or human rally marker can be missed, and players can perform the
physical switch early or late. Cadence therefore proposes a bounded candidate region;
it must not manufacture a switch at the nominal gap.

`beach-source-02` is excluded from every new side-switch fit because the image
becomes blurry roughly halfway through the recording. Historical v1 artifacts remain
unchanged and continue to disclose that they used it. The recording may be retained as
a named blur/quality stress diagnostic, but it must not affect weights, preprocessing,
thresholds, decoder settings, or promotion.

The executable policy is in
[`side_switch_training_policy.py`](../../analysis/side_switch_training_policy.py).

## Existing supervision

The full-NAS manifest has 41 exact `sideSwitches` markers across nine gold recordings.
After the blur exclusion, 37 exact markers across eight recordings remain available for
development. The completed appearance review also has 88 switch-containing candidate
decisions across 22 recordings; removing the six positive windows from the blurry video
leaves 82 reviewed positive windows across 21 recordings.

Exact markers and reviewed windows must not be double-counted. Where both exist, exact
`sideSwitches` are the canonical event time/count labels and reviewed candidate decisions
are observability/candidate labels. Candidate-only raw recordings use the reviewed
switch-containing window as the available event supervision.

An initial alignment audit found that 31 of the 37 retained exact markers occur after a
labeled rally count divisible by seven. The six exceptions are concentrated in a few
recordings and should be audited for missed rally labels, duplicate/ambiguous switch
markers, or timing placement before changing the cadence contract.

## On-device architecture

V3 should lead with the deterministic cadence rather than a rare-event classifier:

1. Count accepted point-like intervals from the existing on-device rally output; when
   score/redo information exists, re-dos must not increment the point proxy.
2. Project each score milestone `7k` onto a rally-gap opportunity.
3. Search an inclusive default margin of `7k-2` through `7k+2` gaps to tolerate re-dos,
   missed/extra rally markers, and an early/late player switch. Treat the margin as a
   development parameter and report ±1 and ±2 sensitivity rather than learning it on
   confirmation data.
4. Rank those local gaps with cheap visual consistency evidence and choose at most one,
   with an explicit no-switch/insufficient-evidence result.
5. Re-anchor the next local search to a confidently observed switch when rally count has
   drifted, while retaining the official `7k` score milestone for audit.
6. Carry a selected switch forward as a parity toggle; never independently classify
   every inter-rally gap.

The first baseline must be cadence-only. The next comparison adds visual ranking. This
separates rally-count errors from appearance errors and establishes whether a learned
visual head is needed at all.

The visual path must reuse on-device-friendly inputs. Do not promote the v2 1,280-pixel
OpenCV HOG proposal path to browser or Android. Instead, derive sparse features from the
existing decoded/cached frame stream around only the expected opportunities:

- stable frames from the rally before and the pre-serve/early portion of the rally
  after the candidate, not players crossing during dead time;
- low-resolution HSV/chroma histograms and signed color moments over adaptive near/far
  ROI bands;
- before/after stability, visibility, blur, camera-shift, and missing-evidence gates
  already available or cheaply reducible on device;
- per-recording unlabeled normalization over the complete game; and
- a small fixed linear head or compact rule that can be implemented identically in
  TypeScript and Java.

Because one set usually exposes only a handful of switches, cadence supplies the event
rate and visual evidence supplies local placement/confidence. Class oversampling alone
must not be treated as the solution.

## Evaluation contract

Freeze recording/source-group-separated development and confirmation scopes before v3
selection. The already-opened v2 confirmation recordings may join v3 development, but
new untouched recordings are required for a promotion claim.

Report separately:

- cadence-only end-to-end switch event precision, recall, and F1;
- cadence-plus-visual local-ranking metrics;
- exact switch-count accuracy per recording;
- rally-index error between predicted and exact markers;
- performance with and without ±1/±2 rally tolerance;
- redo, missing-marker, wrong-marker, and early/late-player-switch failure slices;
- review candidates shown per true switch;
- blur/visibility and missed-rally failure slices; and
- browser/Android feature and decision parity plus incremental runtime/memory cost.

The model remains research-only until the cadence baseline, visual uplift, and on-device
parity all pass on the new confirmation scope.
