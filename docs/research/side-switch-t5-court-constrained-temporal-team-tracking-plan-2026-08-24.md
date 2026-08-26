# Side-switch T5 court-constrained temporal team tracking — 2026-08-24

## Decision and isolation

T5 tests the tracking remediation identified by T4. It preserves T4's native source
frames, full-ROI near detector pass, selective far-court crop, pinned `224 x 224`
detector, score threshold, five endpoint fractions, jersey descriptor, skin/background
rejection, candidate rows, labels, model protocol, and transport equations.

Only temporal ownership changes. T3/T4 attempt to assign individual players between
frames before pooling tracklets into a team. T5 uses the stronger volleyball constraint:
inside one stable rally endpoint, every retained player on a court side belongs to the
same anonymous team. It therefore tracks one near-side and one far-side team state
through the five frames without claiming persistent player identities.

No new frame, optical flow, detector threshold, crop, interpolation, proxy resolution,
model label, candidate source, or decoder change is part of T5. T4 remains rejected and
is used only as the immutable engineering/model comparator.

## Frozen temporal team tracker

Reuse T4's maximum three selected jersey observations per side per frame. For each
non-empty side/frame:

1. set each observation weight to `max(confidence * support, 1e-6)`;
2. form a normalized weighted mean of the 64-value jersey descriptors;
3. set frame reliability to `mean(confidence) * sqrt(mean(support))`; and
4. retain the frame descriptor, reliability, and observation count.

For one court-side team track:

- require non-empty observations in at least two distinct frames; a one-frame team is
  unavailable and cannot enter transport;
- form the temporal descriptor as the normalized frame-reliability-weighted mean of
  all available frame descriptors;
- set reliability to `mean(available frame reliability) *
  (0.40 + 0.60 * available_frames / 5)`;
- set cohesion to one minus the frame-reliability-weighted mean Hellinger distance from
  each available frame descriptor to the temporal descriptor; and
- report available frames and observations explicitly in diagnostics.

The existing x court bounds, canonical-y court bounds, and near/far split at canonical
y `0.56` remain the ownership constraints. Team names remain anonymous and local to an
endpoint. The same team-transport cost and reliability gates compare before-near,
before-far, after-near, and after-far.

The T5 model values use a new prefix:

1. `courtTrackedFarJerseyTeamTransportSwapMargin`;
2. `courtTrackedFarJerseyReliableSwapEvidence`; and
3. `courtTrackedFarJerseyReliableContinuityEvidence`.

All diagnostics use `courtTrackedFarJersey`. No T3 or T4 value enters the T5 head.

## Immutable sources and label-free engineering gates

The T4 feature source is pinned to SHA-256
`443c0ded15cfddbb5e156f670c895375c8c43caede032a9b3ef5ae445ca4113a`.
Its model result is not loaded during extraction.

Output path:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t5-court-constrained-temporal-team-tracking-features-v1.json`.

T5 reaches labels only if every gate passes:

- exact preservation of 704 IDs/order and every T4 prior value;
- 624 eligible boundaries and 80 explicitly ineligible internal rows;
- at least 90% pooled endpoint any-team coverage;
- at least 79.17% pooled endpoint both-team coverage, a gain of at least five
  percentage points over T4's 74.17%;
- no recording below 35% endpoint both-team coverage;
- at least 63.33% four-team boundary visibility, a gain of at least five points over
  T4's 58.33%;
- reliable swap evidence nonzero on at least 20.79% of boundaries, a gain of at least
  three points over T4's 17.79%;
- all three core values finite and nonconstant;
- maximum absolute T5 core/core, T5 core/T4 core, and T5 core/existing-34 Spearman
  correlation below `0.98`;
- exactly 635 endpoint windows, 3,175 frame requests, and 25,400 detector tile calls;
- no frame/detector errors, wall time at most 60 minutes, and peak RSS at most
  768 MiB; and
- no audit, feedback, label, or model result loaded.

Failure stops before labels. Do not lower the two-frame team requirement, retune the
reliability equation, add observations, or fall back to T4 values after observing the
artifact.

## Frozen opened-development evaluation

If engineering passes, train the exact matched boundary T0 and exact `34 + T5 core`
head under the established nested recording-held-out protocol. Also load the immutable
T4 opened-development artifact with SHA-256
`16d617fb8427517b45c31196877d7482cb019f8902b989e5b2db19cfcef0020b`
for a fixed result comparison, not for fitting or threshold selection.

T5 must satisfy every T4 model guardrail: at least +2.0 percentage points pooled
`+/-4 s` F1 over T0; no more than two points lost in precision or recall; no more than
one point lost in strict F1; recovery of at least one T0 miss with nonzero T5 reliable
swap evidence; reduction of T0-selected false boundaries with zero T5 reliable swap
evidence; and recording robustness. In addition, T5 F1 must exceed immutable T4 F1 by
at least 0.5 point. This prevents a different representation from passing without
improving on the experiment that motivated it.

Report event counts, row AP/Brier, recording deltas, all three standardized
coefficients, rank among 37, 11-fold sign stability, and the diagnostic merge with
exact E0 internal selections. Do not run single-value pruning unless the full T5 bundle
passes every gate. Opened-development success cannot authorize promotion or a runtime
port without new recording-held gold.

## Execution ledger

### Preregistration — frozen 2026-08-24

No T5 module, extractor, feature artifact, feature value, model profile, or model result
existed when this contract was committed.

### Extraction — visibility gain, directional-reliability reject 2026-08-24

Implementation checkpoint: `167f7ac`.

The full label-free pass preserves every prior row/value and improves the intended
team-observation coverage, but fails the preregistered reliable-swap gate:

| Engineering metric | T4 | T5 | Requirement | Result |
| --- | ---: | ---: | ---: | --- |
| Endpoint any qualified team | 99.69% | 99.84% | >=90% | Pass |
| Endpoint both qualified teams | 74.17% | 81.10% | gain >=5 pp | Pass (+6.93 pp) |
| Weakest recording both teams | 32.69% | 42.31% | >=35% | Pass |
| Boundary four-team visibility | 58.33% | 68.75% | gain >=5 pp | Pass (+10.42 pp) |
| Reliable swap evidence nonzero | 17.79% | 16.19% | gain >=3 pp | **Fail (-1.60 pp)** |
| Maximum core/core absolute Spearman | — | 0.6100 | <0.98 | Pass |
| Maximum core/T4-core absolute Spearman | — | 0.7553 | <0.98 | Pass |
| Maximum core/existing absolute Spearman | — | 0.2801 | <0.98 | Pass |

All 704 rows, 624 boundaries, 80 internal rows, prior feature values, IDs, and order
remain exact. The run performs 3,175 frame requests and 25,400 detector tile calls
without errors in 1,551.543 seconds (25m51.543s), with 236.172 MiB peak RSS.

Coverage improves in every recording. The weakest recording `161923155` rises from
32.69% to 42.31%; `193307688` rises 65.15% to 80.30%; and `190429172` rises 70.37%
to 79.63%. Thus the rejected result is not caused by missing teams or one bad camera.

The component behavior identifies the failure. Mean minimum team reliability rises
from `0.0876` to `0.1579`, minimum cohesion rises from `0.3921` to `0.4487`, and
nonzero continuity evidence rises from 40.54% to 52.56%. However, mean between-team
separation falls from `0.3739` to `0.3196`, and positive raw swap margins fall from
23.56% to 19.87%. Pooling every retained jersey across a court side makes a stable
side-state descriptor, but blurs the team-specific appearance required for directional
transport. Plausible label-free mechanisms are the differently dressed libero,
within-side jersey outliers, and occasional retained non-player detections; the
artifact alone does not distinguish them causally.

Decision: **reject T5 at engineering and stop before labels**. No T5 model profile,
training, threshold selection, precision/recall result, coefficient importance,
ablation, merge diagnostic, or runtime port exists. The next representation, if
pursued, should retain T4's selective crop and T5's court-side temporal ownership but
replace unconditional averaging with a separately frozen robust dominant-jersey mode
(for example, a multi-frame medoid/consensus cluster with explicit outlier support).
Do not retune the T5 reliability gate or train the failed bundle.

Immutable artifact:

- path: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t5-court-constrained-temporal-team-tracking-features-v1.json`
- SHA-256: `25d8a23563f803058f23ba2da23df0e82283ab893afd143b5f39200d96fcb460`
- module SHA-256: `c415c3146b7a630b5d62812c85e6073a2ba09e071d641985003c5059b0ef0e0b`
- extractor SHA-256: `306fb046a2909cce1c5b954f7c00778d40f6d2c17197654252f3520860239739`

Validation after extraction: all 156 focused `test_side_switch*.py` tests pass.

### User-authorized diagnostic model override — 2026-08-25

After the engineering rejection was committed, the user explicitly requested training
to observe precision and recall. This authorizes one exact diagnostic `34 + T5 core`
run on the already opened 50-marker development scope. It does not revise the failed
engineering decision, make T5 selection-eligible, authorize pruning/tuning, or permit
promotion/runtime work.

Use the already frozen nested recording-held-out model protocol and report matched T0,
immutable T4, and T5. Preserve the originally frozen T5 model checks as descriptive
diagnostics. Commit the model runner before loading labels and record the result once;
do not adapt T5 after inspection.

### Diagnostic model result — reject confirmed 2026-08-25

Model-runner checkpoint: `d0dfdbd`.

| Held-out boundary metric | T0 | T4 | T5 |
| --- | ---: | ---: | ---: |
| Proposals | 52 | 56 | 57 |
| True positives | 26 | 28 | 28 |
| False positives | 26 | 28 | 29 |
| False negatives | 24 | 22 | 22 |
| Precision | 50.00% | 50.00% | 49.12% |
| Recall | 52.00% | 56.00% | 56.00% |
| F1 | 50.98% | 52.83% | 52.34% |
| Strict F1 | 41.18% | 41.51% | 42.99% |
| Row average precision | 0.4356 | 0.4446 | 0.4471 |
| Row Brier score | 0.05453 | 0.05431 | 0.05426 |

T5 gains two TP and three FP over T0. It improves recall by four points but loses
0.88 point of precision, producing only +1.36 F1 points—below the original +2-point
gate—and trails T4 F1 by 0.49 point. It selects none of the four T0 misses with
nonzero T5 reliable-swap evidence, although it reduces the selected zero-evidence
false-boundary slice from 21 to 18. Conflict recording `193307688` is unchanged.

The coefficients explain the mismatch. Raw transport direction remains useful and
stable (`+0.1459`, rank 14/37, positive 11/11 folds), while continuity remains a strong
stable veto (`-0.1725`, rank 11/37, negative 11/11). The supposedly reliable swap
value becomes a stable negative feature (`-0.0413`, rank 29/37, negative 11/11), the
opposite of its intended semantics. Thus unconditional team pooling creates confident
positive swap values that correlate with false rather than true switches.

Decision: **the authorized diagnostic confirms the T5 rejection**. No tuning,
ablation, promotion, or port follows. Robust dominant-jersey consensus must repair the
directional reliability before another selection-eligible model comparison.

Immutable diagnostic model artifact:

- path: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-feature-development-t5-court-tracking-diagnostic-v1.json`
- SHA-256: `e60dff3748f0156dc63de72ddd07ea77b73b5d7f96ef82a4945a33ff72920138`
- runner SHA-256: `3a4c6602a59e0a9de921171cd616064bb18d3ebfe356720d974267367a06cb42`
- model module SHA-256: `d837090a3e0bc014ab2818ee87d6d970273453f3377feefa7df639fffe54d699`
