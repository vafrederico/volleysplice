# Side-switch M1 foreground-motion experiment — 2026-08-24

## Decision and scope

M1 is a new-information experiment after I1, G1, Q1, C1, and P1 failed at least one
promotion gate. It asks whether physical foreground movement through court depth can
separate actual side exchanges from generic palette/scene changes.

The first M1 arm is **boundary-only and feature-only**:

- candidate universe: the frozen 624 adjacent-rally boundaries;
- positive candidate labels: the same 42 boundary labels used by E6;
- human markers: all 50, so candidate coverage remains visible;
- control: the matched boundary-only 34-input profile;
- candidate generation: unchanged;
- loss, hard negatives, nested thresholds, and decoder: unchanged; and
- no P1, Q1, C1, I1, G1, P2, or internal-specialist values in the M1 head.

This arm cannot recover the four current human events with no candidate. It tests only
whether new motion evidence improves ranking and selection of covered boundaries.

## Frozen frame contract

For candidate gap `[g0, g1]`, let `L = g1 - g0`. Extract six ordered optical-flow
pairs. The pair separation is:

```text
delta = min(0.25 seconds, L / 2)
```

Pair midpoints are six evenly spaced values from `g0 + delta/2` through
`g1 - delta/2`, inclusive. Pair `i` uses frames at `midpoint_i - delta/2` and
`midpoint_i + delta/2`. This samples the entire gap while keeping every flow estimate
local in time. Even a 0.125-second gap remains valid. Exact timestamps are stored.

Each frame uses the frozen manifest ROI, court normalization, and `256 x 144` shape.
Every candidate requests exactly 12 frames and computes six flow fields. Timestamp
deduplication is allowed only when exact rounded timestamps agree.

## Frozen motion reduction

For each pair:

1. convert the two court-normalized frames to grayscale;
2. calculate Farneback flow with `(pyr_scale=0.5, levels=3, winsize=15,
   iterations=3, poly_n=5, poly_sigma=1.1, flags=0)`;
3. fit and subtract the existing robust affine camera-flow field from
   `analysis.serving_side_flight._affine_camera_flow`;
4. normalize residual `flow_y` by frame height and residual magnitude by the frame
   diagonal;
5. define foreground energy as `max(magnitude - max(7.5e-4, p90(magnitude)), 0)`;
6. restrict flux measurements to court columns `[0.06, 0.94]` and rows
   `[0.18, 0.94]`; and
7. weight near-origin upward flow by a linear near weight that is zero at normalized
   row `0.45` and one at `0.90`; weight far-origin downward flow by a linear far weight
   that is one at `0.20` and zero at `0.55`.

For pair energy `e`, normalized vertical flow `v`, and side weights `w_near` and
`w_far`:

```text
nearToFar_pair = sum(e * max(-v, 0) * w_near) / sum(e)
farToNear_pair = sum(e * max( v, 0) * w_far ) / sum(e)
```

Zero energy produces zero flux. A pair is coordinated when both directional fluxes
are at least `0.001`, a fixed subpixel normalized-motion floor. Active foreground uses
the already frozen positive-energy mask.

## M1 feature bundle

The first head appends exactly eight values:

| Feature | Definition |
| --- | --- |
| `nearToFarForegroundFluxMean` | Mean of the six near-to-far pair fluxes. |
| `farToNearForegroundFluxMean` | Mean of the six far-to-near pair fluxes. |
| `bidirectionalExchangeMinimum` | Minimum of the two mean directional fluxes. |
| `foregroundExchangeImbalance` | Absolute difference of the two mean directional fluxes. |
| `coordinatedExchangePairFraction` | Fraction of six pairs meeting both `0.001` directional floors. |
| `coordinatedExchangeDurationSeconds` | Gap duration multiplied by coordinated-pair fraction. |
| `netBandMotionFractionMean` | Mean fraction of total residual energy in rows `[0.35, 0.65]`. |
| `edgeStableMotionMinimum` | `1 - active foreground fraction`, clipped to `[0,1]`, for the less-stable of the first and last pair. |

Store all six pair reductions as diagnostics, but do not put pair-specific columns in
the first head. The current before/after proposal coverage remains the stable player
occupancy measurement; M1 does not duplicate it under another name.

## Immutable artifact and validation

The extraction artifact must bind the existing full-union feature artifact, candidate
artifact, manifest, V5 geometry artifact, extractor/module hashes, current feedback
and frozen `initialInference` hashes, and source-video SHA-256 values already recorded
by Visual Summary V2. It must preserve all 704 candidate IDs/order and every existing
feature value exactly; internal rows remain present but have `m1Status = not-eligible`
and no M1 model inputs.

Required extraction checks:

- 624 eligible boundaries and 80 explicitly ineligible internal rows;
- exactly 7,488 boundary frame requests before exact timestamp deduplication;
- no labels read by the extractor;
- no frame errors;
- peak resident memory at or below 512 MiB; and
- workstation extraction wall time at or below 60 minutes.

These are research-compute gates. Product promotion still requires a representative
browser/Android benchmark and the existing 15% median side-switch-stage latency gate.

## Evaluation gates

Use the E6 boundary-only nested protocol. M1 is a standalone contender only if it:

- gains at least 2.0 percentage points pooled outer-held ±4-second F1;
- loses no more than 2.0 points of precision or recall;
- loses no more than 1.0 point of strict F1;
- selects at least one positive boundary candidate missed by the matched control;
- reduces the matched control's selected false-boundary slice whose
  `bidirectionalExchangeMinimum` is at or below the all-row median; and
- passes the E6 recording-robustness rule: reject when a positive total TP gain becomes
  non-positive after removing the largest single-recording TP gain while at least
  three other recordings lose TP.

Report AP, Brier score, strict/±4 counts and metrics, all per-recording deltas, the
predeclared target slices, extraction cost, and `193307688` explicitly. If M1 fails,
do not tune the flux floor, pair count, or gap sampling on these 50 opened events.

## Follow-up policy

- If M1 passes: run leave-one-M1-feature-out pruning, then an incremental comparison
  against the original compact winner. Candidate expansion remains separate.
- If M1 improves AP but fails event gates: preserve the diagnostics; investigate
  threshold/calibration only as a separately declared experiment, not as a retroactive
  M1 rescue.
- If M1 fails transfer robustness: stop motion-formula tuning and collect new
  recordings with explicit physical-exchange annotations.
- Do not port M1 until it passes untouched validation and the product latency budget.

## Relationship to the completed loop

This is the separately specified experiment required by the M1 section of the
[feature development plan](./side-switch-feature-development-plan-2026-08-24.md). It
does not reopen the rejected P1/P2 palette-state branch.
