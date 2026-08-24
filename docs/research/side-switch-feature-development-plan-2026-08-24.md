# Side-switch feature development plan — 2026-08-24

## Decision

The next side-switch feature version should answer a stricter question than the current
model:

> Did near/far team identity specifically exchange, remain exchanged across multiple
> stable observations, and occur during a credible transition rather than during a
> generic visual change?

The current 34-input head cannot express that complete condition. It sees one
before/after comparison, adds independent feature contributions linearly, and then
applies a recording-level decoder. That is enough to recognize many large appearance
changes, but it cannot require swap specificity, observation reliability, and
post-change persistence simultaneously.

Feature work should **not** be implemented as one kitchen-sink model. The recommended
policy has three distinct levels:

1. **Batch shared extraction work by raw-data dependency.** Compute related summaries
   in one video/trace pass so expensive frames are not decoded repeatedly.
2. **Experiment with one semantic feature family at a time.** Keep the candidate
   universe, loss, threshold-selection protocol, and decoder fixed while measuring one
   hypothesis against the exact promoted baseline.
3. **Combine only feature families that independently pass their gates.** Add one
   passing family at a time, measure its incremental value, freeze one version, and
   port only that frozen winner to the browser and Android.

The fastest first experiments are swap-specific interaction terms and production
gap-shape features because neither requires new video frames. The highest-value
structural direction is a candidate-local, multi-rally persistence representation.
Only if that pairwise representation separates change from continuity should it be
promoted into a full anonymous two-state sequence and a new state-change candidate
source.

Do not retry the already failed variants under new names:

- do not use the first three rallies as an absolute state-zero sign boundary;
- do not require one noisy `sameCost > swapCost` comparison as a hard gate;
- do not lower the internal dead-state threshold and send every added row through the
  same global head;
- do not fit another internal specialist from the existing flank/geometry values; and
- do not append the ten existing serve-anchor values.

## Execution ledger

This section is the append-only decision log for the implementation loop. A later
experiment may supersede a decision, but completed results should not be rewritten.

### E0 — exact baseline reconstruction — complete 2026-08-24

Implemented a reusable named-profile runner with the frozen square-root logistic,
top-2/2x recording-balanced hard-negative mining, nested recording-held-out threshold
selection, and the current decoder. The new runner independently reconstructs the
promoted artifacts.

| E0 parity check | Reconstructed result | Expected result | Status |
| --- | ---: | ---: | --- |
| Row AP | 44.4992% | 44.4992% | Exact |
| ±4 proposals / TP / FP / FN | 52 / 29 / 23 / 21 | 52 / 29 / 23 / 21 | Exact |
| ±4 precision / recall / F1 | 55.7692% / 58.0000% / 56.8627% | 55.7692% / 58.0000% / 56.8627% | Exact |
| Strict proposals / TP / FP / FN | 52 / 23 / 29 / 27 | 52 / 23 / 29 / 27 | Exact |
| Strict F1 | 45.0980% | 45.0980% | Exact |
| Outer thresholds | 11/11 exact | 11/11 | Exact |
| Final classifier parameters | Exact to `1e-12` | Promoted classifier | Exact |

Decision: **pass E0 and use this runner for subsequent profiles**. This is machinery
validation, not a new model gain. All 107 focused side-switch tests pass.

Artifact:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-feature-development-e0-baseline-v1.json`

SHA-256: `dffc6f22cf77efa8c90d074c260ca17840372ec20f2680e6c82ed071336df94f`

Next authorized comparison: E1, the fixed seven-value swap-interaction bundle, with no
candidate, loss, threshold-protocol, or decoder change.

## Evidence that determines the direction

The exact recording-held-out reconstruction of the promoted
`side-switch-hard-negative-mining-v1/union34-top2-x2` model is:

| Metric | Result |
| --- | ---: |
| Candidate rows | 704 |
| Positive candidate labels | 46 |
| Human switches | 50 |
| Proposals | 52 |
| True positives | 29 |
| False positives | 23 |
| False negatives | 21 |
| Precision | 55.77% |
| Recall | 58.00% |
| F1 at ±4 seconds | 56.86% |
| Strict F1 | 45.10% |
| Candidate-row average precision | 44.50% |

The 21 false negatives are not one failure mode:

| Recall failure | Count | Implication |
| --- | ---: | --- |
| No candidate in the current union | 4 | Requires a new candidate source; no classifier feature can recover it. |
| Candidate exists but scores below its held-fold threshold | 15 | Primary feature/ranking opportunity. |
| Removed by local suppression | 1 | Decoder sensitivity, not the main bottleneck. |
| Removed by the soft count penalty | 1 | Decoder sensitivity, not the main bottleneck. |

The nested family ablations show that the underlying sources are useful:

| Removed family | F1 after removal | Change from 56.86% |
| --- | ---: | ---: |
| Player assignment/appearance | 40.00% | -16.86 pp |
| Production gap state | 46.96% | -9.91 pp |
| Court-band appearance | 48.98% | -7.88 pp |
| Adjacent-rally production support | 50.51% | -6.36 pp |
| Internal-identity triplet | 53.33% | -3.53 pp |
| Candidate provenance | 53.85% | -3.02 pp |
| Player observation quality | 54.21% | -2.66 pp |

The problem is therefore not that player appearance should be removed. Selected false
positives receive more mean player-assignment logit support than selected true
positives (+1.103 versus +0.897), while true positives receive more court-band and
production-gap support. The model needs a more specific use of the strongest source,
not less of it.

Prior experiments narrow the safe design space further:

- A first-three-rally absolute parity coordinate recognized state zero well but had
  only 21.32% recall for state one. Persistence could not repair the biased emission.
- A conservative, recording-normalized same-side continuity tail removed five false
  proposals without losing a true proposal in an earlier universe. A hard positive
  swap gate reduced recall to 14%. Extreme continuity is useful negative evidence;
  one noisy positive margin is not sufficient switch proof.
- Lowering the internal dead-state threshold increased candidate coverage from 92% to
  100%, but reduced nested F1 from 50.94% to 42.74% and added 13 false positives while
  losing two true positives relative to the matched 704-row ranker.
- A separate internal specialist over transition/geometry features matched zero
  internal events in outer-held evaluation. Its eight positive rows were concentrated
  in four recordings, and its best internal AP was only 6.87%.
- Removing all promoted internal outputs produces a small opened-development gain to
  57.73% F1, but loses one true event. That is a validation candidate, not evidence
  that internal switches should be permanently discarded.
- Adding all ten existing serve-anchor values reduced F1 to 48.48%.

These results support a local, pairwise state-change representation, stronger
interaction/gating features, and new information for internal candidates. They reject
absolute prototype signs, indiscriminate candidate expansion, and feature-count growth
without a specific hypothesis.

## Current production contract and its limitation

The shipped browser and Android beta execute the same graph:

```text
production 4 Hz traces and decoded range union
        |
        +-- every adjacent rally boundary
        +-- selected internal dead-state peaks
        |
two seven-frame 256 x 144 comparison windows per candidate
        |
court-band summary + player-isolated summary
        |
22 visual + 10 production-state + 2 candidate metadata values
        |
one standardized logistic head
        |
local suppression + soft count penalty
```

This graph has four representational limits.

1. **It is candidate-local.** A boundary sees the range immediately before and after
   it, but not whether the new assignment persists through later rallies.
2. **It is additive.** A large generic player change can compensate for weak court or
   gap evidence. The head cannot naturally express `player swap AND court swap AND
   reliable view` without explicit products/minima or a nonlinear model.
3. **It collapses production sources.** Pointwise maximum rally/dead values preserve
   either model's strong response but hide whether the two production bundles agree.
4. **It mixes candidate regimes.** Adjacent boundaries and internal peaks have
   different priors and observation geometry. Three nearly identical inputs mostly
   tell the model that a row is internal; they do not rank internal rows well.

The browser already deduplicates requested frame timestamps, but it recomputes visual
summaries candidate by candidate. Because every adjacent boundary uses full source
ranges, nearly every decoded range is already sampled. A per-range summary cache can
therefore support three-rally context with little or no extra boundary-frame decoding
and may reduce repeated CV work.

## Design principles for version 2

Every proposed feature should satisfy at least one of these roles:

- **Specificity:** distinguish a near/far identity exchange from general activity or
  appearance change.
- **Persistence:** require the inferred exchange to remain consistent across more than
  one observation.
- **Reliability:** determine whether color/assignment evidence is trustworthy rather
  than treating quality as independent positive evidence.
- **Transition shape:** identify a live-to-dead-to-live interval that resembles an
  actual switch.
- **Candidate recovery:** create a row for an event outside the present boundary/high-
  dead-peak union.
- **Candidate-regime separation:** rank internal candidates with information that
  varies meaningfully *within* the internal subset.

Derived values must be deterministic and label-independent. Imputation, means,
scales, robust normalization, feature selection, hard-negative selection, threshold
selection, and any learned state transition must be fit only on the training side of
each recording split.

The recorded-video editor may use whole-recording, label-free normalization because
the complete video is available before inference. If live inference becomes a product
requirement, the normalization contract will need a separate causal version; it must
not be silently assumed to transfer.

## Shared sequence-summary layer

The visual extractor should first be refactored around a reusable intermediate
observation instead of directly returning one final candidate vector.

### `SideObservationV2`

For each decoded rally range, store:

- range ID, start/end, seven exact sampled timestamps, and source support;
- player near, far, and global 52-bin palettes;
- court broad-near, broad-far, tight-near, tight-far, and global palettes;
- before-combination values currently discarded: side separation, proposal coverage,
  proposal count, near/far support, palette instability, maximum camera shift, and
  minimum alignment response;
- translation vectors and alignment responses for all seven frames;
- post-alignment residual summaries and background-only palette summaries; and
- observation validity/quality diagnostics, without converting them to labels.

For adjacent-boundary candidates, compute each range observation once and refer to it
by ID. The current frame plan already requests the seven frames for ranges appearing
on either side of a boundary, so the main change is caching and preserving raw
summaries.

For internal candidates, retain the existing `[t-4, t-1]` and `[t+1, t+4]` flank
summaries as separate observation kinds. Do not pretend they are full stable rallies.
Internal temporal persistence will require a later within-range representation and is
not part of the first boundary-persistence experiment.

### Research artifact contract

The new immutable research artifact should store the intermediate observations and
named feature profiles, not only a flattened final vector. At minimum it should bind:

```text
schema/version
recording and video hashes
label and candidate-union hashes
production model/trace hashes
extractor source hash
candidate ID, kind, timing, and source-range IDs
observation IDs and exact sample times
raw player/court/camera summaries
raw per-source production trace summaries
all derived feature names and values
feature-profile definitions
extraction errors and missing-value reasons
```

This permits many deterministic interaction/ablation experiments without decoding the
video again. Raw frames should not be duplicated in the artifact; compact palettes,
alignment statistics, trace reductions, and provenance are sufficient.

## Feature family I1 — swap-specific interactions

### Purpose

This is the fastest experiment and directly targets the current false-positive
pattern: large player changes that are not specifically supported as a reliable swap.
It requires no new frames and can be calculated from the existing 34 values.

Define:

```text
p  = playerSwapMargin
c  = v4MeanSwapMargin
gp = playerGlobalAppearanceChange
gc = v4GlobalAppearanceChange
s  = minimumPlayerSideSeparation
u  = minimumProposalCoverage
a  = v4MinimumAlignmentResponse
relu(x) = max(x, 0)
clip5(x) = min(5, max(-5, x))
```

Both `p` and `c` are `same assignment cost - swapped assignment cost`; positive values
support a swap and negative values support continuity. Hellinger-derived values are
bounded, so `0.02` is a fixed 2%-of-range numerical floor, not a label-selected
hyperparameter.

| Proposed feature | Exact definition | Intended information |
| --- | --- | --- |
| `jointSwapMarginMinimum` | `min(p, c)` | Positive only when both modalities support a swap. |
| `positiveSwapMarginProduct` | `relu(p) * relu(c)` | Rewards joint positive evidence without rewarding two negative margins. |
| `swapMarginDisagreement` | `abs(p - c)` | Exposes cross-modality conflict to the head. |
| `playerSwapSpecificity` | `clip5(p / max(gp, 0.02))` | Swap evidence as a fraction of generic player change. |
| `courtSwapSpecificity` | `clip5(c / max(gc, 0.02))` | Court swap evidence as a fraction of broad court change. |
| `playerQualityGatedSwap` | `p * s * u` | Player swap scaled by team separability and observation coverage. |
| `courtQualityGatedSwap` | `c * a` | Court swap scaled by alignment reliability. |

The first I1 model should append these seven values to the current 34 and keep the
same top-2/2x hard-negative objective, L2, candidate universe, and decoder. This is one
semantic hypothesis—explicit swap interactions—so test it as one bundle first. If the
bundle improves the outer result, run leave-one-interaction-out ablations and retain
the smallest subset that preserves the gain. Running seven independently selected
single-feature winners first would create unnecessary multiple-comparison noise and
would miss interactions whose purpose is joint gating.

If I1 does not improve the high-player-change false-positive slice, stop. Do not keep
tuning the denominator floor or adding polynomial combinations on the same 46 positive
rows.

### Runtime cost

Seven scalar operations per candidate plus a larger logistic vector. No new frame,
trace, decoder, or model architecture is required.

## Feature family G1 — production gap shape and bundle consensus

### Purpose

Mean dead-state and gap duration are the two strongest thresholded event-selection
features, but the current representation loses temporal shape and takes a pointwise
maximum over the two production bundles. A long interval where only one bundle is
confident can resemble a consensus switch gap.

Let `d1(t)` and `d2(t)` be the two existing 4 Hz dead-state probabilities. For the
candidate gap `[g0, g1]`, define:

```text
D(t) = min(d1(t), d2(t))
E(t) = abs(d1(t) - d2(t))
L    = max(g1 - g0, 0.25 seconds)
```

Use the same inclusive 4 Hz sample-selection convention in Python, browser, and
Android. One-second pre/post windows must be clipped to video bounds; unavailable
values are `NaN` and use fold-fit imputation.

| Proposed feature | Exact definition |
| --- | --- |
| `productionGapConsensusDeadMean` | Mean of `D(t)` in the gap. |
| `productionGapDeadDisagreementMean` | Mean of `E(t)` in the gap. |
| `productionGapConsensusDeadIntegral` | `productionGapConsensusDeadMean * (g1 - g0)`. |
| `productionGapConsensusAbove80Fraction` | Fraction of gap samples where `D(t) >= 0.80`. |
| `productionGapConsensusLongestRun80Seconds` | Longest contiguous 4 Hz run where `D(t) >= 0.80`, multiplied by 0.25 seconds. |
| `productionGapDeadPeakProminence` | `max_gap D - max(mean_pre1s D, mean_post1s D)`. |
| `productionGapDeadEntryContrast` | Mean `D` over the first up-to-one-second of the gap minus mean `D` in the preceding second. |
| `productionGapDeadExitContrast` | Mean `D` over the last up-to-one-second of the gap minus mean `D` in the following second. |

Also emit `productionGapDeadPeakOffsetFraction` as a diagnostic: the absolute distance
between the highest-consensus-dead sample and the gap midpoint, divided by half the
gap duration and clipped to `[0, 1]`. Do not put it in the first G1 head; inspect its
stability first.

Keep the validated current max-combined mean, peak, and duration inputs. G1 asks whether
consensus, shape, and the missing mean-by-duration interaction add information. It does
not replace the current source until an ablation shows that replacement is safe.

### Runtime cost

One linear pass over trace samples already resident in Python, browser, and Android.
No new video decode is required. Implement all trace reductions in one shared batch,
but compare G1 separately from I1.

## Feature family Q1 — directional observation quality

### Purpose

The current extractor collapses several before/after values to `minimum` and absolute
change. That removes direction. A transition where player proposals collapse after a
camera cut is different from one where both sides remain observable, even if the
minimum and absolute difference are identical.

The Q1 profile should replace, not merely duplicate, the following symmetric values:

| Current values | Q1 replacement |
| --- | --- |
| `minimumPlayerSideSeparation`, `playerSideSeparationChange` | `beforePlayerSideSeparation`, `afterPlayerSideSeparation` |
| `minimumProposalCoverage`, `proposalCoverageChange` | `beforeProposalCoverage`, `afterProposalCoverage` |
| `minimumProposalCount`, `proposalCountChange` | `beforeProposalCount`, `afterProposalCount` |
| `minimumNearSupport`, `minimumFarSupport`, `sideSupportImbalanceChange` | before/after near and far support, with exact zero-support state retained |
| `v4MaximumCameraShift` | before and after maximum camera shift |
| `v4MinimumAlignmentResponse` | before and after minimum alignment response |

Keep both palette-instability values; they are already directional. Signed deltas can
be derived by a linear model from the two raw sides, while minima and absolute changes
cannot recover direction.

Compare `Q1 replacement` against the exact baseline at roughly matched dimensionality.
Do not keep both every old collapse and every new raw value in the first experiment;
that would create deterministic redundancy on a very small positive set. If Q1 wins,
derive the interaction gates from its before/after values using the minimum at feature-
construction time.

### Runtime cost

No new frames. The current summaries already calculate these values before discarding
their direction. It requires a feature-contract and artifact change in all runtimes.

## Feature family C1 — camera and scene confounders

### Purpose

`v4MaximumCameraShift` describes only the largest translation estimate. It cannot
distinguish stable translation from zoom, cut, crop, poor alignment, or background
replacement. Those effects can create the generic appearance changes that currently
drive false positives.

The first CPU/on-device-compatible camera bundle should use measurements already
available around translation alignment:

| Proposed feature | Definition |
| --- | --- |
| `cameraShiftDispersion` | Robust dispersion of the seven `(dx, dy)` translation vectors, pooled as the worse before/after value. |
| `alignmentResidualP90` | 90th percentile normalized absolute gray residual after applying the estimated translation. |
| `backgroundAppearanceChange` | Hellinger distance between before/after HSV palettes after excluding motion/player proposal masks. |
| `sceneCutScore` | Maximum adjacent-frame global HSV distance across the ordered before frames, gap edge, and after frames. |

Treat these as confounders. They should receive neutral or negative contributions on
their own, or later multiply swap evidence through a scene-stability term. Do not begin
with homography, dense optical flow, or a learned shot detector. If simple residual and
scene-cut values add no held-recording value, the more expensive variants lack a
foundation.

If C1 helps additively, a separate C2 interaction experiment may add:

```text
sceneStablePlayerSwap = playerSwapMargin * (1 - clip(sceneCutScore, 0, 1))
sceneStableCourtSwap  = v4MeanSwapMargin * (1 - clip(sceneCutScore, 0, 1))
```

Do not include C2 in the first C1 comparison.

### Runtime cost

No additional sampled timestamps, but more per-frame statistics. The background mask
and aligned residual should be produced during the existing visual preparation pass so
images are not converted twice.

## Feature family P1 — local multi-rally swap persistence

### Purpose

This is the main structural feature direction. It asks whether multiple stable ranges
on one side of a candidate agree with one another, multiple ranges on the other side
agree with one another, and cross-candidate comparisons consistently prefer a swapped
assignment.

It deliberately avoids the failed absolute state-zero prototype. State names are
irrelevant; only pairwise continuity versus change matters.

### Pairwise observation margin

For modality `m` and two observations `i` and `j`, let `N` and `F` be their near/far
palettes. Define:

```text
same_m(i, j) = 0.5 * [H(N_i, N_j) + H(F_i, F_j)]
swap_m(i, j) = 0.5 * [H(N_i, F_j) + H(F_i, N_j)]
M_m(i, j)    = same_m(i, j) - swap_m(i, j)
```

`M > 0` means a swapped pairing is cheaper; `M < 0` means the same side assignment is
cheaper. For the player modality, use player-isolated palettes. For the court modality,
average the broad and tight pairwise margins to mirror `v4MeanSwapMargin`.

For the adjacent boundary between ranges `R[k]` and `R[k+1]`, fix context width
`K = 3` before looking at outcomes:

```text
B = last up to three observations ending at R[k]
A = first up to three observations beginning at R[k+1]
X_m = { M_m(i, j) for i in B, j in A }
Wb_m = { M_m(i, j) for i < j in B }
Wa_m = { M_m(i, j) for i < j in A }
```

For each modality calculate:

```text
crossSwapQ25_m       = 25th percentile(X_m)
beforeContinuity_m   = -median(Wb_m)
afterContinuity_m    = -median(Wa_m)
withinContinuityMin_m = min(beforeContinuity_m, afterContinuity_m)
persistentSwapMin_m   = min(crossSwapQ25_m, withinContinuityMin_m)
```

The 25th percentile is intentional: it is positive only when most cross-boundary
comparisons support swapping. Negating within-side margins makes strong continuity
positive. The final minimum is positive only when cross-boundary swap and both stable
sides agree.

The compact first P1 bundle is:

- `playerCrossSwapQ25`;
- `playerWithinContinuityMinimum`;
- `playerPersistentSwapMinimum`;
- `courtCrossSwapQ25`;
- `courtWithinContinuityMinimum`;
- `courtPersistentSwapMinimum`;
- `crossModalityPersistentMinimum = min(playerPersistentSwapMinimum,
  courtPersistentSwapMinimum)`;
- `crossModalityCrossSwapDisagreement = abs(playerCrossSwapQ25 -
  courtCrossSwapQ25)`; and
- `minimumPersistentContextFraction = min(len(B), len(A)) / 3`.

Store cross-pair agreement fractions, medians, both individual within-side continuity
values, and context-quality summaries as diagnostics in the artifact. Do not put all
of them in the first head. If the compact nonlinear summaries help, use family
ablations to decide whether richer raw summaries are needed.

### Boundary-first scope

P1 should first be evaluated on adjacent-boundary rows, because decoded ranges provide
the stable observations and 19/23 current false proposals are boundaries. Compare a
boundary-only baseline with a boundary-only P1 head. Keep internal results separate;
do not impute rally-context features onto internal flank windows and claim that one
shared head handled both regimes.

For the overall composition diagnostic, preserve the current outer-held internal
branch unchanged while replacing only the boundary scorer. Selection of P1 must be
based on the boundary comparison, and the merged event result must state exactly how
cross-kind suppression and probability calibration are handled.

Internal persistence requires several trustworthy observations within the containing
range or a full-sequence state model. That is later work, not a reason to contaminate
the first P1 experiment.

### Runtime cost

For boundaries, the current plan already samples the relevant rally frames. Cache one
`SideObservationV2` per range, then reuse it across all adjacent candidates. The P1
pair costs are tiny relative to CV extraction. Summary caching should reduce duplicate
work and can offset the additional pair comparisons.

## Feature family P2 — anonymous two-state sequence and state-change candidates

### Entry condition

Do not implement P2 merely because persistent state is conceptually attractive. Begin
only if P1 demonstrates that pairwise swap/continuity values transfer across held
recordings. The prior absolute parity experiment proved that persistence cannot repair
a biased state emission.

### Preferred state formulation

Use an anonymous two-state sequence. Set the first decoded state to an arbitrary label
`0`; never assert that it corresponds to a particular team identity. A transition
means only that parity changed.

1. Build one cached observation per decoded stable range.
2. Score each adjacent observation edge with the P1 pairwise player/court values,
   quality, and trace context.
3. Fit the edge-change model only on the training recordings in each outer fold.
4. Decode the held recording with a two-state semi-Markov or run-length model whose
   states alternate on accepted changes.
5. Require post-change persistence through at least two stable observations as one
   preregistered sensitivity, not as a value selected from the held video.

Candidate/ranker features can then include:

- `stateChangePosterior`;
- `stateChangeLogitMargin` over the next-best local edge;
- `preChangeRunLength` and `postChangeRunLength`;
- `postChangePersistenceProbability`;
- pre/post state entropy or abstention confidence; and
- player/court change-posterior disagreement.

Generate a candidate interval from the last stable observation in the old state to the
first stable observation in the new state. The UI timestamp can remain the interval
midpoint, but candidate evaluation must preserve the interval and the existing strict
and ±4-second one-to-one matching contract.

This source can address the four no-candidate false negatives. The earlier oracle
showed that stable observations bracket all 50 reviewed switches, but that is only a
ceiling: the learned edge representation must recover those transitions without truth
states.

### Required evaluation separation

Evaluate P2 in two steps:

1. candidate coverage, proposal count, and localization before fitting a final ranker;
2. a full outer-held pipeline in which edge training, state decoding, candidate union,
   feature extraction, ranker fitting, thresholds, and decoder choices all exclude the
   held recording.

Do not add P2 candidates and simultaneously alter the final classifier loss or output
threshold. That would make any recall or precision change uninterpretable.

## Internal-candidate direction

The current internal path contains 80 candidates and only four positive rows; it emits
one true and four false proposals. Three almost identical values mainly encode
candidate kind. Repackaging those values or fitting another small internal logistic
head has already failed.

The internal branch needs new within-kind information:

- two-bundle dead-peak consensus rather than only max score;
- local peak prominence, width, rise/fall shape, and trace disagreement;
- stable player/court observations on both sides of the internal time;
- within-range state-change posterior from P2; and, if needed,
- direct foreground exchange motion through the dead interval.

Peak rank, normalized position within the source range, boundary distance, and simple
flank instability were already included in the failed specialist and should not be
the core of another model.

Do not select a type-specific threshold or learned internal head from four positives.
A recommended data gate is at least 20 exhaustively reviewed internal-positive
candidates spread across at least eight recordings before a flexible internal head is
eligible for promotion. Until then:

- report boundary and internal metrics separately;
- retain boundary-only as a frozen validation candidate;
- treat any current-internal policy as provisional; and
- use G1/P2 to collect better internal evidence and more informative reviewed rows.

If enough data becomes available, a two-head runtime must define how calibrated
boundary/internal probabilities are merged before cross-kind NMS and the soft count
penalty. Separate thresholds without a merge contract are not a complete decoder.

## Later feature family M1 — foreground side-exchange motion

This is a higher-cost candidate/reliability source, not the first experiment. Use it
only if palette persistence leaves specific uncovered or ambiguous cases.

In court-normalized coordinates, estimate foreground mass or short tracks moving from
near to far and far to near through the net-depth band. Candidate values would include:

- near-to-far foreground flux;
- far-to-near foreground flux;
- `bidirectionalExchangeMinimum = min(nearToFarFlux, farToNearFlux)`;
- imbalance between the two directions;
- duration of coordinated exchange; and
- stable occupancy before and after the exchange.

This requires frames during the candidate gap, which the current visual path does not
sample. It should therefore be a distinct extraction/candidate experiment with an
explicit frame and latency budget. Do not hide its cost inside P1.

## Build batches versus model experiments

The following distinction is the direct answer to whether features should be added in
a batch or one at a time.

| Engineering batch | Implement together because | Model experiments kept separate |
| --- | --- | --- |
| A. Derived-only | Uses the existing 34-value rows; no decode. | I1 interaction bundle versus baseline, then leave-one-interaction-out only if I1 wins. |
| B. Trace-only | All values scan the same two resident 4 Hz production traces. | G1 gap shape; internal peak-shape diagnostics; no candidate expansion. |
| C. Visual summary V2 | Q1, C1, and P1 need the same decoded frames, alignment, palettes, masks, and per-range cache. One extraction pass prevents repeated video work. | Q1, C1, and P1 are three independent heads, each compared with its matched baseline. |
| D. Sequence/candidate | P2 reuses the selected P1 observation layer. | First P2 candidate coverage, then P2 union/ranker. M1 remains a separate expensive arm. |
| E. Runtime port | Browser/Android parity work is costly and should target one immutable model. | Port only the final frozen winner; do not port research arms. |

Thus, raw measurements should be collected in sensible batches, while causal model
comparisons should be one semantic change at a time. “One at a time” means one
hypothesis family, not necessarily one scalar: a minimum, product, and disagreement
term are jointly the interaction hypothesis and should first be tested as that compact
bundle.

## Predeclared experiment sequence

### E0 — freeze and reproduce the control

- Bind the current candidate artifact, labels, source hashes, feature order, top-2/2x
  hard-negative policy, L2, threshold procedure, and decoder.
- Reproduce 52 proposals, 29 TP, 23 FP, 21 FN, and 56.86% ±4-second F1.
- Preserve the exact outer-held probability and contribution inventory for paired
  comparisons.
- Freeze diagnostic slices before training new variants:
  - 23 selected false positives;
  - 15 covered/below-threshold misses;
  - two decoder-removed misses;
  - four uncovered markers;
  - boundary versus internal rows;
  - recording `193307688`; and
  - false positives with high player-change support.

Re-review the 23 FP and 17 covered FN windows to assign a non-training error taxonomy:
camera cut/reframe, zoom/crop, huddle/timeout, player entry/exit, observation dropout,
rally merge/split, generic appearance change, ambiguous timing, duplicate candidate,
and internal-peak error. Keep physical marker truth separate from diagnostic tags.

### E1 — I1 swap interactions

- Inputs: current 34 plus the seven fixed I1 values.
- Candidate and decoder contract: unchanged.
- Question: do explicit agreement/specificity/gating terms reduce high-player-change FP
  without losing covered true events?
- Follow-up only if positive: leave-one-I1-feature-out pruning.

### E2 — G1 gap shape

- Inputs: current 34 plus the eight G1 values.
- Candidate and decoder contract: unchanged.
- Question: do consensus and temporal shape distinguish true switch gaps from other
  long dead intervals?
- Report adjacent and internal changes separately.

### E3 — first incremental combination

Run only if I1 and/or G1 passes standalone gates. Add the stronger standalone family
first, then add the other as one incremental comparison. If the second family adds no
incremental outer-held value, exclude it even if it helped alone.

### E4 — Q1 directional replacement

- Extract the Visual Summary V2 superset.
- Compare Q1 replacement with the matched 34-input baseline reconstructed from the same
  new artifact.
- Question: is the direction of observation collapse more useful than min/absolute
  summaries?

### E5 — C1 scene confounders

- Compare C1 with the same matched visual-artifact baseline.
- Question: does explicit scene instability reduce generic-change FP?
- Run C2 gating only if C1 demonstrates stable value.

### E6 — P1 boundary persistence

- Compare boundary-only current features with boundary-only current plus P1.
- Keep internal scoring frozen for the overall composition diagnostic.
- Question: does multi-rally agreement recover weak true switches and veto transient
  high-change boundaries?
- Inspect `193307688` explicitly because all four events are covered and its current
  player/court evidence conflicts.

### E7 — compact winner assembly

Order passing families by standalone held result, then add them one at a time to the
current compact winner. Stop when an addition fails the incremental gate. Do not allow
inner selection to choose arbitrary subsets from every emitted diagnostic column.

### E8 — P2 state-change candidates

- Conditional on P1 emission success.
- Evaluate candidate coverage/load first.
- Freeze the candidate generator before final-ranker training.
- Retrain/evaluate the entire held-recording pipeline.
- Keep loss and decoder sensitivities separate from the candidate comparison.

### E9 — untouched validation and port

- Freeze formulae, feature names/order, imputation/scaling, model weights, thresholds,
  candidate contract, and decoder.
- Evaluate new exhaustive recordings once.
- Port only after the source-held-out decision and runtime-cost gate pass.

## Training and evaluation protocol

### Splits

Use recording-grouped nested evaluation. In every outer fold, the held recording must
be absent from:

- feature-profile choice;
- imputation, mean/scale, and robust normalization;
- hard-negative identification and weighting;
- L2 or any other hyperparameter selection;
- edge/state model fitting;
- threshold and decoder selection; and
- feature-family pruning.

Never split candidate rows or frames randomly. Rows from one recording share camera,
teams, uniforms, court, production errors, and event base rate.

The 11 recordings are already opened development data. Nested LOO estimates transfer
inside that scope but does not turn repeated feature iteration into untouched evidence.
Final promotion requires a new exhaustively labeled set.

### Fixed training controls

For E1–E7 retain the promoted controls unless the experiment explicitly names a
change:

- square-root class weighting;
- recording-balanced top-two/2x hard-negative mining;
- L2 `0.1`;
- the current candidate universe;
- the current local suppression rule;
- six free predictions and 0.5-logit excess-count penalty; and
- fit-scope threshold selection.

Do not mix a feature test with focal loss, pairwise ranking, recording calibration,
candidate expansion, or a new decoder. Those branches were previously weak or would
prevent causal attribution.

### Required metrics

For every arm report:

- candidate coverage before classification;
- candidate-row AP overall and by candidate kind;
- one-to-one strict and ±4-second TP, FP, FN, precision, recall, and F1;
- proposal count and proposals per recording;
- per-recording metrics and paired change from the baseline;
- boundary and internal metrics separately;
- calibration/Brier diagnostics, without substituting them for event F1;
- the frozen error-slice counts;
- frame requests, decoded-frame reuse, feature-extraction time, peak memory, model size,
  and browser/Android parity status.

The primary development rank remains pooled ±4-second event F1 for this side-switch
task. Strict timing is a guardrail, not a replacement. Candidate coverage must remain
separate from classifier recall so a larger union cannot claim success merely by
making more rows.

### Recommended research gates

These are planning thresholds to freeze before running the new arms, not observed
claims:

| Gate | Recommended requirement |
| --- | --- |
| Standalone feature contender | At least +2.0 percentage points pooled outer-held ±4 F1, with neither precision nor recall falling by more than 2.0 points unless that tradeoff was the feature's declared purpose. |
| Strict timing guardrail | Strict F1 no more than 1.0 point below the matched baseline. |
| Slice validity | The predeclared target slice must move in the expected direction; e.g. I1 must reduce high-player-change FP and P1 must improve persistent/covered cases. |
| Recording robustness | Report all 11 paired changes; reject a gain caused by one recording if several others regress materially. |
| Incremental combination | A family that passed alone must still add at least +1.0 point when added to the compact winner. |
| Candidate-source diagnostic | Reach at least 48/50 opened markers while keeping average load at or below 80 candidates/video, then pass the independent ranker gate. |
| Runtime research budget | No more than 15% median side-switch-stage latency increase on representative browser and Android devices unless the accuracy gain justifies a separately approved budget. |

With only 50 events, a one-event movement can noticeably change F1. Always report raw
counts alongside percentages. Do not treat a bootstrap over 704 correlated rows as
704 independent observations.

For final production consideration, collect a new validation set with enough support
for the intended branch. A practical target is at least ten new recordings and roughly
40 switches. An internal head additionally needs the internal-positive data gate
above; a set containing no internal positives cannot validate that branch.

## Research implementation plan

### Python feature layer

Create a versioned research module rather than extending global tuples in place. A
reasonable split is:

```text
analysis/side_switch_feature_contract_v2.py
    SideObservationV2
    pairwise assignment and persistence reductions
    I1, G1, Q1, and C1 named feature builders
    named feature-profile definitions

analysis/side_switch_sequence_state.py
    P2 edge model and anonymous state decoder
    state-change candidate construction

scripts/extract-side-switch-sequence-features.py
    immutable observation/candidate artifact extraction
    exact reuse of current rows and source-hash validation

scripts/train-side-switch-feature-experiments.py
    E0–E7 nested paired runner
    per-profile thresholds, hard-negative fitting, and reports
```

The exact names can follow repository conventions, but responsibilities should remain
separate. The extractor must not import labels to calculate feature values. The trainer
attaches labels and builds fold-specific preprocessing/model state.

### Named profiles, not column slicing by accident

Represent every arm as an ordered, hashed profile:

```text
union34-v1
union34-plus-interactions-i1
union34-plus-gap-shape-g1
directional-quality-q1
union34-plus-camera-c1
boundary-union34-plus-persistence-p1
```

The model artifact must record the exact ordered names. Derived interactions should be
materialized by name before vectorization, then standardized with statistics learned
inside the fit scope. Never reuse full-development means/scales in outer folds.

### Test coverage

Add focused tests for:

- sign semantics of same versus swapped assignment;
- P1 behavior on perfect continuity, perfect swap, one noisy observation, and missing
  context;
- 4 Hz run lengths and clipped edge windows for G1;
- source-consensus versus source-disagreement traces;
- directional Q1 reconstruction and absence of old min/absolute leakage;
- camera residual invariance under pure translation;
- deterministic feature order and profile hashes;
- fold-local preprocessing and hard-negative selection;
- candidate one-to-one matching and P2 no-label leakage; and
- exact E0 baseline reproduction.

## Browser implementation after selection

Do not port every research profile. Once a single model is frozen:

1. In `prod/src/lib/on-device/side-switch.ts`, refactor the visual functions so one
   range/window produces a reusable observation summary. Cache summaries by feature
   version, range/window ID, and exact sample times.
2. Assemble selected I1/G1/Q1/C1/P1 values from cached summaries and resident
   production traces.
3. In `prod/src/lib/on-device/side-switch-model.ts`, add the exact ordered feature
   signature and parser contract. If P2 changes candidates, bump the candidate
   contract independently from the feature version.
4. Publish a new content-addressed runtime asset in `prod/public/runtime/`; never
   replace the existing asset under the same fingerprint.
5. Preserve `modelId`, `modelFingerprint`, `featureVersion`, and `candidateContract` in
   saved output so stale cached analyses invalidate cleanly.

If the final model remains one logistic head, the current inference loop can be
retained after feature assembly. If it gains boundary/internal heads or a P2 edge head,
the JSON schema must explicitly store each head's feature order, preprocessing,
threshold, and merge/decoder rule.

## Android implementation after browser freeze

Mirror the frozen browser contract only after Python/browser parity is established:

1. Refactor `SideSwitchFeatureExtractor.java` to expose a package-level observation
   summary rather than only a final 22-value array. Preserve exact palette and
   alignment math.
2. In `SideSwitchInference.kt`, cache summaries for unique range/window IDs and build
   candidate features from those summaries.
3. In `SideSwitchModels.kt`, update the content-addressed asset name, feature version,
   exact feature-name list/column count, parser, and candidate contract where needed.
4. Add the frozen asset under `android/app/src/main/assets/`.
5. Update `SideSwitchModelsTest.kt` and `SideSwitchPipelineInstrumentedTest.kt` with
   shared parity fixtures and a real-video smoke/latency check.

Python, browser, and Android should consume a small fixture containing observation
summaries, candidate IDs/kinds, expected ordered vectors, logits, probabilities, and
decoded outputs. Test intermediate features, not only final marker timestamps; final
agreement can otherwise hide compensating feature errors.

## Model artifact versioning

Any selected change requires a new feature version such as
`SIDE-SWITCH-SEQUENCE-V2`; the exact final name should describe the frozen contract,
not a mutable experiment. The runtime JSON should bind:

- ordered feature names;
- imputation, means, scales, weights, bias, and threshold for every head;
- observation-context configuration (`K = 3`, sampling rules, missing-context policy);
- production-trace shape thresholds;
- candidate-generator settings and contract;
- state decoder/merge settings if present;
- training/evaluation source hashes; and
- a content fingerprint derived from the complete payload.

Changing candidate generation without changing visual features still requires a new
candidate contract. Changing feature order or formula requires a new feature version.
Neither change should reuse the current `side-switch-c2570481c30d.json` asset name.

## What should not be combined in one comparison

The following pairs must remain separate until one side is frozen:

- feature family and candidate generator;
- feature family and class-loss/hard-negative policy;
- feature family and threshold/decoder redesign;
- P1 pairwise emissions and P2 persistence length;
- internal feature representation and internal threshold;
- camera features and an expensive optical-flow candidate source;
- score/cadence prior and visual state evidence; and
- model improvement and browser/Android numerical-port changes.

Otherwise an apparent gain cannot be attributed, and a regression offers no useful
direction.

## Prioritized execution backlog

| Priority | Work | Expected value | Cost | Go/no-go result |
| ---: | --- | --- | --- | --- |
| 1 | E0 error taxonomy and exact paired baseline | Makes every later feature claim auditable | Low/manual | Frozen slices and reproducible baseline |
| 2 | I1 derived interaction pilot | Fast precision/specificity test | Very low | Reduce high-player-change FP and pass feature gate |
| 3 | G1 trace consensus/shape | Builds on strongest validated state source | Low | Improve gap-related TP/FP separation |
| 4 | Visual Summary V2 extraction/cache | Enables several tests without repeated decode | Medium | Exact V1 reconstruction plus reusable summaries |
| 5 | P1 boundary persistence | Main precision/recall representation hypothesis | Medium | Transfer across recordings and improve boundary events |
| 6 | Q1 and C1 as separate arms | Reliability and scene-confounder refinement | Medium | Incremental value beyond I1/P1 |
| 7 | Compact winner assembly | Prevents unnecessary column growth | Low once extracted | Each added family retains incremental gain |
| 8 | P2 anonymous state/candidates | Only route here that can recover no-candidate events structurally | Medium/high | Higher candidate coverage followed by ranker gain |
| 9 | Internal head after data gate | Candidate-regime specialization | Data-limited | Enough diverse positives and held-out separability |
| 10 | M1 foreground exchange | Expensive fallback/new evidence | High | Candidate/feature gain large enough to justify frames |
| 11 | Browser and Android port | Product deployment | High validation burden | Frozen unseen-set winner, parity, latency, lifecycle gates |

## Concrete first implementation slice

The first code slice should stop before any runtime port:

1. Add named feature-profile support to the research trainer and reproduce E0.
2. Materialize the seven I1 values from the current immutable feature artifact.
3. Run the exact nested I1 comparison and its frozen error slices.
4. Add a trace-only G1 builder using both production bundles and run E2 separately.
5. Decide whether either cheap family earns entry into the compact-winner sequence.
6. In parallel at the engineering level—not in the same model comparison—design the
   `SideObservationV2` artifact and verify that it reconstructs all current 22 visual
   values exactly before adding Q1/C1/P1 outputs.

The next major code slice is the cached per-range observation extractor and P1 boundary
experiment. P2 candidate work should remain blocked on a positive P1 emission result.

## Expected interpretation of outcomes

- **I1 helps precision but not recall:** retain it as a cheap gate, then rely on P1/P2
  for recall.
- **G1 helps both:** gap consensus was missing; retain it before attempting more
  expensive visual features.
- **P1 reduces FP and recovers covered FN:** proceed to P2, because pairwise state
  changes are now observable.
- **P1 only reduces FP:** retain it as a verifier, but do not assume it can generate
  candidates.
- **P1 fails:** stop the palette-state path; prioritize better team isolation or M1
  motion rather than tuning persistence length.
- **P2 raises coverage but final F1 falls:** the candidate source is useful but its new
  rows lack rankable features; do not lower thresholds to force recall.
- **Q1/C1 add no value:** preserve their raw diagnostics for error review if cheap, but
  exclude them from the model.
- **Internal head remains unstable:** use boundary-only as a validation candidate and
  collect internal-positive data; do not choose another threshold from the same four
  positives.

## Final recommendation

Build one shared summary/trace foundation, but conduct controlled feature-family
experiments. The immediate low-cost sequence is I1 then G1. The main model direction
is P1: robust pairwise swap evidence across three stable observations before and after
an adjacent boundary. P2 is conditional on P1 and is the intended route to recovering
the four current no-candidate events. Q1 and C1 should make reliability directional and
explicit, while internal specialization must wait for genuinely new evidence and more
positive recordings.

The winning feature set should be compact. A new value belongs in the shipped model
only if it improves held-recording event behavior, still adds value after stronger
families are present, transfers to untouched recordings, and can be reproduced exactly
in Python, browser, and Android.

## Related records

- [Feature importance and error attribution](./side-switch-feature-importance-2026-08-24.md)
- [Current winner production-port contract](./side-switch-current-winner-production-port-contract-2026-08-23.md)
- [Parity feasibility](./side-switch-parity-feasibility-2026-08-23.md)
- [Same-side continuity verifier](./side-switch-continuity-verifier-2026-08-23.md)
- [Expanded internal candidates](./side-switch-expanded-internal-candidates-2026-08-23.md)
- [Internal-candidate specialist](./side-switch-internal-specialist-2026-08-23.md)
- [Hard-negative mining winner](./side-switch-hard-negative-mining-2026-08-23.md)
