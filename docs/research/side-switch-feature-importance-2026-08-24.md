# Side-switch feature importance and error attribution — 2026-08-24

## Decision

The promoted `union34-top2-x2` model does have a small set of consistently important
feature families, but its roughly mid-50% precision and recall do **not** come from one
uniformly weak classifier.

- Recall is mostly a representation/ranking failure. Of 21 missed switches, four have
  no candidate, 15 have a covered candidate below the classifier threshold, one is
  removed by local suppression, and one is removed by the soft count penalty.
- The most non-redundant family is player assignment/appearance. Removing it under the
  complete nested protocol drops F1 from 56.86% to 40.00%. Production gap state and
  court-band appearance are the next most important families.
- The player family is also the clearest precision liability: selected false positives
  receive **more** player-assignment logit support than selected true positives on
  average. The head is good at recognizing a large player/appearance change, but that
  evidence is not specific enough to prove a persistent team-side swap.
- Three statistically interchangeable inputs encode internal-candidate identity. The
  internal path has only 1/4 positive-candidate recall and 1/5 selected-candidate
  precision. Its generator score is functioning mainly as candidate-kind identity, not
  as useful within-kind ranking.
- The ten currently unused serve-anchor values are not a missing solution. Adding them
  with semantically invalid internal values masked drops outer-held F1 to 48.48%.

The next feature version should therefore prioritize a persistent two-state team-side
representation, swap-specific interactions, and a separate internal-candidate ranking
contract. Merely changing the global threshold, removing the decoder, or appending the
existing serve-anchor columns is unlikely to solve the current error balance.

The implementation-grade feature definitions, extraction architecture, staged
experiment matrix, decision gates, and browser/Android port plan are specified in the
[side-switch feature development plan](./side-switch-feature-development-plan-2026-08-24.md).

## Scope and interpretation

This analysis uses the exact 704-candidate, 11-recording, 50-marker opened-development
scope bound to the promoted hard-negative winner. It does not use a protected test set.

Every baseline candidate score is from a model fit without that candidate's recording.
Each held recording uses the fixed variant's fit-selected threshold already stored in
the promoted evaluation. The reconstruction exactly reproduces 52 proposals, 29 TP,
23 FP, and 21 FN.

Importance is measured four ways:

1. standardized coefficient magnitude and sign stability across 11 outer-fit models;
2. 200 within-recording permutations of each feature and feature family;
3. deterministic mean-neutralization in each held model; and
4. full nested family ablations that refit hard-negative models and reselect thresholds
   without the held recording.

The nested family ablation is the primary evidence about whether a family contains
unique information. Coefficients and permutation effects can split or exaggerate
importance when inputs are correlated. The permutation percentiles are variability
under feature perturbation, not confidence intervals for unseen recordings.

## Baseline and recall failure decomposition

| Metric | Outer-held result |
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
| F1 | 56.86% |
| Candidate-row AP | 44.50% |

| Cause of missed human switch | Count | Share of 21 FN | Feature work can address it? |
| --- | ---: | ---: | --- |
| No candidate in the ±4-second union | 4 | 19.05% | No; candidate generator must change |
| Covered candidate below threshold | 15 | 71.43% | Yes; representation/ranking/calibration |
| Local adjacent-candidate suppression | 1 | 4.76% | Possibly, but not a primary cause |
| Soft count penalty | 1 | 4.76% | Possibly, but not a primary cause |

This rules out the decoder as the main recall bottleneck. Disabling local cleanup could
recover at most one of the current covered misses directly; lowering the threshold
would expose many additional negatives because candidate ranking is only 44.50% AP.

Candidate coverage is also uneven by recording. `203801418` contributes two of the
four upstream misses; `180646590` and `212717581` contribute one each. In contrast,
all four switches in the zero-recall recording `193307688` are covered: that failure
is entirely classifier ranking, not proposal coverage.

## Feature-family importance

Every row below is a separately refit fixed top-2/2× hard-negative model with threshold
selection repeated inside each outer fold.

| Feature view | Row AP | Proposals | TP | FP | Precision | Recall | F1 | F1 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Current 34 inputs** | **44.50%** | 52 | 29 | 23 | **55.77%** | **58.00%** | **56.86%** | — |
| Drop player assignment/appearance | 38.84% | 55 | 21 | 34 | 38.18% | 42.00% | 40.00% | -16.86 pp |
| Drop production gap state | 34.51% | 65 | 27 | 38 | 41.54% | 54.00% | 46.96% | -9.91 pp |
| Drop court-band appearance | 34.66% | 48 | 24 | 24 | 50.00% | 48.00% | 48.98% | -7.88 pp |
| Drop adjacent-rally production support | 39.39% | 49 | 25 | 24 | 51.02% | 50.00% | 50.51% | -6.36 pp |
| Drop internal-identity triplet¹ | 44.23% | 55 | 28 | 27 | 50.91% | 56.00% | 53.33% | -3.53 pp |
| Drop declared candidate provenance | 44.65% | 54 | 28 | 26 | 51.85% | 56.00% | 53.85% | -3.02 pp |
| Drop player observation quality | 44.40% | 57 | 29 | 28 | 50.88% | 58.00% | 54.21% | -2.66 pp |

¹ The overlapping diagnostic triplet is `productionGapLiveFraction`,
`candidateIsInternalDeadStatePeak`, and `candidateGeneratorScore`; it is not one of the
six disjoint semantic families.

The main conclusions are:

- Player assignment/appearance contains the most unique decision information. The
  other families cannot reconstruct either its recall or its precision.
- Gap-state information is especially important to precision. Without it, only two TP
  are lost but 15 FP are added.
- Court-band appearance is not redundant with player-isolated appearance despite high
  pairwise correlations. Removing it loses five TP.
- Player observation quality has little row-ranking value by itself. Its removal keeps
  all 29 TP but adds five FP, so its useful role is modest precision control rather than
  positive switch evidence.
- Candidate provenance is weak but not useless. Removing it loses one TP and adds three
  FP. Because candidate identity is also encoded by `productionGapLiveFraction`, the
  declared two-feature group understates the total kind effect.

## Most important individual inputs

The table ranks representative inputs by a combination of stable standardized weight
and held-recording permutation loss. A positive permutation loss means shuffling the
feature reduced the baseline metric. All coefficients below keep the same sign in all
11 outer-fit models.

| Feature | Final standardized coefficient | Mean permutation AP loss | Mean permutation F1 loss | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `productionGapMeanDeadStateScore` | +0.426 | 10.20 pp | 12.90 pp | Strongest candidate-rank signal; sustained dead-state context |
| `productionGapDurationSeconds` | +0.254 | 4.57 pp | 13.23 pp | Long gaps strongly affect thresholded event selection |
| `productionMinimumAdjacentRallyPeak` | +0.280 | 6.73 pp | 7.53 pp | Both adjacent rallies need credible live support |
| `v4GlobalAppearanceChange` | +0.217 | 4.32 pp | 10.97 pp | Broad visual change across the boundary |
| `v4BroadSameAssignmentCost` | +0.217 | 4.18 pp | 10.62 pp | Court-band same-side mismatch |
| `playerOrientationFlipEvidence` | +0.226 | 3.91 pp | 6.77 pp | Player-side orientation reversal |
| `playerSwapMargin` | +0.235 | 2.81 pp | 6.20 pp | Swapped assignment cheaper than same assignment |
| `playerGlobalAppearanceChange` | +0.225 | 0.99 pp | 9.04 pp | Player-region change; important at the decoder threshold but confounded |
| `v4MeanSwapMargin` | +0.173 | 1.95 pp | 4.22 pp | Court-band swapped-versus-same evidence |
| `minimumProposalCount` | -0.155 | 1.75 pp | 2.79 pp | Fewer motion proposals are favored in the current fit |

The current model has sharply revalued several features relative to the former
`local-peak-soft-count` winner:

- `productionGapMeanDeadStateScore`: +0.018 to +0.426;
- `productionMinimumAdjacentRallyPeak`: +0.007 to +0.280; and
- `playerSwapMargin`: +0.045 to +0.235.

That comparison is descriptive rather than causal because the candidate universe,
class objective, and hard-negative policy also changed. It nevertheless explains what
the promoted head now depends on: a credible dead gap surrounded by live rallies plus
appearance evidence consistent with swapping sides.

Several individual values are effectively devalued or unstable:

- `playerSwappedAssignmentCost` has mean outer coefficient +0.008 and the same sign in
  only 7/11 folds;
- `minimumFarSupport` is +0.012 with only 6/11 sign agreement;
- `productionMinimumAdjacentSupportCount` is +0.013 with 7/11 agreement;
- `proposalCountChange` is +0.018 with 8/11 agreement, and shuffling it slightly
  improves mean event F1; and
- `sideSupportImbalanceChange` and `beforePlayerPaletteInstability` have small,
  unstable negative effects.

These are candidates for removal, reformulation, or use as quality gates. Their weak
additive coefficients do not prove that their underlying measurements are useless.

## Why recall remains near 50%

### Covered misses are visually weaker in the exact families the model trusts

For each human marker with a candidate, the analysis sums fold-specific standardized
feature contributions. Detected versus missed target candidates differ as follows:

| Family | Mean logit support, detected targets | Mean logit support, missed targets | Detected minus missed |
| --- | ---: | ---: | ---: |
| Court-band appearance | +0.872 | +0.134 | **+0.738** |
| Player assignment/appearance | +0.897 | +0.534 | **+0.363** |
| Production gap state | +0.583 | +0.244 | **+0.338** |
| Adjacent-rally production support | +0.058 | -0.003 | +0.061 |
| Candidate provenance | +0.011 | -0.048 | +0.059 |
| Player observation quality | +0.070 | +0.024 | +0.046 |

The largest individual recall shortfalls are broad/global court appearance, player
global appearance, gap duration, court swap margin, and adjacent-rally peak. This means
the misses do not merely sit a few points below a badly chosen threshold; most look
less like the current feature definition of a switch.

### `193307688` exposes cross-family conflict

All four switches are covered and all four fall below the 0.405 held-fold threshold.
Their player-assignment family is strongly positive (+0.83 to +1.06 logit), but their
court-band family is negative (-0.46 to -0.09), and gap/adjacent support is inconsistent.
Their probabilities are only 0.114–0.306.

The same recording's three false proposals score 0.584–0.834. Two receive very large
player-assignment support (+2.42 and +3.01); the strongest false proposal also receives
+1.63 court-band support. A single additive head cannot express “trust the player swap
only when it persists and the court view is reliable.” A high transient value in one
family can dominate contradictory evidence.

### Internal candidates are a separate weak regime

The union contains 624 adjacent-boundary candidates with 42 positives and 80 internal
dead-state peaks with only four positives. The promoted decoder selects five internal
candidates: one positive and four negative. Three of the four positive internal rows
are missed. This is 25% positive recall and 20% selected precision for the internal
path.

`productionGapLiveFraction` and `candidateIsInternalDeadStatePeak` have correlation
1.000 on this artifact; each correlates 0.99998 with `candidateGeneratorScore`. Their
final standardized coefficients are also nearly identical and negative. The three
inputs collectively impose roughly -0.28 to -0.37 logit on the missed internal
positives. Removing the triplet is not a fix—it loses one overall TP and adds four FP—
because the negative prior is useful against 76 internal negatives. The problem is
that one global linear head has no useful *within-internal* confidence signal.

### Four switches have no row to classify

No reweighting, loss function, or threshold can recover these. The candidate union
needs a continuous state-change source outside decoded rally boundaries and the
current `deadState >= 0.98` internal peaks.

## Why precision remains near 50%

Of the 23 false proposals, 19 are adjacent-rally boundaries and four are internal
peaks. Comparing selected FP to selected TP logit support gives the key asymmetry:

| Family | Mean FP logit support | Mean TP logit support | FP minus TP |
| --- | ---: | ---: | ---: |
| Player assignment/appearance | **+1.103** | +0.897 | **+0.205** |
| Player observation quality | **+0.170** | +0.070 | **+0.099** |
| Candidate provenance | -0.007 | +0.011 | -0.018 |
| Adjacent-rally production support | +0.017 | +0.058 | -0.041 |
| Production gap state | +0.401 | **+0.583** | -0.182 |
| Court-band appearance | +0.628 | **+0.872** | -0.244 |

The false positives are therefore not primarily candidates with no positive evidence.
They often have very strong player changes. `playerGlobalAppearanceChange` is among the
top five positive contributors for 15/23 FP, while `playerOrientationFlipEvidence` is
among the top five for 11/23. The most precision-helpful features are mean dead-state,
minimum adjacent rally peak, court swap margin, broad/global court change, and gap
duration; TP receive more support from each than FP on average.

This is a specificity problem:

- local palette/orientation change is necessary for recall but can also fire on a
  non-switch visual transition;
- the model has no persistent post-change state requirement;
- quality values enter additively rather than deciding whether appearance evidence is
  trustworthy; and
- one linear sum cannot require agreement between player swap, court swap, stable
  geometry, and temporal persistence.

No semantic cause such as timeout, huddle, camera movement, or player exit was assigned
in this analysis because the 23 FP windows were not manually re-reviewed here. Those
categories should be added in the next error-review pass instead of inferred from
feature values alone.

## Missing-feature screen

The full feature artifact contains ten serve-anchor values excluded from the promoted
34-input head. Their strongest recording-held-out univariate screen is
`productionDeadToNextServeSeconds` at 14.72% AP; most others are near the 6.53% positive
prevalence baseline. Adding all ten to the complete nested model, while masking the
semantically invalid internal rows, produces:

| View | Row AP | Proposals | TP | FP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current 34 | 44.50% | 52 | 29 | 23 | 55.77% | 58.00% | 56.86% |
| Current 34 + masked serve-anchor 10 | 40.59% | 49 | 24 | 25 | 48.98% | 48.00% | 48.48% |

The current serve-anchor values should not be appended to the model. A useful score
prior would need actual point-result state and uncertainty, not another proxy for
rally timing.

## Recommended feature versions

### 1. Persistent team-side state features — highest priority

Start with recording-relative pairwise comparisons across 2–3 stable rallies before
and after each boundary. Require within-side continuity plus cross-boundary swapped
assignment, without assigning an absolute `team A near` state or anchoring a universal
sign to the first three rallies. Candidate features should include:

- median same-versus-swapped cost over 2–3 stable rallies before and after;
- lower-quartile cross-boundary swap evidence;
- minimum before/after continuity and post-change persistence;
- agreement between player-isolated and court-band state; and
- observation count and quality for missing/weak context.

This directly targets both failure modes: weak one-window FN can accumulate evidence,
while transient high-change FP should fail the persistence requirement.

Only if these pairwise emissions transfer across held recordings should they feed an
anonymous two-state sequence with state-change posterior, entropy, hysteresis, and run
length. The state names must be arbitrary; the earlier absolute parity sign was
strongly biased toward its initial state.

### 2. Swap-specific interactions instead of generic change magnitude

Add explicit interaction features to the otherwise cheap linear head:

- `playerSwapSpecificity = playerSwapMargin / (playerGlobalAppearanceChange + eps)`;
- court-band equivalent using `v4MeanSwapMargin` and `v4GlobalAppearanceChange`;
- player/court swap agreement, such as their minimum normalized margin;
- sign agreement and disagreement penalties between player and court swap evidence;
- swap margin multiplied by minimum side separation and alignment quality; and
- post-change persistence multiplied by swap evidence.

These let the model distinguish “everything changed” from “near/far team identity
specifically exchanged.” Product/min/agreement terms also express gates that the
current additive logistic head cannot.

### 3. Candidate-kind-specific ranking

Report boundary and internal P/R separately now, but defer a flexible separate internal
head until new within-internal state/trace evidence and more positive recordings exist.
The already tested internal specialist matched zero internal events in outer-held
evaluation, so another intercept, threshold, percentile, or geometry-only head is not
the next experiment.

Do not simply remove the internal prior: the nested diagnostic shows that it controls
false positives. The goal is a useful within-internal ordering signal, not equal scores
for the two candidate regimes. Once that evidence and data exist, define separate
heads/calibration only with an explicit cross-kind merge and decoder contract.

### 4. Continuous high-recall candidate features

Generate candidates from low-rate persistent state flips and coordinated cross-court
motion, not only decoded rally boundaries or very high dead-state peaks. Candidate
features should include change-point strength, coordinated opposite-direction player
flow, stable court-side occupancy before/after, and duration of the new state. This is
the only feature work that can address the four no-candidate FN.

### 5. Camera/scene confounder features

The current `v4MaximumCameraShift` is weak and does not explain large appearance
changes. Add court-line/homography stability, zoom/crop change, affine residual after
alignment, scene-cut score, and player-mask coverage loss. Use these to downweight
appearance evidence, not as positive switch evidence.

### 6. Reformulate quality features as gates

The current observation-quality family preserves recall but only modestly controls FP.
Retain the stronger measurements—side separation, after-palette instability, proposal
count, and alignment—but use them to scale or abstain from swap evidence. Drop or
reformulate the unstable support/count differences. Store signed before/after changes,
not only `min` and absolute difference, so the model knows whether observability
improved or collapsed.

### 7. Gap-shape features

Keep mean dead-state and duration, which are strongly validated. Add contiguous
dead-state duration, rise/fall slopes, time from live-to-dead and dead-to-next-live,
multi-threshold area, and overlap between player cross-court motion and the dead gap.
These should distinguish a switch-shaped gap from another long dead interval better
than mean/duration alone.

## Proposed next experiment order

1. Freeze an error taxonomy for the 23 current FP and the 17 covered FN by reviewing
   their windows without changing labels.
2. Run the fixed swap-interaction bundle and the two-production-bundle gap-shape bundle
   as separate, no-new-frame experiments.
3. Build a cached per-range visual-summary artifact, then compare directional quality,
   scene confounders, and candidate-local multi-rally persistence as separate arms.
4. Combine only independently passing families, one at a time, into a compact winner.
5. If pairwise persistence succeeds, implement the anonymous two-state sequence and
   measure state-change candidate coverage before fitting a new ranker.
6. Defer an internal head until new within-range evidence exists and the positive data
   span is large enough for a held-recording comparison.
7. Repeat grouped nested development evaluation, report strict and ±4-second event
   metrics, and keep candidate coverage separate from classifier recall.
8. Freeze the chosen version before evaluating new untouched recordings. Do not use
   the protected test split to select features or thresholds.

Success should require improvement on the primary pooled event metrics without hiding
recording failures: separately report `193307688`, internal candidates, no-candidate
events, and FP with high player-change evidence. Strict timing should remain a separate
guardrail because the promoted model's strict F1 is only 45.10% versus 56.86% at ±4
seconds.

## Artifacts and reproduction

The complete machine-readable artifact contains every coefficient, mutation result,
family ablation, univariate screen, correlation, per-recording result, missed-event
trace, and false-positive contribution inventory:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-feature-importance-2026-08-24.json`

SHA-256: `b2c501c61e9f7b3aeb2bbb04cf73f3f9793e1831053a7762c08993924daea14d`

Reproduce it from the repository root:

```bash
PYTHONPATH=. /home/developer/volleycut/.venv/bin/python \
  scripts/analyze-side-switch-feature-importance.py \
  --permutation-repeats 200 \
  --overwrite
```

The implementation is
[`analyze-side-switch-feature-importance.py`](../../scripts/analyze-side-switch-feature-importance.py).

## T2 trained-feature follow-up — 2026-08-24

The later endpoint-identity T2 engineering pass was explicitly trained on the same
opened development labels under a matched 624-boundary nested comparison. It does not
improve the model: precision moves from 50.00% to 48.15%, recall remains 52.00%, F1
moves from 50.98% to 50.00%, and strict F1 loses 2.71 points.

Feature importance identifies two different failures. The directional
`appearanceTransportSwapMargin` is nearly ignored (standardized coefficient `+0.0068`,
rank 32/36) and changes sign in five of 11 outer fits. The conditional identity value
is stable but weak (`+0.0679`, rank 24/36, positive in 11/11 fits); by itself it gains
one TP while adding two FP. This is general match/observability evidence, not enough
evidence that teams actually exchanged sides.

This result sharpens the feature direction: improve player/team isolation until
cross-side transport direction transfers across recordings, then test one frozen
joint direction-similarity-reliability condition. Do not append more transformations
of the current unstable margin. Candidate work remains separate because the boundary
scope contains candidates for only 43 of 50 markers and the full union for 46 of 50.

The complete protocol, per-recording changes, ablations, and artifact hashes are in
[the T2 decision record](./side-switch-t2-conditional-identity-transport-plan-2026-08-24.md).

## T4 selective-resolution follow-up — 2026-08-24

T4 applies the same pinned detector to a smaller native far-court crop before the
jersey/team transport stack. This fixes much of the missing-observation problem:
both-team endpoint coverage rises from 44.57% to 74.17%, and four-team boundary
coverage rises from 26.12% to 58.33%.

In the exact held-recording boundary comparison, T4 preserves precision at 50.00%,
raises recall from 52.00% to 56.00%, and raises F1 from 50.98% to 52.83%. That is a
real directional improvement—two net TP for two net FP—but the +1.8498-point F1 gain
misses the preregistered +2.0-point gate. T4 is therefore rejected rather than rounded
up or tuned after inspection.

The importance pattern is more informative than the binary decision. Raw jersey-team
transport is positive in 11/11 outer fits (`+0.1053`, rank 17/37), and continuity is a
stable negative veto in 11/11 (`-0.1577`, rank 13/37). The reliability-gated swap
value is nearly zero (`+0.0059`, rank 34/37) and changes sign in four folds. Selective
resolution made transport direction learnable; sparse/unstable tracking reliability
now limits how confidently that direction can be used.

The next feature direction is not a larger proxy or threshold adjustment. Preserve
the T4 crop and add court-constrained temporal tracking as a separate version, aiming
to convert intermittent far-player observations into longer, reliable anonymous-team
tracklets. Keep resolution, descriptor, model protocol, and candidate source fixed so
the effect remains attributable. Full details and immutable artifacts are in
[the T4 decision record](./side-switch-t4-selective-far-court-detection-plan-2026-08-24.md).

## T5 court-tracking follow-up — 2026-08-24

T5 replaces individual-player linkage with an anonymous team-state track for each
court side across T4's same five frames. This succeeds as an observability mechanism:
both-team endpoint coverage rises from 74.17% to 81.10%, four-team boundary visibility
rises from 58.33% to 68.75%, and every recording exceeds 42% both-team coverage.

It fails before labels because direction gets worse. Reliable swap evidence is nonzero
on 16.19% rather than T4's 17.79%, missing a frozen required gain of three points.
The raw positive swap-margin rate also falls 23.56% to 19.87%, while continuity rises
40.54% to 52.56%. Team reliability and cohesion improve, but between-team separation
falls from 0.3739 to 0.3196.

This narrows the feature direction again: court-side temporal ownership is useful, but
averaging every retained jersey is too permissive. A subsequent representation should
select a dominant multi-frame jersey mode and report outlier/secondary-mode support,
preserving team separation while keeping T5's coverage. T5 was not trained, so there
is no T5 precision, recall, F1, or coefficient importance to compare. Full details are
in [the T5 decision record](./side-switch-t5-court-constrained-temporal-team-tracking-plan-2026-08-24.md).

The later user-authorized diagnostic model confirms this attribution. T5 recall rises
52.00% to 56.00%, precision falls 50.00% to 49.12%, and F1 reaches 52.34%, below T4's
52.83%. Raw direction is positive in 11/11 folds and continuity is negative in 11/11,
but the intended reliable-swap feature is also negative in 11/11. The model learns
that T5's confidently pooled positive-swap evidence is counter-evidence, so stronger
outlier rejection is a representation requirement rather than optional cleanup.

## T6 dominant-consensus follow-up — 2026-08-25

T6 tests hard dominant-mode outlier rejection without labels. It restores the missing
identity specificity: mean team separation rises 0.3196 to 0.5010 and positive raw
swap margins rise 19.87% to 28.04%. But 4,120 rejected observations reduce both-team
visibility to 61.57%, four-team visibility to 42.31%, and reliable swap coverage to
14.42%. It therefore stops before model training.

Together T5/T6 show that neither extreme is suitable. Full pooling has coverage but
misleading identity; hard consensus has identity but insufficient temporal support.
The remaining feature direction is a soft robust mixture that retains team
availability while weighting dominant/secondary appearance modes and exposing mode
entropy/support as reliability. This should be tested on new recording-held evidence,
not by selecting another distance radius on the repeatedly opened scope.

## T7 soft-consensus follow-up — 2026-08-25

T7 retains every observation while continuously downweighting distance from the T6
medoid. It recovers exact T5 visibility and improves directional availability:
positive raw swap margins rise from 19.87% to 25.48%, and reliable swap evidence from
16.19% to 19.39%. The price is incomplete identity recovery. Mean minimum team
separation reaches only 0.3387, well below the preregistered 0.3796 target and still
below T4's 0.3739.

This resolves the hard-versus-soft question without labels. A single weighted mean
cannot simultaneously preserve intermittent team observations and keep distinct
jersey identities sharp. The next feature should retain a bounded set of appearance
modes per court side, expose primary/secondary support and ambiguity, and compare
mode sets across the boundary. That changes representation topology rather than
tuning T6/T7's already observed distance scale.

## T8 explicit-mode follow-up — 2026-08-25

T8 confirms that explicit modes repair most of the separation loss: 0.3387 rises to
0.3707 while exact T5 visibility is retained. Yet symmetric nearest-mode matching is
too permissive. Positive margins fall to 18.43% and reliable swap evidence to 14.58%,
even though secondary modes are substantive on 90.69% of available tracks.

The feature direction should keep the two modes but require mass-preserving
correspondence. Exact two-by-two optimal transport is the smallest attributable next
change: it prevents both source modes from explaining themselves through the same
destination mode while respecting their measured support.

## T9 mass-preserving follow-up — 2026-08-25

T9 confirms that balanced mode transport is active and repairs separation: the mean
available-pair cost rises 0.3470 to 0.3991 and minimum team separation rises 0.3707 to
0.4178, above T4. It does not recover enough same-versus-swapped direction. Positive
margins rise only 18.43% to 20.51%, below T7's 25.48%, and reliable-swap coverage rises
only 14.58% to 16.51%, below T7's 19.39%. T9 therefore fails before labels, so there
is no precision, recall, F1, or coefficient attribution to report.

The remaining problem is upstream of the set-distance formula. T8/T9 mass is the
frequency of retained detector observations, not a stable count of player identities;
balanced transport can preserve that mass exactly without making independently built
endpoint modes directionally comparable. The next representation should build
persistent player tracklets first, assign one appearance unit per stable tracklet,
and expose one-to-one match coverage separately from conditional jersey similarity.
This is a new representation hypothesis, not authorization to tune T9 on the opened
recordings.

## T10 tracklet-unit follow-up — 2026-08-25

T10 validates the stable-player-unit part of the hypothesis. Mean minimum team
separation rises to 0.5845, cross-match coverage remains distinct from conditional
appearance at Spearman 0.7767, and the 364 four-team-visible boundaries contain 148
positive margins versus T4's 147. Exact one-to-one player-unit matching is therefore
directionally at least as useful as T4 pooling on the same observable scope.

The frozen all-boundary gates still fail because T10 inherits T4's 58.33% four-team
visibility: positive margins are 23.72% and reliable-swap coverage is 17.15%. The next
representation should not relax track qualification. It should retain stable T10
units and add one separately typed T5 pooled fallback unit only for an otherwise empty
side, testing whether coverage can be restored without diluting stable identities.

## T11 empty-side fallback follow-up — 2026-08-25

T11 restores exact T5 visibility with 88 typed fallback sides while preserving strong
0.5726 separation. Reliable swap reaches 19.55% and passes its gate; positive margins
rise from T10's 23.72% to 24.36% but remain seven boundaries below T7's target.

The likely remaining mismatch is representational type, not support: T11 can compare
one pooled team fallback directly with one individual stable player. The next
experiment should select player-unit transport only for stable-to-stable pairs and
the exact pooled-team comparison otherwise, leaving both source representations fixed.

## T12 type-consistent hierarchy follow-up — 2026-08-25

Pairwise like-with-like routing raises positive margins to 26.92% and reliable swap to
19.87%, both above the frozen T7 gates, while retaining 0.5348 separation. It still
fails before labels because conditional cross similarity is redundant with T11 at
Spearman 0.9866.

The routing also exposes a new confound: same and swapped alternatives can use
different representation types. A source-consistent router should choose one type for
both destinations from each before-side source, so the margin measures appearance
assignment rather than route availability.

## T13 source-symmetric routing follow-up — 2026-08-25

Enforcing one route per before-side removes T12's directional gain: positive margins
return to 23.72%, while conditional similarity remains >=0.983 correlated with the
prior hierarchy. The routing branch is therefore exhausted before labels.

The next independent representation should target secondary player roles rather than
coverage routing. A medoid over stable player-tracklet jerseys gives each player one
vote while resisting a libero/outlier, and requires no observed distance threshold.

## Limitations

- The 11 recordings are opened development evidence, not an independent generalization
  claim.
- Repeated permutations do not create 200 independent datasets.
- Correlated features share importance. The exact model can depend on a family even
  when one member has a small coefficient.
- Contribution differences describe associations in the held predictions; they do not
  prove the physical reason visible in each video window.
- The new feature ideas are hypotheses. They require predeclared ablations and unseen
  recordings before promotion.
