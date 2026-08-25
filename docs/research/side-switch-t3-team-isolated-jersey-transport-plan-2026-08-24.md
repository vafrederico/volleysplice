# Side-switch T3 team-isolated jersey transport — 2026-08-24

## Decision and experiment boundary

T3 follows the negative opened-development T2 result. T2 conditional identity
similarity was positive in all 11 outer fits but weak, while its directional transport
margin ranked 32nd of 36 and changed sign in five fits. T3 therefore changes the raw
information used for identity transport instead of adding another transformation of
the T2 values.

This loop tests one hypothesis:

> A jersey-focused, background/skin-rejected, multi-frame anonymous team descriptor
> produces a transferable near/far transport direction and reliable continuity veto.

Candidate expansion is deliberately separate. T3 uses the immutable 704-row full
union and is eligible only on its 624 adjacent-rally boundaries. The 80 internal peaks
remain unchanged and receive no T3 values. Candidate IDs/order, labels, old features,
training loss, hard-negative policy, threshold protocol, and decoder do not change.

The extraction pass is label-independent. It must not load the marker audit, feedback,
T2 model result, model scores, or any row label. The later model run on the existing 50
markers is explicitly authorized by the user's request to execute the recommended
direction, but remains opened-development evidence and cannot promote or port T3.

## Frozen endpoint sampling and player selection

Reuse the pinned T1 person detector, native-resolution manifest ROI, and piecewise
court normalization. For every unique endpoint window, sample five frames at fractions
`(0.10, 0.30, 0.50, 0.70, 0.90)`. Keep detections with hip x in `[0.06,0.94]` and
canonical hip y in `[0.18,0.98]`, assign near/far around canonical y `0.56`, and retain
at most three detections per side per frame by detector confidence times square-root
box area.

This produces 635 endpoint windows, 3,175 frame requests, and 12,700 four-tile detector
calls. No label-dependent frame or player selection is permitted.

## Frozen jersey descriptor

Construct a 64-value descriptor from a torso region, not the complete person box.

1. When the detector shoulder-to-hip span is positive and at least 12% of box height,
   use vertical limits at 10% and 95% of that span, center x at the mean shoulder/hip
   x, and half-width `min(0.28 * box_width, 0.42 * torso_span)`.
2. Otherwise fall back to box x `[25%,75%]` and box y `[18%,58%]`.
3. Apply an elliptical central mask.
4. Reject YCrCb skin pixels with Cr `[133,173]` and Cb `[77,127]`.
5. Estimate background color as the median Lab value in the outer 12.5% ring of the
   detector box and reject torso pixels within Lab distance 10 of that median.
6. If fewer than 32 pixels or 15% of the ellipse survive, drop only the background
   rejection while retaining the torso and skin masks. Record this fallback.
7. Compute the T1-compatible `12 x 4` hue/saturation histogram, four value bins, and
   four-bin Lab marginals for L/a/b. Normalize each block and the final 64 values.

Weight retained pixels by `0.75 + 0.25 * saturation`. Descriptor support is the final
retained fraction of the geometric torso ellipse and enters tracklet reliability.

## Frozen five-frame tracklets and team summary

Link observations over the five ordered frames with cost:

```text
0.60 * Hellinger(jersey descriptor) +
0.25 * min(court-position distance / 0.50, 1) +
0.15 * min(abs(log(box-height ratio)) / log(2), 1)
```

Accept links at cost at most `0.50`. Unlinked observations start new tracklets. Pool a
track descriptor by detector-confidence times descriptor-support weighting. Tracklet
reliability is:

```text
mean_detector_confidence *
(0.40 + 0.60 * observed_frame_fraction) *
sqrt(mean_descriptor_support)
```

Only tracklets observed in at least two of five frames can enter a team summary. Keep
at most three per side. A side's anonymous team descriptor is the reliability-weighted
pool of those tracklets. Team reliability is `min(sum(tracklet reliability) / 2, 1)`.
Team cohesion is one minus the reliability-weighted mean Hellinger distance from each
tracklet to the pooled descriptor. Missing teams have reliability and cohesion zero.

These are anonymous visual team summaries; no player name, jersey label, team label,
or absolute near/far state is learned.

## Frozen T3 reductions

Use Hellinger distance between available team descriptors and cost one for a missing
side. Define the mean same-side and cross-side costs exactly as in T1. The artifact
stores raw direction, similarity, separation, observability, and cohesion diagnostics.

```text
jerseyTeamTransportSwapMargin = same_cost - swapped_cost

jerseyCrossSideSimilarityMinimum =
    min(1 - cost(near_before, far_after),
        1 - cost(far_before, near_after))

jerseyReliabilityGate = min(
    jerseyCrossSideSimilarityMinimum,
    four team reliabilities,
    before/after near-vs-far team separation,
    four team cohesions)
```

The first and only T3 model bundle contains:

| Feature | Frozen definition | Intended role |
| --- | --- | --- |
| `jerseyTeamTransportSwapMargin` | Same cost minus swapped cost. | Transferable exchange direction. |
| `jerseyReliableSwapEvidence` | `max(margin, 0) * jerseyReliabilityGate`. | Positive evidence only for reliable, separated, bidirectional team exchange. |
| `jerseyReliableContinuityEvidence` | `max(-margin, 0) * jerseyReliabilityGate`. | Reliable same-side continuity veto. |

Do not append raw conditional similarity or reliability to the first head. They are
diagnostics, not independent additive switch evidence.

## Immutable artifact and engineering gates

Output path:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t3-team-isolated-jersey-transport-features-v1.json`.

The artifact must bind source video, manifest, detector, extractor, and module hashes;
preserve all 704 rows and old values exactly; report per-recording near/far team
observability and mask fallbacks; contain no frame/detector errors; and finish within
60 minutes and 768 MiB peak RSS.

T3 is model-ready only if:

- at least 90% of endpoints have any qualified team descriptor;
- at least 60% have qualified descriptors on both court sides;
- at least 50% of boundaries have all four required teams and nonzero bidirectional
  cross-side similarity;
- all three core values are finite and nonconstant;
- `jerseyReliableSwapEvidence` is nonzero on at least 5% of boundaries;
- no two T3 core values have absolute Spearman correlation at or above `0.98`;
- no T3 core value has absolute Spearman correlation at or above `0.98` with an
  existing 34-input value; and
- all parity, provenance, resource, and no-label checks pass.

Failure stops before labels. Do not relax the two-frame track requirement, mask,
linking threshold, gate, or core formula after seeing an engineering failure.

## Frozen opened-development model evaluation

If engineering passes, compare the exact matched 624-row boundary T0 control with
`T0 + T3 core` under nested recording-held-out threshold selection, fixed square-root
logistic/L2 `0.1`, top-2/2x recording-balanced hard negatives, and the current decoder.

Require at least +2.0 percentage points pooled +/-4-second F1, no more than 2.0 points
lost in precision or recall, no more than 1.0 point lost in strict F1, and the existing
recording-robustness rule. Also require both frozen slices:

1. recover at least one positive candidate row missed by the matched T0 selection;
2. reduce T0-selected false boundaries whose
   `jerseyReliableSwapEvidence == 0`.

Report row AP/Brier, event counts, all recording deltas, full-development standardized
coefficients, absolute coefficient ranks, and sign stability across 11 outer fits.
Do not run individual-feature pruning unless the complete bundle passes every gate.

As a diagnostic only, union T3 boundary selections with the exact full-union E0
internal selections without probability calibration or cross-kind suppression. No
browser or Android work starts from opened-development evidence.

## Execution ledger

### Preregistration — frozen 2026-08-24

No T3 module, extractor, artifact, feature value, or label result existed when this
contract was committed.
