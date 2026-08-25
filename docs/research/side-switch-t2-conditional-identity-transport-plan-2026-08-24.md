# Side-switch T2 conditional identity transport — 2026-08-24

## Decision and scope

T2 is a separately versioned, label-free decomposition of the rejected T1 core. T1
showed viable player-tracklet coverage, but its matched identity mass and transport
coverage were redundant at Spearman `rho = 0.9931`. T2 asks whether appearance quality
conditional on having a match is independent enough from coverage to form a compact
future model input.

T2 is not a T1 model rescue:

- no detector thresholds, descriptors, tracklets, assignments, or video frames change;
- no label, marker audit, feedback file, model score, or T1 event result is read;
- the sole input is immutable T1 artifact SHA-256
  `982c6955b9eda279bbd98a74e274e7644d8fa348fdd59da4949e11a9b2b0e5f4`;
- all 704 rows, candidate IDs/order, and prior feature values must remain exact; and
- the output is an engineering artifact, not a trained model or promotion candidate.

## Frozen formula

For each T1 directional reduction `d` in `nearFar` and `farNear`, define:

```text
conditionalSimilarity_d =
    0                                      if coverage_d = 0
    matchedIdentityMass_d / coverage_d     otherwise
```

The T1 transport construction guarantees matched identity mass is coverage multiplied
by the reliability-weighted mean `(1 - Hellinger cost)` of matched pairs, so the ratio
is finite and lies in `[0,1]`. Use exact zero handling; no learned epsilon, floor, or
imputation is permitted.

The future T2 model core contains exactly two features:

| Feature | Definition |
| --- | --- |
| `appearanceTransportSwapMargin` | Exact T1 value: mean same-side assignment cost minus mean cross-side cost. |
| `conditionalCrossSideIdentitySimilarityMinimum` | Minimum conditional similarity of near-before→far-after and far-before→near-after. |

`transportCoverageMinimum` remains in the artifact as a diagnostic and future
reliability/abstention input, but it is not a T2 core model feature.

Every eligible row also stores both directional conditional similarities as
diagnostics. The 80 internal rows remain explicitly ineligible and receive no T2 model
values.

## Immutable transformation contract

Output path:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t2-conditional-identity-transport-features-v1.json`.

The transformer must verify and record:

- exact T1 input hash and T1 engineering decision `fail` caused only by the frozen
  non-redundancy check;
- 704 rows, including 624 eligible boundaries and 80 ineligible internals;
- exact preservation of every T1 row ID/order and every prior numeric feature;
- finite `[0,1]` conditional values on all boundaries;
- exact zero whenever the corresponding directional coverage is zero;
- no video, detector, manifest, audit, feedback, or label source loaded;
- transformer/module hashes embedded at write time;
- wall time no more than 60 seconds and peak RSS no more than 256 MiB; and
- atomic refusal to overwrite an existing artifact.

## Label-free engineering gates

T2 is validation-ready only if all checks pass:

- both core features are nonconstant;
- `conditionalCrossSideIdentitySimilarityMinimum` is nonzero on at least 50% of the
  624 boundaries, matching the minimum T1 bidirectional-observability requirement;
- absolute Spearman correlation between the two T2 core features is below `0.90`;
- absolute Spearman correlation between conditional similarity and
  `transportCoverageMinimum` is below `0.90`;
- no T2 core feature has absolute Spearman correlation at or above `0.98` with any of
  the existing 34 model inputs; and
- all parity, range, zero-semantics, and resource checks pass.

These thresholds were frozen before calculating any T2 value. If T2 fails, do not
change the ratio, aggregate by mean instead of minimum, or add an epsilon on this
opened artifact.

## Future model evaluation

Even an engineering pass does not authorize evaluation on the repeatedly opened 50
markers. After at least ten new recording-held videos with roughly 40 side switches
exist:

1. reproduce the matched boundary-only 34-input T0 control;
2. add only the exact two-value T2 core;
3. use fold-local preprocessing, hard negatives, thresholds, and decoder;
4. require the existing +2.0 pp F1, precision/recall, strict timing, target-slice, and
   recording-robustness gates; and
5. keep coverage-based abstention and candidate expansion as separate experiments.

No browser or Android port begins from this engineering artifact.

## Execution ledger

### Preregistration — frozen 2026-08-24

No T2 artifact or T2 values had been calculated when this contract was committed.

### Transformation — engineering pass 2026-08-24

Implementation checkpoint: `911ea47`.

Immutable artifact:

- path: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t2-conditional-identity-transport-features-v1.json`
- SHA-256: `fc7e36306f4570e8c93b788f8471cd11d3fe865a1e8dc4699c673e8334aa281c`
- module SHA-256: `138c4e7f5719f035fd1cd4ce188758d1f37d80ce156e458547b86cbd9c8dc3b9`
- transformer SHA-256: `e1f778d4b193c20c86b24b77f758ff2820b7a5205e1cbfd9048591c3af17dbf6`

Every frozen engineering gate passes:

| Gate | Requirement | Observed | Result |
| --- | ---: | ---: | --- |
| Two core values vary | Required | 2/2 | Pass |
| Conditional similarity nonzero | ≥50% | 54.33% | Pass |
| Core-to-core absolute Spearman | <0.90 | 0.1024 | Pass |
| Conditional-similarity/coverage absolute Spearman | <0.90 | 0.8204 | Pass |
| Maximum correlation with existing 34 inputs | <0.98 | 0.4677 | Pass |
| Conditional range and zero semantics | Exact | Exact | Pass |
| Wall time | ≤60s | 0.135s | Pass |
| Peak RSS | ≤256 MiB | 63.387 MiB | Pass |

The conditional value has median `0.52208`, interquartile range `[0, 0.68939]`, and
range `[0, 0.89872]`. Its 45.67% zero fraction is the honest consequence of missing
bidirectional support; among supported boundaries, it now represents appearance
similarity rather than support magnitude. The conditional/coverage correlation remains
material at 0.8204, so future results must still report coverage, but the two are no
longer near-duplicates.

`appearanceTransportSwapMargin` and conditional similarity correlate only 0.1024.
The strongest relationship to an existing current-model input remains the swap
margin versus `minimumNearSupport` at -0.4677, comfortably inside the novelty gate.

The exact dormant future model profile is implemented as
`boundary-union34-plus-conditional-identity-transport-t2`: the current ordered 34
inputs followed by only the two frozen T2 core values. It deliberately excludes
`transportCoverageMinimum`. No opened-label runner is wired to this profile.

Decision: **T2 passes label-free engineering and is validation-ready, but is not a
model winner**. This result validates representation independence, not precision,
recall, feature importance, or event F1.

### New-data evaluation — not run; waiting for eligible labels

No eligible new side-switch gold appeared during this loop. The repeatedly opened 50
markers remain outside the transformation and model-selection path. The next action is
data collection/freeze, followed by one untouched T0-versus-T2 nested comparison under
the already frozen gates. Do not port T2 or tune a threshold before that result.

### Opened-development training amendment — authorized and frozen 2026-08-24

The user explicitly requested that T2 be trained now to determine whether the two
engineering-ready values make a measurable difference. This authorizes one
**exploratory development** run on the existing 50 markers; it does not reclassify
those markers as new validation data and does not authorize promotion or a runtime
port.

Freeze the comparison before reading any T2 label result:

- use exactly the 624 adjacent-rally boundaries and all 50 opened markers;
- T0 is the matched ordered 34-input boundary control;
- T2 appends exactly `appearanceTransportSwapMargin` and
  `conditionalCrossSideIdentitySimilarityMinimum`;
- preserve square-root class balancing, L2 `0.1`, top-2/2x per-recording hard-negative
  mining, nested recording-held-out threshold selection, and the current decoder;
- report outer-held row AP/Brier, +/-4-second and strict event metrics, all 11
  recording deltas, and the existing recording-robustness rule; and
- union each boundary branch with the exact E0 outer-held internal selections only as
  a non-calibrated composition diagnostic. The boundary-only result selects the
  exploratory decision.

The frozen target slices are:

1. **covered boundary misses:** positive candidate rows not selected by T0, with T2
   counted as useful only if it recovers at least one; and
2. **weak-transport false boundaries:** T0-selected false rows where either T2 core
   value is at or below that value's median across all 624 label-independent boundary
   rows. T2 must retain fewer of these rows.

The descriptive pass gate remains at least `+2.0` percentage points outer-held
F1, no more than `2.0` points lost in precision or recall, no more than `1.0` point
lost in strict F1, both target-slice checks, and recording robustness. Passing this
gate means only "promising on opened development data."

Feature-importance reporting is also frozen before the run. Report each T2 value's
full-development standardized logistic coefficient, absolute-coefficient rank among
all 36 inputs, and sign consistency across the 11 outer-fit models. Run the two
single-value additions (`34 + swap margin` and `34 + conditional similarity`) only as
descriptive ablations; they cannot replace, prune, or tune the frozen two-value T2
comparison.

Output path:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-feature-development-t2-conditional-transport-opened-v1.json`.
