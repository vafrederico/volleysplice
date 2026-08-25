# Side-switch T14 dominant tracklet medoid — 2026-08-25

## Decision and isolation

T10–T13 show strong player-unit separation but insufficient clean direction once route
confounds are removed. T14 tests whether secondary player roles, especially a libero,
make equal-unit team matching unstable. It represents a side by one dominant jersey
chosen from stable player tracklets.

Preserve exact T11 videos, detector ownership, frames, jersey observations, T3
tracklets/qualification, T5 empty-side fallback, candidates, labels, model protocol,
and decoder. For a nonempty stable set, choose the tracklet whose descriptor minimizes
the unweighted sum of Hellinger distances to all stable tracklet descriptors; break
ties by existing stable-tracklet order. Use that exact tracklet descriptor and
reliability without averaging. For an empty stable set, use the exact single T5 pooled
fallback. Compare the resulting one-unit teams by Hellinger distance.

The model core is exactly the three familiar direction semantics with prefix
`dominantTrackletJersey`: swap margin, reliable swap evidence, and reliable continuity
evidence. Cross/same similarity, reliability, separation, and gates remain diagnostics.

## Immutable input and label-free gates

Pin T13 artifact SHA-256
`232a26201edf16fe104dec965e73bb9f1864162baa13749187b0d3ad4970972e`.
T13 is a transformation and does not duplicate top-level video audit records, so load
T11 SHA-256 `aa0b6abfa6324350b7944728757ee2867e7ab6c35eadfc1e085f73339e74a6d5`
only for its immutable `extractionAudit` video paths, ROI, and court geometry. T13
remains the sole row/prior-feature source; no T11 feature value is recomputed or added.
Output:
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t14-dominant-tracklet-medoid-features-v1.json`.

T14 reaches labels only if all gates pass:

- exact 704-row prior parity, exact T10 stable-tracklet counts, and exact T5
  availability;
- deterministic medoid selected for every nonempty stable side and fallback only for
  empty stable sides;
- at least one multi-tracklet side and mean medoid-to-pooled descriptor distance >=0.01,
  proving the representation is active;
- mean minimum team separation >=`0.37386011658617885`;
- positive margins >=`25.48076923076923%` and reliable swap >=`19.391025641025642%`;
- all three core values finite/nonconstant and maximum absolute core/core, core/T4
  through core/T13, and core/existing Spearman below `0.98`;
- 635 endpoints, 3,175 frames, 25,400 tile calls, no errors, <=60 minutes, and
  <=2,048 MiB RSS with four recording workers; and
- no label, audit, feedback, or model artifact loaded.

Failure stops before labels. Do not tune medoid weighting, add a distance radius,
change fallback, select a second role, or alter gates after observing T14.

## Conditional model comparison

If engineering passes, commit exact boundary T0 (`34`) versus `34 + T14 three-value
core` wiring before labels. Apply +2 F1 over T0, +0.5 over T4, all prior guardrails,
and positive reliable-swap coefficient in the full fit and >=9/11 outer fits.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T14 code, artifact, feature value, model profile, or model result existed when this
contract was committed.
