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

### Engineering result — pass; model comparison authorized

The video-audit clarification is committed at `aaa3dcb`; implementation is committed
at `387cbe6`. The four-worker extraction wrote the immutable 39,577,763-byte artifact:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t14-dominant-tracklet-medoid-features-v1.json`

SHA-256: `d68e0d9a2aa72e1fe3cb06f413fdb43dacd52b9090dd4241863596459f8b9753`.

Every frozen label-free gate passes:

| Engineering measurement | Result | Frozen requirement | Check |
| --- | ---: | ---: | --- |
| Both-team/four-team visibility | 81.1024% / 68.75% | exact T5 | Pass |
| Multi-tracklet / fallback sides | 795 / 45 | both active | Pass |
| Mean medoid-to-pool distance | 0.224519 | >=0.01 | Pass |
| Mean minimum team separation | 0.406364 | >=0.373860 | Pass |
| Positive raw margin | 33.0128% (206/624) | >=25.4808% | Pass |
| Nonzero reliable swap | 25.00% (156/624) | >=19.3910% | Pass |
| Strongest prior-family correlation | 0.642725 | <0.98 | Pass |

Extraction completed 635 endpoints, 3,175 frames, and 25,400 tile calls without
errors in 436.648 seconds (7m16.648s) at 1,183.723 MiB peak RSS.

Decision: **T14 passes engineering and enters the frozen matched model comparison**.
This authorizes only implementation/commit of the exact `34 + T14 core` runner before
labels are opened; it is not a precision, recall, F1, or promotion result.

### Matched model result — reject

The exact 37-input model wiring was committed at `b01bf07` before the label/audit and
model artifacts were loaded. The one-shot opened-development comparison wrote the
946,875-byte artifact:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-feature-development-t14-dominant-tracklet-medoid-opened-v1.json`

SHA-256: `ba6b7493f4d2c98a5d6bd755947a1f917ce4a5d0b69a9cb07c7b48c903d54f06`.

| Boundary-only metric | T0 | T14 | Change |
| --- | ---: | ---: | ---: |
| Row AP | 43.5623% | 44.0401% | +0.4777 pp |
| ±4 proposals / TP / FP / FN | 52 / 26 / 26 / 24 | 55 / 29 / 26 / 21 | +3 / +3 / 0 / -3 |
| ±4 precision | 50.00% | 52.7273% | +2.7273 pp |
| ±4 recall | 52.00% | 58.00% | +6.00 pp |
| ±4 F1 | 50.9804% | 55.2381% | +4.2577 pp |
| Strict F1 | 41.1765% | 43.8095% | +2.6331 pp |

T14 also exceeds immutable T4 F1 by 2.4079 points. It recovers one of five frozen
covered misses, removes three of 20 zero-reliable-evidence false boundaries, and its
three-TP gain remains positive after removing the best recording. Every event,
target-slice, and recording guardrail therefore passes.

The semantic coefficient gate fails decisively. Raw transport margin is positive in
the full fit (`+0.105827`) and all 11 outer fits, but reliable swap is negative in the
full fit (`-0.077673`) and all 11 outer fits; reliable continuity is likewise negative
in all 11 (`-0.084315` full fit). The required reliable-swap count was at least 9/11,
and the observed count is 0/11.

Decision: **reject T14 despite the F1 gain**. The result says the medoid raw direction
is useful, but the preregistered reliability semantics are contradicted consistently.
Do not prune the bundle, reverse signs, tune gates, promote, or port it on these opened
labels. Any continuation must be a separately preregistered representation rather
than a T14 ablation.
