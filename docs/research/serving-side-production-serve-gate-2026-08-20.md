# Production serve heads as a serving-side gate — 2026-08-20

## Decision

Use both frozen production serve heads to answer **Is this a serve?** in the
serving-side results workflow. A reviewed candidate is a serve when either head
reaches its existing production threshold within one second of the source-aligned
candidate anchor. Otherwise its final prediction is `not-serve`; candidates that
pass the gate keep the frozen serving-side v2 `near`/`far` prediction.

This is an auditable review-UI gate, not a promotion to production rally
suppression. The policy and thresholds were selected without consulting the serving-
side labels or protected test results.

## Frozen gate contract

- Version: `production-dual-serve-head-anchor-window-v1`
- Anchor window: ±1.0 second
- Aggregation: either head at its deployed threshold
- All-labels V2 head: `model-1ca43e38eefc`, threshold `0.85`
- Previous production head: `model-9c92b8e9333f`, threshold `0.85`
- Gate fingerprint:
  `68de299816c29597c4bd2320b3827bff8b09807fbd3cdc8f4111267c4db8384a`

The extractor applies each exact runtime logistic head and production serve decoder
to source-aligned feature matrices. It records both heads' peak probability, peak
time, threshold decision, and nearest decoded detection for every candidate.

Feature coverage spans all 30 available videos: 19 use canonical
`audiovisual-audio-normalized-v3` caches and 11 use the corresponding source-aligned
model-feedback feature matrices. Each source path and SHA-256 is recorded in the
evidence artifact.

Implementation:

- [`analysis/production_serve_gate.py`](../../analysis/production_serve_gate.py)
- [`scripts/extract-serving-side-serve-gate.py`](../../scripts/extract-serving-side-serve-gate.py)
- [`scripts/infer-serving-side-v2-all-videos.py`](../../scripts/infer-serving-side-v2-all-videos.py)

## Current-label diagnostic

The current correction overlay contains 1,126 human serves and three human
`not-serve` candidates among the 1,129 clear serving-side rows.

| | Human serve | Human not-serve |
| --- | ---: | ---: |
| Predicted serve | 1,064 | 1 |
| Predicted not-serve | 62 | 2 |

That is 99.91% serve precision and 94.49% serve recall. The negative-label sample is
too small for a stable specificity estimate. In particular,
`indoor-source-02:rally:9` remains a false serve because both heads score it
above 98%. The results UI exposes the 62 gate misses, the one false serve, both head
scores, and the current human label so these errors can be reviewed and corrected.

The recall cost is material. The production pipeline retains fallback behavior for
missed contacts, so this candidate-level gate should not become a hard production
suppression rule without a development-only policy iteration and broader negative
label coverage.

## NAS artifacts

All paths are relative to
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/`.

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| `features/serving-side-serve-gate-v1/all-reviewed.json` | 1,424 | `41cf930cc064d3ba7e5c2198b23e6cdb92bdea99036d2a88d753ec717b07331a` |
| `reports/serving-side/serving-side-specialist-v2-dual-serve-gate-all-video-inference.json` | 1,129 | `5f859b81faf0ed345ab68d94dbec86ce4757293fa5fb1a8684f4be7454945510` |

The evidence artifact covers clear and unclear report candidates so future label
changes do not require rerunning either serve head. The combined inference artifact
binds the evidence hash, serving-side model and datasets, frozen labels, and
implementation hash.

## Reproduction

The commands refuse to overwrite existing artifacts. Choose a new versioned output
path when intentionally creating a new immutable revision.

```bash
npm run extract:serving-side-serve-gate
npm run infer:serving-side-v2-all-videos
```
