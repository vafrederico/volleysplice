# Serving-side hybrid serve gate — 2026-08-21

## Decision

The serving-side results workflow now uses a conservative second-stage production
rally fallback. The fixed-flight side classifier still always emits `near` or `far`.
The separate serve gate exposes that side when either:

1. either frozen production serve head reaches its deployed `0.85` threshold within
   one second of the candidate anchor; or
2. both serve heads miss, but the anchor is contained in a production rally interval
   supported by both production models.

The second path is recorded as `production-rally-recovery` and always enters the UI
review queue. A candidate with neither form of evidence remains `not-serve`.

This policy was chosen after examining the current all-video corrected labels,
including the historically opened protected recording. It is an assisted review-UI
policy, not a development-selected model improvement or a production rally-model
promotion. The artifacts record that limitation explicitly.

## Current-label result

The comparison uses the same 1,114 candidates, 30 recordings, eight current
human `not-serve` labels, correction overlay, and source-quality exclusions as the
previous results UI.

| Metric | Serve heads only | Hybrid gate |
| --- | ---: | ---: |
| True serves | 1,050 | 1,084 |
| Missed serves | 56 | 22 |
| False serves | 3 | 4 |
| True not-serves | 5 | 4 |
| Serve precision | 99.715% | 99.632% |
| Serve recall | 94.937% | 98.011% |
| Non-serve recall | 62.500% | 50.000% |

The hybrid path recovers 35 rows: 34 true serves and one current human non-serve.
The fixed-flight side output is correct for 30 of the 34 recovered true serves
(88.235%). The side-confidence policy marks two of the recovered rows independently;
the union of side uncertainty and serve recovery contains 57 candidates.

## Versioned artifacts

- Evidence:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/serving-side-serve-gate-v2/all-reviewed.json`
  (`1,395` rows, SHA-256
  `9ac4071f125030b0a8f1de037e48aa48b73f30f71fa6df8a302b70ed2db5344f`).
- Matching fixed-flight inference:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v3-hybrid-serve-gate-all-video-inference-v2.json`
  (`1,114` rows, SHA-256
  `a7135cd509df3b0063e4634c99d314afb06d1df5552e292b5c17339a1671e722`).
- Hybrid gate fingerprint:
  `21395cc10390eefd14e120b3d7078191ddc9bb7ba1b6852fa4471658a7ff7af0`.

Every standard recording binds both source production `analysis.json` files. Raw
model-feedback recordings bind the exact stored initial production-ensemble ranges.
The evidence generator records every source path and SHA-256 and refuses to overwrite
the versioned output.
