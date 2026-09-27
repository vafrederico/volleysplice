# Full-video web model-switch validation

Validated 2026-09-27 against production build `360be998`, using the complete
`recording-044` (1,061.016 seconds). Private media and raw receipts remain outside
Git and resolve through the external ledger. This is a functional UI/inference
validation, not a model-selection or human-label accuracy evaluation.

The real built React application ran in an isolated desktop Chrome
153.0.8010.48 profile using WebGPU, production CSP, and no cross-origin isolation.
Runs were sequential; the Android emulator was off. Source video, browser profile,
and artifacts were on the configured research storage.

| Run | Input cache state | Rallies ready | All results ready | Rallies | Serving-side results |
| --- | --- | ---: | ---: | ---: | ---: |
| Balanced | Cold image, motion and audio caches | 3m 50s | 5m 30s | 34 | 34 |
| Maximum coverage | New encoder; motion/audio reused from Balanced | 1m 55s | 3m 36s | 37 | 37 |
| Balanced forced rerun | Image, motion and audio caches reused | 2.4s | 1m 40s | 34 | 34 |

All-results times include serving-side and court-side-switch inference. The latter
completed with zero detected switch candidates in each run. Rally-ready times are
derived from rendered progress transitions; all-results completion was polled at
2.5-second intervals. These single desktop measurements have different cache
conditions and are not comparative cold-start model benchmarks or phone timings.

## Checks passed

- Full source bounds were used for both variants; the selected model identity was
  preserved and legacy suppression did not replace neural rally cuts.
- Both fresh encoders processed 2,123 sampled images. The UI captured 2,122
  changing image-count states in each run; progress advanced into motion/audio and
  then rally classification.
- Each rally-classifier run displayed 34 distinct increasing progress values,
  from 12% through 95%, before final completion. It did not remain fixed at 42%.
- Loading, image counts, classifier progress, and cached-image reuse were visible
  in the real progress panel. Both fresh runs opened the editor with playable
  video, rally timelines, and serve markers.
- Every detected rally received a serving-side result. This verifies coverage of
  the serving pipeline, not the accuracy of its predicted side.
- Selecting the already analyzed Balanced variant through the normal UI reopened
  its saved result in 68 ms without rerunning inference.
- A separate forced rerun evicted only the saved Balanced project from the
  isolated test profile while retaining feature caches. It made no image-encoder
  asset request and emitted no image-processing updates. Rally boundaries and
  all rally probabilities matched the original Balanced run exactly.
- No unhandled browser errors occurred. The temporary browser and loopback test
  server were closed after completion.

The forced rerun is test setup, not a user-facing force-rerun control. Its remaining
time is predominantly serve/court-side processing; ordinary switching to a saved
result reuses those results as well.

## Reproduction

Use [the built-app validation runner](../../scripts/validate-production-web-model-switch.mjs).
Supply `--source`, `--output`, `--build`, `--harness`, `--browser`, and
`--recording-index` through ledger resolution and ignored local configuration.
The runner uses an isolated browser profile, serves the built application only on
loopback, drives its normal file/model controls, records rendered progress, and
verifies persisted analysis. It refuses to overwrite a completed validation.

No additional production-code fixes were required by this run. Mobile Chrome,
WASM fallback performance, and other recordings were not retested here.
