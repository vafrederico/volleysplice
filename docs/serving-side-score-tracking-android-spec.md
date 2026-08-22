# Serving-side score tracking and video overlay: browser contract and Android spec

Status: production web implementation complete; native Android implementation pending.

This document is the normative product and parity contract for porting the production
browser's serving-side inference, score editor, timeline markers, model-feedback data,
preview, and encoded score overlay to Android. It describes the behavior that exists in
`prod/` as of the `SERVSIDE237-FLIGHT` deployment. Research alternatives under
[`docs/research/`](research/) are historical evidence and must not override this spec.

## Scope and non-goals

Android must implement:

1. fixed-flight serving-side inference for every included merged production interval;
2. the hybrid serve gate and review reasons;
3. durable, editable serve and side-switch markers;
4. deterministic score reduction and point history;
5. the conditional editor UI and guided-tour steps;
6. source-timeline marker rendering and interaction;
7. optional player preview and MP4 score overlay;
8. model-feedback schema v3 export/import parity; and
9. cross-platform fixtures for features, inference, scoring, and rendering state.

This version does not infer side switches, set boundaries, match format, serving player,
penalty points, or the winner of the final rally without a later serve. It does not put
the point-history rail into the encoded video. It does not add serving-side columns to
the existing F104/520 rally model input.

## Canonical browser sources

| Concern | Browser authority |
| --- | --- |
| Runtime identity, feature names, ranking, logistic runner, gate | `prod/src/lib/on-device/serving-side-model.ts` |
| Raw 82 + 155 feature extraction and orchestration | `prod/src/lib/on-device/serving-side.ts` |
| Cache identity | `prod/src/lib/on-device/serving-side-cache.ts` |
| Runtime artifact | `prod/public/runtime/serving-side-85bc3325fbd4.json` |
| Score schema and reducer | `prod/src/lib/score-tracking.ts` |
| Editor score controls | `prod/src/components/ScoreTrackingPanel.tsx` |
| Editor/timeline integration | `prod/src/components/CutEditor.tsx` |
| Overlay snapshot and layout | `prod/src/lib/score-overlay.ts` |
| Encoded Canvas compositor | `prod/src/lib/on-device/export.ts` |
| Persistence/migration | `prod/src/lib/cut-draft.ts`, `prod/src/lib/project-store.ts` |
| Feedback export/import | `prod/src/lib/model-feedback.ts`, `prod/src/lib/model-feedback-import.ts` |
| Guided tour | `prod/src/components/GuidedTour.tsx` |

If prose and code disagree, stop the port and resolve the contract in both places. Do
not silently make Android behave differently.

## End-to-end lifecycle

For a new project, the required order is:

```text
decode selected game window and produce F104 rows
  -> run both three-head rally bundles
  -> retain both serve probability arrays and decoded contacts
  -> build the overlap-union production intervals and agreement provenance
  -> run the suppression specialist
  -> open/reuse the source and sample serving-side windows at every included interval start
  -> compute all raw 237-column rows
  -> tied-rank every column over the complete candidate set
  -> run fixed-flight v3 and the hybrid gate
  -> persist the raw matrix, model identity, evidence, and candidate verdicts
  -> create/open the editor and seed model serve markers
```

Serving-side generation is part of initial inference, before a new project's editor is
shown. It is not controlled by the score-UI toggle. The toggle controls presentation,
not whether the durable model output exists.

The browser treats a serving-side extraction failure as recoverable: it reports the
failure, still opens the editor with the completed rally/suppression analysis, and offers
a rerun when the source and both production serve outputs are available. Android may use
the same recovery policy, but must never present missing serving-side output as a
successfully populated cache.

For a compatible older project without serving-side output, Android may run this stage
after source relinking if both production serve outputs are retained. After a successful
run it must persist the result. Ignored intervals, disabled rallies, suppression changes,
team names, verdict corrections, and side-switch markers must never rerun feature
extraction.

## Candidate and cache identity

The candidate anchor contract is
`merged-production-interval-start-v1`:

- take every valid included production interval;
- sort by `start`, then `end`, then stable interval ID;
- use `interval.start` as the source-time anchor;
- retain the interval ID, start, end, and agreement on the output candidate; and
- require unique, stable non-empty IDs.

The durable output is reusable only when all of the following match:

- model ID `serving-side-fixed-flight-v3`;
- fingerprint
  `85bc3325fbd43abba6ba3726ac091dc6a3a1eafc68d63d979cb2a7450eb49e06`;
- feature version `SERVSIDE237-FLIGHT`;
- anchor contract `merged-production-interval-start-v1`;
- 237 columns and one raw row per candidate; and
- every candidate's ID, anchor, bounds, and agreement exactly match the current ordered
  production intervals.

The serving-side cache is separate from the F104 feature cache. Its identity must also
include source identity, display geometry/rotation, normalized ROI, analysis window,
decode/runtime variant, and ordered feature signature. Editor ignored ranges are
deliberately not part of this identity.

## Runtime artifact and parser

Android must package parameters equivalent to the checked-in browser asset. The browser
asset's file SHA-256 is
`14f18bf0b0f326ccd7ef4b3d614a96a53dd9675df61813fd375677489d0e5a7c`.
At load time validate:

- schema version `1` and kind
  `volleycut-serving-side-fixed-flight-runtime-v1`;
- exact model ID, fingerprint, and feature version;
- exact eight court-flow and nine flight offsets;
- resize `192×108` and flight grid `4×6`;
- all 237 names in exact order;
- 237 imputation, mean, scale, and weight values;
- finite bias/L2/threshold values and strictly positive scales; and
- exact side threshold, review band, and hybrid-gate policy.

A mismatched artifact is a hard failure, not a best-effort migration.

## Frame sampling

All timestamps are seconds from the original source. Crop in display-oriented source
coordinates using the project's normalized ROI, resize to `192×108`, and stretch the ROI
to that size (`fit: fill`). Sample at `anchor + offset`, clamped to
`[0, duration - 0.01]`. Deduplicate equal clamped timestamps across candidates before
decode where practical.

Court-flow offsets:

```text
[-1.25, -0.75, -0.35, -0.10, 0.10, 0.30, 0.55, 0.85]
```

Flight offsets:

```text
[-0.15, 0.05, 0.20, 0.35, 0.55, 0.80, 1.10, 1.40, 1.75]
```

Frame selection must be specified and parity-tested. If native MediaExtractor/MediaCodec
cannot return the same visual sample as browser `CanvasSink`, record the timestamp/frame
selection delta and prove final score/verdict acceptance on the release corpus rather
than assuming compatibility.

## Court-flow features: first 82 columns

Use the browser's exact generated names with the `v2:` prefix. Compute Farnebäck flow
with `(0.5, 2, 13, 2, 5, 1.1, 0)` for these frame pairs:

- `pre`: `0→1`, `1→2`;
- `contact`: `2→3`, `3→4`, `4→5`;
- `post`: `5→6`, `6→7`.

For each pair:

1. compute the median x and y flow over the complete 192×108 image;
2. subtract that translation from every flow vector;
3. mark residual magnitude at least one pixel active;
4. apply a 3×3 morphological open;
5. use 8-connected components; and
6. summarize the bottom 32% image band as `near` and top 32% as `far`.

The frozen runtime does not use annotated service-zone ellipses. Introducing calibrated
geometry would change the production feature distribution and requires a new model and
feature version.

For each phase and side emit `flowMean`, `flowP90`, `activeFraction`,
`largestComponentFraction`, `componentCountDensity`, `centroidX`, `centroidY`, `flowX`,
and `flowY`; then emit the exact near-minus-far and phase-delta bank generated by
`courtFlowFeatureNames()`. The count must be 82 and every raw value must be finite.

## Fixed-flight features: final 155 columns

Use the browser's exact generated names with the `flight:` prefix. Compute eight
Farnebäck fields with `(0.5, 3, 15, 3, 5, 1.1, 0)`. For each field:

1. fit x and y affine flow from normalized x/y plus a constant on the declared sparse
   sampling grid;
2. compute sparse residual magnitudes;
3. retain the 75% smallest residuals and refit when at least six sparse samples remain;
4. subtract the fitted affine field at every pixel;
5. normalize x/y by image width/height and magnitude by the image diagonal;
6. set the energy threshold to `max(0.00075, p90(normalizedMagnitude))`; and
7. define energy as `max(normalizedMagnitude - threshold, 0)`.

Average pair summaries into `launch` (pairs 0–2), `early` (3–5), and `late` (6–7).
For each phase emit 24 normalized 4×6 cell energies, four energy-weighted row vertical
flows, and the 15 global values listed in `FLIGHT_GLOBAL_STATISTICS`. Then emit the 13
declared changes for `launchToEarly` and `earlyToLate`. The count must be 155 and every
raw value must be finite.

## Ranking and classifier

Concatenate the 82 and 155 raw banks in that order. Only after all candidate rows are
available, replace every column with stable tied percentile midranks:

```text
rank = ((first_equal_position + last_equal_position) / 2) / (rowCount - 1)
```

Positions are zero-based. Equal values receive the same rank. A one-row recording uses
`0.5` for all columns. No non-finite raw value is accepted.

For each ranked row:

```text
filled[i] = ranked[i] when finite, otherwise impute[i]
z         = bias + sum(((filled[i] - mean[i]) / scale[i]) * weight[i])
nearProb  = sigmoid(clamp(z, -30, 30))
side      = near when nearProb >= 0.4783744762021848, otherwise far
```

The fitted logistic model always returns a physical `near` or `far` side. It never
returns `not-serve` and does not use audio, human review labels, or serve-head evidence
as model inputs.

## Hybrid serve gate and verdict

At each anchor, inspect each production bundle's aligned serve probability samples whose
absolute distance from the anchor is at most one second. The boundary is inclusive. Use
the first occurrence on a tied maximum. If the window has no sample, use the nearest
sample. Record for each head its model ID, threshold, peak probability/time, threshold
crossing, and nearest decoded contact.

Apply the gate:

1. Either head peak `>= 0.85`: `isServe=true`, source `serve-head`.
2. Otherwise agreement `both-models`: `isServe=true`, source
   `production-rally-recovery`, add mandatory review reason
   `production-rally-recovery`.
3. Otherwise: `isServe=false`, source `none`, verdict `not-serve`.

Independently add review reason `side-score` when
`0.3121748736511044 <= nearProb < 0.5028396703865513`.

For a serve, verdict is `review` when any review reason exists; otherwise it is the
underlying `near` or `far`. Preserve the underlying side and probability even under
review. The gate must never change rally intervals.

## Score state and marker seeding

The persisted score schema is version 2:

```text
enabled: Boolean (default true)
team1Name: non-blank String (default "Team 1")
team2Name: non-blank String (default "Team 2")
serveMarkers: ordered editable markers
sideSwitchMarkers: ordered manual markers
removedModelMarkerIds: unique tombstones
```

A serve marker contains stable ID, source timestamp, editable side
`near|far|review`, origin `model|manual`, optional immutable `modelSide`,
`ignorePreviousPoint`, and optional production rally ID.

Seed one model marker per candidate except an uncorrected `not-serve` candidate:

- automatic `near`/`far` verdict: marker side and `modelSide` are that side;
- `review` verdict: marker side and `modelSide` are `review`;
- marker ID: `serve-<candidateId>`; timestamp: candidate anchor; rally ID: candidate ID.

Reapplying cached output must preserve user side corrections and replay flags, preserve
manual markers, honor model-marker tombstones, and never resurrect a removed prediction.

## Deterministic score reducer

Team identity is anchored to the initial court layout: Team 1 starts near and Team 2
starts far. Sort serves and switches by timestamp, then ID.

- The first visible serve establishes the server and awards no point.
- Each later serve identifies the winner of the preceding rally: the team serving next
  receives one point.
- `ignorePreviousPoint=true` on that later serve records an ignored/replayed point and
  awards nothing.
- `side=review` records an unresolved point and awards nothing.
- Count side switches with `switch.timestamp <= serve.timestamp`; odd parity reverses
  near/far team identity and even parity restores it.
- A side-switch marker awards no point by itself.
- Without a later serve marker, the final rally remains unawarded. The editor must allow
  a manual marker after it.

This reducer is intentionally rules-light. It does not enforce a score limit, win-by-two,
sets, rotations, or sanctions.

Before reduction, remove serve markers whose timestamps fall in ignored source intervals
and model-linked markers whose rallies are disabled or effectively suppressed. Filtering
must not mutate cached inference. Manual markers without a rally ID survive rally filters.

During dead time and a retained rally's leading padding, preview/export uses the closest
next visible serve timestamp as the score boundary. During the core and trailing padding,
use the real source timestamp. This prevents a completed point from advancing early
during the rally while keeping pre-serve padding aligned with the upcoming rally state.

## Editor and timeline requirements

Score tracking is labeled **BETA**, defaults enabled, and is persisted per project. When
disabled, hide every score-specific control/marker and let the desktop video/editor use
the reclaimed width. The score controls appear in a box above the video on desktop and
remain above the player in the mobile flow.

Required controls:

- editable Team 1 and Team 2 names;
- current score and current server;
- selected predicted/manual serve correction to Near or Far (no separate Review button);
- ignore/replay previous point;
- add missing serve at current timestamp with selected side;
- add side switch at current timestamp;
- list all markers with seek and remove actions; and
- point timeline above the marker list with rally number plus one rail per team showing
  that team's running point number.

Timeline rendering:

- serve marker: thin vertical line with small ball icon;
- side switch: thin vertical line with small two-arrows icon;
- marker lines sit visually above ranges but never intercept seek/drag gestures;
- the icon remains clickable and seeks/selects its marker;
- tapping or dragging elsewhere on the rail retains normal timeline behavior; and
- predicted serve markers are removable exactly like manual markers.

The guided tutorial must conditionally include score steps only when score tracking is
enabled. Explain the beta switch, score panel, marker semantics, and the final-video
overlay toggle.

## Final-video overlay

`renderScoreOverlay` is a per-project Boolean, defaults false, and is shown only when
score tracking is enabled. Place it indented under **Play final cut only**. Enabling it
must immediately show the overlay over the editor player and include it in the next MP4
export. The UI must not claim a material speed difference relative to the normal export;
both current platform exporters already encode their requested output.

The overlay is one top-left row:

```text
Team 1 name | 00 | Team 2 name | 00
```

- outer black 1–2 px border;
- only the bottom-right corner rounded;
- Team 1 red `#d9342b`, Team 2 blue `#2367c9`;
- team text white, centered, and name cells grow with measured text;
- score cells white with black centered text;
- scores padded to at least two digits; and
- total overlay width capped at 96% of the display/video width, shrinking name cells
  proportionally but preserving their minimum size when necessary.

The Android preview renderer and Media3 export effect must consume the same pure score
snapshot/layout rules. At export queue time, freeze the score state, filtered rally IDs,
ignored intervals, active rally ranges, names, and render preference into the job. Map
each retained clip's local presentation time back to original source time before score
lookup. Do not read a mutable editor draft from the foreground service.

The browser bakes display rotation into overlay frames and writes rotation zero on that
path. Android must verify 0°, 90°, 180°, and 270° inputs and avoid applying rotation twice.

## Persistence and model feedback

Android must version and atomically migrate its native project/editor schemas. Existing
projects default to score tracking enabled and final-video rendering disabled without
losing cuts or suppression state.

Android's current feedback exporter is schema v1. This feature requires parity with
`volleycut-model-feedback` schema v3:

- retain both component serve outputs;
- export serving-side model identity, raw float64 237-column matrix, anchor contract,
  candidates, probabilities, review reasons, and both heads' evidence;
- export complete score state, tombstones, excluded rally IDs, and derived point history;
- preserve ignored ranges and final materialization provenance; and
- import the same data into a new local project without rerunning inference.

Large numeric arrays use base64 little-endian encoding and explicit dtype/shape as
documented in [`../prod/docs/model-feedback-bundle.md`](../prod/docs/model-feedback-bundle.md).
Raw video bytes are never part of the bundle.

The `renderScoreOverlay` preference is an editor/export preference and is included in
the edit-list JSON. It is not currently part of feedback schema v3; Android should match
that behavior unless the schema is deliberately revised on both platforms.

## Android implementation map

Suggested native modules:

- `ServingSideRuntime`: strict JSON parser and immutable parameter model;
- `ServingSideFeatureExtractor`: sampled grayscale windows, 82/155 raw banks;
- `ServingSideModelRunner`: tied ranks, logistic score, hybrid gate;
- `ServingSideCache`: atomic raw features/evidence bound to source and candidates;
- `ScoreTrackingModels` and `ScoreReducer`: persisted state and pure source-time reducer;
- Compose `ScoreTrackingPanel` and timeline overlay;
- `ScoreOverlayLayout`: platform-independent numeric layout primitives;
- Compose preview overlay; and
- Media3 `CanvasOverlay`/`OverlayEffect` connected to `ExportService`.

Integrate initial inference in `ProjectAnalysisService.kt`, persistence in
`NativeProjectStore.kt`/`EditorDraftStore.kt`, editor UI in `EditorActivity.kt`, feedback
in `ModelFeedbackExporter.kt`, tutorial in `GuidedTour.kt`, and export rendering in
`ExportService.kt`. Reuse the existing source grant, ROI, analysis window, production
ensemble, serve arrays, and export interval snapshot.

## Acceptance gates

The Android implementation is complete only when all of these pass:

1. Exact 82, 155, and 237 generated name/order tests.
2. Runtime asset identity/hash and malformed-artifact rejection tests.
3. Raw feature fixtures covering normal anchors plus source-start/end clamping.
4. Stable tied-rank parity including ties and a singleton candidate.
5. Frozen logistic score parity (the all-0.5 row produces
   `0.5428339645988896` within numerical tolerance).
6. Inclusive ±1 second serve evidence, first-peak tie, and both-model-only recovery tests.
7. Cache invalidation for model, feature version, source, ROI, window, and candidate
   identity; no invalidation for ignored editor ranges.
8. Score tests for first serve, repeated winners, review, replay, equal-timestamp switch,
   disabled/suppressed rallies, ignored ranges, final missing serve, and dead-time bounds.
9. Project migration, marker tombstone, edit-list, and feedback v3 round-trip tests.
10. Compose UI tests for conditional visibility and normal timeline gestures through
    marker lines.
11. Overlay layout/golden tests for SD/1080p/4K, long names, scores above 9, colors,
    centering, width cap, and rotation.
12. Encoded MP4 tests for A/V sync, duration, source-time score changes across multiple
    retained intervals, cancellation cleanup, and playability.
13. Browser/Android comparison on the same physical files, ROI, game window, production
    intervals, and candidate anchors, with every material deviation recorded.

Do not ship Android serving-side scoring by copying only the model JSON or by deriving
points from unfiltered verdicts. Feature generation, candidate identity, gate evidence,
editor correction semantics, filtering, persistence, and export-time source mapping are
one product contract.
