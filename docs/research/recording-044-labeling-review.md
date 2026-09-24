# recording-044: compact boundary review in the labeling tool

The user requested importing this recording and its corrected model feedback as human labels, then exposing the current production-preserving compact boundary recommendation and its signals as a separate labeling reference.

## Scope

- Source: `private-reference-0196`, with matching `.model-feedback.json`.
- Recording: `recording-044`; outdoor grass, 1,061.016489 seconds, 1920x1080, approximately 60 FPS with variable source timestamps.
- Artifacts: `private-reference-0117`.
- Import corrected core ranges after inclusion/suppression decisions, manual additions, ignored intervals and score-marker corrections. Preserve feedback provenance; imported reviewed cuts are not independently verified serve-to-dead-ball annotations.
- Predictions use the existing compact short-boost four-head TCN. Fix the established engineering checkpoint choice, seed 3407 / outer-0, before evaluation on this recording. Reuse validated 104-column feedback features at their actual source timestamps. No retraining or label-driven model selection.
- Reproduce the same checked-in production default used in the preceding experiment, independently of human suppression overrides. Keep its export coverage fixed while showing proposed event boundaries and whole-parent review flags.
- Show original predictions and review guidance, not simulated perfect-human output, as model layers. Serving-side inference is added in the follow-up below; reconstructed-score predictions remain outside this experiment.

## Checklist

The Tasks tool is unavailable; this checklist tracks the work.

- [x] Locate the source/feedback and establish the import, inference and UI contracts.
- [x] Prepare and verify imported human labels and media identity.
- [x] Run frozen production and compact inference and prepare review signals.
- [x] Register the recording and references without replacing existing labels or catalogs.
- [x] Verify API access, video byte ranges, rendered signal components, model layers and export accounting. Interactive browser inspection was unavailable.
- [x] Record artifact identities, test results and the usable labeling URL.

## Open and inspect

Open this recording in the labeling tool (resolve the recording index through the private ledger). The root labeling application is running on port 3000 with the study-specific catalog and research references configured in the ignored `.env.local`.

The editable human draft contains 37 retained rally ranges, 36 serve markers, 18 explicit false-positive negatives, and the ignored opening from 0 to 151.528966 seconds. Corrected core timestamps and manual additions are preserved. The imported draft records its feedback provenance; these reviewed export cuts are not fresh frame-exact serve/dead-ball annotations. Existing catalog entries and labels were preserved.

The model selector offers:

- **Production + compact review**: current production rally boundaries, with recommended whole-rally review regions and typed boundary flags.
- **Proposed boundary preview**: provisional initial-start corrections, additional-rally proposals and separate end boundaries. Its export accounting still uses production coverage.

Both views display the shared compact signal panel: live play, serve/start, rally end and keep. Signals retain all 4,245 actual source timestamps; click the graph or review-region controls to seek. Scores are model outputs, not calibrated confidence. The serve/start trace does not predict serving side.

The recommended 10% playback-budget queue selects **9 production regions totaling 84.76 seconds**. **Show all 27 flagged regions** exposes the complete candidate set. Regions are displayed in video order. The two additional-rally proposals, at **4:19.37** and **12:49.25**, are outside the selected nine regions and can be inspected through that toggle. These are unconfirmed suggestions, not automatic score increments.

| Output | Outside ignored opening | Full file |
| --- | ---: | ---: |
| Suppression-adjusted production rallies | 40 | 44 |
| Compact standalone candidates | 34 | 34 |
| Proposed boundary preview | 42 | 46 |

There are 47 boundary flags: 21 initial starts, 24 ends and two additional starts. The selected nine review regions contain five initial-start and seven end flags. Reviewing a region means inspecting the complete production parent, including potential rallies that have no proposal. Selecting a model reference does not edit the human draft.

## Execution and verification

Inference used the frozen compact short-boost four-head TCN (29,700 parameters), seed 3407 / outer-0 / epoch 60. This is an existing research checkpoint, not a newly promoted deployment model. No training or label-driven model selection occurred. The model consumed the 104 expected audiovisual features at their native timestamps; imported human rallies and serve markers were excluded from inference and proposal generation. The existing ignored opening only limits the review universe.

Independent replay matched the saved neural probabilities within 9.84e-7. Unsuppressed production reconstruction exactly matched the original feedback endpoints. Both displayed references retain identical production export coverage at 0, 1, 2 and 3 seconds of symmetric padding and the strict positive-gap-less-than-3-second join rule.

Validation passed: 14 feedback-import tests, six inference/input-isolation tests, five UI/parser/static-render tests, TypeScript and the optimized Next build. The recording page, editable draft and research-reference endpoints returned HTTP 200. Initial and final video byte-range requests returned correct HTTP 206 responses. Computer-use exposed no browser surface, so interactive visual inspection was not performed.

Publication v2 adapts only the label document's protected task envelope to the labeling catalog. All imported annotation content and provenance are unchanged; the original envelope and immutable import remain archived. The 37 original catalog records are unchanged. The new recording is a challenge item and was not added to training.

## Production ensemble follow-up

The user additionally requested a standalone production ensemble run. Its unsuppressed union is distinct from the aggressive whole-rally production baseline used by the compact review recommendation.

- [x] Replay the production component models and suppression from native features, without human labels.
- [x] Add standalone ensemble, component-model and suppression-adjusted references through a derived catalog.
- [x] Verify live API endpoints, unchanged human labels and research references, and record the usable UI selection.

Fresh checked-in TypeScript inference reproduces every prior production endpoint and all nine probability arrays exactly. The run binds the runtime assets, 58 source files (including the 53 production library TypeScript files), and 22 prior artifacts; none changed. Native features and all source timestamps were reused without human decisions. Serving-side and side-switch models were not part of this rally-only replay.

Refresh the recording and select **Production ensemble** under **Model breakdown**. This is the unsuppressed ensemble union. **Suppression-adjusted ensemble** shows the current aggressive whole-rally result. Both component models are available individually; **All model rails** displays them alongside compact guidance.

| Additional model layer | Outside ignored opening | Full file |
| --- | ---: | ---: |
| Production ensemble, no suppression | 52 | 59 |
| Production · All labels v2 | 42 | 48 |
| Previous production | 51 | 56 |
| Suppression-adjusted ensemble | 40 | 44 |

Suppression removes 15 whole ensemble cores, 12 outside the ignored opening. The suppression layer uses these actual policy-gated removals, not the raw suppression head's decoded spans. Its 44 retained cores exactly match the export source of both compact comparison layers. The standalone unsuppressed ensemble uses its own 59 cores for padding and export comparison. The previously reported 40 production rallies referred to the suppressed result outside ignored time.

The live page and reference/draft APIs passed verification. All 37 human rallies and 36 serve markers remain unchanged, as do both compact references and their signal arrays. No frontend code changes were needed for this follow-up. Browser interaction remains unverified because no browser surface was available.

Fresh outputs are under `production-ensemble-v1/`: `ensemble-inference.json`, `core-inference.json`, complete `production-signals.json`/`.npz`, `production-replay.json`, `registration.json` and `receipt.json`. `labeling-catalog-production-v1.json` adds only the target record's `candidateSource.modelEvalInference` field and is now the active catalog. The earlier catalog is preserved. Publication and HTTP checks are recorded in `production-labeling-registration-v1.json` and `production-labeling-ui-verification-v1.json`.

| Follow-up artifact | SHA-256 |
| --- | --- |
| `production-ensemble-v1/ensemble-inference.json` | `145bb92d7d40c0fb01b17dff531b44af85782eb919582ecbddd1624ac9ab4a5a` |
| `production-ensemble-v1/receipt.json` | `ca8a5898266ecf4280bda4beae18deaad481019873f7c5adb9e38b753ffe3a0c` |
| `labeling-catalog-production-v1.json` | `8e9d76922825129b9276fcc9d4b8326ed1f7a85307bf472aa83d3f9bc57c4ca2` |
| `production-labeling-registration-v1.json` | `4b8c5450d4a583afad9303e3d836ee9e1c717f23104974238b3ccb0955ccb4ae` |

## Serving-side and boundary-detail follow-up

- [x] Run the current production serving-side classifier and serve gate on the unedited ensemble anchors, using validated native cached features.
- [x] Display near/far/review/not-serve decisions with source-head evidence; preserve rejected decisions separately from serve markers.
- [x] Explain every boundary flag with its original parent, proposed boundary, correct comparison, source sample scores, and review instruction.
- [x] Make Model breakdown a multi-select for arbitrary combinations of rails.
- [x] Publish new versioned references and verify the live UI/API, meaningful regression tests, unchanged labels and export coverage.

**Model breakdown** now opens independent checkboxes for production, either component, suppression, compact review, boundary preview and any available Sol reference. **All** selects every rail; **Clear** leaves only the human rail. Production and compact review are selected together when this recording loads. Changing selections preserves the other selected models.

The production serving-side model `serving-side-fixed-flight-v3` was rerun using the validated native **59 × 237 SERVSIDE237-FLIGHT** feature cache. All anchors match the unedited production ensemble starts exactly. Both production serve heads were rerun; neither corrected human serve markers nor manual suppression decisions enter inference. The feature/model fingerprint, anchor contract, tied percentile ranks and all gate decisions were independently verified. Only two archived near-side scores differ from fresh inference, by at most 1.11e-16; verdicts, reasons, evidence and anchors match exactly.

| Serving decision | Outside ignored opening | Full source | Among suppression-retained candidates outside ignored opening |
| --- | ---: | ---: | ---: |
| Near | 18 | 21 | 16 |
| Far | 16 | 19 | 14 |
| Needs review | 9 | 10 | 9 |
| Rejected by serve gate | 9 | 9 | 1 |

The nine visible serve reviews comprise five cases recovered from agreement between the rally models, two with an ambiguous side score, and two with both reasons. Recovery means neither serve head reaches 0.85 within one second of the production start, but both rally models support the interval. The side-review band is `[0.3121748736511044, 0.5028396703865513)`; the underlying side threshold remains 0.4783744762021848. These are production-start anchors, not newly localized serve contacts.

The **Production serve predictions** panel shows every decision, near/far scores, review reason, both head peaks and thresholds, and seek controls. Ignored-opening candidates are hidden by default and can be included. Rejected candidates stay visible in this panel but do not become serve markers or remove rallies. The previously incorrect `not-serve` → near/far mapping is fixed. Review markers display `?`, preserve the guessed near/far side, and use the score for that side. An explicit empty current result no longer falls back to stale predictions.

Every review region now has expandable **Boundary breakdown** details. The original 47 flags are unchanged: 21 initial starts, two additional starts and 24 ends. Comparisons distinguish 21 changes to an original parent start, 22 changes to its final end, and four new internal boundaries (two additional starts and two separate ends). New boundaries have no invented old timestamp or shift. Each entry shows source/candidate lineage, original versus proposed timing where applicable, the exact nearest native signal sample, all four compact scores and what to check. Three proposals use decoded compact boundaries without a selected head peak; their source is identified rather than implying peak confidence.

The nine recommended parents and 84.76-second queue are unchanged. Each region identifies whether it is selected or outside the playback budget; outside-budget regions remain available. Serve-side reviews and compact boundary reviews are distinct decisions and are not assumed to refer to the same rallies.

Published metadata uses `serving-side-v2/ensemble-inference.json`, `research-references-boundary-details-v1.json`, and `labeling-catalog-serve-review-v1.json`. The first strict serving comparison attempt is preserved under `serving-side-v1`; v2 explicitly tolerates floating-point score differences up to 1e-12 while requiring exact decisions and evidence. The original human draft, rally predictions, compact signals and review selection are unchanged. `serve-review-registration-v1.json` records the publication checks.

Final validation: **20 focused tests** pass across model multi-selection, boundary enrichment/rendering, serving verdict presentation and the production serving runtime. The optimized Next build and TypeScript pass. Live reference and draft APIs return the expected 59 serving decisions, 50 non-rejected model markers (43 outside ignored footage), and 47 detailed boundary flags. Served HTML contains the multi-select and All/Clear controls. The 37 human rallies and 36 human serve markers are unchanged. `serve-review-ui-verification-v1.json` records these checks. Browser interaction was not available; static component rendering, event-handler tests and the live HTTP responses were verified.

| Serving/detail artifact | SHA-256 |
| --- | --- |
| `serving-side-v2/ensemble-inference.json` | `c7893ca77bee97c1ddb55ca01a8ca16e659fde8c34e5a5a86c4df2fb521aee79` |
| `research-references-boundary-details-v1.json` | `02a2008cff860284cd9d346d9a08ea7467abde9673d68aeca876807990045760` |
| `labeling-catalog-serve-review-v1.json` | `7fae2050a45b93b0e427d680392d051f62d56f10e937b5de9fc5a6dd2786b7aa` |

## Compact standalone follow-up

- [x] Publish the existing 34 frozen compact events as a separate selectable reference.
- [x] Keep the main blocks at raw compact boundaries and compute exports from compact's own events.
- [x] Show the four compact signals without attaching production-parent review decisions to standalone inference.
- [x] Verify the active UI/API, export-policy separation, unchanged labels and existing references.

Select **Compact standalone** in **Model breakdown**. Its 34 main blocks show decoded, unpadded compact boundaries. Its export strip uses its own events with the selected padding and strict gap-less-than-3-second joins. The decoder itself already applies 1-second smoothing and 0.5-second gap bridging; the UI does not join the main core blocks further. This layer exposes the same four native compact signals without implying that standalone predictions carry the production-parent review queue. Selecting a review layer alongside standalone retains the shared review panel.

The active `research-references-compact-standalone-v1.json` manifest preserves the prior two references exactly. Its SHA-256 is `86f081e71710e21ee8e9239d1bb69524f66d0c6a6d7f5d90147d6220e57d34ce`. No inference, training, threshold selection or label edits were performed. Live references and document APIs verify the 34 own export cores, all 4,245 samples for each of four heads, and the unchanged human draft. Ten focused parser, rendering and multi-selection tests pass, including all four padding cases and the exact 3-second cut boundary. TypeScript and the optimized Next build pass. The receipt is `compact-standalone-ui-verification-v1.json`; interactive browser inspection remains unavailable.

The [compact miss investigation](compact-standalone-misses-2026-09-20.md) distinguishes missing retained rallies from production-only discarded candidates and traces serve-head recovery through a no-serve counterfactual.

## Original artifact identities

All paths below are relative to the study artifact directory declared above. The editable human draft is in `private-reference-0197`.

| Artifact | SHA-256 |
| --- | --- |
| Source MP4 | `b92371af35a49a93726ad36de32382f0ac46c42bf9e4ed1260242ca82e7031af` |
| Source feedback | `28027e3943f91051e85190a444aa4b071e7a8b4f46381b3854df6c3f7055be4e` |
| Immutable human import | `5e6d3b05d797b4c996fd587a4a1c420d83d2be3b82c6542260464175ff5c430b` |
| Published human draft at handoff | `97ab7adce70f57ad521add3a18f18765abec61d36b2ba31cf0c739638164d94c` |
| `research-references.json` | `183d718ccf3567603efe44ceed56797487a5a5949cd935f77fecd39e66d548d5` |
| `labeling-registration-v2.json` | `65af9246d0b8375937bdd61562ff2fa35ff81f977a080c66bf2d475cab20dca8` |
| `inference-audit-v1.json` | `914cb07f240171b894282eaab46b0ff6652793d0f334b47fd841096e9ba0ed81` |
| `labeling-publication-audit-v1.json` | `b8a5acef92ce26537c310ba9ff6d52ce4eaf0f9f7e16f88f20d365c608fc4327` |
| `labeling-ui-verification-v1.json` | `0d45190c9ed6ac306e18f0d1407f18f24ec36049f9b39f6f8090814473848dba` |
