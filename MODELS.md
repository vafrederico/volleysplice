# VolleyCut model registry

This is the durable lineage registry for models trained by the VolleyCut project. It
records what each model consumes, the video files used to fit its learned parameters,
which data was used only for selection or evaluation, what predecessor it was intended
to improve or compare with, and what changed. Feature formulas and production versus
research status live in [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md); inference topology,
decoders, and ensemble behavior live in
[`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md).

This inventory was reconstructed from the checked-in production bundles, named NAS
artifacts, artifact metadata, manifests, and research reports. A field marked “not
recorded” is deliberately unknown; do not infer missing lineage from a version number.

## Registry contract

A registry entry is required for every durable learned artifact and every report-only
training study. A bundle containing independently fitted heads must name every head.
Ephemeral cross-validation fold files may be represented by their parent study rather
than one row per fold, provided the study records the folds, source set, and feature
signature.

Every new entry must record:

- immutable model ID/version and artifact path;
- learned target/head and algorithm;
- exact ordered feature signature, feature profile, and feature-generation config;
- **fit sources** whose labels or frames affected weights, calibration, adapters, or
  learned preprocessing;
- **selection sources** used for thresholds, epochs, seeds, decoders, or promotion but
  not for fitting weights;
- evaluation-only and protected-test sources, explicitly separated from both roles;
- source content hashes, label/manifest hash, code revision, random seed, and artifact
  hash when available;
- `Compared with`, `Change`, `Reason`, and `Result/disposition` fields. When no factual
  predecessor or change record is available, write `not recorded` rather than guessing.

Validation and protected-test video is never a training source merely because a report
shows a metric for it. A later final refit that intentionally includes former
development data must register a new source set and artifact.

This registry includes named linear heads, adapters, and learned suppression/side-switch
classifiers. It excludes fixed decoders and heuristics, third-party pretrained backbones
that VolleyCut did not train, cached feature matrices, and predictions. YOLOX, MobileNet,
and DINOv2 are therefore mentioned only where VolleyCut trained a downstream head or ran
a model study.

## Feature profile keys

| Key | Model inputs | Meaning |
| --- | ---: | --- |
| `F28` | 140 | 28 appearance/frame-difference base features × five offsets; historical no-flow ablation |
| `F42` | 210 | 42 base court-motion/flow features × five temporal offsets; historical v1 |
| `F42-raw` | 210 | Same base/context layout without within-recording percentile normalization |
| `F90` | 450 | 90 audiovisual v2 base features × five offsets |
| `F104` | 520 | 104 noise-normalized audiovisual v3 base features × five offsets; production |
| `F90+serve` | 451 | F90 plus one cross-fitted serve-contact probability |
| `F90+peaks` | 454 | F90 plus four decoded serve-peak-window features |
| `SIDE36` | 36 | Separate side-switch marker appearance/context signature |
| `SIDE34-V2` | 34 | 17 adaptive side-conditioned/derived marker features plus one missingness indicator per scalar; separate from production F104 |
| `LOWRES12` | 12 | Detector-free 192×108 HSV assignment, motion, blur, luma, and edge changes for side-switch v3 |
| `MULTIFRAME-NORMALIZED19` | 19 | Seven-frame 256×144 court-normalized broad/tight side identity, motion, and alignment-quality profile for side-switch v4 |
| `PLAYER-ORIENTATION22` | 22 | Six v4 carry-forward scalars plus 16 motion-component player identity, proposal support, and quality inputs for side-switch v5 |
| `DETECTED-ADAPTIVE29` | 29 | Six v4 carry-forward scalars, 20 quantized-person torso identity/localization inputs, and three adaptive whole-set team-orientation inputs for side-switch v6 |
| `PRODUCTION-STATE20` | 20 | Ten rally/dead-state/agreement/gap inputs plus ten serve-anchor/support inputs reduced from the two shipped production bundles; side-switch add-on, not part of F104 |
| `FULL-UNION-V5-STATE42` | 42 | V5 PLAYER-ORIENTATION22 plus PRODUCTION-STATE20 for all 704 full-union candidates; the trained union head selects a 34-input subset |
| `FULL-UNION-EXPANDED-V5-STATE42` | 42 | Same 42-value contract over 852 candidates after lowering internal dead-state threshold to 0.80 and separation to 10 seconds; 703 rows reused exactly and 149 extracted |
| `PRODUCTION-CONTEXT19` | 19 | Nine rally/dead-state/agreement inputs plus all ten serve-anchor/support inputs from the shipped production bundles; excludes raw gap duration and suppression |
| `PLAYER-ORIENTATION22+STATE10` | 32 | Validation-selected V5 production-state view: frozen V5 appearance plus the ten state/gating inputs; suppression excluded |
| `V5-RALLY-PARITY2` | 2 | Diagnostic per-rally V5 orientation coordinate and quality; no learned head, production feature signature, or promotion |
| `CONTINUITY1` | 1 | Recording-robust-normalized V5 `playerSwapMargin`; strong same-side tail veto over current-winner proposals |
| `RAW-VLM` | n/a | Raw video windows and text targets; no handcrafted feature matrix |

The profile key is a summary only. The ordered `featureNames` embedded in an artifact is
the authoritative signature.

## Current production system

Production browser assets are under `prod/public/runtime/`; byte-equivalent Android
copies are under `android/app/src/main/assets/`.

| Artifact | Learned models | Features | Fit sources | Compared with and change | Disposition |
| --- | --- | --- | --- | --- | --- |
| `model-1ca43e38eefc` (`environment-specialists-v2-all-labels`) | Rally, serve-contact, and dead-state linear heads | F104 | `ENV2-ALL11` | Compared with `environment-specialists-v1/all-labels` and previous production. Added three indoor recordings containing walking/ball-retrieval hard negatives; refit all three heads on six grass plus five indoor recordings. | Primary production bundle. Artifact SHA-256 `1ca43e38eefc…`; component artifact hashes remain in the bundle. |
| `model-9c92b8e9333f` | `full-audiovisual-audio-normalized-v3` rally, `serve-specialist-audio-normalized-v5` serve, and `dead-state-transition-audio-normalized-v5-no-legacy-final` dead-state heads | F104 | `GOLD-NB-T4` | Compared component-by-component with the prior F90 models. Added 14 noise-normalized audio base features; the serve/dead-state heads also tested legacy/new audio ablations. | Previous production bundle, retained in the two-model consensus because it still recovers some rallies missed by the newer refit. Artifact SHA-256 `9c92b8e9333f…`. |
| `suppression-overlap-exclusion-retrained` | Binary false-positive suppression head | F104 | `FEEDBACK16` | Compared with the original feedback suppression v3 fit. Corrected the target builder so false-positive intervals overlapping an included corrected rally core were excluded from suppression positives; production decoder settings were held fixed. | Production suppression model, gated to eligible one-model intervals. Artifact SHA-256 `39eddf581639…`; learned-weights SHA-256 `a943…`. |

## Historical named artifacts

### Earliest excerpt models

Artifact roots are `/mnt/freenas/volleycut/v0-2026-08-09/models/` and the corresponding
no-beach sensitivity root
`/mnt/freenas/volleycut/v0-2026-08-09-no-beach-2026-08-12/models/`.

| Artifact | Target/features | Fit; selection | Compared with and change | Disposition |
| --- | --- | --- | --- | --- |
| `real-rally-v0` | Rally, F42 | `V0-T3`; `V0-V1` | First recorded learned fixed-camera rally classifier; predecessor not recorded. | Historical baseline. |
| `real-rally-v0-no-flow` | Rally, F28 | `V0-T3`; `V0-V1` | Matched ablation of `real-rally-v0`; removed all 14 optical-flow base columns, reducing 210 inputs to 140. | Historical diagnostic. |
| no-beach `real-rally-v0` | Rally, F42 | `V0-NB-T2`; `V0-V1` | Fresh sensitivity refit of the corresponding v0 artifact after removing the beach excerpt from fitting. | Historical no-beach sensitivity. |
| no-beach `real-rally-v0-no-flow` | Rally, F28 | `V0-NB-T2`; `V0-V1` | Combines the no-beach training-scope change with the matched 140-input no-flow ablation. | Historical diagnostic. |

### Full-gold artifact family

Unless a row overrides it, these artifacts live under
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/`, fit on `GOLD-T6`, and use
`GOLD-V2` only for development selection. “Same learned weights” means a separately
named artifact was metadata/re-export/finalization, not another fit.

| Artifact(s) | Target/features | Compared with and change | Disposition |
| --- | --- | --- | --- |
| `pilot-baseline-v0` | Rally, F42-raw; fit `PILOT-T6` | Replaced the tiny excerpt corpus with six 90-second pilot clips; no percentile sequence normalization. | Historical pilot baseline. |
| `pilot-percentile-v1` | Rally, F42; fit `PILOT-T6` | Matched comparison with `pilot-baseline-v0`; added tied within-recording percentile normalization. | Percentile normalization retained. |
| `full-percentile-v1` | Rally, F42 | Compared with `pilot-percentile-v1`; replaced 90-second clips with the six fully labeled source videos while retaining F42. | Historical full-video baseline. |
| `full-audiovisual-v2`; `full-audiovisual-v2-final` | Rally, F90 | Compared with `full-percentile-v1`; expanded 42/210 to 90/450 with quality, camera-compensated motion, temporal formation proxies, and 13 audio columns. `-final` has the same learned weights and freezes final metadata/decoder selection. | v2 reference model; superseded in production. |
| `audiovisual-v2-targeted-pruned-diagnostic` | Rally, F90 signature with 365 active contextual inputs | Compared with full F90; zeroed/removed feature groups implicated by ablation diagnostics while preserving the artifact signature. | Rejected after the live-recall guardrail failed. |
| `serve-specialist-v1`; `serve-specialist-v2` | Serve contact, F42 | Added a dedicated serve-contact head instead of relying on rally probability. v2 has the same learned weights and adds expanded selection/tie-break metadata. | Historical serve baseline. |
| `serve-specialist-audiovisual-v3`; `serve-specialist-audiovisual-v4` | Serve contact, F90 | v3 upgrades the serve head from F42 to F90. v4 has the same learned weights and records corrected cadence/parity evaluation and re-export metadata. | Superseded by normalized-audio v5. |
| `full-audiovisual-audio-normalized-v3` | Rally, F104 | Compared with `full-audiovisual-v2-final`; appended 14 noise-normalized broadband/band audio columns, yielding 70 new contextual inputs. | Became rally head in previous production. |
| `serve-specialist-audio-normalized-v5` | Serve contact, F104, all legacy and new audio | Compared with `serve-specialist-audiovisual-v4`; added the v3 noise-normalized audio bank. | Became serve head in previous production. |
| `serve-specialist-audio-normalized-v6-no-legacy` | Serve contact, F104 signature with 65 legacy-audio contextual inputs disabled | Matched ablation of v5: visual plus only the new audio bank. | Research ablation; not promoted. |
| `serve-specialist-audio-normalized-v7-new-only` | Serve contact, F104 signature using only the 70 new-audio contextual inputs | Matched ablation of v5: removed visual and legacy-audio evidence. | Research ablation; not promoted. |
| `full-audiovisual-serve-prob-stack-v1-control` | Rally, F90 | Matched F90 control for the stacked serve experiment. | Research control. |
| `full-audiovisual-serve-prob-stack-v1-exploratory` | Rally, F90+serve | Compared with its control; appended one source-group-cross-fitted serve probability. | Rejected; did not pass the gate. |
| `full-audiovisual-serve-peak-window-v1-exploratory` | Rally, F90+peaks | Compared with the F90 control; appended four features describing decoded serve-peak membership, proximity, offset, and confidence. | Rejected; did not pass the gate. |
| `dead-ball-specialist-audiovisual-v1`; `dead-ball-specialist-audiovisual-v2` | Local end-boundary/dead-ball, F90 | Added a boundary-local specialist to the rally pipeline. v1 selection chose an exact no-op refinement; v2 retains the same learned weights with re-export/report metadata. | Historical diagnostic; no promoted boundary change. |
| `dead-ball-specialist-audio-normalized-v2-legacy-only` | Local dead-ball, F104 signature using legacy audio | Audio-input control for the normalized-audio boundary study. | Research ablation. |
| `dead-ball-specialist-audio-normalized-v3` | Local dead-ball, F104 full | Compared with v2; enabled legacy plus new normalized audio. | Research candidate. |
| `dead-ball-specialist-audio-normalized-v4-no-legacy` | Local dead-ball, F104 without legacy audio | Compared with v3; disabled legacy audio while retaining visual and new audio. | Research ablation. |
| `dead-ball-specialist-audio-normalized-v5-new-only` | Local dead-ball, F104 using only new audio | Compared with v3; removed visual and legacy-audio evidence. | Research ablation. |
| `dead-state-transition-audio-normalized-v0-legacy-only`; `…-v3-legacy-only-final` | Global dead-state transition, F104 signature using legacy audio | v0 is the legacy-audio control. v3 is the same learned weights finalized after the comparative evaluation. | Historical control/final alias. |
| `dead-state-transition-audio-normalized-v1-full`; `…-v4-full-final` | Global dead-state transition, F104 full | Compared with v0; enabled the full new normalized-audio bank. v4 is the same learned weights finalized. | Research candidate/final alias. |
| `dead-state-transition-audio-normalized-v2-no-legacy`; `…-v5-no-legacy-final` | Global dead-state transition, F104 without legacy audio | Compared with v1; disabled legacy audio while retaining visual and new audio. v5 is the same learned weights finalized. | v5 became dead-state head in previous production. |
| `dead-state-global-audio-normalized-v1-full-final`; `…-v2-full-final` | Algebraic inverse-rally dead-state control, F104 | Compared with transition-trained dead-state heads; uses global inverse-rally targets. v2 is the same learned weights with final re-export metadata. | Research control; not promoted. |
| `side-switch-specialist-v1` | Side-switch marker, SIDE36; fit `SIDE4`, selected on `GOLD-V2` | First dedicated learned side-switch ranker; compared against marker heuristics rather than a prior trained side-switch model. | Rejected for automatic use; retained for review ranking. |
| `side-switch-specialist-v1-no-blurry-beach-2026-08-20` | Counterfactual side-switch marker, SIDE36; fit `SIDE3-NO-BLUR`, same historical selection/evaluation protocol | Exact v1 refit after removing only `beach-source-02`; selected `appearance-context`, L2 1.0, threshold 0.432488. | Rejected; primary raw F1 fell 17.57%→17.02%, ROC AUC 0.538→0.524, and AP 0.110→0.106. Retained only as the blur-exclusion diagnostic. |
| `side-switch-specialist-v2` | Side-switch marker, SIDE34-V2; class-balanced logistic; fit `SIDE-V2-T5`, select threshold/decoder on `SIDE-V2-V2`, confirm once on `SIDE-V2-E4` | Compared with v1 on the same four confirmation recordings. Replaced gap/context-heavy v1 inputs with adaptive near/far assignment costs, frame consistency, color moments, derived stability interactions, and unlabeled recording-bound normalization candidates. L2/family used grouped OOF selection; validation selected a no-op temporal decoder. | Retained research/review ranking, not automatic scoring. Same-scope v1→v2 F1 19.05%→32.73%, ROC AUC 0.590→0.796, AP 0.124→0.373; v2 precision remains 20.45%. |
| `side-switch-specialist-v3-reanchored-capped6` | Side-switch gap, LOWRES12; fit `SIDE-V3-T6`, select L2/threshold/decoder on `SIDE-V3-V4`, retrospectively evaluate `SIDE-V3-E11` | Detector-free 192×108 HSV/motion/blur/edge ranker plus seven-point opportunity decoder. Evaluated ±1/±2/±3/±4 margins; overlapping ±4 windows use monotonic one-to-one assignment. Decoder re-anchors after selection and caps a set at six opportunities. | Rejected; source-held-out exact-gap F1 7.59%, ±2-tolerant F1 32.91%. Cadence-only ±4 reached 12.12% exact and 42.42% ±2-tolerant F1. No browser/Android port. |
| `side-switch-specialist-v4-multiframe-normalized` | Side-switch gap, MULTIFRAME-NORMALIZED19; same `SIDE-V3-T6`/`SIDE-V3-V4`/`SIDE-V3-E11` split | Replaces three fixed-band frames with seven-frame side palettes, first-seven-rally net calibration, piecewise vertical normalization, broad/tight scale pooling, and camera-translation compensation. Cadence/decoder policy is unchanged. | Positive feature result but rejected for automatic use. Exact F1 7.59%→16.67%, ±2-tolerant F1 32.91%→41.67%, and raw-phone row AP 18.41%→26.92%. Still only 6/37 exact predictions correct; no browser/Android port. |
| `side-switch-specialist-v5-player-orientation` | Side-switch gap, PLAYER-ORIENTATION22; same `SIDE-V3-T6`/`SIDE-V3-V4`/`SIDE-V3-E11` split | Adds player-like motion-component proposals, soft foot-position side assignment, first-three-rally team anchors, and optional persistent orientation parity. Validation selected L2 1.0, ±1 margin, distance 0.25, and orientation weight zero. | Best event-level research result but not promoted for automatic use. Candidate-window exact F1 16.67%→27.85%, ±2-tolerant F1 41.67%→45.57%, and row AP 26.92%→43.40%. Its 44 selected proposals are exposed in `/side-switch-review`; no browser/Android port. |
| `side-switch-specialist-v6-detected-adaptive`; bundled `detectedFixedPrototypeAblation` | Side-switch gap, DETECTED-ADAPTIVE29 selected head and 26-input fixed-prototype ablation; same `SIDE-V3-T6`/`SIDE-V3-V4`/`SIDE-V3-E11` split | Replaces v5 motion blobs with the frozen 3.48 MB block-int8 OpenCV Zoo MediaPipe person localizer, torso palettes, and three-frame consistency; adds confidence-gated online team-prototype updates. The Apache-2.0 detector is a third-party frozen input, not a VolleyCut-trained model. | Negative candidate-window event result; not promoted. Raw-phone row AP rises 43.40%→45.47%, but exact F1 falls 27.85%→24.10% and ±2-tolerant F1 falls 45.57%→40.96%. Its 48 selected proposals are exposed in `/side-switch-review`; no browser/Android port. |
| `side-switch-v5-production-state-v1` | Side-switch gap, PLAYER-ORIENTATION22+STATE10; frozen split and common V5/V6 decoder geometry | Compares original and production-serve-grounded V5 appearance with soft summaries of the two shipped rally/serve/dead-state bundles. Validation selects original appearance plus the ten state/gating inputs. Suppression scores are structurally quarantined. | Positive exploratory review-ranking result, not promoted. Exact precision/F1 improve 25.00%/27.85%→28.95%/30.14% while proposals fall 44→38; ±2 F1 moves 45.57%→46.58%, but exact per-recording count accuracy falls 3/11→0/11. Its 38 proposals are exposed as a separate `/side-switch-review` timeline; no inference-runtime port. |
| `side-switch-v5-no-cadence-v1` | Side-switch gap, validation-selected PLAYER-ORIENTATION22+STATE10; same frozen split and rows as V5-state | Refits the V5 base/state heads, verifies exact learned-parameter parity with the cadence model, and replaces seven-point candidate windows, re-anchoring, spacing, and the six-event cap with independent all-gap thresholding. L2, threshold, and feature-view selection remain development-only. | Positive causal diagnostic, not promoted. Exact recall/F1 improve 31.43%/30.14%→80.00%/44.44%, retaining all 11 cadence true positives and recovering 17 more, but proposals rise 38→91 and false positives 27→63. Research artifact only; no review UI or inference-runtime port. |
| `side-switch-v5-peak-cleanup-v1` | Frozen no-cadence V5-state head plus PRODUCTION-CONTEXT19 auxiliary head and cadence-free peak/count decoder | Compares eight locked combinations of adjacent/time NMS, a soft post-six logit penalty, and soft production-context log odds. There is no cadence, re-anchoring, hard count cap, production hard gate, or suppression input. Historical validation selects adjacent-gap peaks plus context weight 0.25. | Former research winner: locked `local-peak-soft-count`, selected by user after the exhaustive 50-event audit and superseded on 2026-08-23. It has 44.64% end-to-end pooled F1, 25 TP/37 FP/25 FN, and 62 proposals (5.64/video) under ±4-second boundary matching. Its manifest is archived byte-for-byte; no review UI/runtime port or production use. |
| `side-switch-v6-production-state-v1` | Side-switch gap, validation-selected serve-grounded DETECTED-ADAPTIVE29+PRODUCTION-STATE20 | Same production-state/serve-grounding ablation applied to V6; all eight candidates tie on validation exact F1 and AP selects grounded+combined. | Rejected. Exact F1 falls 24.10%→17.91%, ±2 F1 40.96%→38.81%, and row AP 45.47%→36.23%; frozen V6 remains unchanged. |
| `side-switch-continuity-verifier-v1` | Former-winner proposal veto, CONTINUITY1; threshold fit on 11 opened development videos after LOO evaluation | Compared with `side-switch-v5-peak-cleanup-v1/local-peak-soft-count`. Recording-normalizes the existing player same-minus-swap cost and vetoes only its strong-continuity tail under a 90% fit-TP retention guardrail. | Promising historical research layer, not a production model. LOO preserves 25 TP, removes 5 FP, and improves pooled precision/F1 40.32%/44.64%→43.86%/46.73%. A raw same-cheaper hard gate collapses recall to 14%; add-only and quality variants are rejected. |
| `side-switch-full-union-ranker-v1` | Side-switch event, FULL-UNION-V5-STATE42 artifact; class-balanced 32/34-input linear views; nested recording LOO over 11 opened-development videos | Expands scoring from 352 reviewed gaps to all 704 union candidates. Inner grouped LOO selects feature view, L2, threshold, adjacent/time suppression, and soft count separately inside each outer fold. Final all-development model selects the 34-input union-native view, L2 0.1, adjacent suppression, and a 0.5 post-six logit penalty. | Historical contender and predecessor to the promoted hard-negative head. At ±4 seconds it reaches 27 TP/29 FP/23 FN, 48.21% precision, 54.00% recall, and 50.94% F1 versus the then-current winner's 44.64%. Strict F1 is 39.62%, feature-view selection is 6/5 across folds, and the expanded continuity veto is rejected. |
| `side-switch-full-union-calibrated-v1` | Side-switch event, same FULL-UNION-V5-STATE42 scope; natural/square-root/full class weighting and label-free recording calibration | Fixes L2 0.1 and the dominant adjacent+soft-count decoder to isolate loss prior, 32/34-input view, raw/robust-logit/percentile score handling, and threshold under nested recording LOO. Final all-development model uses square-root balancing, raw primary32 scores, and threshold 0.446270. | Retained objective diagnostic, not a new winner or production model. Nested selection removes 2 FP at unchanged 27 TP, moving ±4 F1 50.94%→51.92%; variant choices remain split. Calibration variants are rejected. An identically specified no-switch head is the switch complement to `2.22e-16`. |
| `side-switch-internal-peak-penalty-v1` | Side-switch event, same FULL-UNION-V5-STATE42 scope; square-root-balanced 32/34-input logistic views plus a soft candidate-type logit offset | Compared with a matched zero-penalty nested-LOO control. Fixes L2 0.1 and adjacent+soft-count decoding, then selects an internal-peak penalty from 0–3 logits with the feature view and threshold. Final all-development model uses primary32, penalty 1.0, and threshold 0.424653. | Rejected diagnostic, not a new winner or production model. Nested ±4 F1 moves 53.47%→54.90% with two additional boundary TP but loss of the control's one internal TP; 6/11 folds select zero penalty. Artifact SHA-256 `1719ca433cb9…`; evaluation SHA-256 `4e0cc9a4050a…`. |
| `side-switch-full-union-expanded-ranker-v1` | Side-switch event, FULL-UNION-EXPANDED-V5-STATE42; class-balanced 32/34-input linear views; nested recording LOO for ranker/decoder over 11 opened-development videos | Compared with `side-switch-full-union-ranker-v1`. Keeps 624 boundaries but lowers the internal dead-state threshold 0.98→0.80 and separation 14→10 seconds, expanding 80→228 internal candidates and 92%→100% opened-scope coverage. Candidate settings were chosen on the opened grid; ranker selection remains outer-LOO. | Rejected. Nested ±4 F1 falls 50.94%→42.74% with 25 TP/42 FP/25 FN; strict F1 falls 39.62%→32.48%. Only 1/13 emitted internal proposals matches, and only one of four newly covered markers survives. Model SHA-256 `1e1a33bfc84e…`; evaluation SHA-256 `85e2b2e3998a…`. |
| `side-switch-hard-negative-mining-v1` | Side-switch event, retained FULL-UNION-V5-STATE42; square-root-balanced 32/34-input logistic views with fit-only per-recording hard-negative loss multipliers | Compared with a matched no-mining nested control. Fits a base head, upweights each recording's top 0/2/4/8 labeled negative scores by 2× or 4×, refits, and selects view/mining/threshold under outer recording LOO. Final development artifact uses union34, top 2 at 2×, and threshold 0.398850. | **Current research winner:** fixed `union34-top2-x2`, explicitly promoted after reaching 29 TP/23 FP/21 FN and 56.86% F1 under ±4-second matching. Nested selection removes 1 FP at unchanged 27 TP (53.47%→54.00% F1), but strength selection is split and this is opened development. Research-only; no browser/Android port or production use. Model SHA-256 `c2570481c30d…`; evaluation SHA-256 `e67088b36d17…`. |
| `side-switch-pairwise-ranking-v1` | Side-switch event, same union34/top-2/2× head with recording-balanced within-video positive/negative pairwise logistic loss | Holds the promoted feature, mining, decoder, and event contract fixed; compares pair strengths 0.25–2 and 0/0.5-logit margins under nested recording LOO. The final development selector returns the zero-strength pointwise control exactly. | Rejected. The best pairwise row AP is 44.73% versus 44.50%, but it adds 7 FP at unchanged 29 TP and lowers ±4 F1 56.86%→53.21%. Nested selection lowers F1 to 54.72%. The current research winner is unchanged; no runtime port. Model SHA-256 `3fa593b9abed…`; evaluation SHA-256 `e95c529ba676…`. |
| `side-switch-recording-reliability-v1` | Promoted union34/top-2/2× head plus a 13-input recording-summary ridge offset | Predicts a per-video optimal-threshold logit offset from camera/alignment/palette/serve/score-distribution proxies. Reliability targets and predictions are nested by recording; direct blur is unavailable in the retained artifact. | Rejected. Nested selection chooses the zero-offset control in 11/11 folds. The best fixed head gains 1 TP but adds 6 FP (56.86%→55.05% F1); stronger offsets fall near 52%. Final artifact returns the exact promoted classifier/threshold/decoder. No runtime port. Model SHA-256 `076bc7081a89…`; evaluation SHA-256 `9b6ca0eb3460…`. |
| `side-switch-rare-event-losses-v1` | Same union34/top-2/2× linear head with focal or effective-number final loss | Compares exact focal gamma 1/2 and effective-number beta 0.9/0.99/0.999 under nested recording LOO; hard-negative mining and decoder stay fixed. Final development selection returns square-root BCE. | Rejected. Best focal AP moves 44.50%→44.66% but event F1 falls to 53.85%; best effective-number F1 is 54.90%. Nested selection reaches 53.47% versus the 56.86% control. No pointer/runtime change. Model SHA-256 `bca760af4f8f…`; evaluation SHA-256 `cfd89de98636…`. |

The `V5-RALLY-PARITY2` feasibility experiment does not appear as a trained-model row:
it fits no head. Its fixed sign/persistence decoder is rejected after 21.32% swapped
state recall and 5 TP/13 FP/45 FN at persistence two. The rally-level feature and
artifact lineage are registered in [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md) and the
[decision record](docs/research/side-switch-parity-feasibility-2026-08-23.md).

The full-trace side-switch candidate union is a fixed generator over existing production
ranges/dead-state probabilities, not a learned head. Its selected development
configuration and 92% candidate recall are tracked in
[`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md) and the
[candidate-union decision](docs/research/side-switch-candidate-union-2026-08-23.md).
The downstream `side-switch-full-union-ranker-v1` row records the separately trained
ranker now that every union window has score-compatible features.

The Unicode ellipsis in three grouped rows abbreviates only the repeated artifact prefix:
the complete names are
`dead-state-transition-audio-normalized-v3-legacy-only-final`,
`dead-state-transition-audio-normalized-v4-full-final`,
`dead-state-transition-audio-normalized-v5-no-legacy-final`, and
`dead-state-global-audio-normalized-v2-full-final`.

The v2 side-switch implementation revision is
`9382fb932fdd873f7b6e0907645a97add2e5fa05`. Its model SHA-256 is
`376425c2d70d3e2418de9592e376e40a6f2a6c5b398ac9f8f65278c0839b3216`,
its stable deployable fingerprint is
`b24265fdac554eece6d8e1e446290be1797e9f3669005fa4f1c392fb72c1671f`,
and its feature artifact SHA-256 is
`8cad75f8800ca84544751980420ccd6bdf7ddac0ab61f0450eb6c5dcf1a6f321`.
The immutable provenance artifact at
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v2-provenance.json`
has SHA-256
`3b97b66745b9cddf236448022dc49f33565809fb28a53af6d1cc3452b712caa4`
and records every source video's full-file SHA-256, sampled fingerprint, feedback hash,
split role, label-map identity, development dataset hash, and evaluation hash. The full
decision and metric record is
[`side-switch-specialist-v2-2026-08-20.md`](docs/research/side-switch-specialist-v2-2026-08-20.md).

The v1 blur-exclusion counterfactual has fingerprint
`cf761aaaea6f87ec9c5ad0be2320145c728d08d4b85cc71eeef9d89bd61393ae`.
Its model, dataset, and evaluation SHA-256 values and the controlled comparison are in
[`side-switch-v1-blur-exclusion-counterfactual-2026-08-20.md`](docs/research/side-switch-v1-blur-exclusion-counterfactual-2026-08-20.md).
All future side-switch fitting must apply
[`validate_side_switch_fit_recordings`](analysis/side_switch_training_policy.py) before
preprocessing or selection. Historical `SIDE4` remains unchanged so its lineage stays
truthful; v3 and later fitting must exclude `beach-source-02` from weights,
preprocessing, thresholds, margins, and decoder choices.

The selected v3 fingerprint is
`54a31e37b0b89883582a4070d54fd72e76424ce3852b6a8662ef83ae8ab29621`.
The feature/model/development/evaluation SHA-256 values, complete ±1 through ±4
sensitivity, and non-promotion decision are in
[`side-switch-specialist-v3-2026-08-20.md`](docs/research/side-switch-specialist-v3-2026-08-20.md).
The immutable 21-video provenance manifest has SHA-256
`9bcc8d33f3a268f991d311a719205b5657544c23943096629c18e3ca70ba598e`
and binds implementation revision `ecadbdd1cbfb4c0d88455a0bd329a495b17ce2d3`.

The selected v4 fingerprint is
`54447d6dbb3db1c2ab281f3cfef0e5ef05e4cfa374cae21aa6f287508e9a98ae`.
Its feature/model/development/evaluation SHA-256 values, v3 comparison, and non-promotion
decision are in
[`side-switch-specialist-v4-2026-08-20.md`](docs/research/side-switch-specialist-v4-2026-08-20.md).
The immutable v4 provenance manifest has SHA-256
`bd7841b73427f292c864ae5e070bf54a94a276114e93ac335c4949cd8e0e90d1`
and binds implementation revision `c53a0159e39c9f8f4bdb5b0fb9a2e593f340fa64`.

The selected v5 fingerprint is
`c5bc3c86b9b2d35ea05a204929c2d496fdf85e01402c461a634a07c8b4441dff`.
Its feature/model/development/evaluation hashes, pre-fit diagnostic, orientation no-op
result, and v4 comparison are in
[`side-switch-specialist-v5-2026-08-20.md`](docs/research/side-switch-specialist-v5-2026-08-20.md).
The immutable v5 provenance manifest has SHA-256
`479ac49e1aefa0d0e4af841834b9a434575250ec566c58a86e3f57884c01b148`
and binds implementation revision `d0ac779a42d0672cfed29ec1d8fe0239fbd9e2bc`.

The selected v6 fingerprint is
`3abf18c01bff4c68bcbc805d45d302407a7495ad6dfaefb0a78ba856b56a41d4`;
the independently fitted fixed-prototype diagnostic fingerprint is
`df1f9498c21c27b10602baa0632a967fde59bde54afa8d5a2dc0fcf2653e0e33`.
Its pinned third-party detector and feature/model/development/evaluation hashes,
localization runtime audit, ablation, source-group comparison, and non-promotion
decision are in
[`side-switch-specialist-v6-2026-08-20.md`](docs/research/side-switch-specialist-v6-2026-08-20.md).
Both V5/V6 event evaluations are conditioned on reviewed rally-gap candidates and do not
establish exhaustive full-video precision/recall. Their 59-event proposal union is
reviewable without changing the frozen artifacts; see
[`side-switch-v5-v6-review-ui-2026-08-20.md`](docs/research/side-switch-v5-v6-review-ui-2026-08-20.md).
The immutable v6 provenance manifest has SHA-256
`28541061320879877cb348355f92c6a1a80980b3fda754513667b1574c0e2fc5` and binds
implementation revision `2cccddb7420bf845f99162fb09c841edca4a1d58`.

The production-state experiment reuses the exact shipped production head artifacts
and the frozen V5/V6 split. Its selected exploratory V5 fingerprint is
`dc5ea15a4455ac06af9e1d71d81dbd0fbe1886e2b2d0eb708d60f1ab9647e5dd`; the rejected
V6 experimental fingerprint is
`5909cfb20b975f416973b42bd7426e7b4c5fa42e4441caa2dd83c76d50937032`.
Complete feature/model/evaluation hashes, suppression-leakage audit, attribution,
and non-promotion constraints are in
[`side-switch-production-state-experiment-2026-08-20.md`](docs/research/side-switch-production-state-experiment-2026-08-20.md).
The immutable provenance manifest has SHA-256
`174fe3980cdd55fa14dda00c7e27d1b01882f303ec3ece6b6d709276e88643e7` and binds
implementation revision `5221c74f51356e993ebfcad8326469fd6bb80124`.

The no-cadence V5-state follow-up refits both eligible heads against the same frozen
rows and produces exact classifier-parameter parity with the cadence variants before
reselecting independent thresholds on validation. Its selected fingerprint is
`c8e6014892b0012971d288fd60eadb81150a9646ec0b4c5ce6c77e76eeca76ab`.
The model/development/evaluation hashes, exact true-positive cross-tab, decoder-cascade
audit, and non-promotion constraints are in
[`side-switch-v5-no-cadence-2026-08-20.md`](docs/research/side-switch-v5-no-cadence-2026-08-20.md).
The immutable provenance manifest has SHA-256
`b0a6d21df15c4c5f7f968efa600836bd1b13c6d72791ce746c0c89c7725d64e2` and binds
implementation revision `b7c732d61b2d242ee50f28af1686f9fab73d6a21`.

The peak/count/context follow-up inherits that exact no-cadence model and evaluates
eight validation-locked cleanup families. Its historical validation-selected
peak+context fingerprint is
`d37839e0031facd6913d904fd7c81624b7f3d9ce57567a1b2f182d3c0aac0190`.
The current forward research winner is its locked `local-peak-soft-count` mechanism,
fingerprint
`053fde3b92c971542ac9b644bfa8ab7ce70f5e37e0469ed229ac64b8727c0a07`.
The machine-readable pointer, decoder settings, evaluation metrics, and bound source
hashes are in
[`side-switch-current-research-winner-v1.json`](data/side-switch-current-research-winner-v1.json).
Complete grids, hard-versus-soft production semantics, all retrospective ablations,
and non-promotion constraints are in
[`side-switch-v5-peak-cleanup-2026-08-20.md`](docs/research/side-switch-v5-peak-cleanup-2026-08-20.md).
The immutable provenance manifest has SHA-256
`0782d18b53741ba8f63270986c432278ac2d43ff7d8088babfe349acd07d47f9` and binds
implementation revision `8926e69e12bee18b4fd64cea92f621ede81bc201`.

The completed raw-phone continuous review adds a separate exhaustive evaluation; it
does not refit or mutate any model above. Fifty direct human points are canonical for
the 11-recording physical-event audit. With four-second proposal-boundary allowance,
only 33/50 events are covered by any modern candidate gap. No-cadence V5-state had the
highest comparable macro recall at 56.36%; independent soft count had the highest macro
precision at 55.91%; and local peak plus soft count led pooled F1 at 44.64%. A later
704-candidate hard-negative model is now the explicit research winner with 29 TP/23
FP/21 FN and 56.86% pooled F1. These replace candidate-conditioned raw-phone numbers
when discussing end-to-end physical-switch accuracy. Because the fixed mining variant
was selected after label opening, these 11 recordings remain development scope; the
decision does not authorize production promotion. See
[`side-switch-hard-negative-winner-promotion-2026-08-23.md`](docs/research/side-switch-hard-negative-winner-promotion-2026-08-23.md).

### No-beach full-gold refits

The no-beach root is
`/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/models/`.
Every artifact below is a distinct fresh refit, not a pointer to the same-named
full-gold weights. The common change was to compare with the corresponding full-gold
artifact after removing both beach sources from fitting: fit `GOLD-NB-T4`, selection
`GOLD-V2`. Within-family feature changes and alias relationships are the same as in the
full-gold table, except that even same-named final exports belong to this no-beach fit.

| Family | Complete artifact names | Purpose/disposition |
| --- | --- | --- |
| Rally foundations | `pilot-baseline-v0`, `pilot-percentile-v1`, `full-percentile-v1`, `full-audiovisual-v2`, `full-audiovisual-v2-final`, `audiovisual-v2-targeted-pruned-diagnostic`, `full-audiovisual-audio-normalized-v3` | No-beach sensitivity for raw/percentile, F42→F90, pruning, and F90→F104 changes. Pilot artifacts fit `PILOT-NB-T4`, not `GOLD-NB-T4`; all others use the common full-video set. |
| Serve specialist | `serve-specialist-v1`, `serve-specialist-v2`, `serve-specialist-audiovisual-v3`, `serve-specialist-audiovisual-v4`, `serve-specialist-audio-normalized-v5`, `serve-specialist-audio-normalized-v6-no-legacy`, `serve-specialist-audio-normalized-v7-new-only` | No-beach refits of the serve feature/profile progression and normalized-audio ablations. |
| Rally with serve evidence | `full-audiovisual-serve-prob-stack-v1-control`, `full-audiovisual-serve-prob-stack-v1-exploratory`, `full-audiovisual-serve-peak-window-v1-exploratory` | Matched no-beach controls and rejected serve-stack/peak experiments. |
| Local dead-ball | `dead-ball-specialist-audiovisual-v1`, `dead-ball-specialist-audiovisual-v2`, `dead-ball-specialist-audio-normalized-v2-legacy-only`, `dead-ball-specialist-audio-normalized-v3`, `dead-ball-specialist-audio-normalized-v4-no-legacy`, `dead-ball-specialist-audio-normalized-v5-new-only` | No-beach refits of local boundary and audio-input studies. |
| Dead-state transition/global | `dead-state-transition-audio-normalized-v0-legacy-only`, `dead-state-transition-audio-normalized-v1-full`, `dead-state-transition-audio-normalized-v2-no-legacy`, `dead-state-transition-audio-normalized-v3-legacy-only-final`, `dead-state-transition-audio-normalized-v4-full-final`, `dead-state-transition-audio-normalized-v5-no-legacy-final`, `dead-state-global-audio-normalized-v1-full-final`, `dead-state-global-audio-normalized-v2-full-final` | No-beach refits/final exports of transition and inverse-rally dead-state comparisons. |

### Environment-specialist refits

These bundles are under `/mnt/freenas/volleycut/intake-2026-08-13/experiments/`.
Each bundle contains separately fitted rally, serve, and dead-state heads using F104.

| Bundle/head set | Fit sources | Compared with and change | Disposition |
| --- | --- | --- | --- |
| `environment-specialists-v1/models/all-labels/{rally,serve,dead-state}` | `ENV1-ALL8` | Compared with previous no-beach production components; refit all heads across all then-available grass/indoor labels with revised environment/hard-negative weighting. | Research candidate; predecessor to v2. |
| `environment-specialists-v1/models/grass-only/{rally,serve,dead-state}` | `ENV1-GRASS6` | Matched v1 environment specialization using grass sources only. | Research specialist; not current production. |
| `environment-specialists-v1/models/indoor-only/{rally,serve,dead-state}` | `ENV1-INDOOR2` | Matched v1 environment specialization using indoor sources only. | Research specialist; superseded by v2 indoor. |
| `environment-specialists-v2/models/all-labels/{rally,serve,dead-state}` | `ENV2-ALL11` | Compared with v1 all-labels; added `indoor-indoor-source-06`, `indoor-indoor-source-04`, and `indoor-indoor-source-08`, including walking/retrieval hard negatives, then refit all three heads. | Promoted as `model-1ca43e38eefc`. |
| `environment-specialists-v2/models/indoor-only/{rally,serve,dead-state}` | `ENV2-INDOOR5` | Compared with v1 indoor-only; expanded two indoor fitting sources to five with the three intake recordings. | Research specialist; not current production. |

No v2 grass-only bundle is present; do not synthesize one from the v1 artifact.

### Feedback suppression experiments

Both experiment roots use `FEEDBACK16` for fitting and hold the then-current production
decoder fixed. The five feedback projects in that set are raw user recordings with
corrected labels, not development-only metrics.

| Artifact/head | Compared with and change | Disposition |
| --- | --- | --- |
| `feedback-suppression-v3-2026-08-16/models/v3-candidate1-three-head/{rally,serve,dead-state}` | Compared with current production by refitting the three primary heads on the 11 environment sources plus five corrected-feedback sources. | Original candidate 1; not promoted. |
| `feedback-suppression-v3-2026-08-16/models/v3-candidate2-four-head/suppression` | Added a fourth learned suppression head to candidate 1. | Original candidate 2; superseded because overlap-positive target construction was corrected. |
| `feedback-suppression-v3-corrected-2026-08-18/models/v3-candidate1-three-head/{rally,serve,dead-state}` | Corrected experiment rerun; the three learned heads are weight-equivalent to the original candidate 1 because the correction affects only suppression targets. | Corrected comparison control; not promoted. |
| `feedback-suppression-v3-corrected-2026-08-18/models/v3-candidate2-four-head/suppression` | Compared with the original suppression head; excludes false-positive candidates that overlap included corrected rally cores. | Corrected candidate; learned weights match the promoted overlap-exclusion retrain. |
| `feedback-suppression-v3-2026-08-16/models/suppression-overlap-exclusion-retrained` | Standalone export of the corrected suppression target/fit, retaining production decoder behavior. | Promoted production suppression artifact. |

### Raw-video VLM adapters

| Artifact | Inputs and fit sources | Compared with and change | Disposition |
| --- | --- | --- | --- |
| `qwen3vl2b-unsloth-wsl-v1/seed-1729` | Qwen3-VL-2B LoRA adapter over 32-second raw-video windows; fit `GOLD-T6`, select/validate `GOLD-V2`; dataset also contains evaluation-only `GOLD-TEST1` | First recorded raw-video VLM training run in this family; no handcrafted F42/F90/F104 inputs. | Research only; checkpoint/adapter retained, not production. |
| `qwen3vl2b-unsloth-wsl-v1/seed-3407` | Same data, split roles, and RAW-VLM input contract | Matched second-seed comparison with seed 1729; changed random seed only. | Research reproducibility run; not production. |

## Report-only trained studies

These studies trained fold-local or temporary downstream heads but did not publish one
durable production artifact per fit. They are registered at study granularity.

| Study | Features and fit sources | Compared with and change | Result |
| --- | --- | --- | --- |
| [Audiovisual v2 feature ablation](docs/research/audiovisual-feature-ablation-2026-08-11.md) | F90 fold-local rally models; `GOLD-DEV8` | Removed predeclared feature groups from the full F90 reference to identify harmful/helpful groups. | Produced the targeted-pruned diagnostic; not promoted. |
| [Multiscale transition and interaction study](docs/research/feature-experiment-order-execution-2026-08-12.md) | F90 plus generated multiscale/interaction bank; `GOLD-DEV8` | Compared with F90 and component-selector controls; adds centered 0.5/1/2/4/8-second summaries and seven interactions. | 1,084-input combined development winner; retained research, not production. |
| [Rectangle court-relative study](docs/research/feature-experiment-order-execution-2026-08-12.md) | Court-relative reductions over existing grids; `GOLD-DEV8` | Compared orientation-invariant and fixed-endline variants with the transition reference. | Rejected; no variant passed the paired gate. |
| [Multistate and follow-up studies](docs/research/multistate-followup-execution-2026-08-12.md) | Fold-local DEAD/SETUP/SERVE/LIVE emissions and transition/boundary features; `GOLD-DEV8` | Compared a multistate sequence architecture and follow-ups with the frozen transition winner. | Research only; no production promotion. |
| [High-resolution MobileNetV2 study](docs/research/feature-experiment-order-execution-2026-08-12.md) | Frozen 1 Hz embeddings, projected crop features, and fold-local heads; `GOLD-DEV8` | Added four geometric high-resolution crops to the transition reference. | Rejected in every development source group. |
| [Component selector v1/v2](docs/research/feature-experiment-order-execution-2026-08-12.md) | Fold-local diagnostic selectors; `GOLD-DEV8` | v1 was superseded; v2 standardized the component diagnostic against the full reference. | Diagnostic only; not production. |
| [DINOv2 temporal development study](docs/research/dinov2-temporal-execution-2026-08-13.md) | Frozen DINOv2 ViT-S/14 tokens plus audiovisual boundary heads; `GOLD-DEV8` | Compared a learned temporal representation with the handcrafted transition reference. | Passed the development gate, but no checkpoint was published and protected test stayed unopened. |
| DINOv2 no-beach sensitivity | Same DINOv2 contract; fit `GOLD-NB-T4`, select `GOLD-V2` | Removed beach training sources to test environment sensitivity. | Research result only; no published checkpoint. |

The ball-presence pilot is not a VolleyCut-trained model entry: it evaluated a
third-party YOLOX artifact and did not train a downstream candidate after detector gates
failed. Fixed heuristics and decoders are likewise outside this registry.

## Source video catalog

Model rows use the stable IDs below. Paths are the actual source files used by the
training manifests on this machine; relocating the data does not change the recording
ID. For a future rebuild, verify the manifest's content hash rather than trusting only
the pathname. Historical manifests did not consistently record hashes, so this
reconstruction does not invent them.

### Early excerpt and pilot clips

| ID | Role-capable source video path |
| --- | --- |
| `v0-beach-JXM` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/labeled/beach/beach-source-02-deadstart25-duration62.mp4` |
| `v0-grass-qpd` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/labeled/grass/grass-source-09-deadstart2-duration88.mp4` |
| `v0-indoor-9lc` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/indoor/indoor-source-01-start340-duration90.mp4` |
| `v0-grass-rSs` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/labeled/grass/grass-source-10-deadstart6-duration84.mp4` |
| `v0-test-indoor-tds` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/labeled/indoor/indoor-source-05-deadstart41-duration49.mp4` |
| `pilot-beach-JXM` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/beach/beach-source-02-start358-duration90.mp4` |
| `pilot-beach-ey` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/beach/beach-source-01-start291-duration90.mp4` |
| `pilot-grass-Dm` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/grass/grass-source-04-start344-duration90.mp4` |
| `pilot-grass-GYU` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/grass/grass-source-01-start465-duration90.mp4` |
| `pilot-grass-qpd` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/grass/grass-source-09-start366-duration90.mp4` |
| `pilot-grass-rSs` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/grass/grass-source-10-start325-duration90.mp4` |
| `pilot-indoor-9lc` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/indoor/indoor-source-01-start340-duration90.mp4` |
| `pilot-indoor-Y9` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/indoor/indoor-source-07-start304-duration90.mp4` |
| `pilot-test-indoor-tds` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/indoor/indoor-source-05-start387-duration90.mp4` |

### Fully labeled proxy recordings

| ID | Source video path |
| --- | --- |
| `beach-beach-source-02` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/beach/beach-source-02.mp4` |
| `beach-beach-source-01` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/beach/beach-source-01.mp4` |
| `grass-grass-source-04` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/grass/grass-source-04.mp4` |
| `grass-grass-source-01` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/grass/grass-source-01.mp4` |
| `grass-grass-source-09` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/grass/grass-source-09.mp4` |
| `grass-grass-source-10` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/grass/grass-source-10.mp4` |
| `indoor-indoor-source-01` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/indoor/indoor-source-01.mp4` |
| `indoor-indoor-source-07` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/indoor/indoor-source-07.mp4` |
| `test-indoor-indoor-source-05` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/indoor/indoor-source-05.mp4` |

### Intake additions and corrected-feedback recordings

| ID | Source video path |
| --- | --- |
| `grass-grass-source-03` | `/mnt/freenas/volleycut/intake-2026-08-13/proxies/grass/grass-source-03.mp4` |
| `grass-grass-source-05` | `/mnt/freenas/volleycut/intake-2026-08-13/proxies/grass/grass-source-05.mp4` |
| `indoor-indoor-source-06` | `/mnt/freenas/volleycut/intake-2026-08-13/proxies/indoor/indoor-source-06.mp4` |
| `indoor-indoor-source-04` | `/mnt/freenas/volleycut/intake-2026-08-13/proxies/indoor/indoor-source-04.mp4` |
| `indoor-indoor-source-08` | `/mnt/freenas/volleycut/intake-2026-08-13/proxies/indoor/indoor-source-08.mp4` |
| `project-cmh0xj` | `/mnt/freenas/volleycut-raw-no-backup/PXL_20260816_193307688.mp4` |
| `project-kqx9c` | `/mnt/freenas/volleycut-raw-no-backup/PXL_20260816_171720964.mp4` |
| `project-qkf86k` | `/mnt/freenas/volleycut-raw-no-backup/PXL_20260816_161923155.mp4` |
| `project-vxcbv3` | `/mnt/freenas/volleycut-raw-no-backup/PXL_20260816_203801418.mp4` |
| `project-yf69sz` | `/mnt/freenas/volleycut-raw-no-backup/PXL_20260816_190429172.mp4` |

## Source-set definitions

Only IDs in a set's **fit** definition affect learned parameters. Selection and test
sets are separate even when listed on the same model row.

| Set | Exact members and role |
| --- | --- |
| `V0-T3` | Fit: `v0-beach-JXM`, `v0-grass-qpd`, `v0-indoor-9lc` |
| `V0-NB-T2` | Fit: `v0-grass-qpd`, `v0-indoor-9lc` |
| `V0-V1` | Selection: `v0-grass-rSs` |
| `V0-TEST1` | Protected test only: `v0-test-indoor-tds` |
| `PILOT-T6` | Fit: `pilot-beach-JXM`, `pilot-beach-ey`, `pilot-grass-GYU`, `pilot-grass-qpd`, `pilot-indoor-9lc`, `pilot-indoor-Y9` |
| `PILOT-NB-T4` | Fit: `pilot-grass-GYU`, `pilot-grass-qpd`, `pilot-indoor-9lc`, `pilot-indoor-Y9` |
| `PILOT-V2` | Selection: `pilot-grass-Dm`, `pilot-grass-rSs` |
| `PILOT-TEST1` | Protected test only: `pilot-test-indoor-tds` |
| `GOLD-T6` | Fit: `beach-beach-source-02`, `beach-beach-source-01`, `grass-grass-source-01`, `grass-grass-source-09`, `indoor-indoor-source-01`, `indoor-indoor-source-07` |
| `GOLD-NB-T4` | Fit: `grass-grass-source-01`, `grass-grass-source-09`, `indoor-indoor-source-01`, `indoor-indoor-source-07` |
| `GOLD-V2` | Selection: `grass-grass-source-04`, `grass-grass-source-10` |
| `GOLD-DEV8` | Development folds only: all six `GOLD-T6` plus both `GOLD-V2`; each report's held-out source-group fold is excluded from that fold's fit |
| `GOLD-TEST1` | Protected test only: `test-indoor-indoor-source-05` |
| `ENV1-GRASS6` | Fit: `grass-grass-source-04`, `grass-grass-source-01`, `grass-grass-source-09`, `grass-grass-source-10`, `grass-grass-source-03`, `grass-grass-source-05` |
| `ENV1-INDOOR2` | Fit: `indoor-indoor-source-01`, `indoor-indoor-source-07` |
| `ENV1-ALL8` | Fit: union of `ENV1-GRASS6` and `ENV1-INDOOR2` |
| `ENV2-INDOOR5` | Fit: `ENV1-INDOOR2` plus `indoor-indoor-source-06`, `indoor-indoor-source-04`, `indoor-indoor-source-08` |
| `ENV2-ALL11` | Fit: union of `ENV1-GRASS6` and `ENV2-INDOOR5` |
| `FEEDBACK16` | Fit: `ENV2-ALL11` plus `project-cmh0xj`, `project-kqx9c`, `project-qkf86k`, `project-vxcbv3`, `project-yf69sz` |
| `SIDE4` | Fit: `beach-beach-source-02`, `beach-beach-source-01`, `grass-grass-source-01`, `grass-grass-source-09` |
| `SIDE3-NO-BLUR` | Counterfactual fit: `beach-beach-source-01`, `grass-grass-source-01`, `grass-grass-source-09`; only `beach-beach-source-02` is removed from historical `SIDE4` |
| `SIDE-V2-T5` | Fit: `raw-no-backup-PXL_20260816_164327879`, `raw-no-backup-PXL_20260816_171720964`, `raw-no-backup-PXL_20260816_190429172`, `raw-no-backup-PXL_20260816_180646590`, `raw-no-backup-PXL_20260816_183701800` |
| `SIDE-V2-V2` | Threshold and decoder selection only: `raw-no-backup-PXL_20260816_210449857`, `raw-no-backup-PXL_20260816_193307688` |
| `SIDE-V2-E4` | Confirmation evaluation only: `raw-no-backup-PXL_20260816_160023210`, `raw-no-backup-PXL_20260816_161923155`, `raw-no-backup-PXL_20260816_203801418`, `raw-no-backup-PXL_20260816_212717581` |
| `SIDE-V3-T6` | Fit/model-family selection: `beach-beach-source-01`, `grass-grass-source-02`, `grass-grass-source-06`, `grass-grass-source-01`, `grass-grass-source-05`, `grass-grass-source-09` |
| `SIDE-V3-V4` | Threshold/decoder selection only: `grass-grass-source-03`, `grass-grass-source-04`, `grass-grass-source-08`, `grass-grass-source-10` |
| `SIDE-V3-E11` | Retrospective evaluation only: all 11 `raw-no-backup-PXL_20260816_*` recordings in `FROZEN_RECORDING_SPLIT`; source group is disjoint from `SIDE-V3-T6` and `SIDE-V3-V4` |

## Rebuild checklist

Before claiming a model has been reproduced:

1. Resolve every source-set ID to the exact file and verify its content hash.
2. Verify label/ignored-interval revision and split role; never open a protected test
   split during iteration selection.
3. Rebuild the exact ordered features under the artifact's immutable feature profile.
4. Reproduce learned preprocessing, target construction, class/hard-negative weighting,
   seed, optimizer, and calibration.
5. Record both weight and full-artifact hashes. A metadata-only final alias should say
   that it reuses weights; a refit should never do so.
6. Evaluate using the canonical ranking contract in
   [`docs/model-ranking-metric.md`](docs/model-ranking-metric.md), including all required
   padding cases and the declared development scope.
7. Add the new row here in the same change as the durable artifact, including its
   predecessor, intended improvement, actual change, and disposition.
