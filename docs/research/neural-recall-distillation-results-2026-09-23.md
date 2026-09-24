# Recall, detector alternatives and DINO compression results

Completed and independently audited. CUDA FP16 preserves DINO's retained core play at the target 2-second padding, with nearly identical F1. The tested detector configurations do not support reliable player counts, and the DINO-distilled MobileNet does not improve the requested recall-first tradeoff. Raising the selection floor to 99% improves retained play for compact AV, but does not guarantee 99% on held-out sources. No production, serve-side model or editor behavior has changed.

The [frozen protocol](neural-recall-distillation-protocol-2026-09-23.md) defines the four user-requested comparisons. Its initial checklist is historical. The prior [recognition results](neural-recognition-results-2026-09-22.md) and their registrations remain unchanged.

## Person detector alternatives

YOLOX-Nano 416 and SSDLite320 MobileNetV3-Large were evaluated on 32 fixed frames across the four source groups and two first-30-second sequences at 2Hz: 142 distinct frames per detector. YOLOX substantially improves localization over the prior MediaPipe/NanoDet observations, but repeated distant-player omissions and overlap duplicates remain. SSDLite misses more distant players. Both fail the frozen gate for reliable player counts and formation features; this does not establish that confidence-weighted occupancy or flow from incomplete detections cannot help.

No labeled detector recall is claimed, and no person-feature rally model was trained. Fixed thresholds, no tile/score sweep, all sheets reviewed, exact source/asset/frame provenance and SSDLite public-forward parity passed. Desktop two-thread median preprocessing/forward/postprocessing was 49.36ms for YOLOX and 51.91ms for SSDLite; these exclude video decoding and are not phone estimates. See the [full detector report](person-detector-alternatives-2026-09-23.md).

## Raising the recall requirement

The new requirement is **pooled inner R_core >=99%**, followed by maximum F1_padP_coreR among eligible choices. It is a validation eligibility condition, not a promise on an unseen source or a calibrated probability. The target product padding remains 2 seconds before/after; both export unions join only positive gaps strictly below 3 seconds. Ignored time is subtracted afterward without rejoining. Every complete result includes 0/1/2/3-second sensitivity, original rally losses and duration accounting.

This changes checkpoint/decoder selection while preserving the existing short-rally-boosted training loss. It does not mean 99% event recall, accurate serve boundaries, or that a score of 0.99 is a calibrated probability. Those require separate measurements.

The first stage reconstructs all 60 original 95% source/seed selections and their held outputs exactly, then reselects within the same 48 decoders x 4 checkpoints. The independent selection audit replays all 11,520 candidate metrics. Missing newly selected outer checkpoints are refitted using the original numerical recipe and seed; each new refit also saves the original selected epoch and must reproduce its weights, scores and training exposure exactly.

| Existing model | Eligible source/seed folds out of 12 | Interpretation |
| --- | ---: | --- |
| Compact AV + TCN | 12 | All three seeds have a complete eight-recording evaluation. |
| DINO + TCN | 10 | Two infeasible folds; never average only favorable complete seeds as a three-seed model rank. |
| Frozen regional MobileNet + TCN | 11 | One infeasible fold; same restriction. |
| AV transformer | 0 | No eligible operating point in this fixed grid. |
| DINO transformer | 1 | Only one source/seed fold is eligible. |

The minimum live-entry threshold in this registered grid is 0.20. Infeasibility is a limitation of this grid/recipe, not proof that no lower-threshold or more permissive decoder can reach 99%. No infeasible fold silently falls back to 95%. A separate frozen-original-epoch decoder-only sensitivity is reported with its own scope and eligibility.

All 16 supplemental refits passed exact original weights, predictions, training history and exposure checks. The final independent audit passed. Compact AV is the only existing neural arm with a complete three-seed result under the strict requirement:

| Setting, target padding 2s | P_pad | R_core | F1_padP_coreR | Export seconds | Incorrect export seconds | Missed core seconds | Completely lost rallies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Compact AV, original 95% selection | 88.30% | 92.41% | 90.31% | 3,543.47 | 414.82 | 181.20 | 33.00 |
| Compact AV, strict 99% selection | 77.82% | 97.96% | 86.71% | 4,437.94 | 987.35 | 48.70 | 9.67 |
| Production, descriptive baseline | 72.45% | 99.27% | 83.76% | 4,943.03 | 1,361.95 | 17.47 | 3 |

Neural rows are means of three seeds, each measured by pooling duration numerators and denominators across the same eight recordings; fractional rally counts are seed means. Production retains its historical label-exposure caveat. The stricter AV setting gains 5.55 percentage points of held-source recall, loses 10.49 points of precision and loses 3.59 points of F1. It substantially reduces completely missed rallies, but does not reach 99% held-source recall.

For DINO, only seed 20260918 has a complete result: P 86.09%, R 98.84%, F1 92.03%, five completely lost rallies. Seeds 3407 and 1729 remain infeasible when source-group-012 and source-group-007 are held out respectively. Frozen MobileNet has complete results for seed 3407 (85.79% / 96.00% / 90.61%, 14 completely lost) and 1729 (83.33% / 96.93% / 89.62%, 12 completely lost); seed 20260918 is infeasible when source-group-012 is held out. Infeasibility concerns pooled inner validation on the other sources, not measured recall on the untested outer source. These are descriptive seed results, not comparable three-seed model averages. Neither transformer has a complete seed. No complete existing neural seed achieved 99% held-source R_core at the target 2-second padding.

The final report is `recall99-v1/report-complete.json` (SHA-256 `7443fa8e5409a84c2f1708ebaaef13b0333f0e850403a55608c70d6a5e28237f`); its passing supplemental audit is `audit-refits.json` (`f04015c21ed010a61bdf41dc70e6627d50804d0d5a17ba3c4fc5354780bdb4df`). The [detailed recall report](neural-recall-operating-point-results-2026-09-23.md) includes every padding case, failed selection, duration breakdown and original-rally guardrail.

## DINO precision

The precision comparison includes the actual DINOv2-S/14 image encoder at 336 pixels/4Hz, not just the small temporal head. Historical frame ordinals, pixels, pooling and float16 cache storage remain fixed. FP32 ONNX parity against PyTorch and original-cache sampling passed before full precision replay.

The mixed dynamic INT8 graph quantizes 48 constant-weight MatMuls while 24 attention MatMuls, convolution, LayerNorm and Softmax remain floating point. The initial pilot showed significant embedding drift. A separately recorded, label-blind engineering amendment tested exactly two alternatives: reduced-range signed weights and unsigned weights, selecting by embedding RMSE only. Neither improved on the original signed graph, so the declared full replay retains it. The AVX2 saturation hypothesis is not supported as the main explanation by that check.

The actual image encoder was also tested in desktop Chrome WASM on two fixed inputs. FP32 parity passed; INT8 execution produced materially different results from CPU ORT. Therefore CPU INT8 rally scores cannot be transferred to browser deployment. Desktop timings under concurrent load are neither phone measurements nor production throughput.

If INT8 is pursued further, the next engineering step is layerwise activation comparison, followed by a separately registered calibration or quantization-aware approach using only permitted training sources. ONNX Runtime documents [activation/weight matching and selective exclusion](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html); [PTQ4ViT](https://arxiv.org/abs/2111.12293) demonstrates why vision-transformer quantization can need specialized treatment. Neither source establishes the cause of this DINO graph's drift or guarantees a fix. That follow-up is not part of the present experiment.

All 33,382 embeddings per precision path are complete. The independent input audit passed all eight recordings and 851 bound artifact identities; 24 sampled INT8 numerical replays were bit-exact. Replaying the same three original DINO + TCN seeds and historical 95% selections gives:

| Actual DINO encoder precision, target padding 2s | P_pad | R_core | F1_padP_coreR |
| --- | ---: | ---: | ---: |
| FP32 control | 88.475% | 96.955% | 92.513% |
| CUDA FP16 | 88.527% | 96.955% | 92.541% |
| Mixed dynamic INT8, native CPU | 88.079% | 96.548% | 92.105% |

These are means of three complete, matched eight-recording seed evaluations. FP16 preserves retained core seconds for every individual rally at 2-second padding, including complete/partial losses. Its mean F1 changes by +0.028 percentage points. INT8 loses 0.407 points of recall and 0.408 points of F1. These settings isolate precision effects; they are not newly calibrated per precision variant. The result supports the tested CUDA FP16 path while leaving physical-phone FP16 unqualified. The native CPU INT8 accuracy result does not overcome its failed browser parity check.

At the new FP32-selected 99% operating points, only seed 20260918 has a complete scope. Its matched results are:

| Encoder precision, seed 20260918 only | P_pad | R_core | F1_padP_coreR |
| --- | ---: | ---: | ---: |
| FP32 | 86.089% | 98.841% | 92.025% |
| CUDA FP16 | 86.077% | 98.841% | 92.019% |
| Mixed dynamic INT8, native CPU | 85.314% | 98.961% | 91.632% |

Here INT8 slightly increases recall while adding about 47.4 seconds of export and lowering precision/F1. Thus its effect is operating-point dependent; it does not always reduce recall. Precision variants reuse the FP32-selected settings, without precision-specific recalibration or requalification of the inner 99% constraint. The other two seeds remain explicitly partial and are not averaged with this complete seed.

The [detailed precision report](dino-precision-results-2026-09-23.md) contains all four padding cases, per-seed/per-source scope, event and original-rally guards, export accounting, runtime measurements and artifact hashes. Under the original 95% settings, mean completely lost rallies rise from 13.67 to 14.67 with INT8; FP16 leaves them unchanged.

## Fold-local DINO-distilled MobileNet

This is a different experiment from the prior frozen ImageNet-MobileNet arm. Every student starts from the same pretrained MobileNetV3-Small feature encoder and learns from the frozen DINO teacher. Student and teacher receive the same decoded ROI frames at 224/336 pixels. Teaching samples are at most 128 evenly spaced frames per recording selected through the existing valid mask, including ignored-time exclusion and reviewed game windows; live/dead labels do not select teaching frames.

Each inner student excludes both its outer and inner source groups from all student fitting, including auxiliary or unlabeled footage. Each outer student excludes its evaluation source. The temporary shared projection learns cosine agreement for one global and nine spatial teacher tokens for exactly eight epochs; it is discarded for inference. BatchNorm running statistics remain frozen while encoder weights train. Deployment keeps the original four mobile regional pools, eight quality/age/availability values, AV104 and the same 44,692-parameter temporal TCN. Teacher 3x3 pooling and deployed four-band pooling are intentionally distinct.

Before full fitting, 32 real frames from all four sources passed unchanged-student/historical-cache parity, nonzero finite encoder gradients, unchanged BatchNorm buffers and identical repeated two-step optimization. An initial GPU normalization-order discrepancy was fixed by preserving the historical NumPy FP32 preprocessing order before transfer to the GPU; no learning recipe or held-source outcome was changed to resolve it.

Preparation produced 38,308 mobile frames and 2,304 same-frame teacher targets across all 18 recordings. Every selected decoded frame hash matched the original mobile pipeline. Student registration `20b452deb0b78758cb59ef8d3db775a636b4e571881702b38646d2744d2b14ac` binds those inputs, engineering evidence, source exclusions and numerical code before any full student fit.

The prospectively fixed first inner fit (seed 3407, inner-0-1, temporal epoch 60) passed CPU ONNX and desktop Chrome WASM checks: 32 real encoder fixtures, seven true-length temporal fixtures and four chunk-boundary checks. The actual trained encoder graph is 3,718,362 bytes, with prepared-input medians of 2.82ms on CPU and 18.92ms in Chrome. The trained head plus scaler is 213,477 bytes, with medians of 4.00ms and 8.56ms per tested chunk respectively. These are separate FP32 graph checks under concurrent desktop load, not full video throughput. Decode, AV/quality computation, token FP16 storage roundtrip and feature alignment are excluded; physical-phone performance remains unmeasured. The training-only projector is absent from the deployed graph.

All 28 student/head fits are complete. The student clears 10/12 inner selections; frozen MobileNet clears 11/12. Student seeds 3407 and 20260918 are infeasible when source-group-012 is held out (maximum inner recalls 98.7832% and 98.6118%). Their missing outer fits are not replaced with a lower-floor selection. Only seed 1729 has a complete student evaluation; both models' three-seed means remain unavailable.

The sole complete matched seed, 1729, gives the following audited target-padding results.

| Encoder, same seed and strict 99% selection | P_pad | R_core | F1_padP_coreR | Export seconds | Incorrect export seconds | Completely lost rallies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen ImageNet MobileNet | 83.33% | 96.93% | 89.62% | 4,061.18 | 676.96 | 12 |
| DINO-distilled MobileNet | 71.31% | 98.56% | 82.75% | 4,912.88 | 1,409.69 | 11 |

Distillation gains 1.64 percentage points of retained-core recall, but loses 12.02 points of precision and 6.87 points of F1. It adds 851.70 seconds of export, including 732.73 seconds of additional incorrect footage. Completely missed rallies fall by only one; any-core loss falls from 44 to 20 original rallies. Crucially, completely lost short rallies (original duration <=3s) fall from 12 to 7, while completely lost longer rallies (>3s) rise from zero to four. Those four last 3.07-5.22 seconds and all come from source-group-007. Aggregate retained-time recall hides this regression. Event F1 falls from 72.65% to 62.81%. This is a descriptive same-seed result, not a favorable-seed model ranking. The recipe has not demonstrated an improvement at the requested bar.

The final audit passed all three seeds, 28 physical student fits, 28 temporal fits and 36 logical inner views. It verifies every inner selection, source exclusion, unchanged label/AV inputs and skipped infeasible outer, with independently replayed sampled teacher/encoder/head outputs and complete interval metrics. It is not a full training or every-frame numerical replay. Report SHA-256: `0620fe81eee6a852112b4884b1b2f1c9c449d5fc8589ef5f5d9745dda60ac00a`; audit: `09a23abbc24a25befb8fc93e8af8563596d2b36fe3e75bac1712ff8f2df7a12d`; paired summary: `565f4ff478a84ec5d718a86a67990680b7ab6480158a5d3a45e67a96502f5c07`. See the [student results](distilled-mobile-results-2026-09-23.md) for all padding cases, seed statuses, export accounting and rally guards, and the [runtime qualification](distilled-mobile-qualification-2026-09-23.md) for graph checks and deployment limits.

## Interpretation and remaining mobile checks

Selection eligibility, held-source recall, precise rally separation and model-score calibration are distinct. Retained play after padding can remain high while raw boundaries or individual short rallies are wrong. Report complete/partial losses and event metrics alongside the main duration metric. These repeatedly inspected development sources do not constitute an untouched final test.

No physical Android device was connected through ADB. The tested precision/runtime results do not establish sustained phone latency, memory, battery or thermal behavior. Model files fitting in storage is insufficient to qualify the full pixel-to-rally mobile pipeline.

Keep the current production-backed review workflow. For desktop DINO, the tested CUDA FP16 path is the useful compression result from this batch. For mobile, the small CNN's runtime is promising, but this generic feature-distillation recipe is not ready for promotion. A future registered experiment could train it jointly on rally/start/end labels and teacher rally scores, with explicit original-rally loss guardrails. YOLOX's confidence-weighted occupancy or local flow also remains a separate feature hypothesis; unreliable counts do not prove those signals useless. Neither follow-up was trained in this batch.

## Artifacts, validation and storage

New artifacts are on the direct NAS mount at `private-reference-0118` (`private-reference-0100`). Existing pretrained assets are read-only. New images, weights, caches, temporary files and reports are written to NAS; Python bytecode writes are disabled. WSL root and OS-managed swap remain on C:.

- `person-alternatives-v1/`: detector protocols, official assets, results, overlays and passing audit.
- `recall99-v1/`: strict selections, candidate scores, supplemental refit plans/weights and independent audits.
- `dino-precision-v1/`: encoder exports, pilot/precision caches, runtime evidence and downstream metrics.
- `distillation-images-v1/`, `distillation-teacher-v1/`: identical-frame student inputs and frozen teacher targets.
- `distillation-engineering-v1/`: pretraining pixel/pool/gradient/determinism qualification.
- `distilled-mobile-v1/`: fold-specific encoder features, temporal fits, selections, results and final audit.

The first recorded resource amendment allows intermittent low-memory FP16 extraction to overlap one bounded fitting/engineering worker; full student fitting began only after supplemental recall refits exited. A second amendment permits two fixed student seeds concurrently after memory checks, without numerical changes. An external guard verifies the original process identity and suspends it at the start of seed 1729 until the separate seed 20260918 worker has exited successfully with a valid result. This prevents competing writes when the original runner later reaches that seed. Concurrency is reported explicitly; desktop timings are not exclusive throughput measurements.

No unrelated job was stopped and WSL was not restarted. At the recorded mid-run snapshot, C: had 37.53GiB free and WSL had 12.89GiB available memory; measured swap usage was about 1.72GiB. A later snapshot showed 37.10GiB free on C: and 2.48GiB of WSL swap in use; directing experiment files to NAS does not relocate OS-managed swap. Resource amendment 2 records another live pre-launch snapshot.

All 31 focused tests pass together, covering detector adapters, precision handling, strict recall selection, distillation math, VFR audit sampling and rejection of partial-scope model averages. Detector, recall, precision and complete student artifact audits pass. Student image/teacher, first-fit and CPU/Chrome runtime gates also pass. The guarded parallel worker exited successfully, the original worker resumed and completed, and all training workers are finished. Final measured C: free space was about 36.94GiB; the new study's NAS allocation was approximately 15GiB.
