# Visual recognition and temporal transformers for rally recall

Date: 2026-09-22. **Investigation and proposal only.** No training, feature extraction, model downloads, device benchmarks, or production changes were performed. Beach remains excluded. The RTX 3080 is the future training machine; the intended inference targets are phones and browsers.

## Recommendation

Test **court-aware player movement and a small regional visual CNN** as new information sources. Independently compare the current TCN with a **small temporal transformer on identical features**. Qualify ball visibility and localization before committing to a ball model. Use ball/pose evidence initially for additional-rally proposals, boundary guidance and review prioritization; absent detection must not suppress a rally.

The strongest existing evidence is that visual information helps: DINO improves on compact AV-only models. There is no measured evidence yet that attention will improve our recall, or that a generic ball detector provides a reliable enough signal. A larger model could still fail because the relevant contact, ball or player evidence was discarded during feature extraction.

Work completed for this investigation:

- [x] Commit existing experiments and editor lab, then push Forgejo branch (`a6c48336`).
- [x] Audit current architecture, label tiers and prior recognition experiments.
- [x] Check primary sources for detector, visual-encoder, transformer and deployment options.
- [x] Define controlled experiments, annotation needs, evaluation and device gates below.

All proposed experiments below remain unexecuted.

## What the current evidence establishes

The [current comparison](neural-production-combinations-results-2026-09-19.md) uses eight indoor/grass recordings, four source groups and 322 exact gold rallies. At the declared product padding of 2 seconds before/after, with positive gaps joined only when strictly below 3 seconds:

| Existing configuration | P_pad | Retained-play R_core | F1_padP_coreR |
| --- | ---: | ---: | ---: |
| Production default, aggressive suppression | 72.45% | 99.27% | 83.76% |
| Compact TCN, short boost | 88.30% | 92.41% | 90.31% |
| DINO + TCN, global control | 89.50% | 96.96% | 93.07% |

These are historical development results, not a fresh ranking or untouched generalization estimate. Production weights have historical exposure. Neural results average three seeds after pooling durations within each seed. The linked report contains the required four padding cases and export-duration accounting. Do not compare these percentages to older studies with different cohorts or metrics.

Current four-head compact TCN has **29,700 trainable parameters**. DINO + TCN has **46,868 downstream parameters**, excluding the frozen DINO image encoder. Both use 104 audiovisual features at 4 Hz, width 64, five temporal blocks, and a 125-tick receptive field with 62 ticks of context on each side. DINO additionally supplies ten spatial tokens of 384 values, projected to 16 values each. See [model metadata](../../analysis/transfer_temporal_model.py).

The [short-context experiment](neural-short-context-results-2026-09-19.md) regressed both architectures when reducing the field to 33 ticks. Preserve the approximately 31-second field in the first attention comparison. The [expanded transfer study](neural-short-boost-transfer-results-2026-09-19.md) fits up to 18 recordings from seven source groups; reviewed exports add keep/drop supervision, not exact serve-contact/dead-ball labels. More frames from one match are not more independent matches.

Prior recognition results constrain the next experiment:

| Prior experiment | Finding | Consequence |
| --- | --- | --- |
| [Generic ball pilot](minimum-ball-presence-pilot-2026-08-11.md) | YOLOX-s held-source precision/recall 79.0%/39.0%; adding tiles gave 67.7%/31.8%. The eight-video pilot included beach. | Do not transfer those percentages to today's non-beach cohort. Generic sports-ball detection is an unqualified input; temporal volleyball-specific tracking remains untested. |
| [Frozen MobileNetV2 fusion](feature-experiment-order-execution-2026-08-12.md) | Four crops at 1 Hz, random projection and linear fusion reduced event F1 from .5244 to .3498 across its historical scope. | Avoid repeating that recipe. It does not rule out task adaptation, spatial pooling or temporal fusion with a mobile CNN. |
| [Person tracklets for side switches](side-switch-t1-endpoint-identity-transport-plan-2026-08-24.md) | Existing tiled MediaPipe person detector; weak far-court support and redundant appearance-derived features. | Reuse localization plumbing, then test movement/formation separately. Prior tracklet availability is not detection accuracy or demonstrated rally-recall benefit. |
| [Serving-side flight motion](serving-side-flight-motion-2026-08-20.md) | Concentrated residual flow is already useful to serving-side classification. | Reuse this information as a candidate start cue, but do not infer missed-rally performance from side accuracy. |

The existing `player_motion_*` and optical-flow channels already encode motion summaries. The new hypothesis must add **localized players, court membership and relationships between their movements**, rather than merely rename existing motion energy.

## Recognition features worth testing

| Feature family | Concrete signals | Expected use | Main failure to measure |
| --- | --- | --- | --- |
| Player localization and short tracks | In-court occupancy; near/far movement quantiles; proportion moving; acceleration; formation spread; synchronized receiver response; coordinated stand-down | Recover weak/short points; distinguish active play from incidental movement; refine boundaries | Small far players, occlusion, background people, tracking errors |
| Court geometry and camera quality | Net/service regions, court mask, camera-motion residual, obstruction/blur, scene changes | Put movement and serve cues in context; identify unreliable evidence | Geometry changes, incomplete court, wrong court ownership |
| Regional visual embeddings | Full-court, near/far and net-region appearance over time | Recognize ready formation, active reaction, reset, retrieval, huddle | Learning venue/background instead of activity |
| Selective server pose/action | Service-zone presence, arm rise/strike, jump/land, receiver reaction aligned with audio/flow | Additional serve-start proposals; distinguish a new rally from continuation | Practice toss, setting, tiny or obscured server |
| Ball trajectory and visibility | Observed centers, track continuity, image-plane velocity, launch/contact direction change, net-region crossing, terminal flight | Short/failed serves and quiet rallies; contact/end timing | Blur, tiny ball, occlusion, off-frame flight, retrieval and adjacent-court balls |
| Context/event recognition | Dead-reset, celebration, ball retrieval, camera disturbance; optional learned audio contact/whistle cues | Cleanup and review prioritization | Delayed reactions and off-court sounds masquerading as boundaries |

These are hypotheses. A visible or moving ball is not proof of a live rally; a ball can be held, tossed for practice or returned between points. No detection is **missing evidence**, not a negative label. Keep visibility, observed-versus-interpolated status, confidence and feature age as explicit inputs. Do not interpolate across camera cuts or long occlusion.

For player movement, start with anonymous sets and short tracks rather than persistent identity or jersey recognition. Compensate camera motion and retain box/track reliability. The existing side-switch detector's maximum-six and two-per-side selection rules must not silently carry into six-versus-six rally analysis. Track at the available motion rate between sparse detector frames; treat associations over long gaps as uncertain.

Use court-normalized footpoints for ground movement. An airborne ball does not lie on the court plane: a floor homography must not be presented as its true 3D position or physical velocity. Screen-space/net-relative trajectory is sufficient for an initial learned cue.

## Model shortlist and mobile implications

**Person detector:** first qualify the existing pinned `opencv-zoo-mediapipe-person-int8bq-2023mar` with appropriate player coverage. If it fails far-player observability, compare one challenger: **NanoDet-Plus-m at 320**, optionally 416 only when resolution is the identified failure. Its official repository lists 1.17M parameters, ONNX export, Android/ncnn and browser WASM examples. Repository code is Apache-2.0; pin the actual checkpoint and its terms independently. Published device latency is not our end-to-end cost. [NanoDet source](https://github.com/RangiLyu/nanodet), [license](https://github.com/RangiLyu/nanodet/blob/main/LICENSE).

**YOLO:** YOLOX-Nano is another small alternative with an Apache-2.0 repository and official deployment paths, but there is little value in a broad detector sweep before finding the actual visibility bottleneck. Ultralytics YOLO is technically relevant; its current official terms describe AGPL-3.0 or Enterprise licensing for code/models. Exporting to ONNX does not by itself resolve those terms. [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [Ultralytics terms](https://www.ultralytics.com/license).

**Visual CNN:** use **MobileNetV3-Small** as the first portable encoder. The official classification model is 2.54M parameters and 0.06 GFLOPs; removing its classifier and changing pooling creates a different model whose size/cost must be measured. Begin with one court image at 1–2 Hz, one feature map, and learned regional pools; retain existing 4 Hz AV channels. Freeze the encoder initially, then compare limited fine-tuning or DINO task distillation separately. A near/far crop variant is justified only if distant-player resolution is inadequate. Preserve aspect ratio and document preprocessing instead of applying an ImageNet center crop that removes relevant court. [Official model](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html).

**Pose:** MediaPipe Pose Landmarker offers 33 landmarks and configurable person count, with mobile/web support. Its default is one pose; it is not evidence that full-court twelve-player pose will be reliable or cheap. Restrict the first experiment to plausible server crops and close visible players, with landmark-validity masks. [Official pose guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).

**Ball:** TrackNetV3 is a useful temporal heatmap/visibility/trajectory reference. The official project licenses code and checkpoints under MIT, but its badminton domain and CUDA reference pipeline do not establish volleyball transfer or phone performance. First compare a qualified multi-frame heatmap approach against the old generic detector on a small non-beach pilot. A compact task-trained heatmap CNN would be the eventual mobile candidate. Keep inpainted positions distinguishable from observations. [TrackNetV3](https://github.com/qaz812345/TrackNetV3).

**CLIP:** treat image-text embeddings as a lower-priority representation ablation. The same volleyball scene can depict waiting, live play or retrieval; prompt similarity cannot establish serve time or rally identity. If tested, feed embeddings to the temporal head and calibrate on held-group development data. Do not treat prompt similarity as a confidence percentage or automatic human label. [CLIP model card](https://github.com/openai/CLIP/blob/main/model-card.md).

MobileCLIP is **not on the product shortlist under its currently published model terms**: MIT code and model-weight permissions differ; the weights' terms exclude product development/commercial use. Distillation is not assumed to bypass them. TinyCLIP is an alternative with an author checkpoint marked MIT, but still needs artifact qualification and a phone benchmark. Neither is the first experiment while we already have useful DINO evidence. [MobileCLIP model terms](https://github.com/apple-aiml-research/ml-mobileclip/blob/main/LICENSE_MODELS), [TinyCLIP author checkpoint](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M).

Retain the existing **standard DINOv2** encoder as the desktop reference/teacher; its official standard code and weights are Apache-2.0. Distill useful regional/task outputs to the CNN, not necessarily every high-dimensional token. A teacher adapted with rally labels must be fitted inside each training fold; held-out source footage must not enter adaptation or pseudo-label training. [DINOv2 source and terms](https://github.com/facebookresearch/dinov2).

## Transformer experiment: change temporal reasoning independently

There are three different proposals: a transformer over feature sequences; a transformer replacing the image CNN; and a video transformer learning from frame sequences. **The first is the initial test.** It is compatible with cached features and our small dataset, and directly asks whether selective temporal relationships improve on convolutions.

Proposed first temporal transformer:

- Two pre-normalized blocks, width 40, two attention heads, feed-forward width 80, relative temporal positions; approximately 31,000 parameters, to be counted after implementation.
- Local attention radius 31 ticks in each block. Two blocks give an overall radius of 62 ticks, matching the TCN's 125-tick field. Full attention in every chunk would change the context budget and confound the comparison.
- Static 256-tick input chunks, central 128 ticks scored, 64 real context ticks on each side; validity masks at actual segment boundaries. Verify identical central predictions under overlapping chunk placement. Reset context at ignored spans/camera cuts.
- Same live/serve/end/auxiliary-keep outputs, normalization provenance, labels, losses, sampling rate, decoder and hyperparameter-search budget as the TCN. Serving **side** stays the separate specialist; a serve-start head does not classify near/far side.

The hypothesis is that attention can connect a faint toss/contact cue to later receiver motion or recognize a reset between two active spans. It cannot restore visual evidence absent from the input. The approximately ±15.5-second context suits offline analysis of uploaded video; a future live-camera product would need a separate latency design.

ASFormer motivates local temporal attention for action segmentation. ActionFormer motivates explicit onset/offset localization, but its published system uses substantial precomputed video features; copying its benchmark head alone does not copy its information or performance. Neither paper establishes gains on our data. A small image transformer such as MobileViTv2 is a later encoder comparison; a full video transformer is a desktop reference/teacher only after cheaper experiments show a remaining motion-information gap. [ASFormer](https://arxiv.org/abs/2110.08568), [ActionFormer implementation](https://github.com/happyharrycn/actionformer_release), [MobileViTv2](https://arxiv.org/abs/2206.02680).

## Future experiment sequence

### 1. Freeze data and qualify observable evidence

Reuse repaired presentation-timestamp feature alignment. Freeze recording IDs, source families, label revisions, ignored intervals and model hashes. Keep raw/project/export/proxy derivatives together. Use exact labels for boundary evaluation; manually reviewed export inclusion/exclusion supervises keep coverage, while uncertain contact/end times remain masked. The repeatedly inspected recording-044 case is a descriptive challenge/UX example, not a new validation set.

Create an error inventory separating complete misses, partial play loss, false merges, extra splits, late starts and premature ends. Include short/failed serves, long rallies, weak motion, far players, indoor/grass, camera quality and confusing dead time. Include uniformly sampled ordinary clips so a curated failure pack is not presented as population performance.

Player qualification: roughly 200–400 frames across source groups plus short sequences, annotated for boxes/footpoints and in-court membership. Measure near/far recall, background tracks, usable motion coverage and geometry failures. Ball qualification: reuse suitable existing ball labels, excluding beach; begin with a small non-beach trajectory packet before expanding to 30–60 clips. Measure visible ball diameter after candidate crops, visible-center recall, false tracks, visibility calibration and source-held transfer. Existing 2,160 ball-label frames belong to only 48 correlated windows and are not dense full-rally trajectories.

Before expensive ball fitting, run a clearly labeled **oracle-feature diagnostic** using manually annotated ball signals: does correct ball information help distinguish current errors from hard negatives? Its result is an upper bound, not deployable performance. Separately measure whether a realistic detector and its inference gate can provide those signals. Poor downstream utility with correct observations is a reason to stop investing in that family.

### 2. Run the smallest controlled matrix when compute is available

| Inputs, fixed within each row | Current TCN | Tiny temporal transformer | Question |
| --- | --- | --- | --- |
| AV104 | Replayed control | First attention arm | Does architecture help without new information? |
| AV104 + player/court movement | First new-feature arm | Transfer arm | Do localized movements help, and does attention add to them? |
| AV104 + cached DINO regions | Existing richer-feature reference | Transfer check | Does a temporal gain transfer to DINO? |
| AV104 + mobile regional CNN | First mobile visual arm | Only after a promising isolated result | Can useful visual information fit the phone budget? |

Do not run every combination of detector, encoder, loss, threshold and head. First screen player features with TCN and attention with cached AV; then run the predeclared transfer comparisons. Compare a frozen CNN before fine-tuning, and fine-tuning before adding distillation. Add ball or selective pose only after qualification. Ablate each new family from a successful combination to identify what contributes.

Use the same source-held folds and three fixed seeds, fitting every scaler/calibrator inside its training partition. Head selection and threshold calibration stay inside development/inner validation. Teachers, new annotations and auxiliary examples obey the same source exclusion. Acquire more independent non-beach matches for a final locked test; freeze the candidate before opening it.

On the 3080, cache frozen features once, fit small heads first, and use bounded image batches for limited encoder adaptation. Actual VRAM determines batch size; no full-match raw-frame attention or from-scratch video transformer is required.

### 3. Add labels that unlock the missing distinctions

| Additional annotation | What it unlocks |
| --- | --- |
| Exact serve contact, physical dead-ball instant, later stand-down, visibility/uncertainty | Separate true event boundaries from the players' delayed relaxation and export padding |
| First-start correction versus additional rally; ordered rally IDs and explicit between-rally gaps | Train split/new-rally decisions instead of merely improving a broad keep mask |
| Ready / serve action / live / dead-reset / unknown | Phase transitions, reset-before-serve evidence and hard negative supervision |
| Court/net/service geometry, sparse player boxes/footpoints and short track IDs | Court-aware player flow, reception movement and selective server crops |
| Ball centers/visibility/primary-court ownership, then short contact trajectories | Domain-specific heatmaps and flight/contact features; mask unobservable positions |
| Review correction type and elapsed effort | Learn calibrated error/review priority rather than equating disagreement with error |

Priority is exact start/end and additional-rally labels, followed by sparse player supervision. The [typed-boundary study](neural-typed-boundary-results-2026-09-19.md) had only seven true secondary-start targets, so a larger model alone has little evidence for that distinction. Include retrieval, practice toss, nearby-court play, foreground crossings and camera shake as valid hard negatives. Unlabeled or ignored time must not become dead-play supervision.

## Product configurations and evaluation

Evaluate standalone models, unchanged production, production plus additional proposals, and production plus review guidance on exactly the same scope. For each new neural arm, rerun serving-side prediction at its inferred rally starts and account for that cost and uncertainty.

Start with **production-preserving additions/guidance**. A continuous cheap pass must inspect the whole recording; ball/pose bursts can then cover both existing starts and independently proposed starts. A specialist invoked only around production detections cannot recover a completely missed rally. Report the gate's proposal recall, duty cycle and recovered rallies, including missed events never triggered for specialist analysis.

Keep separate decisions for additional rallies, start corrections, independent ends and removed regions. Rally IDs and score events remain separate even when padded export intervals join. For provisional automatic boundary edits, review/undo must cover both removed core and removed export time; [earlier removal-only review](neural-boundary-default-review-2026-09-20.md) could conceal internal split gaps. Export coverage alone cannot certify score tracking.

Rank by pooled **F1_padP_coreR at padding 2/2**, following the [canonical contract](../model-ranking-metric.md). Every result must also report padding 0/0, 1/1 and 3/3; pooled P_pad/R_core; model/human export duration and delta; strict gap threshold 3 seconds; identical ignored-range subtraction without rejoining. Include correctly removed unwanted time, incorrect export, wanted padded export omitted and missed actual core seconds.

Report complete missed rallies, short/long-rally coverage, event precision/recall/F1 with fixed one-to-one IoU matching, merge/split errors, and observed serve-start/end errors. A reasonable predeclared recovery screen is at least 20% fewer complete misses than the matched neural control, with no increase in partial/long-rally loss; freeze numeric tolerances before fitting. Production replacement has a stronger requirement: no new completely lost rallies and no decline in retained-play recall relative to production on the comparison scope. These guardrails can reject a higher-ranked F1 model.

Review policies need their own comparison at equal reviewed minutes and equal decision counts: missed rallies recovered, harmful edits found, unique affected rallies, and actual reviewer time. Simulate perfect-human answers as an explicitly labeled upper bound, then measure real review. Predict review need from held-out errors using calibrated model outputs, disagreement and observability, not only entropy: a confidently wrong merged rally still needs review. Showing a flag without correction leaves output metrics unchanged.

Report per-source results and source-group uncertainty; repeated frames/seeds are not independent samples. No credible percentage improvement can be forecast from a model name or public benchmark.

## Mobile qualification before scaling training

Share decoded frames with current feature extraction; count detector tiles/crops and seeking/GOP decode cost. The [existing Android benchmark](android-serving-side-feature-benchmark-2026-08-23.md) shows decoding dominates specialist feature time. More sparse network calls do not necessarily mean cheap video analysis.

Proposed screening budgets, not measurements: temporal head below 1 MiB of weights, at most 32 MiB incremental peak memory and 5% added whole-analysis time; the initial combined visual path targets at most 25% added whole-analysis time, 150 MiB incremental peak memory and 15 MiB added model downloads. Reject or reduce input rate/crops if those limits fail. Pin the budgets and comparison device before execution.

Test Android native, Android browser and an iPhone browser with fallback. Include a representative midrange phone as well as the existing Pixel reference. Measure cold load, complete decode/preprocess/inference/postprocess duration, peak memory, battery/thermal behavior over a sustained recording, and UI responsiveness. Post-export numerical/event parity must precede accuracy claims for fp16/int8 models. Quantization calibration uses training data only.

ONNX Runtime Web supports a broader operator set through WASM than its GPU providers; qualify the exact runtime/backend. For attention, use decomposed MatMul/Add/Softmax/normalization and a tested static local mask: the current WebGPU table notes limitations for fused masked attention. Verify actual graph execution rather than equating successful export with acceleration. [ORT Web](https://onnxruntime.ai/docs/tutorials/web/), [operator table](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md).

For Android, evaluate the actual LiteRT/ONNX execution graph. LiteRT documents that CPU/GPU partitioning can be slower than CPU alone, and its documented GPU delegate executes quantized models through floating-point operations. Int8 is not an automatic GPU speedup. A tiny temporal head may belong on CPU while image inference uses GPU. [LiteRT GPU behavior](https://developers.google.com/edge/litert/performance/gpu).

The first concrete future package is therefore **a matched AV transformer control, player/court-motion features with the current TCN, and a regional MobileNet student with DINO as reference**. Ball and selective pose follow observability/usefulness checks. This separates missing information, temporal modeling and label quality while keeping a measurable route to phone inference.
