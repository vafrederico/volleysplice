# Recall, detector alternatives and DINO compression execution

The user authorizes four experiments: YOLO or one/two alternatives, DINO FP16/INT8, a DINO-distilled MobileNet, and a 99% recall selection requirement. This is development research, with no production deployment. The completed 2026-09-22 registrations and results remain immutable.

## Work tracking

The Tasks tool is unavailable. This protocol records the frozen initial scope; completion is recorded separately in the results document and NAS artifacts.

- [x] Inspect storage/GPU availability and existing source-held contracts.
- [ ] Complete the two detector alternatives and visual qualification.
- [ ] Re-select existing models at 99% inner recall; refit missing selected outer checkpoints where feasible.
- [ ] Compare FP32/FP16/mixed INT8 DINO embeddings and downstream rally outputs.
- [ ] Prepare label-blind shared image inputs and qualify a fold-isolated student distillation implementation.
- [ ] Fit and evaluate the distilled student with the matched mobile temporal head.
- [ ] Independently audit source isolation, selection, metrics and artifacts; report all outcomes.
- [ ] Commit and push the completed work.

## Shared contract

Use the same repaired 18-recording development corpus: 8 exact, 3 reviewed-draft and 7 reviewed-export recordings. Exact evaluation contains 4 source groups and 322 original rallies. Group raw/project/proxy derivatives together. Beach and protected test sources remain excluded from fitting, unsupervised adaptation, threshold selection and evaluation. Gold and ignored revisions remain unchanged.

The primary target is symmetric 2-second export padding. For each seed, pool duration numerators and denominators across recordings, then compute P_pad, R_core and F1_padP_coreR. Rank eligible models by F1_padP_coreR. Report all symmetric padding cases 0/1/2/3, model/human export durations and their difference. Join positive gaps strictly below 3 seconds on both unions; subtract ignored spans afterward and never rejoin across them. Report event P/R/F1, complete and partial/any-core losses, short (original duration <=3 seconds) losses, long core recall, source groups, correctly removed time, incorrect export and wanted export omitted.

The new eligibility floor is pooled inner R_core >=0.99. It is an operating-point selection requirement, not a probability calibration or guarantee on an unseen source. A fold with no eligible configuration is explicitly infeasible: do not silently lower the floor. Do not rank partial-scope results as a complete model. Neither development F1 nor passing the floor authorizes replacement of production. Production retains its historical label-exposure caveat.

Use seeds 3407/1729/20260918 and temporal checkpoints 5/15/30/60. Keep the existing 48 decoder candidates (192 checkpoint/decoder choices per outer fold), short-boost loss and temporal fit recipe fixed. A broader decoder search, stronger supervised loss or a different distillation recipe would be another declared experiment, not an outcome-driven amendment here.

## Existing-model 99% comparison

Reconstruct the original 95% choices and metrics from saved inner scores as a regression check. Then jointly select epoch and decoder under the strict 99% floor for AV TCN, DINO TCN, AV transformer, DINO transformer and frozen regional MobileNet + TCN. Reuse hash-verified outer scores only when the selected epoch exists; otherwise refit that outer training partition with the exact original seed, inputs, sampling and numerical implementation. Original artifacts are never overwritten.

Also report a secondary decoder-only sensitivity at each original selected epoch. Label its scope explicitly. No old-95% fallback policy is treated as a successful 99% model. Archive every candidate metric, source exclusions, selected/failed status and outer score provenance.

## Person detector alternatives

Test exactly two new configurations: official YOLOX-Nano 416 (0.1.1rc0) and Torchvision SSDLite320 MobileNetV3-Large COCO_V1. Pin artifact/source/preprocessing identities before inference. Use a single existing ROI, no tiling or outcome-driven threshold sweep. Person score >=0.25, official NMS (YOLOX 0.45; SSD 0.55), post-person-filter cap 24 with uncapped counts recorded.

Use fixed times 0/5/10.5/20.5/29.5 seconds and 25/50/75% of duration in four exact sources: grass-source-03, grass-source-01, indoor-source-01 and indoor-source-08. Add first-30-second 2Hz sequences on the prior grass/indoor pilot pair. Inspect all fixed overlays plus duplicate/false-location examples. This is visual engineering qualification, not fabricated person recall ground truth. Repeated gross errors in either environment reject that configuration; otherwise only a bounded person-feature follow-up is justified. No full detector-feature rally training is bundled into this batch.

## DINO precision

Keep DINOv2 ViT-S/14, 336-pixel input, 4Hz sampling and 10x384 pooled embeddings. Compare actual FP32 encoder, CUDA FP16 encoder, and ONNX mixed dynamic INT8 (quantized eligible MatMul weights and dynamic activations; unsupported operations remain floating point). Report exactly which operators/weights are quantized. Dynamic quantization uses no fitted activation-calibration dataset. Do not call it a fully INT8 graph or a phone benchmark.

First qualify FP32 ONNX against the original PyTorch encoder, including pixels/timestamps/pooling. Regenerate label-blind embeddings for the 8 exact recordings and replay all 3 existing DINO + TCN seed models and their frozen decoders. Preserve the existing float16 cache storage convention. Measure embedding and downstream score drift, all padding/rally-loss metrics, bytes and desktop throughput. Compare under the historical 95% settings to isolate precision effects and, when available, the new 99% settings without precision-specific threshold tuning. A precision path that fails conversion or parity is reported as unsupported rather than silently substituted.

## DINO-distilled mobile encoder

The student starts from the same official MobileNetV3-Small ImageNet V1 weights and keeps its 927,008-parameter feature encoder. The deployed downstream inputs remain the same four 576-value regional pools at 2Hz plus the same eight quality/age/availability values and AV104. The temporal TCN remains 44,692 parameters with the same four supervised heads and loss. Thus comparison with the frozen MobileNet arm changes the encoder adaptation, not the deployed input width or temporal architecture.

Distillation supervises the student image encoder from the frozen official DINOv2-S/14 teacher. For learning only, pool the student's final feature map into a global average plus a 3x3 grid and project each 576-vector with a shared learned 576->384 linear layer. Match L2-normalized vectors using cosine loss: half the loss for the global token, half for the mean of nine spatial tokens. The auxiliary projection is discarded for deployment; regional mobile features retain the original pooling contract. This is feature distillation, not a claim of copying the teacher's rally accuracy.

Prepare shared 2Hz 224-pixel uint8 inputs on NAS using the existing mobile PTS/ROI/letterbox path. Preserve exact selected frame identities, timestamps and quality fields against the existing mobile caches. For each recording choose at most 128 evenly spaced valid 2Hz frames for distillation; exclude ignored time. Compute teacher targets freshly from the same decoded frames at 336 pixels, avoiding mismatched old timestamp/frame-selection conventions. Preprocessing and teacher target extraction use sanitized image/source inputs, never rally labels. Selection of valid training frames uses only the ignored/valid mask, not live/dead labels.

For each seed and each of the six excluded-source-pair inner fits/four outer refits, train a separate student using only that fit's permitted exact and auxiliary source groups. No held-out imagery enters student fitting, including unlabeled distillation. Every fit starts from the same pretrained encoder and seed; freeze BatchNorm running statistics, train encoder weights and temporary projector for exactly 8 epochs with AdamW lr1e-4/weight_decay1e-4, batch16, and group-balanced per-frame weights. No teacher/student checkpoint is chosen using a held-out source. Teacher is frozen and has no task adaptation.

Extract fold-specific student features for that fold's fitting and evaluation records, fit the temporal head with the historical recipe and select its checkpoint/decoder inside the outer fold at the new 99% floor. This is 30 physical student fits and at most 30 temporal fits across 3 seeds. Report student loss endpoints and embedding agreement as diagnostics, not rally success criteria. Compare against the frozen-MobileNet arm under the identical 99% selection contract, retaining the historical 95% results only as context.

Before full fitting, verify identical frame/pool/alignment behavior for an unchanged student, strict source exclusion including auxiliary groups, deterministic resume, finite training, unchanged AV/labels/masks and independent metric replay. If engineering fails, repair the implementation and record the change before outcome evaluation; do not alter the declared learning recipe based on held-source outcomes.

## Resources and mobile qualification

All new weights, image inputs, feature caches, temporary files, downloads and reports live under private-reference-0118 (WSL direct NAS mount private-reference-0100). New jobs set NAS TMPDIR/TMP/TEMP/XDG_CACHE_HOME/TORCH_HOME/HF_HOME/CUDA_CACHE_PATH and disable bytecode writes. Existing read-only dependencies can remain in their prior location. WSL root and OS-managed swap remain on C:; monitor RAM/concurrency and physical C: free space.

One GPU workload at a time. Do not restart WSL or stop unrelated work. At inspection the RTX3080 used 2,659/10,240MiB with 0% utilization; C: had 37.6GiB free. Bound CPU workers too. No new physical-phone performance claim is permitted without measurement. Numeric precision, runtime support, weight size and actual end-to-end phone latency are separate findings.
