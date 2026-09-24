# Mobile-TCN native discrepancy and beach follow-up

Large training, indoor publication, and the original native benchmark are complete. Further DINO work is stopped. No model was retrained or recalibrated in this follow-up.

## Why the editor found the rallies that Android missed

The editor displays saved desktop inference. The native prototype independently generates features from the video. For recording-044, these feature pipelines did not match: the desktop used the full frame, while the benchmark used a default crop; native color conversion assumed limited-range BT.601 for a full-range source. AV downsampling and source-frame selection also differed.

Both checkpoints and decoders match their respective editor options. Replaying each saved feature tensor reproduces its original probabilities: maximum absolute error is below 4.18e-7 for Small and 8.05e-7 for Large. Swapping feature families while keeping the checkpoint and decoder frozen identifies the cause of the disagreement.

| Model and human rally | Native maximum smoothed live probability | Desktop maximum | Entry threshold | Desktop detected core |
|---|---:|---:|---:|---|
| Small, rally 16: 09:22.496–09:28.746 | 0.169 | 0.899 | 0.35 | 09:22.375–09:28.375 |
| Large, rally 16: 09:22.496–09:28.746 | 0.077 | 0.843 | 0.20 | 09:22.875–09:27.375 |
| Large, rally 17: 09:37.868–09:42.367 | 0.079 | 0.500 | 0.20 | 09:38.625–09:44.125 |

Replacing only the embeddings recovers none of these misses. Replacing AV104 or the quality/PTS scalar block recovers a detection in each interval. Replacing AV104 and the scalar block, while retaining native embeddings, restores Small's exact desktop boundary for rally 16 and strong detections for both Large intervals. These substitutions diagnose input drift; they are not selectable new model variants or phone accuracy estimates.

The most consequential individually tested Small scalar was clipped-pixel fraction. It averaged approximately 2.947% near the rally on native versus 0.0125% in desktop features. A desktop reproduction of the native limited-range conversion reproduces this outlier; changing ROI alone does not. See the [detailed Small investigation](native-small-tcn-input-drift.md) for narrow substitutions, color evidence, remaining contract differences, and code references.

The code now uses decoder-output color range and matrix, with track metadata as a per-field fallback. AV and specialist cache identities were advanced so stale features cannot be reused. New analyses in both the normal Android app and benchmark use the full frame, matching the web default. Project matching includes ROI geometry, preventing reuse of a cropped project for a new full-frame analysis; existing saved projects keep their stored geometry and edits. Legacy recovery seeds without ROI provenance remain recoverable but cannot satisfy a new full-frame analysis request; new seeds retain exact geometry and provenance. Replacement neural pooling weights were generated from the verified desktop content geometry. Model checkpoints and decoder thresholds are unchanged.

**Device testing has resumed with the color correction and full-frame default together.** The correction is in the normal application sources, including synchronous and asynchronous AV extraction and the shared serve/side-switch decoder. Remaining resize, rounding, frame-choice, and OpenCV color-conversion differences still require qualification. The original native timings and misses below describe the original prototype, not the corrected implementation.

The original Large full run processed the 17m41s recording in **10m56.983s** including both score specialists; rallies were ready at **9m01.275s**. It found 37 rallies and wholly missed the two intervals above after normal export padding. Its 2-minute pilot median was **70.217s** end-to-end. These runs used native hardware video decoding with FP32 ONNX inference on four CPU threads, not GPU inference.

## Corrected full-frame device replay

The Small, Large, and production cases are complete. Each neural case uses its same frozen highest-recall checkpoint and decoder as the original run, and saved-input temporal and decoder replay checks pass. Production weights and decoder policy also remain unchanged. The normal editor-flow check also passes, as described below. Both preprocessing changes are applied together; this comparison cannot isolate the effect of color handling from removing the crop.

Evaluation uses one fixed saved-human-label revision for `recording-044`: 37 rallies and 321.1462 nonignored core seconds, SHA-256 `97ab7adce70f57ad521add3a18f18765abec61d36b2ba31cf0c739638164d94c`. These are manually reviewed export-derived boundaries with later edits, not independently marked precise serve-contact/dead-ball events. This revision differs from the earlier common-selection panel. Target padding is 2s before and after, with positive gaps strictly under 3s joined for both model and human unions. Ignored time is removed and never rejoined across. Wholly missed means zero retained nonignored human core after padding and joining.

The full source on the phone was hash-verified against the frozen desktop source and manifest, including its 4,373,996,025-byte size. The compared pipelines process the same media bytes; the disagreement is not explained by transferring a different video revision.

| Model | Input pipeline | Found rallies | Wholly missed / 37 human rallies | P_pad | R_core | F1_padP_coreR | Export (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Small | Original native | 37 | 1 | 88.12% | 93.23% | 90.60% | 453.375 |
| Small | Corrected full-frame native | 40 | 0 | 78.20% | 97.93% | 86.96% | 565.750 |
| Small | Frozen desktop reference | 41 | 0 | 84.70% | 99.18% | 91.37% | 537.375 |
| Large | Original native | 37 | 2 | 93.39% | 89.73% | 91.52% | 402.250 |
| Large | Corrected full-frame native | 39 | 0 | 86.84% | 96.99% | 91.63% | 492.250 |
| Large | Frozen desktop reference | 36 | 0 | 92.55% | 97.57% | 94.99% | 466.000 |
| Production ensemble | Original native | 56 | 0 | 61.94% | 96.77% | 75.53% | 722.641 |
| Production ensemble | Corrected full-frame native | 60 | 1 | 66.21% | 96.01% | 78.37% | 650.391 |

Small now detects rally 16 at **09:22.375–09:24.125**, with maximum smoothed live probability 0.652 above its 0.35 entry threshold. It is no longer wholly missed, but the end remains early: after 2s padding, **09:26.125–09:28.746** still loses 2.621s of saved human core. Recall improves by 4.70 percentage points overall; precision falls by 9.92 points and primary F1 falls by 3.65 points. This is partial recovery, not desktop parity.

Large recovers both formerly missed rallies. Rally 16 is detected at **09:23.125–09:25.125** with maximum smoothed live probability 0.414, above its 0.20 threshold; after padding, **09:27.125–09:28.746** still loses 1.621s of core. Rally 17 produces two native cores, **09:38.875–09:41.375** and **09:42.875–09:44.125**. Their padded exports join and retain all saved core time, but the extra boundary is not evidence of two real rallies. Large's recall improves by 7.26 points and precision falls by 6.54 points, leaving F1 only 0.11 points higher. Export coverage and rally separation remain distinct qualification questions.

**The production ensemble changes too.** Its precision increases by 4.27 points and F1 by 2.84 points, while retained-core recall falls by 0.76 points. It now wholly misses saved human rally 22 at **11:08.254–11:11.427** (3.173s), which the original native run retained. Total export shrinks by 72.250s. Higher F1 therefore does not mean every rally is preserved better. This follows from changing shared input generation; no production model weights, suppression threshold, or decoder policy changed. Production per-tick rally probabilities and AV tensors were not saved, so this run supports an output comparison, not the neural-style feature-substitution diagnosis.

| Model | Pipeline | Rallies ready (s) | Both score specialists ready (s) | Video decode + AV (s) | Audio decode + features (s) | Embedding pass (s) | TCN inference (s) | Score specialists (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Small | Original native | 516.199 | 647.150 | 220.879 | 58.149 | 235.918 | 0.163 | 130.951 |
| Small | Corrected full-frame native | 556.871 | 724.430 | 223.378 | 80.012 | 252.164 | 0.131 | 167.559 |
| Large | Original native | 541.275 | 656.983 | 221.694 | 72.315 | 245.521 | 0.225 | 115.708 |
| Large | Corrected full-frame native | 595.719 | 746.346 | 240.022 | 83.995 | 270.098 | 0.226 | 150.627 |
| Production ensemble | Original native | 285.633 | 499.300 | 220.483 | 64.758 | n/a | n/a | 213.667 |
| Production ensemble | Corrected full-frame native | 290.228 | 488.151 | 224.588 | 65.247 | n/a | n/a | 197.923 |

These are single full-video observations, not repeated medians; transfer, installation, and export rendering are excluded. Embedding pass includes its own video decode, image preparation, and encoder inference. The displayed components are selected stages, not an exhaustive sum. Longer audio and specialist passes contribute more than the AV stage to Small's observed elapsed-time increase; one run does not establish its cause. Both corrected score specialists completed for both models. Small's serving-side candidate count changes from 37 to 40, and side-switch evaluated proposals from 38 to 43. Corrected Large evaluates 39 serving-side candidates and 39 side-switch proposals. Neither corrected model selects a side-switch marker. Changed rally boundaries change specialist anchors, so these are behavior differences, not specialist accuracy measurements.

Production also completes both score specialists. Serving-side rows increase from 56 to 60, with review verdicts changing from 11 to nine. Side-switch evaluates 61 proposals in both runs, but selected markers fall from two to one. These changes cannot be called more accurate without independent specialist labels. The production rally/serve/dead-state/suppression inference stage takes 0.207s originally and 0.210s after correction; most processing time is feature generation and score-specialist work.

Replaying the actual saved corrected tensors reproduces phone probabilities within **5.07e-7 for Small** and **7.75e-7 for Large**, and reproduces their native decoded boundaries. This validates the frozen graph and decoder replay; it does not make the input features identical to desktop. Embedding differences fall substantially after correction, while AV104 differences remain. With native embeddings and quality scalars retained, substituting desktop AV104 restores Small's rally 16 boundary to **09:22.375–09:28.375**. The same substitution restores Large's rally 16 to **09:22.875–09:27.375** and rally 17 to one core at **09:38.375–09:44.125**. Embedding-only substitution does not resolve either model's early end; changing only Small's selected-frame timestamp scalar also fails and worsens its F1. AV input drift is the dominant remaining discrepancy in these cases. These frozen substitutions are diagnostics, not deployed outputs, retrained models, or selectable variants; their complete metrics and all four padding cases remain in the comparison artifact.

One remaining visual contract difference is explicit: native AV samples one source YUV pixel for each 192×108 output pixel, while desktop uses area downsampling. Focus/blur/visibility formulas match but their inputs and results differ. Audio feature order and timestamp origin also match the desktop contract; unchanged native audio features still differ from desktop, so an audio root cause cannot be assigned without comparing decoded PCM. The next qualification target is matched video reduction and paired audio samples, not another checkpoint or threshold adjustment.

The corrected run saved each neural representation for audit. Decimal MB are used below. AV104 and its production context are calculated from the 4,245 float32 rows because those arrays were not persisted. Embedding/fused/probability files were measured after collection.

| Model | AV104 calculated | AV520 context calculated | Embedding tokens | Fused features | Probabilities | Total saved neural diagnostics | Decoded specialist payloads |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 1.766 MB | 8.830 MB | 19.566 MB | 41.024 MB | 0.068 MB | 60.657 MB | 0.088 MB |
| Large | 1.766 MB | 8.830 MB | 32.609 MB | 67.105 MB | 0.068 MB | 99.782 MB | 0.085 MB |
| Production ensemble | 1.766 MB | 8.830 MB | n/a | n/a | not saved | none | 0.130 MB |

Tokens are duplicated inside the fused representation; these are diagnostic storage sizes, not additive minimum RAM or measured peak memory. The separate one-to-one rally matching diagnostic uses IoU at least 0.5 and excludes events touched by ignored time:

| Corrected model | Evaluable predictions | Matched / 37 human rallies | Event precision | Event recall | Event F1 |
|---|---:|---:|---:|---:|---:|
| Small | 40 | 27 | 67.50% | 72.97% | 70.13% |
| Large | 39 | 33 | 84.62% | 89.19% | 86.84% |
| Production ensemble | 55 | 27 | 49.09% | 72.97% | 58.70% |

Production's five remaining predictions touch ignored time and are excluded from event matching. These boundary-sensitive event metrics remain separate from retained-play recall and from wholly missed counts.

All required Small padding sensitivities:

| Pipeline | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original native | 0s | 81.36% | 78.63% | 79.97% | 310.375 | 321.146 | -10.771 |
| Original native | 1s | 85.95% | 89.32% | 87.61% | 383.375 | 397.146 | -13.771 |
| Original native | 2s | 88.12% | 93.23% | 90.60% | 453.375 | 470.645 | -17.270 |
| Original native | 3s | 89.85% | 95.42% | 92.56% | 523.375 | 544.644 | -21.269 |
| Corrected full-frame native | 0s | 70.73% | 91.32% | 79.72% | 414.625 | 321.146 | +93.479 |
| Corrected full-frame native | 1s | 75.44% | 96.09% | 84.52% | 490.125 | 397.146 | +92.979 |
| Corrected full-frame native | 2s | 78.20% | 97.93% | 86.96% | 565.750 | 470.645 | +95.105 |
| Corrected full-frame native | 3s | 82.34% | 99.18% | 89.98% | 633.125 | 544.644 | +88.481 |
| Frozen desktop reference | 0s | 77.75% | 95.48% | 85.71% | 394.375 | 321.146 | +73.229 |
| Frozen desktop reference | 1s | 81.52% | 98.43% | 89.18% | 469.375 | 397.146 | +72.229 |
| Frozen desktop reference | 2s | 84.70% | 99.18% | 91.37% | 537.375 | 470.645 | +66.730 |
| Frozen desktop reference | 3s | 86.98% | 99.49% | 92.82% | 611.375 | 544.644 | +66.731 |

All required Large padding sensitivities:

| Pipeline | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original native | 0s | 88.17% | 71.04% | 78.69% | 258.750 | 321.146 | -62.396 |
| Original native | 1s | 91.99% | 83.65% | 87.62% | 332.250 | 397.146 | -64.896 |
| Original native | 2s | 93.39% | 89.73% | 91.52% | 402.250 | 470.645 | -68.395 |
| Original native | 3s | 94.50% | 93.79% | 94.14% | 474.500 | 544.644 | -70.144 |
| Corrected full-frame native | 0s | 81.48% | 87.03% | 84.16% | 343.000 | 321.146 | +21.854 |
| Corrected full-frame native | 1s | 85.25% | 94.05% | 89.43% | 417.000 | 397.146 | +19.854 |
| Corrected full-frame native | 2s | 86.84% | 96.99% | 91.63% | 492.250 | 470.645 | +21.605 |
| Corrected full-frame native | 3s | 89.06% | 98.78% | 93.67% | 566.500 | 544.644 | +21.856 |
| Frozen desktop reference | 0s | 88.38% | 88.21% | 88.29% | 320.500 | 321.146 | -0.646 |
| Frozen desktop reference | 1s | 91.53% | 95.38% | 93.42% | 392.500 | 397.146 | -4.646 |
| Frozen desktop reference | 2s | 92.55% | 97.57% | 94.99% | 466.000 | 470.645 | -4.645 |
| Frozen desktop reference | 3s | 94.64% | 98.98% | 96.76% | 536.000 | 544.644 | -8.644 |

All required production padding sensitivities:

| Pipeline | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original native | 0s | 54.06% | 91.09% | 67.85% | 541.141 | 321.146 | +219.995 |
| Original native | 1s | 57.41% | 95.26% | 71.64% | 644.391 | 397.146 | +247.246 |
| Original native | 2s | 61.94% | 96.77% | 75.53% | 722.641 | 470.645 | +251.997 |
| Original native | 3s | 68.23% | 97.86% | 80.40% | 763.391 | 544.644 | +218.748 |
| Corrected full-frame native | 0s | 61.37% | 84.37% | 71.05% | 441.516 | 321.146 | +120.370 |
| Corrected full-frame native | 1s | 64.04% | 94.07% | 76.20% | 557.016 | 397.146 | +159.871 |
| Corrected full-frame native | 2s | 66.21% | 96.01% | 78.37% | 650.391 | 470.645 | +179.747 |
| Corrected full-frame native | 3s | 70.23% | 96.80% | 81.40% | 725.516 | 544.644 | +180.873 |

## Normal Android app check

The actual `EditorActivity` flow was tested through **Choose video**, without a benchmark activity or injected precomputed result. The two-minute excerpt from `recording-044` was selected with the full game window, and **Teams change court sides** enabled. The saved project records full-frame ROI with known provenance, an analysis window of **0–120.05s**, ready status, and no analysis or specialist errors.

It found **six rallies**, ready in **27.732s**; both score specialists completed by **44.929s**. Video decoding and AV features took 23.429s, audio features 3.980s, contextualization 0.062s, rally inference 0.225s, and score specialists 17.197s. Serving-side processed six anchors (two near, three far, one review); side-switch evaluated five proposals and selected no marker. This is one normal UI run with its own analysis window, not a replacement for the earlier pilot medians.

Playback and a moving playhead were visually verified. After force-stop and relaunch, the six-clip ready project reopened successfully; the three preexisting projects remained intact alongside the new project. The normal debug APK's installed hash was verified. This qualifies the shared corrected production pipeline in the actual app; Small and Large neural choices remain in the benchmark flavor and are not yet normal-editor options.

## Frozen Large models on beach footage

Both selected Large variants have been run on both beach recordings and added to editor-lab and labelv2, including live, serve/start, end, and keep signals. Existing model options, human labels, draft revisions, and non-beach entries were preserved. Both UI APIs were checked for both recordings.

These are frozen indoor-selected models: beach labels were used only after inference for evaluation. Both recordings belong to **one source group**, so this is a small domain-transfer check. Recall here measures retained human core time; it is not the fraction of rally events detected. A 99% calibration target is not a guarantee of 99% recall on new footage.

Target product padding is 2s before and after, with positive gaps strictly under 3s joined. Model and human unions use the same rules. Ignored time is subtracted and never rejoined across. Metrics pool time across both recordings.

| Selection | Found rallies | Wholly missed / 71 human rallies | P_pad | R_core | F1_padP_coreR |
|---|---:|---:|---:|---:|---:|
| Highest F1 | 16 | 56 | 99.80% | 15.42% | 26.71% |
| Highest recall | 38 | 40 | 99.21% | 40.14% | 57.15% |

Wholly missed means zero retained nonignored human core after model padding and gap joining. It does not use one-to-one event matching.

| Selection | Recording index | Found | Wholly missed | P_pad | R_core | F1_padP_coreR |
|---|---|---:|---:|---:|---:|---:|
| Highest F1 | beach-source-02 | 5 | 35 | 99.39% | 9.51% | 17.36% |
| Highest F1 | beach-source-01 | 11 | 21 | 100.00% | 23.17% | 37.62% |
| Highest recall | beach-source-02 | 17 | 26 | 98.10% | 29.32% | 45.15% |
| Highest recall | beach-source-01 | 21 | 14 | 100.00% | 54.31% | 70.39% |

All required padding sensitivities:

| Selection | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Highest F1 | 0s | 99.36% | 5.78% | 10.93% | 30.07 | 516.49 | -486.42 |
| Highest F1 | 1s | 99.69% | 11.38% | 20.43% | 62.07 | 658.49 | -596.42 |
| Highest F1 | 2s | 99.80% | 15.42% | 26.71% | 94.07 | 800.49 | -706.42 |
| Highest F1 | 3s | 99.85% | 19.18% | 32.18% | 129.05 | 948.30 | -819.25 |
| Highest recall | 0s | 98.34% | 23.30% | 37.68% | 122.40 | 516.49 | -394.09 |
| Highest recall | 1s | 98.94% | 33.75% | 50.33% | 192.65 | 658.49 | -465.84 |
| Highest recall | 2s | 99.21% | 40.14% | 57.15% | 258.65 | 800.49 | -541.84 |
| Highest recall | 3s | 99.37% | 45.18% | 62.12% | 325.67 | 948.30 | -622.63 |

The frozen indoor models are too conservative on these beach recordings. This is separate from the Android discrepancy: beach inference used the validated desktop feature path. Further beach work would need source-group-separated beach development data and a protected beach evaluation group; neither additional training nor threshold selection was performed here.

## Evidence indexes

- `private-reference-0211`: frozen Large experiment, selections, and graph qualification.
- `private-reference-0212` / `private-reference-0213`: original Large pilot / full native observations.
- `private-reference-0215`: beach report JSON, all padding cases and event diagnostics, feature checks, UI preservation receipt, and frozen source snapshot.
- `private-reference-0217`: Small/Large feature counterfactuals, color diagnostic, private tensors, and corrected full-frame input contract.
- `private-reference-0218`: completed corrected full-frame Small/Large run receipts and collected tensors.
- `private-reference-0219`: completed corrected full-frame production run, including both score specialists.
- `private-reference-0220`: completed normal application editor-flow check, full-frame saved-project receipt, timing, playback/reopen verification, and preservation of existing projects.
- `private-reference-0221`: corrected comparison and storage artifacts, including frozen-input replay checks, counterfactuals, all padding cases, rally ranges, specialist changes, and measured saved-file sizes.

Exact locations resolve through the external private ledger and ignored environment files. No source media or private path is embedded in this report.

Validation: normal-app `:app:testDebugUnitTest` and `:app:assembleDebug` pass without the benchmark flag; 165 tests pass with zero failures, errors, or skipped tests, including recovery-seed geometry coverage. The normal APK uses `com.volleycut.nativeanalysis.debug`. The corrected full-frame benchmark APK builds, was installed, and its installed hash was verified. The later recovery-seed change affects normal-app project reuse and does not modify the frozen benchmark feature pipeline. Normal editor-flow testing passes as detailed above. No release was published.
