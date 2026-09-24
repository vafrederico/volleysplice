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

The code now uses decoder-output color range and matrix, with track metadata as a per-field fallback. AV and specialist cache identities were advanced so stale features cannot be reused. The benchmark can explicitly use full-frame ROI; its replacement pooling weights were generated from the verified desktop content geometry. The model graphs, checkpoints, and decoder thresholds are unchanged.

**Phone work is stopped at the user's request; the device is unavailable.** The correction is in the normal application sources, including synchronous and asynchronous AV extraction and the shared serve/side-switch decoder. Color unit tests and Android builds validate implementation behavior, not actual end-to-end pixel parity or rally recovery. Remaining resize, rounding, frame-choice, and OpenCV color-conversion differences still require qualification. The original native timings and misses describe the original prototype.

The original Large full run processed the 17m41s recording in **10m56.983s** including both score specialists; rallies were ready at **9m01.275s**. It found 37 rallies and wholly missed the two intervals above after normal export padding. Its 2-minute pilot median was **70.217s** end-to-end. These runs used native hardware video decoding with FP32 ONNX inference on four CPU threads, not GPU inference.

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

Exact locations resolve through the external private ledger and ignored environment files. No source media or private path is embedded in this report.

Validation: normal-app `:app:testDebugUnitTest` and `:app:assembleDebug` pass without the benchmark flag; 156 tests pass with zero failures, errors, or skipped tests. The APK reports the normal `com.volleycut.nativeanalysis.debug` application ID. Benchmark build/tests also pass. No APK was installed or published after the correction.
