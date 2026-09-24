# DINO encoder precision experiment — 2026-09-23

FP16 CUDA is the useful result from this precision experiment: it halves encoder weight storage and preserves every rally's retained core time at the primary 2-second padding across all three original operating points. Mixed dynamic INT8 reduces file size further, but lowers mean recall and F1 in that native CPU comparison and fails the measured CPU/browser parity check. In the sole complete strict99% seed it slightly raises recall while reducing precision and F1. Keep FP16 as the desktop candidate; this INT8 graph is not qualified for deployment.

This isolates encoder precision while keeping all three historical DINO+TCN seeds, fold scalers, checkpoints and decoders fixed. These are the original 95%-inner-recall-selected operating points. A separate strict99% replay uses the newly selected controls without precision-specific tuning. No encoder training or protected-test evaluation occurs here.

The scope is eight development recordings from four source groups, 33,382 sampled frames, 322 rallies, with the existing ignored ranges excluded from rally evaluation. FP32 and FP16 refer to encoder arithmetic: every resulting embedding cache is stored as float16, matching the historical representation.

## Frozen-head rally results

Target export padding is 2 seconds before and after, and padded gaps are joined only when strictly below 3 seconds. The table averages three per-seed pooled evaluations. Recall means retained human core play time, not the count of separately identified rallies.

| Encoder | P_pad | R_core | F1_padP_coreR | Export min | Correctly removed min | Incorrect export min | Wanted export omitted min |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 encoder reference | 88.475% | 96.955% | 92.513% | 62.331 | 69.378 | 7.203 | 6.125 |
| FP16 CUDA encoder | 88.527% | 96.955% | 92.541% | 62.296 | 69.413 | 7.168 | 6.125 |
| Mixed dynamic INT8 CPU encoder | 88.079% | 96.548% | 92.105% | 62.319 | 69.127 | 7.454 | 6.388 |

Correctly removed is evaluable time outside both padded human and model exports. Incorrect export is extra footage outside the padded human export. Wanted export omitted is padded human footage missing from the model export.

| Encoder | Complete losses | Incomplete losses | Short complete losses | Long-rally core recall | Event F1 |
|---|---:|---:|---:|---:|---:|
| FP32 encoder reference | 13.67 | 43.00 | 11.00 | 97.649% | 74.165% |
| FP16 CUDA encoder | 13.67 | 43.00 | 11.00 | 97.649% | 74.204% |
| Mixed dynamic INT8 CPU encoder | 14.67 | 44.67 | 12.33 | 97.332% | 72.288% |

Complete loss means no human rally core retained by the padded export. Incomplete includes complete and partial losses. Short means original duration at most 3 seconds. Event F1 uses uncensored one-to-one IoU at least 0.5 and does not certify precise serve boundaries.

| Encoder | ΔP percentage points | ΔR percentage points | ΔF1 percentage points | Δ missed core seconds |
|---|---:|---:|---:|---:|
| FP16 CUDA encoder | +0.0526 | +0.0000 | +0.0279 | +0.000 |
| Mixed dynamic INT8 CPU encoder | -0.3959 | -0.4070 | -0.4077 | +9.715 |

Paired human-rally coverage, averaged over the three seeds with padding of 2 seconds:

| Encoder | Rallies retaining less core | Rallies retaining more core | New complete losses versus FP32 |
|---|---:|---:|---:|
| FP16 CUDA encoder | 0.00 | 0.00 | 0.00 |
| Mixed dynamic INT8 CPU encoder | 17.00 | 8.00 | 3.00 |

## Encoder size and engineering runtime

| Artifact | Actual bytes | Measurement | Median image inference |
|---|---:|---|---:|
| FP32 ONNX | 88,359,163 | Desktop ORT CPU, 2 threads | 138.72 ms |
| FP16 PyTorch state | 44,167,592 | RTX 3080 CUDA | 5.26 ms |
| Dynamic INT8 ONNX | 24,980,885 | Desktop ORT CPU, 2 threads | 93.14 ms |

The same CUDA pilot measured FP32 at 7.33 ms/image. CPU and GPU rows use different backends and are not a precision-only speed comparison across rows. These are prepared-input, batch-one encoder timings on the Ryzen 9 5900X/RTX 3080 desktop. They exclude decoding, preprocessing, AV features, the temporal head and UI. Concurrent experiments share the host, and phone battery, thermals and memory remain unmeasured.

The quantized graph contains 48 MatMulInteger and 48 DynamicQuantizeLinear nodes. Weights are per-channel signed INT8 and activations dynamically quantized to unsigned INT8. The 24 attention MatMuls, patch Conv, LayerNorm, Softmax and other operations remain FP32. It is a mixed graph, not full INT8. No activation-calibration dataset or label fitting is used.

## Browser qualification

Unlike the earlier temporal-head-only experiment, this runs the actual 336px DINO image encoder in desktop Chrome with ONNX Runtime Web 1.22.0, WASM CPU, one thread, on two fixed label-blind grass/indoor inputs.

| Graph | CPU↔browser parity | Browser median | Maximum output difference |
|---|---|---:|---:|
| fp32 | Pass | 1286.0 ms/image | 0.000016 |
| dynamic-int8 | FAIL | 751.6 ms/image | 1.052268 |

**The INT8 CPU accuracy results cannot be assumed to transfer to the browser:** the graph runs but fails CPU/browser numerical parity. FP32 executes with parity, but this desktop single-thread timing does not establish an acceptable phone experience. FP16 was tested with PyTorch CUDA, not a phone or browser delegate.

## Embedding drift and bounded INT8 alternatives

| Encoder | Mean cosine | Minimum token cosine | RMSE | Maximum absolute difference |
|---|---:|---:|---:|---:|
| FP16 CUDA encoder | 0.99999772 | 0.999968 | 0.004294 | 0.060547 |
| Mixed dynamic INT8 CPU encoder | 0.88799514 | 0.676797 | 0.940407 | 6.973633 |

Embedding statistics include all sampled frames, including ignored time; rally metrics exclude ignored time. On the fixed 32-frame pilot, original signed full-range INT8 had RMSE 0.945 and mean cosine 0.8864. Two predeclared label-blind alternatives—signed reduced range and unsigned full range—gave RMSE 0.962 and 0.948. Neither solved the drift; this does not support AVX2 U8S8 saturation as its main cause. The original graph had the lowest pilot RMSE and remained the full replay candidate. No rally outcomes selected this choice.

## Required padding sensitivity

| Encoder | Before/after s | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 encoder reference | 0 | 84.337% | 86.222% | 85.254% | 2441.886 | 2387.151 | +54.735 |
| FP32 encoder reference | 1 | 86.689% | 94.635% | 90.476% | 3101.642 | 3031.151 | +70.491 |
| FP32 encoder reference | 2 | 88.475% | 96.955% | 92.513% | 3739.831 | 3675.151 | +64.680 |
| FP32 encoder reference | 3 | 90.076% | 97.835% | 93.790% | 4364.456 | 4323.263 | +41.193 |
| FP16 CUDA encoder | 0 | 84.332% | 86.255% | 85.267% | 2442.986 | 2387.151 | +55.835 |
| FP16 CUDA encoder | 1 | 86.713% | 94.638% | 90.490% | 3101.069 | 3031.151 | +69.918 |
| FP16 CUDA encoder | 2 | 88.527% | 96.955% | 92.541% | 3737.753 | 3675.151 | +62.602 |
| FP16 CUDA encoder | 3 | 90.122% | 97.835% | 93.814% | 4362.378 | 4323.263 | +39.115 |
| Mixed dynamic INT8 CPU encoder | 0 | 83.662% | 85.758% | 84.667% | 2449.022 | 2387.151 | +61.871 |
| Mixed dynamic INT8 CPU encoder | 1 | 86.057% | 94.111% | 89.882% | 3108.372 | 3031.151 | +77.221 |
| Mixed dynamic INT8 CPU encoder | 2 | 88.079% | 96.548% | 92.105% | 3739.122 | 3675.151 | +63.971 |
| Mixed dynamic INT8 CPU encoder | 3 | 89.641% | 97.562% | 93.424% | 4370.881 | 4323.263 | +47.618 |

## Independently selected 99% recall controls

This replay holds the new FP32-selected strict99% checkpoint and decoder selection fixed across precisions. Eligibility is measured on inner validation groups; held-source recall is not guaranteed. No precision-specific recalibration or requalification of the inner 99% constraint was performed. Two of twelve DINO seed/source folds were infeasible within the frozen grid and remain absent. No relaxed threshold or production fallback fills those folds.

**Compare precisions within a seed only.** The full-scope row and partial feasible-subset rows have different recording populations, so no three-seed aggregate is reported here. The original95% eight-recording table above remains the complete matched precision-isolation comparison.

| Seed | Scope | Encoder | P_pad | R_core | F1_padP_coreR | Complete losses |
|---:|---|---|---:|---:|---:|---:|
| 3407 | Partial: 7/8 recordings | FP32 encoder reference | 84.177% | 97.550% | 90.372% | 10 |
| 3407 | Partial: 7/8 recordings | FP16 CUDA encoder | 84.197% | 97.550% | 90.383% | 10 |
| 3407 | Partial: 7/8 recordings | Mixed dynamic INT8 CPU encoder | 85.200% | 96.775% | 90.619% | 11 |
| 1729 | Partial: 5/8 recordings | FP32 encoder reference | 83.499% | 99.927% | 90.978% | 0 |
| 1729 | Partial: 5/8 recordings | FP16 CUDA encoder | 83.506% | 99.927% | 90.982% | 0 |
| 1729 | Partial: 5/8 recordings | Mixed dynamic INT8 CPU encoder | 84.269% | 99.687% | 91.332% | 0 |
| 20260918 | All 8 recordings | FP32 encoder reference | 86.089% | 98.841% | 92.025% | 5 |
| 20260918 | All 8 recordings | FP16 CUDA encoder | 86.077% | 98.841% | 92.019% | 5 |
| 20260918 | All 8 recordings | Mixed dynamic INT8 CPU encoder | 85.314% | 98.961% | 91.632% | 4 |

Guardrails for those same strict99% seed/scope combinations:

| Seed | Encoder | Incomplete losses | Short complete losses | Long-rally core recall | Event F1 | Missed core s |
|---:|---|---:|---:|---:|---:|---:|
| 3407 | FP32 encoder reference | 30 | 9 | 98.200% | 76.330% | 47.726 |
| 3407 | FP16 CUDA encoder | 30 | 9 | 98.200% | 76.330% | 47.726 |
| 3407 | Mixed dynamic INT8 CPU encoder | 32 | 9 | 97.405% | 75.978% | 62.833 |
| 1729 | FP32 encoder reference | 1 | 0 | 99.924% | 80.710% | 1.280 |
| 1729 | FP16 CUDA encoder | 1 | 0 | 99.924% | 80.710% | 1.280 |
| 1729 | Mixed dynamic INT8 CPU encoder | 2 | 0 | 99.671% | 80.804% | 5.530 |
| 20260918 | FP32 encoder reference | 19 | 4 | 99.127% | 78.313% | 27.657 |
| 20260918 | FP16 CUDA encoder | 19 | 4 | 99.127% | 78.313% | 27.657 |
| 20260918 | Mixed dynamic INT8 CPU encoder | 17 | 3 | 99.121% | 76.347% | 24.812 |

Strict99% sensitivity retains exactly the same per-seed feasible scope and the same FP32-selected decoder for all three precision arms within each fold:

| Seed | Encoder | Before/after s | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3407 | FP32 encoder reference | 0 | 78.307% | 88.912% | 83.273% | 2212.200 | 1948.351 | +263.849 |
| 3407 | FP32 encoder reference | 1 | 82.038% | 95.634% | 88.316% | 2749.917 | 2480.351 | +269.566 |
| 3407 | FP32 encoder reference | 2 | 84.177% | 97.550% | 90.372% | 3288.900 | 3012.351 | +276.549 |
| 3407 | FP32 encoder reference | 3 | 86.080% | 98.296% | 91.783% | 3818.150 | 3545.620 | +272.530 |
| 3407 | FP16 CUDA encoder | 0 | 78.332% | 88.898% | 83.281% | 2211.167 | 1948.351 | +262.816 |
| 3407 | FP16 CUDA encoder | 1 | 82.061% | 95.634% | 88.329% | 2749.150 | 2480.351 | +268.799 |
| 3407 | FP16 CUDA encoder | 2 | 84.197% | 97.550% | 90.383% | 3288.133 | 3012.351 | +275.782 |
| 3407 | FP16 CUDA encoder | 3 | 86.097% | 98.296% | 91.793% | 3817.383 | 3545.620 | +271.763 |
| 3407 | Mixed dynamic INT8 CPU encoder | 0 | 79.280% | 88.235% | 83.518% | 2168.433 | 1948.351 | +220.082 |
| 3407 | Mixed dynamic INT8 CPU encoder | 1 | 83.033% | 94.868% | 88.557% | 2695.558 | 2480.351 | +215.207 |
| 3407 | Mixed dynamic INT8 CPU encoder | 2 | 85.200% | 96.775% | 90.619% | 3226.075 | 3012.351 | +213.724 |
| 3407 | Mixed dynamic INT8 CPU encoder | 3 | 87.112% | 97.677% | 92.093% | 3746.325 | 3545.620 | +200.705 |
| 1729 | FP32 encoder reference | 0 | 78.031% | 95.978% | 86.079% | 2170.358 | 1764.514 | +405.844 |
| 1729 | FP32 encoder reference | 1 | 80.976% | 99.507% | 89.290% | 2624.000 | 2194.514 | +429.486 |
| 1729 | FP32 encoder reference | 2 | 83.499% | 99.927% | 90.978% | 3061.375 | 2624.514 | +436.861 |
| 1729 | FP32 encoder reference | 3 | 85.634% | 99.984% | 92.255% | 3493.483 | 3059.433 | +434.050 |
| 1729 | FP16 CUDA encoder | 0 | 78.039% | 95.978% | 86.084% | 2170.125 | 1764.514 | +405.611 |
| 1729 | FP16 CUDA encoder | 1 | 80.984% | 99.507% | 89.295% | 2623.767 | 2194.514 | +429.253 |
| 1729 | FP16 CUDA encoder | 2 | 83.506% | 99.927% | 90.982% | 3061.142 | 2624.514 | +436.628 |
| 1729 | FP16 CUDA encoder | 3 | 85.640% | 99.984% | 92.258% | 3493.250 | 3059.433 | +433.817 |
| 1729 | Mixed dynamic INT8 CPU encoder | 0 | 78.801% | 94.655% | 86.003% | 2119.517 | 1764.514 | +355.003 |
| 1729 | Mixed dynamic INT8 CPU encoder | 1 | 81.683% | 99.146% | 89.572% | 2582.892 | 2194.514 | +388.378 |
| 1729 | Mixed dynamic INT8 CPU encoder | 2 | 84.269% | 99.687% | 91.332% | 3017.675 | 2624.514 | +393.161 |
| 1729 | Mixed dynamic INT8 CPU encoder | 3 | 86.245% | 99.956% | 92.596% | 3458.925 | 3059.433 | +399.492 |
| 20260918 | FP32 encoder reference | 0 | 81.137% | 91.606% | 86.054% | 2695.150 | 2387.151 | +307.999 |
| 20260918 | FP32 encoder reference | 1 | 83.914% | 97.710% | 90.288% | 3370.367 | 3031.151 | +339.216 |
| 20260918 | FP32 encoder reference | 2 | 86.089% | 98.841% | 92.025% | 4022.350 | 3675.151 | +347.199 |
| 20260918 | FP32 encoder reference | 3 | 87.573% | 99.278% | 93.059% | 4685.867 | 4323.263 | +362.604 |
| 20260918 | FP16 CUDA encoder | 0 | 81.120% | 91.595% | 86.040% | 2695.383 | 2387.151 | +308.232 |
| 20260918 | FP16 CUDA encoder | 1 | 83.901% | 97.710% | 90.280% | 3370.600 | 3031.151 | +339.449 |
| 20260918 | FP16 CUDA encoder | 2 | 86.077% | 98.841% | 92.019% | 4022.583 | 3675.151 | +347.432 |
| 20260918 | FP16 CUDA encoder | 3 | 87.563% | 99.278% | 93.053% | 4686.100 | 4323.263 | +362.837 |
| 20260918 | Mixed dynamic INT8 CPU encoder | 0 | 80.028% | 91.976% | 85.587% | 2743.567 | 2387.151 | +356.416 |
| 20260918 | Mixed dynamic INT8 CPU encoder | 1 | 82.770% | 97.851% | 89.681% | 3425.692 | 3031.151 | +394.541 |
| 20260918 | Mixed dynamic INT8 CPU encoder | 2 | 85.314% | 98.961% | 91.632% | 4069.708 | 3675.151 | +394.557 |
| 20260918 | Mixed dynamic INT8 CPU encoder | 3 | 86.928% | 99.404% | 92.748% | 4733.881 | 4323.263 | +410.618 |

## Integrity and artifacts

The original FP32 caches, videos and checkpoints were not overwritten. The 32-frame engineering gate passed FP32 ONNX/PyTorch parity and reproduced historical frame selection within the preregistered float16-cache tolerance. Sequential decoding preserved the historical `round(timestamp × fps)` ordinal sampler, ROI and letterbox; 6,723 comparisons against historical seeking passed byte-for-byte. This changes IO, not frame-selection semantics.

Full INT8 replay used two disjoint CPU workers, each with two ORT threads and batch one, using the exact same graph. All caches and RGB inputs are on the NAS. A separate audit verified every chunk/receipt/cache hash, timestamps, finite tensor dimensions, source associations, and that FP16 parameters are exactly the original weights cast to half. Probability-to-interval decoding and every padding evaluation were independently replayed by the summary script.

Artifacts: `private-reference-0108`. The immutable protocol, source snapshots, CPU/CUDA/browser qualification, INT8 alternatives, extraction receipts, probability arrays, per-seed predictions, full padding/source-group metrics and `summary.json` are retained there.

Key artifact hashes (SHA-256; paths relative to the experiment folder):

| Artifact | SHA-256 |
|---|---|
| `protocol.json` | `ed14d8076943102bc05592cabeeb29066721cb15b37ffb51274dc1b39a7f724c` |
| `audit-inputs.json` | `faa55b5d14d6aa0d3d6084f89867e2c0e5bac5573c8d5b1b0a8ccfb407d0c673` |
| `report.json` | `ca326263a05b51bd1a14332c749d6dd6b1e922052b6d3a801945d5e32873e552` |
| `recall99-v1/protocol.json` | `d49ef76eaee881a1e7a404c46f89cf969163e5722f9bf049850393055ee02d17` |
| `recall99-v1/report.json` | `add8369eccf9b00a030db6e08b1ede4f4ca6b2125eaaf9ae40213e5593c5afeb` |
| `summary.json` | `469481fda8b6eafc50539e073470807e25b310c734e56c8e847993201725bca9` |
| `browser-v1/report.json` | `a9316781342642d2c86dd4676163e722eb7285b25faa59647bb3daca77fa5d83` |

ONNX Runtime documents dynamic quantization and mixed operator support in its [quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). Quantization quality and target-backend parity require measurement; file size alone is insufficient.
