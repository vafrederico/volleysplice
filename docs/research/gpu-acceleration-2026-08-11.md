# Intel iGPU acceleration evaluation — 2026-08-11

## Outcome

The Intel UHD 630 is functional and useful for opt-in review-proxy transcoding, but it is not a suitable backend for the current learned model or feature extractor.

- Keep model training and inference on NumPy/CPU. The complete classifier fit takes less than one second, and classifier inference is about 0.005% of measured full-video processing time.
- Keep learned feature extraction on the existing OpenCV/CPU path. A VAAPI prototype was 3.55× slower and changed predicted intervals.
- Keep software FFmpeg/libx264 as the canonical proxy backend. VAAPI approximately doubled throughput, but the current hardware outputs failed the fidelity and storage non-regression gates.
- Retain VAAPI as an explicit viewing-proxy option. Its filter now produces exact 30 fps cadence, square pixels, even dimensions, and never upscales a source narrower than 960 pixels.
- Cache validation probabilities during decoder tuning. This preserves the selected decoder and metrics while reducing the real full-corpus grid search from 26.596 to 14.518 seconds (45.4%).

No saved model, feature version, cache identity, or prediction contract changed.

## Test host and data

- KVM guest with a six-core Intel Core i5-9500.
- Intel Coffee Lake GT2 / UHD Graphics 630 (`8086:3e92`) using `i915`.
- Intel render node: `/dev/dri/renderD128`.
- Pinned Jellyfin container with Intel iHD 25.4.6 / VA-API 1.23.
- Full corpus: nine videos, 9,082 seconds, seven 4K60 VP9 sources and two 1080p60 H.264 sources.
- Learned input: 36,332 samples with 210 contextual features; 24,265 training and 7,643 validation samples.

The login user is not in the host `render` group, so direct host VAAPI access fails. The existing Docker backend safely maps the render node and its numeric group without changing host group membership. The container successfully exposed H.264 decode/encode, VP9 decode, HEVC decode/encode, and VAAPI scaling.

The corpus is on NFS, the guest was under memory/swap pressure, and Frigate continuously used the same iGPU. Overlapped runs were discarded, but timings should still be treated as host-specific engineering measurements rather than universal throughput claims.

## Model training and inference

The model is a 210-weight FP32 logistic classifier, not a neural network. It occupies only a few KiB and runs after video decoding, per-frame features, temporal contextualization, and percentile ranking.

Real cached-corpus timings:

| Work | Measured time |
|---|---:|
| Contextualize all nine recordings | 0.174 s |
| Fit model, five runs | 0.631–0.919 s |
| Fit model, median | **0.717 s** |
| Predict all 36,332 samples | **40.84 ms** |
| Recorded cold end-to-end training | 335.521 s |
| Recorded end-to-end inference over all nine videos | 757.545 s |

Model fitting is about 0.21% of the recorded cold training run. Classifier prediction is about 0.0054% of measured full-corpus inference. Even a zero-cost GPU implementation could not materially improve either workflow.

Current Intel tensor-compute support also does not make this a good migration target. Intel's current oneMKL GPU requirements begin at 11th-generation integrated graphics, while this UHD 630 is a legacy Gen9 device. OpenVINO could express the inference graph through a legacy OpenCL runtime, but it would not accelerate the custom Adam training loop and would add compilation, dispatch, transfer, driver, and precision complexity to a millisecond-scale operation. See [Intel oneMKL system requirements](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-math-kernel-library-system-requirements-2025.html) and [Intel's legacy GPU guidance](https://dgpu-docs.intel.com/overview/supported-hardware/legacy-gpus.html).

## Learned feature extraction

The existing extractor was compared with a prototype that used VAAPI decode in the Jellyfin container, selected the same 360 sample timestamps from a real 90-second H.264 proxy, streamed BGR frames back to Python, and retained the existing CPU ROI, resize, optical-flow, feature, and model code.

| Path | Wall time |
|---|---:|
| Existing OpenCV CPU path | **3.219 s** |
| VAAPI decode plus raw-frame pipe | **11.442 s** |

The VAAPI path was 3.55× slower. Timestamps were identical, but decoded pixels differed by 1.57 levels mean absolute error. Those differences propagated through percentile ranks:

- contextual-feature mean absolute difference: 0.01317;
- probability mean absolute difference: 0.01251, maximum 0.08553;
- CPU result: five intervals;
- VAAPI result: four intervals, with two other boundaries shifted.

This fails both performance and prediction-parity gates. Hardware decoding must not be placed behind a transparent switch for existing models. A future attempt would require a new feature version, fresh caches, retraining, and held-out evaluation.

## Review-proxy transcoding

The current software and Docker/VAAPI commands were compared on matched 60-second excerpts. Clean repetitions were used for timing; one overlapping sample and one severe VP9 outlier were discarded.

| Source | Software median | VAAPI median | Speedup | Software max RSS | VAAPI max RSS |
|---|---:|---:|---:|---:|---:|
| 1080p60 H.264 | 14.23 s | 6.93 s | **2.05×** | ~250 MiB | 97–104 MiB |
| 4K60 VP9 | 38.64 s | 20.46 s | **1.89×** | 547–572 MiB | 65–70 MiB |

Both outputs were decodable H.264 High, 960×540, yuv420p, BT.709, SAR 1:1, and exactly 1,800 frames over 60 seconds. The current VAAPI quality/storage result, however, was not equivalent:

| Source | Software SSIM / PSNR | VAAPI SSIM / PSNR | Software size | VAAPI size |
|---|---|---|---:|---:|
| H.264 | 0.9724 / 38.39 dB | 0.9388 / 30.15 dB | 5.37 MB | 13.24 MB |
| VP9 | 0.9711 / 39.09 dB | 0.9298 / 29.32 dB | 4.71 MB | 7.66 MB |

Metrics used the same 30 fps Lanczos-scaled source reference. H.264 ICQ trials improved fidelity only by spending substantially more bytes; they did not match the software proxy's quality/size tradeoff. A hybrid hardware-decode/encode plus CPU Lanczos scale recovered quality but lost the throughput advantage.

The corrected all-hardware filter is:

```text
fps=30,scale_vaapi=w='trunc(min(960,iw)/2)*2':h=-2:format=nv12,setsar=1
```

It improved the tested H.264 SSIM to 0.9502, fixed cadence and the video start timestamp, and prevents upscaling. It remains an explicit review-only backend because it still does not match canonical software fidelity.

Enable it for a disposable review-proxy analysis with:

```bash
VOLLEYCUT_PROXY_BACKEND=jellyfin-vaapi \
  npm run analyze-no-model -- /absolute/path/to/recording.mkv
```

Do not use this switch to regenerate annotation masters, learned-model inputs, or the frozen training corpus.

## Implementation instructions

The benchmark branch tested the following implementation edits, but they are deliberately not part of this decision-only commit. Apply them separately against the current `main`, review them in the context of newer pipeline work, and commit them as an implementation change only if still desired.

### Decoder-search optimization

In `analysis/pipeline.py`:

1. Add an optional keyword-only `probabilities: Sequence[np.ndarray] | None = None` argument to `_evaluate_prepared()`.
2. Validate that a supplied probability sequence has one entry per prepared recording and that each entry has shape `(len(item.sequence.times),)`; raise `ModelError` on a mismatch.
3. Use supplied probabilities when present and retain `model.predict(item.contextual_values)` as the default behavior for every other caller.
4. At the beginning of `_tune_decoder()`, compute one tuple of validation probabilities with `model.predict()` and pass that same tuple to `_evaluate_prepared()` for all 1,584 decoder candidates.
5. Extend `analysis/tests/test_pipeline.py` to prove that decoder search reuses the same cached tuple for all candidates and still selects the expected decoder.

This is prediction-preserving because decoder parameters never affect classifier scores. On the measured full-corpus validation search it selected the same decoder and returned byte-for-byte-equivalent metric values while reducing wall time from 26.596 to 14.518 seconds.

### Opt-in VAAPI review proxy

In `analysis/ffmpeg.py`:

1. Keep `software` as the default `VOLLEYCUT_PROXY_BACKEND` and retain the pinned Jellyfin-container device/group mapping.
2. Replace the fixed `scale_vaapi=w=960:h=-2:format=nv12` plus output `-r 30` combination with this filter and filter-driven CFR output:

   ```text
   fps=30,scale_vaapi=w='trunc(min(960,iw)/2)*2':h=-2:format=nv12,setsar=1
   ```

3. Keep `-fps_mode cfr`; remove output `-r 30` so frame cadence is established once in the filter graph.
4. Prefer a separately testable command-builder function so unit tests can inspect Docker, device-group, filter, codec, and output arguments without invoking FFmpeg.
5. Extend `analysis/tests/test_helpers.py` to assert filter-driven CFR, bounded/even scaling, SAR normalization, and absence of output `-r`.
6. Before release, smoke-test real H.264 and VP9 inputs, verify decode success, dimensions, yuv420p, 30 fps, audio presence, and confirm that a source narrower than 960 pixels is not upscaled.

Do not extend this backend to `analysis/media.py` normalization or `analysis/features.py` extraction without a new quality and held-out prediction evaluation. VAAPI `QP 24` is not equivalent to x264 `CRF 24`.

### Operator documentation

In `.env.example` and `README.md`, label `jellyfin-vaapi` as a disposable viewing-proxy backend only. State that software remains canonical for annotation masters and learned-model inputs, and link to this decision record.

## Revisit criteria

Re-evaluate model GPU execution when VolleyCut gains a materially larger visual backbone. Train that network on a supported accelerator, export it, and compare CPU versus OpenVINO GPU inference using fixed FP32/FP16 parity and held-out interval-metric gates. Revisit media defaults only when a hardware encoder configuration matches software within 0.005 SSIM and 0.5 dB PSNR without a material storage increase.
