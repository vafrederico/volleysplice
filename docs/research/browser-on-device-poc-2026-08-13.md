# Browser-only on-device POC — 2026-08-13

## Outcome

`model-9c92b8e9333f` is feasible as a desktop-browser, on-device workflow. Its inference workload is three small logistic heads over 520 contextual inputs, not a neural network that needs a GPU. CPU inference is effectively negligible; decode, optical flow, phase correlation, audio transforms, and final video encoding dominate runtime.

The implemented `/on-device` POC keeps the selected media in the browser, derives features locally, runs the rally/serve/dead-state stack locally, provides interval review and JSON EDL export, and reopens the original master for a direct-to-disk MP4 export. No upload endpoint is involved. `/on-device-ui` loads a small checked-in prediction cache captured from an actual browser on-device run, so review and timeline work does not repeatedly decode the fixture video.

This proves exact parity from the canonical cached 104-wide features through final intervals. Two initial HTTPS browser runs over the canonical 240 MB proxy produced 39 and then 40 candidates versus the canonical 37: the first exposed ignored AAC priming timestamps, while the second isolated the remaining unfiltered-resampler drift. A third run using the pinned FFmpeg 7.1.5 `libswresample` WASM experiment produced **37 candidates** and an all-nine total of **396**, matching offline counts while retaining different boundaries. Browser canvas color/resize still differs from FFmpeg/OpenCV. The production decision therefore remains “browser-feasible, materially closer with SWR, exact media parity still gated.”

## What was built

The browser path is:

1. `File` is opened through Mediabunny's `BlobSource`, with an 8 MiB read cache. The file is range-read lazily and is not copied to application storage.
2. `CanvasSink` decodes selected video frames and applies the normalized camera ROI before producing transient 192×108 canvases at a requested 4 Hz cadence.
3. OpenCV.js computes the static, difference, camera-quality, Farneback-flow, and residual-motion channels on CPU. OpenCV.js does not expose `phaseCorrelate`, so the POC includes a DFT/cross-power port.
4. `AudioSampleSink` decodes audio locally. The default POC downmixes and linearly resamples to 16 kHz. The explicit `libswresample-wasm-v1` experiment instead streams stereo planar Float32 through FFmpeg 7.1.5 SWR for rematrixing, filtered conversion, S16 quantization, and end flush. Both then compute the same 50 ms FFT/band feature schema.
5. The 104 base channels are whole-recording percentile-ranked, except for the six absolute channels, and expanded at -2, -1, 0, +1, and +2 seconds into the exact 520-column model signature.
6. Rally, serve, and dead-state logistic heads run with FP32 arrays on the CPU. The original decoder, serve composition, and dead-state refinement rules then produce reviewable intervals.
7. Export uses the selected raw master, rejects a duration-mismatched alternate master, decodes only kept ranges, and writes an AVC/AAC MP4 to a `FileSystemWritableFileStream` through Mediabunny's chunked `StreamTarget`.

The review timeline has two tracks: raw model predictions and the actual padded export sections. Padding is clamped to the source, sorted, and merged when sections overlap or touch, so duplicated video is neither displayed nor exported.

The pinned browser dependencies are Mediabunny 1.53.1, OpenCV.js 4.12.0-release.1, and fft.js 4.0.4. The checked-in model JSON is about 100 KB. OpenCV.js is a generated, ignored browser asset of about 10.9 MB, copied from the pinned package by `npm run browser-assets` before development and production builds.

The checked-in `model-9c92b8e9333f` bundle comes from the beach-excluded retraining workspace, not the same-named model directories in the original workspace. It serializes `full-audiovisual-audio-normalized-v3` as the rally head (`ca004bff50fb…`), `serve-specialist-audio-normalized-v5` as the serve head (`5ff951b60ee8…`), and `dead-state-transition-audio-normalized-v5-no-legacy-final` as the dead-state head (`9c92b8e9333f…`), along with the decoder, composition, and refinement settings used by that stack. The frozen Y9 feature rows come from the canonical feature cache in the original labeling workspace, while the 37 expected intervals come from the corresponding beach-excluded analysis. That split is intentional: it checks the deployed no-beach bundle against a stable real feature sequence.

## Proxy versus raw source

The canonical indoor fixture has two relevant sources:

| Source | Container/codecs | Dimensions/rate | Size | Role |
|---|---|---:|---:|---|
| Training proxy | MP4, H.264/AAC | 960×540, 30 fps | 240 MB | Feature-parity target |
| Original master | Matroska, VP9/Opus | 3840×2160, 60 fps | 1.83 GB | Final export source |

For the safest workflow, analyze the existing FFmpeg training-style proxy, then select the timeline-identical raw master for export. This retains the model's trained feature distribution while returning to the 4K source for the deliverable.

The POC implements analysis of the raw master without materializing a proxy: the compressed 4K stream is decoded and immediately reduced to transient 192×108 feature canvases. A 33-second, 51.2 MB VP9/Opus segment copied from the actual 4K60 master opened, decoded, and produced one on-device candidate; the full 1.83 GB master has not yet completed the same browser endurance test. Direct raw remains a new feature distribution. The raw and proxy decoders can select slightly different frames, and the raw path has different compression, scale, color, and audio-resampling behavior. It should be treated as an A/B candidate rather than parity mode.

Analysis and export therefore remain deliberately separate. Feature frames never become output frames. Exact arbitrary cut boundaries require decoding and re-encoding the selected original frames; the POC preserves source display dimensions and uses a very-high-quality single encode, but it does not claim byte-identical stream copy. A later “smart copy” mode could copy keyframe-aligned intervals, at the cost of inexact boundaries.

## Court and ROI clarification

There is no court polygon, court-line detector, or drawn-court label in this model path. The POC does not draw one.

The labeling manifests do, however, provide one coarse normalized rectangular ROI for each of the nine canonical recordings, and the browser profile lookup recognizes all nine stable proxy IDs. For the canonical indoor source it is `(x=.04, y=.14, width=.92, height=.84)`. The grass sources use similarly broad rectangles; both beach proxies use `(x=.02, y=.12, width=.96, height=.86)`. Those rectangles exclude irrelevant frame edges; they do not identify court geometry. Because the checked-in bundle was retrained without beach footage, beach predictions are out-of-domain comparison results even though their extraction geometry is known.

The UI calls this a “camera crop,” shows an adjustable rectangle, and allows full-frame fallback. Full-frame inference works technically but is a distribution shift from the trained non-beach recordings.

## Validation completed

- The exported browser bundle has the exact 520-name signature used by all three heads.
- A frozen real-match fixture contains all 3,474 rows of the canonical 104-wide cached feature matrix.
- Browser-side percentile ranking, contextualization, FP32 logits, rally decode, serve composition, and dead-state refinement reproduce all **37 expected intervals exactly**.
- Eight additional model/decoder tests cover validation, clipping, hysteresis, bridging, short events, peak/NMS behavior, composition, and refinement.
- The custom OpenCV.js phase-correlation port was compared with Python OpenCV on a synthetic `(3, -2)` shift and matched to sub-micro-pixel precision with effectively identical response.
- In the shared desktop browser, pinned OpenCV.js Farneback flow over 192×108 synthetic frame pairs averaged about 2.2 ms per pair across 50 warm iterations. This isolates the dominant feature primitive from media decode and is supportive, not an end-to-end throughput measurement.
- TypeScript, ESLint, 74 web tests, all 466 Python tests, and the optimized Next.js build pass. `/on-device`, `/on-device-ui`, and the gated `/on-device-batch` runner are statically prerendered.
- The supplied HTTPS reverse proxy exposes a secure context. In the shared desktop Chromium instance, `VideoDecoder`, `AudioDecoder`, `VideoEncoder`, `AudioEncoder`, `showSaveFilePicker`, and HMR are available. The live UI reports all four capability badges as ready.
- Two full 868.5-second, 240 MB canonical proxy runs completed 3,474 browser feature frames and all three model heads without upload. The pre-fix run returned 39 candidates, the timestamp-corrected run returned 40, and the canonical cached-feature path returns 37.
- Browser/canonical timestamps match bit-for-bit on all 3,474 rows. Visual parity is already high (`diff_mean` correlation 0.99998, `flow_mean` 0.99990, phase response 0.99959). Audio was the dominant drift: for example RMS correlation was 0.675 at zero lag and 0.874 at the next pooled row.
- The canonical AAC stream begins at -14.333 ms with 688 skip samples. The browser previously concatenated that priming packet at logical zero; the accumulator now uses decoded timestamps, trims negative priming, zero-fills gaps, and removes overlaps before resampling. Focused tests cover each case. The 40-result rerun shows that this necessary correction is not sufficient without filtered resampling.
- The 228 KiB `libswresample` WASM module produces the same deterministic output length as native FFmpeg, is invariant to chunk boundaries, and differs from the native SIMD output in only 4/16,046 probe samples, each by one S16 LSB. Its real Y9 run returned the canonical count of 37.
- `/on-device-ui` stores the latest 40 browser predictions at their full returned precision. At the 3 s / 2 s defaults they render as 38 merged export sections; changing both controls to 8 seconds immediately collapses them to 12 sections.
- The exact export path was round-tripped through browser origin-private storage without opening a native save dialog. A 2.5-second proxy interval produced a 2.517-second 960×540 AVC/AAC MP4 (1.44 MB), which the product's own media probe reopened as “Decode Ready.”
- The raw-master path was validated on the copied 4K60 VP9/Opus segment. Its real on-device prediction (`8.375–19.859`) exported as an 11.499-second 3840×2160 AVC/AAC MP4 (71.2 MB) in about 50 seconds, and the product probe reopened it as “Decode Ready.” This proves a short 4K round trip, not full-match endurance or subjective quality.

## Browser batch comparison completed

The HTTPS desktop runner at `/on-device-batch` completed all nine canonical training-style proxies sequentially in the shared Chromium browser. It range-read and decoded approximately 2.74 GB / 2 h 31 min of source media, ran the unchanged browser feature and inference path, and exported no video. The run saved **410 exact, unpadded prediction ranges**:

| Recording | Environment | Ranges | Core predicted time |
|---|---|---:|---:|
| `indoor-source-07` | indoor | 40 | 5:50 |
| `beach-source-02` | beach, OOD | 25 | 14:18 |
| `beach-source-01` | beach, OOD | 34 | 8:54 |
| `grass-source-09` | grass | 46 | 5:57 |
| `grass-source-01` | grass | 60 | 6:40 |
| `grass-source-10` | grass | 52 | 5:33 |
| `grass-source-04` | grass | 55 | 6:05 |
| `indoor-source-01` | indoor | 42 | 7:18 |
| `indoor-source-05` | indoor | 56 | 6:58 |
| **Total** |  | **410** | **67:34** |

The range count is descriptive, not an accuracy score. In particular, the two beach runs are qualitative out-of-distribution probes because beach footage was excluded from training; the unusually long first-beach result should not be read as model validation.

Each result is an append-only ordinary analysis document under `/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/analyses/model-browser-on-device-9c92b8e9333f--<recording-id>/analysis.json`, with the existing per-recording `court-preview.jpg` copied alongside it. The artifact ID and `Browser on-device · model-9c92b8e9333f` variant label keep these timelines distinct in the comparison UI, while `modelVersion: dead-state-transition-audio-normalized-v5-no-legacy-final` preserves the underlying offline lineage and `analysis.method: browser-on-device-webcodecs-opencv-wasm-v1` identifies the execution path. Each document binds the source metadata and manifest ROI, exact intervals, bundle and component hashes, secure-context/browser provenance, and explicit parity/resampler warnings. Beach documents add an OOD warning.

The final audit found all nine exact recording IDs, 410 sorted/non-overlapping/in-bounds ranges, nine matching court previews, no schema or UI-parser errors, and no partial/staging directories. All nine result URLs returned successfully in the main comparison UI under the “Without beach” corpus. The Y9 URL-sourced run reproduced every start/end boundary in the checked-in 40-range browser cache exactly; repeat-decode confidence differences were small (mean absolute difference `0.000297`, maximum `0.004607`). This is browser-to-browser reproducibility, not parity with the 37-range canonical FFmpeg/OpenCV result.

The [offline-versus-browser padding comparison](./browser-on-device-padding-comparison-2026-08-13.md)
evaluates those nine results at 2 and 3 seconds on each side. On the retrospective held-out
recording, browser adjusted F1 is 86.84% versus 88.15% offline at 2 seconds and 88.30% versus
89.33% at 3 seconds; the browser fully contains one fewer of the 39 expected rallies at both
settings.

The follow-up [libswresample-WASM evaluation](./browser-libswresample-wasm-evaluation-2026-08-13.md)
completed another range-only all-nine run under a distinct append-only artifact prefix. It
produced 396 ranges. On validation, SWR tied linear containment at 68/76 with 2 seconds and
72/76 with 3 seconds while improving `F1_padP_coreR` to 77.43% and 80.29%. Across the seven
in-domain recordings it recovered one and two fully contained rallies, respectively, and
raised adjusted F1 to 87.70% and 89.09%.

The batch route is deliberately operational tooling rather than a public upload service. It is disabled unless a high-entropy bearer token and an existing output root are supplied through environment variables. The token remains in the browser URL fragment, requests are same-origin, and a dedicated bearer-protected media route range-serves only the fixed nine proxies. Videos remain server-local, submissions are size/schema/model/source constrained, and persistence uses an atomic directory rename with append-only collision handling. Only ranges, fixed media metadata, and concise provenance cross back to the server.

Before another token-enabled run, the fronting proxy must redact or drop the `Authorization` request header in every access-log path, including rejected 4xx requests. The current development Traefik configuration keeps request headers by default and is not safe for reusable bearer credentials. The completed run used a temporary token and its token-enabled server was stopped afterward; that credential must not be reused.

## Remaining parity and product gates

The secure-context proxy run, base-channel comparison, and filtered-resampler experiment are complete. The remaining gates are:

1. Capture the full WebCodecs+SWR PCM/features and repeat the per-channel/probability comparison; equal interval counts alone can hide threshold coincidences.
2. Decide whether to keep the offline-trained model with explicit browser tolerances or retrain/calibrate on the exact browser-native feature implementation. `OfflineAudioContext` is not a cross-browser parity guarantee.
3. Analyze the 4K raw master and compare its intervals with proxy mode. This measures the direct-raw distribution shift.
4. Extend the successful single-interval 4K round trip to several disjoint intervals and the full master; verify A/V synchronization, first/last-frame boundaries, absence of frame overlap, endurance, and visual quality.
5. Replace duration-only proxy-to-master matching with a content/timeline fingerprint or an explicit sync confirmation; unrelated footage of similar duration currently passes the guard.
6. Harden failed-export cleanup so disk-full/encoder failures abort rather than leave a partial committed MP4.
7. Measure wall time, peak memory, power use, and hardware-codec failures on representative desktop machines.

The known feature differences are:

- Mediabunny's canvas renderer performs browser scaling/color conversion rather than OpenCV `INTER_AREA` on the decoded 960×540 BGR frame.
- The default linear runtime remains unfiltered. The experimental SWR runtime removes that difference but still begins with browser-decoded PCM, which may differ from FFmpeg's decoder output.
- WebCodecs uses real presentation timestamps; the reference OpenCV reader derives timestamps from constant frame index/FPS.
- OpenCV.js is single-threaded in the pinned package. This is adequate for the POC's 192×108, 4 Hz workload but leaves performance on the table.

## Capacity, privacy, and deployment

The hosted service serves static JavaScript, the roughly 100 KB model, and the generated OpenCV asset. Video decode, features, model execution, review state, and export occur in the browser. This removes server inference capacity, upload bandwidth, media retention, and server-side export queues from the critical path.

It does not eliminate all privacy or capacity work. The host still needs an explicit no-upload contract, a restrictive content-security policy, dependency review, and telemetry that never captures filenames or media-derived data. Compute capacity moves to the user's device, so thermal throttling, tab lifetime, codec support, free disk space, and browser memory become product constraints.

For an approximately 14.5-minute match, the canonical feature sequence is only 3,474 rows. The base and contextual FP32 matrices are roughly 1.4 MB and 7.2 MB respectively. Mediabunny's source cache is capped at 8 MiB, and analysis canvases are pooled. The large transient costs are the browser's 4K decoder surfaces and encoder queues, not the model.

Required deployment properties:

- HTTPS, because WebCodecs and direct-to-disk access are secure-context capabilities;
- desktop Chrome/Edge as the first supported target;
- correct static serving of `/on-device/model-9c92b8e9333f.json` and the generated `/on-device/opencv.js`;
- no service worker or analytics behavior that reads selected `File` objects;
- a visible codec capability probe before analysis/export.

## Native fallback

A native app is not required to make the desktop workflow feasible. It becomes attractive if browser codec coverage, background execution, thermal control, or very long 4K exports fail the product gates.

Android can retain the same small CPU model/decoder code while using platform media decode/encode and a streaming container layer. iOS can do the same with AVFoundation/VideoToolbox and Accelerate/vDSP for the signal processing. Native implementations would still need feature-parity validation; they do not automatically match the FFmpeg/OpenCV training distribution. The lowest-risk sequence is desktop HTTPS first, then Android, then iOS if the measured browser limitations justify the additional platform code.

## Reproduction

```bash
npm install
npm run test:web
npm run build
npm run dev -- --hostname 0.0.0.0 -p 3001
```

Open `${VOLLEYCUT_APP_BASE_URL}/on-device` for media operations, or `/on-device-ui` for the no-decode UI fixture. Additional development hosts can be supplied with the comma-separated `VOLLEYCUT_DEV_ORIGINS` environment variable. Plain HTTP is useful only for inspecting UI and static assets.

The corpus batch writer is off by default. For an explicitly authorized local run, start the server with temporary credentials and the existing no-beach analysis root:

```bash
export VOLLEYCUT_ON_DEVICE_BATCH_TOKEN='<temporary-high-entropy-token>'
export VOLLEYCUT_ON_DEVICE_BATCH_OUTPUT_ROOT='/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/analyses'
npm run dev -- --hostname 0.0.0.0 -p 3001
```

Then open `${VOLLEYCUT_APP_BASE_URL}/on-device-batch#token=<temporary-high-entropy-token>`. Completed artifact directories are skipped on resume and are never overwritten. Unset the two variables or stop the temporary server after the run. Results appear automatically in the main comparison UI; for example, an anonymous browser run is selected by `/?video=indoor-source-07&analysis=model-browser-on-device-9c92b8e9333f--indoor-source-07&corpus=without-beach`.

Do not start this mode until the reverse proxy's access-log header policy explicitly redacts or drops `Authorization`; a default “keep” policy can record the bearer on rejected requests.

Regenerate the model and frozen golden fixture with:

```bash
python3 scripts/export-browser-model.py \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/models \
  public/on-device/model-9c92b8e9333f.json

python3 scripts/export-browser-golden-fixture.py \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3/indoor-source-07-be505db9d7354ce6d2cc.npz \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/analyses/model-9c92b8e9333f--indoor-source-07/analysis.json \
  tests/fixtures
```

The model export must use the beach-excluded models root shown above to preserve the documented three-head lineage. The fixture export intentionally combines the original canonical Y9 feature cache with the beach-excluded stack's Y9 analysis, and should be rerun only when one of those source artifacts changes.
