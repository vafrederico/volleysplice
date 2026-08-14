# Browser libswresample-WASM evaluation — 2026-08-13

## Result

The experiment is complete. All nine training-proxy videos were decoded and inferred
inside the HTTPS browser, ranges only; no video was uploaded or exported. The new
append-only artifacts use:

```text
model-browser-on-device-libswresample-wasm-9c92b8e9333f--<recording-id>
```

The WASM browser run emitted 396 ranges, versus 410 from the previous linear-resampler
browser run and 396 offline. Equal total counts do not imply equal boundaries, but the
resampler removed the three extra Y9 ranges and restored Y9's canonical count of 37.

The predeclared target is symmetric 3-second padding because complete-rally retention
is the product objective; 2 seconds is the secondary sensitivity case. Selection uses
the two validation recordings only. On that scope, SWR did **not** increase the number
of fully contained rallies: it tied linear at 68/76 with 2 seconds and 72/76 with 3
seconds. It did improve `F1_padP_coreR` by 0.82 and 1.09 percentage points, respectively,
by removing excess export time while preserving the same core recall.

Across all seven in-domain recordings, SWR recovered one additional fully contained
rally at 2 seconds and two at 3 seconds versus linear. It remained four and three short
of offline containment, respectively. The protected one-video result also recovered
one rally at both paddings, matching offline, but this is a retrospective guardrail and
was not used to choose the runtime.

## Controlled resampler parity

The module is FFmpeg 7.1.5 `libswresample` 5.3.100 compiled as a separate,
single-threaded Emscripten module. It accepts stereo planar Float32 PCM from WebCodecs
and performs rematrixing, filtered 48 kHz to 16 kHz conversion, and S16 quantization in
one streaming context, including a real end-of-stream flush.

The deterministic native-versus-WASM probe produced the same 16,046 output samples and
was invariant to adversarial chunk boundaries. 16,042 samples were bit-identical; the
other four differed by one S16 least-significant bit. The module is 228 KiB plus 13 KiB
of loader glue. Its WASM SHA-256 is
`c7ed95ed8b6f5e11ea9e86262214b978bd145a6ec1f648dd56616af298449f90`.

## Primary validation sensitivity

Event precision/recall/F1 are chronological one-to-one matches at IoU >= 0.5 after
padding and merging. Adjusted F1 is `F1_padP_coreR`, the harmonic mean of equally padded
human precision (`P_pad`) and unpadded core-human recall (`R_core`). “Full” is the count
of unpadded human rallies wholly contained in one merged export interval. Durations are
ignored-aware pooled export seconds; delta is model minus equally padded human export.

| Runtime | Pad | Event P | Event R | Event F1 | P_pad | R_core | Adj F1 | Full | Model s | Human s | Delta s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Offline | 0 s | 50.96% | 69.74% | 58.89% | 54.48% | 91.84% | 68.39% | 44/76 | 758.8 | 450.1 | +308.7 |
| Offline | 1 s | 41.18% | 55.26% | 47.19% | 58.58% | 96.68% | 72.95% | 66/76 | 965.5 | 602.1 | +363.4 |
| Offline | 2 s | 29.67% | 35.53% | 32.34% | 62.04% | 98.38% | 76.09% | 72/76 | 1,159.9 | 754.1 | +405.8 |
| Offline | 3 s | 18.52% | 19.74% | 19.11% | 65.69% | 98.95% | 78.96% | 74/76 | 1,329.6 | 906.1 | +423.5 |
| Linear browser | 0 s | 46.73% | 65.79% | 54.64% | 57.28% | 88.80% | 69.64% | 36/76 | 697.9 | 450.1 | +247.7 |
| Linear browser | 1 s | 39.42% | 53.95% | 45.56% | 60.66% | 95.00% | 74.04% | 63/76 | 909.4 | 602.1 | +307.3 |
| Linear browser | 2 s | 33.33% | 42.11% | 37.21% | 63.30% | 96.99% | 76.60% | 68/76 | 1,110.8 | 754.1 | +356.7 |
| Linear browser | 3 s | 20.93% | 23.68% | 22.22% | 66.37% | 98.19% | 79.20% | 72/76 | 1,294.8 | 906.1 | +388.7 |
| SWR browser | 0 s | 48.51% | 64.47% | 55.37% | 57.37% | 89.26% | 69.84% | 36/76 | 700.4 | 450.1 | +250.2 |
| SWR browser | 1 s | 40.00% | 52.63% | 45.45% | 61.23% | 95.04% | 74.47% | 61/76 | 901.5 | 602.1 | +299.4 |
| SWR browser | 2 s | 32.26% | 39.47% | 35.50% | 64.44% | 96.99% | 77.43% | 68/76 | 1,094.8 | 754.1 | +340.6 |
| SWR browser | 3 s | 22.50% | 23.68% | 23.08% | 67.91% | 98.19% | 80.29% | 72/76 | 1,269.6 | 906.1 | +363.5 |

At the 3-second target, SWR ranks first on validation adjusted F1 and exports 25.2
fewer seconds than linear, but offline still fully contains two more rallies. At 2
seconds, SWR exports 16.0 fewer seconds than linear with the same 68/76 containment;
its event F1 is 1.71 points lower, so the gain is specific to the product ranking metric.

## Requested 2-second and 3-second comparison

| Scope | Runtime | Pad | Event P | Event R | Event F1 | P_pad | R_core | Adj F1 | Full / expected |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| In-domain 7, descriptive | Offline | 2 s | 49.51% | 55.84% | 52.49% | 77.84% | 99.28% | 87.26% | 262/274 |
| In-domain 7, descriptive | Linear | 2 s | 49.37% | 56.93% | 52.88% | 77.16% | 98.72% | 86.62% | 257/274 |
| In-domain 7, descriptive | SWR | 2 s | 50.81% | 57.30% | 53.86% | 78.75% | 98.96% | 87.70% | 258/274 |
| In-domain 7, descriptive | Offline | 3 s | 33.68% | 35.40% | 34.52% | 79.90% | 99.69% | 88.70% | 271/274 |
| In-domain 7, descriptive | Linear | 3 s | 33.67% | 36.50% | 35.03% | 78.98% | 99.29% | 87.98% | 266/274 |
| In-domain 7, descriptive | SWR | 3 s | 34.84% | 36.50% | 35.65% | 80.69% | 99.45% | 89.09% | 268/274 |
| Protected test 1, retrospective | Offline | 2 s | 52.17% | 61.54% | 56.47% | 78.99% | 99.71% | 88.15% | 37/39 |
| Protected test 1, retrospective | Linear | 2 s | 51.06% | 61.54% | 55.81% | 76.98% | 99.61% | 86.84% | 36/39 |
| Protected test 1, retrospective | SWR | 2 s | 54.35% | 64.10% | 58.82% | 79.19% | 99.62% | 88.24% | 37/39 |
| Protected test 1, retrospective | Offline | 3 s | 38.10% | 41.03% | 39.51% | 80.72% | 100.00% | 89.33% | 39/39 |
| Protected test 1, retrospective | Linear | 3 s | 33.33% | 33.33% | 33.33% | 79.07% | 99.97% | 88.30% | 38/39 |
| Protected test 1, retrospective | SWR | 3 s | 35.71% | 38.46% | 37.04% | 80.69% | 100.00% | 89.31% | 39/39 |
| All 9, beach OOD included | Offline | 2 s | 45.80% | 45.80% | 45.80% | 69.50% | 99.18% | 81.73% | 331/345 |
| All 9, beach OOD included | Linear | 2 s | 45.92% | 47.25% | 46.57% | 68.99% | 98.83% | 81.26% | 325/345 |
| All 9, beach OOD included | SWR | 2 s | 46.69% | 46.96% | 46.82% | 70.15% | 98.93% | 82.09% | 328/345 |
| All 9, beach OOD included | Offline | 3 s | 31.33% | 28.70% | 29.95% | 73.44% | 99.51% | 84.51% | 341/345 |
| All 9, beach OOD included | Linear | 3 s | 31.68% | 29.57% | 30.58% | 72.77% | 99.31% | 84.00% | 336/345 |
| All 9, beach OOD included | SWR | 3 s | 32.28% | 29.57% | 30.86% | 74.01% | 99.33% | 84.82% | 338/345 |

The direct containment lift versus linear is therefore:

- validation: +0 at 2 seconds, +0 at 3 seconds;
- in-domain seven: +1 at 2 seconds, +2 at 3 seconds;
- protected test, retrospective: +1 at both paddings; and
- all nine, descriptive with beach OOD: +3 at 2 seconds, +2 at 3 seconds.

## Decision and limits

Promote `libswresample-wasm-v1` over the linear resampler for continued browser-model
experimentation. It is small, streaming, locally private, materially closer to the
offline feature path, and wins the declared validation ranking metric at 3 seconds.
Do not call it exact offline parity: WebCodecs audio decoding and browser canvas
resize/color behavior still differ, and validation containment itself did not improve.
The current offline-trained model should retain an experimental browser-runtime label
until those remaining feature-path differences are either removed or the model is
retrained/calibrated on browser-native features.

## Reproduction and artifacts

```bash
scripts/build-libswresample-wasm.sh
npm run validate:libswresample-wasm
PYTHONPATH=. python3 scripts/evaluate-browser-parity-padding.py \
  --include-libswresample-wasm \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/reports/browser-libswresample-wasm-padding-comparison-v1.json
```

The evaluator refuses to overwrite a report and validates model/component hashes,
source-media hashes, browser runtime provenance, interval ordering, current ignored
ranges, and the checked-in WASM/loader digests. The machine report SHA-256 is
`9db86795f84a2589d754a37717a09fcbf87f8311e18bebe3a5eff50425397702`.
