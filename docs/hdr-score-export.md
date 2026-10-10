# Score export with source color preserved

This variant keeps the source video transfer function, primaries and range.
HDR score exports use 10-bit HEVC; SDR exports use H.264. HDR score artwork
starts as sRGB and is converted to the video color space before compositing,
using 203-nit reference white. Android SDR keeps its established electrical
RGB blend, with corrected premultiplied alpha. An unsupported encoder must
report an error rather than silently export SDR.

The reported iPhone error matches the native iOS exporter's SDR-only score
overlay guard. The production web editor rejects iOS in
[`App.tsx`](../prod/src/App.tsx). The native export paths are
[`ExportService.kt`](../android/app/src/main/java/com/volleycut/nativeanalysis/ExportService.kt)
and [`VideoExporter.swift`](../ios/App/VideoExporter.swift).

## Analysis and export are separate

Android analysis selects the HEVC base layer for verified Dolby Vision 8.4
(profile 8, BT.2020 HLG) instead of requiring a Dolby Vision decoder. Other
Dolby Vision profiles are not reinterpreted as HEVC. The asynchronous, synchronous,
specialist and independent neural decoder paths request SDR tone mapping for
HLG/PQ input and verify both decoder support
and the decoded SDR transfer before extracting features from 8-bit YUV frames.
Unsupported tone mapping fails explicitly. The visual extraction version has
changed so earlier HDR features cannot silently satisfy a new analysis.

Export reads the original media again, so analysis conversion does not determine
export color or bit depth. The existing iOS analysis conversion is unchanged.
HDR model accuracy remains a separate qualification task.

Preserving the HLG base image of an iPhone Dolby Vision recording does not by
itself preserve Dolby Vision dynamic metadata. See each platform's README for
its export constraints. iOS currently accepts limited-range video and explicitly
rejects full-range input rather than changing its range. Android also rejects
full-range HDR because its pinned Media3 HDR shader assumes a video-range luma
offset. The original Dolby Vision recording is supplied privately at runtime;
it is never bundled in the APK or committed with the test suite.

## Reproducible fixtures

[`prepare-hdr-export-fixtures.py`](../scripts/prepare-hdr-export-fixtures.py)
creates SDR, HLG and PQ fixtures using source and artifact indexes from the
external private ledger. It requires tagged, limited-range BT.709 input and
never overwrites the source or an existing artifact. Configure
`VOLLEYCUT_PRIVATE_LEDGER` (or its platform override) outside Git.

```sh
python scripts/prepare-hdr-export-fixtures.py \
  --source-index hdr-export-source-01-windows \
  --output-index hdr-export-artifacts-01-windows \
  --profile hlg --start 60 --duration 6 --encoder libx265 --name source-hlg
```

Omit `--start` and `--duration` for the full source. `--profile pq` creates the
PQ counterpart; `--profile sdr` creates the SDR control. A compatible FFmpeg
build can use `--conversion libplacebo --encoder hevc_nvenc` for GPU conversion.
Both HDR recipes map SDR reference white to 203 nits; they do not invent
highlight detail absent from the source. Each output has a receipt containing
its SHA-256, color metadata and source/artifact indexes, without private paths.

The prepared artifact set contains six-second SDR, HLG and PQ samples and a
full-length HLG copy (1920x1080, 60 fps, 66,349 frames, 1,105.816 seconds).
The full copy has HEVC Main10 and limited-range BT.2020 HLG tags; sample frames
at its beginning, middle and end decode successfully. Synthetic white-patch
checks place 203-nit white within two 10-bit code values of the HLG/PQ targets.

## Validation scope

On a physical Pixel 10 Pro, minute-long SDR and original Dolby Vision 8.4
samples passed production-model analysis, including matching fresh shared and
cached independent predictions, synchronous decoding and specialist samples.
Both inputs exported with scores on/off. Scored HDR output was HEVC Main10,
limited-range BT.2020 HLG; the no-score control retained Dolby Vision through
transmuxing. The scored HDR minute exported in about 15 seconds using the
hardware HEVC encoder. SDR output remained 8-bit H.264/Rec.709.

Independent first-frame inspection confirmed output tags and score colors.
HDR white measured 725 in limited-range 10-bit luma (target about 721);
SDR white measured 237 (target 235). Nine score GPU/encoded-pixel/rotation
checks also passed on the phone. The decoder update passed 220 JVM tests.
Full exported files remain on the phone; the private workstation artifacts
contain sampled first-frame data, metadata, hashes and test receipts. These
checks do not establish long-recording performance, model accuracy or HDR
appearance on a calibrated display. Real iOS Dolby Vision remains untested.

Android passed 217 JVM tests and 11 targeted emulator tests, plus a separate
PQ-input smoke check. A later full-range HDR preflight guard passed its three
focused policy tests, including the new HLG/PQ range regression. Coverage
includes encoded SDR score pixels and rotation,
HLG/PQ floating-point GPU pixels, opaque colors, fades, and unchanged pixels
outside the score. A software decoder override is used only by tests because
the emulator's host H.264 decoder rejects these inputs.

The emulator reports no HDR encoder. Its HLG/PQ checks therefore validate
compositing and the unsupported-encoder error, not an end-to-end HDR encode.
See [`android/README.md`](../android/README.md) for test input setup and
[`ios/TESTING.md`](../ios/TESTING.md) for the CPU-only iOS lab checks.

iOS passed 165 portable tests. Direct CoreVideo buffer checks passed for SDR,
HLG and PQ, including score pixels, exact untouched pixels and color attachments.
All six three-frame, 64x48 AVFoundation exports passed with scores on/off across
those formats, verifying output color tags, HDR HEVC 10-bit and decoded pixels.
The HLG fixture writer disables Dolby Vision metadata generation because that
generation failed on the lab; these results cover plain HLG, not Dolby Vision.
The unsigned ARM64 iOS build-for-testing also passed, compiling the app and its
hosted/UI test targets. Hosted/UI tests were not executed. Android ARM64 debug
and unsigned iOS build archives are retained separately for physical-device
validation; the iOS archive still requires development signing before install.

Before release, test both short and long recordings on real Android and iOS
devices: SDR, HLG, PQ and the original iPhone Dolby Vision file; scores on/off;
rotation; partial fades; and scoreboard colors on an HDR display. Inspect
output transfer/primaries/range, HEVC bit depth, audio, cut timing and chapters.
Keep the same source and score state when comparing against the HDR-to-SDR
variant in its separate worktree.
