# Android audio feature mismatch: root cause

**The dominant mismatch is an Android batched-audio timestamp reconstruction bug.**
It interprets a 1,106-sample first timestamp gap as the permanent decoded packet
size even though the AAC packets decode to 1,024 samples. It assigns regrouped PCM
to progressively earlier timestamps and trims overlapping samples. The exact
production assembler reproduces a 78.66-second loss; saved phone features
independently show the predicted drift and frozen tail.

This finding supersedes resampling as the primary explanation in the
[native benchmark](distilled-mobile-large-native-benchmark.md). Diagnosis is
complete. The subsequent [shared app and web repair](android-web-audio-fix.md)
implements codec-based framing and startup percentile correction; that report
tracks validation separately from the original frozen diagnosis below.
Evidence resolves through ledger `private-reference-0223`, recording `recording-044`.

## Fault mechanism

1. The source is 48 kHz stereo AAC. Its first compressed-packet timestamps are
   0, 0.023042, 0.044375, 0.065708 and 0.087042 seconds. The unusual first gap
   rounds to 1,106 samples; subsequent gaps round to 1,024. All five inspected
   decoded frames contain 1,024 samples, including the first.
2. [NativeAudioDecoder](../../android/app/src/main/java/com/volleycut/nativeanalysis/NativeAudioDecoder.java)
   lines 614-617 estimates `framesPerUnit` from the first two input timestamps.
   `setFramesPerUnit` at lines 408-411 latches the first estimate and ignores later
   corrections.
3. `DecodedAccessUnitAssembler.accept` groups 1,106 PCM samples together while
   consuming timestamps normally separated by 1,024 samples. Later waveform
   content is assigned to earlier video time.
4. [AudioFeatureExtractor.push](../../android/app/src/main/java/com/volleycut/nativeanalysis/AudioFeatureExtractor.java)
   lines 70-76 trims the resulting apparent overlaps. This discards actual PCM
   samples and shortens the audio timeline.
5. After the shortened audio ends, `finishAndPool` selects the nearest available
   frame (lines 124-129). Audio remains marked available and the last feature
   values repeat through the remaining video.

## Independent evidence

An offline Java harness extracts the exact production assembler into a minimal
enclosing class. Synthetic PCM carries known source-time values, using the real
packet count and the inspected initial timing irregularity followed by regular
AAC spacing. Its consumer mirrors production gap/overlap arithmetic. This
isolates the assembler mechanism from codecs, resampling and model inference.

| Assembler assumption | Reconstructed audio ends | PCM discarded as overlap | Unconsumed timestamps | Audio content assigned to video 10:00 |
|---|---:|---:|---:|---:|
| Current: 1,106 samples/unit | 16:22.332 | 78.660s | 3,687 | 10:48.047 |
| Diagnostic control: 1,024 samples/unit | 17:40.994 | 0s | 0 | 10:00.000 |

The saved phone inputs independently confirm the signature: all 26 nonconstant
audio columns become constant around 16:22.5 (peak-to-RMS at 16:22.25), then repeat
for approximately 78.5 seconds. Both distilled selections contain the same native
AV ranks within 3.44e-8. Feature names/order and desktop cache/source association
were checked and match.

Without fitting a slope or offset, comparing `native(t)` against
`desktop(t * 1106 / 1024)` produces:

| Feature | Correlation at matching timestamps | After fixed time mapping |
|---|---:|---:|
| RMS loudness | 0.1778 | 0.9457 |
| Noise floor | 0.4741 | 0.9947 |
| Signal-to-noise ratio | 0.0385 | 0.9340 |
| Onset cadence | 0.2747 | 0.9826 |
| Spectral flux | 0.0157 | 0.4907 |

The mapping is diagnostic, not a repair. Discarded waveform samples and their
frequency content cannot be recovered by moving feature timestamps. Features must
be regenerated from correctly placed PCM.

## Secondary differences measured separately

A bounded source excerpt covering approximately 60-120 seconds was decoded to
mono PCM. Exact snapshots of five plain-Java DSP classes, compiled with the
bundled JDK without stubs, were compared with Python using identical samples.
After the first ten seconds, every nonconstant same-input audio feature
correlation is at least 0.9999999999998991. This rules out ordinary FFT/feature math
as the explanation for the large full-video mismatch.

| Controlled resampling comparison on identical 48 kHz PCM | Correlation |
|---|---:|
| RMS: Android linear versus desktop filtered resampling | 0.999997 |
| Spectral flux | 0.999796 |
| 80-250 Hz SNR | 0.9999997 |
| 4,000-7,800 Hz SNR | 0.994199 |

Unfiltered resampling remains a feature-contract difference, especially at high
frequencies. It cannot account for the observed full-video RMS correlation of
0.178 or the growing time drift here. These controlled results concern one
excerpt and do not qualify the resampler for all audio.

A separate startup bug exists in `AudioFeatureExtractor.rollingPercentile`
(lines 326-329): when the percentile position is an integer, both interpolation
weights become zero. This produces false zero noise floors and inflated SNR
during the first ten seconds. The fixed 200-frame window then uses a nonintegral
position, so this defect cannot explain late-video compression.

## Scope, prior tests and repair requirements

The normal Android app and neural benchmark use the shared decoder. Its faulty
branch is selected by `AUTO` when Android/codec batching support is available,
or explicitly by batched mode. The synchronous single-access-unit path avoids
this regrouping estimate. The production ensemble is also exposed to the bug;
its accuracy after repair has not been measured.

Existing assembler tests directly supply a known-correct frame length and never
test inference from an irregular first timestamp gap. Earlier batch-versus-single
qualification used another recording whose timing did not expose this case.
Passing temporal-model probability/decoder replay starts downstream of the wrong
inputs, so it could not detect the defect.

The original repair acceptance requirements were:

1. Stop inferring permanent decoded frame size from one container timestamp gap.
   Preserve validated decoded-unit timing/counts, or use a single-unit preflight
   and synchronous fallback when batching cannot establish them. A blanket 1,024
   constant is not valid for every codec/profile.
2. Check PCM accounting, consumed timestamps and reconstructed duration. Detect
   unexplained overlap/discard instead of silently accepting it.
3. Add regressions for irregular initial timestamps, codec frame sizes, gapless
   delay/padding and analysis windows. Compare batch and single-unit extraction
   before timing the corrected pipeline.
4. Repair startup percentile interpolation and qualify filtered resampling
   separately.
5. Invalidate affected audio/context caches and rerun the same production and
   neural checkpoints on-device, including accuracy and full-pipeline timing.
   This is a shared app fix, not a benchmark-only change.

At diagnosis time, a direct Android MediaExtractor timestamp or decoded-PCM trace was not captured.
The exact code reproduction, source metadata, fixed-ratio alignment and saved-phone
frozen-tail evidence strongly corroborate the mechanism. A corrected device run
was the next acceptance check. The [repair follow-up](android-web-audio-fix.md)
now records byte-identical full-file single/batched audio and frozen-model replay;
a complete fresh video-to-score accuracy/timing run remains separate work.

The earlier [audio-only replacement check](distilled-mobile-large-recall-audio-check.md)
establishes the model impact: substituting desktop audio reduces the highest-recall
choice from two wholly missed rallies to zero, while a partial tail remains. It
does not substitute for the corrected device run.

[Structured evidence](android-audio-timeline-root-cause.json) includes source,
code and input hashes, packet accounting, all feature correlations and controlled
PCM results.
