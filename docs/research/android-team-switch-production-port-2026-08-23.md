# Android team-switch production port - 2026-08-23

## Outcome

The Android app can now run the frozen production team-switch pipeline after rally
analysis and seed its selected events into score tracking as editable, model-origin
team-side switch markers. Automatic switch generation is opt-in per project and is
off by default. Serving-side and team-switch sampling share one gap-aware video decode
schedule, so enabling the second specialist does not add a second full-video traversal.

The shipped Android runtime has the same canonical JSON content as the browser
runtime (working-tree line endings may differ):

| Property | Value |
| --- | --- |
| Model | `side-switch-hard-negative-mining-v1/union34-top2-x2` |
| Feature version | `SIDE-SWITCH-UNION34-V1` |
| Inputs | 22 visual + 10 production-state + 2 candidate metadata |
| Threshold | `0.39884973953581804` |
| Canonical runtime SHA-256 | `ab4197545fb916a37ee6ac1d69e74ddfc0123c09039cdfa88c4ef378dd3e27fc` |
| Model fingerprint | `sha256:c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3` |

Gradle verifies the runtime asset hash before every Android build.

## Project opt-in

The New Project card includes a default-off **Generate team side-switch markers**
checkbox below **Prepare score tracking**. Its help text is: “Enable this only for
formats where teams change court sides during the recording.” The checkbox is
available only when score tracking is prepared.

When the option is off, Android still generates serving-side markers but skips switch
state restoration, switch frame requests, feature extraction, runtime loading, and
model inference. Model-origin switch markers are not seeded into score tracking;
manual switch markers remain available and are preserved. The choice is persisted in
both native and editor project schemas. Projects from older schemas migrate to enabled
only when they already contain cached switch output.

## Pipeline

Android ports the frozen browser contract without retraining or retuning:

1. Persist both production rally/dead-state traces with the rally result. For older
   projects, restore the same traces from the retained contextual feature cache.
2. Generate every adjacent-rally boundary candidate plus retained internal dead-state
   peaks, including the four-second edge exclusion and score-ranked 14-second NMS.
3. Union the serving-side and team-switch frame requests. Decode forward while the
   next requested timestamp is at most five seconds away; start a new extractor/codec
   segment across a larger positive gap.
4. Build the seven-frame before/after comparisons at 256x144 BGR, including court/net
   calibration, vertical warp, upper-court phase correlation, HSV motion palettes,
   player-like component filtering, and near/far foot assignment.
5. Append the ten reductions from both production model bundles and the two candidate
   metadata values. Apply the frozen imputation, standardization, logistic head, and
   threshold.
6. Decode in descending probability order with ordinal-distance suppression and the
   post-six soft logit penalty. No cadence, time-distance, re-anchoring, or hard count
   rule is added.

The score-tracking schema stores switch origin, confidence, model event ID, and linked
rally IDs. Deleting an inferred switch creates a model-marker tombstone, so rerunning
inference does not resurrect the user's deletion. Manual markers survive reseeding.
Ignored intervals and excluded rallies hide linked serve and switch markers from score
derivation without deleting their cached decisions. Feedback exports include the full
switch feature matrix, scored candidates, and runtime identity.

## Pixel 10 verification

The opt-in end-to-end instrumentation test ran on a Pixel 10 Pro (`blazer`), Android
17 / API 37, against the existing real project `project-mqmfyy`:

| Property | Result |
| --- | ---: |
| Source | H.264 1920x1080 at 60 fps |
| Source duration | 1,105.817 s |
| Rally ranges | 52 |
| Shared requested frames | 1,210 |
| Gap-aware decoder segments | 39 |
| Team-switch feature rows | 56 |
| Selected model switches | 2 |
| Serving-side rows | 52 |
| Initial shared pipeline wall time | 882.845 s |
| Instrumentation result | Passed; output persisted |

The earlier 12-second device experiment spent too long decoding sparse intervening
video. The fixed five-second policy follows the serving-side schedule diagnostic and
bounded the combined real-video run to 39 segments. During the completed run the
device reported thermal status 0; observed process PSS remained below 252 MB.

The first combined run also exposed two Android-only costs. The fallback sampler was
converting every decoded 60-fps frame for one second before every requested timestamp,
rather than materializing only the requested frame, and the service emitted progress
broadcasts quickly enough to overwhelm the editor main thread. The decoder now limits
fallback conversion to a true end-of-video request, reuses precomputed YUV crop maps,
and reports decode/setup/conversion/model phase timings. UI progress is rate-limited
while stage changes and completion still emit immediately.

The final parity-gated run kept the same 39 segments and decoded source-frame count,
but converted only the 1,208 decoder outputs that satisfied an actual specialist
request. It also generated only the requested specialist image type at each timestamp
and reused each prepared side-switch image bank across its broad and player summaries.

| Final optimized phase | Time |
| --- | ---: |
| Decode setup | 12.649 s |
| Requested-frame pixel conversion | 51.703 s |
| Complete shared decode | 428.822 s |
| Serving-side feature/model evaluation | 9.022 s |
| Team-switch feature/model evaluation | 25.925 s |
| Planning and orchestration | 0.116 s |
| **Complete specialist pipeline** | **463.954 s** |

This is 418.891 seconds, or 47.4%, faster than the initial 882.845-second run. The
test compared every persisted serving-side and team-switch raw feature value with the
previous output and passed exact parity. Decode still accounts for 92.4% of the final
wall time: the 1,210 sparse requests require 43,058 source decoder outputs because
previous-sync GOP preroll makes the optimal five-second plan traverse about 717 seconds
of the 1080p60 source.

After reinstalling the final debug APK without clearing application data, the editor
reported 39 visible serves and 2 visible switches after ignored/excluded filtering,
plus `SERVE + SWITCH MARKERS READY`. The persisted draft contained both inferred
switches as model-origin markers with their confidence, event ID, and rally links. A
pre-existing manual switch was preserved.

## Automated coverage

`testDebugUnitTest`, `assembleDebug`, and `assembleDebugAndroidTest` pass. Coverage
includes frozen model-score parity, candidate and decoder rules, opt-in persistence,
schema migration, disabled-marker suppression, runtime/output validation, state
restoration and persistence, progress-update throttling, marker seeding, tombstones,
manual-marker preservation, and ignored/excluded switch visibility.

The long device test is
[`SideSwitchPipelineInstrumentedTest.kt`](../../android/app/src/androidTest/java/com/volleycut/nativeanalysis/SideSwitchPipelineInstrumentedTest.kt).
It is opt-in because it processes the complete stored source video.
