# Reference project: original serve-marker parity

Checked September 11, 2026 against the saved `tds6-reference.mp4` project
`CF30CE9A-58F4-4DB6-B225-0BEF1474BBDD` and its matching draft checkpoint.
The selected game window is **6:28.065–11:38.938** (388.065–698.938 seconds),
inside the 1,105.817-second source recording. No source video was decoded for
this check and no device, project, checkpoint, or model parameters were changed.

## Result

**Android production code reproduces all 16 original iPad serve markers when
given the same saved inputs.** The extra markers relative to the full-recording
reference are not an iOS classifier or score-seeding discrepancy on those inputs.
They arise from different rally boundaries and serve evidence in the selected
analysis scope; this check does not establish that every proposed serve is real.

The local JVM replay called the compiled Android `FeatureMath.contextualize`,
both production `ModelRunner` decoders, `ProductionEnsemble.merge`,
`ServingSideModelRunner` and `ScoreReducer.seedModelMarkers`:

| Input | Rows | Rally candidates in/overlapping selected window | Original serve gates accepted |
|---|---:|---:|---:|
| Exact iPad partial-window base features and specialist raw features, Android replay | 1,243 × 104; 17 × 237 | 17 | 16 |
| Native Android base-feature cache sliced to the same window and reranked, Android replay | 1,243 × 104 | 16 | 14 |
| Same native Android cache evaluated over the full recording, counting overlapping candidates | 4,424 × 104 | 13 | 12 |

For the exact iPad input, all 17 rally starts, ends, agreement values and Float32
confidence bits match. All 17 serving-side probabilities, both serve-head peak
probabilities, sides, gate decisions and verdicts match exactly (maximum numeric
difference **0**). Reseeding the current edited score state through Android also
returns exactly the saved state, including four tombstones and the manual side
correction at 10:18.875.

The native-cache slice preserves previously generated base-feature values,
including their full-source temporal/audio boundary context. It recomputes
contextual ranks and model inference for the selected rows. It is **not** a fresh
partial-window Android media decode, and no native specialist image features
were regenerated. Its accepted serve count is determined by the production gate;
Near/Far classification is not claimed for that slice.

## Why the marker counts differ

Both platforms anchor original serve candidates to unpadded production-ensemble
rally starts, before editor suppression and short-gap joining. Each candidate
gets a marker if either production serve head reaches **0.85 within ±1 second**.
If neither does, `both-models` rally agreement still creates a **Review** marker
through `production-rally-recovery`. A single-model candidate without serve-head
evidence is `not-serve` and does not create a marker. A marker uses the rally-start
timestamp, not the nearest serve detection or the head's peak timestamp.

The iPad window has 13 serve-head candidates, three recovery candidates and one
rejected candidate. Its original 16 markers include these four later deleted by
the user:

| Marker | Source time | Original reason | Original verdict |
|---|---|---|---|
| `serve-R002` | 6:44.875 | Both-model rally recovery | Review |
| `serve-R003` | 6:54.375 | Previous-production serve-head peak 0.89936 | Far |
| `serve-R004` | 7:00.875 | Previous-production serve-head peak 0.99041 | Near |
| `serve-R008` | 7:47.875 | Both-model rally recovery | Review |

The base snapshot and matching checkpoint agree: **12 markers remain saved**.
The marker at 9:09.125 remains in score state but its rally `R012` is excluded,
leaving **11 visible markers**. `R006` is also excluded but originally had no
serve marker. Suppression suggestions are empty in this partial project.

Compared with the full-project Android snapshot, 12 original accepted anchors
have a counterpart within one second when the rally overlapping the start of the
window is included. Three timestamps match exactly: 8:05.625, 10:40.000 and
11:09.000. The four extra iPad anchors are 6:44.875, 6:54.375, 7:47.875 and
9:09.125. Examples of upstream fragmentation:

- Full-reference 6:27.625–6:58.375 becomes iPad starts at 6:28.500,
  6:44.875 and 6:54.375.
- Full-reference 7:35.625–7:56.625 becomes iPad starts at 7:35.500 and 7:47.875.
- Full-reference 8:59.500–9:17.625 becomes iPad starts at 8:59.875 and 9:09.125.

Both production contextualization and serving-side classification rank features
across the entire input row population. Changing the analysis window can therefore
change predictions away from the window edges. Comparing a single raw specialist
row in isolation would also change its ranks and would not reproduce its original
probability. The 14-versus-16 same-window native/iPad gate difference narrows the
remaining feature/input discrepancy: iPad adds the separate 6:54.375 candidate;
the later iPad 9:09.125 both-model recovery candidate corresponds to a native
9:11.875 single-model candidate whose serve gate rejects it.

## Padding and joining

The ordinary rules match: default 2-second before/after padding, clipped initial
cuts to the game window; manual ranges are not repadded; a configurable 0–10-second
join threshold defaults to 3 seconds; positive gaps join only when **strictly less**
than that threshold. Overlaps/touching ranges merge even with a zero join threshold.
Suppression barriers prevent bridging, and ignored spans are subtracted without
rejoining across them. These export unions do not collapse original serve markers;
a serve can preserve separate editable rallies inside one joined output interval.

One edge difference exists: iOS explicitly clips newly repadded veto fragments to
known source/game bounds (`EditorMath.swift:187`), while Android's materializer
uses `coreEnd + afterPadding` without that explicit clamp
(`EditorModels.kt:438`). It cannot explain this project's original marker count:
the serving candidate path does not consume materialized export intervals, and
this project has no suppression suggestions.

### Actual edited-draft padding replay

Android `EditorMath.applyPadding` and `materialize` were also executed on the
actual saved draft, after verifying it equals the matching checkpoint. This
retains the excluded `R006`/`R012` cuts, the outside-game ignored spans, score
tombstones and user correction. The source bounds are 0–1,105,817ms and the
explicit padding bounds are the game window 388,065–698,938ms. Join threshold
remains 3,000ms in every case; each case starts from the same original draft.

| Symmetric padding | Output intervals | Output union duration | Included joined-gap duration |
|---|---:|---:|---:|
| 0s | 12 | 142.125s | 7.000s |
| 1s | 11 | 167.060s | 2.500s |
| 2s | 10 | 189.560s | 1.500s |
| 3s | 10 | 208.560s | 0s |

`android-edited-padding-cases.json` contains every exact interval, contributing
cut ID and joined gap. `PaddingParityReplay.java` produced these values through
the actual Android classes; it rejects a mismatching checkpoint or unexpected
suppression settings. This is export-math comparison on edited data, not an
accuracy ranking or a padding recommendation.

`PaddingParityProbe.swift` was compiled with `ios/Sources/VolleyCore/*.swift`
and executed on the Mac. Its three arguments are the saved native project,
the Android cases JSON and an output JSON path. **All four Swift cases match
Android exactly**, including interval order, cut IDs and joined gaps; padding
also preserves score state. Results are in `swift-edited-padding-cases.json`.
The existing
`ProjectContractTests.testStrictJoinThresholdAndIgnoredGapAreNotRejoined` provides
the separate exact-3-second boundary assertion; the actual-project results alone
are not presented as that edge-case fixture.

## Evidence and reproduction

All local evidence is under `E:/wslmac/artifacts/serve-parity/`:

- `comparison-summary.json`: assertions, counts, deleted marker times and input hashes.
- `ServeParityReplay.java`, `replay.ps1`: executable harness; outputs are written
  only to this artifact directory.
- `android-on-ipad-partial-input.json`: exact-input inference, candidates and
  original/current Android-seeded score states.
- `android-native-same-window.json`, `android-native-full-replay.json`: native-cache
  scope comparison, including actual Android serve-head evidence and gate results.
- `ipad/project-CF30CE9A-58F4-4DB6-B225-0BEF1474BBDD.volleyproject.json` and its hidden
  `.draft-checkpoint.json`: read-only saved project snapshots.

From the repository root, using existing compiled Android debug classes:

```powershell
& E:/wslmac/artifacts/serve-parity/replay.ps1 -Repo (Get-Location).Path -Mode ipad
& E:/wslmac/artifacts/serve-parity/replay.ps1 -Repo (Get-Location).Path -Mode native-slice
& E:/wslmac/artifacts/serve-parity/replay.ps1 -Repo (Get-Location).Path -Mode native-full
```

The full-project editor snapshot has ID
`import-37bdca07-2465-4ed9-bc90-2a957770a12c`; it is an **imported project**, not
independent proof of native decoding provenance. The native replay instead uses
the cache `c490bc66537f0510ec3359439fb79f9ee88b533ee6767bc843b19f9a26add6a3` under
`E:/wslmac/artifacts/android-reference/files/native-features-v1/`, bound by
`E:/wslmac/artifacts/android-reference/provenance.json` to the original MP4 and ROI. Its retained
`android-analysis.json` SHA256 is
`d11fe35b3f3b111ef9bd4aeaceb4cc27f656ef485644ae3e9ef1058fcc5ff2b8`.
The iPad and imported Android snapshots have identical sampled source fingerprints
and ROIs, but different analysis windows. Full-source native and imported range
boundaries agree for 55 of 59 candidates, so those references are not interchangeable.

Relevant production paths: Android `ServingSideModels.kt:269–400`,
`ScoreTrackingModels.kt:93–131`, `EditorModels.kt:393–540`; iOS
`ServingSideInference.swift:118–246`, `ScoreTracking.swift:239–256`,
`EditorMath.swift:187–249`. This is a parity diagnosis, not model tuning or a new
accuracy ranking on the historical protected reference recording.
