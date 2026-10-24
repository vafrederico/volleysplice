# Courtside Deck rally review

Rallies is the initial workspace. Rallies, the existing full Editor, and Export
share one `ProjectDocument.draft`, score tracking, AVPlayer, source position and
playback rate. Navigation and list position are separate view state. No synthetic
data enters the app; the preview harness is independent of the app target.

The reference is Android Courtside Deck C1 at `0c0f754f`, including
[`RallyReviewWorkspace.kt`](../android/app/src/main/java/com/volleycut/nativeanalysis/RallyReviewWorkspace.kt),
[`RallyReviewModels.kt`](../android/app/src/main/java/com/volleycut/nativeanalysis/RallyReviewModels.kt)
and the [native reference captures](../artifacts/design-experiments/courtside-deck/native/verification.json).
The iOS implementation uses the C1 ground `#eef3f4`, ink `#131416`, muted
`#5d5f60`, teal `#1b6c80`, control `#cdced1`, and raised `#f4f6f8` roles with
native system sans typography. The full Editor retains its established palette.

## Workflow and implementation

- The largest region answers “is this rally worth keeping?” with real video.
  The focused timeline, previous/next retained context, and explicit decisions
  carry the work; metadata stays quiet.
- `RallyReview` coalesces clip, cleanup and serve review work by rally. Orphan
  cleanup/serve stops remain reachable. Fully ignored evidence is skipped.
  Next review navigates without resolving decisions.
- Keep/Remove writes through existing draft validation and saving. Logical
  cleanup fragments resolve together; serve decisions remain independent.
  Near/Far edits existing markers or explicitly creates a manual marker;
  Not a serve uses the existing removal/tombstone reducer.
- The shared top bar centers its pills between equal side lanes. The project
  selector, adjacent arrow/trash and trailing settings stay within the existing
  horizontal safe-area policy. Metrics scroll within the trailing lane. Narrow
  portrait wraps the same bar and allows the project name two lines. Saved
  projects show ready/queue/processing/attention status.
- Both team names edit shared score state. Rate changes update the persisted
  draft and the running player. Previous rally, Next review and Next rally retain
  paused/playing/waiting-to-play intent and replace the preview stop boundary.
- Trimming pauses for inspection. Drag handles and one-second nudges retain the
  current frame within the **updated padded/retained range**, clamping only an
  excluded frame. Rally-bound serve timestamps follow existing editor math.
- The local trim window includes the selected padded range and eight seconds of
  context on either side. Distant retained neighbors are labeled edge actions;
  hour-long gaps never stretch that window. The window stays stable during a trim.
- The browser uses scrollable rally/score/source-time rows without thumbnails.
  Manual list dragging suspends follow and updates its independent center marker.
  Source seeking follows video, selects intersected rallies and scrolls the list;
  gaps retain the previous selection and emphasize Add missing.
- Native source sliders expose adjustable/set-value accessibility in the browser
  and Add sheet. The vertical source rail also has adjustable actions and distinct
  circle/video and diamond/list-center markers. Add can open inside existing
  footage; source seeking then proposes a valid gap. Only Add rally commits it.
  Overlap with existing or ignored footage fails validation; cancellation edits
  nothing.
- Export owns score visibility/fade, point history, MP4 destination/save, chapters
  and project saving. Settings contains project options, processing/storage,
  version, model reanalysis and tutorial replay. The full Editor still exposes its
  score, cleanup, padding, ignored-time, clip and iPad resize tools.
- The persistent tutorial points at actual deck controls and opens the appropriate
  workspace/browser. Advancing it changes navigation only. Completed/dismissed
  progress stays dismissed; unfinished legacy editor steps migrate to the new tour.

## Available validation

The Windows/WSL lab ran **190 portable core XCTest cases with zero failures**,
including twelve new rally-review regressions. The ARM64 iOS app was built and
development-signed through the current-source xtool path using `-NoInstall`.
These results establish source compilation and portable behavior, not native UI
execution. Production bundle identity remains `com.volleysplice.VolleySplice`;
lab-only signing substitutions stay in external staging.

The test invocation uses the actual checkout, serially with the app build:

```powershell
& "$env:VOLLEYCUT_IOS_LAB_ROOT/xtool/test-xctests.ps1" -SourceRoot $PWD -Target portable
& "$env:VOLLEYCUT_IOS_LAB_ROOT/xtool/test-app.ps1" -SourceRoot $PWD -NoInstall
```

`RallyReviewTests` covers orphan work, ignored ranges, overlapping padding/serve
identity, explicit decisions, gap navigation, retained neighbors, nonzero-padding
frame retention, hour-long gaps and Add seeking without draft writes.
`CourtsideDeckUITests` and the updated existing UI tests describe Apple-runtime
checks for the new navigation, shared team edits and native source-slider seeking.
All thirteen hosted integration-test sources and six UI-test sources typechecked
for ARM64 iOS. They have not been packaged or executed as Apple-runtime suites
during this change.

## Local compatible SwiftUI preview

See [the preview harness](Previews/CourtsideDeck/README.md). The exact app source
for decision, serve, nudge and navigation controls compiles locally with Tokamak
StaticHTML. The surrounding layout, score and video are explicitly synthetic.
The three generated HTML probes are checked in for review.

This is a compatible SwiftUI implementation, **not Apple's native SwiftUI
renderer**. StaticHTML does not execute button actions or state changes. Its
frame expansion and grid spacing differ from native SwiftUI: the inspected
controls cluster at intrinsic widths, and the synthetic video slot shrinks.
Consequently these probes establish local source reuse and rendered control
content, not C1 geometry or final native layout fidelity. No project source or
private data was submitted to a hosted playground.

Apple-runtime validation remains required for both landscape rotations, notches,
portrait/tablet split view, Dynamic Type/VoiceOver focus and set-value behavior,
list gestures/scroll anchoring, drag hit testing, AVPlayer seeks/buffering/preview
stops, score timing, export media and system-sheet presentation. No Mac VM,
Simulator, iPad, WDA session, device installation or installed-app data was used.
