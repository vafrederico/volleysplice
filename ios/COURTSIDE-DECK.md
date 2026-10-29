# Courtside Deck rally review

Rallies is the initial workspace. Rallies, the existing full Editor, and Export
share one `ProjectDocument.draft`, score tracking, AVPlayer, source position and
playback rate. Navigation and list position are separate view state. Normal
launches use real projects. An explicit Debug-only device fixture supports native
controls checks; the compatible-renderer preview remains independent.

The reference is Android Courtside Deck C1, including the linear timeline restored
before `91b6e6d3`, with
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
  Next review stays disabled until the current clip, cleanup and serving-side
  decisions are resolved. Advancing never resolves other pending decisions.
- Keep/Remove writes through existing draft validation and saving. Logical
  cleanup fragments resolve together; serve decisions remain independent.
  Near/Far edits existing markers or explicitly creates a manual marker;
  Not a serve uses the existing removal/tombstone reducer.
- The shared top bar centers its pills between equal side lanes. The project
  selector, adjacent arrow/trash and trailing settings stay within the existing
  horizontal safe-area policy. Landscape metrics sit immediately left of Settings,
  with review labels on separate lines. Narrow portrait has a full-width summary
  row and three equal review buttons, with no duplicate row above Game timeline.
  The project name can use two lines. Saved
  projects show ready/queue/processing/attention status.
- Both team names edit shared score state in the first **Project settings** section
  of the Settings modal. The fields are inline; Project settings has no nested
  navigation button. Save project and the existing project actions are directly
  available there. Rate changes update the persisted
  draft and the running player. Previous rally, Next review and Next rally retain
  paused/playing/waiting-to-play intent and replace the preview stop boundary.
  Review and focused editing stop playback at the selected rally's retained end
  through AVPlayer's media boundary. Play after a context seek or at the end restarts
  that rally when needed. All rallies and Add missing rally use the full source
  bounds so playback can inspect the recording and gaps. Editor and Export release this boundary.
  Navigation stays below the scrolling review controls. Native buttons provide
  full touch targets even when the landscape interface is scaled down. Queue opens
  from Settings; there is no persistent bottom queue strip.
  Tablet portrait gives the player most of the available height while reserving
  space for review controls. Tablet landscape has a draggable video/detail divider,
  shared by focus and browser, with an adjustable accessibility action. The duplicate
  team/score strip above the video has been removed on iOS and Android; the video
  retains the configured score overlay.
- Both Rallies and Editor preview the same configured score overlay, point history
  and fade behavior inside the video's fitted bounds. Selection uses an outline;
  the rally fill retains the Editor's pending, kept or removed state colors.
- Trimming pauses for inspection. Drag handles and one-second nudges retain the
  current frame within the **updated padded/retained range**, clamping only an
  excluded frame. Rally-bound serve timestamps follow existing editor math.
- The deck uses one linear source-time scale containing the selected rally, its
  padding and both nearest retained neighbors. Real gaps keep their proportional
  width. Context expands symmetrically around the selected core's midpoint,
  limited only by the available source bounds. It never compresses gaps or zooms
  automatically to enlarge short cores;
  the one-second buttons remain available for those. The full Editor retains its
  core detail scale on wide layouts. Phone portrait Editor uses one track with
  half-height rally core handles; padding is displayed and configured through
  Settings. Both Rallies workspaces and Android portrait Editor use the same
  half-height handle treatment, keeping their full-height touch areas.
  Drag translation uses the original boundary and a
  frozen scale, so touching away from the center cannot jump the edge.
- Trim writes reconcile the current presentation before resolving the selected
  group's retained range. Joining a previous or next rally preserves a frame in
  the newly retained footage; serves and unresolved reviews still separate groups.
- The browser uses scrollable rally/score/source-time rows without thumbnails.
  An ordinary tap on an unselected row selects and seeks while staying in All
  rallies. An ordinary tap on the currently selected row opens its controls,
  regardless of how long it has been selected. Two quick taps naturally select
  then open; no timed double-tap recognizer or delayed selection is used. The
  explicit Edit button remains, and opening makes no review or score decision.
  Manual list dragging suspends follow and updates its independent center marker.
  Source seeking follows video, selects intersected rallies and scrolls the list;
  gaps retain the previous selection and emphasize Add missing.
  On iOS 18+, native scroll phases observe interaction without competing with the
  scroll view's pan gesture. Restoring the list after a workspace switch preserves
  its independent center until the next manual scroll or source seek.
- The browser has two chronological source-timeline columns in a 96-point sidebar:
  the first half on the left and the second half on the right, each with time
  labels and a 44-point touch area. Cross-midpoint rallies render in both. Each
  marker has one owning column; seeking resumes follow and updates selection.
  Hide/Show timeline toggles both columns together. The Play/List legend identifies
  the playback dot and independent list-center diamond. A matching white-filled,
  dark-outlined diamond sits immediately after the centered rally number. Both
  diamonds use the same list-center identity and the same 14-point drawing (5-point
  radius, 2-point stroke), with 6 points after the number. Exactly one row is
  marked. Manual list scrolling moves both diamonds without seeking or changing
  the selected-row styling; source seeking resumes follow. Both the source columns
  and inline Add rows seek directly by tap/drag and expose adjustable accessibility
  actions. Neither view has a separate source-seek slider. Add replaces the detail
  pane while keeping the same player and transport visible. Its source timeline
  wraps across three compact consecutive rows on phone and iPad;
  each row seeks the player and follows its live playhead. There is no separate
  seek slider or nudge row. One fixed bottom action says Mark start at [playhead],
  then Mark end & add rally. Only that second press creates the rally and opens
  its controls. Start/End/Reset share a row above the timeline; Start stays anchored,
  End follows the player, and Reset clears the mark. The Add title is in the
  workspace header, with instructions and Cancel in the lower pane. Existing rallies, ignored
  footage, reverse ranges and ranges under 500 ms cannot be added. Cancellation
  edits nothing and reopening starts a fresh selection.
- Export owns score visibility/fade, point history, MP4 destination/save, chapters
  and project saving. Settings contains project options, processing/storage,
  version, model reanalysis and tutorial replay. The full Editor still exposes its
  score, cleanup, padding, ignored-time, clip and iPad resize tools.
- Phone portrait Editor keeps Play only final video inside Now reviewing. The
  top review counters seek and select review items without moving the outer
  Editor scroll position; marker and clip lists continue following selection.
  The Current rally heading contains previous/number/next navigation. Preview rally,
  Reset extra time, tenth-second nudge buttons and redundant confidence/timestamps
  are absent from that compact section. Manual deletion and exact handle editing
  remain available. Android portrait also omits Preview rally and Reset extra time.
- The persistent tutorial points at actual deck controls and opens the appropriate
  workspace/browser. Advancing it changes navigation only. Completed/dismissed
  progress stays dismissed; unfinished legacy editor steps migrate to the new tour.

The Debug-only [Test as iPhone setting](README.md#testing-phone-layouts-on-ipad)
constrains the native iPad app to phone points and compact layout rules. It keeps
the editor's identity and state while switching. Universal iPad builds support
portrait and landscape with the toggle off.

## Available validation

The Android-parity refinement passed **259 portable core XCTest cases**, an ARM64
Debug build/sign/install, and targeted native checks on the physical iPad using
440 x 956 and 375 x 667 phone test layouts and normal iPad landscape. Hosted and
UI test sources typechecked for ARM64 iOS; their formal XCTest suites were not
executed. Captures use synthetic fixture media; actual iPhone hardware remains
unverified.

Native checks cover both source columns, unique playback/list markers, matching
row diamonds, manual scrolling without seeking or changing selection, preserving
the independent center across workspace switches, resuming follow, Hide/Show,
and rendering a rally across the column midpoint. Ordinary first, later selected
and rapid consecutive row taps select/open correctly. Pending serving review opens
in Review mode; ready rallies open in Editing mode without making decisions.

Inline Add checks cover the three-row cap, no extra slider or seek nudges, direct
tap and horizontal drag seeking, anchored start, invalid ends, Reset, Cancel and
adding exactly one rally straight into its controls. Earlier native checks also
covered player geometry and source playback in phone portrait and iPad landscape.
Add mode retains the current video/detail width and suspends the resize gesture
so it cannot intercept timeline touches. Display preferences are restored and the
device returned to verified awake parking after native checks.

Build **1.0.0 (4)** passed 207 portable core XCTest cases and ten targeted native checks on the physical iPad in
phone portrait and normal iPad landscape: the duplicate bar is absent, the score
overlay remains, and both names are editable inline at the top of Settings and
shared with the Editor. Android **0.10.34-debug** passed 250 JVM tests and the
targeted physical Pixel checks for team-name sharing/recreation, direct project
settings and video seeking. See the [team settings receipt](Previews/CourtsideDeck/team-settings-verification.json).
The fresh data comparison found no changes to unowned documents or preferences;
only the owned synthetic fixture and its checkpoint changed. Release packaging
was not repeated for this UI-only update. The tutorial now directs team-name edits
to the top Project settings section in the cogwheel modal on both platforms.

### Previous refinement checks

Build **1.0.0 (3)** passed **207 portable core XCTest cases** and **32 targeted
native checks** on the physical iPad. The [refinement receipt](Previews/CourtsideDeck/refinement-verification.json)
records the compact Editor, shared trim edits and state colors, landscape metrics,
video resizing in focus/browser, score-overlay settings and the four named phone
presets in both orientations. The smallest preset's scrollable review controls
were exercised in portrait and landscape. Normal iPad layouts were checked in
both orientations. Debug was installed; Release was built/signed with `-NoInstall`
and checked to exclude both debug preference keys and the testing controls.

Android **0.10.33-debug** passed **250 JVM tests** and two physical Pixel checks
covering rendered video seeking, the shared score overlay, and removal of the
retired portrait actions. The emulator's video check hit MediaCodec errors;
playback qualification uses the physical Pixel. Earlier user edits and their
matching draft journal were preserved; no older backup was restored.

### Earlier regression baseline

The preceding build passed **206 portable core XCTest cases** and **76 native
workflow checks** on the physical iPad. The [device receipt](Previews/CourtsideDeck/device-verification.json)
records nineteen checks in each of four layouts: the phone viewport in portrait
and landscape, then the normal iPad workspace in portrait and landscape.

Native checks covered retained-frame trimming and continuous handle dragging,
focused/source/vertical timeline seeking, explicit keep/remove and serve states,
Not a serve removal/replacement, team editing, rate changes, playing navigation,
review progress, list following and its separate center marker, collapse/expand,
explicit Add/cancel, shared Editor edits/playhead, Export settings and destination
cancellation, and Settings reachability. Lists were scrolled where their rows
overflowed; the large landscape iPad viewport fits all six fixture cards.
Toggling phone mode also preserved the selected Export pill, project, rally bounds,
edited team and paused playhead. The original always-visible score preference was
restored, and the device was left in normal iPad mode, parked awake.

Both Debug and Release ARM64 iOS apps built and were development-signed using the
current checkout. Debug was installed without resetting application data. Release
used `-NoInstall`; its executable excludes the debug preference, toggle and fixture
launch keys, with the Debug executable serving as a positive control. Production
bundle identity remains `com.volleysplice.VolleySplice`; lab signing substitutions
stay outside the repository. No Mac VM or Simulator was used.

The serial validation commands are:

```powershell
& "$env:VOLLEYCUT_IOS_LAB_ROOT/xtool/test-xctests.ps1" -SourceRoot $PWD -Target portable
& "$env:VOLLEYCUT_IOS_LAB_ROOT/xtool/test-app.ps1" -SourceRoot $PWD
& "$env:VOLLEYCUT_IOS_LAB_ROOT/xtool/test-app.ps1" -SourceRoot $PWD -Configuration release -NoInstall
```

`RallyReviewTests` and `RallyTrimTests` cover nonzero padding, both merge directions,
ignored/review/serve boundaries, uniform distant-neighbor context, true short-core
handle positions, frozen drag geometry, focus preservation and playback policy.
`PhoneTestLayoutTests` cover logical dimensions, reserved edges, fitting small
windows and the same automatic density/layout rules as a native phone.

The media is a short synthetic UI fixture. These checks do not qualify actual
iPhone hardware, iPhone system dialogs/keyboard, long-video decoding/audio/export,
VoiceOver, Dynamic Type, multitasking or every rotation variant. Existing hosted
integration/UI test sources were typechecked during the initial port; those
formal Apple-runtime suites were not packaged or executed in this round.

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
The shared `CoreTrimSurface` is compiled by the ARM64 app build; it is not part of
the Tokamak control probes. The tutorial's full Editor step opens that workspace.
Rallies shows Reviewing # rallies with All rallies (#) and a right arrow. All
rallies, Add missing rally and Editing rally # each have their own header and a
left-arrow Review rallies action. Opening a rally with any pending clip, cleanup
or serving review returns to Review mode; Editing applies only to a ready rally.
Resolving the current review stays in Review mode. There is no progress bar or
checked fraction; root review metrics appear only in Editor and Export. Seeking focused
context retains the selected rally, while browser source seeking resumes list
following and selection.

The compatible preview remains a source/rendering aid. Current native behavior
and its qualification limits are recorded in the device validation section above.
