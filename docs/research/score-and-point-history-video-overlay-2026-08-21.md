# Score and point-history video overlay — 2026-08-21

> Historical design research, not the implemented product contract. The shipped browser
> feature uses next-serve serving-side markers, renders only the compact current-score
> box, previews it with accessible DOM/CSS, and burns it into export with Canvas 2D. It
> does not store explicit award events, sets, serving players, or an encoded point-history
> rail. See
> [`serving-side-score-tracking-android-spec.md`](../serving-side-score-tracking-android-spec.md)
> for the normative browser behavior and Android handoff.

## Purpose

This note evaluates whether VolleyCut can burn a tracked volleyball score, serving-team indicator,
and point-by-point history into the final exported MP4. It covers both the native Android app and
the production web app.

The conclusion is **yes on both platforms**. Neither export path needs a new server-side rendering
service. Android can use a timestamp-aware Media3 video effect, while the web app can composite the
graphic into each decoded frame before passing that frame to its existing browser encoder.

The encoding work is not the main product risk. The larger requirement is a reliable, editable,
source-timestamped score model that survives project reloads and produces an immutable snapshot for
each export.

## Product state when this research was written

At the time of this research, VolleyCut did not persist a game score, point winner, or serving
team. The production browser now does so through the next-serve contract documented in the
normative Android handoff above; native Android still does not. Other fields named `score` refer
to model confidence or suppression confidence, not volleyball scoring.

The Android editor persists cut decisions in
[`EditorDraft`](../../android/app/src/main/java/com/volleycut/nativeanalysis/EditorModels.kt), and the
web editor persists the corresponding state in
[`CutDraft`](../../prod/src/lib/cut-draft.ts). These are the natural homes for manually confirmed
score tracking because the data is part of the user's edit rather than an immutable inference
result.

The score state should also be included in the downloaded edit-list JSON on both platforms. That
makes the burned overlay reproducible and lets a later exporter distinguish a rendering error from
incorrect source score data.

## Recommended score-event contract

Store events against the **original source-video timeline**, not the shortened export timeline.
Each completed point should retain enough authoritative state that corrections and unusual scoring
cases do not depend on replaying volleyball rules.

A logical point event should contain:

- a stable event ID;
- the source-video time at which the point is awarded;
- set number;
- Team A and Team B scores after the point;
- the team that won the point;
- the serving team before the rally;
- the serving team after the point;
- an optional serving-player label or identifier;
- an optional confidence or confirmation state if some values were inferred; and
- update metadata needed to distinguish a user-confirmed correction from an automatic suggestion.

Match overlay metadata should separately contain:

- team IDs, display names, abbreviations, and colors;
- initial set number and initial score;
- initial serving team;
- sets won, if the overlay will show match score;
- the overlay schema version; and
- visual preferences such as position and whether point history is enabled.

The post-point score and serving team should be stored explicitly instead of always being derived
from the point winner. This handles manual corrections, penalty points, imported partial games, set
boundaries, and competition-specific first-serve rules without hidden assumptions.

An explicit award timestamp is important. If the score is updated only at the next serve, the final
point of a set or match may never appear in the video. The normal display rule should be:

1. During a rally, show the pre-rally score and `servingTeamBefore`.
2. At the point-award timestamp, switch to the stored post-point score and `servingTeamAfter`.
3. Keep that state until the next event.

If the product initially knows only rally end rather than an exact award moment, rally end can be
used as a provisional timestamp. The editor should allow the user to correct it.

## Source-time to export-time mapping

VolleyCut removes dead time and concatenates retained source intervals. Score events must therefore
remain source-aligned while rendering operates on a shortened output.

For retained intervals ordered as `[start, end)`, the output start of interval `i` is the sum of the
durations of all earlier retained intervals. A frame within that interval has:

```text
source time = interval start + clip-local time
output time = output interval start + clip-local time
```

The renderer chooses the latest score event whose award time is less than or equal to the frame's
source time. This has desirable behavior at every type of edit boundary:

- An event inside retained footage changes the graphic at the correct frame.
- An event inside removed dead time is not rendered there, but the next retained frame starts with
  the updated state.
- Padding and retained short gaps naturally use their real source times.
- Joined intervals do not need special score logic.
- A removed rally can still affect later score state if score tracking says that point occurred.

The last case is technically correct but can reveal an editing inconsistency: a point may appear in
the history even though its rally was removed. The editor should warn when a confirmed point event
does not overlap any final retained interval, because that is often an accidental cut rather than a
desired result.

Score lookup should be implemented as a pure, tested reducer or indexed timeline. Exporters should
not contain volleyball scoring rules; they should only render the authoritative state at a source
timestamp.

## Overlay design

The primary graphic should be a compact scoreboard with:

- Team A and Team B names or abbreviations;
- the current set score;
- an unambiguous serving indicator next to the serving team;
- optional set wins; and
- a high-contrast translucent background, outline, or shadow.

The point-history graphic can show one colored pip per point winner. It must reveal only points whose
award timestamps are at or before the current frame so that the graphic never spoils the remainder
of the set.

For readability:

- show the complete current-set sequence as individual pips;
- separate completed sets into their own rows or compact grouped strips;
- scale the pip size within bounded limits rather than allowing the row to run off-screen;
- retain a clear set divider when teams switch court sides;
- use shape or an internal mark in addition to color so the history is still interpretable with
  color-vision deficiencies; and
- respect title-safe margins and phone-camera overlays near screen edges.

Very long deuce sets and full multi-set matches require an explicit product rule. A good default is
full individual history for the current set, compact completed-set strips, and an optional match
score. This still represents every point while keeping the current action readable.

The export control should offer at least:

- `Include scoreboard`;
- `Include point history`; and
- overlay position.

An incomplete or internally inconsistent score timeline should produce a warning before the user
burns it permanently into an MP4. A non-destructive export without the overlay should remain
available.

## Production web app

### Existing export path

The web exporter in
[`prod/src/lib/on-device/export.ts`](../../prod/src/lib/on-device/export.ts) already performs the
required media pipeline locally:

1. `VideoSampleSink` decodes retained source frames.
2. Each frame is clipped and retimestamped onto the shortened output timeline.
3. `VideoSampleSource` encodes the frames as AVC.
4. Audio is independently clipped and encoded as AAC.
5. Mediabunny muxes the result into an MP4 written to a selected file, origin-private storage, or
   the streaming download path.

The current video loop begins around the `videoPump` in that file. The source timestamp is available
before `sample.setTimestamp(...)` replaces it with the output timestamp. This is the exact place to
resolve score state.

### Proposed browser compositing path

Create one reusable full-resolution `OffscreenCanvas` for the export and reuse its 2D context for
every frame:

1. Save the decoded sample's original source timestamp.
2. Resolve the score state and visible point history at that timestamp.
3. Draw the decoded video sample onto the canvas.
4. Draw the scoreboard and history on top.
5. Create an outgoing Mediabunny `VideoSample` from the canvas with the clipped output timestamp and
   duration.
6. Pass the outgoing sample to the existing `VideoSampleSource` encoder.
7. Close both input and outgoing samples promptly and reuse the canvas.

Mediabunny already exposes the required primitives: a `VideoSample` can draw itself into a canvas,
and a canvas can be wrapped as a new `VideoSample`. No FFmpeg build, upload, or server render is
required.

The export function should accept an immutable overlay snapshot alongside the final intervals. The
caller in [`CutEditor.tsx`](../../prod/src/components/CutEditor.tsx) should create that snapshot at
the moment export begins. Changes made after export starts must not affect frames still being
encoded.

### Browser preview

The editor can place a transparent canvas over the existing video preview and redraw it whenever
playback time, score data, or viewport size changes. The preview and exporter should call the same
pure drawing function so typography, spacing, team colors, and point-history behavior match.

CSS or React DOM can still be used for accessible score-entry controls, but the visible export
preview should use the canvas renderer because the final MP4 will not contain DOM elements.

### Web-specific risks

The existing path decodes and re-encodes video, but it can currently pass each decoded sample almost
directly to the encoder. Canvas compositing adds a full-frame draw and an RGB-backed outgoing frame
for every sample. This is likely inexpensive for a simple overlay at 720p or 1080p, but it can be
noticeable for long 4K recordings.

Mitigations are:

- reuse one `OffscreenCanvas` and context;
- allocate no arrays, gradients, fonts, or layout objects inside the per-frame loop;
- cache the resolved score layout until the next score event;
- use device-pixel coordinates derived once from the output dimensions;
- continue respecting encoder backpressure; and
- benchmark Canvas 2D before introducing a WebGL compositor.

Rotated phone footage needs deliberate handling. The simplest correct path is to draw into
display-oriented dimensions, bake the rotation into the pixels, and emit output rotation metadata
of zero. Retaining rotation metadata while also drawing a display-oriented canvas would rotate the
overlay twice. Tests must cover 0, 90, 180, and 270 degree inputs.

Canvas compositing can also affect color handling, especially for HDR or wide-gamut inputs. The
first product version should explicitly target the existing SDR AVC output contract and verify that
the overlay does not introduce unexpected range or contrast changes.

## Native Android app

### Existing export path

[`ExportService.kt`](../../android/app/src/main/java/com/volleycut/nativeanalysis/ExportService.kt)
creates one clipped Media3 `EditedMediaItem` per retained interval, arranges them in an
`EditedMediaItemSequence`, and exports the resulting `Composition` as H.264/AAC.

Media3 supports video effects on each `EditedMediaItem`. The appropriate implementation is a custom
timestamp-aware `CanvasOverlay` wrapped in `OverlayEffect` and `Effects`. The Android module would
need a direct dependency on `androidx.media3:media3-effect` using the same Media3 version already
declared for common, ExoPlayer, Transformer, and muxer.

Relevant upstream documentation:

- [Create a basic video editing app using Media3 Transformer](https://developer.android.com/media/implement/editing-app)
- [CanvasOverlay API](https://developer.android.com/reference/androidx/media3/effect/CanvasOverlay)
- [OverlayEffect API](https://developer.android.com/reference/androidx/media3/effect/OverlayEffect)
- [EditedMediaItem.Builder API](https://developer.android.com/reference/androidx/media3/transformer/EditedMediaItem.Builder)

### Proposed Android compositing path

Create a separate overlay instance or pre-shifted event view for each retained interval. For an
interval starting at source time `S`, map an overlay callback's clip-local presentation time `t` to
`S + t`. This avoids relying on undocumented assumptions about whether presentation timestamps
continue or reset across items in a composition.

At export queue time, copy team metadata, point events, and overlay preferences into the
`ExportJob`. The current job already snapshots the retained interval start and end arrays. Score
data should follow the same principle so an editor correction cannot change a queued job.

The overlay can draw the entire scoreboard and history in one Android `Canvas` pass. Text and pip
layout should be cached between score changes, and rendering should avoid per-frame bitmap
allocation.

Adding a video effect guarantees that the video track must be decoded, composited, and encoded. The
current no-effect path may be able to copy compatible compressed samples in some cases, so an
overlay can increase Android export time and power consumption even though the service already
requests H.264/AAC output. This should be reflected in progress copy and benchmarked on the release
target device.

## Cross-platform structure

The two apps should share a behavioral contract even though Kotlin and TypeScript will have separate
renderers:

- identical score-event JSON schema;
- identical event-boundary semantics;
- identical rules for events in removed time;
- identical point-history ordering and set grouping;
- identical team color defaults and contrast requirements;
- identical export toggles; and
- identical edit-list representation.

Useful independent modules on each platform are:

1. **Score timeline reducer:** validates events and returns authoritative state at a source time.
2. **Export time mapper:** maps frames and intervals without knowing volleyball rules.
3. **Overlay layout:** converts score state and dimensions into drawing primitives.
4. **Platform renderer:** draws those primitives to Android Canvas or browser Canvas.
5. **Export snapshot:** freezes data and preferences for the life of an export.

Separating these responsibilities keeps score correction logic out of the encoder and makes most of
the behavior testable without processing video.

## Validation plan

### Pure logic tests

- state before the first event, exactly at an award timestamp, and immediately after it;
- consecutive points by the same team and a change of server;
- score correction without duplicated history;
- set transition and explicit first server of the next set;
- imported partial game with a non-zero initial score;
- an event in a removed gap;
- an event exactly at an interval start or end;
- multiple retained intervals and accumulated output offsets;
- suppressed rally with a retained score event warning; and
- immutable export behavior after the editor data changes.

### Rendering tests

- golden images at 720p, 1080p, and 4K;
- landscape and portrait/rotated sources;
- short and long team names;
- high and deuce scores;
- one point, a normal complete set, a long deuce set, and a multi-set history;
- light and dark source frames behind the graphic;
- serving indicator distinguishable without color; and
- safe placement under common phone and social-video crops.

### Encoded-output tests

- MP4 video and audio remain playable and synchronized;
- output duration matches the sum of retained intervals;
- score changes on the expected encoded frame;
- the next retained frame reflects events that occurred in omitted time;
- Android and web produce equivalent score state at sampled source timestamps;
- export cancellation releases canvas, frame, encoder, and temporary-output resources; and
- browser and Android performance at 1080p and 4K remains acceptable.

## Suggested implementation sequence

1. Define and test the cross-platform score-event and overlay-settings schema.
2. Add score entry, serving-team selection, correction, and persistence to the web editor first.
3. Add a browser preview using the shared web canvas renderer.
4. Integrate the renderer into the existing web `videoPump` and benchmark 1080p/4K exports.
5. Add the equivalent Kotlin models and editor controls while retaining JSON compatibility.
6. Add the Android Media3 overlay effect and export snapshot.
7. Add score data to both edit-list formats and cross-platform fixtures.
8. Validate output parity, rotation, event boundaries, A/V sync, cancellation, and long-match layout.

Starting on the web is convenient because the current exporter exposes decoded frames directly and
the same canvas renderer can drive both preview and final encoding. This is an implementation-order
recommendation, not a requirement; neither platform depends on the other.

## Product decisions still required

- Is serving tracked only by team, or optionally by player?
- Does the main scoreboard show set wins as well as the current set score?
- Should completed-set point history remain visible throughout the match?
- What should happen when score data is incomplete: warning, block overlay export, or allow a
  confirmed partial overlay?
- Should the score change precisely at rally end, at a user-marked award time, or after a short
  configurable delay?
- Should overlays be enabled by default once a complete score timeline exists?
- Which corners must remain free for common social-platform or broadcast graphics?

These choices affect editor UX and visual layout, but none changes the underlying feasibility.
