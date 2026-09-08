# VolleySplice Android Design Language

This document extends the shared [`../DESIGN_LANGUAGE.md`](../DESIGN_LANGUAGE.md)
for the native Android app. The root document is the source of truth for the
product's common visual hierarchy, semantic colors, interaction states, language,
and accessibility rules. This document describes how those rules are expressed in
Jetpack Compose on phones, tablets, foldables, and Android desktop/freeform windows.

The Android UI is one responsive product, not separate phone and tablet products.
Both layouts expose the same project and editing state; only their organization and
density change.

## Shared foundation

Apply these rules from the root design language without reinterpretation:

- Keep video, score, timeline, current rally, and unresolved review work primary.
- Show real project state. Counts, colors, export details, and selection must never
  be mock data or stale copies.
- Use structure before decoration: flat surfaces, thin rules, compact headings, and
  restrained rounding.
- Give color a second cue through a label, icon, outline, pattern, or explicit state.
- Keep destructive actions explicit and red; keep unresolved review actions visibly
  distinct from optional cleanup.
- Use short operational labels. Remove redundant headings, subtitles, and help text
  when the surrounding structure already explains the control.
- Keep playback, effective final-cut ranges, Keep/Remove state, and export behavior
  semantically identical.

When this document is silent, follow the root design language. When Android needs a
platform-specific mechanic, preserve the root document's intent and use the rules
below.

## Android palette and surfaces

Android uses the same semantic roles as the web editor with a slightly warmer,
higher-contrast Compose palette:

| Android token | Value | Role |
| --- | --- | --- |
| Paper | `#f8f7ee` | App, column, card, and modal surface |
| Ink | `#17382e` | Primary text, important values, active outlines |
| Muted | `#65766f` | Supporting metadata only |
| Green | `#2f6855` | Primary controls and retained state |
| Acid | `#d9ee9e` | Review navigation and selected attention state |
| Pale green / Rail | `#c8d6ca` | Rules, padding, secondary tracks, outlines |
| Orange | `#de7959` | Review-needed labels and excluded-footage actions |
| Danger | `#b3261e` | Explicit removal and destructive actions |
| Warning | `#e8a317` | Required review emphasis |

Use the same off-white Paper surface throughout setup, review, export, and both
sidebars. Do not introduce white card islands merely to separate sections. Rules,
spacing, and small background-state changes should do most of that work.

## Responsive mode

Layout is selected from the current **usable window size**, not the device model:

- `DESKTOP`: width at least `840dp` and height at least `360dp`.
- `COMPACT`: every other window.

This means the desktop layout also applies to sufficiently wide tablets, foldables,
external displays, and Android desktop/freeform windows. A phone can enter it in a
wide desktop window, and a tablet can fall back to compact mode in a narrow split.
There is no separate device allowlist and no manual tablet switch.

Recompose between modes when the window changes. Do not scale the phone layout until
it happens to fit; rearrange the same controls into the appropriate workspace.

## Shared app shell

Setup and ready projects use the same compact shell:

- transparent VolleySplice mark and a clearly labeled current-project selector;
- concise stage navigation;
- settings and project actions at the trailing edge;
- review counters and export summary derived from the current draft;
- thin outer rules and Paper surfaces rather than a decorative app bar.

For ready projects on large screens, the header contains:

1. the current-project selector;
2. compact `Review` and `Export` tabs;
3. `Review cleanup`, `Review clips`, and `Review serves` counters on the right;
4. clips included and planned duration;
5. Settings and More.

Keep the Review and Export tabs narrow. Review counters sit immediately to the left
of the clips-included summary so unresolved work is visible without taking vertical
space. Do not add subtitles beneath the primary header items.

## Desktop and tablet review workspace

The large-screen editor mirrors the production web app's information architecture:

| Area | Contents |
| --- | --- |
| Left | Score tracking, serve and side-switch markers, point timeline, team names and scores, selected serve, clip register |
| Center | Resizable video player, one-row transport, two-row whole-game timeline, timeline legend |
| Right | Current rally, Keep/Remove and trim controls, missed/extra-footage tools, left-out footage, final-cut playback, render controls |

Each column is a flat Paper surface. Separate sidebar sections with horizontal rules
and vertical rhythm; do not wrap each section in a card. Section toggles belong on
the same row as their heading. Small cards remain appropriate for a bounded focused
object, modal, or timeline track, but never as shells around an entire sidebar.

### Resizable panes

- Left sidebar: default `292dp`, constrained to `180–520dp`.
- Right sidebar: default `292dp`, constrained to `220–560dp`.
- Preserve at least `220dp` for the center pane plus required divider/chrome space.
- Show a narrow, discoverable drag handle between each sidebar and the center.
- Drag direction must match the edge being manipulated.

The left default is deliberately wide enough for `Near`, team-switch controls, team
names, and score labels without clipping. Team-name inputs sit above their score
boxes; names may wrap rather than shrink into unreadable text.

### Video and transport

- Review player default height: `320dp`; clamp resizing to `180–1200dp`.
- The player consumes its chosen height even when the source is letterboxed. Do not
  impose a smaller aspect-ratio-derived maximum.
- Dragging the bottom resize handle changes height using display density.
- Clicking anywhere on the video toggles play/pause and exposes an accessibility
  label for the current action.
- Put current time, total duration, speed, and the compact Play/Pause button in one
  row below the player.

### Timeline

- The desktop game timeline always divides the source window into two sequential
  rows, with the midpoint ending row one and starting row two.
- Both rows are seekable and show the same marker/state vocabulary.
- Place the complete legend immediately below the timeline: rally core, padding,
  needs review, serve, side switch, excluded, and joined gap.
- A low-confidence or required-check rally keeps its own review boundary and review
  styling until the user explicitly keeps it. Do not visually or behaviorally merge
  it into surrounding retained footage first.
- Explicit Keep changes the effective timeline state to green. Explicit Remove is
  red; automatic cleanup remains a separate pale/secondary removal state.

### Right-side controls

Keep these as ruled sections in this order when space permits:

1. current rally and focused trim timeline;
2. missed or extra footage tools;
3. left-out footage;
4. `Play only final cut`;
5. `Render scores` and `Render point timeline`;
6. privacy and open-source links.

`Render scores` and `Render point timeline` start enabled for a newly created
project, matching score markers being prepared by default. Disabling score rendering
also disables point-timeline rendering when the latter would no longer be meaningful.

## Desktop Export tab

Export is a real top-level tab, not a permanent sidebar card. It presents three
parallel options on a flat workspace:

- **MP4 video:** included duration and clip count, score-render toggle, persisted
  saved-video status, progress, and Save MP4.
- **YouTube chapters:** concise explanation, ready chapter count, and Get YouTube
  chapters.
- **Project file:** explains that edits and markers are saved without the source
  video, and provides Save project.

The saved-video label must come from persisted project export metadata, not a transient
toast or process-local counter. It should survive app and device restarts.

## Rebuilt compact phone UI

Compact mode is a deliberate vertical version of the same editor, not the old UI
compressed around the new features.

- Keep score markers, point timeline, team names/scores, and selected serve ahead of
  the player so score context is available while reviewing.
- Place the clip register after the player, transport, review queues, timeline, and
  focused controls where it can remain bounded and scrollable.
- Use section cards only where a border meaningfully groups one phone-sized task.
  Every such card needs consistent internal horizontal and vertical padding; content
  must not touch its outline. This applies especially to selected serve, point
  timeline, team names/scores, current rally, and setup controls.
- Keep the same Paper background inside and outside cards. Cards provide grouping by
  outline and spacing, not by a white fill or shadow.
- Put toggles on the heading row and remove explanatory subtitles that repeat the
  heading or visible control state.
- Team-name fields remain above score boxes and may wrap.
- Touch controls should generally be at least `40–42dp` high even when desktop
  controls are denser.

The compact review player is independently resizable, defaults to `176dp`, and is
clamped to `140–720dp`. Clicking it also toggles play/pause.

## Setup screens

Setup uses the same shell, Paper surface, type scale, controls, and responsive trigger
as review. Keep one concise orientation heading and the numbered workflow; do not
turn setup into a marketing page.

Once a source is chosen, its preview player is resizable so the user can set the game
start and end accurately:

- compact setup default `176dp`, range `140–720dp`;
- desktop setup default `320dp`, range `140–1200dp`.

Setup height is stored separately from review height so tuning the game window does
not unexpectedly resize the editing workspace.

## Global layout persistence

Resizable dimensions are device/app preferences, not project data. Persist them in
the shared `desktop_editor_layout` preferences and reuse them across projects and app
restarts.

Keep independent values for:

- compact review player height;
- desktop review player height;
- compact setup player height;
- desktop setup player height;
- desktop left sidebar width;
- desktop right sidebar width.

Clamp stored values again when reading them. Layout preferences must never be included
in project JSON, copied from one project to another, or reset when a project is deleted.

## Android interaction and accessibility

- All drag handles need semantic descriptions and a non-drag fallback is desirable
  when a control becomes essential.
- Video, timelines, serves, side switches, review counters, and rally rows seek or
  navigate immediately; avoid a hidden secondary selection model.
- Keep disabled controls readable and explain why an action is unavailable through
  nearby state, not opacity alone.
- Support system font scaling without clipping team names, timestamps, or action
  labels; prefer wrapping or additional height over shrinking text.
- Preserve Android back behavior for dialogs, menus, project setup, and editor tabs.
- Respect system bars, display cutouts, split-screen insets, and freeform resizing.

## Implementation guardrails

Before accepting an Android UI change, verify:

- compact phone and desktop/tablet modes at their breakpoint edges;
- narrow split-screen and short-wide fallback behavior;
- both setup and review player resizing;
- both sidebar drag directions and width clamps;
- persistence after activity and process restart, and across different projects;
- no clipped team names, `Near`/`Far`, timestamps, or header counters;
- two timeline rows plus the complete legend on desktop;
- Keep/Remove/review-needed colors agree with effective export state;
- Export exposes MP4, YouTube chapters, and project save without duplicating controls;
- all major regions retain the shared off-white surface and flat ruled hierarchy.

The responsive and persistence behavior is implemented in
[`app/src/main/java/com/volleycut/nativeanalysis/EditorActivity.kt`](app/src/main/java/com/volleycut/nativeanalysis/EditorActivity.kt)
and covered in part by
[`app/src/test/java/com/volleycut/nativeanalysis/EditorLayoutModeTest.kt`](app/src/test/java/com/volleycut/nativeanalysis/EditorLayoutModeTest.kt).
