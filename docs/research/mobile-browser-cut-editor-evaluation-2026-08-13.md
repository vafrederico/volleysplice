# Mobile browser cut-editor evaluation — 2026-08-13

## Decision

VolleyCut should not embed a conventional nonlinear editor for the first
correction workflow. The product edits decisions on the original source clock:
users need to see dead-time gaps, find missed rallies, exclude unusable source
sections, retain or remove suggestions, and adjust the retained boundary around
each rally. A conventional output timeline packs retained clips together and
hides the source gaps needed for this review.

The implementation should therefore keep VolleyCut's label and edit-decision
formats canonical and use a mobile-first source-time overview/detail editor.
For the initial UI, operate directly on the cached model or prelabel ranges and
persist a versioned edit draft in browser storage. Treat browser-local rendering
as a later, capability-gated stage.

## Required workflow

The correction surface must let a user:

- keep or remove each inferred range;
- add a missed range by marking its source-time start and end;
- mark source intervals as ignored without converting them into negative
  examples;
- adjust the retained start and end around an individual range without losing
  the original inferred core interval;
- see the whole recording and a precise local view on the same source clock;
- resume an unfinished correction draft on the same device.

On phones, precision cannot depend on dragging a small interval in a compressed
hour-long timeline. The primary interaction is a whole-recording overview plus
a focused 10–30 second detail window. The selected range receives large touch
handles and explicit 0.1-second and 1-second nudge controls. Adding a missed
range uses Mark start / Mark end actions. Keep, remove, and ignore remain
single-tap actions.

## Evaluated open-source options

### Peaks.js

[Peaks.js](https://github.com/bbc/peaks.js) is the strongest existing
interaction reference for the mobile UI. It provides an overview waveform, a
zoomable view, editable segment annotations, and explicit mouse, touch, wheel,
and keyboard support. This overview/detail split matches long source-video
review better than a desktop track stack. It is LGPL-3.0; ongoing development
has moved from the BBC GitHub repository to Codeberg.

Adoption caveat: Peaks.js is waveform-oriented and requires waveform data. The
current VolleyCut cache contains timestamp labels but no guaranteed waveform
artifact. It is therefore a candidate adapter once inference emits lightweight
peaks, not a prerequisite for the first range editor.

### WaveSurfer.js Regions

[WaveSurfer.js](https://github.com/katspaugh/wavesurfer.js) is a mature
BSD-3-Clause alternative. Its Regions plugin supplies draggable and resizable
time ranges. Current releases use pointer events, special handling for coarse
pointers, a short touch delay to distinguish dragging from scrolling, and
multi-touch prevention.

It has the same waveform-data caveat as Peaks.js. Its default resize handles
also require restyling to reach mobile touch-target sizes.

### interact.js

[interact.js](https://interactjs.io/docs/) is the lowest-risk interaction layer
for VolleyCut's existing timeline. The MIT-licensed library normalizes mouse
and touch pointer input and provides dragging, edge resizing, snapping,
constraints, auto-scroll, and multi-touch gestures. It does not provide a media
timeline or edit model.

It remains a reasonable later substitution if native Pointer Events become
difficult to maintain. The first implementation is small enough to use Pointer
Events directly while keeping the state and geometry under VolleyCut's control.

### React Timeline Editor

[React Timeline Editor](https://github.com/xzdarcy/react-timeline-editor) was
initially attractive because it provides React rows, actions, resizing,
constraints, snapping, and custom action rendering. It is not acceptable for a
mobile requirement. An [open issue](https://github.com/xzdarcy/react-timeline-editor/issues/56)
reports that both moving and resizing fail in mobile browsers, and an
[earlier mobile issue](https://github.com/xzdarcy/react-timeline-editor/issues/8)
was closed as not planned.

### FreeCut

[FreeCut](https://github.com/walterlow/freecut) is the most complete open-source
browser editor evaluated. It includes trimming, splitting, cropping,
multi-track editing, local projects, and browser rendering. It is an
application rather than an embeddable SDK, and its documented full workflow
requires a modern Chromium browser, WebGPU, WebCodecs, OPFS, and desktop-style
File System Access.

FreeCut may be useful later as a separately maintained advanced editor or as a
UX/code reference. It is not a dependable phone-browser correction surface,
especially on iOS.

### Elah

[Elah](https://github.com/elahlabs/elah) has an appealing Apache-2.0 React/Next
package structure and implements trim, split, import, project serialization,
preview, and local MP4 export. It is very new, and its documented limitations
include unstable backward scrubbing, stalls on large seeks, in-memory assets,
and audio/export gaps. Mobile behavior is not documented. Long volleyball
recordings make these limitations directly relevant, so it should remain a
prototype candidate rather than the production base.

### OpenCut and Omniclip

The current [OpenCut](https://github.com/OpenCut-app/OpenCut) rewrite lists its
Editor API, plugins, and headless integration as future work. OpenCut Classic is
archived. [Omniclip](https://github.com/omni-media/omniclip) has promising
browser-only WebCodecs and web-component work, but its next major version and
programmatic tooling remain under development. Neither is ready to embed.

### Source-available references

[Twick](https://github.com/ncounterspecialist/twick) is technically capable but
uses the Sustainable Use License, with redistribution, rebranding, and resale
restrictions. [Rescript](https://github.com/wassgha/rescript) closely matches the
desired cut-range UX but now uses the PolyForm Noncommercial license. These are
useful references, not acceptable open-source dependencies for VolleyCut.

## Browser-local processing

[Mediabunny](https://github.com/Vanilagy/mediabunny) is the preferred processing
candidate for a later export spike. It is an MPL-2.0, zero-dependency TypeScript
media toolkit with streaming I/O, MP4 reading and writing, trimming, cropping,
transmuxing, and transcoding. Its APIs are designed for modern browsers rather
than porting FFmpeg into WebAssembly.

[WebAV](https://github.com/WebAV-Tech/WebAV) remains useful on desktop Chromium,
but its documented compatibility target is Chrome/Edge 102+ and Electron. That
is too narrow for a mobile-browser product baseline.

Browser-local export must be capability-gated. Safari added only the video
portion of WebCodecs in 16.4, while complete audio encoding arrived later, and
actual codec support varies by device. The baseline contract is therefore:

1. review, edit, and persist decisions on all supported mobile browsers;
2. offer local export only after codec, storage, and memory qualification;
3. otherwise preserve the edit-decision list for server, native, or later
   desktop export.

## Initial implementation contract

The initial implementation will not add a waveform dependency. It will:

- render the cached analysis labels on a source-time overview;
- show one selected range in a zoomed detail editor;
- keep inferred core bounds immutable while storing optional retained-boundary
  overrides;
- support manual ranges, inclusion decisions, and ignored source spans;
- cache the versioned correction draft by recording and analysis identifier;
- expose large touch targets and precise nudge controls;
- derive the final merged edit list from the local draft without mutating the
  underlying labels.

This preserves a clean adapter boundary: Peaks.js or WaveSurfer can later
replace the overview/detail rendering if waveform artifacts become available,
and Mediabunny can consume the same edit-decision list without changing the
review UI.

## Validation gates

Before enabling browser-local export, test representative long recordings and
50–150 ranges on intended iPhone, iPad, and Android devices. Verify:

- forward and backward seeking;
- touch scrolling versus handle dragging;
- 0.1-second boundary adjustment and range constraints;
- manual add, remove, ignore, undo/reset, and reload recovery;
- memory behavior with full-length source media;
- H.264/AAC input and output capability;
- audio/video synchronization and boundary accuracy across many concatenated
  ranges.

