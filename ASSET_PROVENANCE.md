# Asset and model provenance

This file records the public, non-identifying origin of non-code assets shipped
or committed by VolleySplice. It intentionally omits private recording IDs,
video titles, local paths, and the private source-ID alias map.

## Project-trained models

The JSON bundles in `prod/public/runtime/`, `public/on-device/`, and
`android/app/src/main/assets/` contain models trained by the VolleySplice
project: rally/serve/dead-state heads, false-positive suppression, serving-side,
and side-switch classifiers. They contain learned numeric parameters and
configuration, not source video or audio bytes.

The public source-set names are anonymous aliases. Full training and evaluation
lineage is documented in `MODELS.md`. The project has approved publishing these
trained artifacts under MIT; this does not publish or license the underlying
recordings, labels, or private alias mapping.

No third-party OpenCV Zoo model or third-party pretrained weight file is
committed to public `main` or bundled with the current applications.

## Project WebAssembly

`feature-reductions.wasm` is compiled from the project-authored
`assembly/feature-reductions.ts`. The project source and generated module are
distributed under MIT, while the AssemblyScript toolchain retains its upstream
license.

`libswresample.wasm` is a third-party FFmpeg-derived module. Its origin and
LGPL compliance material are documented in `THIRD_PARTY_NOTICES.md` and
`vendor/libswresample-wasm/README.md`.

## Brand artwork

The VolleySplice wordmark and Play Store feature graphic were generated and
edited for the project using an image-generation tool. Exact prompts and the
mechanical resize steps are recorded in `docs/brand.md` and
`android/play-store/README.md`. Before public release, the project owner must
confirm ownership or authorized reuse of the original VolleyCut logo and symbol
used as edit references. Once confirmed, the resulting repository assets are
published under MIT. Trademark rights, if any, are not granted merely by the
software license.

The corresponding web, Android, iOS, favicon, and Play Store renditions are
copies or mechanical resizes of those project assets.

## Application screenshots

Current Play Store screenshots are captures of the project's Android UI. Their
build identity, dimensions, and SHA-256 hashes are recorded in
`android/play-store/screenshots/capture-manifest.json`. The screenshot build
replaced the decoded source-video layer with black: committed screenshots show
project UI and derived timeline/score state, not video frames.

A visual review of the current phone/tablet capture sources and the historical
`phone-upload-final` set found only project UI, generic media names, scores,
timeline values, and timestamps. No source footage, source title, source ID,
personal name, account identifier, or notification content is visible. The
featured phone images are deterministic compositions of the reviewed phone
captures, project copy, and the project-generated background described below.

Featured screenshots combine those app captures, project-authored copy, and a
project-generated background through
`android/play-store/scripts/compose_featured_screenshots.py`. The generation
prompt is recorded in `android/play-store/README.md`.

The `artifacts/android-youtube-chapters-*.png` documentation captures show only
project UI, generic team/rally labels, scores, and timestamps. A visual review
found no source footage, source title, source ID, personal name, or account ID.

## Third-party assets

The Google Play badge remains a Google brand asset governed by Google's usage
guidelines and is excluded from the MIT license grant. The Android Material
settings icon and copied OpenCV/FFmpeg artifacts retain their upstream licenses.
See `THIRD_PARTY_NOTICES.md`.

## Excluded private material

Raw recordings, audio, extracted frames, private labels, the source-ID alias
map, signing material, and device/account configuration are not public release
assets. Reports containing embedded source frames are treated as media and are
excluded unless separately reviewed and approved.
