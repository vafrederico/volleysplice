# VolleyCut production web app

This directory is a standalone, static, browser-only application. It combines local
video loading, audiovisual feature extraction, rally inference, and the cut editor in
one UI. It has no API routes, server database, media catalog, mounted-media paths, or
upload behavior.

Before queueing inference, the user can seek the local preview and mark the game start
and end. The marked source-time window becomes part of the project and feature-cache
identity. Video and audio decoding only generate features inside that window, inference
is clipped to it, editor padding cannot cross it, and the editor overview shows the game
window instead of unused pre-game or post-game footage. Existing saved projects without
explicit bounds migrate to the full source duration.

The selected video stays in the browser. Each inference request creates a durable local
project, and the top-bar project selector returns to completed inputs without rerunning
the model. Multiple projects can be queued: one generates features at a time while any
completed project remains available in the editor. OpenCV runs in a dedicated worker,
feature reductions and FFmpeg-compatible audio resampling run through bundled WebAssembly,
and the model runs on CPU. Visual checkpoints, completed audio features, project metadata,
and finalized inference results use local IndexedDB; edit drafts use local storage. Source
video bytes are not copied into project storage, so a browser restart only requires
reconnecting the exact local file for playback or export—not rerunning inference. Deleting
a project also deletes its cached features, inference, and edit draft. The editor exports
its final padded and corrected intervals as an MP4 at the
source dimensions, encoded directly from the original local video into a user-selected
file or origin-private storage before iOS sharing. JSON edit-list export is also available.
For model improvement, the editor can additionally download a versioned model-feedback JSON
containing the retained 90-column base feature matrix, source timestamps, probability traces,
untouched initial inference ranges, corrected ranges, explicit false-positive and false-negative
labels, ignored intervals, and finalized export ranges. Video bytes are never included. New
projects retain a sampled source fingerprint, so an identical raw video can be reconnected after
renaming or transfer and aligned using the bundle's source-relative timestamps. See
[`docs/model-feedback-bundle.md`](docs/model-feedback-bundle.md) for the format contract.
Padded ranges separated by less than the configurable join-gap threshold are exported as
one continuous section. The default is 3 seconds, a gap of exactly 3 seconds remains a
cut, and retained join gaps are shown in light gray on the overview rail.
Both analysis and MP4 export show live elapsed-time and estimated-time-remaining counters
while they run.

## Run locally

Requirements: Node.js 24 and a current Chrome or Edge browser, or Safari 26 on iOS/macOS.

```sh
npm install
npm run dev
```

Open the printed localhost URL. Media analysis requires a secure browser context;
localhost is trusted for development, and deployed environments must use HTTPS.
For LAN-device testing, Vite also accepts `VOLLEYCUT_DEV_HTTPS_KEY` and
`VOLLEYCUT_DEV_HTTPS_CERT` paths containing a locally trusted key and certificate.

## Build static files

```sh
npm ci
npm run build
```

Serve `dist/` from any static HTTPS host. The build uses relative asset URLs, so it
works at a domain root or a subpath such as GitHub Pages without changing the config.
Do not open `dist/index.html` directly through `file://`; WebCodecs and module workers
need an HTTP origin.

MP4 export uses direct File System Access when available. On iOS it streams the output
into origin-private file storage without holding the complete video in JavaScript memory,
then presents a separate Share or Save action so Safari has fresh user activation for its
native share sheet. Exact boundaries require a single AVC/AAC transcode; the output keeps
the source display dimensions and uses the very-high-quality encoder preset.

The included `.github/workflows/deploy-pages.yml` is ready when this directory is used
as a repository root. In a monorepo, copy the workflow to the repository-level
`.github/workflows/` directory and set both `npm` steps' `working-directory` to `prod`
and the artifact path to `prod/dist`.

## Docker

```sh
docker compose config --quiet
docker compose up -d --build
```

The included Compose file joins the existing external `websecure` network and exposes
the nginx service through Traefik at `https://volleycut.vafrederico.com`. The route is
intentionally public: it has compression and Let's Encrypt labels, but no Authentik
labels, callback router, ForwardAuth middleware, or published host port. TLS terminates
at Traefik.

The image is a multi-stage build: Node compiles the static site, and nginx serves only
the resulting files on container port 8080. After deployment, verify that an
unauthenticated request returns the app rather than redirecting to a login page:

```sh
curl -I https://volleycut.vafrederico.com
```

## Runtime contents

- `public/downloads/VolleyCut-v0.10.1-arm64-release-signed.apk`: the signed
  arm64 Android release offered from the production header. The production
  build verifies its SHA-256 digest before it can be deployed.
- `public/runtime/model-1ca43e38eefc.json`: the promoted all-labels v2 inference heads and decoders.
- `public/runtime/model-9c92b8e9333f.json`: the previous production heads used by the two-model consensus pass.
- `public/runtime/suppression-39eddf581639.json`: the held corrected suppression
  specialist. Its neighboring manifest records the source artifact, weights, decoder,
  and emitted asset hashes.

The production app extracts media features once, runs both model stacks, and unions
overlapping rally ranges. Ranges emitted by only one model are retained but marked as
disagreements, assigned a conservative review confidence below 50%, and highlighted
with an orange striped treatment so **Review next** visits them before export.

Feature caches are versioned by the feature schema, extraction settings, source,
media metadata, ROI, and runtime variant—not by the model—so compatible features can
be reused across model upgrades. Persisted inference is separately keyed by an
ensemble identity containing both complete browser-bundle SHA-256 digests and the merge
algorithm version. A stale single-model or older-ensemble result is changed to **Needs
source** on load and cannot be presented as current production inference.

The optional suppression review runs on the same cached contextual features. **No
suppression** is the default and preserves the existing export. Conservative,
Balanced, and Aggressive choices differ only in how one-model-only production
components become suggestions; every removal remains visible and reversible in the
editor. Regenerate the checked-in browser artifact deterministically with:

```sh
python scripts/export-browser-suppression-model.py \
  /path/to/suppression-overlap-exclusion-retrained \
  public/runtime/suppression-39eddf581639.json \
  public/runtime/suppression-39eddf581639.manifest.json
```

`npm run preview` serves the verified production build on `0.0.0.0:3000`, matching
the local Traefik target for `https://internal.example`.
- `public/runtime/feature-reductions.wasm`: fused visual feature reductions.
- `public/runtime/libswresample.*`: filtered local audio resampling.
- `public/runtime/opencv*.js`: generated from the pinned npm dependency by the
  `prepare-assets` script before development and production builds.
- `licenses/`: redistributed FFmpeg license and wrapper source. These and the OpenCV
  package license are copied into the built site during asset preparation.

No runtime file is fetched from the parent repository or from an external CDN.
