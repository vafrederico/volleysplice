# VolleyCut production web app

This directory is a standalone, static, browser-only application. It combines local
video loading, audiovisual feature extraction, rally inference, and the cut editor in
one UI. It has no API routes, server database, media catalog, mounted-media paths, or
upload behavior.

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

- `public/runtime/model-9c92b8e9333f.json`: all three inference heads and decoders.
- `public/runtime/feature-reductions.wasm`: fused visual feature reductions.
- `public/runtime/libswresample.*`: filtered local audio resampling.
- `public/runtime/opencv*.js`: generated from the pinned npm dependency by the
  `prepare-assets` script before development and production builds.
- `licenses/`: redistributed FFmpeg license and wrapper source. These and the OpenCV
  package license are copied into the built site during asset preparation.

No runtime file is fetched from the parent repository or from an external CDN.
