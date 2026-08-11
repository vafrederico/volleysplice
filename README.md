# VolleyCut

VolleyCut is an AI-assisted volleyball video editor. This feasibility-stage implementation turns one local recording into a browser-friendly review proxy, estimates visible court lines, suggests likely activity/rally intervals, and loads them into a human review timeline.

The suggestions are heuristic and require review. They are not yet produced by a trained volleyball classifier and should not be treated as exact serve-contact, end-of-play, or scoring decisions.

## Prerequisites

- Node.js and npm.
- Python 3 with `venv` support.
- FFmpeg and FFprobe, including H.264 encoding support.

## Install

```bash
npm install
npm run analysis:setup
```

The second command creates a local `.venv` and installs NumPy plus headless OpenCV. Both `.venv` and all generated video artifacts are ignored by Git.

Set the durable media location in an ignored `.env.local` file. This machine currently uses `/mnt/freenas/volleycut`:

```bash
cp .env.example .env.local
# Edit VOLLEYCUT_DATA_ROOT in .env.local.
```

The default proxy backend is portable FFmpeg/libx264. On a Linux host with Intel VAAPI and the pinned Jellyfin image already available, `VOLLEYCUT_PROXY_BACKEND=jellyfin-vaapi` enables an opt-in Docker-backed hardware path without changing host group membership. See `.env.example` for the corresponding pinned image setting.

## Analyze a recording

Put a source recording under `$VOLLEYCUT_DATA_ROOT/raw/`, then run:

```bash
npm run analyze -- /mnt/freenas/volleycut/raw/indoor/my-set.mkv
```

Optional arguments include:

```bash
npm run analyze -- /mnt/freenas/volleycut/raw/indoor/my-set.mkv \
  --title "Indoor practice — set 1" \
  --id indoor-practice-set-1 \
  --analysis-fps 4
```

Each run creates an immutable, ignored directory:

```text
$VOLLEYCUT_DATA_ROOT/analyses/<analysis-id>/
  analysis.json
  court-preview.jpg
  proxy.mp4
```

Existing IDs are never overwritten. The app opens the most recently modified valid analysis by default and provides a dataset picker for switching between every valid run. The selected analysis is stored in the URL, so a review can be bookmarked. To try the whole flow without a real recording:

```bash
npm run fixture:analysis
npm run analyze -- /mnt/freenas/volleycut/raw/synthetic/synthetic-two-bursts.mp4
```

Refresh the external dataset inventory at any point with:

```bash
npm run dataset:status
```

The report is written to `$VOLLEYCUT_DATA_ROOT/manifests/status.md` and summarizes raw-download and analysis progress without tracking media in Git.

To analyze every complete source listed in the external `manifests/sources.json`, while safely skipping existing analyses and incomplete downloads:

```bash
npm run analyze:dataset
```

Evaluate generated intervals against a completed label manifest with:

```bash
npm run evaluate:labels -- \
  --labels /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/pilot-gold-v1.json \
  --parameter-search
```

The optional parameter search is diagnostic: it includes a leave-one-source-group-out result, but nine short segments are not enough evidence to change production thresholds without a larger held-out label pack.

`--labels` also accepts a directory of full-video `*.labels.json` documents.

The first pilot evaluation and its prioritized improvement plan are documented in [`docs/research/analysis-vs-pilot-gold-2026-08-09.md`](docs/research/analysis-vs-pilot-gold-2026-08-09.md).

After changing court validation or rally decoding, create immutable analyses from the existing proxies without transcoding the videos again:

```bash
npm run reanalyze:dataset
```

This writes suffixed analysis directories, hard-links their proxies when supported, and recomputes motion only when the validated court region changed. The full-video ground-truth comparison that selected analyzer v2 is documented in [`docs/research/full-video-ground-truth-analyzer-v2-2026-08-10.md`](docs/research/full-video-ground-truth-analyzer-v2-2026-08-10.md).

## Review locally

```bash
npm run dev -- --hostname 0.0.0.0
```

Open `http://<host-lan-ip>:3000` from another machine. The Next.js development allowlist automatically includes this host's active non-loopback IPv4 interfaces so dev chunks and HMR work over the LAN. If you use a custom DNS name or reverse proxy, add its hostname (without a scheme or port) to the optional comma-separated `VOLLEYCUT_DEV_ORIGINS` setting.

The review screen provides:

- A dataset picker for every valid local analysis.
- Proxy video playback and byte-range seeking.
- Suggested activity intervals with uncalibrated confidence.
- Detected court-line overlays and a court diagnostic image.
- Include/exclude decisions.
- Configurable pre-roll and post-roll.
- A merged edit decision list, so overlapping padding is counted only once.
- Explicit warnings for missing audio, fallback court regions, low camera stability, and zero detected rallies.

Generated videos are served only through an allowlisted local media route. Original source recordings are never exposed by that route.

## What the analyzer currently does

1. Uses FFprobe to inspect the source.
2. Uses FFmpeg to normalize rotation, timestamps, frame rate, color format, and audio into a low-resolution H.264 proxy.
3. Builds a temporal-median representative frame and uses edge plus Hough-line detection to estimate a court activity region.
4. Samples proxy frames at 4 fps, compensates small camera translations, and measures motion inside that region.
5. Measures mono audio energy when audio is available.
6. Combines normalized motion and supporting audio into activity scores.
7. Applies hysteresis, bridges at most 0.75 seconds of inactivity, filters short noise, and adds no automatic ending tail before producing review candidates.

The line detector currently estimates a rectangular activity region from stable line segments; it does not yet solve full court calibration or distinguish court lines from every similar gym marking. Severely bottom-cropped estimates fall back to a broad activity region. Audio can support a visual candidate but cannot create one by itself.

## Tests

```bash
npm test
npm run lint
npm run build
```

The tests cover analysis catalog loading, interval merging and clamping, timeline formatting, synthetic activity segmentation, signal normalization, and detected/fallback court regions. The generated fixture provides a codec-level smoke test for the full pipeline.

## Project layout

- `analysis/` — Python/FFmpeg feasibility pipeline and tests.
- `app/` — Next.js application and private local media route.
- `components/` — interactive review editor.
- `lib/` — analysis validation and edit decision list calculations.
- `$VOLLEYCUT_DATA_ROOT/raw/` — external source recordings organized by surface.
- `$VOLLEYCUT_DATA_ROOT/analyses/` — external proxies, JSON results, and diagnostics.
- `data/` — ignored fallback storage when `VOLLEYCUT_DATA_ROOT` is unset.
- `docs/research/` — retained feasibility and implementation research.
- `docs/analysis-format.md` — versioned processing/UI contract.

## Current limitations

- Supported footage is still intended to be stationary, landscape, full-court video from behind an end line.
- Motion is not equivalent to live play; warmups, celebrations, walking, neighboring courts, and camera movement can create false candidates.
- Quiet or visually subtle rallies can be missed.
- Grass, beach ropes, cropped views, and dense indoor floor markings can cause the court detector to use the fallback region.
- YouTube downloading, upload UI, background jobs, rendered exports, model training, player tracking, and action labels are not implemented yet.
- Corrections currently live only in browser state; persistent edit revisions are a later milestone.

The product should favor retaining extra footage over deleting live play. Always inspect every proposed boundary before export work is added.
