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

Existing IDs are never overwritten. The app opens the most recently modified valid analysis. To try the whole flow without a real recording:

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

## Review locally

```bash
npm run dev
```

The review screen provides:

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
7. Applies conservative hysteresis, gap filling, duration filters, and bounded confidence to produce review candidates.

The line detector currently estimates a rectangular activity region from stable line segments; it does not yet solve full court calibration or distinguish court lines from every similar gym marking. Audio can support a visual candidate but cannot create one by itself.

## Tests

```bash
npm test
npm run lint
npm run build
```

The tests cover interval merging and clamping, timeline formatting, synthetic activity segmentation, signal normalization, and detected/fallback court regions. The generated fixture provides a codec-level smoke test for the full pipeline.

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
