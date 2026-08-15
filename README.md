# VolleyCut

VolleyCut is an AI-assisted volleyball video editor. This feasibility-stage implementation supports a no-model audiovisual heuristic, trained rally classifiers, blind Sol prelabels, a shared comparison UI, and a dedicated gold-label workstation.

Every suggestion source requires review. Heuristic and trained-model confidence values are not calibrated probabilities and should not be treated as exact serve-contact, end-of-play, or scoring decisions.

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

## Analyze a recording with the no-model heuristic

Put a source recording under `$VOLLEYCUT_DATA_ROOT/raw/`, then run:

```bash
npm run analyze-no-model -- /mnt/freenas/volleycut/raw/indoor/my-set.mkv
```

Optional arguments include:

```bash
npm run analyze-no-model -- /mnt/freenas/volleycut/raw/indoor/my-set.mkv \
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
npm run analyze-no-model -- /mnt/freenas/volleycut/raw/synthetic/synthetic-two-bursts.mp4
```

Refresh the external dataset inventory at any point with:

```bash
npm run dataset:status
```

The report is written to `$VOLLEYCUT_DATA_ROOT/manifests/status.md` and summarizes raw-download and analysis progress without tracking media in Git.

To analyze every complete source listed in the external `manifests/sources.json`, while safely skipping existing analyses and incomplete downloads:

```bash
npm run analyze-no-model:dataset
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
npm run reanalyze-no-model:dataset
```

This writes suffixed analysis directories, hard-links their proxies when supported, and recomputes motion only when the validated court region changed. The full-video ground-truth comparison that selected analyzer v2 is documented in [`docs/research/full-video-ground-truth-analyzer-v2-2026-08-10.md`](docs/research/full-video-ground-truth-analyzer-v2-2026-08-10.md).

Run a saved learned model over every full-video manifest entry and evaluate the practical
0–3 second crop-padding tradeoff with:

```bash
npm run infer:model-dataset -- \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-percentile-v1

# Materialize the frozen v4+v5 fusion and all persisted specialist iterations.
npm run infer:dual-serve-fusion-dataset
npm run infer:specialist-model-dataset

npm run evaluate:model-padding -- \
  --model-version full-percentile-v1 \
  --padding-seconds 0 1 2 3 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-model-padding-v1.json
```

Padding is applied only to the hypothetical exported crops. The report preserves the core
model intervals, merges overlapping padded crops, and separates train, validation/tuning,
and held-out test results.

Rally-model iterations are ranked by **Padded P/Core R F1**
(`F1_padP_coreR`): precision compares the padded model export with equally padded human
labels, while recall compares that same padded model export with the core human labels.
See the [model iteration ranking metric](docs/model-ranking-metric.md) for the exact
interval, aggregation, and split-discipline contract. This is not chronological event F1.
Every iteration report includes the four symmetric before/after padding cases of
0, 1, 2, and 3 seconds; the primary rank uses the predeclared target-padding case.

## Review locally

```bash
npm run dev -- --hostname 0.0.0.0
```

Open `http://<host-lan-ip>:3000` from another machine. The Next.js development allowlist automatically includes this host's active non-loopback IPv4 interfaces so dev chunks and HMR work over the LAN. If you use a custom DNS name or reverse proxy, add its hostname (without a scheme or port) to the optional comma-separated `VOLLEYCUT_DEV_ORIGINS` setting.

The review screen provides:

- A video picker for the nine prepared full recordings.
- A per-video source picker for human reference labels, blind Sol prelabels,
  no-model heuristic versions, and every available trained-model inference.
- Stacked timelines on one video clock, including the labeling workstation's
  vertical playhead and click-to-seek behavior.
- Explicit training, validation/tuning, and evaluation-only badges for each
  trained model and recording.
- Proxy video playback and byte-range seeking.
- Suggested activity intervals with uncalibrated confidence.
- Detected court-line overlays and a court diagnostic image.
- Include/exclude decisions.
- Configurable pre-roll and post-roll.
- A merged edit decision list, so overlapping padding is counted only once.
- Per-model core, padded, and Padded P/Core R F1 metrics plus export-duration cost.
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

Open `/label` for the local gold-label workstation. Its batch-aware selector loads full-corpus or pilot tasks, streams their exact NAS proxies, seeds fresh full-corpus tasks from the production model, shows blind Sol labels on a separate read-only reference timeline, displays ready/saved progress, and resumes atomically saved drafts; local file pickers remain available as a fallback. It supports precise rally/ignored/hard-negative intervals and exports resumable or completed labels. See [`docs/labeling-guide.md`](docs/labeling-guide.md).

## Rally-analysis baseline

The worktree now includes a CPU-only v0 that can normalize and validate recordings, train a temporal rally classifier, infer rally intervals into `analysis.json`, and evaluate an untouched test split. It can be exercised before real video arrives with:

```bash
npm run analysis:setup
npm run test:analysis
.venv/bin/python -m analysis smoke
```

See [`analysis/README.md`](analysis/README.md) for the annotation contract and end-to-end commands. Research, licensing, camera-fit findings, and adoption decisions are indexed under [`docs/research/`](docs/research/).

Run one saved model across every video in a manifest with the resumable dataset command:

```bash
npm run infer:model-dataset -- \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-percentile-v1
```

Completed immutable runs are skipped, incomplete destinations are rejected, and new results
appear in the comparison UI automatically.

## Project layout

- `analysis/` — Python/FFmpeg feasibility pipeline and tests.
- `app/` — Next.js application and private local media route.
- `components/` — interactive review editor.
- `lib/` — analysis validation, local catalogs, and edit-decision calculations.
- `$VOLLEYCUT_DATA_ROOT/raw/` — external source recordings organized by surface.
- `$VOLLEYCUT_DATA_ROOT/analyses/` — immutable heuristic and trained-model inference runs.
- `$VOLLEYCUT_LABELING_WORKSPACE/` — proxies, Sol candidates, labels, manifests, models, and reports.
- `data/` — ignored fallback storage when external roots are unset.
- `docs/research/` — retained feasibility, model, and source-reuse research.
- `docs/labeling-guide.md` — exact boundary policy, keyboard workflow, and label validation/import.
- `docs/analysis-format.md` — versioned processing/UI contract.

## Current limitations

- Supported footage is still intended to be stationary, landscape, full-court video from behind an end line.
- Motion is not equivalent to live play; warmups, celebrations, walking, neighboring courts, and camera movement can create false candidates.
- Quiet or visually subtle rallies can be missed.
- Grass, beach ropes, cropped views, and dense indoor floor markings can cause the court detector to use the fallback region.
- YouTube downloading, upload UI, background jobs, rendered exports, player tracking, and action labels are not implemented yet.
- Corrections currently live only in browser state; persistent edit revisions are a later milestone.

The product should favor retaining extra footage over deleting live play. Always inspect every proposed boundary before export work is added.
