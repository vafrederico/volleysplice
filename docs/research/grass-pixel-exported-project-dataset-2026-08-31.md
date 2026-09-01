# Grass Pixel exported-project dataset — 2026-08-31

## Outcome

Seven already-reviewed grass recordings from `/mnt/freenas/volleycut-raw-no-backup`
were imported as the durable source group `grass-pixel-20260829`. No LLM labeling
was run. The import preserves four distinct layers instead of treating every edited
cut fragment as one frame-exact rally:

- retained export coverage;
- human score-tracking serve events;
- human side-switch markers; and
- ignored intervals.

The immutable import is at
`/mnt/freenas/volleycut/exported-project-datasets/grass-pixel-20260829/dataset.json`.
It is registered as `challenge`, `grass`, and `reviewed-export-coverage` so it can be
referenced by corpus-building and future evaluation workflows without silently
promoting weak boundaries to gold labels.

## Imported annotations

| Item | Count |
| --- | ---: |
| Recordings | 7 |
| Raw retained edited ranges | 279 |
| Usable non-micro coverage ranges | 275 |
| Serve events | 263 |
| Side switches | 33 |
| Ignored intervals | 3 |
| Micro-range shortening artifacts | 4 |
| Serve timestamps realigned to retained coverage | 15 |
| Unassociated non-micro coverage ranges | 0 |

Realigned serve events retain both `rawTime` and the normalized `time`. This makes
the correction reversible and preserves evidence from the original export. A joined
coverage range may contain more than one serve, and a serve may cover adjacent export
fragments; the importer therefore does not invent one serve per corrected range.

## Missing exported features

All seven feedback bundles have `features: null`. Their warnings identify the old
import/export cache-loss bug, while retained ranges, corrections, serving-side
features, serve markers, and side switches remain present.

The repository audit compared `cc44ca718928d9f65a8741c0bd1843617192ce31` with
`34b4cb8f3b275e9bd7d56a5a45e7e0a2e188ea40`. The only relevant later behavior
change is `10478aed9d4194af9eaaba0497fc770e8b43cf78`, which hydrates Android's native
feature cache when an imported feedback bundle already contains complete features.
It does not regenerate absent features. The committed Python feature implementation,
Android feature schema/math/engine, and the two production runtime JSON files did not
change in that range. Consequently, deterministic feature regeneration is still
required for this batch, but no earlier regenerated feature artifact is invalidated
by the post-`cc44ca7` commits.

## Regeneration execution contract

`scripts/regenerate-exported-project-inference.py` now:

- uses one feature process on hosts with at most eight logical CPUs, otherwise up
  to logical CPUs minus two processes capped by recording count (seven workers,
  three OpenCV threads each, on this 24-thread host and seven-recording batch);
- probes a real source frame and uses FFmpeg CUDA/NVDEC when available;
- uses `ffprobe`, not an OpenCV decode, for metadata on the NVDEC path; every
  recording owns a fresh FFmpeg/CUDA process and per-recording feature state;
- preserves the historical nearest-frame sampling timeline and BT.601 conversion;
- gives NVDEC and OpenCV caches distinct versioned identities;
- validates feature version/configuration, runtime hashes, cache hashes, and trace
hashes before reusing checkpoints; and
- defaults to the imported file size plus the production `sampled-sha256-v1`
  fingerprint for replay verification.

The sampled fingerprint reads only the first and last 1 MiB of each source. The full
SHA-256 was computed once during import and remains the content/cache identity.
`--source-verification full` is available for an intentional deep audit; it re-reads
all source bytes and is not the normal replay path.

NVENC is not involved. NVENC is NVIDIA's encoder, while this workload decodes input.
The original regeneration used OpenCV's CPU FFmpeg decoder with hardware acceleration
reported as disabled. The canonical regenerated run uses NVDEC via
`-hwaccel cuda -hwaccel_output_format cuda`. Audio is decoded separately by FFmpeg on
CPU, and no video is transcoded. If the NVDEC probe fails and automatic mode falls
back to OpenCV video decoding, recordings run serially so CPU decoders do not compete
with one another.

The completed CPU/OpenCV replay was preserved at
`regenerated-inference-opencv-v1/`. The canonical NVDEC replay and its separately
keyed feature cache are:

- `regenerated-inference/index.json`
- `features/audiovisual-audio-normalized-v3-nvdec-v1/`

Both frozen production runtimes are replayed. Their SHA-256 values are:

- all-labels-v2: `d2c2c11e8fed8b6c6ad77d244b613e81d5bab101939a8f57be5166b45ebca78f`
- previous-production: `d8cc42f70bc10576a5e03251b05981ceeee1a61a15c61cc5dfb68dd631e6f90d`

The reusable full-NAS manifest was published at
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-nas-video-corpus-v3.json`.
It contains 37 deduplicated recordings, including these seven reviewed-export rows
with 275 usable coverage ranges, 263 serve markers, 33 side switches, explicit
`llmLabelingUsed: false`, and links to their regenerated inference artifacts.

Final artifact SHA-256 values:

- imported dataset: `df4bfd8e450e87e145bdf41e6e2a5e9cefe209152be9c82b954b693c8ad85269`
- regenerated NVDEC index: `77d9da92c2e3277945dab7a6c2e1f88fb5b00369ae34c7b104fa94613eb10c1d`
- full-NAS corpus v3: `167bd600f404db0d36d137901fc2f507a9861bbd2aa892d6536f5f5bb1a3074b`
