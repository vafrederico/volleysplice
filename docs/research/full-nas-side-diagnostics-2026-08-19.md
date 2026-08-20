# Full-NAS serving-side and side-switch diagnostics — 2026-08-19

## Purpose

This run extends the first existing-label heuristics beyond the nine completed
training recordings. The goal is to inspect transfer on the rest of the NAS
corpus and to compare future feature/model iterations on the same immutable
video inventory.

The source directory mentioned during planning is named
`/mnt/freenas/volleycut-raw-no-backup` on this NAS (singular). The manifest also
includes the completed and intake proxy recordings so the reports cover every
deduplicated video source currently available to these diagnostics.

## Corpus contract

The reproducible manifest is:

```text
/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-nas-video-corpus-v1.json
```

It contains 30 recordings and 1,424 rally candidates:

| Source | Recordings | Target status | Use |
| --- | ---: | --- | --- |
| `completed/full-v1` | 9 | `gold` | Existing completed labels and side-switch markers |
| `intake/labels/full` | 8 | `reviewed-draft` | Human-reviewed continuous drafts, not final gold |
| `intake/blind-sol/prelabels` | 2 | `candidate-only` | Unvalidated blind candidates missing a human draft |
| `volleycut-raw-no-backup` model feedback | 11 | `candidate-only` | Non-training MP4s with model-derived rally windows |
| **Total** | **30** |  | **1,424 rally windows** |

The builder deduplicates by recording ID, preferring completed labels over
intake drafts, drafts over blind prelabels, and any label source over the raw
no-backup namespace. Candidate-only rows are deliberately retained for review
but are excluded from gold/diagnostic accuracy denominators.

The raw no-backup videos do not have human rally or side-switch labels. Their
`initialInference.ranges` from the matching `.model-feedback.json` are used as
serve/rally candidates. Side-switch rows for those recordings are gaps between
candidate rallies, marked `label: null`; they are not no-switch negatives.

## Variants run

### Serving side

Every rally candidate is sampled around its existing serve anchor using the
current fixed protocol:

- pre-serve offsets: `-1.25`, `-0.75`, `-0.35` seconds;
- action offsets: `+0.05`, `+0.30`, `+0.55` seconds;
- signed near-minus-far evidence after the recording ROI is applied;
- upper/background ROI half = `far`, lower/foreground ROI half = `near`;
- full-half and outer 35% baseline-band motion, HSV palette change, and OpenCV
  HOG proposal-area variants; and
- a diagnostic abstention margin of `0.05`.

The current labels have no structured serving-side field. The report therefore
uses only strict near/far serving phrases in human notes as weak targets. Rows
without an unambiguous phrase remain unknown. Candidate-only rows never become
targets merely because an unvalidated note happens to contain a matching word.

### Side switching

For completed and reviewed-draft labels, positives come from `sideSwitches` and
controls come from unmarked inter-rally gaps of at least eight seconds. For
candidate-only recordings, every eligible gap is a review candidate with no
binary target. The existing OpenCV HOG person proposal and player-palette
features are run on two frames per side of each gap, with an eight-second flank
and 0.75-second edge margin.

## Reports and review UI

The completed report paths are:

```text
/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-existing-label-variants-full-nas-v1.json
/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/appearance-diagnostic-full-nas-v1.json
```

The development app exposes both report queues:

- `/serving-side-review` — near/far/unclear review for every rally candidate;
- `/side-switch-review` — switch/no-switch/unclear review for labeled controls
  and candidate-only gaps.

Both queues stream video through a report-indexed development media route, so
raw no-backup MP4s do not need to be registered as prepared labeling tasks.
Decisions are debounced and atomically saved to NAS review files; they do not
mutate labels or research reports.

## Labels needed next

For useful accuracy claims, add a per-rally field beside `start`/`end`:

```json
{
  "servingCourtSide": "near|far|unknown",
  "servingConfidence": 0.0,
  "servingObservability": "visible|partially-visible|offscreen|ambiguous"
}
```

For side switching, add marker confidence/observability, approximate transition
intervals, and hard no-switch categories such as timeout, huddle, celebration,
and ball retrieval. A small subset should also receive person boxes/footpoints,
anonymous team IDs, court/net anchors, and near/far physical side labels. This
would allow the color-mass idea to be tested as team association instead of
scene-change correlation.

The score tracker additionally needs an initial near/far team mapping, initial
score, format/scoring rule, and an explicit recovery state when a serving-side
or side-switch decision is unknown.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python scripts/build-full-nas-corpus.py \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-nas-video-corpus-v1.json

PYTHONPATH=. .venv/bin/python scripts/evaluate-side-switch-appearance.py \
  --corpus-manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-nas-video-corpus-v1.json \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/appearance-diagnostic-full-nas-v1.json \
  --environments beach grass indoor unknown --samples-per-side 2 --minimum-gap-seconds 8

PYTHONPATH=. .venv/bin/python scripts/evaluate-serving-side-existing-labels.py \
  --corpus-manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-nas-video-corpus-v1.json \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-existing-label-variants-full-nas-v1.json \
  --environments beach grass indoor unknown
```

Future iterations should build a new manifest/report filename, retain the
candidate source model IDs in `candidateSource`, and compare the same source
scope and target-status filters. They should not overwrite this report or
promote candidate-only review decisions to gold labels automatically.

