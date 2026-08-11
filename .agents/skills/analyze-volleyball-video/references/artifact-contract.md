# Candidate artifact contract

## Paths

For the current workspace:

```text
Candidates: /mnt/freenas/volleycut/direct-sol-prelabels-2026-08-10/results/<recording-id>.candidates.json
Tasks:      /mnt/freenas/volleycut/labeling-v1-2026-08-09/tasks/full/
Prelabels:  /mnt/freenas/volleycut/labeling-v1-2026-08-09/prelabels/sol-xhigh/
```

Use the full MP4 basename without `.mp4` as `recordingId`. For `grass-match-full.mp4`, use `grass-match-full`, not `grass-match`.

## JSON schema

```json
{
  "schemaVersion": 1,
  "recordingId": "grass-match-full",
  "videoPath": "/absolute/path/grass-match-full.mp4",
  "durationSeconds": 100.0,
  "analysisMethod": "blind-gpt-5.6-sol-xhigh-audiovisual",
  "events": [
    {
      "serveContact": 10.5,
      "rallyEnd": 18.0,
      "serveConfidence": "medium",
      "endConfidence": "high",
      "notes": "Serve is audio-anchored; collective stand-down makes the end clear."
    }
  ],
  "ambiguities": ["Distant server is outside the frame for several points."],
  "analyzedAt": "2026-08-10T19:00:00Z"
}
```

Requirements:

- `events` is chronological and non-overlapping. It may be empty only when the complete blind pass found no plausible rally; explain that in `ambiguities`.
- All times are finite and satisfy `0 <= serveContact < rallyEnd <= durationSeconds`.
- Confidence is exactly `high`, `medium`, or `low`.
- Notes describe evidence or uncertainty rather than asserting correctness.
- `videoPath` and duration must match the prepared task exactly.

## Import

```bash
npm run analyze -- import-model-prelabels \
  --candidates-dir /mnt/freenas/volleycut/direct-sol-prelabels-2026-08-10/results \
  --tasks-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/tasks/full \
  --output-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/prelabels/sol-xhigh
```

The importer rejects identity/duration mismatches, invalid values, overlaps, and attempts to overwrite a changed prelabel. Human drafts under `labels/full/` always take precedence over these isolated prelabels.

## Report

For each batch, record:

- event count and duration per video;
- evidence preparation used;
- focus, framing, occlusion, audio, or recording-edge problems;
- timestamps that deserve priority human review;
- actual positive, negative, and disambiguation cues;
- structural validation performed.
