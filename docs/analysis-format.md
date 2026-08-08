# Local analysis output

The feasibility pipeline writes one immutable analysis directory per run under `data/analyses/<analysis-id>/`. Generated directories are local artifacts and are intentionally ignored by Git.

```text
data/analyses/<analysis-id>/
  analysis.json
  court-preview.jpg
  proxy.mp4
```

`analysis.json` is the boundary between video processing and the review application. Times are floating-point seconds relative to the normalized proxy.

## Version 1 shape

```json
{
  "schemaVersion": 1,
  "id": "practice-set-20260807-221500",
  "title": "Practice set",
  "createdAt": "2026-08-08T05:15:00Z",
  "source": {
    "filename": "practice-set.mov",
    "duration": 342.1,
    "width": 1920,
    "height": 1080,
    "fps": 59.94,
    "hasAudio": true
  },
  "assets": {
    "proxyUrl": "/api/media/practice-set-20260807-221500/proxy.mp4",
    "courtPreviewUrl": "/api/media/practice-set-20260807-221500/court-preview.jpg"
  },
  "analysis": {
    "method": "court-motion-audio-heuristic-v1",
    "analysisFps": 4,
    "warnings": [],
    "court": {
      "confidence": 0.72,
      "source": "detected-lines",
      "roi": { "x": 0.08, "y": 0.22, "width": 0.84, "height": 0.73 },
      "lines": [{ "x1": 0.1, "y1": 0.7, "x2": 0.9, "y2": 0.7 }]
    }
  },
  "rallies": [
    {
      "id": "R01",
      "start": 24.25,
      "end": 37.5,
      "confidence": 0.76,
      "included": true,
      "evidence": { "motionPeak": 0.88, "audioPeak": 0.64 }
    }
  ],
  "signals": [{ "time": 0, "motion": 0.1, "audio": 0.04, "activity": 0.08 }]
}
```

The detector owns only the objective core interval from estimated serve contact through estimated end of live play. User-selected pre-roll and post-roll remain edit-list settings and never modify these timestamps.

## Compatibility rules

- Consumers must reject unknown `schemaVersion` values.
- New optional keys may be added without changing the version.
- Rally intervals must be ordered, non-negative, non-overlapping, and bounded by `source.duration`.
- Confidence describes the heuristic's certainty, not a calibrated probability.
- The local API exposes only known generated asset names and never exposes source recordings.
