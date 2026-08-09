# Rally labeling guide

The `/label` route is a local-only annotation workstation. It reads a task JSON and an MP4 through browser file pickers; neither file is uploaded or copied into the web application.

## Required label contract

For every continuous recording, review from the first frame to the last and label every live-play interval:

- **Start:** the instant the server contacts the ball.
- **End:** the first instant live play has ended because the ball is down/out, a fault is complete, or play is stopped.
- Use half-open `[start,end)` timestamps on the exact normalized proxy named by the task.
- Include aces and service faults even when the interval is only a few seconds.
- If an interrupted point is replayed, label each actual serve-to-dead-ball sequence and tag the interrupted one `interrupted-replay`.
- Do not include celebration, ball retrieval, walking to position, scorekeeping, or pre-serve setup.

Aim for boundaries within 0.25 seconds. Use the ±0.1 second controls (`J`/`K`) and slow playback when necessary. If a partial rally at a file edge, camera gap, or occlusion makes the boundary genuinely unresolvable, mark that whole ambiguous sequence as an ignored span instead of guessing. Ignored time is excluded from fitting and evaluation.

For each file, also confirm:

- annotator name;
- players per team;
- target points for that game/set;
- the format or scoring rule when the target alone is ambiguous;
- that the whole continuous video was reviewed.

Target points are metadata, not a request to label the running score. If the target cannot be established from the video or known event format, leave it blank and explain why in annotation notes.

## Optional hard negatives

Ordinary time between rallies is automatically negative, so it does not need manual segmentation. If possible, mark three to five especially confusing dead-time examples per video:

- adjacent-court play;
- foreground player crossing;
- celebration;
- setup between points;
- timeout;
- camera motion;
- warmup.

These tags help construct targeted failure suites. They do not replace rally labels.

Do **not** label player identity, individual touches, ball trajectories, scores per rally, winners, kills, errors, or rotations for this increment. Those would add time without helping the next live/dead sequence model.

## Workstation controls

1. Start the app with `npm run dev` and open `/label` on the printed URL.
2. Load one `*.labels.json` task or previously downloaded draft.
3. Select the matching MP4 shown in the task metadata.
4. Use `S` at serve contact and `E` at end of play.
5. Use `[` then `]` for an ignored span and `H` twice for an optional hard negative.
6. Use `Space` to play/pause, `J`/`K` for ±0.1 seconds, and Shift+`J`/`K` for ±1 second.
7. Download drafts often. Reload a draft through the same task picker to resume.
8. After the entire video is reviewed, check the confirmation box and export completed labels.

The browser downloads `<recording-id>.labels.json`. Move it into the workspace's `labels/pilot/` or `labels/full/` directory rather than modifying the original task.

## CLI validation and manifest creation

Validate one completed document and its exact proxy:

```bash
npm run analyze -- validate-labels --labels /path/to/recording.labels.json
```

Combine every completed document in one task pack into an immutable dataset manifest:

```bash
npm run analyze -- build-manifest \
  --labels-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/labels/pilot \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/pilot-gold.json \
  --name volleycut-pilot-gold-v1
```

Source groups and preassigned splits are retained automatically. The command rejects incomplete documents, video/hash mismatches, overlapping intervals, invalid categories, duplicate recordings, and source groups that cross splits.

## Full-corpus proxy preparation

The current NAS/host produces full annotation proxies at approximately real time. This resumable command needs roughly 2.5 hours for the present corpus and skips every completed proxy/task on rerun:

```bash
npm run analyze -- prepare-labeling-workspace \
  --plan /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-corpus-plan.json \
  --workspace /mnt/freenas/volleycut/labeling-v1-2026-08-09 \
  --fps 30 --max-width 960 --crf 24 --preset ultrafast --threads 1
```

The 960×540 files are seekable annotation proxies. Raw 1080p/4K sources remain immutable and should be used for later learned-frame extraction where feasible.
