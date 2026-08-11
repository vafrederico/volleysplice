# Rally labeling guide

The `/label` route is a local-only annotation workstation. Prepared tasks and their MP4 proxies stream from the NAS through the local application; fallback browser file pickers remain available. Nothing is uploaded to a cloud service or copied into the web application.

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

1. Start the app with `npm run dev -- --hostname 0.0.0.0` and open `/label` on the printed LAN URL.
2. Choose the **Full corpus** or **Pilot** batch, then choose a prepared task. The app loads both its task JSON and matching NAS proxy. Ready and saved counts appear in the batch selector; the catalog refreshes while full proxies are being prepared.
   When an isolated GPT-5.6 Sol prelabel exists and no human draft has been saved yet, the task picker reports its AI rally count and loads it as an explicitly unvalidated starting point. AI rows are marked `AI`; inspect and correct both boundaries rather than accepting them as ground truth.
3. To resume a downloaded draft, use the two local fallback pickers for the draft and its matching MP4.
4. Use `S` at serve contact and `E` at end of play. The rally strictly containing the playhead is highlighted in both the timeline and rally table. Within that rally, `S` moves its start and `E` moves its end to the playhead. In dead time after a rally, `E` extends that immediately preceding rally to the playhead; the editor rejects an extension that would overlap the next rally, an ignored span, or a hard negative.
5. Delete the highlighted rally with **Delete selected**, `Delete`, or `Backspace`. Keyboard deletion is disabled while focus is in a form control.
6. If teams switch court sides in this video or format, press `X` at the switch. Side switches are point markers rather than rally intervals; their time and optional note can be edited or deleted in the **Side switches** section.
7. Use `[` then `]` for an ignored span and `H` twice for an optional hard negative.
8. Use `Space` to play/pause, `J`/`K` for ±0.1 seconds, and either Shift+`J`/`K` or `←`/`→` for ±1 second. The vertical line on the label timeline tracks the current video time.
9. Use **Save draft to NAS** often. Selecting that prepared task later resumes the saved draft automatically; downloading a backup JSON is optional. Independently, the browser records the last prepared task and playhead time in versioned local storage during playback and seeking. Reloading `/label` reopens that prepared task at the saved time; local fallback files are never stored.
10. After the entire video is reviewed, check the confirmation box and export completed labels.

Direct drafts are written atomically to `labels/full/<recording-id>.labels.json` or `labels/pilot/<recording-id>.labels.json`, according to the selected batch. The browser can also download the same naming format for offline backups. Do not modify the originals under `tasks/`.

Model-generated candidates are kept separately under `prelabels/sol-xhigh/`. The editor only uses one when a human draft does not exist. The first **Save draft to NAS** writes a human draft under `labels/full/`, after which that draft always takes precedence. Candidate provenance, confidence tags, and ambiguity notes remain in the saved document so that later evaluation can distinguish assisted labels from independently created labels.

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
  --fps 30 --max-width 960 --crf 24 --preset ultrafast --threads 2
```

The width-limited, constant-frame-rate files are seekable annotation proxies. Each completed proxy and task appears automatically in the **Full corpus** picker. Raw 1080p/4K sources remain immutable and should be used for later learned-frame extraction where feasible.
