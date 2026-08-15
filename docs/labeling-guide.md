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

## Court-geometry clicks

For the court-relative feature experiment, pause on one clear frame with the full playing area visible and use the **Court geometry · 4–8 clicks** strip below the video. “Near” always means the camera side, independent of which team is serving.

The minimum useful geometry is four named court corners:

- near-left;
- near-right;
- far-left;
- far-right.

When visible, also click the left and right net anchors and one representative service-zone anchor on each of the near and far sides. This produces six or eight anchors. A draft may be saved partway through, but a document that contains partial geometry cannot be completed until all four corners and both points of every started optional pair are present. Coordinates are stored as normalized frame coordinates, so they remain valid across proxy and source resolutions. Do not guess through a crop or obstruction; leave an optional pair absent and explain it in annotation notes.

## Per-rally transition cues

After correcting a rally boundary, use the **Optional transition cues** panel on the current-rally card. These sparse labels support the serve-anchored multi-state and end-state experiments:

- **Receiver reaction time:** the first coordinated receiving-side response to the serve, as an absolute video timestamp no more than five seconds after rally start.
- **Collective stand-down time:** the first clear collective shift out of live-play posture, as an absolute timestamp within five seconds of rally end.
- **Terminal cue:** `ball-down-or-out`, `whistle-or-stoppage`, `no-recovery`, or `unobservable`.
- **End observability:** `observable`, `partially-observable`, or `unobservable`.
- **Start confidence / end confidence:** independent values from 0 (guess) to 1 (frame-clear).
- **Immediate result verified:** yes for a serve whose outcome is immediate (for example, an ace or completed service fault), no when a continued exchange is visibly verified, and unlabeled when it cannot be established.

Transition fields are optional and legacy label documents remain valid. The first readiness target is five fully cued rallies per recording. Within that small pilot, cover both immediate-result strata (aces and service faults) when available, ordinary multi-touch rallies, and at least one uncertain or partly unobservable endpoint rather than annotating only the easiest examples.

## Sparse anonymous player tracklets

The optional player-tracklet panel prepares the step-6 player-motion experiment without requiring these labels to finish ordinary rally annotation. Start with a small, stratified pilot of five rallies per recording: include ordinary rallies, an ace or service fault when available, and at least one visually difficult example. For each chosen rally, cover both the `serve` and `rally-end` windows with at least two visible players and at least two frames per short track.

- Choose the rally and boundary window explicitly. Serve observations may span two seconds before through three seconds after serve contact. Rally-end observations may span three seconds before through two seconds after the rally end.
- Use short temporary IDs such as `P1` and `P2`. An ID is only consistent within one rally boundary window; it is not a player identity and may be reused in another window.
- Assign anonymous `team-a`, `team-b`, or `unknown` grouping plus `near`, `far`, `outside`, or `unknown` court side. Near/far refers to the camera, not serving team.
- At a paused frame, use **Click footpoints** for the point between a player's feet or **Draw boxes** for a tight full-player box. Seek to another frame and repeat with the same short ID.
- Optionally assign `ready`, `playing`, `jumping`, `stand-down`, or `walking` at each observation.

The schema rejects names, persistent player IDs, or any other identity field inside a tracklet. It accepts footpoints, boxes, or both as normalized frame coordinates. One-frame drafts remain valid, but the readiness report counts a track as usable only after two observations; it counts a boundary window as ready only after two usable tracks.

## Optional hard negatives

Ordinary time between rallies is automatically negative, so it does not need manual segmentation. If possible, mark three to five especially confusing dead-time examples per video:

- adjacent-court play;
- foreground player crossing;
- walking or ball retrieval;
- celebration or huddle (distinct from the legacy `celebration` category);
- setup between points;
- timeout;
- camera motion;
- warmup.

Include current model false positives when available, and include randomly selected dead-time controls so the failure set is not composed only of model-chosen examples. The workstation has explicit `model-false-positive` and `random-dead-control` categories for those two sources. Preserve old `celebration` labels as-is; use `celebration-huddle` for the new increment.

These tags help construct targeted failure suites. They do not replace rally labels.

Do **not** label player identity, individual touches, ball trajectories, scores per rally, winners, kills, errors, or rotations. Outside the explicit sparse-tracklet pilot, player annotations are not part of the court/hard-negative/transition-cue increment.

## Workstation controls

1. Start the app with `npm run dev -- --hostname 0.0.0.0` and open `/label` on the printed LAN URL.
2. Choose the **Full corpus** or **Pilot** batch, then choose a prepared task. The app loads both its task JSON and matching NAS proxy. Ready and saved counts appear in the batch selector; the catalog refreshes while full proxies are being prepared.
   When no human draft has been saved yet, the task picker first loads an existing completed human document as an editable copy, when available; saving later writes a separate draft without modifying that completed source. Otherwise it loads the exact offline predictions from the model bundle promoted to production (`model-1ca43e38eefc`, all-labels v2) as the editable starting point. The same production inference and the blind GPT-5.6 Sol labels appear on synchronized read-only timelines immediately below the editable track. Separate production rows recompute `P_pad`, `R_core`, and `F1_padP_coreR` live at two and three seconds of symmetric padding as the human labels change. Overlapping or touching padded ranges are merged before durations and metrics are calculated, as are positive gaps strictly shorter than the configurable join threshold (3 seconds by default). Each model row has a final-export rail: black is retained export, light gray is a joined short gap, and red is missed unpadded human rally time. If the production-model artifact is unavailable, the editor falls back to the Sol prelabel. AI rows are marked `AI`; inspect and correct both boundaries rather than accepting either source as ground truth.
3. To resume a downloaded draft, use the two local fallback pickers for the draft and its matching MP4.
4. Pause on a clear full-court frame, click the four named court corners, and add the two net and two service-zone anchors when visible. The overlay advances to the next anchor; **Finish court clicks** returns control to video playback.
5. Use `S` at serve contact and `E` at end of play. The rally strictly containing the playhead is highlighted in both the timeline and rally table. Its number, boundaries, duration, classification, notes, transition cues, and available AI confidence cues also appear in the rally card beside the video, where it can be classified without scrolling to the interval table. During dead time the card retains the preceding rally, so a rally can be classified immediately after its end is marked. Within a rally, `S` moves its start and `E` moves its end to the playhead. In dead time after a rally, `E` extends that immediately preceding rally to the playhead; the editor rejects an extension that would overlap the next rally, an ignored span, a hard negative, or an existing sparse-player observation window. If a new `S` marker is open and `E` is pressed inside another rally, that rally is split at the playhead: the first part begins at the new `S`, and the remainder keeps the old rally end and metadata. Sparse tracklets are repartitioned by their valid boundary windows. This intentionally leaves the remainder selected for correction; move its start to the next real serve or delete it if it is only the old AI tail. Exactly touching rallies cannot be completed because the live/dead model needs positive dead time between distinct events.
6. Delete the highlighted rally with **Delete selected**, `Delete`, or `Backspace`. Keyboard deletion is disabled while focus is in a form control.
7. If teams switch court sides in this video or format, press `X` at the switch. Side switches are point markers rather than rally intervals; their time and optional note can be edited or deleted in the **Side switches** section.
8. For the optional player pilot, choose the rally and boundary window, enter a short anonymous track ID, then click footpoints or draw boxes on two or more paused frames. **Finish player clicks** restores native video controls.
9. Use `[` then `]` for an ignored span and `H` twice for an optional hard negative.
10. Use `Space` to play/pause, `J`/`K` for ±0.1 seconds, and either Shift+`J`/`K` or `←`/`→` for ±1 second. The vertical line on the label timeline tracks the current video time.
11. Use **Save draft to NAS** often. Selecting that prepared task later resumes the saved draft automatically; downloading a backup JSON is optional. Independently, the browser records the last prepared task and playhead time in versioned local storage during playback and seeking. Reloading `/label` reopens that prepared task at the saved time; local fallback files are never stored.
12. After the entire video is reviewed, check the confirmation box and export completed labels.

Direct drafts are written atomically to `labels/full/<recording-id>.labels.json` or `labels/pilot/<recording-id>.labels.json`, according to the selected batch. The browser can also download the same naming format for offline backups. Do not modify the originals under `tasks/`.

Production-model predictions are loaded from the exact `model-1ca43e38eefc--<recording-id>/analysis.json` artifact under the intake analysis root. Sol candidates remain separate under `prelabels/sol-xhigh/` or the intake workspace's `blind-sol/prelabels/` directory and are never copied into the editable track while the production artifact is available. The first **Save draft to NAS** writes a human draft under `labels/full/`, after which that draft always takes precedence. Candidate provenance, confidence tags, and ambiguity notes remain in the saved document so that later evaluation can distinguish assisted labels from independently created labels.

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

The manifest builder preserves court geometry, transition fields, sparse anonymous player tracklets, hard-negative provenance categories, and side-switch points in each raw recording row. Existing model readers that do not use these optional fields continue to ignore them.

## Read-only readiness report

Audit a label directory without modifying any label document:

```bash
PYTHONPATH=. .venv/bin/python scripts/report-label-readiness.py \
  --labels-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/labels/full \
  --output /path/to/new-label-readiness.json
```

The report lists geometry anchor counts, hard-negative category coverage, transition-field completion by rally outcome stratum, sparse-tracklet window/geometry/state coverage, invalid documents, and an aggregate debt checklist. Transition-cue and tracklet debt each use the five-rally-per-recording pilot targets described above. The output path must not already exist; the command refuses overwrite. Omit `--output` to print JSON to stdout. The scan opens labels read-only and does not freeze, rewrite, or normalize them.

## Full-corpus proxy preparation

The current NAS/host produces full annotation proxies at approximately real time. This resumable command needs roughly 2.5 hours for the present corpus and skips every completed proxy/task on rerun:

```bash
npm run analyze -- prepare-labeling-workspace \
  --plan /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-corpus-plan.json \
  --workspace /mnt/freenas/volleycut/labeling-v1-2026-08-09 \
  --fps 30 --max-width 960 --crf 24 --preset ultrafast --threads 2
```

The width-limited, constant-frame-rate files are seekable annotation proxies. Each completed proxy and task appears automatically in the **Full corpus** picker. Raw 1080p/4K sources remain immutable and should be used for later learned-frame extraction where feasible.
