# Full-corpus labeling increment — 2026-08-10

## Outcome

The next labeling increment is available at `${VOLLEYCUT_APP_BASE_URL}/label`. The **Full corpus** batch contains nine complete, seekable task/proxy pairs totaling 9,082.23 seconds (2:31:22). The original raw recordings remain unchanged under `${VOLLEYCUT_DATA_ROOT}/raw`.

The pack contains two beach 2v2 recordings, four grass reverse-2s/KOB recordings, and three indoor reverse-4s recordings. Six tasks are assigned to train, two grass tasks to validation, and one indoor task to test. All recordings from the same source group remain in the same split.

All nine task documents passed video hash, duration, schema, ROI, capture, and initial interval validation. Preparation left nine proxies, nine provenance records, nine tasks, and no hidden temporary artifacts.

## Workstation decisions

- The prepared-task selector is batch-aware and defaults to **Full corpus**; the completed 90-second pilot remains available separately.
- Ready and saved counts are visible in the batch selector. The catalog polls every 15 seconds, so newly prepared tasks appear without restarting the app.
- Each option shows its priority, environment, original raw filename, duration, and saved rally count.
- Selecting a task loads its exact NAS proxy. A saved draft takes precedence over the blank task and resumes automatically.
- **Save draft to NAS** writes atomically to `labels/full/<recording-id>.labels.json`. Pilot saves remain isolated under `labels/pilot/`.
- Incomplete proxy/provenance/task triplets are not exposed. Task IDs and resolved paths are constrained to the prepared workspace.

## Annotation requested

For every full video:

1. Enter the annotator name.
2. Confirm the prefilled players-per-team value and enter target points when the game format makes it known.
3. Mark every rally from serve contact (`S`) to the first instant live play has ended (`E`), including short aces and service faults.
4. Mark only censored edge play, camera gaps, or genuinely unresolvable boundaries as ignored.
5. Optionally mark three to five confusing dead-time examples as hard negatives.
6. Save drafts frequently. After reviewing the entire continuous video, check the review confirmation and export completed labels.

Player identities, individual touches, ball tracks, running score, point winner, rotations, and play outcome taxonomy are not needed for this increment.

## Acceptance checks

- API catalog: full 9/9 ready and 0 saved; pilot 9/9 ready and 9 saved.
- Browser playback: a full indoor task loaded with media ready state 4 and the expected duration.
- HTTP media serving: byte-range requests return `206 Partial Content` with exact range and content-length headers.
- Task validation: all nine full task documents pass with `--allow-incomplete`; only the expected not-yet-labeled and unknown-target-points warnings remain.

Once the nine full drafts are completed, freeze a new immutable manifest and retrain/evaluate the learned visual sequence model using the existing source-group-disjoint splits. Do not tune against the single indoor test recording.
