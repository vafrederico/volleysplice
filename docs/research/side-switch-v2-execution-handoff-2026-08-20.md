# Side-switch v2 execution handoff — 2026-08-20

## State at handoff

No v2 implementation or NAS artifact has been written yet. The completed v1 model,
dataset, evaluation, tests, and research report are committed in `a946009`.

The full review decision file remains:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/appearance-review-decisions-full-nas-v1.json`

Its SHA-256 is
`5adde135b226753eebc0acecfe108f1b1994e752687e017deddce312cd422c88`.
It contains all 1,096 marker decisions: 88 switch, 1,004 no-switch, and 4
unclear.

## Frozen raw recording split

The v2 split was fixed before extracting or scoring any v2 feature. Recording IDs
were ordered by SHA-256 of `side-switch-v2-split:<recording-id>`. The first five are
training, the next two validation, and the final four evaluation.

### Train

- `raw-no-backup-PXL_20260816_164327879`
- `raw-no-backup-PXL_20260816_171720964`
- `raw-no-backup-PXL_20260816_190429172`
- `raw-no-backup-PXL_20260816_180646590`
- `raw-no-backup-PXL_20260816_183701800`

### Validation

- `raw-no-backup-PXL_20260816_210449857`
- `raw-no-backup-PXL_20260816_193307688`

### Evaluation

- `raw-no-backup-PXL_20260816_160023210`
- `raw-no-backup-PXL_20260816_161923155`
- `raw-no-backup-PXL_20260816_203801418`
- `raw-no-backup-PXL_20260816_212717581`

All raw recordings share source group `volleycut-raw-no-backup`; this is therefore a
recording-held-out split, not a source-group-independent split. The raw labels and
pooled v1 behavior have already informed the v2 feature hypotheses, so the four
evaluation recordings are a preregistered confirmation set rather than a pristine
never-observed test corpus.

## V2 execution contract

Create new immutable v2 artifacts; never overwrite the v1 report or model.

1. Add side-conditioned color evidence without requiring team identity labels:
   near/far same-assignment cost, swapped-assignment cost, swap margin, and a
   continuous area-and-position-weighted color moment.
2. Add frame-level consistency: median and lower-quartile swap margin, fraction of
   frame pairs supporting a swap, variance, usable pair count, and minimum
   before/after coverage.
3. Add derived existing-signal features: mean/min/disagreement of equal- and
   area-weighted palette distances, player-minus-background evidence, geometry
   stability, and palette-by-stability interactions.
4. Add unsupervised per-recording median/MAD or percentile normalization. Compute it
   without decisions and bind it to the complete candidate sequence.
5. Remove raw gap duration from the learned feature bank. V1 learned it strongly,
   but the train relationship disappears on raw footage.
6. Add a separate temporal decoder using rally order, switch-spacing priors, and
   toggle consistency. Do not expose evaluation decisions during decoder selection.
7. Fit feature-family/L2 choices with recording-grouped OOF predictions from the
   development recordings. Select thresholds and decoder settings on validation,
   freeze them, and open the four evaluation recordings once.
8. Report the static classifier and temporal decoder separately, including per-video
   precision, recall, F1, ROC AUC, AP, and confusion counts. Keep indoor fixed to
   no-switch and outside specialist ranking metrics.

The strongest immediate derived feature hypothesis is
`mean(playerPaletteEqual, playerPaletteArea) * geometryStability`. Its earlier
retrospective raw diagnostic was encouraging, but that observation is hypothesis
generation only and is not a valid v2 promotion result.
