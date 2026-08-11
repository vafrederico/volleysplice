# Full-corpus model training — 2026-08-10

## Outcome

The nine continuously reviewed full-video label drafts now produce a reproducible, immutable training set and a new CPU baseline model. The corrected snapshot contains **345 rallies across 9,082.23 seconds (151.37 minutes)** of footage. The model was trained only on the full-video manifest; the overlapping 90-second pilot excerpts were not added again.

The validation-selected baseline is useful for review assistance, but not unattended cutting. On the one-video indoor test it matched 28 of 39 rallies, with **0.629 event F1**, **0.671 time IoU**, **0.490-second start-boundary MAE**, and **1.921-second end-boundary MAE**. It still predicts too many events and tends to end rallies late.

## Pre-training label audit

The saved manual drafts were preserved unchanged under:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/labels/full`

The audit found seven exact-touching interval pairs. In every case the short second interval had the same AI tags and note as the preceding interval, which identifies it as the remainder created by the editor's split shortcut. Four indoor cases were also checked against dense frames and audio: players were already relaxing, huddling, or remaining down after the terminal play, with no new serve/receiver response. A binary live/dead target cannot represent two distinct events with zero dead time.

The seven removed tails totaled 6.399 seconds:

| Recording | Removed interval |
|---|---:|
| `grass-source-01` | 724.091–725.200 |
| `grass-source-09` | 368.092–369.800 |
| `grass-source-09` | 955.687–957.000 |
| `indoor-source-01` | 190.448–191.750 |
| `indoor-source-01` | 466.483–467.050 |
| `indoor-source-07` | 652.409–652.500 |
| `indoor-source-07` | 779.191–779.500 |

The final snapshot ledger stores each removed interval verbatim, its reason, source-draft and snapshot hashes, and the referenced proxy hash. The provisional uncorrected snapshot was retained under `completed/full-v0-touching-artifacts` rather than deleted.

Completed labels now require positive dead time between successive rallies. The UI may temporarily create a touching remainder during a split, but completion is blocked until its start is moved to the next serve or the stale remainder is deleted.

## Frozen data and split

- Completed read-only snapshot: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/completed/full-v1`
- Snapshot ledger: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/completed/full-v1/snapshot.json`
- Training manifest: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json`
- Manifest SHA-256: `b662c5078d37858fd979a0c68759c0033152a3aefdce4886037351b4afb6b6fe`

| Split | Videos | Source groups | Rallies | Live seconds | Aces | Service faults |
|---|---:|---:|---:|---:|---:|---:|
| Train | 6 | 3 | 230 | 1,842.199 | 25 | 28 |
| Validation | 2 | 1 | 76 | 450.121 | 10 | 13 |
| Test | 1 | 1 | 39 | 322.419 | 4 | 7 |

Source groups do not cross splits. Validation contains only grass footage, test contains one indoor source, and beach appears only in training. The test video also contains the earlier TDS pilot excerpt, so it is a regression test rather than a pristine unseen final benchmark. There are only five independent source groups.

The full and pilot manifests must not be concatenated: the pilot clips are re-encoded excerpts of these same recordings and would duplicate frames, overweight those timestamps, and risk leakage.

## Model and training

- Model: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-percentile-v1`
- Feature cache: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/full-percentile-v1`
- Model metadata SHA-256: `005cbd86438ac3877db0f31d3a3031d3120d5f0f734506ce399483921a1e06b7`
- Weights SHA-256: `bc19cece24ee90bcf93ae6cd0381c472f5c9a323172153c44a0fdecc61f2ddc3`

The model is the existing 4 fps CPU feasibility baseline: low-resolution appearance, frame difference, regional motion, and optical flow; tied within-recording percentile normalization; five temporal samples at -2, -1, 0, +1, and +2 seconds; and a class-weighted logistic classifier. Training used 24,265 samples, stopped after 28 epochs with epoch 8 selected, and took 335.5 seconds with a cold feature cache.

The validation-selected decoder uses:

- smoothing: 0.5 seconds;
- enter/exit thresholds: 0.50 / 0.45;
- minimum live duration: 3.0 seconds;
- gap bridging: 1.5 seconds.

Outcome tags (`ace`, `service-fault`) and Sol confidence/provenance tags are preserved in the labels but are not model inputs or sample weights. The current model learns only live versus dead time.

## Results

| Split | True / predicted / matched | Event precision | Event recall | Event F1 | Time IoU | Start MAE | End MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation (tuning) | 76 / 93 / 48 | 0.516 | 0.632 | 0.568 | 0.503 | 0.724s | 2.685s |
| Test (one indoor source) | 39 / 50 / 28 | 0.560 | 0.718 | **0.629** | **0.671** | **0.490s** | **1.921s** |

Immutable reports:

- `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-percentile-v1-validation.json`
- `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-percentile-v1-test.json`

The earlier pilot percentile model scored 0.706 validation event F1 on two 90-second grass excerpts but 0 event F1 on its 90-second indoor test excerpt. Those numbers are not directly comparable to full-video evaluation. The full model's 0.629 test F1 and 28 matched rallies are nevertheless a meaningful regression improvement over zero matches on the earlier excerpt.

## Decision and next increment

Use `full-percentile-v1` as the reproducible review-assist baseline. Do not use it for automatic exports or treat the logistic score as a calibrated probability.

The strongest next model work is:

1. add audio onset/cadence-collapse features and a learned stand-down/end-of-play signal, because end boundaries are the largest systematic error;
2. add a pretrained visual backbone with short and long temporal branches while retaining this manifest, decoder, and interval metrics;
3. report metrics by rally outcome and explicitly evaluate immediate-result service faults/aces;
4. collect new source groups so every environment has validation/test coverage, with a genuinely untouched multi-recording test set.

Validation is already used for both classifier early stopping and decoder selection, so its score is tuning evidence, not independent generalization evidence.
