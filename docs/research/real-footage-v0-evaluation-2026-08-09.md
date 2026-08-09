# Real-footage v0 evaluation — 2026-08-09

## Outcome

The current v0 is useful as an end-to-end data, training, inference, and evaluation scaffold, but the learned court-motion baseline is not accurate enough to cut these recordings without review. It generalized partially from three training source groups to an independent indoor 4v4 clip: 68% of labeled live time was retained, but only 48% of predicted live time was actually live, and none of the two complete rallies met the strict 0.5 interval-IoU event match.

This result strengthens the earlier decision: keep the beach thesis's dense live/dead task, temporal context, match-disjoint splitting, and interval metrics; replace the v0 motion representation with a learned visual model trained on our camera domain. VNL-STES remains optional pretraining or a later event branch, not the cutter.

The derived footage workspace is `/mnt/freenas/volleycut/v0-2026-08-09`. Its `README.md`, raw inventory, manifest, proxies, provenance sidecars, model artifacts, reports, and inference outputs are outside Git. The raw files under `/mnt/freenas/volleycut/raw` were treated as immutable.

## Corpus inventory

Nine recordings were found: four grass, three indoor, and two beach, totaling about 2.52 hours and 19.10 GB. All are 60 fps with stereo audio. Seven are 3840×2160 VP9; two are 1920×1080 H.264. Filename/match relationships reduce them to five independent source groups, which is the unit used for leakage prevention.

The footage includes confirmed 2v2 beach/grass and 4v4 indoor games. Target point counts were not reliably established from the excerpts, so `targetPoints` is explicitly `null`. The manifest schema and evaluation report now carry `playersPerTeam` and `targetPoints`; results can be stratified as more formats are labeled.

## Camera fit

| Check | Observed in this corpus | Assessment |
|---|---|---|
| Locked camera | All reviewed excerpts appear stationary with no broadcast cuts | Good fit for the beach-derived capture contract. |
| End-line view | All are behind an end line or close to it, generally centered | Good geometric match; substantially closer to the beach thesis than VNL broadcast views. |
| Full court and service areas | Present in the reviewed excerpts | Meets the essential acceptance test. |
| Elevation | Outdoor views are visibly low; physical height is unknown | Usable for rally cutting, but near players occlude court/players and cross the foreground. Raise future cameras where practical. |
| Background isolation | Grass recordings include adjacent courts/spectators; indoor frames include benches/people at edges | Significant hard-negative motion. A court mask and semantic features are needed. |
| Source quality | Raw media is 1080p/4K at 60 fps | More than sufficient. Several emergency 960×540 pilot proxies fall below the intended 720p floor and should not become benchmark masters. |

The papers do not justify an exact physical height or distance. For our own capture, acceptance should remain coverage-based: fixed landscape framing, centered behind an end line, full sidelines and both service areas visible with margin, no pan/zoom, retained audio, and 1080p30 or 1080p60. Commercial guidance suggests roughly 8–10 feet of elevation, but that is not a measured requirement from either research paper.

## Pilot protocol

The executable manifest contains five recordings in five independent source groups:

- Train: one beach 2v2, one grass 2v2, and one indoor 4v4 excerpt; eight rallies.
- Validation: one independent grass 2v2 excerpt; four rallies.
- Test: one independent indoor 4v4 excerpt; two rallies.

Related sets/games remain in the same source group. There is only one beach source group, so beach cannot be both trained and held out in this pilot. Labels follow half-open serve-contact-to-dead-ball intervals and were estimated from 4 fps contact sheets. They are adequate for exercising the pipeline, not for comparing architectures; a continuous-video second pass and independent label review are required first.

The default model samples the court ROI at 4 fps and 192×108, calculates motion/optical-flow grid features with ±2 seconds of context, fits a regularized logistic classifier, and tunes only the interval decoder on validation. Training stopped with epoch 2 as the best validation epoch. Model and report artifacts bind to the exact immutable manifest and normalized-video hashes.

## Results

| Model | Split | Live recall | Live precision | Time IoU | Event F1 | Predicted / true rallies |
|---|---|---:|---:|---:|---:|---:|
| Motion + optical flow | Validation grass 2v2 | 1.00 | 0.58 | 0.58 | 0.57 | 3 / 4 |
| Motion + optical flow | Held-out indoor 4v4 | 0.68 | 0.48 | 0.39 | 0.00 | 3 / 2 |
| Motion without optical flow | Validation grass 2v2 | 0.97 | 0.59 | 0.58 | 0.50 | 4 / 4 |
| Motion without optical flow | Held-out indoor 4v4 | 0.72 | 0.42 | 0.36 | 0.00 | 3 / 2 |

Optical flow is retained for v0 because it improved held-out time IoU and precision, though this sample is far too small to establish a real ablation result. Evaluation took 2.30 seconds for 49 seconds of video with a warm feature cache (0.047 processing/video ratio); inference itself took roughly 1.8–5.3 seconds for 49–90 second proxies on this host.

The held-out truth windows were 7.0–18.5 and 28.0–37.75 seconds. The optical-flow model predicted 2.875–4.858, 12.375–21.358, and 29.375–48.625. This shows the core failure: it detects active motion but does not reliably recognize serve/dead-ball boundaries, and it treats post-play movement as live.

Qualitative inference on four unlabeled sister recordings produced:

- Beach: zero intervals despite visible play in the 1 fps contact sheet—a clear domain/threshold false-negative warning.
- Grass, KB group: two long intervals (15.375–43.358 and 60.608–89.858) broadly overlap visible play but retain substantial setup/dead time and end-censor the excerpt.
- Grass, Shoreline group: only two short intervals (40.608–43.625 and 52.608–57.125) despite other visibly active sequences, indicating strong viewpoint/lighting sensitivity.
- Indoor sister set: five plausible active windows, but no metric is claimed without labels.

These observations are qualitative because those four clips were intentionally left outside the labeled manifest.

## Implementation changes prompted by real footage

- Normalization supports bounded excerpts with recorded source start/duration provenance.
- H.264 preset and thread limits are configurable for constrained workers.
- The manifest validates optional game format, players per team, and target points.
- Evaluations include `byPlayersPerTeam` and `byTargetPoints`, including an `unknown` group.
- A real, source-group-disjoint manifest, model, no-flow ablation, held-out reports, and five inference artifacts now exist in the external workspace.

The 4K VP9 transcodes exposed a host constraint: high-quality/concurrent encodes were killed while swap was saturated. The pilot therefore used bounded excerpts and some 960-pixel-wide ultrafast proxies. Production ingestion should serialize or resource-limit jobs and create at least 1280×720 benchmark proxies.

## Decision and next implementation increment

Retain v0 as a transparent baseline and regression test. Do not use its boundaries for unattended exports. The next model should follow the beach thesis more closely: a pretrained image backbone on low-rate frame sequences, separate short and long context branches, temporal ensembling, then conservative decoding. Court masking should use a polygon/corners rather than the current rectangle so sky, spectators, and adjacent courts do not dominate motion.

Before that comparison, annotate continuous video from at least 20–30 independent source groups, balanced across beach/grass/indoor, 2v2/4v4/6v6, camera heights, lighting, and known scoring targets. Include warmups, timeouts, celebrations, adjacent courts, service faults, foreground crossings, and very short rallies as explicit hard negatives/positives. Split by match/session/device; never split random frames or sister sets.

The user review surface remains part of the product requirement: expose predicted intervals with configurable pre/post-roll, rapid split/merge/delete, and save corrections as separately consented training labels.

## Artifacts

- Workspace guide: `/mnt/freenas/volleycut/v0-2026-08-09/README.md`
- Raw inventory: `/mnt/freenas/volleycut/v0-2026-08-09/manifests/raw-inventory.json`
- Exploratory manifest: `/mnt/freenas/volleycut/v0-2026-08-09/manifests/real-v0.json`
- Default model: `/mnt/freenas/volleycut/v0-2026-08-09/models/real-rally-v0`
- Held-out report: `/mnt/freenas/volleycut/v0-2026-08-09/reports/test-evaluation.json`
- No-flow report: `/mnt/freenas/volleycut/v0-2026-08-09/reports/test-evaluation-no-flow.json`
