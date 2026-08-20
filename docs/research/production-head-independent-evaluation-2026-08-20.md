# Independent precision/recall of production model heads

Generated: `2026-08-20T22:09:43.006800+00:00`.

## Outcome

No production head should yet be treated as a universally complete standalone gate. The serve heads are the closest match for a region proposal criterion, but their decoded 1-second recall on the two human-reviewed development recordings is only 80.9% (all-labels v2) and 80.9% (previous production). Even a 2-second matching region reaches 94.4% and 92.1%, not 100%. The dead-state head is explicitly local and needs an existing candidate end anchor. The suppression head is a veto specialist with a restricted evaluation universe, not a general dead-time detector.

The protected test (`indoor-source-05`) was not opened. There is currently no fully held-out, completed human boundary-gold scope available without opening it. The two human-reviewed development documents remain `in-progress` and share source groups with model development, so serve/dead results are provisional. Held-out feedback is valid for reviewed inclusion/veto analysis, but all retained boundaries are untouched model-origin ranges and are not boundary-gold evidence.

## Raw classifier metrics at production thresholds

These are pooled sample counts. Each row uses that head's own target and universe, so rows are useful for understanding a head but are not a cross-head leaderboard.

### held-out-feedback

Serve and dead-state rows in this scope use untouched model-origin boundaries and are diagnostic only.

| Bundle | Head | Threshold | Positives | Predicted positives | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| All-labels v2 | rally | 0.50 | 10152 samples | 8828 samples | 0.8828 | 0.7676 | 0.8212 |
| All-labels v2 | serve | 0.85 | 1857 samples | 1300 samples | 0.5454 | 0.3818 | 0.4492 |
| All-labels v2 | deadState | 0.90 | 1707 samples | 1430 samples | 0.6161 | 0.5161 | 0.5617 |
| Previous production | rally | 0.50 | 10152 samples | 9380 samples | 0.8357 | 0.7722 | 0.8027 |
| Previous production | serve | 0.85 | 1857 samples | 2402 samples | 0.4338 | 0.5611 | 0.4893 |
| Previous production | deadState | 0.90 | 1707 samples | 3029 samples | 0.4946 | 0.8776 | 0.6326 |
| Suppression | suppression | 0.75 | 1521 samples | 797 samples | 0.6361 | 0.3333 | 0.4374 |

### development-human-reviewed

| Bundle | Head | Threshold | Positives | Predicted positives | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| All-labels v2 | rally | 0.50 | 1975 samples | 2762 samples | 0.5673 | 0.7934 | 0.6616 |
| All-labels v2 | serve | 0.85 | 709 samples | 799 samples | 0.6120 | 0.6897 | 0.6485 |
| All-labels v2 | deadState | 0.90 | 704 samples | 207 samples | 0.8116 | 0.2386 | 0.3688 |
| Previous production | rally | 0.50 | 1975 samples | 3040 samples | 0.4951 | 0.7620 | 0.6002 |
| Previous production | serve | 0.85 | 709 samples | 858 samples | 0.5478 | 0.6629 | 0.5999 |
| Previous production | deadState | 0.90 | 704 samples | 663 samples | 0.5762 | 0.5426 | 0.5589 |
| Suppression | suppression | 0.75 | 2259 samples | 1640 samples | 0.7579 | 0.5502 | 0.6376 |

### development-all-diagnostic

| Bundle | Head | Threshold | Positives | Predicted positives | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| All-labels v2 | rally | 0.50 | 3257 samples | 4066 samples | 0.6387 | 0.7974 | 0.7093 |
| All-labels v2 | serve | 0.85 | 1039 samples | 1210 samples | 0.6248 | 0.7276 | 0.6723 |
| All-labels v2 | deadState | 0.90 | 1032 samples | 360 samples | 0.8528 | 0.2975 | 0.4411 |
| Previous production | rally | 0.50 | 3257 samples | 4541 samples | 0.5759 | 0.8029 | 0.6707 |
| Previous production | serve | 0.85 | 1039 samples | 1338 samples | 0.5613 | 0.7228 | 0.6319 |
| Previous production | deadState | 0.90 | 1032 samples | 914 samples | 0.6225 | 0.5514 | 0.5848 |
| Suppression | suppression | 0.75 | 2976 samples | 2267 samples | 0.7671 | 0.5843 | 0.6634 |

### combined-diagnostic

| Bundle | Head | Threshold | Positives | Predicted positives | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| All-labels v2 | rally | 0.50 | 13409 samples | 12894 samples | 0.8058 | 0.7749 | 0.7900 |
| All-labels v2 | serve | 0.85 | 2896 samples | 2510 samples | 0.5837 | 0.5059 | 0.5420 |
| All-labels v2 | deadState | 0.90 | 2739 samples | 1790 samples | 0.6637 | 0.4337 | 0.5246 |
| Previous production | rally | 0.50 | 13409 samples | 13921 samples | 0.7510 | 0.7796 | 0.7650 |
| Previous production | serve | 0.85 | 2896 samples | 3740 samples | 0.4794 | 0.6191 | 0.5404 |
| Previous production | deadState | 0.90 | 2739 samples | 3943 samples | 0.5242 | 0.7547 | 0.6187 |
| Suppression | suppression | 0.75 | 4497 samples | 3064 samples | 0.7330 | 0.4994 | 0.5941 |

## Serve heads on held-out export feedback

Production does not merge the two serve peaks into one detector. It decodes each serve head inside its own three-head model path and later unions the resulting rally intervals. The combined rows below mirror those parallel paths: duplicate path alerts remain in the precision denominator, while each feedback start is counted only once for recall if either path covers it.

| Tolerance | Path | Precision | Recall | Matched / predicted | Covered / true |
|---:|---|---:|---:|---:|---:|
| 0.5s | All-labels v2 | 0.3009 | 0.2881 | 68 / 226 | 68 / 236 |
| 0.5s | Previous production | 0.2784 | 0.3432 | 81 / 291 | 81 / 236 |
| 0.5s | Both production paths | 0.2882 | 0.4280 | 149 / 517 | 101 / 236 |
| 1.0s | All-labels v2 | 0.5044 | 0.4831 | 114 / 226 | 114 / 236 |
| 1.0s | Previous production | 0.4433 | 0.5466 | 129 / 291 | 129 / 236 |
| 1.0s | Both production paths | 0.4700 | 0.6186 | 243 / 517 | 146 / 236 |
| 2.0s | All-labels v2 | 0.6903 | 0.6610 | 156 / 226 | 156 / 236 |
| 2.0s | Previous production | 0.5739 | 0.7076 | 167 / 291 | 167 / 236 |
| 2.0s | Both production paths | 0.6248 | 0.7542 | 323 / 517 | 178 / 236 |

These feedback boundaries are untouched production-model starts, not verified serve-contact annotations. This section measures agreement with export feedback starts; it cannot establish semantic serve recall.

## Operational interpretation on the strongest applicable scope

| Bundle/head | Operational unit | Precision | Recall/coverage | Counts |
|---|---|---:|---:|---|
| All-labels v2 rally | decoded live seconds against held-feedback retained ranges | 0.9729 | 0.8167 | 2126.4s predicted / 2533.0s retained |
| All-labels v2 serve | decoded contacts within 1.0s on human-reviewed development | 0.7200 | 0.8090 | 72/89 serves; 100 detections |
| All-labels v2 dead-state | oracle-end-anchored transitions on human-reviewed development | — | 0.1685 | 15/89 ends emitted |
| Previous production rally | decoded live seconds against held-feedback retained ranges | 0.9196 | 0.8198 | 2258.2s predicted / 2533.0s retained |
| Previous production serve | decoded contacts within 1.0s on human-reviewed development | 0.6990 | 0.8090 | 72/89 serves; 103 detections |
| Previous production dead-state | oracle-end-anchored transitions on human-reviewed development | — | 0.1011 | 9/89 ends emitted |
| Suppression | decoded selected-universe samples | 0.6626 | 0.3241 | rally-sample survival 0.9753 |

## Can a head be used by itself?

- **Serve-region proposals:** useful, but not a complete gate. A consumer must keep a fallback path for rallies whose serve head does not emit a matched peak. The JSON artifact includes event precision/recall at thresholds 0.05–0.95 and 0.5/1.0/2.0-second tolerance sensitivity.
- **Rally inclusion:** the rally heads can independently propose live regions, but neither should be interpreted as complete coverage. Production unions them and also uses serve rescue specifically because one head alone misses live time. Held-feedback precision is optimistic as an absolute truth estimate because the accepted reference ranges themselves originated from the production ensemble.
- **End refinement:** dead-state scores are meaningful only in local windows around an existing proposed end. Oracle-anchored coverage is reported to isolate the head; it does not establish global standalone precision.
- **Suppression/veto:** use only inside the current one-model-only eligibility gate. Its precision/recall universe intentionally excludes arbitrary dead time, so it cannot justify whole-video dead-time classification.

## Metric contracts

- Rally raw target: human live-play samples outside `ignoredIntervals`; operational metrics apply only that rally head's production temporal decoder and measure unpadded duration overlap.
- Serve raw target: samples within ±1.0 seconds of every human serve contact. Operational metrics apply the production peak threshold, 10-second NMS, and a primary 1.0-second one-to-one matching tolerance.
- Dead-state raw target: the production local end-transition target (2 seconds before/after each end plus pre-serve negative controls), at the production 0.90 dead threshold.
- Suppression raw target: valid non-rally samples selected by the production ensemble or explicit hard-negative labels versus valid human rally samples. The production 1.0-second smoothing and 0.75/0.65 decoder is used for the operational row.
- All sample metrics pool confusion counts across recordings; they do not average per-recording precision or recall.
- Held-feedback serve/dead rows remain in JSON for auditability, but they are diagnostic only: `userTouchedCutIds` is empty for all six held files, so those model-origin starts and ends are not independent semantic boundary labels.

Full threshold profiles, per-recording diagnostics, model hashes, source hashes, and exact counts are in the companion JSON artifact.
