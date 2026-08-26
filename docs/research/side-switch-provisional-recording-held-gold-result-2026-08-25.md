# Side-switch provisional recording-held gold result — 2026-08-25

## Decision

Do not promote or tune any T1–T20 representation from these two recordings. T19 and
T20 are the strongest experimental transfers because they match the deployed and
boundary-T0 controls without adding a false proposal, but neither recovers another
event. T2, T4, T14, T16, and T18 each add the same false proposal at unchanged recall.

This remains provisional evidence. Both label snapshots are `in-progress`,
`grass-source-11` is not continuously reviewed, the source groups appeared in
historical work, and one of seven markers is a retained deployed-model proposal.

## Frozen scope and fit audit

The evaluation binds these label snapshots:

| Recording | Label SHA-256 | Continuous review | Markers | Retained model-origin markers |
| --- | --- | --- | ---: | ---: |
| `grass-source-11` | `ced51c22114541acec74b10d6992109e90425f6a53f0bc7e4581a6f4429fdaa5` | No | 4 | 0 |
| `grass-source-07` | `8bada37b00829755ef28e82fc2d549cba4c782b8e9fefbc58a31e6f087ac7a4e` | Yes | 3 | 1 |

The T1–T20 engineering-pass audit admits exactly T2, T4, T14, T16, T18, T19,
and T20. T1, T3, T5–T13, T15, and T17 are excluded. T15's label-free diagnostic is
materialized only because passing T16 consumes its two source advantages; the failed
T15 model profile is never scored.

Every admitted artifact already had a frozen `fullDevelopment.classifier` trained on
the original 11 opened-development recordings. No admitted model was missing a fit,
so no training used either new recording. Thresholds, feature orders, weights, and the
ranked-candidate decoder are applied unchanged.

## Extraction result

Feature extraction loaded no label file. It reconstructed candidates from frozen
production traces and cached F104 inputs, then extracted T2/T4/T14-derived appearance
features from the immutable native videos in `volleycut-raw-no-backups`. The 4K AV1
recording used system FFmpeg to emit lossless BGR24 native frames because the installed
OpenCV codec cannot decode AV1; the other native source used OpenCV.

| Check | Result |
| --- | ---: |
| Recordings | 2 |
| Candidate rows | 83 |
| Adjacent-boundary / internal rows | 82 / 1 |
| Rows after ignored intervals | 78 |
| Boundary rows after ignored intervals | 77 |
| ±4 candidate coverage | 7 / 7 markers |
| Maximum deployed-probability parity error | `1.305e-8` |

The candidate generator is therefore not the immediate ceiling on this snapshot: all
seven markers are coverable under the frozen ±4 interval matcher. The miss is in
ranking/threshold/decoding, not absent candidate intervals.

## Frozen evaluation

Strict and ±4-second results are identical for every profile:

| Profile | Proposals | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deployed union34 | 1 | 1 | 0 | 6 | 100.00% | 14.29% | 25.00% |
| Boundary T0 | 1 | 1 | 0 | 6 | 100.00% | 14.29% | 25.00% |
| T2 conditional transport | 2 | 1 | 1 | 6 | 50.00% | 14.29% | 22.22% |
| T4 selective far jersey | 2 | 1 | 1 | 6 | 50.00% | 14.29% | 22.22% |
| T14 dominant medoid | 2 | 1 | 1 | 6 | 50.00% | 14.29% | 22.22% |
| T16 source-resolved medoid | 2 | 1 | 1 | 6 | 50.00% | 14.29% | 22.22% |
| T18 representative medoid | 2 | 1 | 1 | 6 | 50.00% | 14.29% | 22.22% |
| T19 cross-representation consensus | 1 | 1 | 0 | 6 | 100.00% | 14.29% | 25.00% |
| T20 compact medoid/disagreement | 1 | 1 | 0 | 6 | 100.00% | 14.29% | 25.00% |

Every profile emits zero proposals on `grass-source-11`, missing all four
markers. On `grass-source-07`, every profile selects the retained model-origin
marker at 340.742 seconds and misses both human-added markers. T2/T4/T14/T16/T18 also
select the same unmatched boundary anchored at 173.983 seconds; T19 and T20 suppress
it like the controls.

Consequently, the only true positive is not independent of the deployed model that
seeded the labeling task. On the six human-added markers, every evaluated profile has
zero recall. This is the most important limitation when interpreting the nominal
100% precision of the one-proposal profiles.

## Artifacts and reproduction

- Feature artifact:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-provisional-recording-held-gold-r1-features.json`
  — SHA-256 `1b0fbc1ccc67a1d636e374b4052c310299532967c78196a937b5caaa2552dd5d`
- Evaluation artifact:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-provisional-recording-held-gold-r1-evaluation.json`
  — SHA-256 `fe1e6c43d3ecd98303bbab978b02b0325f908bf225333b3880adf6a8d9ee5873`

Reproduction commands from the repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-side-switch-provisional-gold.py extract \
  --recording-workers 2 --opencv-threads 10 \
  --features /new/output/side-switch-gold-features.json
PYTHONPATH=. .venv/bin/python scripts/evaluate-side-switch-provisional-gold.py evaluate \
  --features /new/output/side-switch-gold-features.json \
  --evaluation /new/output/side-switch-gold-evaluation.json
```

The runner refuses to overwrite either artifact and verifies label, prediction,
runtime, classifier-artifact, proxy-video, and native-video hashes.

## Next action

Wait for both labels to be completed and continuously reviewed. If either bound hash
changes, retain this R1 result and create an R2 artifact. Do not tune a threshold,
prune T14, or select T19/T20 on R1. A future experiment should target why coverable
human-added boundaries remain low-scoring and must be preregistered before any later
gold revision is opened.
