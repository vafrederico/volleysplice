# Offline versus browser on-device padding comparison — 2026-08-13

## Outcome

The browser implementation of `model-9c92b8e9333f` is close to the canonical offline
implementation at the export level, but it is not parity-equivalent.

On the one held-out indoor recording, the browser trails offline adjusted
`F1_padP_coreR` by **1.31 percentage points at 2 seconds** of padding on each side and
**1.03 points at 3 seconds**. It preserves one fewer complete core rally in both cases:
36/39 rather than 37/39 at 2 seconds, and 38/39 rather than 39/39 at 3 seconds.

Across the supported seven-recording, no-beach corpus, the adjusted-F1 gaps are smaller:
0.64 points at 2 seconds and 0.73 points at 3 seconds. The browser preserves five fewer
complete rallies at either setting. These aggregates include training and tuning footage,
so they diagnose implementation parity rather than generalization.

Three seconds improves full-rally containment and adjusted F1 for both implementations,
but retains more footage. Event F1 falls because padding expands and merges neighboring
rallies, making a one-to-one IoU match less likely. Event F1 is therefore a boundary/event
diagnostic here; `F1_padP_coreR` and fully-contained rally count better represent export
utility.

The machine-readable report is
`/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/reports/browser-on-device-padding-comparison-v1.json`
(SHA-256 `82a0994fc06534ea88641a22f50b55512344bbd31a54a70e802f430817208312`).

## Retrospective held-out result

This is the only recording that was held out from model fitting and tuning. It contains 39
expected human rallies. The result is a retrospective regression check, not a new blind
acceptance test.

| Padding each side | Runtime | Event precision | Event recall | Event F1 | `P_pad` | `R_core` | Adjusted F1 | Fully present |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2s | Offline | 52.17% | 61.54% | 56.47% | 78.99% | 99.71% | **88.15%** | **37 / 39** |
| 2s | Browser on-device | 51.06% | 61.54% | 55.81% | 76.98% | 99.61% | **86.84%** | **36 / 39** |
| 3s | Offline | 38.10% | 41.03% | 39.51% | 80.72% | 100.00% | **89.33%** | **39 / 39** |
| 3s | Browser on-device | 33.33% | 33.33% | 33.33% | 79.07% | 99.97% | **88.30%** | **38 / 39** |

At 2 seconds, offline would export 52 merged sections totaling 623.0 seconds; browser would
export 53 totaling 639.8 seconds. At 3 seconds, those become 48 sections / 722.5 seconds
offline and 45 sections / 739.4 seconds in the browser. Scoring masks the held-out video's
114.134-second ignored pregame range, leaving respectively 46/47 evaluable fragments at
2 seconds and 42/39 at 3 seconds.

## Validation/tuning result

The two grass validation recordings contain 76 expected rallies. They informed model
selection, so these are tuning diagnostics rather than held-out performance.

| Padding each side | Runtime | Event precision | Event recall | Event F1 | `P_pad` | `R_core` | Adjusted F1 | Fully present |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2s | Offline | 29.67% | 35.53% | 32.34% | 62.04% | 98.38% | **76.09%** | **72 / 76** |
| 2s | Browser on-device | 33.33% | 42.11% | 37.21% | 63.30% | 96.99% | **76.60%** | **68 / 76** |
| 3s | Offline | 18.52% | 19.74% | 19.11% | 65.69% | 98.95% | **78.96%** | **74 / 76** |
| 3s | Browser on-device | 20.93% | 23.68% | 22.22% | 66.37% | 98.19% | **79.20%** | **72 / 76** |

The browser's event and adjusted scores are slightly higher on this slice, while its strict
full-containment count is lower. More candidate sections can improve one-to-one event matches
without reproducing the same boundaries or preserving every rally end-to-end.

## Supported no-beach corpus diagnostic

This scope contains all seven recordings in the model corpus and 274 expected rallies. It
mixes four training, two validation, and one held-out recording; use it to summarize runtime
parity, not to claim generalization.

| Padding each side | Runtime | Event precision | Event recall | Event F1 | `P_pad` | `R_core` | Adjusted F1 | Fully present |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2s | Offline | 49.51% | 55.84% | 52.49% | 77.84% | 99.28% | **87.26%** | **262 / 274** |
| 2s | Browser on-device | 49.37% | 56.93% | 52.88% | 77.16% | 98.72% | **86.62%** | **257 / 274** |
| 3s | Offline | 33.68% | 35.40% | 34.52% | 79.90% | 99.69% | **88.70%** | **271 / 274** |
| 3s | Browser on-device | 33.67% | 36.50% | 35.03% | 78.98% | 99.29% | **87.98%** | **266 / 274** |

The browser produces 410 unpadded candidates across all nine recordings versus 396 offline.
On the supported seven, 2-second padding yields 325 actual merged browser sections versus
318 offline; 3-second padding yields 306 versus 297. After ignored pregame spans are masked
for scoring, these become 316 versus 309 evaluable fragments at 2 seconds and 297 versus 288
at 3 seconds. The additional candidates explain some of the browser's event-recall gains and
padded-precision loss.

## All-nine descriptive aggregate

The combined result is included for completeness because both implementations ran all nine
videos. It is not the headline: the 71 beach rallies are out of distribution because beach
footage was explicitly excluded from this model's corpus.

| Padding each side | Runtime | Event precision | Event recall | Event F1 | `P_pad` | `R_core` | Adjusted F1 | Fully present |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2s | Offline | 45.80% | 45.80% | 45.80% | 69.50% | 99.18% | **81.73%** | **331 / 345** |
| 2s | Browser on-device | 45.92% | 47.25% | 46.57% | 68.99% | 98.83% | **81.26%** | **325 / 345** |
| 3s | Offline | 31.33% | 28.70% | 29.95% | 73.44% | 99.51% | **84.51%** | **341 / 345** |
| 3s | Browser on-device | 31.68% | 29.57% | 30.58% | 72.77% | 99.31% | **84.00%** | **336 / 345** |

The beach-only event F1 is just 9.35% offline / 12.73% browser at 2 seconds and 4.04% /
4.17% at 3 seconds. Its high full-containment counts come from overly long exports and do not
indicate useful beach performance.

## Fully-present rallies by recording

“Fully present” always refers to an unpadded human rally core that is contained from start to
end within one padded, clipped, and merged export section.

| Recording | Role | 2s offline | 2s browser | 3s offline | 3s browser | Expected |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `beach-source-02` | Beach OOD | 38 | 38 | 39 | 39 | 39 |
| `beach-source-01` | Beach OOD | 31 | 30 | 31 | 31 | 32 |
| `grass-source-04` | Validation/tuning | 34 | 31 | 35 | 33 | 36 |
| `grass-source-01` | Training | 42 | 41 | 44 | 42 | 45 |
| `grass-source-09` | Training | 39 | 40 | 40 | 40 | 40 |
| `grass-source-10` | Validation/tuning | 38 | 37 | 39 | 39 | 40 |
| `indoor-source-01` | Training | 36 | 36 | 38 | 38 | 38 |
| `indoor-source-07` | Training | 36 | 36 | 36 | 36 | 36 |
| `indoor-source-05` | Retrospective test | 37 | 36 | 39 | 38 | 39 |

## Metric contract

The two padding cases mean **2 seconds before plus 2 seconds after** and **3 seconds before
plus 3 seconds after** each model interval. Every result is clipped to video bounds; touching
or overlapping padded sections are merged before evaluation and export.

Event precision, recall, and F1 use micro-pooled chronological one-to-one matches between
merged padded exports and unpadded human rallies at IoU >= 0.5. This intentionally penalizes
a single export section that merges multiple rallies.

The canonical adjusted score is:

```text
P_pad  = duration(model padded export intersect human padded export)
         / duration(model padded export)
R_core = duration(model padded export intersect human unpadded rally cores)
         / duration(human unpadded rally cores)

F1_padP_coreR = harmonic_mean(P_pad, R_core)
```

Numerators and denominators are pooled across recordings before division. The human export
uses the same padding, clipping, and merging policy as the model export. Current label-document
ignored ranges are subtracted from model export, human padded export, and human core time.
Those three ignored pregame spans do not intersect any of the 345 frozen rallies.

For additional context, conventional core live-time precision / recall / F1 on the held-out
recording are 55.55% / 99.71% / 71.35% offline and 53.93% / 99.61% / 69.97% browser at
2 seconds. At 3 seconds they are 48.46% / 100.00% / 65.29% offline and 47.25% / 99.97% /
64.17% browser. Precision falls as the export retains more non-rally footage; adjusted
precision is higher because it compares against the equally padded human export.

## Interpretation and decision

The browser port is operationally viable and aggregate export quality is close to offline,
but these results do not close the feature-parity gate. The browser still uses a linear audio
resampler rather than the canonical FFmpeg/libswresample path, and the full media pipeline
produces 410 rather than 396 offline intervals. The small aggregate adjusted-F1 gap can hide
specific boundary substitutions and missed complete rallies.

Use 3 seconds on each side when maximizing complete-rally preservation is worth the extra
footage. Use 2 seconds when shorter exports matter and accepting a small containment loss is
reasonable. This report compares fixed padding settings; it does not use the held-out result
to retune the model, decoder, or padding policy.

## Reproduction

```bash
PYTHONPATH=. python3 scripts/evaluate-browser-parity-padding.py \
  --output /path/to/new-report.json
```

The evaluator refuses to overwrite an existing report. It validates all analysis identities
and ranges, loads included unpadded intervals from both runtime variants, uses rallies from the
frozen nine-video manifest (SHA-256 `b662c5078d37858fd979a0c68759c0033152a3aefdce4886037351b4afb6b6fe`),
and overlays only the current label documents' ignored ranges. The JSON records every input,
label, model, analysis, and evaluator SHA-256 plus per-scope and per-recording components.
