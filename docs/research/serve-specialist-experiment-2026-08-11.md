# Serve-specialist composition experiment — 2026-08-11

## Outcome

A separate serve-contact classifier learns a useful signal, including on short rallies, but the
first composition rule does **not** improve strict held-out event recall. Keep it as an opt-in
experiment; do not make it the default cutter.

On the one-source indoor regression test, the serve head found 33 of 39 contacts within one
second, including 7 of 10 rallies at most three seconds. Composing that head with the existing
live/dead model raised short-rally any-overlap recall from 8/10 to 9/10 and 95%-coverage recall
from 6/10 to 8/10. It did not add an IoU-0.5 match: overall event recall stayed at 28/39 while
event F1 fell from 0.629 to 0.602 because four extra predictions were retained.

This is evidence that the specialist can gate useful serve context, but not yet that it adds a
signal unavailable to the full model: the newly recovered test fault already exists in the full
model's permissive decode. It is also not evidence that a serve timestamp alone can infer the
dead-ball boundary.

## Frozen data and protocol

The experiment uses only `full-gold-v1`: 345 rallies in nine continuously reviewed videos,
split by source group.

| Split | Videos / source groups | Rallies | At most 3 s | Aces | Service faults |
|---|---:|---:|---:|---:|---:|
| Train | 6 / 3 | 230 | 42 | 25 | 28 |
| Validation | 2 / 1 | 76 | 21 | 10 | 13 |
| Test | 1 / 1 | 39 | 10 | 4 | 7 |

The existing `full-percentile-v1` live/dead artifact stays frozen. Serve-model weight updates use
only the train rows, but validation selects its early-stopping epoch and then 8,640
contact-decoder/composition choices. The baseline had already used the same validation source for
its own early stopping and rally decoder. This heavy three-way validation reuse makes the result
tuning evidence, not a generalization estimate. The frozen serve parameters are applied unchanged
to test. Outcome tags are evaluation slices only; they are not inputs, targets, or sample weights.
Pilot excerpts are not included because they duplicate portions of these full videos.

The test source has been inspected and evaluated in earlier work, so it is a retrospective
regression set rather than a pristine final benchmark. Validation contains only one grass
source group. New source-group-disjoint data is required before any product promotion.

## Model and composition

The specialist uses the same cached 210 contextual inputs as the baseline, so inference decodes
the video once and adds only a second 210-weight logistic dot product. Its positive target is
every 4 Hz sample within one second of a labeled serve contact. All other evaluable samples,
including the rest of each rally, are negative. Training used 1,858 positive and 22,407 negative
samples; epoch 9 was selected and fitting plus validation search took 61.0 seconds from the warm
feature cache.

Validation searched 8,640 combinations, breaking equal event-score ties with time IoU,
live-time precision, retained dead time, and prediction count. The selected serve decoder uses a
0.75 threshold, 10-second score-first non-maximum suppression, and a -0.25-second timestamp
correction. The composition:

1. keeps every standard full-model interval;
2. moves a nearby full-model start back to a detected serve;
3. runs a permissive live decoder at 0.40/0.30 enter/exit thresholds, 0.25-second minimum live
   duration, and 0.5-second gap bridging;
4. retains a permissive interval only when a serve detection is nearby and the interval is at
   most five seconds;
5. adds no unconditional fixed-duration fallback—the validation search selected zero seconds.

The artifact records `predictionTask: serve-contact`. Older artifacts without that field load as
`rally-live`. Inference rejects role swaps, incompatible feature signatures, disagreement between
the two artifacts' recorded training-manifest hashes, or a serve artifact selected against a
different rally-model hash. Dataset evaluation additionally verifies the current manifest digest;
standalone inference has no current manifest to compare.

The heads do use different coefficients: only 6 of their 15 largest absolute standardized
coefficients overlap. The serve head places large negative weight on central motion at a candidate
sample's -1-second context and positive weight on central difference/motion at +2 seconds, which
is consistent with a setup-lull-to-play transition. Because positive candidate samples span ±1
second around contact, those offsets cover roughly 0–2 seconds before and 1–3 seconds after the
true contact across the positive window. Coefficients are associations among correlated features,
not causal importance.

## Validation result (tuning only)

The contact detector at a one-second tolerance matched 55/76 serves (0.573 precision, 0.724
recall, 0.640 F1). It matched 11/21 short-rally contacts, 7/10 ace contacts, and 7/13 service-fault
contacts.

| Metric | Full model | Composed | Delta |
|---|---:|---:|---:|
| Event precision | 0.516 | 0.510 | -0.006 |
| Event recall | 0.632 (48/76) | 0.684 (52/76) | +0.053 |
| Event F1 | 0.568 | 0.584 | +0.016 |
| Time IoU | 0.503 | 0.504 | +0.001 |
| Live-time recall | 0.889 | 0.917 | +0.028 |
| Missed live seconds | 49.91 | 37.34 | -12.57 |
| Dead seconds retained | 346.27 | 369.03 | +22.76 |

| Validation slice | Full strict / overlap | Composed strict / overlap |
|---|---:|---:|
| At most 3 s | 4/21 / 17/21 | 5/21 / 18/21 |
| Ace | 2/10 / 7/10 | 4/10 / 9/10 |
| Service fault | 3/13 / 10/13 | 4/13 / 11/13 |
| Ordinary long | 43/51 / 51/51 | 44/51 / 51/51 |

The short, ace, and service-fault slices overlap and therefore must not be added together.

## Retrospective test result

At one second, the serve head matched 33/39 contacts (0.635 precision, 0.846 recall, 0.725 F1).
Slice contact recall was 7/10 short rallies, 4/4 aces, 4/7 service faults, and 25/28 ordinary
long rallies. The stricter half-second recall was 23/39.

| Metric | Full model | Composed | Delta |
|---|---:|---:|---:|
| Event precision | 0.560 | 0.519 | -0.041 |
| Event recall | 0.718 (28/39) | 0.718 (28/39) | 0.000 |
| Event F1 | 0.629 | 0.602 | -0.027 |
| Time IoU | 0.671 | 0.657 | -0.014 |
| Live-time recall | 0.927 | 0.936 | +0.010 |
| Missed live seconds | 23.60 | 20.52 | -3.09 |
| Dead seconds retained | 123.08 | 136.96 | +13.88 |

| Test slice | Full strict / overlap / 95% coverage | Composed strict / overlap / 95% coverage |
|---|---:|---:|
| At most 3 s | 1/10 / 8/10 / 6/10 | 1/10 / 9/10 / 8/10 |
| Ace | 2/4 / 4/4 / 3/4 | 2/4 / 4/4 / 3/4 |
| Service fault | 0/7 / 5/7 / 4/7 | 0/7 / 6/7 / 6/7 |
| Ordinary long | 26/28 / 27/28 / 16/28 | 26/28 / 27/28 / 19/28 |

The exploratory, post-selection gain is coverage: one more short/service-fault rally overlaps a
retained interval and two more short/fault rallies reach 95% coverage. The validation objective
selected strict short-event recall, overall F1, and overall recall—not 95% coverage—and those
headline held-out metrics did not improve. The cost is 13.88 extra dead seconds and four extra
predicted intervals. Boundary-sensitive IoU recall does not improve.

The recovered test fault was already represented by the full model's permissive interval from
332.108 to 334.858 seconds; the serve head gated and anchored it rather than discovering wholly
new evidence. A simple one-second symmetric padding baseline reaches 0.968 live-time recall and
8/10 short-rally 95% coverage, but retains 209.88 dead seconds. The paired model is more selective
(0.936 live-time recall, 8/10 coverage, and 136.96 dead seconds) and finds one additional short
overlap, but does not dominate padding on every metric.

## Decision and next experiment

Do not enable the composition by default. Keep the serve model and optional paired inference as
a reproducible diagnostic. The contact head is strong enough to justify a second iteration, but
that iteration needs a better end decision rather than a more aggressive fixed crop.

The [audiovisual follow-up](./serve-specialist-audiovisual-experiment-2026-08-11.md) completes the
next feature iteration. It recovers strict short events but still fails the precision/F1 guardrail.

Recommended next steps:

1. compare against a rally-only permissive rescue and equal-footage padding before attributing a
   gain to the separate serve head;
2. use the serve peak to open a short-event state and train a separate dead-ball/cadence-collapse
   head to close it;
3. repeat the contact experiment with the audiovisual feature branch, particularly onset,
   transient-cadence, and post-contact collapse features;
4. let a serve peak split a long permissive interval that may be swallowing multiple rallies;
5. evaluate paired source-group-out folds and require gains in at least three development groups;
6. collect untouched indoor, grass, and beach sources before deciding whether to ship;
7. retain coverage and footage-cost metrics alongside strict IoU—short faults can be usefully
   saved even when a fixed-duration prediction cannot match their exact boundary.

## Reproduction and artifacts

```bash
.venv/bin/python -m analysis train-serve \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-percentile-v1 \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-v2 \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/full-percentile-v1 \
  --target-radius 1.0 --epochs 120 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serve-specialist-v2-validation.json

.venv/bin/python -m analysis evaluate-serve \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-percentile-v1 \
  --serve-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-v2 \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/full-percentile-v1 \
  --split test \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serve-specialist-v2-test.json
```

- Serve model metadata SHA-256: `be49ee17bf188fb7f62986afaa1152d00e85c72ae4ba62265571793bdf30de5a`
- Serve weights SHA-256: `fe066e03a1e7317fc393509b2b8995ea5c73b56ad8735a306043912eaca4595d`
- Validation report SHA-256: `b8124115183bbb34756f1fd73aba268c8b1a28d7e87a13721728eab57d29d6eb`
- Test report SHA-256: `191cd8dfe5fc6f58eb6945f5dcc09ebf46f34fb3805151bff59f1eba99aa5133`

All destinations are immutable: reruns must use new artifact names.
