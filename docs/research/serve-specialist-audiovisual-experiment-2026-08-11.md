# Audiovisual serve-specialist experiment — 2026-08-11

## Outcome

The 450-input audiovisual serve head is materially better than the visual-only specialist on
validation and recovers two strict short-rally matches on the retrospective test. It is still not
ready to become the default cutter: test event recall rises from 29/39 to 31/39, but 14 additional
predictions reduce precision from 0.644 to 0.525 and F1 from 0.690 to 0.633.

The useful result is targeted. Short-rally strict recall rises from 0/10 to 2/10 and any-overlap
recall from 4/10 to 9/10. Ace strict recall rises from 1/4 to 3/4. Service-fault contact spotting
reaches 6/7, but strict interval recall remains 0/7, confirming that finding the serve does not
solve the dead-ball boundary.

Keep `serve-specialist-audiovisual-v4` as an opt-in research artifact. The next composition should
control false rescues and learn a short-event end decision.

## Data and protocol

This uses the same immutable `full-gold-v1` manifest as the visual experiment: 230 train, 76
validation, and 39 retrospective-test rallies, with source groups disjoint across splits. The
short, ace, and service-fault slices overlap and are not additive. Outcome tags are evaluation
slices only.

The promoted `full-audiovisual-v2-final` rally model stays frozen. Its 450 contextual inputs come
from 90 base signals at five temporal offsets and include advanced motion/formation/quality proxies
plus audio transient, cadence, RMS, and spectral channels. The serve head uses exactly the same
feature cache and extractor signature, so paired inference still decodes the video once.

Serve positives are samples within one second of each labeled rally start. Training has 1,858
positive and 22,407 negative samples; early stopping selected epoch 13. Validation is reused for
the rally model's earlier selection, the serve head's early stopping, and an 8,640-candidate
composition search. The validation result is therefore tuning evidence, not a generalization
estimate. Test parameters were frozen before the retrospective test run.

The selected contact decoder uses a 0.90 threshold, eight-second non-maximum suppression, and a
+0.25-second correction. Composition uses a one-second association window, a two-second fallback,
and a validation-selected permissive rally decoder with 0.45/0.35 enter/exit thresholds,
0.5-second minimum duration, and no gap bridging.

An initial `v3` run is excluded from all conclusions. It exposed an evaluator integration error:
the paired path used nominal 4 Hz while main's canonical evaluator used the recording's actual
median sample cadence. That changed the untouched test baseline from 45 to 47 intervals. `v4`
uses the canonical cadence and outcome-slice implementation; a regression test now locks parity.

## Serve-contact result

| Split | Visual-only v2 | Audiovisual v4 | Audiovisual short | Audiovisual ace | Audiovisual fault |
|---|---:|---:|---:|---:|---:|
| Validation recall at ±1 s | 55/76 | 68/76 | 18/21 | 8/10 | 12/13 |
| Retrospective-test recall at ±1 s | 33/39 | 32/39 | 8/10 | 3/4 | 6/7 |

Audiovisual validation contact precision/recall/F1 is 0.680/0.895/0.773. Test is
0.681/0.821/0.744, versus 0.635/0.846/0.725 for visual-only v2. The new head is more selective and
much stronger on validation short/fault contacts, but the one-source test comparison is mixed.

Two of the serve head's 15 largest absolute coefficients are audio channels:
`t+1s/audio_seconds_since_transient` and `t+1s/audio_onset_cadence`. Advanced visual motion spread,
collapse, formation-change, camera-response, and spatial-entropy channels dominate the rest. This
shows that the full head uses the added feature bundle; it does not isolate a causal contribution
from audio. An audio-off/advanced-visual-on ablation is still required.

## Validation result (tuning only)

| Metric | Frozen audiovisual | Composed | Delta |
|---|---:|---:|---:|
| Predicted / matched rallies | 88 / 52 | 112 / 56 | +24 / +4 |
| Event precision | 0.591 | 0.500 | -0.091 |
| Event recall | 0.684 | 0.737 | +0.053 |
| Event F1 | 0.634 | 0.596 | -0.038 |
| Time IoU | 0.547 | 0.532 | -0.014 |
| Live-time recall | 0.909 | 0.938 | +0.029 |
| Missed live seconds | 41.07 | 27.88 | -13.19 |
| Dead seconds retained | 297.99 | 343.05 | +45.06 |

| Slice | Frozen strict / overlap / 95% coverage | Composed strict / overlap / 95% coverage |
|---|---:|---:|
| At most 3 s | 3/21 / 17/21 / 11/21 | 6/21 / 21/21 / 16/21 |
| Ace | 5/10 / 8/10 / 6/10 | 7/10 / 10/10 / 8/10 |
| Service fault | 1/13 / 10/13 / 6/13 | 3/13 / 13/13 / 9/13 |
| Ordinary long | 46/51 / 51/51 / 31/51 | 46/51 / 51/51 / 36/51 |

## Retrospective test result

| Metric | Frozen audiovisual | Composed | Delta |
|---|---:|---:|---:|
| Predicted / matched rallies | 45 / 29 | 59 / 31 | +14 / +2 |
| Event precision | 0.644 | 0.525 | -0.119 |
| Event recall | 0.744 | 0.795 | +0.051 |
| Event F1 | 0.690 | 0.633 | -0.058 |
| Time IoU | 0.672 | 0.652 | -0.020 |
| Live-time recall | 0.910 | 0.939 | +0.029 |
| Missed live seconds | 29.00 | 19.60 | -9.40 |
| Dead seconds retained | 114.52 | 142.05 | +27.53 |

| Slice | Frozen strict / overlap / 95% coverage | Composed strict / overlap / 95% coverage |
|---|---:|---:|
| At most 3 s | 0/10 / 4/10 / 4/10 | 2/10 / 9/10 / 5/10 |
| Ace | 1/4 / 2/4 / 2/4 | 3/4 / 4/4 / 3/4 |
| Service fault | 0/7 / 3/7 / 3/7 | 0/7 / 6/7 / 3/7 |
| Ordinary long | 28/28 / 28/28 / 16/28 | 28/28 / 28/28 / 18/28 |

A one-second symmetric padding control retains 189.41 dead seconds yet leaves short overlap at
4/10 and strict short recall at 0/10. The specialist reaches 9/10 short overlap with 142.05 dead
seconds, so its targeted rescue is not reproduced by simple padding. Padding still has higher
overall F1 (0.651) and live-time recall (0.956), which is why neither approach is a complete win.

## Decision and next steps

Do not promote the composition. Preserve it for comparison and pursue:

1. a validation objective or gating head that explicitly prices false rescues;
2. a learned cadence-collapse/dead-ball head to close detected short events;
3. an audio-off ablation and grouped source-out serve study to isolate feature-family value;
4. splitting a permissive interval when a later high-confidence serve indicates two swallowed
   rallies;
5. untouched indoor, grass, and beach sources before any product decision.

## Reproduction and artifacts

```bash
/home/developer/volleycut/.venv/bin/python -m analysis train-serve \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-v2-final \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-audiovisual-v4 \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2 \
  --target-radius 1.0 --epochs 120 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serve-specialist-audiovisual-v4-validation.json

/home/developer/volleycut/.venv/bin/python -m analysis evaluate-serve \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-v2-final \
  --serve-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-audiovisual-v4 \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2 \
  --split test \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serve-specialist-audiovisual-v4-test.json
```

- Rally artifact SHA-256: `e08a3db18e4fd81fc6f5c10816b7c17f4c7661d4424a98d800aa0b1b0ccfd6a0`
- Serve artifact SHA-256: `e28af5f43e78f9708676c046cc5636d1a7276a0e176042ea40eada49aa0a36e7`
- Serve metadata SHA-256: `151e491ca4d3f6b65c2ffccd914212bcc52b77226fc7e696a5945bab119c556d`
- Serve weights SHA-256: `add2eab0339f4f8e3f330c09e4fea64509455f52fe30f25b674a681a5c787436`
- Validation report SHA-256: `af86cf447cc9aea44529c5f4268959f8d3644874da26016a325e5f219792d4bb`
- Test report SHA-256: `24f7eacf733675097a873902e02e6e424d56dd872ae685a93c4793a47d9c8323`

All destinations are immutable. Use new names for future runs.
