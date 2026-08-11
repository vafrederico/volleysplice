# Rally model with serve-probability feature — 2026-08-11

## Outcome

Appending the serve specialist's continuous probability as a 451st rally-model input does not
improve the retrospective test result. The validation-selected stacked model gains one strict
match on validation, including three additional service faults, but it gives up too much live
footage. On test it recovers one short ace, loses five ordinary-long rallies, and lowers event F1
from 0.690 to 0.602. Service-fault strict recall remains 0/7.

Do not promote `full-audiovisual-serve-prob-stack-v1-exploratory`. The frozen audiovisual rally
model remains the primary model; the post-decoder serve composition remains the more useful
serve-specialist experiment, though that composition is also not ready for promotion.

## What the audiovisual serve composition still misses

There are two distinct meanings of “missed.” At a one-second contact tolerance, the audiovisual
serve head misses seven starts:

| Truth start | Duration | Classification | Nearest serve-score peak error |
|---:|---:|---|---:|
| 123.500 s | 6.671 s | ordinary long | +1.250 s |
| 203.500 s | 10.196 s | ordinary long | -11.517 s |
| 348.500 s | 11.203 s | ordinary long | -1.250 s |
| 439.950 s | 2.722 s | short ace | -1.200 s |
| 484.822 s | 9.910 s | ordinary long | +1.161 s |
| 747.209 s | 6.888 s | ordinary long | +1.274 s |
| 1007.571 s | 1.564 s | short service fault | +15.179 s |

That is five ordinary-long rallies, one short ace, and one short service fault. Durations range
from 1.564 to 11.203 seconds; the median is 6.888 seconds.

After serve-anchored composition, eight truth intervals still fail strict IoU >= 0.5. Every one is
a short outcome:

| Truth interval | Duration | Classification | Best IoU | Truth coverage |
|---|---:|---|---:|---:|
| 191.716–193.344 s | 1.628 s | service fault | 0.434 | 83.6% |
| 267.224–268.931 s | 1.707 s | service fault | 0.488 | 100.0% |
| 316.308–317.148 s | 0.840 s | service fault | 0.131 | 47.4% |
| 332.452–333.795 s | 1.343 s | service fault | 0.071 | 12.3% |
| 439.950–442.672 s | 2.722 s | ace | 0.358 | 100.0% |
| 733.500–735.166 s | 1.666 s | service fault | 0.446 | 100.0% |
| 959.500–962.025 s | 2.525 s | service fault | 0.169 | 100.0% |
| 1007.571–1009.135 s | 1.564 s | service fault | 0.000 | 0.0% |

The median duration is 1.647 seconds. Six of these eight contacts are found within one second, and
four intervals cover the entire truth event but are too long to satisfy strict IoU. The dominant
failure is therefore the dead-ball boundary, not merely serve recall.

## Stacked experiment design

The specialist and rally heads are logistic models over the same 450 contextual audiovisual
inputs. Appending the serve logit would be algebraically redundant with another linear head, so
this experiment appends the raw sigmoid output as a nonlinear derived feature named
`derived/serve_contact_probability`.

Training scores are leakage-safe within the available training split. For each of the three train
source groups, a serve head is fit on the other two groups for 13 epochs and scores only the held-out
group. The rally head trains on those out-of-fold scores. Validation and test use the frozen full
`serve-specialist-audiovisual-v4` scorer. A matched 450-input control is retrained with the same
optimizer and decoder search. Its weights and predictions exactly reproduce the frozen baseline.

The experiment evaluates five variants:

1. the frozen 450-input audiovisual baseline;
2. the matched 450-input retraining control;
3. the 451-input head with the frozen baseline decoder;
4. the 451-input head with a validation-selected decoder; and
5. the selected 451-input head with its serve feature clamped to the train mean.

The clamp makes the serve feature's standardized contribution zero while preserving the other 450
learned coefficients. It is the closest ablation of whether the dynamic serve score changes the
selected stacked model's decisions.

Validation remains heavily reused: it previously selected the baseline and specialist, then
selects both new heads' epochs and both decoders here. The one-source test has already been
inspected in earlier experiments. Validation is tuning evidence and test is retrospective
regression evidence, not a fresh estimate of generalization.

## Validation result (tuning only)

| Metric | Frozen / matched control | Stacked, frozen decoder | Stacked, selected decoder | Serve feature clamped |
|---|---:|---:|---:|---:|
| Predicted / strict matches | 88 / 52 | 91 / 44 | 85 / 53 | 85 / 52 |
| Event precision | 0.591 | 0.484 | 0.624 | 0.612 |
| Event recall | 0.684 | 0.579 | 0.697 | 0.684 |
| Event F1 | 0.634 | 0.527 | 0.658 | 0.646 |
| Time IoU | 0.547 | 0.511 | 0.567 | 0.567 |
| Live-time recall | 0.909 | 0.901 | 0.786 | 0.788 |
| Live-time precision | 0.579 | 0.542 | 0.670 | 0.669 |
| Missed live seconds | 41.07 | 44.67 | 96.36 | 95.29 |
| Dead seconds retained | 297.99 | 343.10 | 174.23 | 175.36 |

| Slice | Frozen strict / overlap / 95% coverage | Selected stack | Clamped stack |
|---|---:|---:|---:|
| At most 3 s | 3/21 / 17/21 / 11/21 | 5/21 / 14/21 / 7/21 | 4/21 / 14/21 / 6/21 |
| Ace | 5/10 / 8/10 / 6/10 | 5/10 / 8/10 / 1/10 | 5/10 / 8/10 / 1/10 |
| Service fault | 1/13 / 10/13 / 6/13 | 4/13 / 8/13 / 6/13 | 3/13 / 8/13 / 5/13 |
| Ordinary long | 46/51 / 51/51 / 31/51 | 44/51 / 51/51 / 17/51 | 44/51 / 51/51 / 17/51 |

The validation-selected decoder trades 55.29 additional missed-live seconds for 123.76 fewer dead
seconds. The dynamic serve score accounts for only one strict match relative to the clamped
version. Its standardized coefficient is 0.0172, rank 362 of 451 by absolute magnitude.

## Retrospective test result

| Metric | Frozen / matched control | Stacked, frozen decoder | Stacked, selected decoder | Serve feature clamped |
|---|---:|---:|---:|---:|
| Predicted / strict matches | 45 / 29 | 48 / 28 | 44 / 25 | 44 / 25 |
| Event precision | 0.644 | 0.583 | 0.568 | 0.568 |
| Event recall | 0.744 | 0.718 | 0.641 | 0.641 |
| Event F1 | 0.690 | 0.644 | 0.602 | 0.602 |
| Time IoU | 0.672 | 0.651 | 0.633 | 0.632 |
| Live-time recall | 0.910 | 0.893 | 0.734 | 0.733 |
| Live-time precision | 0.719 | 0.706 | 0.822 | 0.822 |
| Missed live seconds | 29.00 | 34.52 | 85.91 | 86.14 |
| Dead seconds retained | 114.52 | 120.07 | 51.19 | 51.19 |

| Slice | Frozen strict / overlap / 95% coverage | Selected stack | Clamped stack |
|---|---:|---:|---:|
| At most 3 s | 0/10 / 4/10 / 4/10 | 1/10 / 4/10 / 3/10 | 1/10 / 4/10 / 3/10 |
| Ace | 1/4 / 2/4 / 2/4 | 2/4 / 3/4 / 3/4 | 2/4 / 3/4 / 3/4 |
| Service fault | 0/7 / 3/7 / 3/7 | 0/7 / 2/7 / 1/7 | 0/7 / 2/7 / 1/7 |
| Ordinary long | 28/28 / 28/28 / 16/28 | 23/28 / 28/28 / 4/28 | 23/28 / 28/28 / 4/28 |

The selected stack's one recovered strict event is a 1.653-second ace. It loses five
ordinary-long events of 36.426, 15.231, 6.888, 15.347, and 6.850 seconds. Clamping the serve
feature produces the same strict matches and outcome coverage; only one ordinary-long start moves
by a single 0.233-second sample. The dynamic feature therefore contributes no held-out event gain.

For context, post-decoder v4 composition reaches 31/39 strict matches and 2/10 strict short
matches, with F1 0.633 and live-time recall 0.939. It is noisier than the baseline but materially
better than putting the same specialist score directly into this linear rally head.

## Decision and next step

Do not integrate the stacked head into ordinary inference. Keep its explicit
`rally-live-stacked-serve` artifact role and dedicated evaluator so it cannot be mistaken for a
450-input rally model.

The miss audit and stack result point to a different next model: retain serve contact as a separate
anchor, then learn or gate the **end** of a short event. A useful candidate would model cadence
collapse/dead-ball probability after a detected serve and optimize short-event coverage together
with dead-time cost. Additional untouched source groups are required before another promotion
decision.

## Reproduction and artifacts

```bash
/home/developer/volleycut/.venv/bin/python -m analysis train-rally-with-serve \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --baseline-rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-v2-final \
  --serve-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-audiovisual-v4 \
  --control-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-prob-stack-v1-control \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-prob-stack-v1-exploratory \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-audiovisual-serve-prob-stack-v1-validation.json \
  --epochs 180 --batch-size 2048 --learning-rate 0.02 --seed 7

/home/developer/volleycut/.venv/bin/python -m analysis evaluate-rally-with-serve \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --baseline-rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-v2-final \
  --serve-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-audiovisual-v4 \
  --control-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-prob-stack-v1-control \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-prob-stack-v1-exploratory \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2 \
  --split validation \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-audiovisual-serve-prob-stack-v1-final-validation.json

/home/developer/volleycut/.venv/bin/python -m analysis evaluate-rally-with-serve \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --baseline-rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-v2-final \
  --serve-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-audiovisual-v4 \
  --control-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-prob-stack-v1-control \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-prob-stack-v1-exploratory \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2 \
  --split test \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-audiovisual-serve-prob-stack-v1-final-retrospective-test.json
```

- Manifest digest: `67cfcf46a94dad05a38aa1fdb146059cc21cff8a906cf08abc4989f4980d20db`
- Frozen rally artifact: `e08a3db18e4fd81fc6f5c10816b7c17f4c7661d4424a98d800aa0b1b0ccfd6a0`
- Frozen serve artifact: `e28af5f43e78f9708676c046cc5636d1a7276a0e176042ea40eada49aa0a36e7`
- Matched-control artifact: `8e928a470947404930a5d560bf7049814bbe3b1898e34fd069655d277690f8df`
- Stacked artifact: `418babe628f9a239fb4c5742a186396a51ce068dca6bf7e4a73b3cef45276360`
- Final validation report: `563d0a7b00323beb457b530b7836f6a1a756898e8ace341c8cbf26ca25f8cd64`
- Final retrospective-test report: `04c15d153ed13b053f7e008691e2fd9fee18691321fb2129ef8ed9cb634e3a47`

Model and final-report destinations are immutable. The earlier reports without `final` in their
names are superseded metadata-only drafts; their metrics are identical. Use new destination names
for any future run.
