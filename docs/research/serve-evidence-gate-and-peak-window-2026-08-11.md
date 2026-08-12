# Serve-evidence gate and peak-window rally feature — 2026-08-11

## Outcome

Two related serve-evidence variants were tested against the frozen audiovisual rally and serve
models:

1. remove the serve composer's unconditional two-second fallback and require primary or
   permissive rally evidence; and
2. train a rally head with four decoded serve-peak features covering a symmetric two-second
   window.

Neither variant passes its validation promotion guardrails. The evidence gate is nevertheless a
useful retrospective diagnostic: on the previously inspected test source it removes eight false
intervals without losing a strict match, raising event F1 from 0.633 to 0.689. On validation it
removes 14 intervals but loses one strict short ace, so it remains an experimental configuration
rather than the inference default.

The learned peak-window head is not suitable for promotion. It gains three strict service-fault
matches on validation and two on test, but reduces live-time recall and loses ordinary-long
rallies. Keep the frozen audiovisual rally model and v4 serve artifact unchanged.

## What was tested

The evidence gate reuses the exact frozen v4 serve decoder and permissive rally decoder. It changes
only the composition settings:

- `associationSeconds`: 1.0 to 2.0;
- `fallbackSeconds`: 2.0 to 0.0; and
- `maxRescueSeconds`: unchanged at 5.0.

With a zero fallback, a serve peak can still anchor a primary interval or rescue a nearby short
permissive interval. An unsupported peak produces no interval. The association predicate is not
strictly symmetric: it accepts a peak up to two seconds before a predicted interval or anywhere
inside that interval.

The learned rally head has the original 450 contextual audiovisual inputs plus four derived
columns:

- binary membership within two seconds of the nearest decoded serve peak;
- triangular proximity, `max(0, 1 - abs(delta)/2)`;
- signed offset, `delta/2`, inside that window; and
- decoded peak confidence inside that window.

These are constructed from predicted peaks only. Whether a peak matches a gold serve within two
seconds is used exclusively for evaluation and is never an input.

Training features are leakage-safe within the available train split. Each of the three train
source groups receives peaks decoded from a serve head fit on the other two groups. Validation and
test use the frozen full-training `serve-specialist-audiovisual-v4`. The 454-input rally head uses
the same class-weighted logistic trainer as the baseline; validation selected epoch 20 and a
decoder with 0.5-second smoothing, 0.70/0.55 enter/exit thresholds, two-second minimum live time,
and one-second gap bridging.

## Validation result

Validation contains 76 truth rallies, including 21 at most three seconds, 10 aces, 13 service
faults, and 51 ordinary-long rallies. It is tuning evidence from one previously reused grass source
group.

| Variant | Predicted / strict | F1 | Time IoU | Live recall | Dead retained | Short | Fault | Ordinary long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen/matched rally control | 88 / 52 | 0.634 | 0.547 | 0.909 | 297.99 s | 3/21 | 1/13 | 46/51 |
| Frozen v4 composition | 112 / 56 | 0.596 | 0.532 | 0.938 | 343.05 s | 6/21 | 3/13 | 46/51 |
| Evidence gate, no fallback | 98 / 55 | 0.632 | 0.547 | 0.935 | 319.00 s | 5/21 | 3/13 | 46/51 |
| Peak-window head, frozen decoder | 88 / 50 | 0.610 | 0.548 | 0.912 | 299.28 s | 3/21 | 2/13 | 44/51 |
| Peak-window head, selected decoder | 94 / 56 | 0.659 | 0.570 | 0.863 | 230.70 s | 6/21 | 5/13 | 46/51 |
| Peak features clamped to train means | 93 / 51 | 0.604 | 0.552 | 0.839 | 233.28 s | 4/21 | 3/13 | 44/51 |

The gate fails the predeclared requirement to preserve both overall and short strict matches. The
peak head improves F1 and outcome recall, and its five-match advantage over the clamped ablation
shows that the derived bundle affects decisions. It fails the live-recall guardrail: missed live
footage rises from 41.07 to 61.79 seconds.

At a two-second contact tolerance, validation has 72 matched and 28 unmatched serve peaks. Of the
28 unmatched peaks, 12 are unsupported and dropped by the gate, seven have permissive evidence,
and nine occur in primary predictions. Two matched peaks are also unsupported, explaining why an
evidence-only gate cannot be assumed to preserve every true serve.

## Retrospective test result

The one-source test contains 39 truth rallies and has been inspected in several earlier
experiments. These numbers are regression diagnostics, not fresh generalization evidence.

| Variant | Predicted / strict | F1 | Time IoU | Live recall | Dead retained | Short | Fault | Ordinary long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen rally baseline | 45 / 29 | 0.690 | 0.672 | 0.910 | 114.52 s | 0/10 | 0/7 | 28/28 |
| Frozen v4 composition | 59 / 31 | 0.633 | 0.652 | 0.939 | 142.05 s | 2/10 | 0/7 | 28/28 |
| Evidence gate, no fallback | 51 / 31 | 0.689 | 0.675 | 0.939 | 126.05 s | 2/10 | 0/7 | 28/28 |
| Peak-window head, frozen decoder | 43 / 29 | 0.707 | 0.675 | 0.920 | 116.98 s | 0/10 | 0/7 | 28/28 |
| Peak-window head, selected decoder | 50 / 28 | 0.629 | 0.700 | 0.867 | 76.94 s | 2/10 | 2/7 | 25/28 |
| Peak features clamped to train means | 49 / 26 | 0.591 | 0.681 | 0.847 | 79.05 s | 0/10 | 0/7 | 25/28 |

The evidence gate removes exactly the eight unsupported false serve intervals identified in the
test audit. It recovers no truth and loses no truth relative to v4. Of the ten serve peaks unmatched
within two seconds, eight are dropped, one has permissive evidence, and one falls inside a false
primary prediction.

The selected-decoder peak head recovers two short service faults:

- `267.224–268.931`, duration 1.707 seconds, IoU 0.528; and
- `733.500–735.166`, duration 1.666 seconds, IoU 0.515.

It loses three ordinary-long rallies of 36.426, 6.888, and 17.394 seconds. Relative to the frozen
baseline, false intervals also increase from 16 to 22. The outcome gain is therefore not an
acceptable replacement for full-rally recall.

The standalone serve detector finds 37/39 test starts within two seconds from 47 peaks: 78.7%
precision, 94.9% recall, and 0.860 F1. This tolerance describes evaluation only; it does not prove
that every nearby peak is a correctly timed serve contact.

## Decision

Do not promote either variant from this corpus. The low-risk production hypothesis is still the
zero-fallback evidence gate, because its retrospective behavior is event-neutral and less noisy,
but the validation miss must be resolved on new source groups before enabling it. The peak-window
rally head should remain an explicit research artifact.

The next dataset increment should test the gate without revising its parameters and should record
which real serves lack even permissive live evidence. For the learned path, a sequence model or an
explicit state transition conditioned on a serve is more appropriate than asking one linear head
to trade short-event rescue against long-rally coverage.

## Reproduction and immutable artifacts

The frozen semantic manifest digest is
`67cfcf46a94dad05a38aa1fdb146059cc21cff8a906cf08abc4989f4980d20db`.

Model artifact hashes:

- rally: `e08a3db18e4fd81fc6f5c10816b7c17f4c7661d4424a98d800aa0b1b0ccfd6a0`;
- serve: `e28af5f43e78f9708676c046cc5636d1a7276a0e176042ea40eada49aa0a36e7`;
- matched rally control: `8e928a470947404930a5d560bf7049814bbe3b1898e34fd069655d277690f8df`;
  and
- peak-window rally: `e4567657d7bf47ef831d30485046fc4b505c58f1e50d5ce3234026de8548e8ff`.

Peak-window files:

- `model.json`: `41b515eea9cb014c06dace298358b5bd01564630264c5fd92d4dd3ba3ea1ea28`;
- `weights.npz`: `4a5ffbd912d702c7396b58ad40c41aea98cb7874cefac0405250941b82f389e1`;
- reviewed validation report:
  `f484f26b382fa832a8733c5f77db5169fe99cca18e481df06a2eb3c82fe87a40`;
  and
- reviewed retrospective-test report:
  `a71ed6b6d8bc85aafef7bc96a825fa9b4aabcafbc99b559b6a83f47c1b5afefa`.

The model is stored at
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-peak-window-v1-exploratory`.
The reviewed reports are `serve-evidence-v1-reviewed-validation.json` and
`serve-evidence-v1-reviewed-retrospective-test.json` under the corpus `reports` directory.

```bash
PYTHONPATH=. .venv/bin/python -m analysis train-serve-evidence \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --baseline-rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-v2-final \
  --serve-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-audiovisual-v4 \
  --control-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-prob-stack-v1-control \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-peak-window-v1-exploratory \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serve-evidence-v1-validation.json \
  --epochs 180 --batch-size 2048 --learning-rate 0.02 --seed 7

PYTHONPATH=. .venv/bin/python -m analysis evaluate-serve-evidence \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --baseline-rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-v2-final \
  --serve-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-audiovisual-v4 \
  --control-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-prob-stack-v1-control \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-serve-peak-window-v1-exploratory \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2 \
  --split test \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serve-evidence-v1-reviewed-retrospective-test.json
```

Artifacts are immutable; use new destinations when reproducing these commands.
