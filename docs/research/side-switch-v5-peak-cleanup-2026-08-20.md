# Side-switch V5 peak/count/context cleanup — 2026-08-20

## Decision

Use local peak suppression as the next cadence-free decoder direction. Keep a soft
post-six count penalty as the most promising follow-up for new reviewed recordings.
Do **not** turn the production heads into hard gates, and do not promote the new
production-context auxiliary head: its validation gain does not transfer reliably to
the user-reviewed raw-phone scope.

The validation-selected mechanism is local peak suppression plus a `0.25` production
context log-odds weight. It cuts retrospective proposals from 91 to 60, but exact F1 is
essentially unchanged at 44.21% versus 44.44% for unconstrained no-cadence inference.
It improves ±1/±2 F1 to 54.74%/56.84% by removing nearby duplicates, at a cost of exact
recall from 80.00% to 60.00%.

The locked local-peak + soft-count ablation is the strongest raw-phone diagnostic:
62 proposals, 38.71% precision, 68.57% recall, 49.48% exact F1, and 57.73% ±1/±2 F1.
It did not win the historical validation selection and therefore cannot replace the
selected mechanism after retrospective labels are opened. It is the preregistered
candidate to confirm on new user-reviewed sets.

No mechanism is promoted to automatic use. The raw-phone labels are correct within the
reviewed gap inventory, but that inventory is candidate-conditioned rather than
exhaustive full-video truth, and every raw-phone recording was opened by earlier
side-switch work.

## Were production heads gates before this experiment?

No. The name `original:state-gate` is historical shorthand, not its execution
semantics. In the no-cadence V5-state model, ten values derived from the two production
rally/dead-state bundles—including raw gap duration—are ordinary inputs to the same
32-weight logistic classifier as the V5 appearance features. A weak production value
can lower the probability, but no threshold makes a gap categorically ineligible.

Actual hard gates were already evaluated and rejected:

- requiring both production models on both adjacent rallies reduced exact recall from
  31.43% to 11.43% and exact F1 to 15.09%;
- requiring serve support on both sides reduced exact F1 to 21.54%; and
- requiring both conditions reduced exact F1 to 12.00%.

The raw-phone recordings contain valid one-model rally intervals that are uncommon in
the historical sets. Production support must remain graded evidence.

## New production-context use

This experiment adds a separate `PRODUCTION-CONTEXT19` auxiliary linear head using:

- nine production rally/dead-state values from the state bank;
- all ten production serve/support/anchor values; and
- no raw gap-duration value.

The head is fitted on the same six training recordings. L2 is selected by
recording-held-out training AP. Its probability is never thresholded as a gate. Instead,
the decoder adds a validation-selected fraction of its log odds to the frozen
no-cadence V5-state log odds:

```text
combined_logit = v5_state_logit + context_weight * production_context_logit
```

Weights `0.25`, `0.5`, and `1.0` are compared. Weight zero is the exact no-cadence
control. Every candidate remains eligible, `productionContextIsHardGate` is frozen
`false`, and the suppression model remains quarantined.

The auxiliary head is weak by itself:

| Production-context-only ranking | AP |
| --- | ---: |
| Recording-held-out training | 21.19% |
| Historical validation | 19.49% |
| Raw-phone retrospective | 14.18% |

This drop explains why a small validation cleanup does not transfer as an exact-event
improvement. The production heads are useful as context features, but this experiment
does not support using them as a stronger veto or ranking head.

## Peak and count decoder

All settings are cadence-free and score every frozen reviewed gap. There are no
seven-point opportunity centers, candidate margins, re-anchoring transitions, or hard
maximum count.

Local peak suppression is score-ranked non-maximum suppression:

- the higher-scoring candidate wins;
- `minimumGapSeparation = 2` suppresses only immediately adjacent gap orders; and
- time-separation candidates of 0, 30, and 60 seconds are compared on validation.

Validation selects zero additional time separation in every winning mechanism. The
adjacent-gap rule alone removes duplicate clusters without assuming how many rallies
were correctly counted.

The count prior is deliberately soft. The first six selections in a set are free,
matching the stated usual maximum. Selection seven and later pay an increasing logit
penalty of `penalty * excess_count`. Penalties `0.25`, `0.5`, and `1.0` are compared.
A sufficiently strong seventh or later candidate can still be emitted; there is no
hard cap and no selection changes the location of later candidate windows.

## Frozen validation selection

The eight mechanism families below are selected only on the four historical validation
recordings. Each family searches its declared parameter grid and classifier threshold.
The global validation winner remains locked before retrospective evaluation.

| Mechanism | P | R | Exact F1 | Proposals |
| --- | ---: | ---: | ---: | ---: |
| Independent control | 48.15% | 76.47% | 59.09% | 27 |
| Production context | 50.00% | 76.47% | 60.47% | 26 |
| Soft count | **72.73%** | 47.06% | 57.14% | 11 |
| Soft count + context | 45.16% | **82.35%** | 58.33% | 31 |
| Local peak | 59.09% | 76.47% | 66.67% | 22 |
| **Local peak + context (selected)** | 61.90% | 76.47% | **68.42%** | 21 |
| Local peak + soft count | 60.00% | 70.59% | 64.86% | 20 |
| Local peak + soft count + context | 57.89% | 64.71% | 61.11% | 19 |

The selected decoder uses adjacent-gap peak suppression, zero time separation, no
count penalty, production-context weight `0.25`, and threshold
`0.4367987574848957`.

## Raw-phone retrospective result

All eight rows below use the settings locked within their validation family. The bold
selected row is the global historical-validation winner; the local-peak + soft-count
row is a retrospective diagnostic and cannot be substituted after label opening.

| Mechanism | P | R | Exact F1 | ±1 F1 | ±2 F1 | Proposals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No-cadence control | 30.77% | **80.00%** | 44.44% | 47.62% | 47.62% | 91 |
| Production context | 33.33% | 68.57% | 44.86% | 48.60% | 50.47% | 72 |
| Soft count | **43.18%** | 54.29% | 48.10% | 50.63% | 53.16% | **44** |
| Soft count + context | 30.59% | 74.29% | 43.33% | 48.33% | 50.00% | 85 |
| Local peak | 35.62% | 74.29% | 48.15% | 55.56% | 55.56% | 73 |
| **Local peak + context (selected)** | 35.00% | 60.00% | 44.21% | 54.74% | 56.84% | 60 |
| Local peak + soft count | 38.71% | 68.57% | **49.48%** | **57.73%** | **57.73%** | 62 |
| Local peak + soft count + context | 37.50% | 60.00% | 46.15% | 57.14% | 57.14% | 56 |

For reference, the old cadence V5-state control has 38 proposals and 30.14% exact F1.
Every cleanup mechanism remains materially better than cadence; even the weakest row
by exact F1, soft count + context, exceeds cadence by 13.20 points.

## What caused the cleanup?

### Local peaks transfer

Local peak suppression alone reduces 91 proposals to 73. It removes 16 false positives
while losing only two exact true positives, raising exact F1 from 44.44% to 48.15%.
It eliminates all 22 adjacent selected pairs from the independent output. This is the
most stable result: the same mechanism improves both validation and retrospective
metrics without recreating a seven-point path.

### Soft count is promising but not confirmed

Adding the validation-selected `0.25` post-six logit penalty to local peaks reduces the
raw-phone result to 62 proposals and raises exact F1 to 49.48%. It removes nine more
false positives and two more true positives relative to local peaks. Historical
validation preferred no count penalty, so this gain needs confirmation on newly
reviewed sets rather than promotion from the opened scope.

### Production context does not transfer cleanly

Production context removes one false positive on validation when added to local peaks,
which makes it the formal winner. On raw-phone data, however, it changes local peaks
from 73 proposals/26 true positives to 60 proposals/21 true positives. Exact F1 falls
from 48.15% to 44.21%, and row AP falls from 44.30% to 42.91%.

The context score is therefore too domain-sensitive to serve as a gate or strong veto.
It can remain visible as an explanatory score or weak feature, but the next decoder
should not depend on it to establish eligibility.

## Recommended next confirmation

On newly user-reviewed games, freeze and test exactly two cadence-free decoders before
opening labels:

1. local adjacent-gap peak suppression; and
2. the same peak suppression plus the post-six `0.25` logit penalty.

Keep the production-context weight at zero for that confirmation. Report proposal
count, exact/±1/±2 event metrics, and per-recording count error. If the soft-count row
continues to improve precision without a large recall loss, it is the better on-device
decoder. Otherwise retain local peaks alone.

## On-device cost

Peak suppression and the soft count prior require only sorting a small set of gap
scores and simple scalar comparisons. They add no video decode, visual model, or cadence
state. The optional context head adds 19 weights over outputs already computed by the
production ensemble, but its transfer result does not justify a runtime port.

No browser or Android inference implementation and no review-UI timeline are added.

## Reproduction

The feature artifact and no-cadence V5-state head are already frozen:

```bash
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v5-peak-cleanup.py freeze
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-v5-peak-cleanup.py evaluate
PYTHONPATH=. .venv/bin/python scripts/build-side-switch-v5-peak-cleanup-provenance.py \
  --implementation-revision 8926e69
```

Every destination refuses overwrite.

## Immutable artifacts

All paths are under `/mnt/freenas/volleycut/labeling-v1-2026-08-09/`.

| Artifact | SHA-256 |
| --- | --- |
| `models/side-switch-v5-peak-cleanup-v1/model.json` | `82e64c69564d17c3dfbea9c5f4a276cdd161a0c400507d3df7589c019e44ee3f` |
| `models/side-switch-v5-peak-cleanup-v1/dataset-development.json` | `cc249e6ca75b18dc038bace1c3a26fa5f252f1f32a626eb409bcacae435bf0d0` |
| `reports/side-switch/side-switch-v5-peak-cleanup-v1-evaluation.json` | `d38d85e6b4fec379786fbab390dd111e3b0cfc212adf6d2fcb95b6bba936c89f` |

- selected fingerprint:
  `d37839e0031facd6913d904fd7c81624b7f3d9ce57567a1b2f182d3c0aac0190`;
- implementation revision:
  `8926e69e12bee18b4fd64cea92f621ede81bc201`; and
- provenance:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v5-peak-cleanup-v1-provenance.json`,
  SHA-256 `0782d18b53741ba8f63270986c432278ac2d43ff7d8088babfe349acd07d47f9`.
