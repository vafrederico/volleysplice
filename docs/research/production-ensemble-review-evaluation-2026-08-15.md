# Production two-model ensemble with disagreement review — 2026-08-15

## Decision

The production ensemble of previous production (`model-9c92b8e9333f`) and all-labels
v2 (`model-1ca43e38eefc`) is a useful **recall-safety workflow**, but it is not the best
fully automatic `F1_padP_coreR` system. At the declared two-second product padding, the
reviewed ensemble reaches **99.87% pooled evaluation recall**, compared with 98.71% for
all-labels v2 and 97.83% for previous production. Its 69.65% precision remains below
v2's 72.05%, leaving reviewed-ensemble F1 at 82.07% versus 83.30% for v2.

The intended product interpretation is therefore: use all-labels v2 as the primary
stack, use previous production as a recall challenger, and require disagreement review
before export. The unreviewed union is not acceptable at the target padding: its
precision is 61.16% and its F1 is 75.86%, below both standalone models.

This is an evaluation of the current whole-range merge behavior. It is not evidence
that model agreement guarantees correctness. On the three evaluation recordings, 19
zero-core-overlap components were emitted by both models and therefore were not marked
as disagreements. Preserving provenance for model-only boundary wings is the clearest
next precision improvement.

## Scope and frozen inputs

The declared product target is symmetric **2-second** padding. Zero, one, and three
seconds are required sensitivity cases. All cases use the canonical short-gap join
threshold of **3 seconds**: a positive gap strictly shorter than 3 seconds is retained
and joined, while an exact 3-second gap remains a cut.

Model selection evidence is limited to these evaluation-only recordings:

- `grass-source-06` — grass evaluation, label SHA-256
  `009db2c64aaf1b696e1c201810e5398bf1af0d99e854050c6946c3c419983406`
- `grass-source-08` — grass evaluation, label SHA-256
  `b0c049a716db3622a1f675e3f1b719f5f1afb739a49f53f04a14828ddbfbde23`
- `indoor-source-03` — indoor evaluation, label SHA-256
  `f69d0a6212e8f1ff0b6b2ddf88b96d07331b658266f32cef0a8805df44994870`

The historical protected recording `indoor-source-05`, label SHA-256
`48fd12d57a2b0265609421a65f108c3e3b39c2c67a76fd28622371f25248d1df`,
is reported separately and was not used to select the ensemble, model, decoder,
threshold, padding, or review policy.

The grass evaluation documents are continuously reviewed human drafts but remain
`in-progress`. The indoor evaluation document for `indoor-source-03` still
identifies its annotator as `GPT-5.6 Sol high (unvalidated)`, so indoor evaluation
results are provisional. The protected label is human-complete.

The exact analysis-input SHA-256 values are:

| Recording | Previous production analysis | All-labels v2 analysis |
| --- | --- | --- |
| `grass-source-06` | `fada990bc6f2f9c61d516a58505dd1a3c197ef84d039d39a931d4a63d1331565` | `08fd08a24b41312e180a2a8eb2f8f4fa80220df340edea88abecac0a9c22362f` |
| `grass-source-08` | `ce848858efe3f3f930f5f638283411705ccb2f05bbb6e39ba3a923f570b3ce61` | `22576141d172136dbf1e928f4f89d6a37e10f195ee28de2d804f9501321fad21` |
| `indoor-source-03` | `dc17d6633b7144ffa92b27d6dc1f767f16b206b09cc4b2c777594a84cec0ae24` | `350c60df6277e8e1312e2772957441dd73b76b6d24543dae961d7c606ad7d2e9` |
| `indoor-source-05` | `4f9c61cd6fc628be1a626f59a062d08f2a27a05d0db2ea3888f2b46342c4d202` | `9e30738b19383b53c8f1cf7521b19d03986460fe096a858d6f742ce7454d82a3` |

## Review simulation

The simulation reproduces the production `overlap-union-disagreement-v1` behavior:

1. Included intervals from both models are ordered chronologically.
2. Transitively overlapping intervals form one component. Touching intervals do not
   establish model agreement.
3. A component containing both sources is emitted from the earliest start through the
   latest end and marked `both-models`.
4. A single-source component is marked as a disagreement.
5. For the requested review assumption, a disagreement is retained when it overlaps
   any non-ignored human core rally time and disabled when it has zero core overlap.
   Both-model components are always retained. Retained ranges are not boundary-trimmed.

This is an optimistic oracle for whether the user makes the correct keep/disable
decision. It is intentionally not a perfect-edit oracle: false boundary overhang, false
positive both-model components, padding, and retained short gaps remain in the export.
Removing invalid disagreements before padding matches the editor's included/excluded
range behavior.

For each recording, retained model and human intervals receive identical padding, are
clipped to video bounds, merged when overlapping or touching, and joined across positive
gaps strictly below 3 seconds. Label `ignoredIntervals` are subtracted after those
operations and fragments are not rejoined across ignored time. Dataset scores pool
intersection numerators and duration denominators across recordings:

```text
P_pad  = duration(padded_model intersection padded_human) / duration(padded_model)
R_core = duration(padded_model intersection core_human)   / duration(core_human)
F1_padP_coreR = 2 * P_pad * R_core / (P_pad + R_core)
```

## Evaluation-only results

Durations are seconds after ignored-time subtraction. `Difference` is model export
minus padded human export. The reviewed ensemble is the product behavior under the
requested user-review assumption; the raw ensemble is included as a no-review
guardrail.

| Pad | Strategy | `P_pad` | `R_core` | `F1_padP_coreR` | Model export | Human export | Difference |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0s | Previous production | 57.96% | 88.03% | 69.90% | 1,237.05 | 814.47 | +422.58 |
| 0s | All-labels v2 | 65.41% | 89.11% | **75.44%** | 1,109.55 | 814.47 | +295.08 |
| 0s | Raw ensemble | 53.38% | **94.28%** | 68.16% | 1,438.54 | 814.47 | +624.07 |
| 0s | Reviewed ensemble | 59.40% | 94.09% | 72.82% | 1,290.17 | 814.47 | +475.70 |
| 1s | Previous production | 61.25% | 95.37% | 74.59% | 1,612.64 | 1,072.57 | +540.07 |
| 1s | All-labels v2 | **68.88%** | 96.25% | **80.30%** | 1,444.22 | 1,072.57 | +371.66 |
| 1s | Raw ensemble | 57.15% | **98.94%** | 72.45% | 1,822.17 | 1,072.57 | +749.60 |
| 1s | Reviewed ensemble | 65.06% | 98.69% | 78.42% | 1,592.10 | 1,072.57 | +519.53 |
| **2s** | **Previous production** | **65.28%** | **97.83%** | **78.30%** | **1,927.79** | **1,332.57** | **+595.23** |
| **2s** | **All-labels v2** | **72.05%** | **98.71%** | **83.30%** | **1,752.48** | **1,332.57** | **+419.91** |
| **2s** | **Raw ensemble** | **61.16%** | **99.87%** | **75.86%** | **2,136.78** | **1,332.57** | **+804.21** |
| **2s** | **Reviewed ensemble** | **69.65%** | **99.87%** | **82.07%** | **1,867.54** | **1,332.57** | **+534.97** |
| 3s | Previous production | 67.98% | 98.89% | 80.57% | 2,245.78 | 1,591.35 | +654.43 |
| 3s | All-labels v2 | **75.41%** | 99.59% | **85.83%** | 2,027.43 | 1,591.35 | +436.08 |
| 3s | Raw ensemble | 64.78% | **100.00%** | 78.63% | 2,417.73 | 1,591.35 | +826.38 |
| 3s | Reviewed ensemble | 73.33% | **100.00%** | 84.61% | 2,127.17 | 1,591.35 | +535.82 |

At the two-second target, the reviewed ensemble changes `P_pad`, `R_core`, and
`F1_padP_coreR` by **+4.38, +2.05, and +3.77 percentage points** versus previous
production. Versus all-labels v2, it changes them by **-2.40, +1.16, and -1.23
points**. At three seconds, reviewed-ensemble F1 is 84.61%, 4.04 points above previous
production and 1.22 points below all-labels v2.

## Protected-test sensitivity

This table is descriptive only and was not used to select the system. It uses only
`indoor-source-05`.

| Pad | Strategy | `P_pad` | `R_core` | `F1_padP_coreR` | Model export | Human export | Difference |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0s | Previous production | 73.15% | 93.41% | **82.05%** | 411.72 | 322.42 | +89.31 |
| 0s | All-labels v2 | 64.58% | 88.00% | 74.49% | 439.31 | 322.42 | +116.89 |
| 0s | Raw ensemble | 61.01% | **94.39%** | 74.12% | 498.79 | 322.42 | +176.38 |
| 0s | Reviewed ensemble | 64.00% | **94.39%** | 76.28% | 475.56 | 322.42 | +153.14 |
| 1s | Previous production | 72.39% | 98.42% | **83.42%** | 523.79 | 400.42 | +123.37 |
| 1s | All-labels v2 | 68.19% | 94.81% | 79.33% | 536.37 | 400.42 | +135.96 |
| 1s | Raw ensemble | 63.54% | **98.50%** | 77.25% | 601.73 | 400.42 | +201.31 |
| 1s | Reviewed ensemble | 67.48% | **98.50%** | 80.09% | 566.59 | 400.42 | +166.17 |
| **2s** | **Previous production** | **72.17%** | **99.71%** | **83.73%** | **636.97** | **478.42** | **+158.55** |
| **2s** | **All-labels v2** | **70.44%** | **97.35%** | **81.74%** | **630.01** | **478.42** | **+151.59** |
| **2s** | **Raw ensemble** | **64.70%** | **99.71%** | **78.47%** | **713.26** | **478.42** | **+234.84** |
| **2s** | **Reviewed ensemble** | **69.39%** | **99.71%** | **81.83%** | **664.99** | **478.42** | **+186.57** |
| 3s | Previous production | 73.09% | **100.00%** | **84.46%** | 736.14 | 556.42 | +179.72 |
| 3s | All-labels v2 | **71.78%** | 98.70% | 83.12% | 732.84 | 556.42 | +176.42 |
| 3s | Raw ensemble | 66.63% | **100.00%** | 79.97% | 812.39 | 556.42 | +255.97 |
| 3s | Reviewed ensemble | 71.33% | **100.00%** | 83.26% | 758.86 | 556.42 | +202.44 |

At two seconds on the protected recording, the reviewed ensemble matches previous
production's 99.71% recall but trails its F1 by 1.90 points. Against all-labels v2 it
gains 2.35 recall points and 0.10 F1 points while losing 1.04 precision points.

## Per-recording target-padding diagnostics

| Recording | Reviewed `P_pad` | Reviewed `R_core` | Reviewed `F1_padP_coreR` | F1 vs previous | F1 vs v2 | Invalid disagreements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `grass-source-06` | 60.75% | 99.58% | 75.47% | +0.85 | -0.18 | 10 / 11 |
| `grass-source-08` | 68.47% | 100.00% | 81.29% | +6.51 | -3.73 | 15 / 15 |
| `indoor-source-03` | 81.24% | 100.00% | 89.65% | +4.68 | -0.10 | 12 / 13 |
| `indoor-source-05` (protected) | 69.39% | 99.71% | 81.83% | -1.90 | +0.10 | 6 / 6 |

Across the three evaluation recordings, 37 of 39 disagreement components are invalid;
including the protected recording, 43 of 45 are invalid. The resulting 95.6% invalid
rate confirms that the disagreement queue is highly concentrated review work. The two
valid disagreements are one v2-only grass component and one previous-production-only
indoor component.

The whole-component agreement rule leaves an important blind spot. There are 19
evaluation components, or 27 including protected test, that are emitted by both models
but have zero overlap with core human rally time. They are not surfaced as low-confidence
disagreements. In addition, when the two models overlap only partially, the production
merge expands the output to the earliest start and latest end and marks that full range
as agreement. Model-only boundary wings can therefore lower precision without entering
the disagreement review queue.

## Product follow-up

Keep mandatory disagreement review if the two-stack recall safeguard remains enabled.
Do not treat the raw union as a safe automatic export. To preserve the recall lift while
closing the precision gap to all-labels v2, split overlapping components at every model
boundary and retain provenance for each subrange. The editor can then highlight
old-only and v2-only boundary wings instead of converting the entire min/max union into
agreement. Low-confidence or unusually long both-model components should remain a
secondary review class because agreement alone did not eliminate false positives in
this evaluation.
