# Serving-side temporal embeddings with specialist features

## Registered follow-up

This development experiment follows the embedding-only temporal study. Keep its
968 reviewed non-beach serves, 26 recordings, eight conservative source groups,
labels, exclusions, 13-frame windows, frozen encoder, three seeds, and nested
source-held partitions. The previous experiment is immutable. No protected test
labels, new annotations, or production changes are part of this study.

Add the incumbent's exact ordered SERVSIDE237-FLIGHT signature: 82 court-flow
recording-rank features and 155 recording-rank flight features from the
192x108-r4c6 configuration. Reconstruct tied flight ranks within each recording;
reuse the frozen court-flow ranks. These are label-independent, recording-wide
features over the historical reviewed-candidate population, not online features.
Verify row identities and feature signatures against the frozen specialist.

Refit the incumbent class-balanced logistic algorithm with its fixed L2=0.1
inside every fitting partition. Its scaler, class weights, and coefficients see
only fitting rows. This supplies both a newly nested logistic control and a
frozen linear skip path in the neural heads. Do not use the incumbent's final
all-development weights to predict held sources. Its historical saved results
also included beach training rows and development-wide threshold selection;
the newly nested non-beach control removes those differences.

The hybrid retains the original four-token LayerNorm/projection and width-64
temporal network over embeddings from -3 to +3 seconds at 2 Hz. Concatenate its
64-value center representation with a learned 237-to-32 ReLU projection of the
fold-standardized specialist features. A 96-to-64 ReLU/dropout-to-one correction
head adds a bounded residual (3*tanh) to the frozen logistic logit. Initialize
the last layer to zero. This starts each fit at its freshly fitted specialist.

A matched specialist-only neural control replaces the temporal representation
with zeros; the feature branch and correction head are identical. The controls
measure whether embeddings add information beyond the same specialist inputs.
The original embedding-only head remains a historical neural comparison; the
hybrid also changes initialization and regularization, so its entire gain must
not be attributed solely to feature concatenation.

Use seeds 3407, 1729, 20260918, epochs 5/15/30/60, deterministic FP32 AdamW
(learning rate 0.001, weight decay 0.0001), batch 32, gradient clipping 1.0,
dropout 0.1, and the previous source-balanced/square-root-class-weighted BCE.
Add 0.01 times the weighted squared residual to discourage overwriting the
specialist without supporting evidence. Record fitting loss, training and held
predictions at each checkpoint for diagnosis. No adaptive hyperparameter search.

Report full-correction heads (residual multiplier 1) and inner-selected mixtures
with multipliers 0, 0.25, 0.5, 1. Multiplier zero is the logistic control. Select
epoch, multiplier, and threshold only on inner source-held predictions; threshold
grid 0.05 to 0.95 by 0.025. Rank by group-macro BA, pooled BA, pooled macro F1,
worst-group BA; ties prefer less correction, earlier epoch, then threshold
closest to 0.5 and the smaller threshold. Multiplier zero has epoch zero.
Report three-seed mean metrics, per-source results, paired errors, and secondary
probability ensembles using the mean inner-selected thresholds. Do not select a
favorable seed or tune on outer labels. Preserve final full-correction hybrid
heads for all seeds as well as final inner-selected mixtures, using full-
development source-held selection before all-development refitting.

Primary metric is source-group macro balanced accuracy. Compare the hybrid with
the newly nested logistic and specialist-only neural controls. A production
follow-up additionally requires improvement over the historical specialist's
macro BA without regressing its worst-group BA. This is serving-side prediction,
not rally interval selection; export-padding F1 is inapplicable.

## Diagnostic plan

Before interpreting results, independently verify feature construction, label
alignment, excluded groups, scalers, checkpoint reloads, selections, and metrics.
For the original head, measure training/held gaps, per-source ROC AUC, class-wise
probability shifts, fixed-0.5 sensitivity, and hypothetical per-source best
thresholds. The last are explicitly label-exposed diagnostic upper bounds, never
deployable estimates or model selectors. Measure inference sensitivity to
removing/reversing context as a perturbation diagnostic, not a trained ablation.
Distinguish observed failures from hypotheses about spatial pooling, the 16-value
projection bottleneck, and the frozen encoder's task mismatch.

The encoder's prior exposure to held sources' unlabeled imagery and rally-based
model selection remains a limitation. Anchors are reviewed serves; serve gating,
inferred-anchor robustness, phone parity, and phone latency remain unmeasured.
Only stable ledger indexes and aggregate results belong in Git.

## Work tracking

The Tasks connector is unavailable; this checklist tracks the work.

- [x] Inspect the incumbent feature signature and earlier temporal errors.
- [x] Freeze the follow-up registration and reconstruct the 237 features.
- [x] Qualify fold-local fitting, fusion, and checkpoint behavior.
- [x] Train nested hybrid and matched specialist-only heads.
- [x] Diagnose the earlier head and compare all controls.
- [x] Refit final heads and independently audit results and privacy.
- [x] Record findings and artifact indexes.

## Earlier-head diagnosis

The embedding-only head's mean fitting-source macro BA is 94.29%, versus 77.37%
on held sources. Mean held-source ROC AUC is 93.71%; mean fitting AUC is 98.31%.
Thus the scores often order the two sides better than their selected decision
threshold suggests, while ranking also degrades on unseen sources.

Choosing a separate threshold using each held source's own labels would produce
90.36% mean macro BA. This is a label-exposed diagnostic upper bound for these
fixed predictions, not an estimate of a usable calibrated model. It supports
threshold transfer as a substantial problem, but it still trails the historical
specialist. Simply using 0.5 must not be confused with source-specific calibration.

The ensemble's largest source has 373 serves: near serves average 0.315 near
probability, versus 0.060 for far serves, and its inner-selected ensemble cutoff
is 0.567. It misses 154 of 195 near serves. Another 145-serve source averages
0.878 for near and 0.408 for far, but uses a cutoff of 0.283, incorrectly calling
50 of 76 far serves near. These opposite shifts explain why a common threshold
does not transfer reliably. Stable source indexes and full distributions are in
the [diagnostic report](serving-side-temporal-diagnostics-2026-10-13.json).

Removing context from the trained temporal head reduces mean macro BA from
77.37% to 65.78%; reversing visual time reduces it to 66.64%. Removing past frames
reduces it to 61.45%, and removing future frames to 72.83%. These are inference
perturbations, not newly trained ablations, but they confirm that temporal order
and both directions affect the model. The earlier trained anchor-only control
was 70.54%, a separate and fairer estimate of the benefit from learning context.

The specialist has evidence the first head lacked: camera-motion-compensated
near/far flow, concentrated flight motion, phase differences, and recording-
relative ranks. In contrast, the encoder supplies four image-relative pooled
appearance tokens at 2 Hz and projects each 960-value token to 16 values. The
loss of fine spatial/motion information and the projection bottleneck are
plausible contributors, not isolated causal findings. The hybrid controls below
test the value of adding the specialist evidence without claiming to identify
which encoder limitation is responsible.

The follow-up's immutable registration and source archive are under ledger
artifact `private-reference-0231`; registration SHA-256 is
`dc2fdf8b932724a1b0958163c4335b01a35351ce0658f118ad615d66cb7f7176`.

## Completed model comparison

All 216 nested fits completed. Adding the specialist features and its fold-fitted
linear skip recovers most of the original deficit. The hybrid's remaining gain
over the newly nested specialist is small, and it fails the historical
specialist's primary-metric promotion guardrail. Production remains unchanged.

Neural values below are three-seed means; worst-group BA is the minimum across
seeds. All newly fitted controls use the same non-beach population and nested
threshold selection. The two historical comparisons retain their previously
documented selection and training-scope differences.

| Model | Source-group macro BA | Pooled BA | Worst-group BA |
|---|---:|---:|---:|
| Hybrid, inner-selected correction strength | 93.51% | 93.89% | 85.83% |
| Hybrid, full neural correction | 93.28% | 93.72% | 87.30% |
| Same neural correction, specialist features only, inner-selected strength | 93.33% | 93.42% | 82.89% |
| Same neural correction, specialist features only, full correction | 92.87% | 92.86% | 84.36% |
| Freshly refitted logistic specialist | 93.41% | 93.48% | 85.83% |
| Historical existing specialist | 93.97% | 94.00% | 85.83% |
| Earlier embedding-only temporal head | 77.37% | 70.72% | 52.52% |

The selected hybrid is 16.15 percentage points above the earlier embedding-only
head, 0.10 points above the new logistic control, and 0.19 points above the matched
specialist-only neural control. Full-strength embeddings add 0.40 points to the
full-strength neural control, but that hybrid is 0.14 points below the logistic
control. These comparisons do not establish a practical replacement advantage.

| Hybrid seed | Full-correction macro BA | Inner-selected-strength macro BA |
|---|---:|---:|
| 3407 | 93.35% | 93.18% |
| 1729 | 93.39% | 93.80% |
| 20260918 | 93.09% | 93.56% |

The selected-strength ensemble makes 61 errors, versus 58 for the historical
specialist and 63 for the newly nested logistic control. Against the historical
specialist it fixes five errors and introduces eight. Ensemble results are
secondary and do not replace the seed-mean primary metric.

## Final research heads

Final checkpoint, correction strength, and threshold were selected using
full-development source-held predictions, then refitted on all 968 rows. These
are deliverable research heads, not a new held-out evaluation.

| Seed | Configuration | Epoch | Correction multiplier | Near threshold |
|---|---|---:|---:|---:|
| 3407 | Full correction | 15 | 1.00 | 0.525 |
| 3407 | Selected strength | 30 | 0.25 | 0.525 |
| 1729 | Full correction and selected strength | 5 | 1.00 | 0.325 |
| 20260918 | Full correction | 5 | 1.00 | 0.325 |
| 20260918 | Selected strength | 15 | 0.25 | 0.525 |

Each hybrid contains 48,785 trainable parameters and 712 frozen scaler/linear-skip
values. Five distinct checkpoints supply the six configurations, with compressed
checkpoint sizes of approximately 188 KB. Weights, fitting receipts, prepared
features, logs, and the immutable source snapshot reside in ledger artifact
`private-reference-0231`. Full checkpoint hashes and evaluation breakdowns are in
the [aggregate result](serving-side-hybrid-head-2026-10-13.json).

## What explains the result

The missing specialist inputs are the largest practical difference. With trained
weights and thresholds fixed, label-blind permutations within each held recording
produce the following mean macro BA for the full-correction hybrid:

| Inference perturbation | Macro BA |
|---|---:|
| Original inputs | 93.28% |
| Shuffle court-flow features | 74.78% |
| Shuffle flight-motion features | 70.19% |
| Shuffle all 237 specialist features together | 50.47% |
| Shuffle complete embedding windows | 92.93% |
| Keep only the anchor embedding frame | 93.11% |

The strong dependence on both specialist feature families and much smaller
dependence on embeddings support keeping the specialist inputs. These are
post-hoc perturbations with fixed operating points, not retrained ablations;
they do not measure the accuracy of a model trained with a feature family removed.

The frozen embeddings also carry substantial source-specific appearance.
Source-group means explain 69.90% of feature variance on average across raw
anchor embedding channels, and 71.68% after per-token LayerNorm without learned
affine parameters. The same descriptive quantity is 5.03% for after-minus-before
embedding differences. Recording-rank specialist features have equal recording
means by construction. This observation is consistent with source-dependent
score shifts and suggests explicit temporal differences or recording-relative
embedding normalization as a focused future experiment. It does not prove that
removing all source information would improve classification.

Additional epochs mainly improve fitting accuracy. At the fixed diagnostic 0.5
threshold, full-hybrid training macro BA rises from 99.41% at epoch five to
99.92% at epoch 60, while source-held macro BA declines from 93.24% to 92.76%.
The specialist-only neural control degrades more, from 93.05% to 91.44% held BA.
This is evidence of overfitting on the current eight-source development set;
training longer alone does not resolve the transfer problem. These fixed-0.5
curves were not used to alter the registered inner selection.

The selected hybrid's gain over the logistic control is only 0.10 percentage
points. A paired 20,000-resample bootstrap over the eight source groups gives a
descriptive 95% interval of -0.47 to +0.54 points. Its gain over the matched
selected-strength neural control is 0.19 points, with an interval of -0.50 to
+0.81 points. These intervals condition on the fitted development predictions
and are not protected-test claims. They include both improvement and regression.

The evidence supports three findings: explicit motion/rank features recover the
lost signal; source-dependent score shifts explain much of the original neural
head's deficit; and the current embedding branch provides little demonstrated
incremental benefit. Fine spatial pooling, the projection bottleneck, and encoder
task mismatch remain hypotheses rather than individually isolated causes. The
[hybrid diagnostics](serving-side-hybrid-diagnostics-2026-10-13.json) contain the
perturbations, source differences, bootstrap intervals, and epoch curves.

## Verification and reproduction

- Twenty-four focused unit tests passed, covering the new hybrid, earlier temporal
  head, specialist fitting, and recording-rank helpers. Thirty additional synthetic
  operating-point cases agreed with an independent brute-force implementation.
- The independent audit reconstructed all 968 temporal windows and all 968 by 237
  specialist features. The ordered signature matches the packaged production
  specialist. Only the -3.0 and -2.5-second embedding context slots are masked;
  the specialist's shorter motion windows remain inside valid context.
- All 216 neural fitting partitions and 37 logistic fits passed verification.
  Fold scalers match training-only statistics and the maximum logistic objective
  gradient is below 0.00000000000000004.
- All 120 outer operating-point selections and reported confusion matrices were
  independently replayed. Seventy distinct selected held checkpoints reproduced
  saved base logits and neural residuals exactly, maximum difference 0.0.
- The largest final-head CPU/GPU probability difference is 0.000000124 on the
  fixed 16-row probe. This verifies head inference, not phone embedding extraction
  or end-to-end native-device accuracy.
- The [audit receipt](serving-side-hybrid-audit-2026-10-13.json) records the hashes
  and counts. Targeted privacy checks cover the new source, tests, protocol, and
  aggregate reports. No protected labels were opened; no production model was
  replaced. No commit, pull request, or publication was performed.

Use `scripts/train-serving-side-hybrid.py` with `prepare`, `fit`, and `summarize`.
Run in the qualified WSL PyTorch environment and resolve the study through
`VOLLEYCUT_PRIVATE_LEDGER` or its `_POSIX` override; missing mappings fail closed.
The frozen source snapshot under the study's
ledger artifact is the exact reproduction source. Preserve the original study
and this registration when running follow-ups.

Independent audit: `scripts/audit-serving-side-hybrid.py`. Post-hoc diagnostics:
`scripts/diagnose-serving-side-temporal.py` and
`scripts/diagnose-serving-side-hybrid.py`. The archived preregistration predates
training; the completed findings and additional post-hoc diagnostics above do
not change its candidate search or selectors.
