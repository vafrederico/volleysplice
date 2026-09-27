# DINO-distilled MobileNetV3-Large + TCN

## Completed work

- [x] Match the frozen Large study's 24 split configurations, including export-label variants.
- [x] Fix calibration to the 99% retained-core recall target; no lower-target fallback.
- [x] Qualify the Large student implementation and source isolation.
- [x] Train each fold-local student, then its TCN; calibrate on that split's held sources.
- [x] Select highest common-unseen exact-label F1, then highest recall within the winning variant.
- [x] Run both frozen selections on all 44 catalog videos, including beach inference only.
- [x] Publish both options and signals to the comparison index used by labels and editor lab.
- [x] Validate the publication and report all required padding cases and rally-loss guardrails.

All 24 registered fits are complete; 15 achieved the strict 99% calibration bar.
The two published options use distinct checkpoints and decoders within
`expanded-large`, both with training seed 3407. Results on the one-recording
common exact-label selection panel are:

| UI selection | Split draw | TCN epoch | P_pad | R_core | F1_padP_coreR | Wholly missed human rallies |
|---|---:|---:|---:|---:|---:|---:|
| Highest F1 | 20260918 | 60 | 97.76% | 97.68% | 97.72% | 1 |
| Highest recall | 3407 | 15 | 70.88% | 100.00% | 82.96% | 0 |

The highest-F1 option improves common-panel F1 from frozen Large's 96.32% to
97.72%. The highest-recall option removes its predecessor's one complete rally
miss but lowers F1 from 89.61% to 82.96% while retaining more footage. These are
selected-panel comparisons, not independent generalization estimates.

Both choices and their four signals are available in labelv2 and editor lab for
all 44 recordings: 42 non-beach and two beach. Thirty-six recordings support
scoring: nine non-beach exact-label videos, six reviewed drafts, nineteen reviewed
exports and two exact-label beach videos. The other eight have predictions/counts
only. The [results report](distilled-mobile-large-results.md) and
[aggregate JSON](distilled-mobile-large-results.json) keep these label types
separate and include all padding cases and production-training exclusions.

Beach remains unqualified: the highest-F1 option retains only 7.63% of human core
time and wholly misses 62 rallies across the two beach videos. The recall option
retains 78.01% and wholly misses 16. Neither option is promoted to production,
and this experiment does not establish native-phone performance.

The student is MobileNetV3-Large initialized from ImageNet V1, with a 960-to-384
training-only projection to cached frozen DINO targets. The distillation recipe
matches Small: eight epochs, batch 16, AdamW at 0.0001 with weight decay 0.0001,
fixed BatchNorm statistics, source-balanced global/spatial cosine loss. The
student learns only from each registered training membership. Calibration,
common-selection and beach footage never train the student or its TCN.

Inference uses the trained Large encoder's four 960-value regional pools at
2 Hz, aligned to 4 Hz audiovisual features: AV104 plus 3,840 visual values and
eight quality/age/availability values. The FP32 TCN has the same short-boost loss
and checkpoints at epochs 5, 15, 30 and 60 as the prior study. Cached embeddings
use FP16 storage, as before; this is not FP16 model inference. DINO and the
distillation projector are unnecessary on the eventual device.

Each split selects a feasible epoch/decoder under the strict 99% calibration
constraint. Among feasible splits, select the highest common-unseen
`F1_padP_coreR`; select the highest-recall draw within that winning dataset variant,
breaking recall ties by F1. This follows the previous UI methodology. The common
exact-label panel contains one recording/source group and is already selection
data, not a protected test. A 99% calibration target is not a guarantee on other
videos. Infeasible configurations remain explicitly infeasible.

Target product padding is 2 seconds each side; positive gaps strictly below
3 seconds join. Report pooled precision, retained-core recall, primary F1 and
export durations for symmetric 0/1/2/3-second padding. Subtract ignored intervals
after padding/joining without rejoining across them. Exact rally-event and
whole-rally-loss metrics are separate from retained-time recall; export-only
labels require their own coverage evaluation.

All private inputs, checkpoints and output locations are resolved using the
external ledger and ignored environment configuration. No original media names
or private filesystem locations belong in this report. Publication preserved
existing artifacts and editor drafts while adding the two new model IDs.

## Execution and qualification

The external run is registered as ledger artifact `private-reference-0222`.
It contains separate fitting and completion progress records. The completion
worker waits for every registered fit, validates checkpoint ownership, performs
selection and all-video inference, evaluates saved predictions, produces an
identifier-free report, then validates and publishes the two options. A final
browser check covers both options, their signals on all 44 videos, suppression
combinations and reload behavior. A failure stops publication or reports the
failed stage; the 99% eligibility floor is never silently relaxed.

The final completion receipt and browser check passed for both options on all
44 recordings, including their signals, suppression combinations, human comparison
and reload behavior. Scoring replay was corrected to use the frozen catalog
duration after one final endpoint differed by 44 microseconds from an AV-cache
duration. Saved predictions, rally counts, checkpoints and selections were unchanged.

After five complete fits, execution moved to two persistent workers with disjoint
task assignments. The handoff left no partial fit directories. The original plan,
numerical sources, task contracts, seeds and training recipe remain unchanged;
an additional execution receipt binds the dispatcher and validates all 24 fits
before selection may start. Concurrent wall times are not speed benchmarks.

A later calibration check found ten candidates with retained-core recall
`1.0000000000000002`: summation order made a complete intersection exceed its
denominator by about `1.14e-13` seconds, while accurately summed missed core was
zero. An execution amendment bounds only out-of-domain metric values within
32 float64 epsilons of zero or one. Values already inside the domain, including
those immediately below the 99% bar, are unchanged. Raw and bounded candidates,
corrections and source/checkpoint hashes are retained in the indexed artifact.
Completed training is reused; this correction does not change model predictions,
boundaries, loss, thresholds or source membership.

The GPU qualification used deterministic FP32 with TF32 disabled. Batch 16
produced 960-channel spatial features and ten 384-value teaching tokens, with
finite gradients and fixed BatchNorm buffers. The first real student audit
verified 28 training recordings, seven training source groups and 3,584 teacher
frames: all 138 BatchNorm buffers were unchanged and all 170 encoder parameter
tensors changed. Distillation took 118.15 seconds for that student; this is not
the complete fit or video-processing time.

Focused implementation tests passed, covering student geometry and
training, inference ownership/resume, publication rollback/draft preservation,
report privacy, label-policy separation and disjoint worker ownership. The branch publication audit found
no current-file or historical private-reference findings. Independent report
checks verified the selections against saved scores, all four padding cases,
same-gold frozen-Large comparisons, separate label-policy panels and aggregate-only
public output. The completed results above remain research findings.
