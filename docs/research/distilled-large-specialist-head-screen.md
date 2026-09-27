# Frozen Distilled Large recall embeddings: lightweight specialist-head screen

## Scope and method

The selected `neural-distilled-mobile-large-tcn-fp32-high-recall` encoder and rally TCN were frozen. Only two separate class-balanced logistic heads were fitted to saved regional image embeddings. The direct heads use 3,840 serving-side inputs and 3,844 side-switch inputs; each final linear head is about 15.4 KB in float32 parameters. A fixed, label-independent 128-column sign projection was also screened. No AV, quality scalar, existing specialist feature, or human label enters the frozen encoder at inference.

The serving-side population has 968 non-beach, human-reviewed near/far candidates from 26 recordings in eight source groups. Evaluation holds out each entire source group when fitting the new head. Anchors are the reviewed candidates, not neural rally starts. The comparison is the existing serving-side specialist's saved out-of-group decisions on precisely those rows.

The switch population is the existing opened-development union of 704 candidates and 50 exhaustive human markers across 11 recordings. All 11 recordings belong to **one source group**. Evaluation holds out one recording at a time; it does not measure source-group transfer. Candidate matching uses the existing four-second one-to-one marker contract. The new scores use the frozen union decoder with a fixed 0.5 threshold. The comparison is the existing specialist's saved recording-held-out result on the same candidate universe.

## Results

| Head input | L2 | Serving pooled balanced accuracy | Serving group-macro balanced accuracy | Switch candidate AP | Switch event F1, fixed threshold | Switch event F1, post-hoc best threshold |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 128-column projection | 0.1 | 70.88% | 68.47% | 10.23% | 6.35% | 18.05% |
| Direct frozen embeddings | 0.01 | 75.69% | 78.81% | 13.34% | 14.46% | 18.03% |
| Direct frozen embeddings | 0.1 | 79.63% | 82.68% | 12.20% | 17.28% | 18.42% |
| Direct frozen embeddings | 1.0 | **82.83%** | **84.36%** | 11.88% | 13.51% | 22.22% |
| Existing task-specific specialists | — | **94.00%** | **93.97%** | **44.50%** | **56.86%** | — |

The direct serving head at L2 1.0 is 11.17 percentage points below the current specialist in pooled balanced accuracy. The best switch candidate AP in this small grid is 13.34%, versus 44.50% for the current specialist. At L2 0.1 the direct head emits 31 switch proposals with 7 TP, 24 FP, and 43 FN; the incumbent emits 52 with 29 TP, 23 FP, and 21 FN. Post-hoc threshold sweeps use held-out outcomes to select the threshold and are optimistic diagnostics, not deployable estimates.

The switch study's positive candidate prevalence is 46/704 (6.53%). The new head ranks above that base rate but does not approach the task-specific model. This screen does not support replacing the existing serving-side or switch specialists, nor removing their video decode pass.

## Limits and reproducibility

- The projection, feature windows, L2 grid, and 0.5 switch threshold were development screens. Choosing the best row after viewing results is development selection, not untouched-test performance.
- The selected encoder learned label-blind DINO targets and may have seen held-out videos' pixels; only the new task-head labels are held out by group or recording.
- The saved inputs are desktop embeddings. Native feature parity and device runtime for these heads were not measured.
- A nonlinear head or new task-specific image sampling could behave differently. These results isolate a compact linear head on the already available embeddings.
- The output JSON files contain aggregate metrics and content hashes only. The runner accepts all private inputs as explicit arguments and records no original source names or paths in its outputs.

Runner: `scripts/experiment-distilled-large-specialist-heads.py`. The four adjacent JSON files preserve the aggregate observations and selected frozen encoder hash. The union decoder implementation matched the reference decoder in 40 deterministic cases on the real candidate population.
