# Temporal serving-side head on frozen Distilled Large embeddings

## Preregistered scope

Train a near/far serving-side head on the frozen highest-recall DINO-distilled
MobileNetV3-Large encoder. The encoder and rally model remain unchanged. This is
development research; production continues to use the existing specialist.

Use the correction-clean, human-reviewed serving-side development snapshot from
the previous embedding screen: 968 non-beach serves, 26 recordings, eight source
groups. Near is the positive class. Unclear and not-serve rows, beach, protected
sources, ignored footage, and source-quality exclusions cannot supply supervision.
Verify current effective review decisions against every frozen row. Bind the
dataset, corrections, current review, catalog ignored ranges, encoder, and embedding
receipts by content hash. Private inputs and outputs resolve through the external
ledger; publish only stable indexes and aggregate results.

## Models and training

- Ordered embeddings at 2 Hz from -3 to +3 seconds relative to the reviewed serve
  anchor: 13 frames, four 960-value image-relative regional pools per frame.
- Nearest-frame selection uses earlier ties and a maximum 0.25-second error.
  Context outside video bounds or the anchor's contiguous valid segment is masked;
  never sample across ignored or source-quality intervals. Include availability,
  relative time, and actual selected-frame offset as label-independent scalars.
- Follow the rally head's per-token LayerNorm, shared learned 960-to-16 projection,
  and depthwise residual temporal convolutions. Use width 64, kernel three,
  dilations 1/2/3, dropout 0.1, and a single near/far logit at the anchor. Its
  receptive field covers the full 13-frame window.
- Train two predeclared arms: all ordered context, and the identical network with
  only the anchor frame available. The second arm measures the benefit of temporal
  context rather than conflating it with a different nonlinear architecture.
- Seeds 3407, 1729, 20260918; checkpoints 5, 15, 30, 60. AdamW learning rate 0.001,
  weight decay 0.0001, batch size 32, gradient clipping at 1.0. Shuffle all training
  rows each epoch. Source-balanced loss and square-root class weights are fitted
  exclusively on that training partition. Use deterministic FP32 training.

## Testing and selection

Hold out an entire source group in each of eight outer folds. Within the other
seven groups, hold out each group in turn to choose checkpoint and threshold.
An excluded-pair fit may supply two inner evaluations, since neither excluded
group enters fitting. All normalization, weighting, checkpoint and threshold
choices exclude the outer group. Threshold candidates are fixed at 0.05 through
0.95 in steps of 0.025, including 0.5.

Rank inner candidates by source-group macro balanced accuracy, pooled balanced
accuracy, macro F1, worst-group balanced accuracy, then earlier checkpoint and
threshold closest to 0.5. Evaluate each arm and seed using its nested outer
decisions. Select the final arm by mean three-seed outer macro balanced accuracy,
then mean pooled balanced accuracy and worst-group accuracy, preferring the
single-frame arm on an exact tie. Select a final epoch/threshold for each seed
using full-development leave-one-group-out scores, then refit on all development
labels. Report three-seed performance and a probability ensemble using only
inner-selected per-seed thresholds; do not choose a seed using outer outcomes.
Preserve full-development temporal heads for all three seeds even if the control
arm wins. This delivery requirement does not alter candidate search or testing;
any additional temporal refits use the same frozen recipe and full-development
source-held checkpoint selection.

Compare the existing specialist's saved source-held decisions on the exact same
rows. Include the prior frozen linear-embedding screen as a historical control;
its grid selection was development-exposed and is not a new nested estimate.
The specialist comparator also retains its historical development feature and
threshold selection; its saved decisions are not a newly nested refit.
Report pooled and group-macro balanced accuracy, macro F1, both classes' precision
and recall, worst group, environments, per-group results, paired errors, and
model parameter/storage counts. The research winner must improve macro balanced
accuracy without regressing the existing specialist's worst group before a
production follow-up is justified. This study creates no rally intervals, so
export-padding F1 is not a serving-side selection metric.

## Interpretation limits

The selected frozen encoder may have seen evaluation groups' unlabeled images
during DINO distillation, and its historical selection used rally outcomes.
Nested testing isolates the new serving-side supervision; it does not establish
fully unseen end-to-end encoder generalization. Anchors are reviewed candidate
starts, not newly inferred rally starts. Device embedding parity, phone latency,
and end-to-end serve-gating accuracy require separate measurements.

## Work tracking

The Tasks connector is unavailable; this checklist is the durable task record.

- [x] Inspect the prior serving-side screen and rally fitting/testing contracts.
- [x] Locate the frozen encoder, embedding bank, reviewed labels and GPU runtime.
- [x] Register hash-bound inputs and qualify temporal masking and fold isolation.
- [x] Complete all nested development fits for both arms and three seeds.
- [x] Select and refit final heads, verify checkpoint inference parity.
- [x] Independently replay metrics, run focused tests and privacy checks.
- [x] Record the results and artifact indexes here.

## Execution checkpoint

Ledger artifact `private-reference-0225` contains the immutable registration,
source snapshot, prepared windows, fitting receipts, logs, nested decisions, and
final heads. The prepared tensor is 968 by 13 by 3,843: 3,840 embedding values
plus three scalars per frame. All 968 anchors are valid; five context frames are
masked. Training covers 216 fits: 28 excluded-source pairs and eight excluded
single groups, for each of two arms and three seeds.

The current review document has additional content since the older feature
snapshot, but every effective reviewed label in the 968-row population agrees
after applying the unchanged correction overlay. The embedding catalog has finer
source subdivisions for one population; all those subdivisions remain together
in the original conservative serving-side source group. No catalog source group
crosses a serving-side fold.

Qualification passed 28 focused tests covering the new head, the reused compact
rally TCN, and regional embedding utilities. Tests verify ordered past/future
inputs, masking without crossing excluded spans, earlier nearest-frame ties,
both temporal directions' gradients, exclusion of held labels from weights,
checkpoint selection, exact checkpoint reloads, and corrupted-resume rejection.

## Completed results

The temporal arm wins the predeclared comparison against the identical
single-frame control, but fails the existing specialist's quality guardrails.
Keep the existing serving-side specialist operational.

The neural values below are means across the three nested source-held seed
evaluations; the worst-group column is the minimum across seeds. The incumbent
and historical linear head are their saved development comparisons, with the
selection-exposure limitations stated above.

| Head | Source-group macro BA | Pooled BA | Worst-group BA |
|---|---:|---:|---:|
| Ordered temporal embeddings | 77.37% | 70.72% | 52.52% |
| Identical head, anchor frame only | 70.54% | 68.84% | 50.00% |
| Existing serving-side specialist | 93.97% | 94.00% | 85.83% |
| Historical direct linear embedding head | 84.36% | 82.83% | Not recorded |

Temporal context adds 6.83 percentage points of mean group-macro balanced
accuracy over the matched control. It remains 16.60 points below the existing
specialist and 6.99 points below the historical linear embedding head. These
results reject this compact head/window as a specialist replacement; they do
not establish the performance of other temporal architectures or encoder
fine-tuning.

| Temporal seed | Nested group-macro BA | Nested pooled BA | Nested worst-group BA |
|---|---:|---:|---:|
| 3407 | 76.44% | 71.64% | 65.03% |
| 1729 | 79.66% | 73.14% | 62.10% |
| 20260918 | 76.01% | 67.37% | 52.52% |

The predeclared three-seed probability ensemble, using the mean of each outer
fold's independently inner-selected thresholds, reaches 79.31% group-macro BA,
69.79% pooled BA, and 59.95% worst-group BA. The anchor-only ensemble reaches
73.07%, 67.41%, and 53.33%, respectively. The temporal ensemble fixes 39 incumbent
errors but introduces 273 new errors: 292 errors versus the incumbent's 58 on
the same 968 rows. Ensemble results are secondary; they do not replace the
predeclared seed-mean selector.

Three final temporal heads were selected using all-development source-held
predictions and refitted on all 968 reviewed development serves:

| Seed | Selected epoch | Near-probability threshold |
|---|---:|---:|
| 3407 | 60 | 0.925 |
| 1729 | 30 | 0.875 |
| 20260918 | 15 | 0.200 |

Each head has 34,961 parameters and 139,844 bytes of FP32 parameter storage.
The wide threshold variation is recorded rather than hidden by selecting a
favorable seed. Model weights, complete fitting receipts and source snapshots
remain under ledger artifact `private-reference-0225`. The
[aggregate results](serving-side-temporal-head-2026-10-13.json) include both
classes' precision/recall, per-group and environment breakdowns, selected outer
checkpoints and thresholds, paired errors, and final checkpoint hashes. The
[delivery receipt](serving-side-temporal-head-delivery-2026-10-13.json) identifies
all three fully trained temporal heads without disclosing private paths.

## Verification and reproduction

- Independent replay checked all 216 fitting partitions, all 968 input windows,
  and all 48 outer checkpoint/threshold selections. It reproduced the reported
  confusion matrices and aggregate metrics without importing the study's metric
  or selection functions.
- Restored selected checkpoints reproduce their training-time held predictions
  exactly: maximum difference 0.0. Final-head CPU/GPU prediction differences are
  at most 0.000001133 on the fixed 16-row probe. This is head inference parity,
  not native-phone embedding or end-to-end qualification.
- Twenty-eight focused unit tests and 30 synthetic independent metric/selection
  replay cases passed. The study opened no protected-label file.
- Privacy checks passed for the new research sources, protocol and aggregate
  reports. Private input mappings, training tensors and checkpoints remain
  external to Git. No production serving-side, Android or website behavior was
  changed.
- Runtime: Python 3.12.3, NumPy 2.2.4, PyTorch 2.11.0+cu128; deterministic FP32
  training with TF32 disabled. Registration SHA-256:
  `e724a2f152dca33ac28a1137f21785bcf5afa165426fff19b980dffc6869abb3`.

Runner: `scripts/train-serving-side-temporal.py` (`prepare`, `fit`, `summarize`).
Independent verification: `scripts/audit-serving-side-temporal.py`.
Temporal-head delivery: `scripts/refit-serving-side-temporal-delivery.py`.
Pass private inputs/output locations explicitly after resolving the external
ledger; preserve the immutable registration and source snapshot when resuming.
