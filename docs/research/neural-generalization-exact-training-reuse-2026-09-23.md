# Exact training and raw-inference reuse amendment

This additive amendment preserves the registered 162 logical training tasks,
270 task/precision evaluations, four checkpoint epochs, 192 epoch/decoder candidates,
11 recall floors and all final evaluation populations. It changes execution of
identical training recipes, without consulting accuracy outcomes. The original
fitter, data loader, student recipe and numerical auditors remain unchanged.

The mapping is chosen from registered task order, before running the affected
tasks. Within each registration, the first task with an identical ordered
training recipe owns its physical optimizer execution. Every later task in that
class receives that complete checkpoint bank. No cross-registration reuse is
allowed, including between coverage and approximate four-head proxy labels.

A recipe binds the registration and its complete frozen source closure,
manifest, feature index, model/configuration, training seed, epochs, loss arm,
raw ordered training IDs, tier-ordered IDs, source groups and excluded student
groups. Calibration IDs, display variant and split-draw identifier are excluded
because they do not enter optimization. Differences in any bound input, training
order or supervision tier prevent reuse.

The frozen temporal fitter evaluates calibration only with inference mode at
fixed epochs. It has no validation-dependent optimizer, scheduler, stopping
rule, scaler or sampling. Training has explicit independent NumPy streams and
step-specific dropout seeds. The student fitter has no validation argument;
its fixed eight epochs use only the ordered training images and teacher targets.
A source-bound synthetic CPU qualification changes calibration population and
order for all five distinct temporal architectures, retaining exact/draft/export
streams. It requires bit-identical training history, exposures, scaler and
checkpoint arrays at epochs 5,15,30,60, and shared calibration probabilities.
This qualification concerns training independence; it is not accuracy evidence.

The expected execution accounting is:

| Registration | Logical head fits | Physical head fits | Logical students | Physical students |
|---|---:|---:|---:|---:|
| Original corpus | 18 | 18 | 3 | 3 |
| Randomized composition | 96 | 60 | 16 | 10 |
| Export-proxy experiments | 48 | 42 | 8 | 7 |
| Total | 162 | 120 | 27 | 20 |

Historical nested fits are separate. These counts are checked against each
final registered mapping, rather than inferred from successful results. Draws
sharing an owner remain correlated experimental views, not additional independent
training replications. In particular, identical weights at each epoch make the
medium/wider-calibration contrast isolate calibration composition. Each task's
selected epoch can still differ after its independent calibration.

`register-neural-generalization-reuse.py` writes the immutable companion
`reuse-plan.json`. It binds this amendment, the CPU qualification, adapter,
independent provenance auditor and execution queue. The ordinary registrations
and tasks are not rewritten.

`reuse-neural-generalization-fit.py` requires the canonical owner's passing
numerical fit gate. It copies all four weight/scaler archives byte-for-byte and
explicitly rebinds inherited temporal/student metadata. The student encoder,
eight-epoch history, BatchNorm buffers, teaching membership and training-feature
lineage remain visible. Immutable student feature arrays may be shared through
receipts bound to the target encoder. New calibration imagery is extracted
without labels if needed.

Every target receives fresh label-blind CUDA predictions for every calibration
record at all four epochs. Predictions for overlapping calibration records must
match the owner's saved probabilities bit-for-bit. When overlap is empty, the
receipt explicitly reports no shared control; the target's separate numerical
audit still replays all calibration records at the declared sampled CPU scope.
No selected epoch, decoder, score threshold, candidate ranking or performance
result is copied. Every task independently evaluates its own 192 candidates and
all 11 strict recall floors, including infeasible floors.

The target must pass the unchanged numerical fit audit and the independent
reuse provenance audit, written as `reuse-audit.json`. Existing or incomplete
target fit directories cause a closed failure requiring investigation. Inherited
physical training time is explicitly labelled; adapter/materialization time is
reported separately.

For full-video inference, `reuse-neural-generalization-inference.py` may copy the
owner's already audited raw probability archives for the exact same frozen
42-record panel and DINO precision. Both complete checkpoint banks/scalers and
the distilled encoder must have identical bytes. All 42 raw archives are copied
byte-for-byte, with target-specific task and checkpoint references. Student
feature receipts are rebound; any existing target arrays must exactly equal the
owner's arrays. Decoded intervals, selections and metrics are never reusable.

Each target still passes the unchanged all-record numerical inference audit
and a separate `reuse-inference-audit-{precision}.json` provenance gate. Its
task-specific floor selections are then decoded and evaluated normally. The
final result auditor requires both companion mappings and all applicable reuse
gates, verifies logical/physical counts and binds the displayed execution lineage.
Every registered evaluation remains present. Reuse changes neither gold labels,
protected/common-panel exclusion, precision-transfer policy nor any ranking rule.

All new plans, qualification outputs, copies, receipts and logs reside on NAS.
No real reuse is executed until the final feature-bound registration and passing
qualification are available. There is no additional exploratory training or
outcome-dependent choice of owners.

The queue preserves registration order within each requested model. A variant
filter whose earlier physical owner is unavailable stops; run that owner first.
Existing CPU, RAM, GPU and encoder scheduling limits still apply. This amendment
grants no additional concurrency.

The CPU qualification passed on all five distinct architectures in 82.37 seconds,
using two threads on CPUs 10–11. Its immutable NAS receipt is
`reuse-qualification-v1/qualification.json`, SHA-256
`97040912ea48a415b2beac8bd68c53a8483aabd224cc0169e9ce918531788540`.
This result authorizes the execution adapter only; it makes no claim about the
recall or accuracy of any trained model.
