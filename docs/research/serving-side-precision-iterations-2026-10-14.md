# Ten serving-side precision experiments

## Frozen protocol

The user authorized up to ten further experiments, prioritizing 98% precision
among automatic serving-side answers and comparing 95%/99% targets. Review
workload uses all 968 reviewed true serves from 26 recordings and eight source
groups as its denominator. This is development research, not a protected-test
or end-to-end serve-detection evaluation. Prior encoder exposure and
reviewed-candidate recording ranks retain the previously documented limitations.

Freeze all ten arms before their new results are inspected. Use the same nested
source exclusions, labels, anchors, encoder, three neural seeds, epochs
5/15/30/60, and precision-policy selector as the previous precision study.
Every outer group is excluded from training and all inner selection models.
The selector requires both sides and source-macro precision to meet the target,
20 accepted examples per side, and four supporting inner groups per side.
Maximize source-macro automatic coverage among feasible policies; otherwise
review all. No outer-label threshold tuning or best-seed selection is allowed.
Descriptive precision is not a transfer guarantee. All ten development outcomes
will be disclosed; selecting a promising arm requires later independent validation.

| Iteration | Hypothesis / change |
| --- | --- |
| 01 | Add four absolute motion-quality averages to the 237 specialist ranks. |
| 02 | Preserve those four measurements separately in launch, early, and late phases (12 values). |
| 03 | Add the 82 cached raw court-flow values alongside ranks and the 12 quality values. |
| 04 | Add 64 fixed random projections of temporal embedding changes alongside ranks and 12 quality values. |
| 05 | Add 24 interactions between four quality averages and six near-minus-far flow/activity ranks. |
| 06 | Learn correctness/review from honestly cross-fitted specialist mistakes, margins, family agreement, and 12 quality values. |
| 07 | Extend the correctness head with squared evidence terms and evidence-by-quality interactions. |
| 08 | Add the 12 quality values to the previous asymmetric-cost temporal neural head. |
| 09 | As 08, but subtract each window's valid-frame embedding mean before the temporal head. |
| 10 | As 08, but use consecutive-frame embedding differences with valid-pair masking. |

Arms 01–07 use class-balanced logistic fitting with fold-local scaling and inner
selection among L2 values 1, 0.1, 0.01 (in that tie-break order). A matched
rank-only control gets the same L2 search, separate from the ten experiments.
The previous fixed-L2 specialist, BCE hybrid, and asymmetric-cost hybrid remain
frozen historical controls under the same policy selector. Deterministic models
are evaluated once, not represented as three independent replicates.

Absolute quality means energyMean, activeFraction, smallComponentEnergyFraction,
and abs(flowY), from the already registered flight configuration. Existing raw
court features join by exact private recording/rally/anchor identity; only the
current frozen labels are used. No protected labels or new media are loaded.
Feature 04 uses two signed contrasts: valid post-anchor minus pre-anchor mean,
and late-post minus early-post mean. Each 3840-dimensional contrast is projected
onto 32 Gaussian columns drawn once with seed 271828 and scaled by sqrt(32).
Missing-side contrasts are zero. This fixed, label-free map is not fitted on
held sources. Feature 05 centers six pre/contact/post near-minus-far flowMean
and activeFraction ranks at 0.5 before multiplying by the four absolute means.

For correctness arms, three fixed-L2=0.1 teachers use full, court, and flight
ranks. Teacher probabilities used to label a gate-training example exclude
that example's entire source group in addition to the gate's excluded groups.
Thus inner gates use three-group-excluded teachers; global out-of-fold teacher
scores are not reused. The gate predicts whether the full teacher's routed
side is correct, using seven probability/margin/agreement terms and 12 quality
values. Its routing remains the teacher's side. Arm 07 additionally uses seven
squared evidence terms and 84 evidence/quality products. Correctness scores
are class-balanced decision scores, not calibrated correctness probabilities.

Arms 08–10 retain the previous 237-feature fold-fitted logistic skip, frozen
encoder, compact temporal architecture, bounded residuals, false-positive
weight 49, and residual penalty 0.001. Their specialist representation expands
from 237 to 249 inputs with quality scaling fitted only on training rows.
Use AdamW 0.001, weight decay 0.0001, batch 32, dropout 0.1, gradient clipping 1,
source-balanced/square-root-class-weighted examples, deterministic FP32, and
seeds 3407/1729/20260918. Each neural arm requires 108 sequential nested fits.
The centered transform uses valid frames only. The difference transform marks
the first frame and any invalid pair unavailable and preserves time coordinates.

Report all three targets, accepted precision by side and source, correct and
wrong automatic counts, reviews/counts/rates, seed variation, and predeclared
three-seed consensus. Main neural results are means across seeds. Candidate
qualification at the primary target requires pooled, both-side, and source-macro
precision >=98% on these development summaries; among qualified candidates rank
by review rate. If none qualify, state that directly instead of calling the
highest point estimate a validated winner. This is a development ranking only.

Select final research configurations on full-development source-held scores,
then refit each arm on all development data. Final in-sample predictions are
never reported as evaluation. Preserve all selected models and policies privately.

Durable artifacts use ledger `private-reference-0240`, SSD work storage uses
`private-reference-0241`. The initial implementation registration and stopped
run remain archived under `private-reference-0238` / `private-reference-0239`.
Before any nested comparison was evaluated, an empty-batch reshape in final
correctness-head export was found and fixed. All fitting is restarted under the
corrected registration; completed and partial initial fits are not reused.
Run a single fitting process with two CPU threads.
Do not train arms in parallel. Load the embedding array once; reuse it and
release per-arm GPU tensors. Record resident memory and WSL swap counters.
Independent checkpoint replay starts after fitting exits. Public artifacts use
only stable indexes and aggregates; exact rows, locations, and logs stay private.

The Tasks connector is unavailable; this checklist tracks the work:

- [x] Qualify new feature, transformation, and cross-fitting code (nine tests).
- [x] Freeze input/code hashes and the ten-arm contract.
- [x] Run ten sequential experiments and matched controls.
- [x] Independently replay policies, metrics, models, and provenance.
- [x] Report every experiment, 95%/98%/99% review workload, and next steps.

See the [completed results and next steps](serving-side-precision-iterations-results-2026-10-14.md)
and [independent audit](serving-side-precision-iterations-audit-2026-10-14.json).
