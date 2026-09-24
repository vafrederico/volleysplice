# Production and neural rally combinations: prospective protocol

The user requested a same-scope comparison of production, the current neural candidates, automatic combinations, and lower-confidence human-review configurations. This is a fixed-policy, adaptive development comparison using already-audited out-of-fold neural predictions. No new model fitting, threshold selection, protected-test evaluation or production deployment is included.

Primary ranking uses pooled `F1_padP_coreR` at symmetric two-second padding, with positive export gaps joined only when strictly below three seconds. Zero/one/two/three-second padding sensitivity is mandatory. Human and model ranges use identical canonical padding/clipping/merging/joining; ignored intervals are subtracted afterward and never rejoined. Each seed pools the same eight exact-label recordings/four development source groups before calculating ratios. Tables average the three seed scores and recording-summed seconds, without treating repeated seeds as new videos. Production is a single fixed result.

**Duration columns, as clarified by the user**

Let V be video time minus ignored intervals, M the model export, Hp the equally padded human export, and Hc core human play. Report export `|M|`, correctly removed unwanted footage `|V minus (M union Hp)|`, incorrectly removed wanted export `|Hp minus M|`, incorrectly exported unwanted footage `|M minus Hp|`, and missed actual play `|Hc minus M|`. The first three quantities partition V. Missed padded export and missed core play are different; recall uses Hc. Also retain human export duration and model-minus-human duration. Units are seconds, with readable minutes in the summary.

**Frozen candidate population**

- Compact TCN: reviewed-export short boost (prior best compact F1), reviewed-export baseline plus keep-head rescue, and the reviewed-export baseline reference.
- DINO+TCN: draft/global-positive control (prior best DINO F1), reviewed-export short boost (recovery result), reviewed-export baseline plus keep-head rescue, and the reviewed-export baseline reference.
- The first two compact candidates and first three DINO candidates participate in combinations. All use the original 125-tick temporal context and seeds 3407, 1729 and 20260918. Shorter context was already rejected and is not searched again.
- Production: shipped previous/v2 components and their union, source-group-held refitted components and union, and current fresh-draft whole-rally suppression defaults and fixed options. Current checked-in runtime/defaults and cached feature/label identities must be verified before reuse. Historical headline scores are not reused on a different population.

Production shipped weights have historical training exposure. Source-held refitted rally heads provide a separate sensitivity baseline, while their historical decoder/hyperparameter choices remain exposed. The NN candidates were chosen from prior development results. This is not an untouched generalization test.

**Fixed automatic configurations**

For each of the five participating neural models N, evaluate the following with three production anchors P: actual current-default core output, shipped unsuppressed union, and source-held refit unsuppressed union. Refit anchor source tags come from its own two refitted components; shipped/default anchors use shipped components.

1. Raw core union `P union N`.
2. Strict raw intersection `P intersection N` (aggressive research control).
3. Tolerant intersection `P intersection dilate(N, 2 seconds)` (aggressive research control).
4. Guarded trimming: keep all protected production time, trimming only one-model-eligible raw time outside two-second neural support.
5. Guarded whole-component rejection: remove an eligible one-model component only if none of its retained raw time has positive overlap with two-second neural support. Otherwise preserve its original boundaries.

Agreement protection uses source-tagged previous/v2 raw intervals, symmetric two-second agreement padding and positive agreement gaps strictly below 0.5 seconds. A component is protected when both sources occur anywhere in it. Protection covers full original raw heads and tails; agreement padding is not exported. Neural support dilation clips and merges overlaps only, without the final export's three-second joining. All automatic combinations happen before final canonical export processing. Strict/tolerant whole-production vetoes are research diagnostics, not accepted product options based on F1 alone.

Two predeclared compact/DINO pairs are used: compact-short-boost with DINO-draft-global, and compact-keep-rescue with DINO-reviewed-export-short-boost. Pair identical seeds. Evaluate their NN-only union/intersection, plus production union with both NNs and timewise two-of-three majority among P/compact/DINO for each production anchor. Do not select seeds or treat all nine cross-seed pairings as independent.

This gives 107 automatic configurations: nine production rows, seven single-NN rows, four NN-only pair rows, and 87 production/NN combination rows. Production aliases with identical defaults are not duplicated. These recipes are fixed before new combination quality is read; all results will be reported, not only the best row. No inner-fold policy optimizer is introduced. Any highest development score is descriptive and is not an unbiased post-selection estimate.

**Human-review configurations**

For every anchor/participating-NN pair, report two label-blind queues at each padding case: production export absent from neural export (suppression review), and the symmetric export disagreement (bidirectional review, including potential missed play). Automatic export remains the production anchor. A reviewer gets two seconds of playback context around disputed spans, merged with strictly-below-three-second review gaps and with ignored time removed. Playback context counts toward workload but grants no additional correction area.

Report review clips/seconds, disputed seconds, unwanted production export flagged, wanted production export flagged, omitted padded-human export within the disagreement, and omitted actual play within the disagreement. This is 30 queue policies across three matched seeds. Flags alone do not change automatic precision, recall, F1 or exports.

A separately labeled optimistic upper bound assumes perfect correction of disputed export seconds only: `(M minus D) union (Hp intersection D)`. It uses ground truth, performs no further padding or joining, cannot correct errors where both models agree, and is neither automatic model performance nor measured human-review performance. It is excluded from automatic ranking and event-F1 comparisons. No actual human-review outcomes are fabricated.

**Current app export fidelity**

Current whole-rally suppression can block short-gap rejoining across suppressed spans. The adapter must retain both exact app-materialized exports and canonical exports of the same core intervals for all four padding cases, and quantify any difference. Primary ranking follows the repository's canonical contract. If exact app exports differ, publish a separately named actual-app fidelity table, using the same padded-human/core-human references and ignored universe, without silently substituting it for canonical primary ranking. Preserve policy names, source hashes, suppression masks and barrier evidence. Do not change the app.

**Guardrails and verification**

Retain event F1, complete/partial/incomplete rally losses, short-rally (at most three seconds) losses, longer-rally core recall, per-seed and per-source-group effects, and exact newly lost/recovered rally identities versus each production anchor. A conservative development screen requires mean F1 gain at least 0.02, mean overall/long recall regressions no worse than 0.005, event-F1 regression no worse than 0.01, and no new complete or worsened rally coverage in any seed. It is separate from F1 ranking and does not authorize promotion.

Verify label/source-group/duration/ignored identity and immutable input hashes. Independently reconstruct duration metrics using endpoint sweeps, including all four padding cases and actual-export/oracle overrides without extra padding. Test strict three-second gaps, the distinct 0.5-second agreement gap, clipping, ignored-time subtraction, full-component protection, source-tag choice, majority voting, pooling, and review label independence. Save normalized inputs, frozen protocol/recipe/source hashes, every fixed result, audits, report and a reproducible source archive on the NAS.

Execution checklist (Tasks tool unavailable):

- [x] Verify and normalize current production/default and reference inputs.
- [x] Implement and qualify fixed combinations and independent accounting.
- [x] Freeze source/input/recipe identities before opening new hybrid outcomes.
- [x] Execute the complete fixed matrix and review diagnostics.
- [x] Independently audit metrics, duration partitions and candidate construction.
- [x] Write standalone/combination/review comparisons and all padding sensitivities.
- [x] Archive final sources and completed evidence; leave production unchanged.

Completed matrix report SHA-256: `556f44e55b56425ddd0693cfe7f997819fd68365aa29a18653296699062462e4`.
Registration: `c0656f85658e6eade8f4e8feacbeb88453186c3f1c3b7d38ab332eb105f47b27`.
The NAS `protocol-initial.md` preserves the prospective text and unchecked execution list; this working document records completion without modifying that frozen snapshot.
