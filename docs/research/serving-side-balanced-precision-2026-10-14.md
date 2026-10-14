# Balanced-encoder serving-side precision: registered comparison

Train the leading centered temporal serving-side head using the production
Balanced encoder (`high-f1`) instead of Maximum coverage (`high-recall`). This
is one matched encoder comparison, declared before Balanced serving-side outcomes
are inspected. It does not rerun or select among the earlier ten hypotheses.

Use the same 968 reviewed true non-beach serves, 26 recordings, eight source
groups, labels, anchors, ignored time, frame timestamps, FP16-rounded embedding
precision, 237 specialist ranks, and 12 phase-specific absolute quality features.
Require exact matching temporal masks and time coordinates. The frozen Balanced
weights must match the public production manifest; never substitute another
encoder or a missing ledger entry. Existing Balanced caches must prove the same
image-input lineage and labels-free extraction as the Maximum coverage caches.

Retain the centered 13-frame input spanning -3 to +3 seconds at 2 Hz, fold-local
scaling and logistic skip, asymmetric loss (false-positive weight 49), bounded
residuals, training weights, deterministic FP32, AdamW settings, and seeds
3407/1729/20260918 from experiment 09. Fit all 28 double-source and eight
single-source exclusions for each seed: 108 nested fits. Then freeze evaluation
and final policies before three all-development refits. No historical trained
serving head is reused. The encoders remain frozen.

Use the identical nested selector: 95%, 98%, and 99% precision targets; at least
20 accepted examples and four supporting inner source groups per predicted side;
both-side and source-average precision constraints; maximize source-average
automatic coverage, with the existing deterministic tie breaks. Infeasible
policies review all. Outer labels never choose epochs or thresholds.

Report means across all three seeds, individual seeds, and the predeclared
three-seed agreement policy requiring all three heads to accept and agree.
Precision compares automatic near/far answers with human labels; review workload
uses all 968 serves. Compare Balanced with the already frozen Maximum coverage
centered result at each target, including pooled/both-side/source-average
precision, review counts/rates, wrong automatic answers, and paired outcomes.
Qualification at 98% requires pooled, both-side, and source-average precision at
least 98%; minimize reviews among qualified policies. Do not select a best seed.

This is development head evaluation on known serves. Prior encoder pixel exposure,
historical rally-based encoder selection, and reviewed-candidate recording ranks
retain their limitations. Untouched-source and full production candidate-pipeline
confirmation are required before promotion. Do not open protected labels or
change production as part of this task.

Resolve durable output through ledger `private-reference-0242`, SSD working output
through `private-reference-0243`, Balanced encoder through `private-reference-0244`,
and its embedding bank through `private-reference-0245`. Run one fitting process
with two CPU threads and sequential seeds/folds; record resident memory and WSL
swap counters. Independently replay policies, metrics, input provenance, and
selected checkpoints after training exits. Private row predictions, exact paths,
and execution logs remain outside Git.

The Tasks connector is unavailable; this checklist tracks completion:

- [x] Verify encoder, matched input lineage, and unchanged fitting code.
- [x] Freeze the comparison and run 108 sequential nested fits plus three final refits.
- [x] Independently audit provenance, policies, metrics, and checkpoint replay.
- [x] Report all three targets, paired errors/reviews, resources, and next steps.

See the [completed comparison](serving-side-balanced-precision-results-2026-10-14.md)
and [independent audit](serving-side-balanced-precision-audit-2026-10-14.json).
