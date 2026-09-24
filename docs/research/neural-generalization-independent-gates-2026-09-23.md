# Independent generalization checks

All generated evidence is under
`private-reference-0141`.
These checks validate provenance and computation, not the accuracy of an unscored model.

## Production replay

`scripts/audit-generalization-production-replay.py` compared a fresh production
replay with the historical original-eight exact-label replay. The new input
manifest removed rallies, ignored intervals, hard negatives, keep targets and
game windows before inference. The shipped code and three rally assets were pinned.

All eight recordings passed. Default suppression, unsuppressed union, previous-head
and v2-head interval endpoints had exactly zero error. All seven probability heads
and timestamps were bit-identical. The same cached AV features were intentionally
reused; this does not qualify a new feature extractor or establish new-video accuracy.

`production-replay-qualification-v1/audit.json` SHA-256:
`5bc132c04ec32f3ffc7766d1877fb93b78dcba4367a0a1c252176b03b16af83b`.

## Approximate export-rally registration

Five dedicated tests exercise the actual proxy registrar, including preservation of
base gold, source association, ignored spans and outside-game-window masks, rejection
of overlapping cores/protected inputs, and paired source-disjoint training/calibration
movement. The actual normalized inputs and 674 saved proxy cores also passed a
48-task membership audit. Calibration alternates the entire Aug29/Aug16 source group;
the common protected and September17 sources never enter fitting or selection.

`inventory-v1/proxy-registration-audit-v1.json` SHA-256:
`2889ada391f393492af4cef4ef9a898dd52fef3093f21b5acf135403bc900673`.

## Recall-floor selection

`scripts/audit-neural-generalization-selection.py` is an independent CPU-only gate
for each frozen selection. It reconstructs the calibration labels, full-timeline
validity and saved score ownership; the original-corpus arm instead reconstructs
the historically masked, source-held original-eight OOF population. It separately
checks exact and approximate export-core selection policies.

The gate decodes all 192 registered candidates and computes precision, recall and
F1 using the pinned independent endpoint-sweep arithmetic. It checks every saved
recall/F1 value and reselects all eleven floors. Eligibility uses the exact saved
IEEE value without tolerance or fallback, including 100%; equal F1 retains the
first registered candidate. No protected/common-panel outcomes select a point.

Registration and its archived source closure, actual fitting/scaler memberships,
head architecture, checkpoints, source groups, and student encoder/image/teacher
identities are hash-bound. The student initialization is checked against the prior
frozen MobileNet registration. Each passed output names its precise replay limits:
this selection gate does not re-run the neural network, reconstruct optimizer
arithmetic or independently inspect final BatchNorm buffers.

Four mutation tests passed: altered candidate recall, strict-100% and ordered-tie
selection, same-source auxiliary leakage, and preserving all inference ticks while
attaching approximate proxy boundaries/ignored spans. A real original-corpus AV fit
also passed its owner/code/source gate. Complete selection outputs are pending the
historical refit audit and frozen selection artifacts; no blanket study pass is
claimed before those outputs exist.

Single-artifact integration:

```sh
python scripts/audit-neural-generalization-selection.py \
  --selection private-reference-0142 \
  --output private-reference-0143
```

For multiple artifacts, `--jobs` accepts a JSON array with `task`, `fitDirectory`,
`selection`, `output` and optional `historicalDirectory`. A shared immutable-file
verification cache avoids repeatedly hashing historical evidence; every reference
is checked again for mutation before writing a passed result.

## Final result and report gate

`scripts/audit-neural-generalization-numerics.py` supplies the fit and inference
gates. The fit phase independently reconstructs source membership, exact-tier
normalization, supervision counts, class weights and deterministic sampler
exposure. It checks all four saved checkpoint arrays and replays the first,
middle and last real-context calibration chunks on CPU. Original full-corpus
fits have only a one-tick, label-free output probe; their external predictions
are checked in the separate inference phase.

For distilled models, the fit gate also checks the permitted teacher-frame
population, eight-epoch student exposure, frozen BatchNorm buffers and initial
checkpoint lineage. Sampled image-to-embedding replay binds the learned student
to the features used by its temporal head. The inference gate checks all 42
recordings and all four epochs against their saved full-timeline inputs, with
three sampled context chunks per recording and checkpoint. This is explicitly
sampled numerical replay, not an exhaustive second forward pass or optimizer
replay. Inference receives no human-label masks.

The queue exposes `audit-fit` and `audit-inference` actions for these CPU-only
checks, runs the fitting gate before calibration selection, and runs the
inference gate before scoring. Outputs are immutable NAS receipts. Four
mutation tests cover checkpoint ownership, label-blind declarations, malformed
probabilities, actual model-forward mismatches and teacher-frame membership.

`scripts/audit-neural-generalization-results.py` is prepared for the complete
162-task, 270-neural-precision result index, both fixed production comparators and
the three historical precision reports. It requires the separate fitting and
sampled neural-inference numerical gates, re-decodes every selected point from
saved probabilities, and independently checks every declared panel and exposure
filter. Production receives a fresh CPU TypeScript replay on all 42 cached AV
recordings. No labels enter that replay.

The interval oracle checks all four padding cases under each of the three label
policies. Exact-label results additionally retain independent original-rally
coverage slices and lost-event details in a separate JSONL supplement, alongside
canonical event matching. Historical projections must retain the original eight
recordings, every required model and all three seeds. Complete-draw means require
every registered draw and exactly the same recording scope.

Four synthetic regression tests passed, including rejection of a narrowed
historical projection. The real normalized42 gold reconciliation also passed:
exact/draft intervals and tags, coverage keep targets, reviewed windows and effective
ignored universes agree with the authoritative inventory. Frozen original18
documents remain unchanged. The four tiny Aug29 cores are not coverage evaluation
truth, and the 0.0002-second raw ignored-tail difference is already excluded by its
reviewed game window.

The final gate is pending completed numerical artifacts. It will bind
`reportContentSha256` over `{inventory, scopes, series, tasks}`. After it passes,
the producer may add the audit flag and receipt while preserving that exact
numerical payload; real browser QA/publication then uses the audited payload.
