# Precision-first serving-side training

## Prototype status

This initial learned-rejection formulation was stopped after 20 completed fitting
partitions, with the next incomplete partition also preserved. Its fitting-row
acceptance commonly became very small; the fixed cutoff grid also could not
resolve fine rankings among scores below 0.01. No completed nested evaluation or
held-source superiority/inferiority claim is made for this prototype.

The frozen preregistration, exact original source snapshot, completed checkpoints,
partial checkpoint directory, training log, and stop receipt remain under the
original artifact indexes below. The active follow-up uses a separately registered
asymmetric-cost objective, documented in
[`serving-side-precision-head-2026-10-14.md`](serving-side-precision-head-2026-10-14.md).
The protocol below records the initial intended experiment, not completed results.

## Registered question and target

The user prioritizes avoiding wrong automatic serving-side predictions over
avoiding human review. The primary target is 98% precision among accepted
predictions, with 95% and 99% comparisons. Report accepted serves, wrong automatic
answers, and reviews as fractions of all 968 serves. Correct automatic answers
divided by all serves is also reported; review fraction is the direct measure
of the user's review workload.

Preserve the existing 968 reviewed non-beach serves, 26 recordings, eight source
groups, reviewed anchors, exclusions, frozen encoder, 13-frame windows, and 237
specialist features. Keep the existing studies immutable. This scope evaluates
near/far on true serves. The serve/no-serve recovery gate still requires its own
review decision. No protected test labels participate in this experiment.

## Changed training objective

The existing hybrid supplies a fold-fitted logistic logit plus a bounded neural
correction over temporal embeddings and specialist features. Add a learned
acceptance output from the same fused representation. Let q be near probability,
a be acceptance probability, and p_true be q or 1-q according to the human label.
The three outputs are far=a*(1-q), near=a*q, review=1-a. Initialize a to 0.9.

Train five warmup epochs with the existing BCE and residual penalty. Thereafter:

```text
loss = -log(a*p_true + 0.98*(1-a)) + 0.1*BCE + 0.01*residual^2
```

This is a modified [Deep Gamblers loss, equation 6](https://proceedings.neurips.cc/paper/2019/file/0c4b1eeb45c90b52bfb9d07943d855ab-Paper.pdf),
with payoff 1/0.98 and an auxiliary classification loss. The 0.02 rejection cost
expresses the intended training preference; it does not guarantee 98% observed
precision. Epoch five is a warmup control and must be identified if selected.
Human-unclear labels and the incumbent review flag are not used as artificial
training targets. Acceptance is learned jointly from the reviewed side labels.

Use the prior seeds 3407, 1729, 20260918, epochs 5/15/30/60, source-balanced and
square-root-class-weighted examples, deterministic FP32 AdamW at 0.001 with
weight decay 0.0001, batch 32, dropout 0.1, and gradient clipping 1.0. The encoder
and logistic skip remain frozen within each fitting partition.

## Selection and controls

Use nested source-held evaluation: exclude each outer source from all fitting and
operating-point selection. Inner validation excludes one additional source.
The 28 distinct two-source exclusions and eight single-source exclusions require
108 new neural fits across three seeds. All transformations and supporting
classifiers are fitted only on the applicable training sources.

Route near when q>=0.5, otherwise far. Accept only when the routed side's score
a*q or a*(1-q) reaches its selected cutoff. The common cutoff grid is 0 through
0.99 by 0.01, plus 0.995, 0.999, 0.9999, 1, and 2 (review all). This prevents
overlapping class decisions even if both cutoffs are below 0.5. A zero acceptance
score always requests review, including when a cutoff is zero.

For each target, select epoch and separate near/far cutoffs using only inner
held-source predictions. Feasibility requires both near and far precision at
least the target, at least 20 accepted predictions per side, each accepted side
represented in at least four inner sources, and source-macro accepted precision
at least the target. Sources with no accepted predictions have undefined
precision and zero coverage; they are excluded only from the precision mean.

Among feasible policies maximize source-macro coverage, then pooled coverage,
minimum side precision, fewer errors, earlier epoch, larger cutoff sum, larger
minimum cutoff, larger far cutoff, and larger near cutoff. An infeasible search
returns an explicit review-all policy with undefined precision. Never describe
zero accepted predictions as perfect precision. Review of an entire difficult
source is allowed and is visible in the per-source coverage report.

Apply the same policy search to these controls:

- The earlier full-correction BCE hybrid, using its frozen fold checkpoints.
- The fold-fitted logistic specialist, with a=1.
- A new inexpensive evidence-agreement control: fit separate full-237,
  court-flow-82, and flight-155 logistic classifiers at L2=0.1 within each fold.
  If their side decisions disagree, acceptance score is zero; otherwise use
  their minimum probability for the full classifier's chosen side. This tests
  whether existing feature families provide useful disagreement evidence.

All controls receive the same precision targets and rejection-policy freedom.
The new learned-rejection model is trained at the 98% tradeoff once; the three
precision targets change its inner-selected policy, not the training labels.

Primary summaries report all seeds and their mean precision/coverage, without
choosing a winning seed. A predeclared secondary ensemble accepts only when all
three seed policies accept and agree on the side. The deterministic logistic
controls have identical predictions for all seed entries. Final research heads
use development source-held policy selection before all-development refitting.

## Reporting and verification

Report near/far and pooled accepted precision, coverage, review count/fraction,
wrong automatic answers per 100 serves, correct automatic answers per 100 serves,
source-macro and worst-source results, and how many underlying wrong guesses
were caught by rejection. Report infeasible folds, warmup selections, and
training/held acceptance behavior. Accepted precision is conditional on these
reviewed true-serve anchors. An empirical 98% target can be missed on held sources.
Descriptive Wilson intervals do not certify precision under source clustering or
repeated selection. Retain the prior frozen-encoder exposure limitations.

Verify loss behavior and gradient direction, identity and source isolation,
cutoff boundaries, empty acceptance, independent brute-force selector agreement,
saved checkpoint replay, final CPU/GPU consistency, and artifact hashes/privacy.
No production promotion is implied by this development experiment.

Fast artifacts are resolved through ledger index `private-reference-0235`;
durable artifacts through `private-reference-0234`. Previous data are resolved
through the existing ledger and their immutable registrations. Exact runtime
locations, row predictions, model files, and logs remain outside Git.

The Tasks connector is unavailable; this checklist tracks the work.

- [x] Confirm precision targets and review-workload reporting with the user.
- [x] Review the objective, leakage controls, and available feature alternatives.
- [x] Freeze the protocol and verify training/selection code.
- [ ] Train the reject-aware neural heads and evaluate the common controls.
- [ ] Refit research heads and independently audit decisions and metrics.
- [ ] Record precision versus review workload and evidence-based next steps.
