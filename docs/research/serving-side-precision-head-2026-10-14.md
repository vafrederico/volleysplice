# Serving-side precision with asymmetric training costs

## Completed result

The new objective did **not** meet the main 98% precision target on held
sources. At the policy selected for 98%, its mean accepted precision was
**97.49%**, with **28.44%** of serves needing review. The previous BCE hybrid
reached 97.66% precision with 32.40% review. The new objective accepted more
serves but also accepted more wrong answers; this is not a precision improvement.
No production change is warranted from this comparison.

These are nested source-held development results on **968 reviewed true serves,
26 recordings, and eight source groups**. The denominator for review workload
is all 968 serves. Precision is correct automatic side answers divided by all
automatic side answers. It does not measure serve detection or clear an
independent serve/no-serve review flag. Protected test labels were not opened.

The new head uses the same frozen DINO-distilled MobileNetV3-Large embeddings
over 13 ordered frames from three seconds before through three seconds after
the anchor, plus all 237 existing specialist features. Its changed objective
weights false-positive side evidence 49 times more heavily than missed-positive
side evidence. Final research weights are saved, with no best-seed selection.

### New head: requested precision versus review workload

Counts and metrics below are means across the three registered seeds; fractional
counts are averages, not fractional serves. Requested precision is the inner
selection target, not an achieved outer precision claim.

| Selection target | Measured precision | Automatically handled | Reviews / 968 | Review rate | Correct automatic / 968 | Wrong automatic / 968 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 95% | 95.57% | 94.08% | 57.3 | 5.92% | 870.3 | 40.3 |
| **98%** | **97.49%** | **71.56%** | **275.3** | **28.44%** | **675.3** | **17.3** |
| 99% | 98.17% | 37.60% | 604.0 | 62.40% | 357.3 | 6.7 |

The stricter 99%-target policy reduced errors further but still missed 99%.
It must not be relabeled as an independently validated 98% policy after seeing
these results. At the main target the new head left 1.79 wrong automatic answers
per 100 total serves, compared with 1.58 for the previous BCE head.

### Controls under identical nested selection

Each cell gives measured accepted precision / percentage of all serves needing
review. The logistic controls are deterministic; repeated seed entries do not
provide additional independent evidence.

| Model | 95% target: precision / review | 98% target: precision / review | 99% target: precision / review |
| --- | ---: | ---: | ---: |
| Full-feature logistic specialist | 95.53% / 9.81% | 98.10% / 56.51% | 99.27% / 85.85% |
| Previous BCE neural hybrid | 95.67% / 6.96% | 97.66% / 32.40% | 98.27% / 65.98% |
| Full/court/flight evidence agreement | 95.79% / 19.01% | 94.19% / 83.99% | 100.00% / 95.56% |
| New asymmetric-cost neural hybrid | 95.57% / 5.92% | 97.49% / 28.44% | 98.17% / 62.40% |

Overall precision can hide a weaker side. At the main target:

| Model | Near precision | Far precision | Correct automatic / 968 | Wrong automatic / 968 |
| --- | ---: | ---: | ---: | ---: |
| Full-feature logistic specialist | 98.47% | 97.48% | 413.0 | 8.0 |
| Previous BCE neural hybrid | 98.15% | 97.15% | 639.0 | 15.3 |
| Evidence agreement | 97.50% | 82.86% | 146.0 | 9.0 |
| New asymmetric-cost neural hybrid | 97.64% | 97.36% | 675.3 | 17.3 |

Thus none meets 98% for both sides at the main target on outer predictions.
The specialist's 98.10% pooled result also carries a large review burden: 547
of 968. Two of its eight outer policies had no feasible inner solution and
therefore reviewed their entire outer source. The agreement control had three
such policies. Neural variants had none at this target. The report's counts
of six and nine infeasible policies include three identical control seed copies.

Evidence agreement did not solve confidence errors. At its 99% target it was
correct on all 43 accepted serves, but reviewed 925; that small accepted sample
does not establish reliable 99% precision. Different selected policies can also
accept different source mixes, so higher selection targets need not produce
monotonically higher held-source precision.

### Seed agreement and the historical production policy

The predeclared secondary consensus requires all three seed policies to accept
and agree. For the new head, the 95/98/99-target policies respectively measured
96.34% / 97.87% / 98.39% precision, with 94 / 359 / 720 reviews out of 968.
At the main target it made 13 wrong automatic answers among 609 accepted serves.
The previous BCE consensus made 11 wrong answers among 561 accepted, measuring
98.04% precision with 407 reviews. Its far-side precision was still only 97.38%.
Consensus therefore did not give the new head a reliable 98% solution either.

For context, the historical held-source specialist with production's original
side-uncertainty rule accepted 931 serves, reviewed 37, and made 45 wrong
automatic side answers: 95.17% accepted precision and 3.82% review. This policy
was not designed for the newly requested precision target. Its separate serve
gate is excluded from this direct side-head comparison. Exact deployed-weight
results are in-sample and are not substituted for held-source evidence; see the
[earlier review analysis](serving-side-review-2026-10-13.md).

### Why the new objective is insufficient, and next experiments

The loss changes the incentive and the score scale, but does not add evidence
about an ambiguous serve. Its validation thresholds did not transfer uniformly
to unseen source groups. At the main target, the three new seeds reached
98.14%, 97.22%, and 97.12% pooled precision. Selecting the first after observing
these outcomes would invalidate the registered comparison. In the secondary
consensus, source-group-012 still contributed four errors among 35 accepted
serves (88.57% precision), while three other source groups had no accepted errors.
Agreement among similar heads can preserve shared mistakes.
Before consensus, that source measured only 84.21–89.47% accepted precision
across seeds despite having just 56 of the 968 serves. Its weakness affected
both sides, so the aggregate failure is not explained by one uniformly weak side.

The [scalar quality diagnostic](serving-side-quality-diagnostics-2026-10-14.json)
used an existing frozen specialist at a fixed 0.9 confidence cutoff, separately
from policy selection. Ten of 632 accepted guesses were wrong. Those ten had
lower median absolute motion energy and vertical flow, and a larger median
small-component energy fraction. Five errors came from one source, seven were
near labels, and the relationship varied across sources. Existing relative
ranks already encode related information; this is evidence for a focused
experiment, not a validated weak-motion rejection rule.
Those diagnostics mainly describe specialist errors in source-group-005;
they do not establish the physical cause of the new head's errors in
source-group-012.

Prioritize these follow-ups in a new frozen development protocol:

1. Add four cached absolute quality measurements alongside the existing rank
   features: motion energy, active fraction, small-component energy fraction,
   and absolute vertical flow. Test whether they help identify unreliable
   evidence while preserving coverage at the same precision target. Active
   fraction is effectively capped near 10% by the extractor's percentile
   threshold; it is not an unrestricted moving-pixel occupancy estimate.
2. Add temporal embedding differences and stability under small anchor shifts.
   Prior diagnostics found large source variation in embedding levels and much
   less in before/after differences. Preserve court context while testing whether
   explicit motion changes reduce dependence on recording appearance.
3. Train a separate correctness/review head on honestly cross-fitted mistakes,
   including specialist margin, family disagreement, and absolute quality.
   Inner calibration would require teachers excluding outer, calibration, and
   teacher-validation groups; reusing global out-of-fold mistakes would leak
   outer information into training targets.
4. If cached features remain insufficient, add explicit server/ball position
   relative to court geometry and validate on additional independent sources.
   Eight source groups provide limited evidence for transfer at 98–99% precision.

### Verification and saved artifacts

All 108 nested fits and three final refits completed sequentially. The 34 focused
tests and 24 independent selector self-checks passed. The
[independent audit](serving-side-precision-audit-2026-10-14.json) verified the 108
new fits, 108 prior BCE fits, three final fits, 37 control partitions, all 288
outer policies and 36 final policies, and all aggregate/consensus metrics.
It reloaded 105 distinct selected outer checkpoints with exact GPU logit replay;
maximum final CPU/GPU score difference was 2.83e-7. Public and private aggregate
reports agree, and frozen input/source hashes match.

The [resource receipt](serving-side-precision-resources-2026-10-14.json) recorded
at most one training/evaluation process, peak resident memory of 1,861 MiB,
zero process swap, and no increase in WSL swap-in/out counters over 22.5 minutes
of monitoring. WSL available memory stayed above 12.66 GiB. These are sampled
Linux/WSL observations, not a host-wide Windows paging audit. The independent
checkpoint audit ran only after training/refitting exited.

All private rows, checkpoints, provenance, and logs are preserved under ledger
artifact `private-reference-0236`, with SSD work storage at
`private-reference-0237`. The immutable registration hash is
`c8743c0d643226d405f21f5315f6dee2bb3848ab3eee3ff080141081f2feb25d`.
The [complete aggregate report](serving-side-precision-head-2026-10-14.json)
contains all per-source, per-side, seed, consensus, and delivery details.
These remain development results: prior frozen-encoder exposure and the use
of reviewed-candidate recording ranks limit an end-to-end production claim.

## Registered objective

The user selected 98% precision among automatically accepted serving-side
predictions, with 95% and 99% comparisons. The practical recall measure is how
many of all serves can be handled automatically and how many need human review.
Always report correct automatic answers and wrong automatic answers as well as
accepted/review counts. A target is an empirical operating-point requirement,
not a guaranteed accuracy level on a new source.

Use the previous 968 reviewed non-beach serves, 26 recordings, eight source
groups, fixed anchors/exclusions, frozen encoder, ordered 13-frame embeddings,
and all 237 specialist features. Keep protected test labels out of fitting,
selection, and reporting. These are known true serves; side acceptance cannot
clear the independent serve/no-serve review gate.

The earlier rejection-loss prototype is archived under `private-reference-0234`,
with its fast store `private-reference-0235`. It was stopped after training
acceptance commonly became extremely small. It has no completed nested
comparison and is not claimed to be inferior on held data. Its coarse acceptance
cutoff grid also limits what can be inferred about those tiny scores' ranking.
This follow-up freezes a different objective before its own fitting.

## Training target

Use the same temporal/237-feature hybrid representation and fold-fitted frozen
logistic base b. Replace the single correction with two bounded corrections,
r_far and r_near, each 3*tanh of a learned output. Their logits are:

```text
u_far  = -b + r_far  - log(49)
u_near =  b + r_near - log(49)
```

Each output independently learns whether its side is correct. Weight a false
positive 49 times as strongly as a missed positive, using stable BCE with logits:

```text
w_near = 1 when y=near, else 49
w_far  = 1 when y=far,  else 49
loss = (w_near*BCE(u_near, y) + w_far*BCE(u_far, 1-y))/50
       + 0.001*mean(r_far^2, r_near^2)
```

Unlike assigning a cheap constant review loss, positive labels keep a gradient
that rewards useful correct predictions even when both outputs are very small.
For a calibrated underlying probability p, the population optimum of either
weighted binary loss is p/(p+49*(1-p)); a raw score of 0.5 therefore corresponds
to p=0.98. Finite data, regularization, group weighting, and source shift prevent
interpreting this relationship as a measured precision guarantee.

Let f=sigmoid(u_far), n=sigmoid(u_near), q=n/(n+f), and a=n+f. The shared evaluation
API receives [q,a], so its routed accepted-side score is exactly f or n.
Because u_far+u_near <= 6-2*log(49)<0, f+n<1 and the remaining mass can express
review. These are cost-transformed decision scores, not calibrated probabilities
of correctness or of needing review. Low a alone is not a collapse criterion.

Train all epochs with this loss, without warmup. Use seeds 3407, 1729, 20260918;
checkpoints 5/15/30/60; deterministic FP32 AdamW at 0.001, weight decay 0.0001;
batch 32; dropout 0.1; gradient clipping 1.0; and the prior source-balanced,
square-root-class-weighted examples. Preserve fold-local base fitting at L2=0.1.

## Evaluation and operating-point selection

Use nested source-held evaluation: eight outer groups, with the other seven used
for inner source-held predictions. Twenty-eight two-group exclusions and eight
single-group exclusions yield 108 sequential neural fits across the three seeds.
Every inner model excludes the outer source and its own validation source.
Select operating points exclusively on the concatenated inner predictions.

Route near when q>=0.5, otherwise far. Accept when that side's score reaches its
cutoff, except that zero acceptance always requests review. Search separate
far/near cutoffs from 0 through 0.99 by 0.01, plus 0.995, 0.999, 0.9999, 1 and 2.
The value 2 rejects everything. The route rule prevents overlapping decisions.

For each target 95%, 98%, and 99%, require both near and far accepted precision
at least the target, at least 20 accepted predictions per side, each accepted
side present in at least four inner source groups, and source-macro accepted
precision at least the target. Sources with no accepted predictions have zero
coverage and undefined precision. They are excluded only from the precision
mean; reviewing an entire difficult source is allowed and explicitly reported.

Among feasible policies maximize source-macro coverage, then pooled coverage,
minimum side precision, fewer errors, earlier epoch, larger cutoff sum, larger
minimum cutoff, larger far cutoff, and larger near cutoff. If infeasible, return
review-all with undefined precision and an explicit failure flag. Do not claim
100% precision when no predictions were accepted.

Use the same selection rule for four predeclared variants:

1. New asymmetric-cost neural head.
2. Previous full-correction BCE hybrid, reusing all frozen fold checkpoints.
3. Fold-fitted logistic specialist, with acceptance factor one.
4. Evidence agreement: separately fit full-237, court-flow-82, and flight-155
   logistic models at L2=0.1 inside each partition. Request review on any side
   disagreement; otherwise use the weakest probability for the chosen side.

The neural head is trained once with the 98% cost tradeoff. Targets 95/98/99
change its validation-selected policy, not its training labels. Main summaries
include all three seeds and their mean metrics. No best-seed selection. A
secondary consensus accepts only if all seed policies accept and agree. Apply
that same consensus rule to both neural variants; deterministic controls have
identical seed entries. Report per-source results and infeasible outer policies.

Freeze all outer predictions before final research refitting. Select final
epoch/cutoffs on full-development source-held predictions, then fit each seed on
all development rows. Final in-sample predictions are not an evaluation result.

## Interpretation, storage, and work tracking

Report near/far and pooled accepted precision; accepted/review counts and rates;
wrong automatic answers per 100 serves; correct automatic answers per 100;
source-macro coverage/precision; worst supported-source precision; and capture
of underlying incorrect guesses by review. Descriptive Wilson bounds do not
certify performance under source clustering and repeated policy selection.
All prior encoder exposure and reviewed-candidate rank-population limits remain.

Durable artifacts use ledger index `private-reference-0236`; fast artifacts use
`private-reference-0237`. Exact paths, media identities, row scores, checkpoints,
and logs remain external to Git. Use one model-training process with sequential
fits. GPU checkpoint audits run after training, and monitor host/WSL memory and
swap. Training arrays are loaded once into the process and reused across folds.

The Tasks connector is unavailable; this checklist tracks the work.

- [x] Confirm the 98% target and 95%/99% review-workload comparison.
- [x] Inspect prior features and prototype rejection-loss behavior.
- [x] Freeze and qualify the revised objective and common policy selector.
- [x] Run sequential nested training and the four model comparisons.
- [x] Refit final heads and independently audit all policies and checkpoints.
- [x] Report the measured precision/review tradeoff and grounded next options.
