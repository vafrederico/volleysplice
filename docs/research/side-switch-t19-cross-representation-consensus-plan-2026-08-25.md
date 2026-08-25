# Side-switch T19 cross-representation jersey consensus — 2026-08-25

## Decision and isolation

T14's stable-medoid raw direction is useful but its internal reliability semantics
fail. T4's selective-far pooled-team direction is independently constructed and only
moderately correlated with T14. T19 tests whether agreement between those two
representations supplies reliability without using within-representation gates.

Use only T18 feature artifact SHA-256
`0efab13db25849a74413fc595e2753c735d1c9eb00ae4afaf8b73b9007ef8949`.
For each eligible boundary define from the exact T4 and T14 raw signed margins:

```text
cross-representation consensus    = (T4 margin + T14 margin) / 2
cross-representation disagreement = abs(T4 margin - T14 margin)
```

The two model inputs are exactly:

- `crossRepresentationJerseySwapConsensus`
- `crossRepresentationJerseySwapDisagreement`

Internal rows receive zeros. Do not add either raw margin separately, gate or clip
consensus, sign-code disagreement, change weights from `1/2`, decode video, or load
labels during transformation.

## Engineering gates

Require exact 704-row/prior parity, 624 eligible and 80 zero-filled internal rows,
formula reconstruction within `1e-12`, finite/nonconstant cores, consensus positive
and negative on at least 15% each, disagreement nonzero on at least 20%, core/core
absolute Spearman below `0.98`, consensus absolute Spearman below `0.95` with each raw
T4 and T14 margin, all other prior correlations below `0.98`, <=60 seconds, and
<=2,048 MiB RSS.

Failure stops before labels. Do not alter fusion weights or transform disagreement
after observing T19.

## Conditional model comparison

If engineering passes, commit exact boundary T0 (`34`) versus `34 + T19 two-value
core` before labels. Require +2 F1 points over T0, +0.5 point over immutable T14, all
prior precision/recall/strict/covered-miss/recording guardrails, reduction of
T0-selected false boundaries where T4 and T14 signs disagree or include zero, a
positive consensus coefficient in the full fit and >=9/11 folds, and a negative
disagreement coefficient in the full fit and >=9/11 folds.

This is adaptive opened-development evidence only. Even a pass cannot authorize
promotion or runtime work.

## Execution ledger

### Preregistration — frozen 2026-08-25

No T19 code, values, artifact, profile, or result existed at this commit.

### Engineering result — pass; model comparison authorized

Implementation is committed at `7f9de96`. The transformation wrote the 40,537,567-byte
artifact:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t19-cross-representation-consensus-features-v1.json`

SHA-256: `ab4b474fdbaee20bf884093a37eba25e79b6e93172f0dbfe619c3b0546c421e5`.

Consensus is positive on 24.52% and negative on 75.16%; disagreement is nonzero on
98.88%. Core/core Spearman is `-0.209400`, while consensus correlates `0.866367` with
T4 raw margin and `0.881434` with T14 raw margin. Every frozen parity, support,
formula, novelty, runtime, and memory gate passes.

Decision: **T19 passes engineering**. Only exact `34 + T19 two-value core` model
wiring may now be committed before label access.

### Matched model result — reject; adaptive loop stops

The exact 36-input wiring was committed at `c466730`. The one-shot comparison wrote
the 938,387-byte artifact:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-feature-development-t19-cross-representation-consensus-opened-v1.json`

SHA-256: `b67e2502f05cba27ba9a0b1fc19fd614506264b8319db0befd0bd17b33c615c7`.

Boundary F1 improves 50.98%→53.47% (+2.48 points), with one added TP and two removed
FP. Consensus is positive in the full fit (`+0.107102`) and all 11 folds;
disagreement is negative in the full fit (`-0.064633`) and all 11 folds. Despite that
perfect semantic transfer, T19 trails T14 by 1.77 points and retains all seven frozen
T4/T14 sign-disagreement false boundaries. The T14-comparator and target-slice gates
fail.

Decision: **reject T19 and stop before T20**. T15–T19 exhaust the credible transforms
of the cached T4/T14 representations: bilateral minimum, source decomposition, source
normalization, medoid representativeness, and cross-representation fusion. A T20 on
these same opened artifacts would require forbidden post-result source pruning,
softened consensus, fusion-weight selection, or quality-gate tuning. Resume only with
a genuinely new representation or new recording-held gold; do not manufacture T20 to
reach an experiment number.
