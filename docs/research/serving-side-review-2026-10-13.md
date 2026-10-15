# Serving-side review frequency and hybrid correctness

This post-hoc check applies the deployed review policy to the hybrid study's
968 reviewed true serves across 26 recordings and eight source groups. It does
not train models, select operating points, open protected labels for scoring,
or change production. Human-unclear and not-serve labels remain excluded.

The deployed side-score review band is near probability greater than or equal
to 0.3121748736511044 and strictly below 0.5028396703865513. A side guess still
exists inside that band, using the frozen 0.4783744762021848 near threshold.
Production also requests review when both serve heads fall below 0.85 in the
one-second anchor window but a both-model or neural rally retains the candidate.

The primary comparison uses historical source-held specialist scores and the
hybrid's already frozen nested source-held scores. Exact deployed-weight results
are a separate descriptive replay: those weights were fitted on this corpus.
The hybrid ensemble uses mean probabilities and mean inner-selected thresholds
across all three seeds. No seed or threshold is chosen using this subset.

The Tasks connector is unavailable; this checklist tracks the analysis.

- [x] Read the deployed side-score and serve-gate policy.
- [x] Join the reviewed population to stored specialist and serve-gate evidence.
- [x] Calculate review subsets and frozen hybrid correctness.
- [x] Verify against the production TypeScript policy and record limitations.
- [x] Check the public artifacts for private identifiers.

Artifact indexes: hybrid study `private-reference-0231`, historical gate evidence
`private-reference-0232`, deployed-weight replay `private-reference-0233`.
The reproducible entry point is
[`analyze-serving-side-review.py`](../../scripts/analyze-serving-side-review.py).

## Results

The table reports ordinary count accuracy of the three-seed hybrid ensemble on
each flagged subset, not the whole-study source-macro balanced accuracy reported
previously. Each model's operating points remain fixed. The specialist column is
its underlying near/far guess, even though its displayed verdict requests review.

| Specialist scores and review rule | Flagged / 968 serves | Specialist side correct | Hybrid side correct |
| --- | ---: | ---: | ---: |
| Source-held, uncertain side score | 37 (3.82%) | 24/37 (64.9%) | 26/37 (70.3%) |
| Source-held, historical gate plus side score | 66 (6.82%) | 48/66 (72.7%) | 49/66 (74.2%) |
| Source-held, all anchors treated as neural candidates | 87 (8.99%) | 65/87 (74.7%) | 65/87 (74.7%) |
| Deployed weights, uncertain side score | 21 (2.17%) | 11/21 (52.4%) | 6/21 (28.6%) |
| Deployed weights, historical gate plus side score | 49 (5.06%) | 36/49 (73.5%) | 30/49 (61.2%) |
| Deployed weights, all anchors treated as neural candidates | 70 (7.23%) | 53/70 (75.7%) | 46/70 (65.7%) |

On the 37 source-held side-score cases, the hybrid fixes four specialist errors
and introduces two; nine remain wrong for both. Individual seeds get 23, 27,
and 26 correct (62.2%, 73.0%, and 70.3%). Their mean accuracy is 68.5%; the
ensemble is 70.3%. Fourteen cases are near and 23 far; the ensemble's pooled
balanced accuracy is 63.5%, so the 70.3% ordinary accuracy should not be confused
with the earlier 93.51% whole-study source-macro balanced accuracy.

On the broader 87-case neural-candidate scenario, the hybrid fixes four errors
and introduces four. It offers no net gain there. On the deployed-weight
21-case side subset, the individual held-out hybrid seeds get six, eight, and
seven correct. These small subsets do not establish a reliable automatic
replacement for human review.

The historical gate evidence passes at least one serve head for 916 cases.
Both heads are weak for 52: the historical both-model rally gate recovers 31
for review and rejects 21 as not-serve. A neural rally candidate would instead
retain all 52 for review. The latter calculation assumes candidate detection;
it does not establish that the current rally model actually finds those anchors.

## Interpretation and limits

The deployed weights' 21-case subset answers the literal stored production-model
question on these recordings. It is not an unbiased estimate for new videos:
the weights were fitted on this development corpus, while the compared hybrid
predictions exclude the entire evaluated source. This unequal exposure makes
the deployed-weight comparison descriptive, not a fair model ranking.

The historical source-held specialist comparison is more useful for generalization,
but its review band and side threshold were selected on development data, and
its original fits included beach examples. The hybrid uses the already audited
nested non-beach protocol. The prior frozen encoder's exposure limitations still
apply. No parameters were selected using these review slices.

The saved deployed replay and the hybrid study use slightly different historical
candidate populations for within-recording feature ranks. Recomputing the exact
deployed model on the hybrid's frozen 237-feature matrix changes some scores
(maximum absolute change 0.0240), but changes zero review-band memberships. The
21 flagged cases and the hybrid's six correct outputs are identical. The
specialist's binary guess changes on one of those cases (12 correct instead of
11); the report keeps the saved production replay as its literal comparison.

All 968 cases already have a reviewed near/far serve label. This study does not
measure production traffic, missed rallies, false serve detections, or rows humans
marked unclear. Correct side predictions do not resolve a weak-head candidate's
serve-versus-no-serve uncertainty. Historical gate counts describe known anchors,
not a fresh end-to-end run. Production remains unchanged.

## Verification

All 968 identity joins and gate anchor timestamps agree. The gate models, runtime
fingerprint, threshold, window, ordered features, and frozen hybrid input hashes
are checked. Production TypeScript reproduces 8,712 verdicts across the three
score sources and three rally-agreement cases. Its exact-weight probability
calculation agrees within 5.56e-16. Four boundary checks verify review-band and
serve-head threshold inclusion. The aggregate machine-readable results are in
[`serving-side-review-2026-10-13.json`](serving-side-review-2026-10-13.json).

An independent pass reconstructs all 48 frozen outer-source selections for both
hybrid variants from their hash-verified saved logits, with zero probability
error and exact threshold agreement. It also verifies aggregate counts from
the private row artifact, public/private report equality, and the three new
public files against the private-ledger privacy detectors. No branch publication
or production deployment was performed.
