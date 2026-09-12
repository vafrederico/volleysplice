# v4/v5 add-only and boundary fusion — 2026-08-11

## Outcome

V4 and v5 are complementary, but the safe way to combine them is boundary refinement, not a raw
union. Validation contains eight v5 intervals that do not overlap v4 and **all eight are false**.
No threshold on v5 confidence, v4/v5 serve agreement, or v5-minus-v4 live support produces a new
strict validation match. The frozen add-only decision is therefore `disabled`.

For overlapping outputs, a small 17-option validation grid selected an intersection rule for an
unambiguous one-to-one v4/v5 component when all of these hold:

- v4/v5 interval IoU is at least 0.30;
- their starts differ by at most 1 second;
- v5 ends at least 1 second before v4;
- both composed confidence scores are at least 0.80.

All unmatched and one-to-many/many-to-one components keep v4. The rule preserves prediction count,
event precision, and ordinary-long strict matches on validation before optimizing event F1,
short/fault strict matches, and time IoU.

## Result

| Split and variant | Predictions / matches | Precision | Recall | F1 | Short | Ace | Fault | Ordinary-long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation v4 | 112 / 56 | .5000 | .7368 | .5957 | 6/21 | 7/10 | 3/13 | 46/51 |
| Validation v5 | 110 / 57 | .5182 | .7500 | .6129 | 8/21 | 6/10 | 6/13 | 45/51 |
| Validation selected | 112 / 59 | **.5268** | **.7763** | **.6277** | **9/21** | 7/10 | **6/13** | 46/51 |
| Retrospective test v4 | 59 / 31 | .5254 | .7949 | .6327 | 2/10 | 3/4 | 0/7 | 28/28 |
| Retrospective test v5 | 60 / 30 | .5000 | .7692 | .6061 | 5/10 | 3/4 | 3/7 | 24/28 |
| Retrospective test selected | 59 / 31 | .5254 | .7949 | .6327 | 2/10 | 3/4 | 0/7 | 28/28 |

Category values are strict-recall counts at interval IoU at least 0.5. Categories overlap, and the
predictions do not carry outcome labels, so category-specific precision and F1 are not defined.

The frozen selector changes 21 validation components and six retrospective-test components. Test
strict event counts do not change, but boundary quality does:

- time IoU: `.6520 -> .6689`;
- matches at IoU at least 0.7: `23 -> 25`;
- F1 at IoU at least 0.7: `.4694 -> .5102`;
- mean matched IoU: `.8051 -> .8227`;
- end-boundary MAE: `1.424 s -> 1.124 s`;
- retained dead time: `142.048 s -> 126.942 s`;
- live-time recall: `.9392 -> .9323`.

Thus the conservative selector generalizes as a trimming/refinement rule, but it does not recover a
new test rally at the headline threshold.

## Aggressive boundary diagnostic

Intersecting every unambiguous one-to-one overlap component is stronger but violates the ordinary
guardrail. It reaches validation precision/recall/F1 `.5357/.7895/.6383`, with 11/21 short and
8/13 fault matches. Retrospectively it reaches `.5424/.8205/.6531`, with 4/10 short and 2/7 fault
matches, but ordinary-long falls from 28/28 to 27/28. This is an ablation, not the frozen choice.

## Confidence and disagreement audit

The useful uncovered test fault has v5 composed confidence `.915`, below the two false uncovered
test candidates at `.986` and `.999`. Several false validation candidates also score higher.
A high-confidence cutoff therefore cannot isolate the useful addition. Requiring both serve heads
to agree does not help either: the useful candidate's local v4/v5 serve maxima are `.875/.915`,
while a false test candidate is `.992/.986`.

The implemented validation sweep includes:

- v5 composed-confidence lower bounds;
- lower bounds on the minimum v4/v5 raw serve maximum within one second;
- lower bounds on `v5 live raw max - v4 live raw max`.

Validation selects none because its eight uncovered candidates contain zero novel strict matches.
Retrospectively, live-score uplift above `.25` happens to isolate the one useful fault and would move
test predictions/matches from `59/31` to `60/32`, or precision/recall/F1
`.5333/.8205/.6465`. That threshold is an exploratory hypothesis only: validation has no positive
addition with which to validate it, and choosing it from test behavior would be leakage.

The score semantics also require care. These class-weighted sigmoid outputs are uncalibrated ranking
scores. A composed interval confidence can mean rally-run mean, serve peak, or the maximum of merged
sources. The report therefore preserves the separate raw rally/serve evidence for each uncovered
candidate instead of treating the merged value as a probability.

## Decision

Do not deploy v4+v5 fusion yet. Preserve v4 as the cutting reference. Keep the selected boundary
rule as a promising validation-frozen research candidate, and keep uncovered additions disabled
until a fresh validation source includes true and false additions. On new data, test a learned
overlap-component selector using explicit provenance, both rally heads, both serve heads, and the
dead-ball/end score. Do not select thresholds from this repeatedly inspected single indoor test.

## Reproduction and immutable artifacts

```bash
ROOT=/mnt/freenas/volleycut/labeling-v1-2026-08-09

PYTHONPATH=. .venv/bin/python -m analysis evaluate-dual-serve-fusion \
  --manifest $ROOT/manifests/full-gold-v1.json \
  --v4-rally-model $ROOT/models/full-audiovisual-v2-final \
  --v4-serve-model $ROOT/models/serve-specialist-audiovisual-v4 \
  --v4-cache-dir $ROOT/features/audiovisual-v2 \
  --v5-rally-model $ROOT/models/full-audiovisual-audio-normalized-v3 \
  --v5-serve-model $ROOT/models/serve-specialist-audio-normalized-v5 \
  --v5-cache-dir $ROOT/features/audiovisual-audio-normalized-v3 \
  --split validation \
  --output $ROOT/reports/dual-serve-v4-v5-fusion-v1-validation.json

PYTHONPATH=. .venv/bin/python -m analysis evaluate-dual-serve-fusion \
  --manifest $ROOT/manifests/full-gold-v1.json \
  --v4-rally-model $ROOT/models/full-audiovisual-v2-final \
  --v4-serve-model $ROOT/models/serve-specialist-audiovisual-v4 \
  --v4-cache-dir $ROOT/features/audiovisual-v2 \
  --v5-rally-model $ROOT/models/full-audiovisual-audio-normalized-v3 \
  --v5-serve-model $ROOT/models/serve-specialist-audio-normalized-v5 \
  --v5-cache-dir $ROOT/features/audiovisual-audio-normalized-v3 \
  --split test --retrospective \
  --decision $ROOT/reports/dual-serve-v4-v5-fusion-v1-validation.json \
  --output $ROOT/reports/dual-serve-v4-v5-fusion-v1-retrospective-test.json
```

- Semantic manifest digest: `67cfcf46a94dad05a38aa1fdb146059cc21cff8a906cf08abc4989f4980d20db`
- v4 rally artifact: `e08a3db18e4fd81fc6f5c10816b7c17f4c7661d4424a98d800aa0b1b0ccfd6a0`
- v4 serve artifact: `e28af5f43e78f9708676c046cc5636d1a7276a0e176042ea40eada49aa0a36e7`
- v5 rally artifact: `6ed21cff27eef53990f72ab26ebac17d910d3fa41d50e7818359f2e8e09f234f`
- v5 serve artifact: `951689b0f1146aba02f4074affaf95a355f162baa81e222f7f811de108992979`
- Validation report: `79d9a626a95b3790f8a8dbdf9443dc7ff65f97219fd9a749bdb8465c2a634f66`
- Retrospective-test report: `e4e36f156bb8156b823f7e957ce6b46d6e208e4ab80459db414d3aec39106bce`

The test report embeds the exact validation-report SHA and rejects a changed manifest or any changed
component model hash. Validation was already reused for both underlying pairs; the test has been
opened repeatedly. These results are engineering diagnostics, not an unbiased generalization claim.
