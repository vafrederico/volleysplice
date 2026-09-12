# v3 production-ensemble support-count ablation

The earlier corrected-v3 comparison included `old + v3` for both v3 candidate
outputs. It did not include `v2 + v3` or `old + v2 + v3`. This follow-up adds
those combinations and applies the three retained suppression-agreement policies
to the three-model union.

Candidate 1 (`model-3a3738bffa6b`) is the unsuppressed v3 voter. Candidate 2
(`model-6dc36c67401a`) is candidate 1 after its own suppression head, so its
unions are diagnostic; it is not treated as another independent vote.

## Result

At the declared 2-second export padding, pooled across all 28 evaluable
recordings:

| Variant | `P_pad` | `R_core` | `F1_padP_coreR` | Export Δ vs current prod | Non-exempt complete / partial losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current production (`old + v2`) | `0.7008` | `0.9971` | `0.8231` | `0.0s` | `0 / 0` |
| `old + v3` candidate 1 | `0.6728` | `0.9959` | `0.8031` | `+740.9s` | `1 / 15` |
| `v2 + v3` candidate 1 | `0.7162` | `0.9920` | `0.8319` | `-712.5s` | `3 / 25` |
| `old + v2 + v3` candidate 1 | `0.6612` | `0.9992` | `0.7958` | `+1295.1s` | `0 / 0` |
| Triple, aggressive, suppress exactly 1-of-3 | `0.6804` | `0.9992` | `0.8095` | `+686.7s` | `0 / 0` |
| Triple, zero-miss, suppress exactly 1-of-3 | `0.6768` | `0.9992` | `0.8070` | `+798.1s` | `0 / 0` |
| Triple, raw, suppress 1-or-2-of-3 | `0.7068` | `0.9964` | `0.8270` | `-194.8s` | `4 / 7` |

The fixed three-model union can preserve recall by suppressing only components
supported by exactly one model, but its extra v3 predictions erase the product
benefit: the safe aggressive and zero-miss rows both export more time and have a
lower `F1_padP_coreR` than current production. Extending suppression eligibility
to two-model-supported components is where the non-exempt recall losses appear.
The raw 1-or-2-of-3 row is the only triple suppression row that beats current
production's aggregate `F1_padP_coreR` while saving export time, but it completely
loses four non-exempt rallies and partially loses seven.

This evidence supports the existing decision not to move forward with either v3
candidate. The product path remains current `old + v2`, the corrected external
suppression specialist, and a one-model-only eligibility gate.

## Rules

- `exactly 1`: suppression is eligible only on an agreement-connected component
  supported by exactly one of `old`, `v2`, or v3 candidate 1.
- `1 or 2`: suppression is eligible on anything short of unanimous 3-of-3
  support.
- `exactly 2`: reported separately to isolate the incremental risk of opening
  eligibility to two-model-supported components.
- Any positive-duration overlap connects the complete source intervals,
  including non-overlapping heads and tails. The selected agreement padding and
  strict `<0.5s` joining policy are then applied before counting source models.
- The corrected specialist and held decoder are fixed; nothing was tuned in this
  ablation.
- Selection/ranking uses only the three-recording development scope. Protected
  test, held feedback, and all-evaluable aggregates are post-selection disclosure
  and guardrails.
- Metrics subtract `ignoredIntervals`; final metric export uses symmetric
  2-second padding and joins only positive gaps strictly below 3 seconds.

## Artifacts

- Machine-readable report:
  `/mnt/freenas/volleycut/intake-2026-08-13/experiments/v3-production-ensemble-support-2026-08-18/report.json`
- Markdown report:
  `/mnt/freenas/volleycut/intake-2026-08-13/experiments/v3-production-ensemble-support-2026-08-18/report.md`
- Reproduction script:
  `scripts/evaluate-v3-ensemble-support.py`
