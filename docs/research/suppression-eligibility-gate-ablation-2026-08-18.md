# Suppression one-model eligibility gate ablation (2026-08-18)

## Outcome

Retain the **one-model-only production eligibility gate** for all three selected
suppression options.

With the held production decoder, removing the gate raises precision and
`F1_padP_coreR`, but it does so by deleting real rally coverage:

- development Core Recall falls from `0.9381` to `0.8106` and `R_core` from
  `0.9987` to `0.9434`;
- all-evaluable Core Recall falls from `0.9746` to `0.8590` and `R_core` from
  `0.9971` to `0.9461`; and
- it introduces 75 complete plus 101 partial non-exempt rally losses, totaling
  492.2 seconds of lost covered core.

Removing the gate also makes Raw connected, Aggressive intermediate, and Zero
non-exempt misses produce identical output. Their padding/joining rules exist only
to decide which one-model-only production components are eligible. The gate is thus
both a recall safeguard and the mechanism that makes the three aggressiveness
choices meaningfully different.

## Fixed experiment contract

- Production base: current previous-production union all-labels-v2 ensemble.
- Specialist: corrected overlap-safe suppression artifact
  `39eddf58163901930434ea422a802686ae921ae1e8fe23c20a3c12e5f453da93`.
- Decoder: 1.0-second smoothing, `0.75` enter, `0.65` exit, 0.5-second minimum
  live duration, 0.5-second bridge, 0.25-second short-event minimum, and `0.9`
  short-event threshold.
- Product padding: 2 seconds before and after; join only positive gaps strictly
  below 3 seconds.
- Ignored intervals are outside the evaluation universe and are never rejoined.
- Decision scope: the three predeclared development recordings. Protected test and
  held feedback were not used for threshold selection.
- `PXL_20260816_164327879.mp4` remains in pooled metrics but is excluded from the
  non-exempt rally-loss guardrail because its camera rotates away from the court.

The gated cut is `P intersect S intersect E_policy`, where `E_policy` is raw
production time in one-model-only agreement components. The ungated cut is simply
`P intersect S`.

## Target-padding comparison

All rows pool the 28 evaluable recordings at the declared 2-second padding.

| Variant | Core P | Core R | Core F1 | Padded P | Padded R | Padded F1 | P_pad | R_core | F1_padP_coreR | Export saved | Correct FPs | Non-exempt complete / partial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No suppression | .6783 | .9746 | .7999 | .7008 | .9889 | .8203 | .7008 | .9971 | .8231 | 0.0s | 0 | 0 / 0 |
| Raw connected, gated | .7067 | .9717 | .8183 | .7380 | .9851 | .8438 | .7380 | .9950 | .8474 | 1086.4s | 125 | 1 / 5 |
| Aggressive intermediate, gated | .6955 | .9735 | .8113 | .7272 | .9867 | .8373 | .7272 | .9960 | .8406 | 772.0s | 89 | 0 / 1 |
| Zero non-exempt misses, gated | .6902 | .9739 | .8079 | .7210 | .9875 | .8335 | .7210 | .9966 | .8367 | 590.0s | 71 | 0 / 0 |
| Gate removed, all production | .8220 | .8590 | .8401 | .8130 | .8984 | .8536 | .8130 | .9461 | .8745 | 4358.3s | 174 | 75 / 101 |

The ungated row wins the aggregate ranking metric, but fails the explicit recall
guardrail by a wide margin. It must not be promoted on F1 alone.

## Recall failure by environment

| Environment | Baseline Core R / R_core | Ungated Core R / R_core | Complete / partial losses | Lost core |
|---|---:|---:|---:|---:|
| Grass | .9469 / .9959 | .7377 / .9065 | 32 / 44 | 178.4s |
| Indoor | .9584 / .9980 | .8322 / .9725 | 12 / 34 | 62.8s |
| Beach | .9604 / .9971 | .3332 / .5130 | 31 / 20 | 250.1s |
| Export feedback | .9958 / .9971 | .9785 / .9950 | 0 / 3 | 0.9s |

## Stricter ungated threshold check

An ungated entry-threshold sweep from `0.75` through `0.9999` held smoothing,
minimum duration, bridging, and the 0.1 enter/exit separation fixed. Feasibility
required development Core Recall and `R_core` to match or exceed no suppression
while `P_pad` improved.

Only `0.9999` passed that development constraint. It saved just 5.1 seconds on the
three development recordings, versus 87.5 to 199.6 seconds for the gated options.
It also passed the protected-test and held-feedback guardrails, but failed the
post-selection all-evaluable safety audit:

| Scope | Baseline Core R | 0.9999 Core R | Baseline R_core | 0.9999 R_core | Complete / partial losses |
|---|---:|---:|---:|---:|---:|
| Development | .9381 | .9381 | .9987 | .9987 | 0 / 0 |
| Protected test | .9439 | .9439 | .9971 | .9971 | 0 / 0 |
| Held feedback | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 / 0 |
| All evaluable | .9746 | .9655 | .9971 | .9884 | 13 / 0 |

All 13 losses are complete rallies in
`beach-source-02.mp4`, totaling 85.4 seconds of core. The apparent
development-safe threshold therefore does not generalize and is not a viable
replacement for the gate.

## Required padding sensitivity

The following rows report pooled `P_pad / R_core / F1_padP_coreR` and model export
duration for all evaluable recordings. Human export durations for 0/1/2/3-second
padding are 9876.1 / 12077.6 / 14243.5 / 16381.1 seconds respectively.

| Variant | 0s | 1s | 2s | 3s |
|---|---:|---:|---:|---:|
| No suppression | .6678 / .9778 / .7936 · 14462.9s | .6818 / .9924 / .8083 · 17443.9s | .7008 / .9971 / .8231 · 20099.4s | .7245 / .9984 / .8397 · 22425.1s |
| Raw connected, gated | .6990 / .9743 / .8140 · 13766.7s | .7177 / .9893 / .8319 · 16502.4s | .7380 / .9950 / .8474 · 19013.0s | .7613 / .9966 / .8632 · 21252.5s |
| Aggressive intermediate, gated | .6840 / .9767 / .8046 · 14103.1s | .7050 / .9910 / .8239 · 16837.3s | .7272 / .9960 / .8406 · 19327.4s | .7524 / .9973 / .8577 · 21535.3s |
| Zero non-exempt misses, gated | .6791 / .9771 / .8013 · 14210.3s | .6976 / .9918 / .8191 · 17033.6s | .7210 / .9966 / .8367 · 19509.4s | .7470 / .9978 / .8544 · 21706.2s |
| Gate removed | .8116 / .8697 / .8397 · 10582.7s | .8070 / .9233 / .8613 · 13276.8s | .8130 / .9461 / .8745 · 15741.1s | .8261 / .9567 / .8866 · 17948.4s |

The machine-readable report contains the equivalent four padding cases for the
development scope and target-padding metrics for every environment, provenance,
feedback split, and protected-test scope.

## Product decision

No suppression-system option changes:

- Raw connected keeps its raw-connected one-model eligibility rule.
- Aggressive intermediate keeps 1.5-second agreement padding and joins strictly
  below 0.5 seconds before determining one-model-only eligibility.
- Zero non-exempt misses keeps 2-second agreement padding and joins strictly below
  0.5 seconds before determining one-model-only eligibility.

Do not add an ungated product option and do not replace the gate with the `0.9999`
threshold.

## Artifacts

- Reproducible evaluator:
  `scripts/evaluate-suppression-eligibility-gate.py`
- Full Markdown report:
  `/mnt/freenas/volleycut/intake-2026-08-13/experiments/suppression-eligibility-gate-ablation-2026-08-18/report.md`
- Machine-readable metrics and rally audit:
  `/mnt/freenas/volleycut/intake-2026-08-13/experiments/suppression-eligibility-gate-ablation-2026-08-18/report.json`
