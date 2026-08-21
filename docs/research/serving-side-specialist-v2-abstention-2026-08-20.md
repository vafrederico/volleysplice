# Serving-side v2 abstention experiment

Date: 2026-08-20

## Question

Can separate FAR and NEAR probability thresholds improve serving-side precision by
abstaining on uncertain rows, without changing the frozen v2 model?

## Evaluation contract

- Scope: 1,090 development rows only.
- Scores: leave-one-source-group-out cross-fit predictions from the already-selected
  `court-flow-recording-rank` logistic candidate.
- Reproduction check: the scores must match the prediction digest recorded in the
  frozen model (`3b35b43615c72b995ab93d5a40b22c13906b9984ddb6e6857450afe78b6a0bc0`).
- Threshold rule: emit FAR below the lower threshold, emit NEAR at or above the
  upper threshold, and abstain in between.
- Search constraint: the lower and upper thresholds must surround the frozen
  `0.5408551369` cutoff. Existing side decisions may only become abstentions; the
  experiment cannot flip one emitted side to the other.
- Selection: maximize coverage while both pooled class precisions meet the target.
  At least 50 predictions of each side are required.
- Recall denominators include abstained human rows.
- The protected test dataset and evaluation were not loaded or used.

## Results

| Pooled precision floor | FAR threshold | NEAR threshold | Coverage | NEAR precision | FAR precision | NEAR recall | FAR recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.5409 | 0.5409 | 100.0% | 91.3% | 83.9% | 82.4% | 92.1% |
| 90.0% | 0.2728 | 0.5409 | 84.2% | 91.3% | 90.1% | 82.4% | 70.3% |
| 92.5% | 0.1377 | 0.6335 | 65.2% | 92.7% | 92.7% | 78.4% | 42.4% |
| 95.0% | 0.0771 | 0.6897 | 50.5% | 95.2% | 95.6% | 75.9% | 20.1% |
| 97.5% | 0.0525 | 0.9695 | 15.5% | 98.1% | 98.4% | 19.2% | 11.2% |

The 95% pooled point emits 550 of 1,090 rows and abstains on 540. Its selective
accuracy is 95.3%, but it retains only 109 of 543 human FAR rows. Precision is also
not stable by source group: at that pooled point, `kb-kob-20250614` has 50.0% FAR
precision and `ymca-kob-rev4s-20260312` has 76.2% NEAR precision.

## Decision

Do not deploy the 95% policy. Threshold abstention alone cannot provide both high
precision and useful symmetric recall with the current feature representation.

The 90% pooled point is a plausible review-mode candidate when 84.2% automatic
coverage is acceptable: it improves the weaker FAR precision from 83.9% to 90.1%
without changing NEAR predictions. It is not yet a production threshold because
source-group precision remains uneven.

The next durable precision work should improve the evidence itself: exclude
not-serves before side classification, add calibrated court-relative server/player
location features, and use the corrected high-confidence errors as hard examples.

## Artifacts

- NAS evaluation:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-specialist-v2-development-abstention.json`
- Evaluation SHA-256:
  `a7ace09d1b4c4ac7914c247cb3b990f91af5b4b35bd121f3f04556bff13ffc2e`
- Re-run: `npm run evaluate:serving-side-v2-abstention`
