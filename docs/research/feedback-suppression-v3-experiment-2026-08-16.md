# Feedback suppression v3 experiment (2026-08-16)

## Outcome

The suppression specialist removes substantial false-positive export time and raises
aggregate `F1_padP_coreR`, but it is not recall-safe enough to deploy as a veto on the
current production ensemble. At the declared product padding of 2 seconds, current
production plus suppression:

- improved held-feedback `F1_padP_coreR` from **0.8965 to 0.9267**;
- raised held-feedback `P_pad` from **0.8124 to 0.8686**;
- retained **0.9932** `R_core`, down from 1.0000;
- removed **375.9 seconds** of padded export from the six held-feedback files; and
- improved protected-test `F1_padP_coreR` from **0.8246 to 0.8547**.

The subsequent recall-first audit reverses the earlier follow-up recommendation.
Across suppression entry thresholds from `0.75` through `0.999`, no setting both
preserved the current production baseline's unpadded core recall and canonical
`R_core` while improving `P_pad` on the development scope. Even `0.999` reduced
development core recall from `0.93814` to `0.93327` and `R_core` from `0.99873` to
`0.99822` for only a small `P_pad` increase from `0.61162` to `0.61850`.

Keep current production unchanged. Do not promote either v3 model or deploy the
current suppression veto. With only binary component intervals, protecting everything
detected by either production component leaves no vetoable output. A useful follow-up
must use stronger preserve evidence—raw rally confidence, serve/rally anchors, and
connected-rally continuity—so it can trim weakly supported fringe without deleting an
entire rally.

## Recall-safety and missed-rally audit

At the current suppression decoder (`enter=0.75`, `exit=0.65`, 1-second smoothing,
0.5-second minimum suppression duration, and 0.5-second bridge), the 2-second padded
output still leaves evaluable core time uncovered in 148 reviewed rallies. Of these,
53 are completely missed after padding, merging, and the canonical strictly-less-than
3-second short-gap join; all 53 complete misses were introduced by suppression.

The production-side component attribution for those 53 complete misses is:

- 44 were inferred by both the previous-production and all-labels-v2 models;
- 5 were inferred only by the previous-production model;
- 4 were inferred only by the all-labels-v2 model; and
- 0 were inferred by neither model.

The canonical short-gap join does not rescue any of the 53 complete misses. It helps
only 2 of the 95 partially missed rallies, recovering 2.5 seconds of core coverage in
total, and makes no listed rally fully covered. The complete per-rally list includes
file, range, duration, production-component attribution, each component's overlap,
and before/after-join coverage in the HTML report and the JSON/CSV audit artifacts.

On the eight grass recordings, suppression lowers pooled `R_core` from `0.99590` to
`0.90000`, a loss of 191.5 seconds of covered rally core across 72 affected rallies
(32 complete and 40 partial misses). Of these, 66 rallies and 175.8 lost seconds were
produced by both production components. Previous production alone accounts for 4
rallies and 10.7 seconds; all-labels v2 alone accounts for 2 rallies and 5.0 seconds.

For correctly suppressed false-positive regions, the previous-production model is
the largest source. Using the strict product-safe definition—removed time outside the
2-second padded human export after canonical gap joining—1783.5 seconds were removed:
1056.0 seconds (59.2%) came from previous production only, 444.5 seconds (24.9%) from
both models, and 283.0 seconds (15.9%) from all-labels v2 only. Under the looser raw
non-rally definition, the split is previous-only 1267.8 seconds (53.5%), both 656.5
seconds (27.7%), and v2-only 445.8 seconds (18.8%).

Applying suppression only to previous production, then unioning the unchanged
all-labels-v2 output, is much safer but still not lossless. After 2-second padding and
canonical gap joining it completely misses 4 rallies and partially reduces coverage
on 16 more, losing 31.3 seconds of core coverage in total. Across all evaluable videos,
`P_pad` moves from `0.70078` to `0.75066`, `R_core` from `0.99706` to `0.99388`, and
`F1_padP_coreR` from `0.82307` to `0.85531`. On grass, 10 rallies are affected, 3 are
complete misses, and 15.3 core seconds are lost; grass `R_core` is `0.98826` versus
`0.99590` for production.

The mirror composition—unchanged previous production unioned with suppressed
all-labels v2—completely misses 3 rallies and partially reduces 8 more, losing 29.3
core seconds. Its pooled `P_pad / R_core / F1_padP_coreR` is
`0.71558 / 0.99409 / 0.83215`. Five grass rallies are partially affected for 14.8
lost core seconds, but none is completely missed; grass `R_core` is `0.98846`.
Compared with suppressing previous production only, suppressing v2 only is marginally
safer on pooled recall but provides much less precision and F1 improvement.

## Experiment design

Target product padding was declared as `(before, after) = (2, 2)` seconds. The four
required symmetric padding cases `(0,0)`, `(1,1)`, `(2,2)`, and `(3,3)` are in the
machine-readable and full Markdown reports. All metrics pool duration numerators and
denominators. Model and human ranges receive identical padding, are clipped and
merged, then join positive gaps strictly below 3 seconds. Ignored intervals are
subtracted without rejoining across them.

The feedback store contained 11 files, so an exact half was impossible. A recorded
PCG64 split with seed `20260816` used 5 files (45.5%) for fitting and 6 files (54.5%)
for held-out evaluation. Feedback features were decoded directly from each feedback
JSON bundle; no feedback video feature extraction was performed.

The candidates were:

1. Current production: previous production plus all-labels v2.
2. Current production plus the suppression specialist.
3. v3 candidate 1: the same three heads as v2, refit on all-labels-v2 labels plus the
   five feedback fit files and their reviewed negatives.
4. v3 candidate 2: candidate 1 plus the suppression head.
5. Previous production plus candidate 1.
6. Previous production plus candidate 2.

The suppression positive class is valid non-rally time selected by current
production, plus explicitly reviewed false-positive, hard-negative, and side-switch
samples. Human rally samples form its negative class. Thus the head learns to veto
ball retrieval/toss setup, player transitions, and side switches that were being
exported as play.

Suppression decoders were selected only on `grass-source-06`,
`grass-source-08`, and `indoor-source-03`. The protected test recording
and all six held-feedback files were closed until final evaluation.

## Target-padding results

The table reports `(P_pad / R_core / F1_padP_coreR)` at 2-second symmetric padding.

| Scope | Current | Current + suppression | v3 three-head | v3 four-head | Old + v3 three-head | Old + v3 four-head |
|---|---:|---:|---:|---:|---:|---:|
| All evaluable | .7008 / .9971 / .8231 | **.7950 / .9662 / .8723** | .7526 / .9731 / .8488 | .8521 / .8964 / .8737 | .6728 / .9959 / .8031 | .7062 / .9891 / .8240 |
| Training dataset | .6667 / .9966 / .7989 | **.7702 / .9350 / .8446** | .7316 / .9803 / .8378 | .8490 / .8380 / .8435 | .6570 / .9960 / .7917 | .6957 / .9867 / .8160 |
| Evaluation + validation + test only | .5754 / .9979 / .7299 | **.6934 / .9518 / .8023** | .6797 / .9275 / .7845 | .7819 / .8088 / .7951 | .5625 / .9943 / .7185 | .5881 / .9914 / .7383 |
| All export feedback | .8012 / .9971 / .8884 | **.8601 / .9926 / .9216** | .8028 / .9838 / .8841 | .8789 / .9664 / .9206 | .7447 / .9963 / .8523 | .7774 / .9900 / .8709 |
| Feedback fit files | .7892 / .9939 / .8798 | .8511 / .9918 / .9161 | .7999 / .9872 / .8837 | **.8760 / .9725 / .9217** | .7450 / .9963 / .8525 | .7733 / .9869 / .8671 |
| Feedback held-out files | .8124 / 1.0000 / .8965 | **.8686 / .9932 / .9267** | .8054 / .9807 / .8844 | .8816 / .9608 / .9195 | .7443 / .9962 / .8521 | .7812 / .9928 / .8744 |
| Grass | .5921 / .9959 / .7427 | **.6946 / .9000 / .7841** | .6393 / .9653 / .7691 | .7640 / .7664 / .7652 | .5758 / .9956 / .7296 | .6219 / .9825 / .7617 |
| Indoor | .7300 / .9980 / .8432 | .8308 / .9794 / .8990 | .8003 / .9903 / .8852 | **.8937 / .9166 / .9050** | .7178 / .9968 / .8346 | .7490 / .9929 / .8539 |
| Beach | .4784 / .9971 / .6465 | .5959 / .9116 / .7206 | **.6907 / .8216 / .7505** | .7565 / .6455 / .6966 | .4848 / .9881 / .6504 | .4845 / .9881 / .6502 |
| Feedback environment unknown | .8012 / .9971 / .8884 | **.8601 / .9926 / .9216** | .8028 / .9838 / .8841 | .8789 / .9664 / .9206 | .7447 / .9963 / .8523 | .7774 / .9900 / .8709 |

Feedback is reported as environment `unknown` because that field is not present in
the feedback schema. It was not inferred from filenames or capture appearance.

## Core and padded metric movement

On the six held-feedback files, current production plus suppression changed:

- core precision: **0.8698 → 0.9189**;
- core recall: **1.0000 → 0.9556**;
- core F1: **0.9304 → 0.9369**;
- padded precision: **0.8124 → 0.8686**;
- padded recall: **1.0000 → 0.9731**; and
- padded F1: **0.8965 → 0.9179**.

Across the combined evaluation/validation/test-only scope, the same composition
changed core P/R/F1 from `0.4749 / 0.9462 / 0.6324` to
`0.6310 / 0.7981 / 0.7048`, and padded P/R/F1 from
`0.5754 / 0.9813 / 0.7254` to `0.6934 / 0.8833 / 0.7769`.
This is a material precision win but also a meaningful recall cost, especially on
grass. Manual review should focus on the removed grass spans before deployment.

## Single-model-only veto visual reviews

Two definitions of one-model-only support are now available. Under the original
pointwise rule, only the exact seconds of cross-model overlap are protected. It
affects 31 reviewed rallies across 16 videos (7 complete and 24 partial misses),
loses 60.6 seconds of core coverage, and saves 2153.9 seconds of padded export.
Core P/R/F1 moves from `0.6783 / 0.9746 / 0.7999` to
`0.7650 / 0.9375 / 0.8426`; padded P/R/F1 moves from
`0.7008 / 0.9889 / 0.8203` to `0.7691 / 0.9690 / 0.8576`.

Under the new any-overlap rule, an entire overlap-connected union is protected when
either component overlaps the other anywhere in that span, including all
non-overlapping leading heads and trailing tails. For example, previous `15–20` plus
v2 `10–18` protects `10–20`: both the `10–15` head and `18–20` tail. This affects 10 reviewed rallies across
7 videos (5 complete and 5 partial misses), loses 23.9 seconds of core coverage, and
saves 1162.5 seconds of padded export. Core P/R/F1 becomes
`0.7089 / 0.9714 / 0.8197`; padded P/R/F1 becomes
`0.7406 / 0.9846 / 0.8453`.

Both policies fully remove the same 135 production prediction ranges that have zero
overlap with the padded human export target, totaling 419.1 seconds of raw prediction
time. The additional pointwise export savings come from trimming partially protected
overlap-connected spans, which also accounts for its larger recall loss.

The interactive reviews order complete-miss videos first and display source video,
human label, previous-production, all-labels-v2, production union, veto, and post-veto
rails. They distinguish raw, 2-second padding, and joined-gap regions and
mute/texture ranges outside the affected context. Raw model and veto intervals include
their uncalibrated decoder scores; derived union, padding, and joined-gap ranges have
no standalone score.

## Artifacts

- Full report: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/report.md`
- Machine-readable report: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/report.json`
- Split policy: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/split-policy.json`
- Recall-constrained threshold sweep: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/production-suppression-threshold-recall-constraint.json`
- Missed-rally audit: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/production-plus-suppression-missed-rallies.json`
- Missed-rally CSV: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/production-plus-suppression-missed-rallies.csv`
- Correct-suppression attribution: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/production-plus-suppression-correct-attribution.json`
- Old-only suppression rally audit: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/old-only-suppression-rally-audit.json`
- Old-only suppression rally CSV: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/old-only-suppression-rally-audit.csv`
- V2-only suppression rally audit: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/v2-only-suppression-rally-audit.json`
- V2-only suppression rally CSV: `/mnt/freenas/volleycut/intake-2026-08-13/experiments/feedback-suppression-v3-2026-08-16/v2-only-suppression-rally-audit.csv`
- Interactive HTML report: `https://internal.example/reports/volleycut-feedback-suppression-v3-2026-08-16.html`
- Missed-rally visual review: `https://internal.example/suppression-review`
- Any-overlap full-span visual review: `https://internal.example/suppression-review/any-overlap`
- Candidate 1: `model-dfbb67c7c0c2`
- Candidate 2: `model-cacb15849ba8`
- Inference artifacts: 180 JSON files (6 variants × 30 videos)

All 19 historical/challenge and 11 feedback videos received inference. The two
challenge videos `grass-source-02` and `indoor-source-02` have no
reviewed label documents, so they are inference-only and excluded from metrics.
