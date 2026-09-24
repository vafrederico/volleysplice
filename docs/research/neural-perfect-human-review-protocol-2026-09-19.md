# Perfect-human review of neural and production rally decisions

The user asks how model combinations and individual neural models perform when flagged sections are reviewed by a human who always makes the correct keep/remove choice. This study reports two distinct assumptions: whole-candidate binary decisions, and perfect boundary editing within flagged sections. It does not fabricate observed human accuracy, time spent interacting with controls, calibrated confidence, or a newly trained review head.

## Fixed scope and measurements

Reuse the audited production-combinations input: eight exact-label grass/indoor recordings, four source groups, 322 rallies, seeds 3407/1729/20260918. Beach and the protected test remain excluded. No fitting, new video extraction, deployment, or alteration of prior registered artifacts. Target product padding remains symmetric two seconds with strict less-than-three-second joining; report all 0/1/2/3-second cases. Ignore excluded intervals in every decision, count and duration. Pool recording duration counts within a seed, then average three seed metrics and recording-total seconds; do not count seed replicas as additional videos.

Primary ranking remains F1_padP_coreR at the declared two-second product padding. Report P_pad, R_core, F1_padP_coreR, model/human export duration and difference, correctly removed unwanted time, incorrectly omitted human export, incorrect export, missed actual play, and complete/partial rally losses. Binary decisions retain raw-event F1 and the usual canonical guardrails. The exact-boundary export diagnostic has no invented raw-event matching result.

## Label-blind candidate and flag construction

Models: compact short boost, compact keep rescue, DINO global control, DINO short boost and DINO keep rescue. Production anchors: current checked-in aggressive-suppression default, shipped unsuppressed union, and source-group-held refit unsuppressed union. Preserve their differing exposure limitations. Input probabilities are the selected outer-held-fold live-head scores on the exact original frame-quantized feature times, not recreated nominal timestamps. They are not assumed calibrated.

All review candidate inventories and flags are constructed without human rally labels and remain identical across product padding cases. Merge overlapping/touching raw intervals, subtract ignored spans, and treat each resulting connected interval as a separate decision. Candidate IDs are deterministic start/end-ordered ordinals. Count candidates, visible playback clips, and distinct true rallies separately.

Six fixed production/neural policies per anchor and neural model:

1. `guarded_trim`: use the existing protected-component neural trimming rule only as a trigger. Flag a whole production candidate if any of its raw time would be trimmed. Protection restricts flags, not what a human can inspect within the containing candidate. Production candidates already suppressed are not silently restored.
2. `suppression_zero`: flag a production candidate having no positive overlap with neural detections expanded two seconds each side.
3. `suppression_half`: flag when less than half of a production candidate has that tolerant neural support.
4. `suppression_any`: flag when any production-candidate time lacks that tolerant support.
5. `bidirectional_half`: build candidates from the raw production/neural union; flag a containing candidate if either a production component has less than half tolerant neural support or a neural component has less than half tolerant production support. This can propose missed rallies.
6. `bidirectional_any`: same, using any unsupported raw time as trigger.

Tolerant support clips and merges overlaps only, with no short-gap joining. Zero/any support is decided by interval emptiness, not rounded fractions; half means strictly twice supported duration below component duration. Refit protection uses its own old/v2 components. This is 90 policies, each evaluated at three matched seeds.

For a neural model operating individually, candidates are its positive raw components plus all remaining valid-time gaps split at an absolute five-second grid. These negative sections allow review of potential missed rallies. The candidate universe covers evaluable time, but only flagged candidates are reviewed. For positive candidates use the arithmetic mean live score at feature times in [start,end); for negative candidates use the maximum live score. A section containing no score tick is always flagged by applicable uncertainty rules, not assigned an invented score. Six fixed policies per model:

- `positive_uncertain`: review positive candidates whose mean score is below 0.8; no negative proposals.
- `uncertain_narrow`: positives below 0.7 and negative sections above 0.3.
- `uncertain_medium`: positives below 0.8 and negative sections above 0.2.
- `uncertain_wide`: positives below 0.9 and negative sections above 0.1.
- `all_positive`: review every predicted positive candidate, with no negative search.
- `all_candidates`: review every positive candidate and every negative section; a full-review reference, not a practical recommendation.

These are 30 individual-model policies across three seeds. Thresholds are fixed before reading new review outcomes; all thresholds are reported, with no gold-tuned choice or refit. Total new policy matrix: 120 policies, 360 seed cells. The prior study's 30 export-disagreement policies are also retained as a separately named, finer-grained reference, augmented with counts of distinct labeled rallies touched.

## Two human assumptions

**Binary whole-candidate decisions (main keep/remove interpretation):** preserve automatic raw output outside flagged candidates. Keep an entire flagged candidate if it has any positive overlap with evaluable human rally core; otherwise drop it. This explicitly conservative definition protects any real play in a mixed candidate but cannot repair false tails, exact event boundaries or omitted portions outside the candidate universe. Newly accepted negative/bidirectional candidates can add play. Apply the ordinary product padding/joining/ignored exclusion to the resulting raw decisions. This is a correct binary classifier under the stated definition, not a promise that every kept frame is correct. Identify mixed candidates containing both true core and other time.

**Perfect boundary editing (separate optimistic assumption):** let M be the baseline export and W the final padded/joined export of flagged candidates. Correct only W: `(M minus W) union (paddedHuman intersection W)`. Do not pad or join the edited export again. This allows cutting mixed candidates and recovering only human time inside the flagged section. It cannot correct mistakes outside W. It is not measured human performance; because binary changes are applied before export joining, do not assume these two workflows have a strict numerical ordering in every case.

For either workflow, playback is W with an additional two seconds of context each side and strict less-than-three-second joins, clipped and with ignored time removed. This context is workload only and grants no extra correction permission. Report playback union minutes, fraction of evaluable footage, number of playback clips, positive/negative candidate decisions, distinct gold rally IDs intersecting raw flagged candidates, and distinct gold IDs merely present in playback. Deduplicate rally IDs within each recording, then sum across recordings. Gold-derived counts are post-hoc workload descriptions, never used for flags. Report raw decision time and padded decision/export time separately. Viewing duration is footage at 1x, not measured wall-clock reviewer labor.

## Verification and interpretation

Freeze input/source/recipe/protocol hashes before executing new quality comparisons. Independently reconstruct flags/candidates, binary selections, edit permissions, interval accounting and distinct rally counts. Verify all four padding cases, ignored splits, exact joins, no-score sections, false-positive-only candidates, mixed candidates and uncertain-negative recovery. Reuse prior standalone scores only after hash-bound same-scope replay. Preserve full per-seed, per-recording and source-group evidence.

Report F1 versus review workload for all fixed policies. A highest development score is descriptive, not an unbiased chosen-policy estimate. Shipped production exposure, development-selected neural candidates and uncalibrated score thresholds remain limitations. A dedicated learned needs-review head or budget-tuned policy would need separate training/validation; this study evaluates practical flag rules from existing outputs.

Execution checklist (Tasks tool unavailable):

- [x] Verify and normalize held-fold live probabilities and prior inputs.
- [x] Implement and test candidate, flag and human-decision semantics.
- [x] Freeze the complete policy and input/source manifest.
- [x] Execute all fixed policies and independent reconstruction/accounting checks.
- [x] Summarize binary versus boundary-editing results and review workload.
- [x] Archive reproducible artifacts and leave production unchanged.

Registration SHA-256: `7f47932b9a18e030840449b7e4eceec489ac474d8eb5fa9dee13ee0a32a71b56`.
The NAS `protocol-initial.md` preserves the prospective text; this working document tracks completion without changing that snapshot.
