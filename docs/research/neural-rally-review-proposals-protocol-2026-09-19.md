# Rally-preserving review proposals: prospective development experiment

The user authorized executing improved review selection/boundaries and clarified that score tracking needs proper rally separation for serve detection, even when exporting footage can use joined regions. This experiment changes review proposals and queue ordering; it does not retrain rally detectors, train a learned review head, modify production, or evaluate score/team/winner prediction.

## Scope and primary contract

Reuse the same eight exact-label grass/indoor recordings, four source groups, 322 labeled rallies, and three seeds (3407, 1729, 20260918). Beach and the protected test remain excluded. Models are compact short boost, DINO global, and DINO short boost. Each is evaluated as a challenger to the checked-in production default and individually. The held-group four-head probabilities and original neural event lists are normalized separately; current production event identities are its actual ensemble output, not reconstructed hidden rallies. Earlier studies and their registered sources remain immutable.

Primary ranking is pooled `F1_padP_coreR` at symmetric two-second product padding, with the canonical strict positive-gap <3-second join. Report all four symmetric 0/1/2/3-second cases, pooled precision, core recall, F1, model/human export durations, duration difference, incorrect exported time, correctly removed unwanted time, incorrectly removed human export, and missed core. Ignore excluded intervals consistently, without rejoining after subtraction. Pool recordings within each seed before averaging seed metrics; seed replicas are not additional videos.

Rally identity is a separate output, never derived from padded/export unions. A duration-F1 gain is not sufficient to recommend a score-tracking workflow if rally separation or observed serve-start localization worsens. Report event one-to-one IoU>=0.5 precision/recall/F1, complete misses, merge/split errors, matched boundary errors, and independent start/end timestamp localization at 0.25/0.5/1/2-second tolerances. Headline serve proxy is observed-start localization at one second: synthetic review-window edge starts without an actual model or reviewed gold start marker are excluded from predicted observed contacts. Also report all-boundary localization and the censored counts, so artificial clipping cannot masquerade as observed serves. The exact labels declare serve-contact-to-dead-ball; sparse separate serve markers are not an independent complete serve-contact benchmark. These metrics do not measure serving side or score reconstruction.

## Fixed comparison matrix

Three candidate inventories × two queue orders × two baseline modes × three neural models × three seeds = 108 configuration/seed runs. Each run has four nested per-recording review budgets: 5%, 10%, 20%, 40% of evaluable video, giving 432 outcome cells. Automatic baselines are also reported. No hyperparameter or rule is chosen using outcome scores during this run.

Candidate inventories:

1. `legacy`: the previous whole-component `bidirectional_any` flag for production combinations; previous `uncertain_medium` positive/five-second-negative flag for an individual model. This controls for the old inventory under the new common event-aware human and budget semantics; these are not identical to the previous unbudgeted export-edit oracle results.
2. `local_events`: retain raw production-only and neural-only spans; explicitly flag gaps/boundaries inside one-to-many, many-to-one or many-to-many positive-overlap graph components before tolerant support can erase them. Individual models flag existing positives with mean live below 0.8. Both modes add event-shaped missed-play proposals from sustained live evidence instead of fixed negative grid windows.
3. `local_heads`: the second inventory plus start/end-head-guided proposal boundaries and internal-start review bands. This isolates the additional value/cost of already trained boundary heads.

All flags and ranking features are label-blind. Exact candidate rules are frozen with source hashes. Low-live components use actual-time midpoint cells, live>=0.15 with a peak>=0.30 and at least 0.5 seconds. A missed proposal needs >=0.5 seconds and >=25% of its duration outside the baseline. Head snapping searches ±3 seconds for a head peak>=0.25; existing endpoint disagreements require >=0.25-second displacement. Internal starts require serve-head>=0.35, at least two seconds from event edges, one-second peak suppression, and preceding three-second end evidence>=0.25 or live valley<0.35. Ignored ticks are excluded using half-open intervals. Proposal windows shorter than one second are centered/expanded to one second then clipped/split around ignored spans. Exact duplicate windows combine their reasons; other overlaps remain distinct proposals.

High-live qualification is repeated separately after splitting around ignored spans, including narrow ignored spans containing no tick center. Boundary peak and preceding-context searches stay inside the same valid-time component as their center. Ignored gaps therefore cannot supply evidence or connect candidate evidence across an excluded barrier.

Queue orders:

- `chronological`: start, end, deterministic ID.
- `evidence`: descending fixed evidence per standalone playback second, then start/end/ID. Evidence uses mean/max live and max start/end scores according to proposal reason; event-related proposals receive a fixed four-second-equivalent term. This is a transparent heuristic, not a learned or calibrated error probability. Source code fixes the complete formula.

For each recording, process budgets ascending, retain all selections from smaller budgets, scan the fixed order and accept a proposal if the union playback cost stays within the current budget (1e-9 numeric tolerance). Continue after oversized proposals; context overlap is counted only once. This guarantees nested selected IDs, but need not exhaust the budget. It is not an oracle knapsack or label-based selection. Differences in unused budget are reported; compare realized workload as well as nominal caps.

## Viewing, edit permission, and event correction

A proposal is a raw review range, not necessarily a detected rally. Expand selected proposals by two seconds, union overlaps/touching and subtract ignored spans to obtain raw edit permission W. Do not join positive gaps in W. Playback adds two more seconds of context and the canonical strict <3-second join, then removes ignored spans. Playback does not grant edit permission. Workload is footage at 1x, not observed reviewer labor. Report proposed decisions, merged edit regions, playback clips, playback minutes, fraction of footage, distinct true rallies intersecting W, and those only visible in playback.

The common simulated human corrects raw occupancy and event boundaries only inside W. Raw occupancy becomes `(baselineRawUnion minus W) union (goldRawUnion intersection W)`. Preserve original model start/end markers outside W; substitute gold start/end markers only inside W (closed endpoint membership). Split resulting continuous occupancy at these markers, including touching rally boundaries. Original base occupancy inside ignored time is untouched; evaluation/export masks it separately. Events clipped at an edit-window edge do not receive unseen true endpoints. Report censored start/end and unresolved boundary counts; mark each output start/end as observed only when backed by a retained original marker or a gold marker inside W. Then compute canonical exports from this event list for all padding settings. Padding/joining can change time outside W; raw occupancy and original raw markers outside W must remain unchanged.

This is a perfect localized editor assumption, not binary keep/remove and not observed human performance. It can remove a gap between true rallies while preserving two event identities even when padding makes the exported footage continuous. It can also improve timing without changing export duration. Existing raw event lists must have nonoverlapping interiors; touching events stay distinct. Event metrics censor ignored-time gold identities per the independent metric contract, preserving prediction identity through masking. Partial edge events mean event F1 need not improve monotonically with additional review; no monotonic event-quality claim is assumed.

## Verification and interpretation

Execution amendment before any result publication: the initial registered run stopped at an independent playback-count adapter that supplied a gold interval mapping where the auditor expected a pair. No result files were produced and no outcome tables were read. The initial registration, qualified source bytes and failure closure are preserved in the original study directory. The v2 run changes only that adapter, the output directory, and adds a synthetic end-to-end runner test; the candidate rules, matrix, budgets, human semantics and metrics are unchanged.

Freeze inputs, protocol, implementation and independent audits before running outcomes. Verify label-blind proposal invariance, split/merge and zero-gap cases, actual-time cells, ignored-time exclusion, nested budgets, context cost, bounded edits and original markers outside permission. Independently reconstruct candidates, queue selection and event correction; independently audit all four export accounting cases. Event matching receives exhaustive small-matrix qualification and synthetic merge/split/serve-edge checks.

Retain every fixed arm and source-group/recording result. Rank at equal declared budget and primary padding, inspect realized playback and event/serve guardrails. Production weights historically saw these recordings; decoder/model choices are development-exposed even though neural predictions use held-source fits. No selected configuration is a fresh protected-test deployment estimate. A learned review scorer is a possible later experiment, not silently included here.

A descriptive separation screen requires mean event F1, observed-start one-second F1 and recall not to decrease, and complete missed rallies not to increase, against the same automatic baseline. This is a predeclared non-worsening screen, not proof of score-tracking correctness. All individual metrics, seed ranges and material/any-overlap merge/split diagnostics remain visible.

Execution checklist (Tasks tool unavailable):

- [x] Read repository/vault instructions and identify prior immutable artifacts.
- [x] Verify original event identities and normalize all four held-source neural heads.
- [x] Implement and qualify proposals, queues, event correction and event/serve metrics.
- [x] Freeze the prospective protocol, inputs, sources and complete fixed matrix.
- [x] Execute all arms and independent audits without outcome-driven changes.
- [x] Summarize export quality, rally separation, observed serve-start proxy and review workload.
- [x] Preserve reproducible artifacts; leave production unchanged.
