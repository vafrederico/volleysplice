# Compact typed start and independent end boundaries

The user authorized testing start correction versus additional-rally detection with separate end boundaries. This experiment changes the proposed rally-event timeline while preserving the previous checked-in production export exactly. It does not alter production code or weights, train new models, or evaluate serving side, winners or scores.

## Scope, controls and prospective comparison

Reuse the previous immutable eight indoor/grass recordings, four source groups, 322 manually reviewed rally labels, compact short-boost predictions and three seeds (3407, 1729, 20260918). No beach or protected test. Same actual float64 timestamps, float32 live/start/end/keep scores and production parent IDs. Freeze normalized inputs, source hashes and rules before reading new outcome tables. This is another development comparison on previously exposed footage, not independent generalization evidence.

Each compact event is associated with an original production parent and valid-time component if their intersection lasts at least 0.5 seconds. Order these associated compact events chronologically within that component. The first is an `initial_start` candidate, subsequent ones are `additional_start` candidates. This typing uses only model predictions. If compact missed the true first rally, the inferred type can be wrong; report that error rather than hiding it.

Four fixed automatic policies:

1. `first_start`: use only the first associated compact start, retain the component's original end, and create no additional rallies.
2. `typed_starts`: use the first and additional compact starts, partitioning at successive starts and retaining the original final end. Internal partition ends are synthetic and cannot count as observed dead-ball markers.
3. `separate_ends`: retain each associated compact event's independent end, allowing dead time between event cores. Starts and ends are constrained to the original parent and same valid component.
4. `head_refined`: refine those independent pairs using existing heads. Process events chronologically. Select the strongest start-head tick >=0.25 within three seconds of the original clipped compact start, inside [max(component start, previous refined end), original compact end minus 0.5 seconds]. Then select the strongest end-head tick >=0.25 within three seconds of the original clipped end, inside [chosen start plus 0.5 seconds, min(component end, next original clipped start)]. Ties use nearest original endpoint then earliest time. Retain the original endpoint if no eligible peak exists. Domains preserve nonoverlap and minimum 0.5-second duration. Evidence never crosses ignored time.

Unsupported parents remain exactly their original event. Partially supported parents are processed by valid-time component, retaining a fallback component event where compact has no support. Ignored-edge fragments are censored and do not create observed serve/dead-ball claims. Clipping a neural boundary to a production/ignored edge is not an observed neural boundary; actual unchanged neural timestamps and eligible head ticks are observed. Retained production endpoints honor original marker flags. Every event keeps parent/component/candidate lineage.

Candidate strength is mean live score over the original associated compact interval's actual tick centers, zero if there are none. This is not a calibrated confidence. Actionable flags are an initial-start displacement of at least 0.25 seconds, every proposed additional start, and (for paired policies) an independent end at least 0.25 seconds before the component end. Report all candidate quality separately from actionable flag counts; unchanged candidates are not new corrections.

There are 12 automatic outcomes. Four policies, two fixed queue orders, four nested per-video budget caps (5/10/20/40%), three seeds and two human modes produce 192 reviewed outcomes. Retain every arm. No outcome-driven threshold, label, epoch or seed changes. Production is the common baseline; compare previous split-only findings descriptively without relabeling them as fresh controls.

## Review and human actions

A job is an original production parent with an actionable proposal. Multiple candidate/endpoint flags share one job. Playback is the union of full parent intervals plus two seconds of context, clipped, joined only for positive gaps strictly below three seconds, then ignored time removed without rejoining. Rank chronologically or by maximum flag strength per standalone playback second. Nested greedy selection retains prior selections at larger caps, skips oversized jobs and continues. No labels inform ranking. Report actual time, jobs, clips, boundary flags, candidate events and distinct real rallies touched.

`proposal_confirmation`: inside each selected parent, consider only model-proposed event candidates. A candidate is eligible for ideal human acceptance if its observed start is within one second of a true start in the same component and its proposed interval materially overlaps that true rally (>=min(0.5 seconds, 10% of gold duration)). Match one-to-one, maximum cardinality then smallest start error. The human can correct a mistaken initial/additional type. An accepted true first start replaces the fallback first event's start; accepted subsequent starts add events. Reject unmatched proposals and retain the baseline if none are accepted. If a parent has accepted candidates, operate within its valid components, preserving a fallback first event when the first true rally has no accepted candidate. Starts-only arms retain/partition the old ends. Paired arms correct the accepted event's end to the true end clipped to edit permission; an unseen outside endpoint is unobserved. Accepted ends and later starts remain separate. This restricted human cannot discover an unproposed rally. Correcting one proposed event's end can expose the omission of another unproposed event: report event-core loss and new event-timeline misses even though export coverage stays fixed.

`full_parent`: the explicitly more capable ideal human corrects every rally core inside a selected original parent, including ones without a model proposal. Replace selected parent event cores with individual gold intervals intersected with that parent's valid components; preserve identities at touching boundaries and mark clipped endpoints unobserved. A wholly false selected parent yields no rally records. Unselected parents are untouched. This is a full-parent review ceiling, not a model-generated improvement or observed reviewer result. It must retain all gold-core time originally covered by production. Neither mode imports a true boundary outside the selected parent or changes the export.

For proposal confirmation in paired policies, an accepted start authorizes correcting that rally's entire end within the reviewed parent even if the model end was inaccurate or censored. There is no additional end-nearness acceptance gate. This stronger human capability is distinct from automatic end accuracy, which is evaluated directly.

## Metrics and safeguards

The primary remains pooled `F1_padP_coreR` at symmetric two-second padding with strict positive gap <3-second joining. All arms must equal production exactly, so all tie on primary export ranking. Report symmetric 0/1/2/3-second sensitivity cases, P_pad, retained-play R_core, F1_padP_coreR, model/human export duration and difference, correctly removed unwanted time, incorrectly omitted wanted export, and incorrect export. Do not confuse fixed export recall with rally-event recall.

On the separate event timeline report one-to-one IoU>=0.5 rally precision/recall/F1, observed start AND end localization, merge/split errors, complete misses, additional misses, raw gold-core time lost/added versus production and raw core coverage. Starts use serve-contact labeling policy; these remain start/end proxies, not score accuracy. Automatic and restricted-human event cores may lose play even though exports do not; those losses are material guardrails. Full-parent ideal editing must not lose previously covered gold-core time.

Report candidate initial/additional typing and boundary quality independently: per-parent/component material-overlap gold identities define first and later targets; type labels use chronological gold order. Keep inaccessible/outside-start targets visible, report accessible denominators separately, exclude ignored-touched gold according to the prior identity contract, and never match across masks. Match observed candidate starts one-to-one at 0.5/1/2 seconds, with material interval overlap, reporting both strict type matching and untyped matching with wrong-type counts. Report end and joint start/end accuracy for those associated candidates. Counts are parent-specific and do not replace global rally recall. Parent-specific target duplication and unique true-rally counts remain explicit.

For reviewed outcomes, typed candidate metrics describe the original model candidates inside selected parents, before human correction. They do not describe snapped human boundaries. The separate event identity/start/end metrics describe the resulting human-edited event timeline. This prevents ideal human markers from being reported as neural proposal accuracy.

Pool within seed before averaging across seeds; show ranges and all source-group/recording diagnostics. Preserve every fixed result, including regressions. No production promotion based on the unchanged export F1. A useful automatic boundary policy must also improve event/start/end quality without unacceptable new event-core omissions. Perfect human assumptions and finite review budget remain explicit.

## Execution checklist

The Tasks tool is unavailable; this checklist tracks the work.

- [x] Read instructions and identify immutable inputs and prior findings.
- [x] Implement and qualify typed plans, metrics, both human modes and independent audits (88 tests passed; 96 real-input plans independently checked).
- [x] Freeze protocol, sources and inputs before outcome inspection (44 sources; contract `93697ad54ff3392e30cedb8282b77841736e23ab4afb7db933b7f99d8b6b4d5c`).
- [x] Execute 12 automatic and 192 reviewed outcomes with inline independent checks (all passed).
- [x] Audit aggregate results and interpretation; report both recall definitions and event-core losses (69 arms, 897 scopes and 3,588 padding rows independently checked; interpretation receipt alongside results).
- [x] Preserve source/input/results/figures and durable pending memory (byte archive and verification receipt in the study reproducibility directory; pending vault draft updated, canonical memory unchanged).
