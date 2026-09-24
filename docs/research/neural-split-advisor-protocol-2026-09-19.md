# Production-preserving compact split adviser

The user authorized the next experiment after prioritizing retained rallies and asking for compact neural guidance for rally breakdown and low-confidence cleanup. This fixed development experiment separates export footage, proposed score-tracking event identities, and human-confirmed identities. Production app code and weights remain unchanged.

## Scope and fixed matrix

Reuse the exact eight indoor/grass recordings, four source groups, 322 labels and compact short-boost predictions from seeds 3407, 1729 and 20260918 in the preceding registered review experiment. Reuse its original float64 timestamps, float32 four-head scores and original production event IDs. No beach, new fitting, learned confidence calibration, or protected test. Prior artifact and source hashes are verified before execution. Earlier development exposure remains a limitation.

Three label-blind split inventories:

- `event_starts`: compact event starts strictly more than two seconds inside a production parent/valid component, with this and an earlier compact event each overlapping the parent in that valid component by at least 0.5 seconds. Strength is current compact-event mean live probability over actual tick centers in that intersection.
- `head_evidence`: local serve-head maxima >=0.35 with preceding three-second end maximum >=0.25 or live minimum <0.35, strictly more than two seconds inside the parent/valid component. Strength is serve score. Thresholds reuse the previous head-guided proposal study.
- `corroborated`: one-to-one event/head candidates within one second, paired deterministically by distance then head strength then time; use the head timestamp and minimum of the two strengths.

One-second nonmaximum suppression uses descending strength then earliest time; distances strictly below one second conflict. Evidence never crosses ignored intervals. These strengths are heuristics, not calibrated probabilities. Exact source implementations are frozen after synthetic qualification and before outcome inspection.

Automatic outputs apply all proposed splits, partitioning each original production parent into touching children without removing any raw occupancy. A new start is model-evidenced; the preceding child's cut end is synthetic, not an observed dead-ball time. Preserve parent lineage and original external endpoint markers. No automatic cleanup deletion.

Cleanup flags mark original production parents with less than half of their evaluable raw duration supported by compact events dilated two seconds (clipped, ignored time removed, no short-gap joining). Intersect neural events with each valid-time component before dilation and clip dilation to the same component; ignored detections cannot create support. Nonmaximum suppression is also isolated by valid-time component. Flag strength is one minus support fraction. Flags do not remove footage or events automatically.

Review inventories are `split_only` and `combined` for each of the three split policies, plus a single `cleanup_only` control: seven inventories. Each uses chronological and evidence-per-playback-second order, three seeds, and nested 5/10/20/40 percent per-video budgets: 42 queue/seed runs and 168 reviewed outcomes. There are also nine automatic split outputs and the production baseline. All fixed arms remain reported; no outcome-driven threshold changes.

## Review workload and human operation

One review job is an original production parent containing at least one relevant flag. Multiple split proposals and cleanup flags share that job. Viewing cost is the union of parent intervals expanded two seconds, clipped to video bounds, joined only for positive gaps strictly below three seconds, with ignored time removed without rejoining. Edit permission is selected raw parents minus ignored time. Sort chronological by start/end/ID, or by maximum relevant flag strength divided by standalone playback seconds, then start/end/ID. Greedy selection retains previously selected jobs at ascending caps, skips oversized jobs and continues. No gold informs selection. Report actual minutes, jobs, clips, edit regions, unique true rallies and unused budget.

The simulated human checks only proposed splits within selected parents. Match proposals one-to-one to eligible secondary gold starts within one second; accepted proposals snap to those starts, unmatched proposals are rejected. Unreviewed proposals remain unapplied. Do not invent unproposed splits or adjust any original parent endpoint. This deliberately limited ideal-human operation is not full boundary editing: it leaves inter-rally dead time inside event partitions and cannot recover production omissions.

If a selected parent carries a cleanup flag and has zero evaluable gold-core overlap, remove its event records from the score-tracking timeline. Mixed true/false parents are retained. This is ideal whole-false-event classification, not ideal trimming. **The frozen production export remains unchanged even after this event-timeline cleanup.** Removing footage after confirmed cleanup is outside this experiment. Report event-timeline occupancy loss separately so it cannot be mistaken for export loss. Start/end confidence flags remain distinct from event existence and export inclusion.

## Evaluation and decision contract

Export primary is pooled `F1_padP_coreR` at symmetric two-second padding, strict positive-gap <3-second join. Report all symmetric 0/1/2/3-second cases, pooled P_pad/R_core/F1, model/human export duration and difference, correct removed time, incorrectly omitted wanted human export and incorrect export. Export values must equal production exactly for every arm. They therefore tie on the mandatory primary ranking; event diagnostics do not become a replacement export ranking.

Evaluate actual proposed/confirmed event lists separately with the preceding qualified one-to-one IoU>=0.5 event metrics, any/material merge/split errors, complete misses, new complete-miss IDs and observed start/end localization. Headline start tolerance is one second. Start labels are a serve-contact proxy, not an independent complete serve marker, serving-side, winner or score benchmark. Synthetic split ends do not count as observed ends.

Split targets are later gold rally starts within an original production parent and same valid-time component as an earlier materially overlapping gold identity. Material overlap is >=min(0.5 seconds, 10 percent of gold duration). Exclude gold identities touching ignored time consistently with prior identity metrics. The first overlapping gold is never a split target. Include near-edge targets in recall denominators and report two-second-guard accessibility separately. Match proposed timestamps one-to-one per parent using maximum cardinality then minimum time error at 0.5/1/2 seconds; duplicates count as false proposals. Report split precision, recall, F1, missed targets and spurious proposals. This target definition does not treat a shifted initial start as an additional rally.

Report pooled per-seed metrics, mean/range across three seeds, per-source-group and per-recording results. Source replicas are not extra independent videos. For cleanup flags report wholly false, mixed/real parent counts, real rallies touched, and their ideal-review yield. A useful split candidate must preserve export and original raw coverage automatically, avoid new full rally misses, and improve event/start metrics without unacceptable spurious splits. No automatic split policy is promoted based only on unchanged duration F1. Review results assume perfect restricted actions, not observed human performance.

## Verification and execution checklist

The Tasks tool is unavailable; this checklist tracks the work.

- [x] Read instructions and identify immutable inputs.
- [x] Implement and synthetically qualify split proposals, target matching, review queues and oracle actions (105 tests passed).
- [x] Freeze protocol, input references, sources and matrix before numerical outcomes (contract `617b92ee2d1d101afd3e06568e6773fead23f21f69ed67f15164ccc5388211ac`, 32 sources).
- [x] Run all fixed arms and independently audit candidates, coverage, lineage, queues, human actions and four-padding accounting (nine automatic and 168 reviewed outcomes).
- [x] Summarize automatic split quality, cleanup flag yield, reviewed event metrics and workload without claiming score accuracy.
- [x] Preserve reproducible source/input/results artifacts and record durable conclusions (frozen source/input snapshots, archive byte verification, and pending vault draft capture).
