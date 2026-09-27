# Recognition and local-attention execution — 2026-09-22

User authorized execution of the [recognition proposal](mobile-recognition-and-transformer-plan-2026-09-22.md). This record freezes the initial experiment before new quality outcomes. No production rollout is part of the experiment.

## Work checklist

Tasks tool is unavailable; this checklist tracks execution.

- [x] Inspect resource availability and existing source/label/feature contracts.
- [ ] Implement and test matched-context local attention, player-motion extraction and mobile regional CNN extraction.
- [ ] Freeze source/input hashes and pass real-data engineering checks.
- [ ] Complete the AV and DINO temporal-attention comparisons.
- [ ] Qualify player and mobile visual extraction, then complete their TCN comparisons where engineering passes.
- [ ] Audit selections, coverage, feature provenance and runtime conversion; report all padding cases and failures.
- [ ] Record remaining annotation/device qualifications explicitly.

## Frozen initial scope

Use the existing repaired 18-recording manifest, 8 exact evaluation recordings / 4 source groups, and the separate reviewed-draft and reviewed-export training tiers. Beach and protected sources remain excluded. All raw/project/proxy derivatives remain in the same source family. Existing gold and ignored intervals are immutable; new feature extractors receive sanitized video/ROI/timestamp manifests without labels.

The initial temporal comparison uses reviewed-export supervision and short-boost live loss for both AV and DINO. This preserves the completed short-context study's original-context control recipe. DINO global-control results remain descriptive references, not a different loss secretly assigned to the new architecture. Use seeds 3407, 1729, 20260918; checkpoints 5, 15, 30, 60; identical optimizer, sample exposure, real context and decoder grid. Six unordered excluded-source-pair inner fits supply the twelve isolated logical views, followed by four outer refits per seed. No outer fold enters its own selection.

Training retains historical core 128 / halo 62 chunks (maximum 252 real ticks), rather than introducing padding to reach the proposal's round-number 256 input length. Both heads still see the same 125-tick receptive field. Static 256-token export is an engineering qualification only. Attention uses two radius-31 layers; changing to full-chunk attention is a separate experiment.

The feature arms add one family at a time to AV: continuous court-relative player movement, or frozen MobileNetV3-Small regional tokens. Both first use the unchanged TCN temporal backbone and short-boost loss. CNN fine-tuning/distillation, ball, pose and new phase supervision follow separate qualification rather than being bundled into these initial comparisons.

## Evaluation and acceptance

Rank by pooled `F1_padP_coreR` at symmetric 2-second padding; always report 0/1/2/3 seconds with model/human durations and delta. Join positive gaps strictly below 3 seconds on both model and human export unions. Subtract identical ignored intervals afterward and never rejoin across them.

Report precision, retained core recall, event F1, complete and incomplete rally losses, short rallies (original duration <=3 seconds), longer-rally core recall, and each source group. Compute counts/durations per seed; seeds are not independent recordings. Keep production's checked-in baseline and historical exposure caveat visible.

The recovery gate versus each matched neural control requires at least 20% fewer completely missed rallies overall and among short rallies, no increase in incomplete losses, and no decline in pooled core recall or longer-rally core recall. Primary F1 may decline at most 0.005 for a recovery-only finding, which does not count as an F1 improvement. Require the directional result in at least two of three seeds and report every group. A general F1 improvement requires at least +0.02 F1 with no additional complete losses and no core-recall decline. Production replacement additionally requires no newly lost rally or reduction of retained core coverage versus production; no automatic promotion follows development results.

Review/addition configurations are evaluated separately from suppression. A flag alone does not improve output metrics. Rally identity and observed boundaries remain distinct from padded export coverage. New ball/person ground truth and real human review/device measurements must not be fabricated from existing rally labels or synthetic tests.

## Resource policy

One bounded GPU training worker initially; CPU extraction pilots can run independently. Do not stop other jobs, restart WSL, or change global WSL settings. Write caches/checkpoints/downloads to `private-reference-0173`; C: had approximately 3.2 GiB free at start. Inspect engineering throughput before scheduling full extraction. Never claim desktop extraction time as phone inference speed.
