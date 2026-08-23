# Side-switch rare-event improvement plan — 2026-08-23

> Execution update: fixed `union34-top2-x2` superseded the original control as the
> current research winner. The subsequent within-recording pairwise/AUC-surrogate
> direction was rejected. See the
> [promotion](./side-switch-hard-negative-winner-promotion-2026-08-23.md) and
> [pairwise decision](./side-switch-pairwise-ranking-2026-08-23.md).
> The subsequent recording-reliability head was also rejected after every outer fold
> selected zero offset; see the
> [reliability decision](./side-switch-recording-reliability-2026-08-23.md).
> Focal and effective-number losses were then rejected after both fixed and nested
> event-F1 regressions; see the
> [loss decision](./side-switch-rare-event-losses-2026-08-23.md).
> The non-reanchored latent score prior was also rejected because rally ordinal is not
> point count; see the
> [soft-prior decision](./side-switch-soft-score-prior-2026-08-23.md).
> The separate expanded internal head also failed to add a held-out internal TP;
> boundary-only is retained as a small precision candidate. See the
> [internal-specialist decision](./side-switch-internal-specialist-2026-08-23.md).

## Decision

Execute a sequence of small, separately committed side-switch experiments. Preserve
`side-switch-v5-peak-cleanup-v1/local-peak-soft-count` as the current research control;
do not mutate its immutable artifacts or treat any result on the opened 11-video scope
as an automatic-production promotion.

The first architectural direction is to predict persistent court-side parity and then
detect a state change. A separate same-side continuity verifier is the first precision
follow-up. Both are more useful meanings of “predict the opposite” than adding a second
`no-switch` logit to the existing binary classifier.

## Evidence that determines the order

The exhaustive raw-phone audit contains 50 physical switches across 11 one-set videos.
All videos begin with the set at score zero. Under the primary full-gap ±4-second
matching contract:

- only 33/50 events lie in any of the 352 modern candidate gaps, so re-ranking the
  current universe has a 66% recall ceiling;
- the current local-peak+soft-count control emits 62 proposals and produces 25 TP,
  37 FP, and 25 FN: 40.32% pooled precision, 50.00% pooled recall, and 44.64% pooled
  F1;
- its proposals average 2.27 TP, 3.36 FP, 2.27 FN, and 5.64 proposals per video;
- correct selected proposals have mean/median scores 0.587/0.574 versus 0.537/0.515
  for false proposals;
- score discrimination is stronger within recordings than across them: all eight
  mixed TP/FP recordings have a higher TP mean, while `190429172` and `193307688`
  produce only relatively high-scoring false proposals; and
- a post-hoc global threshold improves the opened scope but fails to improve
  leave-one-video-out F1, so another global cutoff is not the next experiment.

The primary bottleneck is therefore candidate/state representation, followed by
recording calibration and hard-negative rejection. Rare-class loss weighting alone
cannot recover the 17 events that never enter the proposal universe.

## Does predicting the opposite help?

### Ordinary binary complement: no

For one binary classifier and one feature vector,
`P(no switch | x) = 1 - P(switch | x)`. A two-logit softmax expresses the same decision.
Adding that output supplies no new evidence and does not fix foreground/background
imbalance.

### Same-side continuity verification: potentially yes

A separately designed verifier can ask whether team-to-side correspondence remains
unchanged across two stable windows. This is not the complement of the existing score:
it has different inputs, supervision, and abstention behavior. It should compare:

- `sameCost`: pre-near to post-near plus pre-far to post-far; and
- `swapCost`: pre-near to post-far plus pre-far to post-near.

A high-quality, strongly lower `sameCost` is affirmative continuity evidence and may
veto a proposal. Low-quality comparisons must abstain rather than veto. Production
serve outputs may choose stable comparison frames, but prior results prohibit using
generic production outputs as hard eligibility gates.

### Persistent side-state prediction: yes, highest-value formulation

Assign the teams arbitrary identities from the first stable rally:

```text
state 0 = initial Team A near / initial Team B far
state 1 = initial Team A far  / initial Team B near
```

The ordered manual switch markers toggle this parity. Mask the physical transition
region, and label stable observations between markers with the current state. A rare
50-event problem then supplies many state observations. Emit an event only when the
estimated state flips and remains flipped over multiple stable anchors.

This follows the general methodology of recognizing actions through before/after state
changes and of detecting change points in a persistent sequence. The transfer to
volleyball is a project hypothesis, not a claim made by the cited papers.

### One-class “normal play” prediction: auxiliary only

A one-class model can learn ordinary no-switch appearance and expose anomalies. A
side switch is not the only anomaly: camera pans, huddles, timeouts, blur, zoom, and
players retrieving the ball can also leave the normal manifold. Use an anomaly score
only as candidate evidence or for hard-negative mining, never as the sole event rule.

## Proposed on-device architecture

```text
low-rate court-normalized frames
       ├── initial per-video team prototypes
       ├── current near/far team-to-side assignment
       ├── same-versus-swapped continuity score
       ├── court/camera/blur/team-separation quality
       └── coordinated cross-court motion evidence
                         ↓
             persistent side-state sequence
                         ↓
       high-recall change candidates over the full video
                         ↓
       local peak + soft count cleanup and confidence
```

The preferred runtime remains detector-free unless a detector supplies a measured
source-held-out gain. Use low-resolution HSV/color, occupancy, frame-difference, and
camera-compensated motion summaries; a two-state filter or small linear/temporal head
has negligible on-device state compared with video decoding.

## Execution loops

Every loop must implement the experiment, test it, write a decision record, commit the
implementation and record together, and leave the worktree clean before continuing.

### Loop 1 — side-parity feasibility on existing artifacts

1. Derive parity intervals from the full-video markers.
2. Audit which existing V4/V5/V6 frames and features can provide stable state anchors
   without decoding new video.
3. Build a leakage-safe parity/correspondence dataset grouped by recording.
4. Evaluate state separability and persistent flip localization.
5. Decide whether continuous extraction is justified and freeze its contract before
   decoding video.

This diagnostic must not call candidate-conditioned events exhaustive state samples.
If existing features cannot represent stable state, report that directly and proceed
to a preregistered extractor rather than tuning labels around the failure.

### Loop 2 — same-side continuity verifier

1. Use full-marker TP/FP assignments to form switch and hard no-switch pairs.
2. Compare same-versus-swapped assignment margins, stable-window aggregation, and
   quality-conditioned abstention.
3. Evaluate the verifier alone and as an add-only/veto-only layer over the current
   winner.
4. Require leave-one-recording-out improvement; an opened-scope global improvement is
   diagnostic only.

### Loop 3 — full-video candidate union

1. Extract low-rate state/change features outside the current rally-gap universe.
2. Union state flips, continuity breaks, coordinated cross-court motion, long dead
   intervals, and existing gap candidates.
3. Measure candidate recall before fitting a classifier. The first target is at least
   90% recall on development while recording proposal volume and on-device cost.
4. Pass the high-recall union through the frozen current winner and then through any
   independently validated continuity verifier.

### Loop 4 — rare-event objectives and calibration

Compare one change at a time:

- recording-balanced hard-negative mining;
- focal loss;
- effective-number class-balanced loss;
- within-recording pairwise or AUC-margin ranking; and
- a recording-reliability/abstention head using blur, court visibility, camera shift,
  team separability, and score-distribution summaries.

Do not use frame-level random splits, SMOTE-like interpolation of temporal examples, or
the raw number of easy negative frames as independent evidence. Report event precision,
recall, F1, PR-oriented ranking, per-video counts, candidate recall, and proposal load.

### Loop 5 — soft temporal and score-count prior

Retain local peak suppression and a soft event-count cost. Replace hard seven-point
re-anchoring with a state-space prior over uncertain cumulative points. Permit `+0`
redo observations, normal `+1` points, and missing/incorrect rally markers. Increase
switch hazard near totals 7, 14, and 21, but require visual state evidence and never
re-anchor future opportunities from a predicted switch.

## Label and evaluation contract

- The 11 fully reviewed raw-phone recordings are development scope for all decisions
  made after 2026-08-21.
- Preserve the manual point-marker artifact as canonical physical-event truth.
- A marker identifies an instant inside a switch, not the entire transition. Derive
  stable state labels only outside masked transition windows.
- Split and resample by recording, never by frame, gap, or candidate row.
- Measure candidate recall separately from decoder precision/recall.
- Preserve strict and ±4-second one-to-one event matching for continuity with the full
  audit.
- New untouched sets are required before selecting a production threshold or claiming
  generalization.
- No loop is a browser/Android promotion unless it separately passes quality, latency,
  artifact-parity, and source-held-out gates.

## Methodology sources

Sources were accessed on 2026-08-23. They support general methodology; none evaluates
VolleyCut footage or volleyball side-switch detection.

1. Alireza Fathi and James M. Rehg, “Modeling Actions through State Changes,” CVPR
   2013. Demonstrates action recognition and continuous segmentation through learned
   environment-state changes.
   [CVF paper](https://openaccess.thecvf.com/content_cvpr_2013/html/Fathi_Modeling_Actions_through_2013_CVPR_paper.html)
2. Ryan Prescott Adams and David J. C. MacKay, “Bayesian Online Changepoint
   Detection,” 2007. Provides sequential inference over run length and change-point
   probability.
   [Paper](https://arxiv.org/abs/0710.3742)
3. Tsung-Yi Lin et al., “Focal Loss for Dense Object Detection,” ICCV 2017. Introduces
   downweighting of abundant easy background examples for dense rare-foreground
   detection.
   [CVF paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
4. Yin Cui et al., “Class-Balanced Loss Based on Effective Number of Samples,” CVPR
   2019. Reweights long-tailed classes using effective rather than raw sample counts.
   [CVF paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)
5. Zhuoning Yuan et al., “Large-Scale Robust Deep AUC Maximization,” ICCV 2021.
   Develops a margin-based AUC objective for imbalanced classification.
   [CVF paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html)
6. Prannay Khosla et al., “Supervised Contrastive Learning,” NeurIPS 2020. Studies
   supervised contrastive objectives and reports robustness and reduced-data benefits;
   this motivates, but does not validate, paired same/swapped embeddings here.
   [NeurIPS paper](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
7. Lukas Ruff et al., “Deep One-Class Classification,” ICML 2018. Introduces Deep
   SVDD for directly trained one-class anomaly detection.
   [PMLR paper](https://proceedings.mlr.press/v80/ruff18a.html)
8. Ryuichi Kiryo et al., “Positive-Unlabeled Learning with Non-Negative Risk
   Estimator,” NeurIPS 2017. Provides a robust risk estimator when positives are known
   and remaining samples are unlabeled. This is reserved for future incompletely
   reviewed corpora, not the exhaustive 11-video scope.
   [NeurIPS paper](https://papers.nips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html)

## Related VolleyCut records

- [Full-video marker audit](./side-switch-full-video-marker-audit-2026-08-21.md)
- [Current peak/count/context experiment](./side-switch-v5-peak-cleanup-2026-08-20.md)
- [No-cadence causal diagnostic](./side-switch-v5-no-cadence-2026-08-20.md)
- [Production-state and serve-grounding experiment](./side-switch-production-state-experiment-2026-08-20.md)
- [Current winner manifest](../../data/side-switch-current-research-winner-v1.json)
