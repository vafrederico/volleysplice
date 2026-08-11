# Generated analysis vs. pilot gold labels

## Scope

The current `court-motion-audio-heuristic-v1` outputs under `/mnt/freenas/volleycut/analyses` were evaluated against `labeling-v1-2026-08-09/manifests/pilot-gold-v1.json`. The label pack covers nine 90-second segments (810 seconds total), all three surfaces, five independent source groups, and 30 continuously reviewed serve-contact-to-dead-ball rallies.

Each full-video prediction was clipped to the source offset encoded in the corresponding `*-startN-duration90.mp4` filename and translated to segment-local time. Event matches use one-to-one chronological matching at interval IoU >= 0.5. Time metrics use the union of non-overlapping intervals.

The exact machine-readable output is `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/heuristic-analysis-evaluation-v2.json`.

## Current baseline

| Scope | Rally F1 | Time IoU | Live recall | Live precision | Matched / truth |
|---|---:|---:|---:|---:|---:|
| All | 27.7% | 39.9% | 95.0% | 40.8% | 9 / 30 |
| Beach | 12.5% | 33.4% | 99.6% | 33.5% | 1 / 7 |
| Grass | 14.3% | 32.0% | 85.0% | 33.9% | 2 / 13 |
| Indoor | 57.1% | 51.3% | 100.0% | 51.3% | 6 / 10 |

The current system retains 244.5 of 257.4 labeled live seconds, but it selects 599.8 seconds total. That means it misses only 12.9 live seconds while retaining 355.3 seconds of dead time. This is a useful high-recall proposal generator, but not yet a rally cutter.

For the nine strictly matched rallies, starts average 0.57 seconds early (0.61-second MAE), while ends average 5.43 seconds late (5.43-second MAE; 8.61-second P90). The dominant boundary defect is therefore late ending, not onset.

## Failure analysis

- Ten of 30 labels are short aces or service faults under three seconds. The baseline overlaps 80% of them, but only 10% reach interval IoU 0.5 because they are absorbed into long setup/dead-time candidates. Lowering `min_rally_seconds` alone did not change meaningful metrics; the issue is separation and semantics, not only a duration filter.
- The 20 longer rallies have 95% any-overlap recall, but only 40% strict interval matches. Motion finds play, then continues through walking, ball retrieval, player rotation, and the next setup.
- Grass is the only surface with material live-time misses. In `grass-grass-source-10`, the automatic ROI is `y=0.6261, height=0.3539`, while the reviewed ROI is `y=0.22, height=0.76`. A sampled missed rally visibly has the ball above the net and players in motion outside most of the automatic region.
- Recomputing that segment with the broad reviewed ROI recovered 97.9% live time, but precision fell to 25.3% because neighboring courts, foreground walking, and setup also move. A better ROI is necessary but motion-only classification remains insufficient.
- Visual checks of false-positive spans across beach, grass, and indoor footage consistently show walking, retrieving the ball, or waiting in formation—not camera instability.

## Decoder experiments

These experiments reuse stored signals; they do not retrain or recompute video features.

| Variant | Rally F1 | Time IoU | Live recall | Live precision |
|---|---:|---:|---:|---:|
| Current settings, re-decoded | 28.1% | 40.0% | 95.0% | 40.9% |
| Remove 1-second ending tail | 29.0% | 41.4% | 93.3% | 42.7% |
| Reduce bridged gap from 2.5s to 0.75s | 41.1% | 40.8% | 92.6% | 42.2% |
| High seed 0.70 + 0.75s gap | 45.5% | 43.3% | 88.2% | 45.9% |
| Exploratory combined profile | 44.8% | 45.1% | 90.5% | 47.3% |
| Leave-one-source-group-out selection | 40.0% | 43.6% | 85.9% | 47.0% |

The exploratory combined profile uses high/low activity thresholds `0.70/0.20`, a `0.75s` bridge gap, the existing `3.0s` minimum, `0.75s` onset lead, and no ending tail. It cuts retained dead time from 355.3 to 259.0 seconds on the same labels, but misses 24.5 rather than 12.9 live seconds.

The same-label result is optimistic. Leave-one-source-group-out selection improves precision and time IoU but loses more live play, and selected parameters vary between indoor and outdoor folds. This pack is too small to replace production defaults confidently.

Audio weight `0.18` remained the best of the tested weights under the combined decoder. Weights of `0.30` and `0.50` reduced both F1 and time IoU, so increasing audio influence is not the next improvement.

## Existing learned-v0 context

Five existing `court-motion-temporal-logistic-v0` inference outputs could be mapped exactly or by a documented sub-crop to the new gold labels (409 seconds, 17 rallies). They achieve 20.7% rally F1, 28.0% time IoU, 58.4% live recall, and 35.1% live precision. On the same subset the heuristic achieves 17.1% rally F1, 31.3% time IoU, 98.7% live recall, and 31.5% live precision.

Most of those learned runs share a source group with training or validation material, so the aggregate is diagnostic only. The independent TDS sub-clip is more encouraging: learned-v0 and the heuristic both score 40% rally F1, while learned-v0 improves time IoU from 40.7% to 55.5%. This supports learning boundary/state semantics, but the existing model should not replace the heuristic because its aggregate live recall is far too low.

## Recommended improvements

1. Add the combined decoder settings as an explicitly named experimental profile, not the default. Re-evaluate it after labeling longer continuous spans and at least one new source group per surface.
2. Fix court-region validation before tuning motion further. Reject a full-court ROI that starts below roughly 35% of frame height or covers less than roughly 60% vertically; fall back to a capture-profile region and record the fallback. Use separate foreground, court-interior, and neighboring-court motion zones rather than one rectangle.
3. Make end-of-play a separate decision. Train or derive an ending state that requires evidence of dead ball, not merely continued player motion. Walking and ball retrieval should be explicit hard negatives.
4. Preserve the current heuristic as a high-recall proposal signal, then use a learned temporal state/boundary model with spatial motion-grid features and serve/ball cues to trim and split proposals.
5. Do not address short aces and service faults only by lowering the minimum duration. Add serve-contact and fast terminal cues; the duration ablation was effectively neutral.
6. Expand labels before product claims: full continuous games, a second independent beach source group, and an annotation-agreement pass. Keep source-group-separated validation and test sets.

## Reproduce

```bash
cd /home/developer/volleycut
npm run evaluate:labels -- \
  --labels /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/pilot-gold-v1.json \
  --parameter-search \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/<new-report-name>.json
```

Report destinations are immutable; choose a new filename for a new run.
