# Beach-exclusion retraining — 2026-08-12

All 33 deployable rally/serve/stacked/dead-ball/dead-state model versions in the established lineage were retrained with the beach recordings removed from training. The excluded full-corpus recordings were `beach-source-02` and `beach-source-01`; the pilot and v0 derived manifests exclude their corresponding beach rows as well.

The machine-readable comparison and the complete Markdown report are stored in the external labeling workspace:

- [comparison JSON](/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/reports/beach-exclusion-retraining-comparison.json)
- [comparison report](/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/reports/beach-exclusion-retraining-comparison.md)
- [interactive HTML report](/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/reports/beach-exclusion-retraining-comparison.html)
- beach-free models: `/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/models`
- beach-free inference timelines: `/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/analyses`

## Result

The held-out test comparison improved for 29 versions, declined for 2, and was unchanged for 2 in Event F1. The mean Event F1 change was +0.089 and the median was +0.056. The largest full-lineage gains were:

- normalized audiovisual rally v3: +13.2 percentage points;
- normalized-audio serve specialist v5: +10.4 points;
- normalized-audio dead-state versions: +8.3 to +12.4 points;
- audiovisual stacked/peak variants: about +9 points.

The two declines were the visual-only serve specialist v1/v2 pair, each −0.6 points. The standard audiovisual v2/final pair was essentially unchanged in Event F1 (+0.3 points) but improved temporal IoU by 6.5 points. The targeted-pruning diagnostic was retrained with its original 365-feature retained signature and improved test Event F1 by +0.2 points; the beach-free feature-selection study did not promote a new pruned candidate.

Validation moved in the opposite direction overall: 31 versions decreased and 2 increased in Event F1. This reflects the smaller beach-free training distribution and newly selected decoders; the held-out test result is the more relevant paired generalization check here. Test remains retrospective because this corpus had already been inspected.

Every retrained model was inferred on all nine full recordings, producing 297 fresh artifacts. The two beach videos are included for qualitative out-of-domain inspection only and are not scored.

## Method notes

The comparison uses each lineage’s original and beach-free validation/test report, with “without beach − original” deltas. The full-corpus beach-free manifest retains the same two grass validation recordings and the same indoor test recording. Boundary-head training required a small protocol adaptation: after beach removal the full training split has two source groups, so dead-ball/dead-state epoch selection uses two-fold leave-one-source-group-out rather than the historical three-fold selector.

The fold-level `feature-order-2026-08-12` OOF artifacts and the separate ball-presence detector are not included because they are diagnostic components, not deployable versions of the rally/serve/dead-state model lineage.

The worktree changes also make the boundary epoch selectors and nested feature-study planner valid for the two-/three-group beach-free development layouts, and add an explicit inference-only manifest-mismatch flag for running beach-free models on the excluded beach recordings. Training and evaluation manifest checks remain strict.
