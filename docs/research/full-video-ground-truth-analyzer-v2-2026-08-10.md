# Full-video ground truth evaluation — analyzer v2

## Outcome

Analyzer v2 improves rally separation and end boundaries across the authoritative full-video corpus. It is a materially better review cutter, but it still over-segments and retains too much dead time for unattended exports.

The corpus contains nine full videos, 9,082 seconds, and 352 serve-contact-to-dead-ball rallies. Source groups do not cross the declared train, validation, and test splits. The user confirmed the intervals are authoritative ground truth; stale `in-progress` and `ai-prelabel` metadata must not be used to filter them.

Strict event matches use one-to-one chronological matching at interval IoU >= 0.5. Time metrics use interval unions.

## Changes

- Reduced the maximum bridged inactive gap from 2.5 seconds to 0.75 seconds.
- Removed the automatic 1-second ending tail.
- Retained the existing activity thresholds, 0.75-second onset lead, and 3-second minimum. Lower minimum-duration settings added fragments without a reliable held-out gain.
- Reject a detected court activity region only when it is severely bottom-cropped (`top > 0.55` or `height < 0.42`), then use the existing broad fallback ROI.
- Record exact decoder settings in each new `analysis.json`.

Only the decoder and ROI rule were selected for production. A richer logistic temporal model improved event separation but reduced untouched-test live recall to 77.1% while leaving time IoU effectively flat, so it was not integrated.

## Results

| Scope | Analyzer | Rally F1 | Time IoU | Live recall | Live precision |
|---|---|---:|---:|---:|---:|
| All | v1 | 21.9% | 35.8% | 93.8% | 36.7% |
| All | v2 | **33.4%** | **38.6%** | 91.2% | **40.1%** |
| Validation | v1 | 14.4% | 27.5% | 78.1% | 29.8% |
| Validation | v2 | **28.3%** | **31.6%** | **87.0%** | **33.2%** |
| Untouched test | v1 | 10.8% | 35.8% | 99.9% | 35.8% |
| Untouched test | v2 | **28.6%** | **38.8%** | 99.5% | **38.9%** |

Across all videos, strict matches increased from 78 to 152. End-boundary MAE fell from 3.81 seconds to 2.32 seconds, and its P90 fell from 9.36 seconds to 6.49 seconds. Retained dead time fell by 661.7 seconds, from 4,239.9 to 3,578.2 seconds. The tradeoff is 67.1 additional missed live seconds, leaving overall recall above 91%.

By environment, rally F1 changes are:

- Beach: 24.6% to 28.7%.
- Grass: 16.0% to 26.2%.
- Indoor: 28.7% to 45.4%.

The severely cropped `grass-grass-source-10` court estimate changed from `y=0.6261, height=0.3539` to the broad fallback. With the v2 decoder, its rally F1 improves from 12.4% with the old motion signal to 26.3%, time IoU from 18.9% to 29.8%, and live recall from 42.0% to 77.8%.

## Artifacts

- Baseline: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-ground-truth-baseline-v1.json`
- Analyzer v2: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-ground-truth-analyzer-v2.json`
- Reviewable outputs: `/mnt/freenas/volleycut/analyses/*-v2/`

The nine v2 proxies are hard links to the existing normalized proxies, so they do not duplicate video storage. The review UI lists the v2 analyses separately.

## Caveats and next work

- V2 produces 558 candidates for 352 ground-truth rallies. It separates more real rallies, but manual exclusion and occasional merging are still required.
- Seven ground-truth rally pairs share an exact boundary. A gap-based decoder cannot distinguish them without serve/contact semantics.
- Beach has no independent held-out source group; validation is grass-only and test is indoor-only.
- The next model should classify explicit live, setup, and post-play states using spatial motion features, with walking, retrieval, and timeout intervals as hard negatives. The current v2 outputs establish the acceptance baseline.

## Reproduce

```bash
cd /path/to/volleysplice
npm run reanalyze-no-model:dataset
npm run evaluate:labels -- \
  --labels /mnt/freenas/volleycut/labeling-v1-2026-08-09/labels/full \
  --analysis-suffix=-v2 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/<new-name>.json
```

Both analysis directories and reports are immutable; choose a new suffix or report filename for another run.
