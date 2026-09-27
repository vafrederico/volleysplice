# F1 and recall model comparisons in the editor lab and labelv2

The two lab UIs expose ten frozen options (five model families/precisions, each with an F1 pick and a recall pick) for all 42 non-beach recordings in the completed recall-sweep inventory. The editor lab has a video selector. Existing production, suppression, compact review, and human-export modes remain available. Model boundaries are separate rally cores; export padding and joining do not merge those identities.

The later beach extension adds the two completed-exact beach recordings with only the requested high-recall DINO-TCN and Mobile-TCN choices. It reuses the selections below without beach fitting, calibration, decoder search, or model selection. Frozen DINO caches are reused; label-blind MobileNet features and temporal-head scores are new. Per-recording model catalogs let these two recordings expose two choices while the original 42 continue to expose all ten. Labeling Editor v2 keeps its editable human document and production-ensemble reference alongside the new read-only model rails.

## Selection requested for this UI

Select the highest **F1_padP_coreR** on `common-unseen / exact-rallies / all` among individual registered variant/draw fits with an available 99% calibration operating point. Fall back to 98% only if none qualify. Rank at symmetric 2-second padding and strictly-under-3-second gap joining. Do not average per-video F1 or choose a model's best padding. This is selection of a runnable checkpoint, not a claim that its variant wins across every draw.

The user explicitly requested common-unseen selection after the study finished. That panel is now development/selection evidence for these UI choices, not an independent test of the winners. Its exact-label stratum contains one recording (indoor source); reviewed drafts and export coverage remain separate strata. No training or new threshold search was performed. The immutable original study results were not changed.

The added recall option keeps each F1-selected variant fixed, chooses its highest-recall eligible draw, and breaks ties by F1. All ten options have a 99% calibration target.

| UI option | Pick | Variant | Draw | P_pad | R_core | F1_padP_coreR |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| INT8 DINO-transformer | F1 | expanded-large | 20260923 | 92.66% | 96.82% | 94.69% |
| INT8 DINO-transformer | Recall | expanded-large | 20260918 | 83.61% | 98.80% | 90.58% |
| FP32 DINO-transformer | F1 | expanded-large | 20260923 | 92.53% | 97.19% | 94.80% |
| FP32 DINO-transformer | Recall | expanded-large | 20260918 | 83.05% | 98.80% | 90.24% |
| DINO-TCN | F1 | original-medium | 20260923 | 94.13% | 97.72% | 95.89% |
| DINO-TCN | Recall | original-medium | 3407 | 81.32% | 100.00% | 89.70% |
| Mobile-TCN | F1 | expanded-large | 20260918 | 96.68% | 97.21% | 96.94% |
| Mobile-TCN | Recall | expanded-large | 3407 | 69.04% | 99.51% | 81.52% |
| Distilled Mobile-TCN | F1 | expanded-large | 20260923 | 89.69% | 96.50% | 92.97% |
| Distilled Mobile-TCN | Recall | expanded-large | 3407 | 66.09% | 99.51% | 79.43% |

**The target is not measured recall on this panel or any selected video.** INT8 refers to the saved mixed-INT8 DINO embedding extractor; the temporal head remains FP32. Within each selection mode, the two transformer winners share checkpoint and decoder, allowing a direct precision comparison.

## NAS data and reproducibility

The output root is supplied at runtime. The beach publisher records provenance with recording and model indexes from the frozen inventory, phase-one feature manifest, and comparison catalog; its receipts, reference manifests, and beach audit do not retain host media, feature, checkpoint, or output paths.

- `index.json`: selected models and per-recording reference locations.
- `recordings/*.json`: ten sets of core predictions and full native-timestamp live/start/end/keep signals.
- `audit.json`: checkpoint/decoder identity, 340 exact matches to saved scored model/video outputs, and all four padding cases (0, 1, 2, 3 seconds) for each common-unseen label stratum, including export durations and differences.
- `labeling-catalog.json`: merged catalog with reviewed human imports. Existing saved drafts take priority; no gold label files are overwritten. Ignored intervals are unioned and sorted for the labeling schema.
- `qa/`: API coverage, browser screenshots/results, tests and build log.
- `beach-audit.json`: the two label-blind beach feature/inference runs, frozen checkpoint identities, per-video prediction counts, and proof that the human-label catalog did not change.

Regenerate with `scripts/prepare-neural-comparison-ui.py --study <recall-sweep-root> --output <NAS-output> --catalog <previous-catalog>`, using the existing neural Python environment and `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.`. It decodes already saved probabilities; human labels are never passed to the decoder. This explicitly uses common-unseen outcomes to choose the UI checkpoints.

Set `VOLLEYCUT_NEURAL_COMPARISON_INDEX_PATH` to `index.json`, and `VOLLEYCUT_FULL_NAS_CORPUS_PATH` to `labeling-catalog.json`. Preserve the existing legacy research and editor-lab manifest settings to retain earlier review modes. Per-video identities are validated before displaying results. Legacy August raw imports use their existing feedback-document identity where no video-byte SHA was available; this is not a new claim of video hashing.

## Behavior and limits

Each editor mode keeps a separate revision-bound local draft and its own model signals. Existing F1-mode draft revisions are preserved when their boundaries and signals match exactly; adding recall options does not erase those edits. A model switch never substitutes another model's scores. The generic neural rally confidence is its mean live-head score within the decoded interval, not calibrated event correctness. Saved human labels remain a separate editable lab copy and comparison reference.

These new modes contain rally predictions and serve-start timing signals. Serving-side inference has **not** been computed at their new anchors; sides from production or other boundaries are not borrowed. Earlier modes retain their existing serving predictions. Imported reviewed exports are useful coverage labels, but their boundaries are approximate and UI core metrics do not establish frame-exact rally accuracy.

The local labeling server runs on port 3000. Its `.next` output, temporary files, browser profiles and generated datasets are on the NAS; `.next` is a local symlink to the NAS build directory. No production app or training worker was started.

## Verification

All 42 editor-lab model APIs and video range requests passed. All 42 label-reference APIs expose ten model options and aligned signals. Browser checks cover every mode, four signal heads, video switching, the human comparison, playback advancing, and the labelv2 multiselect on an imported export; no page or console errors. The 30 focused tests passed, including the real frozen legacy checks. Next's optimized build and TypeScript check passed.

The broader web suite returned 300 passes, 3 failures and 2 opt-in skips. The same three failures reproduce against a NAS-extracted copy of the preceding commit: the shared-specialist source-contract assertion and two production-suppression geometry assertions. Those production files were not changed. Logs are in `qa/all-web-tests.log` and `qa/baseline-tests.log`.
