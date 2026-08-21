# VolleyCut model registry

This is the durable lineage registry for models trained by the VolleyCut project. It
records what each model consumes, the video files used to fit its learned parameters,
which data was used only for selection or evaluation, what predecessor it was intended
to improve or compare with, and what changed. Feature formulas and production versus
research status live in [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md); inference topology,
decoders, and ensemble behavior live in
[`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md).

This inventory was reconstructed from the checked-in production bundles, named NAS
artifacts, artifact metadata, manifests, and research reports. A field marked “not
recorded” is deliberately unknown; do not infer missing lineage from a version number.

## Registry contract

A registry entry is required for every durable learned artifact and every report-only
training study. A bundle containing independently fitted heads must name every head.
Ephemeral cross-validation fold files may be represented by their parent study rather
than one row per fold, provided the study records the folds, source set, and feature
signature.

Every new entry must record:

- immutable model ID/version and artifact path;
- learned target/head and algorithm;
- exact ordered feature signature, feature profile, and feature-generation config;
- **fit sources** whose labels or frames affected weights, calibration, adapters, or
  learned preprocessing;
- **selection sources** used for thresholds, epochs, seeds, decoders, or promotion but
  not for fitting weights;
- evaluation-only and protected-test sources, explicitly separated from both roles;
- source content hashes, label/manifest hash, code revision, random seed, and artifact
  hash when available;
- `Compared with`, `Change`, `Reason`, and `Result/disposition` fields. When no factual
  predecessor or change record is available, write `not recorded` rather than guessing.

Validation and protected-test video is never a training source merely because a report
shows a metric for it. A later final refit that intentionally includes former
development data must register a new source set and artifact.

This registry includes named linear heads, adapters, and learned suppression,
side-switch, and serving-side classifiers. It excludes fixed decoders and heuristics,
third-party pretrained backbones
that VolleyCut did not train, cached feature matrices, and predictions. YOLOX, MobileNet,
and DINOv2 are therefore mentioned only where VolleyCut trained a downstream head or ran
a model study.

## Feature profile keys

| Key | Model inputs | Meaning |
| --- | ---: | --- |
| `F28` | 140 | 28 appearance/frame-difference base features × five offsets; historical no-flow ablation |
| `F42` | 210 | 42 base court-motion/flow features × five temporal offsets; historical v1 |
| `F42-raw` | 210 | Same base/context layout without within-recording percentile normalization |
| `F90` | 450 | 90 audiovisual v2 base features × five offsets |
| `F104` | 520 | 104 noise-normalized audiovisual v3 base features × five offsets; production |
| `F90+serve` | 451 | F90 plus one cross-fitted serve-contact probability |
| `F90+peaks` | 454 | F90 plus four decoded serve-peak-window features |
| `SIDE36` | 36 | Separate side-switch marker appearance/context signature |
| `SERVSIDE38` | 38 | Nineteen serving-side motion/palette/HOG scalars plus paired missingness indicators |
| `SERVSIDE237-FLIGHT` | 237 | 82 court-flow recording ranks plus 155 fixed-anchor residual-flight grid recording ranks |
| `RAW-VLM` | n/a | Raw video windows and text targets; no handcrafted feature matrix |

The profile key is a summary only. The ordered `featureNames` embedded in an artifact is
the authoritative signature.

## Current production system

Production browser assets are under `prod/public/runtime/`; byte-equivalent Android
copies are under `android/app/src/main/assets/`.

| Artifact | Learned models | Features | Fit sources | Compared with and change | Disposition |
| --- | --- | --- | --- | --- | --- |
| `model-1ca43e38eefc` (`environment-specialists-v2-all-labels`) | Rally, serve-contact, and dead-state linear heads | F104 | `ENV2-ALL11` | Compared with `environment-specialists-v1/all-labels` and previous production. Added three indoor recordings containing walking/ball-retrieval hard negatives; refit all three heads on six grass plus five indoor recordings. | Primary production bundle. Artifact SHA-256 `1ca43e38eefc…`; component artifact hashes remain in the bundle. |
| `model-9c92b8e9333f` | `full-audiovisual-audio-normalized-v3` rally, `serve-specialist-audio-normalized-v5` serve, and `dead-state-transition-audio-normalized-v5-no-legacy-final` dead-state heads | F104 | `GOLD-NB-T4` | Compared component-by-component with the prior F90 models. Added 14 noise-normalized audio base features; the serve/dead-state heads also tested legacy/new audio ablations. | Previous production bundle, retained in the two-model consensus because it still recovers some rallies missed by the newer refit. Artifact SHA-256 `9c92b8e9333f…`. |
| `suppression-overlap-exclusion-retrained` | Binary false-positive suppression head | F104 | `FEEDBACK16` | Compared with the original feedback suppression v3 fit. Corrected the target builder so false-positive intervals overlapping an included corrected rally core were excluded from suppression positives; production decoder settings were held fixed. | Production suppression model, gated to eligible one-model intervals. Artifact SHA-256 `39eddf581639…`; learned-weights SHA-256 `a943…`. |

## Historical named artifacts

### Earliest excerpt models

Artifact roots are `/mnt/freenas/volleycut/v0-2026-08-09/models/` and the corresponding
no-beach sensitivity root
`/mnt/freenas/volleycut/v0-2026-08-09-no-beach-2026-08-12/models/`.

| Artifact | Target/features | Fit; selection | Compared with and change | Disposition |
| --- | --- | --- | --- | --- |
| `real-rally-v0` | Rally, F42 | `V0-T3`; `V0-V1` | First recorded learned fixed-camera rally classifier; predecessor not recorded. | Historical baseline. |
| `real-rally-v0-no-flow` | Rally, F28 | `V0-T3`; `V0-V1` | Matched ablation of `real-rally-v0`; removed all 14 optical-flow base columns, reducing 210 inputs to 140. | Historical diagnostic. |
| no-beach `real-rally-v0` | Rally, F42 | `V0-NB-T2`; `V0-V1` | Fresh sensitivity refit of the corresponding v0 artifact after removing the beach excerpt from fitting. | Historical no-beach sensitivity. |
| no-beach `real-rally-v0-no-flow` | Rally, F28 | `V0-NB-T2`; `V0-V1` | Combines the no-beach training-scope change with the matched 140-input no-flow ablation. | Historical diagnostic. |

### Full-gold artifact family

Unless a row overrides it, these artifacts live under
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/`, fit on `GOLD-T6`, and use
`GOLD-V2` only for development selection. “Same learned weights” means a separately
named artifact was metadata/re-export/finalization, not another fit.

| Artifact(s) | Target/features | Compared with and change | Disposition |
| --- | --- | --- | --- |
| `pilot-baseline-v0` | Rally, F42-raw; fit `PILOT-T6` | Replaced the tiny excerpt corpus with six 90-second pilot clips; no percentile sequence normalization. | Historical pilot baseline. |
| `pilot-percentile-v1` | Rally, F42; fit `PILOT-T6` | Matched comparison with `pilot-baseline-v0`; added tied within-recording percentile normalization. | Percentile normalization retained. |
| `full-percentile-v1` | Rally, F42 | Compared with `pilot-percentile-v1`; replaced 90-second clips with the six fully labeled source videos while retaining F42. | Historical full-video baseline. |
| `full-audiovisual-v2`; `full-audiovisual-v2-final` | Rally, F90 | Compared with `full-percentile-v1`; expanded 42/210 to 90/450 with quality, camera-compensated motion, temporal formation proxies, and 13 audio columns. `-final` has the same learned weights and freezes final metadata/decoder selection. | v2 reference model; superseded in production. |
| `audiovisual-v2-targeted-pruned-diagnostic` | Rally, F90 signature with 365 active contextual inputs | Compared with full F90; zeroed/removed feature groups implicated by ablation diagnostics while preserving the artifact signature. | Rejected after the live-recall guardrail failed. |
| `serve-specialist-v1`; `serve-specialist-v2` | Serve contact, F42 | Added a dedicated serve-contact head instead of relying on rally probability. v2 has the same learned weights and adds expanded selection/tie-break metadata. | Historical serve baseline. |
| `serve-specialist-audiovisual-v3`; `serve-specialist-audiovisual-v4` | Serve contact, F90 | v3 upgrades the serve head from F42 to F90. v4 has the same learned weights and records corrected cadence/parity evaluation and re-export metadata. | Superseded by normalized-audio v5. |
| `full-audiovisual-audio-normalized-v3` | Rally, F104 | Compared with `full-audiovisual-v2-final`; appended 14 noise-normalized broadband/band audio columns, yielding 70 new contextual inputs. | Became rally head in previous production. |
| `serve-specialist-audio-normalized-v5` | Serve contact, F104, all legacy and new audio | Compared with `serve-specialist-audiovisual-v4`; added the v3 noise-normalized audio bank. | Became serve head in previous production. |
| `serve-specialist-audio-normalized-v6-no-legacy` | Serve contact, F104 signature with 65 legacy-audio contextual inputs disabled | Matched ablation of v5: visual plus only the new audio bank. | Research ablation; not promoted. |
| `serve-specialist-audio-normalized-v7-new-only` | Serve contact, F104 signature using only the 70 new-audio contextual inputs | Matched ablation of v5: removed visual and legacy-audio evidence. | Research ablation; not promoted. |
| `full-audiovisual-serve-prob-stack-v1-control` | Rally, F90 | Matched F90 control for the stacked serve experiment. | Research control. |
| `full-audiovisual-serve-prob-stack-v1-exploratory` | Rally, F90+serve | Compared with its control; appended one source-group-cross-fitted serve probability. | Rejected; did not pass the gate. |
| `full-audiovisual-serve-peak-window-v1-exploratory` | Rally, F90+peaks | Compared with the F90 control; appended four features describing decoded serve-peak membership, proximity, offset, and confidence. | Rejected; did not pass the gate. |
| `dead-ball-specialist-audiovisual-v1`; `dead-ball-specialist-audiovisual-v2` | Local end-boundary/dead-ball, F90 | Added a boundary-local specialist to the rally pipeline. v1 selection chose an exact no-op refinement; v2 retains the same learned weights with re-export/report metadata. | Historical diagnostic; no promoted boundary change. |
| `dead-ball-specialist-audio-normalized-v2-legacy-only` | Local dead-ball, F104 signature using legacy audio | Audio-input control for the normalized-audio boundary study. | Research ablation. |
| `dead-ball-specialist-audio-normalized-v3` | Local dead-ball, F104 full | Compared with v2; enabled legacy plus new normalized audio. | Research candidate. |
| `dead-ball-specialist-audio-normalized-v4-no-legacy` | Local dead-ball, F104 without legacy audio | Compared with v3; disabled legacy audio while retaining visual and new audio. | Research ablation. |
| `dead-ball-specialist-audio-normalized-v5-new-only` | Local dead-ball, F104 using only new audio | Compared with v3; removed visual and legacy-audio evidence. | Research ablation. |
| `dead-state-transition-audio-normalized-v0-legacy-only`; `…-v3-legacy-only-final` | Global dead-state transition, F104 signature using legacy audio | v0 is the legacy-audio control. v3 is the same learned weights finalized after the comparative evaluation. | Historical control/final alias. |
| `dead-state-transition-audio-normalized-v1-full`; `…-v4-full-final` | Global dead-state transition, F104 full | Compared with v0; enabled the full new normalized-audio bank. v4 is the same learned weights finalized. | Research candidate/final alias. |
| `dead-state-transition-audio-normalized-v2-no-legacy`; `…-v5-no-legacy-final` | Global dead-state transition, F104 without legacy audio | Compared with v1; disabled legacy audio while retaining visual and new audio. v5 is the same learned weights finalized. | v5 became dead-state head in previous production. |
| `dead-state-global-audio-normalized-v1-full-final`; `…-v2-full-final` | Algebraic inverse-rally dead-state control, F104 | Compared with transition-trained dead-state heads; uses global inverse-rally targets. v2 is the same learned weights with final re-export metadata. | Research control; not promoted. |
| `side-switch-specialist-v1` | Side-switch marker, SIDE36; fit `SIDE4`, selected on `GOLD-V2` | First dedicated learned side-switch ranker; compared against marker heuristics rather than a prior trained side-switch model. | Rejected for automatic use; retained for review ranking. |
| `serving-side-specialist-v1` | Camera-space serving side, SERVSIDE38; fit six declared train recordings, threshold selected on two validation recordings | First reviewed near/far classifier using the existing whole-half, baseline-band, and HOG evidence bank; compared with the nine fixed signed-score variants. | Research baseline only: 90.41% raw balanced accuracy, but 62.30% on the single protected-test indoor recording. Model SHA-256 `97356fe4ad38…`; see [`serving-side-specialist-v1-2026-08-20.md`](docs/research/serving-side-specialist-v1-2026-08-20.md). |
| `serving-side-fixed-flight-v3` | Camera-space serving side, SERVSIDE237-FLIGHT; correction-clean development fit with leave-one-source-group-out selection | Adds fixed-anchor residual-motion grid ranks to the court-flow bank; compared with the correction-clean court-flow baseline, trajectory/high-resolution variants, quality mixtures, and group-balanced fits. | Selected next-production serving-side model and current results-UI source: 94.12% development source-group macro balanced accuracy and 94.05% pooled balanced accuracy. Final fingerprint `85bc3325fbd4…`; the frozen 95% precision review band routes 3.90% of development cross-fit rows. Browser/Android artifact export and parity remain deployment requirements. The later 24-row assisted uncertainty review confirmed every current human label, so no refit was required. See the production-target contract below. |
| `serving-side-hybrid-serve-gate-v2` | Next-production serving-side composition; two frozen production serve heads plus both-model production rally containment | Compared with the serve-head-only UI gate; recovers a side only when both serve heads miss and the saved anchor is inside a rally interval supported by both production models. | Selected policy to ship with fixed-flight v3. It is deterministic composition, not another trained head, and does not change production rally ranges. It recovers 34 of 56 current true-serve gate misses while adding one false serve; all 35 recovered candidates require review. Browser/Android integration and parity remain pending. See the production-target contract below. |

The Unicode ellipsis in three grouped rows abbreviates only the repeated artifact prefix:
the complete names are
`dead-state-transition-audio-normalized-v3-legacy-only-final`,
`dead-state-transition-audio-normalized-v4-full-final`,
`dead-state-transition-audio-normalized-v5-no-legacy-final`, and
`dead-state-global-audio-normalized-v2-full-final`.

### Next-production serving-side contract

The serving-side production target is the indivisible composition
`serving-side-fixed-flight-v3` + `serving-side-hybrid-serve-gate-v2`. Do not deploy
fixed-flight v3 with the older serve-head-only gate, and do not describe the hybrid
gate as a learned replacement for either production serve head.

The learned side model is one deterministic class-balanced logistic regression:

- Target: physical camera-space `near=1`, `far=0` at a saved serve/rally anchor.
- Inputs: `SERVSIDE237-FLIGHT`, ordered as 82 tied within-recording court-flow ranks
  followed by 155 tied within-recording concentrated-flight ranks. The formulas and
  generated order are specified in [`FEATURE_PIPELINE.md`](FEATURE_PIPELINE.md).
- Fit universe: 1,027 correction-clean rows from 28 recordings and nine development
  source groups: 507 near and 520 far. Fifteen source-quality-affected rows and all
  current `not-serve`/unclear rows are excluded. The protected recording was not used
  for fitting, feature/candidate selection, calibration, or threshold selection.
- Parameters: median imputation, per-column mean/standard-deviation scaling,
  class-balanced sample weights, L2 `0.1`, 237 weights, and one bias.
- Output: `sigmoid(clamp(z, -30, 30))` is the near-side score. Scores at or above
  `0.4783744762021848` emit `near`; lower scores emit `far`.
- Calibration/review: identity calibration won. Automatic far is below
  `0.3121748736511044`; `[0.3121748736511044, 0.5028396703865513)` requires review;
  automatic near begins at `0.5028396703865513`. The review band does not flip the
  underlying side decision.
- Identity: model fingerprint
  `85bc3325fbd43abba6ba3726ac091dc6a3a1eafc68d63d979cb2a7450eb49e06`.

Development cross-fit metrics at the learned side threshold are 94.1186% source-group
macro balanced accuracy, 94.0483% pooled balanced accuracy, 85.8289% worst-source-
group balanced accuracy, and 94.0604% accuracy. Near precision/recall are
94.7791%/93.0966%; far precision/recall are 93.3837%/95.0000%. At the frozen 95%
review operating point, selective accuracy is 95.1368%, coverage is 96.1052%, and
3.8948% of development rows require side-score review.

The hybrid serving decision is evaluated after the side score:

1. Inspect both frozen production serve heads inside ±1 second of the source-aligned
   anchor. Either head reaching its unchanged `0.85` threshold emits the side with
   source `serve-head`.
2. If both heads miss, emit the side only when the anchor is contained in an
   overlap-connected production rally interval marked `both-models`. Record source
   `production-rally-recovery` and require review.
3. Otherwise emit `not-serve` with source `none`.

On the current corrected-label 1,114-row results universe, the hybrid gate has
99.6324% serve precision and 98.0108% serve recall. It reduces missed serves from 56
to 22, recovers 35 rows (34 true serves and one false serve), and the side model is
correct on 30/34 recovered true serves. These are post-hoc assisted all-video results,
including a historically opened protected recording; they establish the selected
product behavior but are not clean held-out evidence for future model selection.

Authoritative artifacts are immutable NAS JSON documents:

| Role | Path | SHA-256 |
| --- | --- | --- |
| Development feature bank | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/serving-side-flight-v3/development.json` | `c4ddf9f94bec00ba0fa62f8267dc166fdeeb6f9419180acad2e7eadf4910af7d` |
| Fitted model, 237 names/parameters, selection, and development predictions | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v3-development.json` | `c867e8a2a141a5231fb1ceb5a0ba289457ac353fab6a9115dc672b84046f2414` |
| Frozen calibration/review policy | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v3-calibration-abstention-development.json` | `68e6429e104f1cd76464eb2c786ceebc88406301b6743d6e1236d26227393c77` |
| All-video development/unprotected features | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/serving-side-flight-v4/all-reviewed-inference.json` | `8bdf003a2fbb1295609e6051e14767614c4915006fcb2cfcac764d25bfdcb895` |
| Post-selection protected features | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/serving-side-flight-v4/protected-test.json` | `58fdb6b0aa7c57fc77ec4393334daf1637f5175b8d11b86e5b6c7e417308ce2e` |
| Hybrid gate evidence | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/serving-side-serve-gate-v2/all-reviewed.json` | `9ac4071f125030b0a8f1de037e48aa48b73f30f71fa6df8a302b70ed2db5344f` |
| Composed all-video inference consumed by the UI | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v3-hybrid-serve-gate-all-video-inference-v2.json` | `a7135cd509df3b0063e4634c99d314afb06d1df5552e292b5c17339a1671e722` |

The hybrid gate fingerprint is
`21395cc10390eefd14e120b3d7078191ddc9bb7ba1b6852fa4471658a7ff7af0`.
Detailed iteration history and the post-hoc limitation are retained in
[`serving-side-improvement-todo.md`](docs/research/serving-side-improvement-todo.md)
and
[`serving-side-hybrid-serve-gate-2026-08-21.md`](docs/research/serving-side-hybrid-serve-gate-2026-08-21.md).

### No-beach full-gold refits

The no-beach root is
`/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/models/`.
Every artifact below is a distinct fresh refit, not a pointer to the same-named
full-gold weights. The common change was to compare with the corresponding full-gold
artifact after removing both beach sources from fitting: fit `GOLD-NB-T4`, selection
`GOLD-V2`. Within-family feature changes and alias relationships are the same as in the
full-gold table, except that even same-named final exports belong to this no-beach fit.

| Family | Complete artifact names | Purpose/disposition |
| --- | --- | --- |
| Rally foundations | `pilot-baseline-v0`, `pilot-percentile-v1`, `full-percentile-v1`, `full-audiovisual-v2`, `full-audiovisual-v2-final`, `audiovisual-v2-targeted-pruned-diagnostic`, `full-audiovisual-audio-normalized-v3` | No-beach sensitivity for raw/percentile, F42→F90, pruning, and F90→F104 changes. Pilot artifacts fit `PILOT-NB-T4`, not `GOLD-NB-T4`; all others use the common full-video set. |
| Serve specialist | `serve-specialist-v1`, `serve-specialist-v2`, `serve-specialist-audiovisual-v3`, `serve-specialist-audiovisual-v4`, `serve-specialist-audio-normalized-v5`, `serve-specialist-audio-normalized-v6-no-legacy`, `serve-specialist-audio-normalized-v7-new-only` | No-beach refits of the serve feature/profile progression and normalized-audio ablations. |
| Rally with serve evidence | `full-audiovisual-serve-prob-stack-v1-control`, `full-audiovisual-serve-prob-stack-v1-exploratory`, `full-audiovisual-serve-peak-window-v1-exploratory` | Matched no-beach controls and rejected serve-stack/peak experiments. |
| Local dead-ball | `dead-ball-specialist-audiovisual-v1`, `dead-ball-specialist-audiovisual-v2`, `dead-ball-specialist-audio-normalized-v2-legacy-only`, `dead-ball-specialist-audio-normalized-v3`, `dead-ball-specialist-audio-normalized-v4-no-legacy`, `dead-ball-specialist-audio-normalized-v5-new-only` | No-beach refits of local boundary and audio-input studies. |
| Dead-state transition/global | `dead-state-transition-audio-normalized-v0-legacy-only`, `dead-state-transition-audio-normalized-v1-full`, `dead-state-transition-audio-normalized-v2-no-legacy`, `dead-state-transition-audio-normalized-v3-legacy-only-final`, `dead-state-transition-audio-normalized-v4-full-final`, `dead-state-transition-audio-normalized-v5-no-legacy-final`, `dead-state-global-audio-normalized-v1-full-final`, `dead-state-global-audio-normalized-v2-full-final` | No-beach refits/final exports of transition and inverse-rally dead-state comparisons. |

### Environment-specialist refits

These bundles are under `/mnt/freenas/volleycut/intake-2026-08-13/experiments/`.
Each bundle contains separately fitted rally, serve, and dead-state heads using F104.

| Bundle/head set | Fit sources | Compared with and change | Disposition |
| --- | --- | --- | --- |
| `environment-specialists-v1/models/all-labels/{rally,serve,dead-state}` | `ENV1-ALL8` | Compared with previous no-beach production components; refit all heads across all then-available grass/indoor labels with revised environment/hard-negative weighting. | Research candidate; predecessor to v2. |
| `environment-specialists-v1/models/grass-only/{rally,serve,dead-state}` | `ENV1-GRASS6` | Matched v1 environment specialization using grass sources only. | Research specialist; not current production. |
| `environment-specialists-v1/models/indoor-only/{rally,serve,dead-state}` | `ENV1-INDOOR2` | Matched v1 environment specialization using indoor sources only. | Research specialist; superseded by v2 indoor. |
| `environment-specialists-v2/models/all-labels/{rally,serve,dead-state}` | `ENV2-ALL11` | Compared with v1 all-labels; added `indoor-indoor-source-06`, `indoor-indoor-source-04`, and `indoor-indoor-source-08`, including walking/retrieval hard negatives, then refit all three heads. | Promoted as `model-1ca43e38eefc`. |
| `environment-specialists-v2/models/indoor-only/{rally,serve,dead-state}` | `ENV2-INDOOR5` | Compared with v1 indoor-only; expanded two indoor fitting sources to five with the three intake recordings. | Research specialist; not current production. |

No v2 grass-only bundle is present; do not synthesize one from the v1 artifact.

### Feedback suppression experiments

Both experiment roots use `FEEDBACK16` for fitting and hold the then-current production
decoder fixed. The five feedback projects in that set are raw user recordings with
corrected labels, not development-only metrics.

| Artifact/head | Compared with and change | Disposition |
| --- | --- | --- |
| `feedback-suppression-v3-2026-08-16/models/v3-candidate1-three-head/{rally,serve,dead-state}` | Compared with current production by refitting the three primary heads on the 11 environment sources plus five corrected-feedback sources. | Original candidate 1; not promoted. |
| `feedback-suppression-v3-2026-08-16/models/v3-candidate2-four-head/suppression` | Added a fourth learned suppression head to candidate 1. | Original candidate 2; superseded because overlap-positive target construction was corrected. |
| `feedback-suppression-v3-corrected-2026-08-18/models/v3-candidate1-three-head/{rally,serve,dead-state}` | Corrected experiment rerun; the three learned heads are weight-equivalent to the original candidate 1 because the correction affects only suppression targets. | Corrected comparison control; not promoted. |
| `feedback-suppression-v3-corrected-2026-08-18/models/v3-candidate2-four-head/suppression` | Compared with the original suppression head; excludes false-positive candidates that overlap included corrected rally cores. | Corrected candidate; learned weights match the promoted overlap-exclusion retrain. |
| `feedback-suppression-v3-2026-08-16/models/suppression-overlap-exclusion-retrained` | Standalone export of the corrected suppression target/fit, retaining production decoder behavior. | Promoted production suppression artifact. |

### Raw-video VLM adapters

| Artifact | Inputs and fit sources | Compared with and change | Disposition |
| --- | --- | --- | --- |
| `qwen3vl2b-unsloth-wsl-v1/seed-1729` | Qwen3-VL-2B LoRA adapter over 32-second raw-video windows; fit `GOLD-T6`, select/validate `GOLD-V2`; dataset also contains evaluation-only `GOLD-TEST1` | First recorded raw-video VLM training run in this family; no handcrafted F42/F90/F104 inputs. | Research only; checkpoint/adapter retained, not production. |
| `qwen3vl2b-unsloth-wsl-v1/seed-3407` | Same data, split roles, and RAW-VLM input contract | Matched second-seed comparison with seed 1729; changed random seed only. | Research reproducibility run; not production. |

## Report-only trained studies

These studies trained fold-local or temporary downstream heads but did not publish one
durable production artifact per fit. They are registered at study granularity.

| Study | Features and fit sources | Compared with and change | Result |
| --- | --- | --- | --- |
| [Audiovisual v2 feature ablation](docs/research/audiovisual-feature-ablation-2026-08-11.md) | F90 fold-local rally models; `GOLD-DEV8` | Removed predeclared feature groups from the full F90 reference to identify harmful/helpful groups. | Produced the targeted-pruned diagnostic; not promoted. |
| [Multiscale transition and interaction study](docs/research/feature-experiment-order-execution-2026-08-12.md) | F90 plus generated multiscale/interaction bank; `GOLD-DEV8` | Compared with F90 and component-selector controls; adds centered 0.5/1/2/4/8-second summaries and seven interactions. | 1,084-input combined development winner; retained research, not production. |
| [Rectangle court-relative study](docs/research/feature-experiment-order-execution-2026-08-12.md) | Court-relative reductions over existing grids; `GOLD-DEV8` | Compared orientation-invariant and fixed-endline variants with the transition reference. | Rejected; no variant passed the paired gate. |
| [Multistate and follow-up studies](docs/research/multistate-followup-execution-2026-08-12.md) | Fold-local DEAD/SETUP/SERVE/LIVE emissions and transition/boundary features; `GOLD-DEV8` | Compared a multistate sequence architecture and follow-ups with the frozen transition winner. | Research only; no production promotion. |
| [High-resolution MobileNetV2 study](docs/research/feature-experiment-order-execution-2026-08-12.md) | Frozen 1 Hz embeddings, projected crop features, and fold-local heads; `GOLD-DEV8` | Added four geometric high-resolution crops to the transition reference. | Rejected in every development source group. |
| [Component selector v1/v2](docs/research/feature-experiment-order-execution-2026-08-12.md) | Fold-local diagnostic selectors; `GOLD-DEV8` | v1 was superseded; v2 standardized the component diagnostic against the full reference. | Diagnostic only; not production. |
| [DINOv2 temporal development study](docs/research/dinov2-temporal-execution-2026-08-13.md) | Frozen DINOv2 ViT-S/14 tokens plus audiovisual boundary heads; `GOLD-DEV8` | Compared a learned temporal representation with the handcrafted transition reference. | Passed the development gate, but no checkpoint was published and protected test stayed unopened. |
| DINOv2 no-beach sensitivity | Same DINOv2 contract; fit `GOLD-NB-T4`, select `GOLD-V2` | Removed beach training sources to test environment sensitivity. | Research result only; no published checkpoint. |

The ball-presence pilot is not a VolleyCut-trained model entry: it evaluated a
third-party YOLOX artifact and did not train a downstream candidate after detector gates
failed. Fixed heuristics and decoders are likewise outside this registry.

## Source video catalog

Model rows use the stable IDs below. Paths are the actual source files used by the
training manifests on this machine; relocating the data does not change the recording
ID. For a future rebuild, verify the manifest's content hash rather than trusting only
the pathname. Historical manifests did not consistently record hashes, so this
reconstruction does not invent them.

### Early excerpt and pilot clips

| ID | Role-capable source video path |
| --- | --- |
| `v0-beach-JXM` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/labeled/beach/beach-source-02-deadstart25-duration62.mp4` |
| `v0-grass-qpd` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/labeled/grass/grass-source-09-deadstart2-duration88.mp4` |
| `v0-indoor-9lc` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/indoor/indoor-source-01-start340-duration90.mp4` |
| `v0-grass-rSs` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/labeled/grass/grass-source-10-deadstart6-duration84.mp4` |
| `v0-test-indoor-tds` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/labeled/indoor/indoor-source-05-deadstart41-duration49.mp4` |
| `pilot-beach-JXM` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/beach/beach-source-02-start358-duration90.mp4` |
| `pilot-beach-ey` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/beach/beach-source-01-start291-duration90.mp4` |
| `pilot-grass-Dm` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/grass/grass-source-04-start344-duration90.mp4` |
| `pilot-grass-GYU` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/grass/grass-source-01-start465-duration90.mp4` |
| `pilot-grass-qpd` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/grass/grass-source-09-start366-duration90.mp4` |
| `pilot-grass-rSs` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/grass/grass-source-10-start325-duration90.mp4` |
| `pilot-indoor-9lc` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/indoor/indoor-source-01-start340-duration90.mp4` |
| `pilot-indoor-Y9` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/indoor/indoor-source-07-start304-duration90.mp4` |
| `pilot-test-indoor-tds` | `/mnt/freenas/volleycut/v0-2026-08-09/normalized/indoor/indoor-source-05-start387-duration90.mp4` |

### Fully labeled proxy recordings

| ID | Source video path |
| --- | --- |
| `beach-beach-source-02` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/beach/beach-source-02.mp4` |
| `beach-beach-source-01` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/beach/beach-source-01.mp4` |
| `grass-grass-source-04` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/grass/grass-source-04.mp4` |
| `grass-grass-source-01` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/grass/grass-source-01.mp4` |
| `grass-grass-source-09` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/grass/grass-source-09.mp4` |
| `grass-grass-source-10` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/grass/grass-source-10.mp4` |
| `indoor-indoor-source-01` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/indoor/indoor-source-01.mp4` |
| `indoor-indoor-source-07` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/indoor/indoor-source-07.mp4` |
| `test-indoor-indoor-source-05` | `/mnt/freenas/volleycut/labeling-v1-2026-08-09/proxies/indoor/indoor-source-05.mp4` |

### Intake additions and corrected-feedback recordings

| ID | Source video path |
| --- | --- |
| `grass-grass-source-03` | `/mnt/freenas/volleycut/intake-2026-08-13/proxies/grass/grass-source-03.mp4` |
| `grass-grass-source-05` | `/mnt/freenas/volleycut/intake-2026-08-13/proxies/grass/grass-source-05.mp4` |
| `indoor-indoor-source-06` | `/mnt/freenas/volleycut/intake-2026-08-13/proxies/indoor/indoor-source-06.mp4` |
| `indoor-indoor-source-04` | `/mnt/freenas/volleycut/intake-2026-08-13/proxies/indoor/indoor-source-04.mp4` |
| `indoor-indoor-source-08` | `/mnt/freenas/volleycut/intake-2026-08-13/proxies/indoor/indoor-source-08.mp4` |
| `project-cmh0xj` | `/mnt/freenas/volleycut-raw-no-backup/PXL_20260816_193307688.mp4` |
| `project-kqx9c` | `/mnt/freenas/volleycut-raw-no-backup/PXL_20260816_171720964.mp4` |
| `project-qkf86k` | `/mnt/freenas/volleycut-raw-no-backup/PXL_20260816_161923155.mp4` |
| `project-vxcbv3` | `/mnt/freenas/volleycut-raw-no-backup/PXL_20260816_203801418.mp4` |
| `project-yf69sz` | `/mnt/freenas/volleycut-raw-no-backup/PXL_20260816_190429172.mp4` |

## Source-set definitions

Only IDs in a set's **fit** definition affect learned parameters. Selection and test
sets are separate even when listed on the same model row.

| Set | Exact members and role |
| --- | --- |
| `V0-T3` | Fit: `v0-beach-JXM`, `v0-grass-qpd`, `v0-indoor-9lc` |
| `V0-NB-T2` | Fit: `v0-grass-qpd`, `v0-indoor-9lc` |
| `V0-V1` | Selection: `v0-grass-rSs` |
| `V0-TEST1` | Protected test only: `v0-test-indoor-tds` |
| `PILOT-T6` | Fit: `pilot-beach-JXM`, `pilot-beach-ey`, `pilot-grass-GYU`, `pilot-grass-qpd`, `pilot-indoor-9lc`, `pilot-indoor-Y9` |
| `PILOT-NB-T4` | Fit: `pilot-grass-GYU`, `pilot-grass-qpd`, `pilot-indoor-9lc`, `pilot-indoor-Y9` |
| `PILOT-V2` | Selection: `pilot-grass-Dm`, `pilot-grass-rSs` |
| `PILOT-TEST1` | Protected test only: `pilot-test-indoor-tds` |
| `GOLD-T6` | Fit: `beach-beach-source-02`, `beach-beach-source-01`, `grass-grass-source-01`, `grass-grass-source-09`, `indoor-indoor-source-01`, `indoor-indoor-source-07` |
| `GOLD-NB-T4` | Fit: `grass-grass-source-01`, `grass-grass-source-09`, `indoor-indoor-source-01`, `indoor-indoor-source-07` |
| `GOLD-V2` | Selection: `grass-grass-source-04`, `grass-grass-source-10` |
| `GOLD-DEV8` | Development folds only: all six `GOLD-T6` plus both `GOLD-V2`; each report's held-out source-group fold is excluded from that fold's fit |
| `GOLD-TEST1` | Protected test only: `test-indoor-indoor-source-05` |
| `ENV1-GRASS6` | Fit: `grass-grass-source-04`, `grass-grass-source-01`, `grass-grass-source-09`, `grass-grass-source-10`, `grass-grass-source-03`, `grass-grass-source-05` |
| `ENV1-INDOOR2` | Fit: `indoor-indoor-source-01`, `indoor-indoor-source-07` |
| `ENV1-ALL8` | Fit: union of `ENV1-GRASS6` and `ENV1-INDOOR2` |
| `ENV2-INDOOR5` | Fit: `ENV1-INDOOR2` plus `indoor-indoor-source-06`, `indoor-indoor-source-04`, `indoor-indoor-source-08` |
| `ENV2-ALL11` | Fit: union of `ENV1-GRASS6` and `ENV2-INDOOR5` |
| `FEEDBACK16` | Fit: `ENV2-ALL11` plus `project-cmh0xj`, `project-kqx9c`, `project-qkf86k`, `project-vxcbv3`, `project-yf69sz` |
| `SIDE4` | Fit: `beach-beach-source-02`, `beach-beach-source-01`, `grass-grass-source-01`, `grass-grass-source-09` |

## Rebuild checklist

Before claiming a model has been reproduced:

1. Resolve every source-set ID to the exact file and verify its content hash.
2. Verify label/ignored-interval revision and split role; never open a protected test
   split during iteration selection.
3. Rebuild the exact ordered features under the artifact's immutable feature profile.
4. Reproduce learned preprocessing, target construction, class/hard-negative weighting,
   seed, optimizer, and calibration.
5. Record both weight and full-artifact hashes. A metadata-only final alias should say
   that it reuses weights; a refit should never do so.
6. Evaluate using the canonical ranking contract in
   [`docs/model-ranking-metric.md`](docs/model-ranking-metric.md), including all required
   padding cases and the declared development scope.
7. Add the new row here in the same change as the durable artifact, including its
   predecessor, intended improvement, actual change, and disposition.
