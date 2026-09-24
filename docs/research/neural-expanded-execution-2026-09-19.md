**Expanded grass/indoor supervision experiment — 2026-09-19**

The expanded run completed all 288 fits on the RTX 3080. Additional reviewed drafts improved the compact TCN's development F1, but reduced retained playing time. Adding the manually reviewed exports did not improve average F1 further. Every neural cohort failed the predeclared core-retention comparison against its matched linear control, so this study does not support replacing production.

The user confirmed that project exports were all manually reviewed, including kept and discarded sections, and authorized using more videos and continuing execution. Completed rally labels, reviewed drafts and reviewed export decisions retain different target meanings.

These are means of three seed runs. Within each seed, metric numerators and denominators are pooled over the same eight held-source recordings / 322 exact rallies. The declared product padding is two seconds before and after each rally, with gaps strictly below three seconds retained.

| Training cohort | Model | P_pad | R_core | F1_padP_coreR | Export seconds | Difference from human export |
|---|---|---:|---:|---:|---:|---:|
| Exact 8 | Linear | .6027 | .9797 | .7462 | 5,797.967 | +2,122.816 |
| Exact 8 | TCN | .8166 | .9456 | .8763 | 4,021.708 | +346.557 |
| Exact + draft, 11 | Linear | .6516 | .9612 | .7766 | 5,146.091 | +1,470.940 |
| Exact + draft, 11 | TCN | .8896 | .9130 | .9011 | 3,480.808 | -194.343 |
| Exact + draft + export, 18 | Linear | .6516 | .9612 | .7766 | 5,146.091 | +1,470.940 |
| Exact + draft + export, 18 | TCN | .8747 | .9176 | .8956 | 3,568.533 | -106.618 |

The padded human export is 3,675.151 seconds in every row. Draft augmentation raises TCN F1 by 2.478 percentage points while reducing core recall by 3.267 points. Export augmentation changes F1 by -0.556 points and core recall by +0.468 points relative to the draft TCN. Adding export supervision leaves the linear control exactly unchanged in every seed, including selection, predictions and evaluation. Its independent output rows cannot transfer keep supervision into live/serve/end predictions.

Draft TCN augmentation improves F1 in all three seeds and all four groups, but fails the allowed recall-regression limit of .005. Export augmentation improves F1 in only one seed and no group on mean paired comparisons. All nine registered comparisons fail the development screen. All 72 selected inner candidates satisfy their .95 recall eligibility constraint; this does not guarantee recall on the outer source groups. No padding value or seed was chosen from outer results.

Rally retention gives a stronger reason to reject these production candidates than the aggregate score alone. Out of 322 original rallies, the exact TCN completely loses 14-27 rallies per seed after product padding; the draft TCN loses 29-42 and the export TCN 34-43. Mean completely/partially lost counts are 21.00/31.33, 35.33/43.00 and 37.67/38.33 respectively. Fewer partial losses must not be read as an improvement when some rallies become completely lost. Production pointers, app behavior and the protected test set remain unchanged.

Short points need explicit protection. The 49 rallies lasting at most two seconds are 15.22% of points but only 3.16% of core time. Missing all of them while keeping all other play would still give .9684 duration recall. All have at least four valid feature ticks, so these failures cannot be explained by zero sampled frames alone. The final diagnostic replays every result with the same gold/ignored revision:

| TCN training cohort | Completely lost <=3s points, out of 73 | <=3s core recall | Completely lost tagged aces, out of 25 | Completely lost tagged faults, out of 32 |
|---|---:|---:|---:|---:|
| Exact | 14.33 | .8039 | 4.00 | 7.33 |
| Exact + draft | 22.33 | .6704 | 5.67 | 13.00 |
| Exact + draft + export | 23.33 | .6541 | 7.67 | 13.33 |

Counts are means across seeds after fixed two-second export padding; outcome tags and duration slices overlap. Longer rallies also regress: core recall for points above three seconds is .9541/.9275/.9334 across these TCN cohorts. The diagnosis therefore does not establish a single short-point cause. Exact loss identities and raw-versus-padded coverage are preserved under `short-event-diagnostic/results/` and `study/loss-identities.json`.

The next bounded experiment should test per-rally weighting of live supervision with unchanged inputs, capacity, source folds and exact exposure. It should predeclare short-point and original-rally retention guardrails in addition to the existing duration metric, and compare against these frozen baselines. That is a hypothesis about the objective, not a demonstrated fix. Feature or decoder changes should follow separately if the lost-point evidence warrants them. More exact serve/end labels from additional sessions would also strengthen evaluation and boundary training; increasing network width alone is not supported by this study. No such follow-up was run here.

Execution checklist:

- [x] Verify 18 recordings, cache identities, annotation provenance and source families.
- [x] Freeze exact8 / draft3 / reviewed-export7 supervision tiers.
- [x] Implement a four-head research model and matched exposure training.
- [x] Complete preflight tests and independent implementation review (66 tests passed).
- [x] Train all three cohorts, two architectures and three seeds with nested source-group validation.
- [x] Audit predictions, checkpoints, group separation and paired exact exposure.
- [x] Report all padding cases, paired improvements and rally retention failures.

The dataset is frozen under `private-reference-0138`. `manifest.json` SHA-256 is `48fe4fb5bcb74b2ee998562f61c3a7e8df43e921f8882e1b44fe1c12db5d451f`; its standard `exact-manifest.json` is `d582e7ae2f75136bf0419f97466e36d4929004a8474c90146b20530c8f311c16`. All 18 caches contain the current 104 audiovisual signals. The prior immutable studies are retained separately.

| Cohort | Training supervision available | Exact evaluation population |
|---|---|---|
| `exact` | 8 completed recordings, 322 exact rallies, 4 source groups | Same 8 recordings |
| `draft` | Exact cohort + 3 continuously reviewed draft recordings, 123 approximate rally ranges | Same 8 recordings |
| `reviewed_export` | Previous cohort + 7 reviewed raw/project pairs, 252 actual final export ranges | Same 8 recordings |

The largest cohort has 18 recordings across seven groups, totaling 5.319 hours of source video. All seven reviewed exports belong to one Aug29 source group. Each outer fold excludes one of the four exact groups from every supervision tier; inner validation also excludes its group from every tier. Derivatives remain in the same source family. Drafts and exports never supply selection or evaluation labels. Beach and protected `source-group-008` footage are excluded. Four development groups still give limited evidence about unseen venues and capture conditions.

The model consumes the same 104 cached audiovisual signals at approximately 4 Hz. The linear control has 2,084 parameters; the compact TCN has 64 hidden channels, five residual temporal blocks and 29,700 parameters. Both output live, serve, end and auxiliary keep logits. Only the first three feed the existing decoder. The extra keep head lets reviewed exports train the TCN's shared representation without treating padded cuts as exact rally boundaries. The linear head has no shared learned representation and provides a useful negative control for this transfer.

Completed labels supervise live/serve/end with weights 1/.5/.5 and a canonical ±2-second, strict-gap-under-3-second keep target weighted .25. Drafts supervise only live with weight .5; all ticks within one second of approximate rally boundaries or ignored boundaries are masked. Fully censored short rallies do not become negative examples. Exports supervise only keep with weight .25, using actual `finalExportIntervals` without another padding or joining pass. Their valid universe is `gameWindow` minus ignored time. All seven game windows cover their videos apart from submillisecond rounding that excludes no cached tick.

At equal epoch prefixes, every optimizer step contains the same exact batch across the three cohorts. Adding auxiliary batches does not replace exact exposure or increase optimizer steps for a fixed epoch count; selected outer refit lengths can differ. Group/recording-balanced replacement sampling, exact dropout and draft dropout each have separate deterministic random streams. Auxiliary batches add compute, which is recorded. Each head's loss divides by its own valid tick count before applying its fixed weight. Scalers and positive class weights use exact fitting rows only. Linear gradients are clipped independently per output head so keep-only gradients cannot rescale primary-head updates; TCN uses global norm clipping at 1. AdamW uses learning rate .001 and weight decay .0001. Inputs use real temporal halos and reset at ignored spans; artificial input padding is avoided.

Before any new outcomes are examined, all arms receive checkpoints at epochs 5/15/30/60 and the same decoder grid: entry thresholds .2/.35/.5/.65/.8/.9, smoothing .5/1 second, minimum live .25/1 second, and boundary snapping disabled/enabled. Selection pools exact inner-fold durations. It first requires `R_core >= .95`, then ranks eligible candidates by `F1_padP_coreR` at declared ±2-second padding. If no candidate qualifies, it retains highest F1 and explicitly flags infeasibility. It never selects on an outer fold. Stable ties favor the first checkpoint/grid entry.

This is a new common training/selection recipe, following the earlier recall failures and linear grid saturation. Differences between these three new cohorts measure augmentation under that recipe. Differences from the earlier six-video, three-head study combine population, objective and selection changes and cannot isolate the effect of more data. The previously explored MLP and DINO branches are not retrained here; only six exact videos currently have aligned DINO caches.

The predeclared comparative screen requires mean F1 gain of at least .02 over the matched linear control, at least two positive seeds, a positive mean paired change in a majority of the four groups, and mean core-recall regression no worse than .005. Report complete/partial original-rally loss identities, event F1 and short/ace/service-fault diagnostics independently. Passing the pooled screen does not approve production. All results include 0/1/2/3-second symmetric padding, pooled precision/recall/F1, export durations and differences; ignored spans are subtracted after padding/joining and never rejoined.

The RTX 3080 runs training in the existing WSL PyTorch environment. There are 216 inner fits and 72 outer refits, producing 18 pooled evaluations. Code hashes, dataset hash, environment and exact protocol are written to `study/preregistration.json` before fitting. Each fit saves tensor-only checkpoints, raw probabilities, full input membership, per-head supervised counts, optimizer steps and cumulative exposure hashes for every epoch. Independent verification can compare equal-epoch prefixes even when outer refits select different lengths.

Eleven additional Aug16 raw/project pairs remain deferred because their available embedded features use the Android native backend. Three sampled frames show grass, but all-source environment checks and feature regeneration/equivalence are still needed. Manual review authorization is already established and is not the blocker.

**Training-only output removal.** A predetermined engineering checkpoint (`reviewed_export/tcn`, seed 20260918, outer 0 / inner 0, epoch 5) was qualified independently of model ranking. [The converter](../../scripts/trim-neural-auxiliary-head.py) removes only the fourth output row, preserving every shared weight and scaler tensor. The inference-only derivative has 29,635 parameters. [ONNX CPU qualification](../../scripts/qualify-neural-primary-derivative.py) passed 38 comparisons on three actual inner-validation recordings, covering full sequences, real-halo chunks, the original four-head network's first three outputs, the trimmed network, ONNX Runtime, and saved GPU probabilities. Maximum probability error was 1.79e-7.

The self-contained ONNX graph is 126,331 bytes (SHA-256 `6909f927cd2259ccc1158be982f8b635ad9efc93efa7037afa61a21403b4d43d`). Prepared-feature inference for 252 ticks took 0.264 ms median / 0.416 ms p95 on one Ryzen 5900X CPU thread; feature extraction, decoding and UI work are excluded. Artifacts are under `engineering-onnx-cpu/reviewed_export-tcn-seed20260918-outer0-inner0-epoch5/`, with the derivative's provenance under `engineering-primary-head-derivative/`. These measurements do not establish browser/phone performance and do not select or promote a model.

**Production references on the same exact scope.** The unchanged production-baseline runner completed CPU refits and replayed all six references with the same 322 rallies and ignored-range revision. At ±2 seconds, source-held v2 reaches F1/core recall .7837/.9462; the source-held previous+v2 union reaches .7352/.9832. The shipped union replay reaches .8015/.9964 but is exposed to historical training and selection. These descriptive references retain their historical decoder and fitting recipe; neural checkpoint/decoder selection continues to use only the registered inner exact folds and matched linear controls. The previous-model allowlist contains four of the eight exact videos, while v2 uses all eight eligible videos. No auxiliary labels enter these references.

`production-baseline/baseline.json` SHA-256 is `4c7dc5d0b23cc1e9d749fb93e4b21f1ec20b4a28fe46a90976037643e161baf0`; `reference-summary.json` contains all four padding cases, exact-scope caveats and verification of 24 refitted head artifacts. An immutable `production-baseline-input/` adapter holds a byte-identical exact manifest and an audit binding its hash, because the legacy runner expects that hash under a generic `manifestSha256` key. Original manifests, audits and analysis modules remain unchanged.

**Required padding sensitivity.** These are means of separate seeds, each pooled over recordings. Every row uses its original selected predictions; the ranking case remains two seconds.

| Cohort | Model | Before/after s | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Exact | Linear | 0 | .5261 | .9060 | .6654 | 4114.9 | 2387.2 | +1727.8 |
| Exact | Linear | 1 | .5607 | .9649 | .7092 | 5072.2 | 3031.2 | +2041.1 |
| Exact | Linear | 2 | .6027 | .9797 | .7462 | 5798.0 | 3675.2 | +2122.8 |
| Exact | Linear | 3 | .6531 | .9864 | .7859 | 6347.2 | 4323.3 | +2023.9 |
| Exact | TCN | 0 | .7658 | .8525 | .8066 | 2658.7 | 2387.2 | +271.6 |
| Exact | TCN | 1 | .7951 | .9217 | .8536 | 3353.9 | 3031.2 | +322.8 |
| Exact | TCN | 2 | .8166 | .9456 | .8763 | 4021.7 | 3675.2 | +346.6 |
| Exact | TCN | 3 | .8353 | .9596 | .8931 | 4676.4 | 4323.3 | +353.1 |
| Draft | Linear | 0 | .5895 | .8637 | .7006 | 3498.5 | 2387.2 | +1111.3 |
| Draft | Linear | 1 | .6157 | .9374 | .7432 | 4415.5 | 3031.2 | +1384.3 |
| Draft | Linear | 2 | .6516 | .9612 | .7766 | 5146.1 | 3675.2 | +1470.9 |
| Draft | Linear | 3 | .6899 | .9725 | .8072 | 5788.2 | 4323.3 | +1464.9 |
| Draft | TCN | 0 | .8517 | .8065 | .8285 | 2260.4 | 2387.2 | -126.7 |
| Draft | TCN | 1 | .8732 | .8825 | .8778 | 2880.8 | 3031.2 | -150.3 |
| Draft | TCN | 2 | .8896 | .9130 | .9011 | 3480.8 | 3675.2 | -194.3 |
| Draft | TCN | 3 | .9020 | .9288 | .9152 | 4082.0 | 4323.3 | -241.3 |
| Export | Linear | 0 | .5895 | .8637 | .7006 | 3498.5 | 2387.2 | +1111.3 |
| Export | Linear | 1 | .6157 | .9374 | .7432 | 4415.5 | 3031.2 | +1384.3 |
| Export | Linear | 2 | .6516 | .9612 | .7766 | 5146.1 | 3675.2 | +1470.9 |
| Export | Linear | 3 | .6899 | .9725 | .8072 | 5788.2 | 4323.3 | +1464.9 |
| Export | TCN | 0 | .8341 | .8228 | .8282 | 2355.9 | 2387.2 | -31.3 |
| Export | TCN | 1 | .8567 | .8908 | .8733 | 2968.7 | 3031.2 | -62.5 |
| Export | TCN | 2 | .8747 | .9176 | .8956 | 3568.5 | 3675.2 | -106.6 |
| Export | TCN | 3 | .8900 | .9310 | .9100 | 4156.9 | 4323.3 | -166.3 |

**Completed verification and artifacts.** The main runner exited successfully. Independent audits verified all 288 fit memberships, 1,872 NPZ hashes, 936 finite and correctly shaped checkpoints, and bit-exact recomputation of all exact-only scalers. Paired exposure verification covered 192 fit prefixes / 9,715 epoch prefixes; 312 linear primary-trace pairs matched exactly. All 18 evaluations and 72 selected inner candidates replayed correctly. All 16 precomputed selections matched main-run selections; the 15 reused refits retained their original hashes. Current and both previous studies' registered source hashes still match. All workers have exited.

Artifacts below are relative to `private-reference-0138`. The full summary (ledger `private-reference-0139`) includes every seed, paired group direction, event metric and training runtime. Training fit durations sum to about 1.40 process hours, with at most 108.8 MiB of recorded PyTorch peak allocation per fit; overlapping workers, CUDA/runtime overhead, feature preparation and selection overhead make this different from elapsed wall time or total GPU memory.

| Artifact | SHA-256 |
|---|---|
| `study/report.json` | `5371829b616bd46fea07446662fc3bb2b53c2a1221e3455e1e2aedd94882b4fc` |
| `study/summary.json` | `e2adfe24cd9cad42c6c918d73dbc0b74d9830fa2a9bb2b4b4a699792ceed4bf3` |
| `study/summary.md` | `454246758b3e34456b5b52dcc82462d29907ee50149cbad46077bb3f7b3d6539` |
| `study/loss-identities.json` | `4aaf61cf9c5354f2c0386523280264e31189c404d3b698155e3839fa5b2ec061` |
| `tensor-scaler-audit-v1.json` | `274c30cd1e9a92cefd59f390079951d00983b811eb6da6848f82e8482a311810` |
| `precompute-main-selection-agreement-v1.json` | `6b9ca1d21feba041ee82769894aaebf758ad226f1d50accce384e6a84479bade` |
| `short-event-diagnostic/results/report.json` | `1cb2735760dfc36d5eec925e22a0470e5b839758c80964518878fd0dfeef93ca` |

The registered contract hash is `63d1553cd412113112012140abda656ec09d6ca761d7e0d32d8d776674200806`. The bounded source snapshot is stored separately under `source-snapshot/`, with its archive hash and per-file manifest in `verification.json` and `snapshot-manifest.json`.

Reproduce into a new directory from the repository (the completed `study` artifacts are immutable):

```bash
private-reference-0110 -u -m analysis.neural_expanded_development \
  --manifest private-reference-0134 \
  --output private-reference-0140
```
