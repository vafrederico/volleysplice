**Grass and indoor neural experiment — 2026-09-19**

The user has narrowed the current scope to grass and indoor volleyball. This follow-up excludes beach from training, fitted normalization, inner checkpoint/decoder selection, and evaluation. It preserves the [first experiment](./neural-development-execution-2026-09-18.md) and changes the dataset scope only. The [frozen protocol](./neural-nonbeach-protocol-2026-09-19.json) records the model/seed budget, target metric and feasibility screen before the new fits.

**Completed result.** DINO+TCN has the highest mean development `F1_padP_coreR` in this restricted experiment, at 0.8741, followed by the compact TCN at 0.8628. Both substantially exceed the same-population linear control, but every neural arm still exceeds the allowed mean recall regression. No model earns production replacement. The larger-corpus audit prompted by the user is a separate next step; these results remain tied to the six-recording scope below.

| Model | Mean `F1_padP_coreR` at ±2 s | Mean `R_core` | F1 gain vs linear | Recall change | Mean complete rally losses |
|---|---:|---:|---:|---:|---:|
| Linear | 0.7502 | 0.9422 | — | — | 18.7 |
| MLP | 0.7988 | 0.9308 | +0.0486 | -0.0114 | 18.3 |
| Compact TCN | 0.8628 | 0.9269 | +0.1126 | -0.0153 | 20.0 |
| DINO + TCN | **0.8741** | 0.9306 | +0.1239 | -0.0116 | 22.3 |

Every neural arm improves F1 in all three seeds and all three mean paired source-group comparisons. Complete-loss counts are means across the three runs, out of 235 original rallies. They cannot replace identity-based checks: recovering one rally does not excuse losing another. Event F1 is 0.4099/0.4248/0.6016/0.5908 for linear/MLP/TCN/DINO+TCN respectively. The TCN's strict short/ace/service-fault recall is 0.2941/0.3467/0.3438, versus linear's 0.1046/0.2933/0.1250. Better event matching still coexists with losses in retained core time.

All required padding cases follow. Each run first pools durations across six recordings; cells below then average the three separately pooled run values. The two-second case alone ranks iterations.

| Model | Before/after padding, s | `P_pad` | `R_core` | `F1_padP_coreR` | Model export, s | Human export, s | Difference, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear | 0 | 0.5622 | 0.8159 | 0.6656 | 2577.6 | 1775.8 | +801.7 |
| Linear | 1 | 0.5891 | 0.9099 | 0.7151 | 3288.9 | 2245.8 | +1043.1 |
| Linear | 2 | 0.6233 | 0.9422 | 0.7502 | 3844.7 | 2715.8 | +1128.9 |
| Linear | 3 | 0.6693 | 0.9616 | 0.7892 | 4293.1 | 3187.9 | +1105.2 |
| MLP | 0 | 0.6589 | 0.7700 | 0.7101 | 2075.3 | 1775.8 | +299.5 |
| MLP | 1 | 0.6709 | 0.8861 | 0.7636 | 2783.3 | 2245.8 | +537.5 |
| MLP | 2 | 0.6997 | 0.9308 | 0.7988 | 3338.8 | 2715.8 | +623.0 |
| MLP | 3 | 0.7269 | 0.9537 | 0.8250 | 3849.6 | 3187.9 | +661.7 |
| TCN | 0 | 0.7517 | 0.8026 | 0.7762 | 1896.8 | 1775.8 | +120.9 |
| TCN | 1 | 0.7816 | 0.8904 | 0.8323 | 2401.9 | 2245.8 | +156.1 |
| TCN | 2 | 0.8071 | 0.9269 | 0.8628 | 2882.0 | 2715.8 | +166.2 |
| TCN | 3 | 0.8259 | 0.9461 | 0.8818 | 3353.2 | 3187.9 | +165.3 |
| DINO + TCN | 0 | 0.7849 | 0.8215 | 0.8001 | 1871.4 | 1775.8 | +95.6 |
| DINO + TCN | 1 | 0.8060 | 0.8976 | 0.8473 | 2371.8 | 2245.8 | +125.9 |
| DINO + TCN | 2 | 0.8268 | 0.9306 | 0.8741 | 2841.2 | 2715.8 | +125.4 |
| DINO + TCN | 3 | 0.8432 | 0.9455 | 0.8902 | 3303.9 | 3187.9 | +116.0 |

Execution checklist:

- [x] Freeze the exact consented grass/indoor subset and verify inherited provenance.
- [x] Re-pool old predictions on the reduced scope as a descriptive reference.
- [x] Refit the same four models with three fixed seeds and nested group selection.
- [x] Recompute production references on the same six-recording scope.
- [x] Verify metrics, coverage losses and selection limits; publish the decision.

The scope contains six recordings, 235 rallies and 6,122.47 seconds (1.701 hours) from three source groups: two grass sessions and one indoor session. Every retained row exactly matches its parent, including gold labels, consent, ignored intervals and cache lineage. Kept label/provenance/cache files and complete proxy bytes were hash-verified. Outer fits train on two groups; each inner fit has only one training group and the other for validation. That is a stricter and noisier selection setting than the original four-group experiment. These recordings have already informed development; neither the user-directed scope change nor seed repeats create fresh independent test evidence.

The architecture, current audiovisual features, frozen DINO caches, losses, optimizer, checkpoint epochs and decoder grid remain unchanged. The compact TCN has 64 hidden channels in each of five temporal blocks, three output heads and 29,635 trainable parameters. Models are ranked by pooled `F1_padP_coreR` at two seconds before/after, with all four required padding cases reported and positive gaps joined only when strictly below three seconds.

The pre-fit screen requires a mean score gain of at least 0.02 over the matched linear control, at least two positive seeds, a positive mean paired gain in at least two of three groups, and mean core-recall regression no worse than 0.005. Coverage identities remain separate deployment guardrails. The previous observation that linear thresholds hit the grid ceiling is retained as a limitation; widening the grid during this scope-only comparison would confound two changes.

All eight bound model/training/evaluation module hashes match the first study. The new manifest and a copy of the scope protocol were written before fitting under `private-reference-0157`:

- `manifest.json`: SHA-256 `04365c5fe26a3186acad90cba5b4f595588d17d24240251c26a2f9b2aab60532`.
- `scope-protocol.json`: SHA-256 `48e518acedb18b8abcb5068d965d1e60aaddc28ee7b73a8f56b3c17044091b8b`.
- `dataset-audit.json`: inherited row identities, full input verification and excluded beach recordings, created by [the subset preparation script](../../scripts/prepare-neural-nonbeach.py).

The unchanged production-baseline module completed CPU refits on this scope. At two-second padding, `F1_padP_coreR` is 0.69955 for the previous bundle, 0.78222 for all-labels v2, and 0.73278 for their union. The union has `P_pad=0.58531`, `R_core=0.97959` and 4,417.1 seconds of export versus 2,715.8 seconds for padded human ranges. Shipped-union retrospective replay is 0.81910 and remains training/selection-exposed. Here, the v2 refit uses the same outer-fitting recordings as the neural models; the previous component retains its narrower historical allowlist. The targets, fitting recipe and fixed historical decoder still differ, so the matched nested linear arm remains the capacity control. Artifact: `production-baseline/baseline.json`, SHA-256 `fac994e8368b05df146cd216ec9aa674a30e889a1bb65efeb25854a4baabc14e`.

**Scope-only rescore versus actual refit.** Removing beach only from old evaluation rows gives F1/core recall of 0.7860/0.9656 (linear), 0.8058/0.9543 (MLP), 0.8918/0.9518 (TCN), and 0.8634/0.8788 (DINO+TCN). Those models still used beach in fitting and inner selection. They are descriptive references, not the beach-free models in the main table. Comparing either table with the original eight-recording pooled score would also mix evaluation populations. Changes between these two six-recording tables combine less fitting data with different inner-fold selection; they do not isolate the causal value of beach imagery alone. The [rescore script](../../scripts/rescore-neural-scope.py) preserves and replays the original predictions; `prior-run-rescore/summary.json` SHA-256 is `1056e64488629be03fd450be4c73a2bdc71a43412aee9b7aa9bd982794556114`.

**Remaining failures.** The fixed-seed-3407 review pack (ledger `private-reference-0158`) contains six contact sheets/30 frames. The largest additional losses include a 7.25-second internal gap in `indoor-source-01` rally 160.807–190.448, another indoor rally with 6.535 seconds lost, and a 5.576-second truncated grass-rally tail. Four newly complete losses in this seed include two aces and a service fault. The other two seeds have ten and seven newly complete losses versus their linear controls. These are failures after two-second export padding, not merely different raw boundaries. JSON SHA-256: `9fb780980e1304bb507b1db552efd2baac1b6a25dc303bd1110e1f2e571eb0ff`.

**Verification and artifacts.** All 108 fits finished: 72 inner fits and 36 outer refits. An independent audit verified all 648 saved prediction/weight hashes and finite tensors, recomputed each scaler from its fitting records, and replayed all twelve final decoded evaluations exactly. All retained fold IDs are inside the six-recording scope, with no held-group leakage. The parent and new training contracts differ only in manifest hash and source groups. The summary now derives group/video/seed counts from the verified manifest instead of the original fixed counts; six summary regression tests pass, including the new three-group scope case. No architecture/runtime port changed, so the prior browser portability result remains engineering context rather than a qualification of these new weights.

The linear arm again selects the upper entry-threshold edge, 0.65, in all nine outer selections. It reaches epoch 30 in three selections; TCN chooses epoch 20 in eight and epoch 10 in one. The linear-calibration search limit therefore remains unresolved. Summed per-fit timers are 252.4 seconds, excluding initial feature loading and decoder searches; maximum PyTorch tensor allocation is 249.4 MiB, excluding CUDA context and encoder extraction.

- `nested-study-v1/report.json`: SHA-256 `d3b872376ce5e4cae381ecbef1747e08f776d44f89deab10ca7adf359d303c76`.
- Per-seed metrics, all padding cases, production references and guardrails (ledger `private-reference-0159`): SHA-256 `21d0fd1bb5e3f0248b8be0f6f8119818c630b1065a6df2f76baf4163c05dba92`.
- `nested-study-v1/summary.json`: SHA-256 `f00d9ebed2ebf501c7794deed7b64af1d703dbe333f5254e1ec87d586e8b82b6`.

Reproduction from the repository in the existing WSL neural environment, using a new destination:

```bash
private-reference-0110 -u \
  scripts/run-neural-development.py \
  --manifest private-reference-0160 \
  --output private-reference-0161
```

**Decision:** retain grass/indoor as the current product research scope, keep production unchanged, and prioritize [the verified larger training cohorts](./neural-corpus-expansion-2026-09-19.md) and coverage-preserving selection before increasing network size. The follow-up corpus audit identifies eight completed exact-label recordings and eleven with historically reviewed drafts; reusable AV caches already exist. Neither a higher score nor the removal of beach makes the current recall losses acceptable. No protected test was opened.
