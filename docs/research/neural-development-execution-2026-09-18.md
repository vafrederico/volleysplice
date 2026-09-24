**Neural development execution — 2026-09-18**

This executes the first stage of the [neural proposal](./neural-network-investigation-and-proposal-2026-09-18.md). Production assets and labels remain unchanged. Later mobile visual training, new annotation work, and motion-video desktop extensions are separate stages; this first run establishes whether compact nonlinear/temporal heads and existing DINO representations improve the current feature baseline.

**Result: temporal modeling improves the development score, but none of these neural candidates is acceptable as a replacement.** The compact TCN gains 0.0731 absolute `F1_padP_coreR` over the matched linear control while losing 0.0429 core recall. Adding frozen DINO tokens lowers the mean score relative to the compact TCN and loses still more rally coverage. This establishes a useful precision/context signal within the trial's fixed budget, not a solved rally detector or proof of a linear-capacity plateau.

At the declared ±2-second padding, means of the three separately pooled seed runs are:

| Model | `F1_padP_coreR` | Change vs linear | `P_pad` | `R_core` | Export seconds | Event F1 |
|---|---:|---:|---:|---:|---:|---:|
| Linear control | 0.7644 | — | 0.6737 | 0.8835 | 4,318.9 | 0.3990 |
| MLP | 0.7835 | +0.0191 | 0.7166 | 0.8651 | 3,936.9 | 0.3931 |
| Compact TCN | **0.8374** | **+0.0731** | 0.8358 | 0.8406 | 3,247.1 | 0.5396 |
| DINO + TCN | 0.8264 | +0.0620 | 0.8627 | 0.7931 | 2,964.2 | 0.4851 |

Padded human export is 3,516.3 seconds. Export duration alone is not accuracy: smaller exports can remove true rallies as well as dead time. The TCN's seed scores span 0.8019–0.8628, and the four already inspected source groups cannot support a strong claim about new recording conditions.

All three neural candidates improve aggregate F1 in all three seeds. Mean paired source-group F1 improves in 4/4 groups for MLP, 3/4 for TCN, and 2/4 for DINO+TCN. The TCN's beach-group mean F1 falls 0.1059 and core recall falls 0.1430. Every neural candidate fails the proposal's recall tolerance of a 0.005 regression; MLP additionally misses the +0.02 score target, and DINO lacks a strict majority of improving groups. The summary operationalizes the proposal as mean paired score/recall and at least two of three improving seeds. That descriptive screen was not a training-selection rule or a separately hash-bound promotion gate.

Required padding sensitivity follows. Each cell is the mean of three runs whose metrics first pool durations over all eight recordings; seed averaging does not add independent source groups. Ranking remains fixed at two seconds.

| Model | Padding before/after, s | `P_pad` | `R_core` | `F1_padP_coreR` | Model export, s | Human export, s | Difference, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear | 0 | 0.6152 | 0.7438 | 0.6733 | 2771.9 | 2292.3 | +479.6 |
| Linear | 1 | 0.6435 | 0.8420 | 0.7294 | 3607.2 | 2904.3 | +702.9 |
| Linear | 2 | 0.6737 | 0.8835 | 0.7644 | 4318.9 | 3516.3 | +802.6 |
| Linear | 3 | 0.7101 | 0.9036 | 0.7952 | 4932.0 | 4136.2 | +795.8 |
| MLP | 0 | 0.6756 | 0.7138 | 0.6938 | 2424.2 | 2292.3 | +131.9 |
| MLP | 1 | 0.6908 | 0.8199 | 0.7494 | 3252.5 | 2904.3 | +348.2 |
| MLP | 2 | 0.7166 | 0.8651 | 0.7835 | 3936.9 | 3516.3 | +420.6 |
| MLP | 3 | 0.7431 | 0.8892 | 0.8093 | 4570.8 | 4136.2 | +434.6 |
| TCN | 0 | 0.7952 | 0.7095 | 0.7483 | 2046.3 | 2292.3 | -246.0 |
| TCN | 1 | 0.8163 | 0.7994 | 0.8066 | 2666.7 | 2904.3 | -237.6 |
| TCN | 2 | 0.8358 | 0.8406 | 0.8374 | 3247.1 | 3516.3 | -269.3 |
| TCN | 3 | 0.8532 | 0.8613 | 0.8566 | 3798.6 | 4136.2 | -337.6 |
| DINO + TCN | 0 | 0.8152 | 0.6361 | 0.7145 | 1786.9 | 2292.3 | -505.4 |
| DINO + TCN | 1 | 0.8417 | 0.7389 | 0.7869 | 2391.1 | 2904.3 | -513.2 |
| DINO + TCN | 2 | 0.8627 | 0.7931 | 0.8264 | 2964.2 | 3516.3 | -552.2 |
| DINO + TCN | 3 | 0.8768 | 0.8240 | 0.8495 | 3519.1 | 4136.2 | -617.1 |

Strict event recall (IoU ≥0.5, unpadded events) further distinguishes the tradeoffs:

| Model | Short ≤3 s | Ace | Service fault | Mean complete rally losses after padding |
|---|---:|---:|---:|---:|
| Linear | 0.1376 | 0.3905 | 0.1463 | 30.7 |
| MLP | 0.2434 | 0.4190 | 0.2114 | 40.3 |
| TCN | 0.2646 | 0.2952 | 0.3333 | 52.3 |
| DINO + TCN | 0.2275 | 0.2381 | 0.2927 | 51.7 |

Better strict event matching for some short outcomes can coexist with worse retained footage. Complete-loss counts are seed means, not fractional real rallies. The detailed summary contains each integer count, any-overlap recall, boundary errors and original-rally identities against both linear and the retrospective shipped union.

Execution checklist:

- [x] Audit and freeze exact, consented, non-protected development labels and reusable caches.
- [x] Implement compact linear/MLP/TCN controls and a matched DINO-fusion TCN.
- [x] Implement canonical ranking, ignored-aware guardrails, and all four padding cases.
- [x] Complete runner preflight and desktop CPU ONNX qualification (browser/device checks separate).
- [x] Run nested source-group development evaluation with three fixed seeds.
- [x] Run descriptive shipped-production replay and restricted-scope production refits.
- [x] Inspect coverage failures, preserve checkpoints/predictions, and publish a decision.

**Frozen data.** Eight recordings, four source groups, 306 rallies, 7,976.43 seconds and 31,908 sampled ticks. Every proxy's complete bytes were hashed; immutable label, provenance, and feature-cache hashes were validated. All rows have explicit training consent. The protected source group and its challenge relatives, weak export coverage, candidate labels, and later intake outside this initial frozen scope are excluded. No new holdout has been manufactured from already inspected groups; results are development/regression evidence.

**2026-09-19 provenance correction:** the subsequent full-corpus audit found affirmative training consent for five later additions in a historical all-labels training manifest. Their intake-plan fields had been incomplete. The initial scope was therefore narrower than the available training-authorized corpus; missing intake metadata was not sufficient evidence that those recordings lacked authorization. This correction does not change the frozen experiment or its results. The [grass/indoor follow-up](./neural-nonbeach-execution-2026-09-19.md) records the narrower product scope; corpus expansion should use the fuller provenance chain.

This is the original `full-gold-v1` exact-label revision. All eight frozen rows have empty `ignoredIntervals`. The implementation tests ignored-time subtraction and loss masking, but this dataset does not exercise them empirically; later mutable-draft ignored revisions are not silently mixed into these results. All comparisons below use the same frozen revision.

NAS artifacts:

- `private-reference-0125`
- `private-reference-0126`

Manifest SHA-256: `d2723dd95c884a07bfe48df974975607b86afa0e540966be070db151a4ebff26`.

Completed results under the same phase directory:

- `nested-study-v1/report.json`, SHA-256 `7b0e308f3df74dba6bdec448dbc1a4e02c66a1b72e490d4a7099afe4c1389280`.
- Full per-seed metrics and guardrails (ledger `private-reference-0127`), SHA-256 `7858520f1ade01b70432c4333129abb24bd66aaeb57ded454320553d12d13c3d`.
- `nested-study-v1/summary.json`, SHA-256 `8fe1c2d82d00aa2c9ef2af2c66adf85876cd5ff6e76c6161366e29dcef751285`.

The [summary tool](../../scripts/summarize-neural-development.py) verifies the preregistration and manifest hashes, checks exact raw gold/ignored revisions, and replays the production references before writing new artifacts. All 12 neural evaluations were independently replayed without a discrepancy, and all 96 outer-refit weight/prediction hashes matched.

**Frozen experiment.** Linear (1,563 parameters), MLP (33,539), and depthwise temporal network (29,635) consume the same current 104 audiovisual channels. The fourth candidate adds ten frozen DINO tokens through shared 384-to-16 projections to the same temporal network (46,803 trainable parameters, excluding the frozen encoder). This fourth arm isolates added representation; it is not an exact reproduction of the older full-convolution DINO architecture.

Each outer source group is assessed only after inner leave-one-source-group-out predictions choose an epoch and decoder. Checkpoints are at epochs 5, 10, 20, and 30; seeds are 3407, 1729, and 20260918. Each seed's dataset score pools recording-duration numerators/denominators; reported means across seeds summarize repeated fits, not averages across videos. Selection uses `F1_padP_coreR` at symmetric 2-second padding, strict positive-gap joining below 3 seconds, and identical ignored ranges. All reported candidates include 0/1/2/3-second padding results.

Ground truth supplies live targets and Gaussian serve/end event targets (sigma 0.35 seconds, radius 1 second). Ignored intervals mask live loss; their additional 1-second halo masks boundary losses and boundary class-weight fitting. Temporal state resets at ignored spans. Per-recording percentile transforms match current feature semantics; scaler statistics use only fitting groups. Training uses AdamW, learning rate 0.001, weight decay 0.0001, head weights 1/0.5/0.5, source-group/recording-balanced chunk sampling, and square-root class weights capped at 20. The models use real 62-tick halos around 128-tick scored centers, fixed per-tick transformations, deterministic kernels, and TF32 disabled.

The common decoder grid tests smoothing 0.5/1 seconds, entry 0.35/0.5/0.65, exit entry-minus-0.1, minimum live 0.25/1 seconds, and optional bounded event-head snapping. It never passes an end-event pulse into the production sustained-dead decoder. Checkpoint and decoder selection are exclusively inner-fold operations. The outer models are refitted for the selected epoch count. No best-seed selection, protected-test selection, or production promotion is permitted.

**Environment and reproducibility.** WSL2 Ubuntu, RTX 3080 with 10 GB, isolated environment `private-reference-0128`. PyTorch 2.11.0 CUDA 12.8 and TorchVision 0.26.0 follow the [official version pairing](https://docs.pytorch.org/get-started/previous-versions/). Cached features are loaded into memory; all new result directories are separate from source datasets. The runner records the manifest, source-code and checkpoint hashes, folds, seeds, transforms, optimizer settings, CUDA memory, and per-fit loss histories. Historical experiment code is preserved.

All 192 fits completed: 144 inner fits and 48 outer refits, producing twelve pooled model/seed results. Summed fit timers were 726.4 seconds, including per-fit checkpoint prediction/saving but excluding initial feature loading, inner decoder search and final reporting. The largest recorded PyTorch CUDA tensor allocation was 261,543,424 bytes (249.4 MiB), in the DINO-fusion arm. This does not include driver/context overhead or DINO encoder extraction: frozen encoder features were reused from verified caches. The experiment demonstrates that head training fits comfortably on this 3080; it does not measure full-video DINO inference throughput.

Reproduction command, from this repository in WSL, using a new output directory:

```bash
private-reference-0110 -u \
  scripts/run-neural-development.py \
  --manifest private-reference-0129 \
  --output private-reference-0130
```

The output preregistration is written before model fitting. Resuming requires identical code/data/configuration hashes and verified completed checkpoint artifacts. A changed experiment requires a new output directory.

**Interpretation limit.** The linear control chooses the highest tested entry threshold (0.65) in all 12 outer selections and the epoch cap (30) in 8 of 12. DINO chooses the lowest entry threshold (0.35) in 9 of 12 and reaches the epoch cap in 8 of 12. This is therefore a comparison within the frozen training/decoder budget, not proof that an optimally trained and calibrated linear model has plateaued. A later, separately registered inner-fold convergence and wider-threshold diagnostic is needed to distinguish architecture gains from calibration or optimization limits. The opened run is not retuned.

**Preflight results.** The 48 compact-model/runner/evaluator/regression tests pass, as do 20 production-baseline/decoder/schema tests. All three compact models pass synthetic ONNX numerical, full-sequence, and overlapping-chunk parity. Portable weights/graph sizes are 18,836 bytes (linear), 147,304 (MLP), and 124,186 (TCN); maximum logit errors are below 1.2e-6. These are float32 synthetic models, not quantized trained deployment artifacts.

Seven additional exporter/summary regression tests pass, bringing the relevant test count to 75. These include checkpoint tampering and changes to ignored spans outside rally cores or equal-duration shifted holes; comparing only retained core duration would not detect those revision mismatches.

On a single Ryzen 5900X CPU thread, ONNX Runtime median inference for 252 ticks is 0.061/0.169/0.228 milliseconds respectively. These measurements exclude video decode, features, download and thermals and are not phone/browser timings. No Android device was connected. The runtime report is `runtime-qualification/runtime-qualification.json` under the phase directory, SHA-256 `0d6038645840b33090909b4b3baffd8fa40b83cf09cc6f68e00854d2c31ea271`.

An isolated headless Chrome 153.0.8010.48 run also passed all 36 synthetic numerical checks using ONNX Runtime Web 1.22.0, WASM CPU, one thread. Median 252-tick inference was 0.187/0.527/0.696 milliseconds for linear/MLP/TCN; full 1,024-tick inference was 0.652/2.217/2.819 milliseconds, and overlapping chunks were 1.318/4.303/7.201 milliseconds. Maximum errors were below 1.4e-6. The first session, including WASM initialization, took 168.4 milliseconds. Loaded runtime assets totaled 11.28 MB uncompressed, approximately 2.92 MB gzip, substantially more than the model graph. This is desktop-browser portability evidence, not a complete application or phone benchmark. Production dependencies were untouched; the isolated browser and test server were closed.

The browser artifact is `browser-runtime-qualification-v2/browser-runtime-qualification.json`, SHA-256 `5e895b436574a5c95963529a19cd2947a5527949737bc83eab0f275bbc0662ae`. Its fixtures, copied models, dependency lock, and source hash make the run repeatable with [the browser harness](../../scripts/qualify-neural-browser.mjs).

**Trained checkpoint portability.** The predetermined MLP and TCN seed-3407/outer-0 engineering checkpoints (both inner-selected epoch 20) were also exported with the fitted scaler, clipping and sigmoid inside the graph. The MLP graph is 140,176 bytes and the TCN graph 126,331 bytes. Both reproduce saved GPU predictions, PyTorch CPU predictions, complete sequences, and overlapping ONNX chunks on two full held recordings (5,311 and 4,182 ticks). The resulting decoded intervals are exactly equal; maximum probability errors are 3.58e-7 and 5.36e-7. Dynamic lengths from 1 to 411 ticks cover actual short segments and recording edges. Three exporter tests cover hash tampering, dynamic lengths, and fused scaling. These exports are particular outer-fold research checkpoints, not final all-development models or production replacements. Full-recording feature percentile transforms remain outside the graph.

Artifacts under `trained-runtime-qualification/`:

- `mlp-seed3407-outer0/trained-runtime-qualification.json`, SHA-256 `8f85673a8fe1f2ddbd670baeebcd0f5d17ec808c84a3c2e23f769298a56345f9`.
- `tcn-seed3407-outer0/trained-runtime-qualification.json`, SHA-256 `74b7baad2ac255aefbcc8e868ef8ecef0d65c0dd90574600d5c87c9261cbf96f`.

Each directory contains the ONNX graph, feature/decoder metadata, scaler, and parity traces. The [exporter](../../scripts/export-neural-checkpoint.py) verifies the frozen code, manifest, fold, and checkpoint hashes before qualification.

A follow-up [trained browser check](../../scripts/qualify-trained-neural-browser.mjs) ran the same TCN checkpoint in isolated desktop Chrome WASM. Short inputs and both full/chunked held traces passed with maximum probability error 3.28e-7. Replaying actual browser probabilities through the canonical Python decoder exactly reproduced all 117 saved intervals. This verifies trained graph portability, not a JavaScript decoder port or the full video pipeline. The browser and local test server were closed. Artifact: `trained-browser-runtime-qualification/report.json`, SHA-256 `db5054e40e0b6246990341d9d514146de0aa41a22144f0ffe2b19b952d83fbe2`.

**Production references.** At 2-second padding, restricted-scope group-held production refits score 0.69035 (previous), 0.75710 (all-labels v2), and 0.71533 (union). The shipped union replay scores 0.77177 retrospectively. The union favors recall at a precision/export-duration cost. Both historical training populations exclude beach; the refits intersect the previous four and v2 six eligible original non-beach recordings, then remove the outer group. Later v2 additions are excluded from this frozen study. Fixed historical epoch caps (22/23/1 for rally/serve/dead) and decoders retain historical development-selection exposure. These are useful system references, but the new same-population linear control is the clean comparison for attributing neural capacity/representation effects.

The production artifact contains all four padding cases, guardrails, held-group audits and 24 fitted heads: `production-baseline/baseline.json`, SHA-256 `2fc0580a2ddaf60454163ff4e7cefdae2e0c7deed97769128d18ce20ee7aa426`.

**Concrete failure review.** A fixed-seed-3407 TCN-versus-linear pack contains the three largest additional original-rally coverage losses and three largest removed false-positive export spans. Selection is descriptive, after evaluation, and does not change labels or training. All three highlighted losses are in the held beach source group: `beach-source-01` at 628.853–644.573 seconds, and `beach-source-02` at 633.037–641.389 and 424.300–430.456 seconds. They lose another 15.720, 6.963 and 6.156 seconds of true rally footage respectively. The largest removed false-positive spans are grass-court intervals of 18.359, 13.836 and 12.950 seconds. Sampled frames show distant beach players and between-point standing/walking in the largest grass case; sparse images are context, not independent boundary validation.

| Seed | Linear complete / partial losses | TCN complete / partial losses | Newly complete losses | Newly partial from full coverage | Recovered complete losses |
|---|---:|---:|---:|---:|---:|
| 3407 | 27 / 47 | 43 / 38 | 25 | 19 | 9 |
| 1729 | 30 / 55 | 43 / 51 | 20 | 30 | 7 |
| 20260918 | 35 / 50 | 71 / 55 | 44 | 32 | 8 |

These are losses after the declared ±2-second export padding and gap joining, out of 306 original rallies. A lower partial-loss count can reflect a transition to complete loss; it is not automatically an improvement. The pack preserves exact identities so recovered rallies cannot conceal newly lost ones.

Review the timestamped pack (ledger `private-reference-0131`), which links six contact sheets containing 30 sampled frames. `review-pack-v1/review-pack.json` SHA-256: `008efe8a83fa7a2b3d84f3aed4952d96324687e260a0ec83cff2ce4acd08beee`. The [review script](../../scripts/build-neural-review-pack.py) records source/result/image hashes and requested/decoded frame timestamps.

**Decision and next stage.** Keep the current production models. The compact TCN is the strongest candidate for further investigation, but its new complete rally losses and recall regression rule out replacement. The MLP misses the proposal's +0.02 mean score target and loses recall; DINO fusion has a larger recall deficit and does not justify its added extraction cost in this configuration. This result does not rule out other visual encoders, fine-tuning, or motion-video representations.

Before more model complexity, register a separate inner-fold diagnostic that lets the linear control converge and explores thresholds beyond the current grid edge. Pair it with recall-constrained neural epoch/decoder selection, with the recall requirement fixed before fitting or outer evaluation. Keep this completed run immutable and label the follow-up as adaptive development on already opened groups. Add independent, explicitly consented exact-label source groups, particularly varied beach/camera conditions, before claiming generalization. New phase/transition labels and a compact task-trained visual branch remain later options from the proposal; the current evidence does not yet justify a larger desktop video model or mobile production integration.

**Status:** first development experiment completed; no production promotion, protected-test opening, or physical-phone performance claim.
