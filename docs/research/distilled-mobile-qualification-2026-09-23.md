# Distilled MobileNet: independent gates and runtime qualification

Status: all 28 student/temporal fits and the final independent three-seed audit
are complete. Prepared-image, teacher-target, first-fit, CPU ONNX, and desktop
Chrome WASM gates also passed. Only seed 1729 has a full evaluation scope:
retained-core recall improves from 96.93% to 98.56%, while precision falls from
83.33% to 71.31%, primary F1 from 89.62% to 82.75%, and four previously retained
rallies longer than 3 seconds become completely lost. The two other student
seeds are infeasible under the fixed inner99 grid, so no model mean is reported.
See the [complete results](distilled-mobile-results-2026-09-23.md).

The runtime result establishes that actual distilled weights can execute in the
chosen browser graph; it does not establish an overall quality gain or phone speed.

This follows the [frozen study protocol](neural-recall-distillation-protocol-2026-09-23.md).
All generated artifacts are under
`private-reference-0118`
(`private-reference-0100` in WSL).
Production and prior studies are unchanged. Beach and the protected test remain
excluded.

## Trained artifact fixed before outcomes

Runtime qualification uses seed **3407**, physical fit **inner-0-1**, student
encoder after its registered eight epochs, and the temporal checkpoint at
**epoch 60**. This choice was frozen before accuracy outcomes. It excludes both
of that inner pair's source groups from student training and temporal auxiliary
training.

The deployed encoder is MobileNetV3-Small's 927,008-parameter feature trunk.
Its ImageNet classifier and 221,568-parameter training-only DINO projection are
omitted. The temporal TCN has 44,692 parameters; its graph includes the fitted
scalar normalization. The two graphs total **3,931,839 bytes** before compression.
These are FP32 graphs, not INT8 or FP16 runtime qualifications.

The complete registered study has 18 inner student/temporal fits plus **at most**
12 outer fits across three seeds. An outer fit is skipped if no registered
candidate reaches 99% inner retained-core recall. Thus 30 is an upper bound,
not a requirement to train unauthorized fallback models.

## Independent input and first-fit checks

The independent prepared-image audit passed for all **18 recordings**, **38,308
images**, and **2,304 teaching images**. It hashes every prepared array, checks
all saved presentation times and quality values against the historical Mobile
cache, and replays the first/middle/last teaching sample for each recording:
**54 decoded source frames**. The source-frame SHA, presentation time, and both
224-pixel student and 336-pixel teacher preprocessing outputs match exactly.

OpenCV ordinal seeking is inaccurate for some Pixel variable-frame-rate files.
The independent auditor initially landed one frame late on one middle sample;
another file's average-frame-rate seek could land seconds late or fail near the
end. The audit now tries a fixed nine-frame neighborhood and falls back to
sequential display-order decoding when needed. Both paths require the original
decoded-frame SHA and the original PTS at the unchanged two-microsecond tolerance.
This repaired audit sampling only; the registered preparation already decoded
sequentially, and no prepared input or model source was changed.

The first-fit audit independently checked:

- Held-source exclusion in exact, draft, and reviewed-export student membership
  and temporal auxiliary membership.
- Eight student epoch permutations/exposure hashes and finite loss history.
- Final BatchNorm buffers identical to ImageNet initialization, while encoder
  weights changed.
- Feature cache lineage and sampled CPU encoder replay, retaining the documented
  FP16 cache rounding tolerance and unchanged audiovisual channels/scalars.
- Exact-only fitted scalers, temporal sampling/exposure, all four checkpoint
  inventories, and sampled CPU temporal-head replay.

That early fit gate is restricted to the fixed first inner fit. The separate
teacher audit also passed: it independently reconstructs teaching valid-mask
membership for all 18 recordings and replays 54 same-frame DINO targets on CPU.
The maximum absolute difference is `4.0531e-5`, within the prospectively fixed
`atol=3e-4, rtol=1e-4` elementwise tolerance. The validity mask includes ignored
intervals and the existing reviewed game-window/coverage scope; ignored spans
are never used as dead-time training negatives.

The final audit verified all 28 physical student fits and 28 temporal fits,
36 logical inner views, every strict99 candidate selection, infeasible-fold
behavior, report lineage, all four padding cases, and complete versus incomplete
seed scopes. Two infeasible source-group-012 outer fits were skipped, without fallback.

## CPU and browser graph results

Both runtimes use one CPU/WASM thread, two warmups, and ten measured calls. The
encoder timing is one prepared 224×224 image. The head timing is one 252-tick
feature tensor containing 128 central ticks and 62 real context ticks on either
side. These inputs are already decoded and preprocessed.

| Actual trained graph | ONNX bytes | CPU median / p95 | Desktop Chrome WASM median / p95 |
|---|---:|---:|---:|
| MobileNet encoder plus four regional pools | 3,718,362 | 2.82 / 4.70 ms | 18.92 / 20.03 ms |
| Dynamic temporal TCN plus fitted scaler | 213,477 | 4.00 / 5.71 ms | 8.56 / 11.22 ms |

The encoder passes on **32 real image fixtures**, eight from each of four
development source groups. The head passes on **seven fixtures** covering
lengths 1, 7, 190, 252, and 411 ticks, with **four** additional whole-sequence
versus chunk-boundary checks. Short segments and both recording edges use true
lengths, not synthetic input padding.

Maximum CPU ONNX versus PyTorch absolute differences were `2.6703e-5` for the
encoder and `1.9073e-5` for the head. Browser differences were `3.7909e-5` and
`1.9073e-5`. Every element passed the prospectively fixed combined relative and
absolute tolerances; maximum absolute error alone is not the acceptance rule.

The actual served ONNX Runtime Web 1.22.0 JavaScript/WASM files total 11,279,437
bytes (2,923,978 gzip bytes), separate from the model graphs. Browser profiles,
temporary files, and results were written to NAS. Existing Node dependencies
were read from the prior harness; nothing was installed to C: or WSL root.

### Limits for mobile deployment

- No physical phone, Android delegate, thermal, or peak-memory measurement was
  performed. Other experiments shared this desktop, so timings are engineering
  diagnostics.
- Encoder and head are separate prepared-input graphs. Video decoding,
  normalization, ROI/pool construction, quality signals, audiovisual extraction,
  2 Hz/4 Hz alignment, and UI cost are outside their timing and graph parity.
- Study tokens are stored as FP16 and restored as FP32. The graph checks compare
  fresh FP32 encoder outputs; they do not qualify the complete fresh-encoder →
  FP16 cache-roundtrip → aligned temporal-head path. Deployment must preserve or
  explicitly requalify that path before claiming end-to-end parity.
- The DINO teacher and distillation projection are unnecessary at deployment.
  The sole complete matched seed trades higher retained-time recall for lower
  precision/event F1 and four newly complete rally losses; this is not evidence
  to promote the present distilled recipe.

## Accuracy and final audit: complete

The final result includes all three seeds, even when a seed is infeasible.
Target product padding is ±2 seconds, with positive padded gaps strictly below
3 seconds retained. Ignored time is outside the evaluation universe. Each
complete seed pools duration numerators and denominators over all eight exact
recordings/four held source groups. Inner 99% eligibility is not an unseen-source
recall guarantee.

The [descriptive summary](distilled-mobile-results-2026-09-23.md) was generated
only after passing, hash-bound independent audits for the student report and the
strict99 frozen-Mobile baseline. It reports all four padding cases, export and
human-export duration, correctly removed time, incorrect export time, wanted
human-export time omitted, missed core time, original complete/partial rally
losses, original short/long rally coverage, and raw event precision/recall/F1.

It emits a model mean only when all three seeds have the full prespecified
evaluation scope. The distilled model has 10/12 feasible folds, with only seed
1729 complete; seeds 3407 and 20260918 are infeasible when holding out source-group-012
(maximum inner recall 98.7832% and 98.6118%). The frozen-Mobile baseline has
11/12 feasible folds: seeds 3407 and 1729 are complete, while seed 20260918
holding source-group-012 is infeasible. Both three-seed means and the three-seed paired
difference remain unavailable. All seed statuses stay visible; the same-seed
1729 pair is descriptive, never a favorable-subset model mean.

At target padding, the distilled seed exports 851.7 more seconds, including
732.7 more seconds of incorrect footage, to retain 39.1 more core-play seconds.
Complete rally losses change 12 to 11, partial losses 32 to 9, and fully retained
rallies 278 to 302. Complete losses among originals longer than 3 seconds change
0 to 4; those four source-group-007 events last 3.073, 3.115, 3.661, and 5.216 seconds.
Three were fully retained by the frozen baseline and one partly retained.
They are not extended rallies. Event F1 falls from 72.65% to 62.81%, illustrating
why retained-time recall alone cannot establish good rally separation.

## Artifact bindings

| Artifact relative to NAS experiment root | SHA-256 |
|---|---|
| Combined `protocol.md` | `1b2c764fde6553e0f19ac3bf6379aa02b9a12f1a19dec86fe61ab934a888eea4` |
| Runtime protocol `runtime-distilled-mobile-protocol-v1.json` | `8019eceb9b91ca62a9437c36b96f72776a648ffdb0ec16e3a1e34f89af129943` |
| CPU `runtime-distilled-mobile-cpu-v1/cpu-report.json` | `b866f1340a562cc548ae00b48a557c56b0a5a749c154f5fe0dc6459c2e80051b` |
| Fixtures `runtime-distilled-mobile-cpu-v1/fixture-manifest.json` | `47546c9745dacf305594d26e42207b76f1fc8e47c5da862c7beb2ab2468d5647` |
| Browser `runtime-distilled-mobile-browser-v1/browser-report.json` | `6c78a507bbf489072f70c04d2b4280a1f2f3056c659bbe0686df15cd03251ba9` |
| Images `distillation-images-v1/audit-prepared-images.json` | `85034af46a960547dabd1a7e6ec8ccac6c35dfabfbe372328218f160e8322820` |
| Teacher `distillation-teacher-v1/audit-inputs-teacher.json` | `6b39fc7ef27b62dc22a550ede32bc82768df78b5933c289529b38910d17a84a4` |
| First fit `distilled-mobile-v1/audit-first-fit.json` | `2ce1dd8a5aa2a2f293b3baaf8f8e6aee0e7cd2ff33720244dc763516567adee3` |
| Final result `distilled-mobile-v1/report.json` | `0620fe81eee6a852112b4884b1b2f1c9c449d5fc8589ef5f5d9745dda60ac00a` |
| Final audit `distilled-mobile-v1/audit.json` | `09a23abbc24a25befb8fc93e8af8563596d2b36fe3e75bac1712ff8f2df7a12d` |
| Paired summary `distilled-mobile-v1/summary.json` | `565f4ff478a84ec5d718a86a67990680b7ab6480158a5d3a45e67a96502f5c07` |

The student registration's canonical contract SHA is
`20b452deb0b78758cb59ef8d3db775a636b4e571881702b38646d2744d2b14ac`.
Each audit receipt binds its input files and independent auditor source.
