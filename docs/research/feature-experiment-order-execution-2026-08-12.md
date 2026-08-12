# Feature experiment order execution — 2026-08-12

## Purpose and decision rule

This is the execution ledger for the ordered experiments in the
[future-feature backlog](./future-feature-experiment-backlog-2026-08-12.md). It records measured
development evidence, protected retrospective checks when a frozen gate allows them, immutable
artifact identities, and label debt. A retrospective result can check for a regression, but it
cannot revise a development-selected candidate. The one indoor test source has been inspected
repeatedly and is not an unbiased final test.

The corpus has nine recordings, 345 frozen rallies, four independent development source groups,
and one protected retrospective source group. All model selection that claims source-group
evidence uses train plus validation only. Centered temporal summaries are offline cutter features,
not causal live-stream features.

## 1. Multiscale transitions and explicit interactions

The official full nested development run compared the 450-input control, multiscale summaries, the
seven predeclared interactions, their combination, and seven individual interaction ablations
across all four outer source-group folds. No protected row was prepared in development. All three
primary additions classified as helpful, and `combined` was selected from the eligible primary
candidates.

| Development OOF candidate | Inputs | F1 @ .5 | Time IoU | Objective | End MAE |
|---|---:|---:|---:|---:|---:|
| 450-input control | 450 | .4655 | .4862 | .5313 | 2.419 s |
| + multiscale | 1,077 | .5008 | .5022 | .5557 | 2.028 s |
| + interactions | 457 | .4890 | .5024 | .5460 | 2.174 s |
| combined | 1,084 | **.5244** | **.5131** | **.5681** | **1.938 s** |

The combined candidate's paired source-group objective delta versus control had mean `+.0397` and
median `+.0318`, satisfying the predeclared `+.01`/three-of-four-source gate. It predicted 308
rallies for 306 truth intervals and found 161 strict matches, versus 360 predictions and 155 strict
matches for the control. Strict development recall nevertheless remained weak for short rallies
(`4/63`) and service faults (`4/41`). No individual interaction earned a helpful ablation
classification; the bank should therefore be retained or removed as a unit until fresh-source
evidence can isolate its terms.

The frozen single-source retrospective was mixed rather than a clean promotion. Against the older
450-input frozen-v2 test result, combined reduced predictions from 45 to the exact truth count of 39
and dead time retained from 114.516 to 87.021 seconds. It also nudged time IoU from .6715 to .6744
and recovered one strict short/service-fault event. However, F1 @ .5 fell from .6905 (29 matches) to
.6667 (26 matches), live recall fell from .9101 to .8564, ordinary-long strict recall fell from
`28/28` to `25/28`, and end MAE rose from 1.511 to 1.613 seconds. Keep `combined` as the frozen
development winner and the substrate for the ordered ablations, but do not replace the current
production baseline from this repeatedly inspected retrospective source.

Immutable reports under the frozen labeling workspace:

- `reports/feature-order-2026-08-12/multiscale-interactions-v1-development.json`, SHA-256
  `010d5021695f73406607f5a6dcfaa8e04fd43626a5594fc8466c6267e653ea1f`;
- `reports/feature-order-2026-08-12/multiscale-interactions-v1-retrospective-test.json`, SHA-256
  `cfcd3011cca9c6ced50211df4a5feedf7a2fd03896dc921f00ea9dcdab79a63e`.

## 2. Rectangle-ROI court-relative summaries

Implementation is ready. This study will reconstruct the exact frozen step-1 winner and compare it
with orientation-invariant, signed fixed-endline, and combined rectangle-ROI summaries. Rectangle
thirds remain rough proxies; they are not a court homography and do not infer which team is serving.

## 3. Serve-anchored multistate model

Implementation is ready. The study will compare a legal `DEAD -> SETUP -> SERVE -> LIVE -> DEAD`
semi-Markov path with a same-feature binary control. Its development promotion gate requires a
positive paired objective result, higher aggregate objective, better short and service-fault strict
recall, preserved ordinary-long strict recall, and no more than one percentage point of live-recall
loss. The protected retrospective stays closed if any condition fails.

## 4. Out-of-fold component selector

Implementation is ready. Fresh v4/v5 rally and serve candidates will be generated out of fold and
compared with the frozen conservative intersection. The available assessment is leakage-controlled
but has only one held grass validation source, so it is explicitly diagnostic and non-promotable;
the frozen intersection remains selected regardless of its raw validation delta.

## 5. Labeling system and current label debt

The optional annotation schema and labeling UI now support:

- four court corners plus optional net and service-zone anchors;
- categorized hard-negative intervals, including model false positives and random dead controls;
- receiver reaction, collective stand-down, terminal cue, observability, confidence, and verified
  immediate-result transition fields; and
- anonymous serve/rally-end player tracklets with normalized footpoints or boxes, short within-window
  track IDs, court/team side, and coarse posture/motion state.

The read-only readiness scan covered all nine current drafts with zero invalid documents. It found:

- 36 required court-corner clicks; optional full geometry also needs 18 net and 18 service anchors;
- two existing hard negatives and 25 more intervals to reach three per recording; all nine
  recordings still need examples of walking/ball retrieval, celebration/huddle, model false
  positives, and random dead-time controls;
- zero fully cued transitions and a 45-rally pilot target, five stratified rallies per recording;
  and
- zero tracklets and a 45-rally pilot target, with both serve and rally-end windows, at least two
  anonymous usable tracks per window, and at least two observations per usable track.

When geometry labels are ready, run rectangle versus polygon versus homography ablations. When hard
negatives are ready, test an auxiliary dead-state head and targeted weighting with source-separated
error mining. When transition cues are ready, test direct endpoint and auxiliary outcome/transition
supervision. When tracklets are ready, test count, pairwise-distance, formation-symmetry,
reaction-synchrony, and stand-down features only after a separately evaluated player detector gate.

Readiness report:
`reports/feature-order-2026-08-12/label-readiness-before-new-labels-v2.json`, SHA-256
`215c1ae7fdd59691c6127d161295dd663687d80d53623c853081a52f52614b95` under the frozen labeling
workspace. The earlier `v1` report is superseded.

## 6. Frozen higher-resolution embeddings and tracklets

Development-only extraction is complete for all eight train/validation recordings: 7,981 one-fps
timestamps and four crops per timestamp. The pinned official OpenCV Zoo MobileNetV2 penultimate
1,280-dimensional output is ImageNet-normalized, L2-normalized by crop, and projected by a frozen
content-addressed Rademacher matrix to 64 dimensions. Crop names are geometric (`full_roi`,
`near_endline_third`, `far_court_half`, and `net_strip_proxy`) and do not claim serving/receiving
roles. The extractor records OpenCV 4.12.0 with the OpenCV CPU backend and target.

The development index is
`reports/feature-order-2026-08-12/highres-mobilenetv2-development-extraction-v1.json`, SHA-256
`6c6fe71701fb204342f44577c20bda525785b15eb3f9c0fe638256e9735ddf2b`. The official backbone
SHA-256 is `c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5`; the extractor
configuration SHA-256 is `16361cf310edb75a090ab01f07946dc23fe030a476666f4da84b626b3fab4f92`.

The nested development ablation will use the frozen step-2 report directly as its binary upstream.
Protected test caches will be extracted only if the high-resolution candidate passes the strict
paired development gate. The player-tracklet half of this step waits on the label pilot above.

## 7. Ball-dependent trajectory features

Skipped. The independent ball-presence detector gate required out-of-fold precision at least .85,
recall at least .60, every source-group recall at least .35, and every environment recall at least
.45. Full-frame detection measured precision .790 and recall .390 with zero beach recall; tiled
detection regressed to precision .677 and recall .318, also with zero beach recall. Therefore neither
oracle-guided nor detected trajectory features are being selected from this round, consistent with
the user observation that the ball features did not work as implemented.

## Validation status

The experiment implementations currently pass 70 focused Python unit tests (six transition,
thirteen court, twenty-three multistate, seventeen selector, nine high-resolution, and two readiness
tests). Thirteen annotation tests, all 39 web tests, ESLint, Python compilation, and whitespace checks
also pass. A LAN-served `/label` page, task catalog, draft label, and ranged video endpoint returned
200/200/200/206 respectively in headless Chromium with no console or HTTP errors before task load;
the deeper interaction pass timed out while the official nested run was consuming host memory, so
the UI has static and automated schema coverage but not a completed end-to-end click audit yet.
