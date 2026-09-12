# Audiovisual feature feasibility, ablation, and final regression — 2026-08-11

## Outcome

The feasible cues from the repository's video-analysis inventory were implemented in a new CPU audiovisual extractor and trained on the frozen `full-gold-v1` corpus. The extractor has **90 base signals and 450 centered temporal inputs**, up from 42 and 210. It uses real proxy audio, camera-compensated motion, quality gates, and explicitly named formation/occlusion proxies. It does not claim semantic player, formation, ball, ace, or service-fault detection.

Source-group-nested development evaluation says the added families help collectively. The clearest retrained family gains are formation geometry, residual player motion, legacy frame difference, and legacy optical flow. Audio cadence and level are mixed; the audio-onset family and legacy-appearance family are harmful in this correlated full model even though a few individual channels inside them are useful.

On the single indoor regression-test source, the full audiovisual model improves event F1 from **0.629 to 0.690**, reduces predicted events from 50 to 45, and improves end-boundary MAE from **1.921 to 1.511 seconds**. Time IoU is essentially flat at **0.672**, while live-time recall falls from 92.68% to 91.01%. This is a better event segmentation result, not a universal improvement.

A follow-up targeted-pruning pass removed the signals implicated by the ablation while retaining three individually helpful exceptions. The 365-input candidate improved pooled development F1 and IoU, but its live recall fell by **11.92 percentage points**, it improved only two of four source groups, and its paired group effect was uncertain. It was therefore **not promoted**. A frozen retrospective run on the one test video favored the diagnostic on F1 and IoU but slightly reduced recall and short-event coverage; that single-source result does not override the grouped development decision.

## Cue feasibility and disposition

| Requested cue | Feasibility now | Disposition |
|---|---|---|
| Audio RMS, peak, spectral flux, onset cadence, and cadence collapse | High | Implemented from 16 kHz mono PCM decoded by FFmpeg. Added noise floor, SNR, RMS novelty, onset strength, contact-like transient, and availability as supporting channels. |
| Time since last contact-like transient | High | Implemented as `audio_seconds_since_transient`; it is one of seven individually helpful signals. |
| Player motion and synchronized stand-down | Medium as proxies; low for identities/poses | Implemented camera-compensated residual flow, regional activity, onset/collapse, coherence, and synchronized stand-down proxies. No player detector or identity semantics are claimed. |
| Receiving-formation changes | Medium as geometry proxies | Implemented motion centroid, spread, entropy, and `receiving_formation_change_proxy`. The family is helpful, but the named receiving-formation channel alone is uncertain. |
| Ball trajectory where visible | Low | Not implemented. At 4 fps and 192×108 the ball is often subpixel; there are no ball annotations, detector weights, or high-resolution aligned derivatives. |
| Camera motion, blur, focus, and occlusion gates | High/medium as low-level gates | Implemented phase-correlation shifts/response, focus, blur, texture, visibility, dark/bright, and occlusion proxies. There are no focus/occlusion labels, so these remain heuristic gates. |
| Separate short-event logic for aces and service faults | Medium for a generic exception; low for distinct outcome inference | Implemented and tuned a high-confidence short-run exception plus ace/fault/short evaluation slices. The exception is neutral and the final decoder disables it. Distinct ace-versus-fault inference remains deferred because training is binary live/dead and only five source groups exist. |

All nine 960-wide training proxies have 48 kHz stereo AAC audio, so the audio implementation does not require returning to the 18 GiB raw sources. Missing or undecodable audio is represented by zero channels plus `audio_available=0`.

## Data and protocol

- Frozen manifest: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json`
- Manifest file SHA-256: `b662c5078d37858fd979a0c68759c0033152a3aefdce4886037351b4afb6b6fe`
- Corpus: 9 recordings, 9,082.233 seconds, 345 rallies, and 2,614.739 live seconds.
- Outcomes: 39 aces, 48 service faults, and 73 rallies shorter than 3 seconds; 68 of those short rallies have an ace/fault tag.
- Development: train plus validation, 8 recordings from 4 source groups and 306 rallies.
- Final regression test: 1 indoor recording/source group and 39 rallies.

Feature and ablation decisions use nested leave-one-`sourceGroup`-out development evaluation:

1. An outer source group is held out for measurement.
2. Leave-one-group-out folds inside the other three groups select epoch caps and decoder settings.
3. The candidate is refit without using the outer labels.
4. Full, legacy-only, added-only, and nine full-minus-family candidates use identical folds and paired seeds.
5. Headline labels require a 0.01 objective margin, compatible mean and median effects, and direction agreement in at least three of four groups.

The weighted objective is `0.55 × event F1 + 0.30 × time IoU + 0.15 × live-time recall`. The original test labels were not loaded until the immutable development report was complete.

## Implementation

The original 42 visual signals remain load-compatible. The v2 extractor adds 48 base signals:

- 11 camera/quality signals;
- 18 camera-compensated player-motion/stand-down signals;
- 6 formation-change geometry signals;
- 6 audio-level signals;
- 4 audio-onset/transient signals;
- 3 audio-cadence/time-since-transient signals.

Centered offsets at -2, -1, 0, +1, and +2 seconds produce 450 classifier inputs. Within-recording percentile ranks remain the default, while absolute quality and availability gates bypass rank normalization. Old v1 models load with the new families disabled and retain prediction parity.

The experiment tooling also adds:

- outcome slices for ace, service fault, rallies at most 3 seconds, and ordinary long rallies;
- event F1 at IoU 0.3, 0.5, and 0.7 plus 0.25/0.5/1/2-second boundary rates;
- exact development-only nested group evaluation;
- retrained leave-one-family-out ablation;
- grouped circular-shift family and individual-signal importance;
- standardized coefficient profiles as supporting evidence;
- 0/1/2/3-second out-of-fold padding evaluation;
- a final-test gate that verifies manifest, recording snapshot, feature signature/version, and code hashes.

## Development performance

These are out-of-fold predictions: every recording is scored by a model whose fitting and decoder selection excluded its source group.

| Feature set | Inputs | Objective | Event F1 | Time IoU | Live recall |
|---|---:|---:|---:|---:|---:|
| Full audiovisual | 450 | **0.5313** | **0.4655** | **0.4862** | **86.26%** |
| Legacy only | 210 | 0.4860 | 0.4170 | 0.4491 | 81.24% |
| Added only | 240 | 0.5051 | 0.4393 | 0.4827 | 79.09% |

Against legacy-only, the added families are **helpful collectively**: mean paired objective gain 0.0408, median 0.0553, positive in three of four groups. Against added-only, the legacy families are also **helpful collectively**: mean 0.0316, median 0.0457, positive in three of four groups. The two cue sets are complementary.

The full OOF result matches 155 of 306 events at IoU 0.5. F1 is 0.658 at IoU 0.3 but 0.243 at IoU 0.7, and end-boundary MAE remains 2.419 seconds. This is still review-assist performance.

## Retrained family ablation

Delta is `full objective − full-minus-family objective`: positive means the family helps. Retrained ablation is the primary classification; circular-shift permutation is supporting evidence with the fitted model frozen.

| Family | Kind | Mean delta | Median delta | Fold range | Retrained label | Frozen permutation |
|---|---|---:|---:|---:|---|---|
| Formation-change proxy | Added | **+0.0693** | **+0.0904** | -0.0183 to +0.1147 | Helpful | Helpful |
| Legacy frame difference | Previous | +0.0279 | +0.0321 | -0.0359 to +0.0832 | Helpful | Helpful |
| Player motion | Added | +0.0270 | +0.0217 | -0.0025 to +0.0674 | Helpful | Helpful |
| Legacy optical flow | Previous | +0.0186 | +0.0158 | -0.0101 to +0.0528 | Helpful | Helpful |
| Camera quality | Added | +0.0182 | +0.0067 | -0.0494 to +0.1087 | Uncertain | Uncertain |
| Audio cadence | Added | +0.0121 | -0.0021 | -0.0130 to +0.0655 | Uncertain | Uncertain |
| Audio level | Added | -0.0057 | -0.0097 | -0.0390 to +0.0357 | Uncertain | Helpful |
| Legacy appearance | Previous | **-0.0255** | **-0.0423** | -0.1096 to +0.0924 | Harmful | Helpful |
| Audio onset | Added | **-0.0303** | **-0.0420** | -0.0604 to +0.0232 | Harmful | Neutral |

The disagreement for legacy appearance is a correlation warning: the fitted full model uses appearance channels, but removing the whole group and retraining improves three of four source groups. These are operational column groups, not independent causal sensors. Formation channels also derive from residual player motion, so their family effects overlap conceptually.

## Targeted pruning follow-up

The follow-up locked its candidates before computing any new scores. Every listed base signal was removed with all five temporal offsets. The primary candidate removes `audio_onset_cadence`, removes the audio-onset family except `audio_rms_novelty`, and removes legacy appearance except `luma_std` and `luma_grid_0`. It retains 73 of 90 base signals, or 365 of 450 contextual inputs.

The same 8 development videos and 4 source groups were evaluated out of fold. This is post-selection exploratory evidence because those groups also generated the original importance findings; it is not an independent confirmation.

| Candidate | Inputs | Pooled objective | Event F1 | Time IoU | Live recall | Mean paired candidate−full | Label |
|---|---:|---:|---:|---:|---:|---:|---|
| Frozen full | 450 | 0.5313 | 0.4655 | 0.4862 | **86.26%** | — | Reference |
| Drop `audio_onset_cadence` | 445 | 0.5246 | 0.4435 | 0.4846 | **90.18%** | -0.0097 | Neutral |
| Audio-onset rescue | 435 | 0.5236 | 0.4608 | 0.4798 | 84.16% | -0.0060 | Uncertain |
| Appearance rescue | 385 | 0.5155 | 0.4680 | 0.4998 | 72.11% | -0.0301 | Uncertain |
| **Targeted primary** | **365** | **0.5354** | **0.4914** | **0.5121** | 74.35% | -0.0116 | **Uncertain** |

The targeted primary's pooled objective is slightly higher because the recordings have different sizes, but its equally weighted source-group objective is lower (0.5215 versus 0.5331). Its per-group objective changes are +0.0270, +0.0706, -0.0186, and -0.1254. It fails both the paired 3-of-4 rule and the live-recall guardrail. The result also shows why “harmful” is conditional: a correlated family can look harmful in a full-minus-family refit without every rescue or combined removal being beneficial.

The cadence-only diagnostic is the only one that raises live recall, but it loses 0.0220 event F1 and is neutral across groups. None of the four new pruned candidates qualifies as a production replacement.

## Individual-signal importance

Every base signal was circularly shifted within each held-out recording, after which all five contextual offsets were rebuilt. The outer model and decoder stayed frozen. Of 90 signals, 7 are helpful, 1 harmful, 56 neutral, and 26 uncertain under the same group-consistency rule.

Helpful signals:

| Rank | Signal | Family | Mean objective drop when shifted |
|---:|---|---|---:|
| 1 | `player_motion_spread_y` | Formation proxy | **0.0738** |
| 2 | `luma_std` | Legacy appearance | 0.0364 |
| 3 | `audio_seconds_since_transient` | Audio cadence | 0.0229 |
| 4 | `audio_rms_novelty` | Audio onset | 0.0229 |
| 9 | `flow_grid_5` | Legacy optical flow | 0.0157 |
| 14 | `player_motion_centroid_y` | Formation proxy | 0.0117 |
| 16 | `luma_grid_0` | Legacy appearance | 0.0109 |

The only individually harmful signal is `audio_onset_cadence` (mean -0.0130, median -0.0105). Notable neutral signals include raw audio RMS, spectral flux, onset strength, focus, blur, occlusion, camera-shift magnitude, synchronized stand-down, and player-motion collapse. Audio peak, contact-like transient, cadence collapse, the named receiving-formation proxy, and several residual-motion grids are uncertain.

Neutral does not mean physically irrelevant. For example, synchronized stand-down has the sixth-largest coefficient profile but neutral single-signal permutation, which is consistent with information shared across correlated motion channels. The complete 90-row ranking and every per-group repeat are retained in the machine report.

## Short events and outcome slices

The generic short-event decoder path is **neutral**: mean objective delta -0.00019 when compared with disabling it. Only two of four outer decoders selected the exception, and its aggregate gain was too small to survive the consistency rule. The fixed validation decoder and final model both disable it by selecting a 3-second minimum and threshold 1.0.

On development OOF predictions:

- ordinary long rallies: 98.22% any-overlap recall, 63.56% strict IoU-0.5 recall;
- aces: 85.71% any-overlap recall, 20.00% strict recall;
- service faults: 68.29% any-overlap recall, 12.20% strict recall;
- rallies at most 3 seconds: 76.19% any-overlap recall, 9.52% strict recall.

On the final test source, all 28 ordinary long rallies match strictly, but none of 10 short rallies and none of 7 service faults match at IoU 0.5. Padding cannot recover events with no nearby predicted crop. A distinct short-outcome model remains necessary.

## Fixed validation and final regression test

For direct comparison with the previous saved baseline, the final artifact uses the same six-train/two-validation/one-test split. Validation is tuning evidence, and the test is a single indoor regression source.

| Split/model | True / predicted / matched | Event F1 | Time IoU | Live recall | Live precision | Start MAE | End MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation, previous | 76 / 93 / 48 | 0.5680 | 0.5025 | 88.91% | 53.61% | 0.724s | 2.685s |
| Validation, audiovisual | 76 / 88 / 52 | **0.6341** | **0.5468** | **90.87%** | **57.85%** | **0.604s** | **2.233s** |
| Test, previous | 39 / 50 / 28 | 0.6292 | 0.6707 | **92.68%** | 70.83% | **0.490s** | 1.921s |
| Test, audiovisual | 39 / 45 / 29 | **0.6905** | **0.6715** | 91.01% | **71.93%** | 0.548s | **1.511s** |

The final decoder uses 0.5-second smoothing, enter/exit thresholds 0.50/0.40, 3-second minimum live duration, and 1-second gap bridging. Training selected epoch 20 of 35 completed epochs.

## Padding sensitivity

Development padding is evaluated on source-group OOF predictions and is the appropriate selection evidence.

| Symmetric padding | Live recall | Live precision | Time IoU | Event F1 | Missed live | Dead retained | Video retained |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0s | 86.26% | 52.71% | 48.62% | 46.55% | 314.89s | 1,774.45s | 47.04% |
| 1s | **93.18%** | 47.88% | 46.26% | 44.58% | **156.38s** | 2,325.06s | 55.93% |
| 2s | 95.54% | 42.99% | 42.14% | 35.61% | 102.21s | 2,904.35s | 63.87% |
| 3s | 96.52% | 39.21% | 38.67% | 26.07% | 79.76s | 3,430.05s | 70.74% |

The same frozen settings on the final test source are:

| Symmetric padding | Live recall | Live precision | Time IoU | Event F1 | Missed live | Dead retained | Video retained |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0s | 91.01% | 71.93% | 67.15% | 69.05% | 29.00s | 114.52s | 36.89% |
| 1s | **95.62%** | 61.94% | 60.23% | 65.06% | **14.13s** | 189.41s | 45.01% |
| 2s | 96.83% | 54.31% | 53.36% | 50.67% | 10.23s | 262.68s | 51.99% |
| 3s | 97.17% | 48.98% | 48.29% | 34.29% | 9.13s | 326.33s | 57.84% |

One second remains the practical knee: it roughly halves missed live time on both OOF development and final test while retaining materially less dead footage than 2 or 3 seconds. Two seconds remains a conservative option. Three seconds has little marginal coverage value.

### Pruned-candidate padding comparison

Padding remains descriptive and did not participate in pruning selection. “Pruned” below is the non-promoted 365-input diagnostic.

| Scope | Padding | F1 full / pruned | IoU full / pruned | Live recall full / pruned | Video retained full / pruned |
|---|---:|---:|---:|---:|---:|
| 8-video development OOF | 0s | 0.4655 / **0.4914** | 0.4862 / **0.5121** | **86.26%** / 74.35% | 47.04% / **34.35%** |
| 8-video development OOF | 1s | 0.4458 / **0.4757** | 0.4626 / **0.4932** | **93.18%** / 81.95% | 55.93% / **42.57%** |
| 8-video development OOF | 2s | 0.3561 / **0.3952** | 0.4214 / **0.4506** | **95.54%** / 85.09% | 63.87% / **49.99%** |
| 8-video development OOF | 3s | 0.2607 / **0.2877** | 0.3867 / **0.4090** | **96.52%** / 86.31% | 70.74% / **56.71%** |
| 1-video retrospective test | 0s | 0.6905 / **0.7160** | 0.6715 / **0.7084** | **91.01%** / 89.37% | 36.89% / **33.69%** |
| 1-video retrospective test | 1s | 0.6506 / **0.7000** | 0.6023 / **0.6490** | **95.62%** / 95.05% | 45.01% / **41.26%** |
| 1-video retrospective test | 2s | 0.5067 / **0.6933** | 0.5336 / **0.5699** | **96.83%** / 96.53% | 51.99% / **48.38%** |
| 1-video retrospective test | 3s | 0.3429 / **0.4167** | 0.4829 / **0.5075** | **97.17%** / 96.64% | 57.84% / **54.54%** |

On the one test video, the pruned diagnostic predicts 42 crops rather than 45 and retains less dead footage. It matches the same 29 rallies, but short-rally any-overlap recall falls from 40% to 30% and service-fault overlap from 42.86% to 28.57%. This test source was already opened for the earlier full-model regression, so these numbers are retrospective rather than fresh selection evidence.

## Artifacts

- Final model: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-v2-final`
- Model artifact SHA-256: `e08a3db18e4fd81fc6f5c10816b7c17f4c7661d4424a98d800aa0b1b0ccfd6a0`
- Model metadata SHA-256: `61b58eb5dbfe20fe2684a1b70ce371d0b4fef40fe495d3f57f31abe10b650c8e`
- Weights SHA-256: `c971bbd13cedb6dcbeef949bdee86adb4e7f98beff83e04ceb00c4ab3ca48982`
- Feature cache: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2`
- Development report: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/audiovisual-v2-feature-study-development.json`
- Development report SHA-256: `56af84c296f2a50c9becc804019bb688751f109426d2020a3c70d79e88888eef`
- Final report: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-audiovisual-v2-final-test.json`
- Final report SHA-256: `7e82ad543e9eb7b95a3c4791ec90c83e7d4a9f5d885a7884589ea08a08f120ad`
- Targeted-pruning development report: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/audiovisual-v2-targeted-pruning-development.json`
- Targeted-pruning report SHA-256: `6ef738b289306ca356af351d7d542314bbce9a3846978d09743d0db4b5b39b34`
- Non-promoted diagnostic model: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/audiovisual-v2-targeted-pruned-diagnostic`
- Diagnostic model artifact SHA-256: `594cd3f9ea590ccf8a728d7be7b8fb99751c7a9fa17256e9af54b770efa189c8`
- Diagnostic model metadata SHA-256: `12300a6e18312f6d23186a39bc1c7d7498bb61b530c73e5f13273bd1b1c4d80c`
- Diagnostic weights SHA-256: `4909738b3c5e9e9408b9ea9061d2c1c231900afb3054672d62eb41c5492cf98b`
- Diagnostic final report: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/audiovisual-v2-targeted-pruned-regression-test.json`
- Diagnostic final report SHA-256: `0cdfce7ccb49df679a695d5f400dcaa62715be208f4cf741ec0a75139d0c6226`

The original development study took 2,591.1 seconds. The warm-cache targeted pruning study took 748.3 seconds for 64 logistic fits, and its fixed-split diagnostic training plus regression test took 24.4 seconds.

## Reproduction

Run the development-only pruning pass first. It refuses to overwrite an existing report and never prepares the test split:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-pruned-features.py development \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --baseline-report /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/audiovisual-v2-feature-study-development.json \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2 \
  --output /path/to/new-targeted-pruning-development.json
```

Only after that report is frozen, an explicit retrospective diagnostic can open the already-used test labels. `--allow-unpromoted-diagnostic` records that this candidate did not pass development promotion:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-pruned-features.py final-test \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --pruning-report /path/to/new-targeted-pruning-development.json \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2 \
  --model /path/to/new-pruned-model-directory \
  --output /path/to/new-pruned-regression-test.json \
  --candidate targeted_pruned \
  --allow-unpromoted-diagnostic \
  --open-test
```

## Decision and limitations

Keep `full-audiovisual-v2-final` as the production candidate, with 1-second symmetric export padding as the default. The full model still contains the two families and cadence signal flagged by the first ablation: the follow-up deliberately tested removing them, but the combined pruned model failed the grouped recall and consistency criteria. Keep `audiovisual-v2-targeted-pruned-diagnostic` only as an auditable experiment, not as the replacement model. Do not treat either model's scores as calibrated probabilities or use them for unattended exports.

For the next model iteration:

1. investigate sparse/group-regularized selection rather than deleting correlated families as a block;
2. retest the simpler complete audio-onset-family removal on genuinely new source groups; the individually useful `audio_rms_novelty` did not rescue that family here;
3. build a dedicated short-outcome branch rather than relying on generic duration cleanup;
4. collect new source groups with untouched indoor, grass, and beach test coverage;
5. add high-resolution aligned derivatives and ball labels before revisiting trajectory features.

There are only four independent development groups, one random seed, and one final indoor test group. Beach appears only in development training, validation is grass, and the final test includes a previously inspected pilot excerpt. Helpful/harmful labels are therefore exploratory evidence for this corpus and weighted-logistic architecture, not statistical or external-generalization claims.
