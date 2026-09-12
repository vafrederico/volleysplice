# End-boundary and dead-state audio experiment — 2026-08-12

## Outcome

The new noise-normalized frequency-band audio does **not** improve end-of-rally selection. It can
raise raw end-peak recall, but the extra peaks do not become additional strict rally matches. A
local live-to-dead transition head using visual plus legacy audio is the only tested variant that
improves the enhanced v5 rally/serve pair: validation F1 rises from `.6129` to `.6344`, and the
retrospective test rises from `.6061` to `.6263` by recovering one 1.343-second service fault.

That candidate is still below the current v4-pair retrospective-test reference (`.6327` F1), adds
4.015 seconds of retained dead time on test, and does not use the new audio channels. Preserve it as
a research candidate, but do not promote it. The global dead-time model is an inverse-rally
control. Decoding its complement directly as rallies regresses validation F1 from `.6541` to
`.5823`; its retrospective test F1 happens to rise from `.5882` to `.6190`, but test cannot override
the negative selection evidence. Using the same head only to refine v5 endpoints regresses test F1
to `.5859`.

## Protocol

All specialists use the frozen enhanced upstream pair and its 520-input cache:

- rally artifact `full-audiovisual-audio-normalized-v3`;
- serve artifact `serve-specialist-audio-normalized-v5`;
- 385 visual, 65 legacy-audio, and 70 noise-normalized/band-audio contextual inputs.

The profile ablations keep the same 520-input artifact signature. Removed columns are zeroed before
normalization during fitting and have exactly zero persisted coefficient weight:

| Profile | Retained inputs |
|---|---:|
| Full old + new | 520 |
| Visual + legacy audio | 450 |
| Visual + normalized-band audio, no legacy | 455 |
| Normalized-band audio only | 70 |

Three targets were tested:

1. **End pulse:** positive samples are within 0.5 seconds of each annotated rally end. Peaks are
   evaluated at one-second tolerance and used by the existing serve-gated end-refinement path.
2. **Local dead transition:** samples in the two seconds before an annotated end are live and those
   in the two seconds after it are dead; one second of pre-serve setup supplies extra live controls.
   Samples outside these local windows are masked, so this score is not a globally calibrated
   dead-time probability.
3. **Global dead state:** every valid sample outside gold rally intervals is dead and every sample
   inside is live. This is exactly `1 - rally-live`; with matched inputs, class balancing, and
   optimization it is the sign-inverted logistic objective, so it is a capacity-equivalent control.
   The v2 experiment evaluates it both as an endpoint refiner and by decoding `1 - p(dead)` directly
   as rally-live.

Epoch selection uses three leave-one-training-source-group-out fits and the median best epoch, then
fits train-only data for that fixed duration. Decoder/refinement selection uses the two-recording
grass validation split. Local/global refinement searches around an existing enhanced-v5 interval
end, requires a preceding live-low state followed by two dead-high samples, and is an exact no-op
when no transition is found. It cannot create a missing rally. The validation grid has 73 candidates
per dead-state profile. The direct global-dead control separately validation-tunes 1,597 ordinary
rally-decoder configurations.

## End-pulse result

Raw one-second end spotting shows the requested audio tradeoff. The no-legacy head has the highest
recall, but lower precision; the new-audio-only head collapses. The full head has the best
retrospective raw F1, but its validation advantage over visual plus legacy audio is only `.0033`.

| Split | Specialist inputs | Peaks / matched ends | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Validation | Visual + legacy | 270 / 40 | .1481 | .5263 | .2312 |
| Validation | Full old + new | 214 / 34 | **.1589** | .4474 | **.2345** |
| Validation | Visual + new, no legacy | 301 / 41 | .1362 | **.5395** | .2175 |
| Validation | New audio only | 31 / 5 | .1613 | .0658 | .0935 |
| Retrospective test | Visual + legacy | 117 / 22 | .1880 | .5641 | .2821 |
| Retrospective test | Full old + new | 110 / 23 | **.2091** | .5897 | **.3087** |
| Retrospective test | Visual + new, no legacy | 148 / 28 | .1892 | **.7179** | .2995 |
| Retrospective test | New audio only | 25 / 5 | .2000 | .1282 | .1563 |

The selected interval compositions do not turn those peaks into better event detection:

| Split | Composition | Predictions / matches | Precision | Recall | F1 | Time IoU |
|---|---|---:|---:|---:|---:|---:|
| Validation | Enhanced v5 baseline | 110 / 57 | .5182 | .7500 | .6129 | .5505 |
| Validation | End pulse, visual + legacy | 109 / 57 | **.5229** | .7500 | **.6162** | **.5524** |
| Validation | End pulse, full | 110 / 57 | .5182 | .7500 | .6129 | .5522 |
| Validation | End pulse, no legacy | 110 / 57 | .5182 | .7500 | .6129 | .5513 |
| Validation | End pulse, new only | 110 / 57 | .5182 | .7500 | .6129 | .5505 |
| Retrospective test | Enhanced v5 baseline | 60 / 30 | .5000 | .7692 | .6061 | .6702 |
| Retrospective test | End pulse, visual + legacy | 60 / 30 | .5000 | .7692 | .6061 | **.6762** |
| Retrospective test | End pulse, full | 60 / 30 | .5000 | .7692 | .6061 | .6721 |
| Retrospective test | End pulse, no legacy | 60 / 30 | .5000 | .7692 | .6061 | .6666 |
| Retrospective test | End pulse, new only | 60 / 30 | .5000 | .7692 | .6061 | .6702 |

Every test composition remains 5/10 short, 3/4 ace, 3/7 service fault, and 24/28
ordinary-long strict matches. The no-legacy peak head replaces one short fault with another: it
recovers the 332.452–333.795 fault but loses the 733.500–735.166 fault, so its higher raw recall has
zero net interval benefit. Categories overlap and are recall slices; category-specific precision
and F1 are not defined because predictions do not carry outcome labels.

## Local transition and global-dead result

At threshold 0.5, local-target sample classification favors visual plus legacy audio on validation.
No-legacy happens to have the highest retrospective local F1, but that does not survive interval
decoding. Global rows use all valid samples and therefore are not directly comparable with the
window-masked local rows.

| Target | Specialist inputs | Validation P / R / F1 | Retrospective-test P / R / F1 |
|---|---|---:|---:|
| Local transition | Visual + legacy | **.6122 / .6908 / .6491** | .7859 / .7885 / .7872 |
| Local transition | Full old + new | .6055 / **.6941** / .6467 | .7260 / .8237 / .7718 |
| Local transition | Visual + new, no legacy | .6132 / .6727 / .6416 | .7692 / **.8333 / .8000** |
| Global dead | Full old + new | .9480 / .7244 / .8213 | .9340 / .8096 / .8674 |

The downstream comparison is decisive:

| Split | Refiner | Predictions / matches | P / R / F1 | Time IoU | Short | Ace | Fault | Ordinary | Dead retained |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | Enhanced v5 baseline | 110 / 57 | .5182 / .7500 / .6129 | .5505 | 8/21 | 6/10 | 6/13 | 45/51 | 241.655 s |
| Validation | Local, visual + legacy | 110 / **59** | **.5364 / .7763 / .6344** | .5502 | **9/21** | 6/10 | **7/13** | **46/51** | 241.254 s |
| Validation | Local, full | 110 / 57 | .5182 / .7500 / .6129 | **.5535** | 8/21 | 5/10 | 6/13 | 45/51 | **235.256 s** |
| Validation | Local, no legacy | 110 / 58 | .5273 / .7632 / .6237 | .5521 | 8/21 | 6/10 | 6/13 | 46/51 | 248.866 s |
| Validation | Global dead, full | 110 / 57 | .5182 / .7500 / .6129 | .5523 | 8/21 | 6/10 | 6/13 | 45/51 | 246.285 s |
| Retrospective test | Enhanced v5 baseline | 60 / 30 | .5000 / .7692 / .6061 | .6702 | 5/10 | 3/4 | 3/7 | 24/28 | 76.314 s |
| Retrospective test | Local, visual + legacy | 60 / **31** | **.5167 / .7949 / .6263** | **.6707** | **6/10** | 3/4 | **4/7** | 24/28 | 80.329 s |
| Retrospective test | Local, full | 60 / 30 | .5000 / .7692 / .6061 | .6684 | 5/10 | 3/4 | 3/7 | 24/28 | **74.489 s** |
| Retrospective test | Local, no legacy | 60 / 29 | .4833 / .7436 / .5859 | .6692 | 4/10 | 3/4 | 2/7 | 24/28 | 87.725 s |
| Retrospective test | Global dead, full | 60 / 29 | .4833 / .7436 / .5859 | .6664 | 4/10 | 3/4 | 2/7 | 24/28 | 83.068 s |

The legacy local refiner's extra test match is truth interval `332.452–333.795` (1.343 seconds),
tagged `service-fault` and `serve-confidence:low`; its selected IoU is `.5103`, just across the
strict `.5` threshold. This is a useful short-fault recovery, but it comes with live precision
falling from `.7779` to `.7708` and retained dead time increasing by 4.015 seconds. The full local
head trims 1.825 seconds of test dead time but adds no strict event and slightly reduces time IoU.

### Direct global-dead-as-rally control

The global head can also be treated as a rally model by decoding `1 - p(dead)`. Its decoder was
selected on validation and then frozen for test. This is the requested direct dead-time-detector
control, distinct from the endpoint-refinement row above.

| Split | Direct model | Predictions / matches | Precision | Recall | F1 | Time IoU | Short | Ace | Fault | Ordinary |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | Enhanced rally v3 | 83 / 52 | **.6265** | **.6842** | **.6541** | **.5677** | **5/21** | **5/10** | **4/13** | **43/51** |
| Validation | `1 - global dead` | 82 / 46 | .5610 | .6053 | .5823 | .5216 | 3/21 | 4/10 | 2/13 | 40/51 |
| Retrospective test | Enhanced rally v3 | 46 / 25 | .5435 | .6410 | .5882 | .6572 | 1/10 | 2/4 | 0/7 | **23/28** |
| Retrospective test | `1 - global dead` | 45 / 26 | **.5778** | **.6667** | **.6190** | .6573 | **4/10** | 2/4 | **3/7** | 21/28 |

The direct control gains one net retrospective match and recovers three fault matches, but loses two
ordinary-long matches. More importantly, it loses six validation matches, including two faults and
three ordinary-long rallies, and drops validation F1 by `.0718`. Its favorable test result is an
exploratory observation, not a promotable result.

There is no new target information or model capacity here: `global-dead = 1 - rally-live` on every
valid sample. The fitted global artifact is not an exact numerical sign inversion of the frozen v3
rally artifact because it was retrained independently with a source-group-selected six-epoch
schedule and a separately selected decoder. Differences between the rows therefore measure
training/decoder variance, not a distinct dead-time representation.

## Decision and caveats

Do not promote any new end/dead model. Keep the visual-plus-legacy local transition artifact for
evaluation on fresh sources. The next test should freeze this decoder unchanged and evaluate it on
new indoor, grass, and beach source groups; the normalized-band channels need a new representation
or interaction model before another end-boundary ablation.

These are engineering diagnostics, not an unbiased generalization estimate:

- validation is one reused grass source group and was used for the decoder grid;
- the single indoor test source has been inspected repeatedly and is explicitly retrospective;
- the transition head only refines existing enhanced-v5 intervals and cannot recover a rally that
  both upstream heads miss;
- the centered `-2/-1/0/+1/+2`-second context is offline and non-causal;
- the local sample P/R/F1 describes only the annotated-end windows plus pre-serve controls;
- the direct global-dead decoder's positive test delta contradicts its validation regression and
  must not be selected retrospectively;
- the legacy local result is below both the v4-pair test F1 (`.6327`) and the original frozen-v2
  rally-only test F1 (`.6905`).

## Reproduction

Use single-thread numerical kernels on this host to avoid repeating the observed memory pressure:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2
ROOT=/mnt/freenas/volleycut/labeling-v1-2026-08-09
PYTHONPATH=. .venv/bin/python -m analysis train-dead-ball \
  --manifest $ROOT/manifests/full-gold-v1.json \
  --rally-model $ROOT/models/full-audiovisual-audio-normalized-v3 \
  --serve-model $ROOT/models/serve-specialist-audio-normalized-v5 \
  --cache-dir $ROOT/features/audiovisual-audio-normalized-v3 \
  --model $ROOT/models/dead-ball-specialist-audio-normalized-v3 \
  --output $ROOT/reports/dead-ball-specialist-audio-normalized-v3-validation.json \
  --dead-ball-input-profile full \
  --target-radius 0.5 --epochs 120 --batch-size 2048 --learning-rate 0.02 --seed 7

PYTHONPATH=. .venv/bin/python -m analysis evaluate-dead-ball \
  --manifest $ROOT/manifests/full-gold-v1.json \
  --rally-model $ROOT/models/full-audiovisual-audio-normalized-v3 \
  --serve-model $ROOT/models/serve-specialist-audio-normalized-v5 \
  --cache-dir $ROOT/features/audiovisual-audio-normalized-v3 \
  --model $ROOT/models/dead-ball-specialist-audio-normalized-v3 \
  --split test --retrospective \
  --output $ROOT/reports/dead-ball-specialist-audio-normalized-v3-final-retrospective-test.json

PYTHONPATH=. .venv/bin/python -m analysis train-dead-state \
  --manifest $ROOT/manifests/full-gold-v1.json \
  --rally-model $ROOT/models/full-audiovisual-audio-normalized-v3 \
  --serve-model $ROOT/models/serve-specialist-audio-normalized-v5 \
  --cache-dir $ROOT/features/audiovisual-audio-normalized-v3 \
  --model $ROOT/models/dead-state-transition-audio-normalized-v3-legacy-only-final \
  --output $ROOT/reports/dead-state-transition-audio-normalized-v3-legacy-only-final-validation.json \
  --target-mode end-transition \
  --dead-state-input-profile visual-plus-legacy-audio \
  --before-end-seconds 2 --after-end-seconds 2 --pre-serve-setup-seconds 1 \
  --epochs 120 --batch-size 2048 --learning-rate 0.02 --seed 7

PYTHONPATH=. .venv/bin/python -m analysis evaluate-dead-state \
  --manifest $ROOT/manifests/full-gold-v1.json \
  --rally-model $ROOT/models/full-audiovisual-audio-normalized-v3 \
  --serve-model $ROOT/models/serve-specialist-audio-normalized-v5 \
  --cache-dir $ROOT/features/audiovisual-audio-normalized-v3 \
  --model $ROOT/models/dead-state-transition-audio-normalized-v3-legacy-only-final \
  --split test --retrospective \
  --output $ROOT/reports/dead-state-transition-audio-normalized-v3-legacy-only-final-retrospective-test.json

PYTHONPATH=. .venv/bin/python -m analysis train-dead-state \
  --manifest $ROOT/manifests/full-gold-v1.json \
  --rally-model $ROOT/models/full-audiovisual-audio-normalized-v3 \
  --serve-model $ROOT/models/serve-specialist-audio-normalized-v5 \
  --cache-dir $ROOT/features/audiovisual-audio-normalized-v3 \
  --model $ROOT/models/dead-state-global-audio-normalized-v2-full-final \
  --output $ROOT/reports/dead-state-global-audio-normalized-v2-full-final-validation.json \
  --target-mode global-dead --dead-state-input-profile full \
  --epochs 120 --batch-size 2048 --learning-rate 0.02 --seed 7

PYTHONPATH=. .venv/bin/python -m analysis evaluate-dead-state \
  --manifest $ROOT/manifests/full-gold-v1.json \
  --rally-model $ROOT/models/full-audiovisual-audio-normalized-v3 \
  --serve-model $ROOT/models/serve-specialist-audio-normalized-v5 \
  --cache-dir $ROOT/features/audiovisual-audio-normalized-v3 \
  --model $ROOT/models/dead-state-global-audio-normalized-v2-full-final \
  --split test --retrospective \
  --output $ROOT/reports/dead-state-global-audio-normalized-v2-full-final-retrospective-test.json
```

Repeat the first and third commands with the other input profiles and artifact names in the table
below to reproduce the full ablation.

## Immutable artifacts

Semantic manifest digest:
`67cfcf46a94dad05a38aa1fdb146059cc21cff8a906cf08abc4989f4980d20db`.
Frozen rally and serve artifact hashes are respectively
`6ed21cff27eef53990f72ab26ebac17d910d3fa41d50e7818359f2e8e09f234f` and
`951689b0f1146aba02f4074affaf95a355f162baa81e222f7f811de108992979`.

| Artifact | Model SHA-256 | Validation report SHA-256 | Retrospective-test report SHA-256 |
|---|---|---|---|
| End pulse, visual + legacy (`dead-ball-specialist-audio-normalized-v2-legacy-only`) | `735df75dd1e10424c54512860c34fa203f2183e449ebeae98a99084dd9338fd7` | `50b7582c3f1b95ff5c51f4e8df86712b188c243c3f00ac07f4dddc915bd7babd` | `049f4636a2e589cf30f02dc2cba6700c9cb1766c487c7fbcffa97f26c28b8d46` |
| End pulse, full (`dead-ball-specialist-audio-normalized-v3`) | `f2d3908c2d288a5f36d48784cdcc4cb786ecff389f1df9f4d1a509fd4870834e` | `f6e54317eedd9f87e45ecc48c10433adaa580616bb961ff37efd03160c55b80d` | `b15d3eab479effd177fcb3047edfb8fdea4c3030db6d7511f2ed2bd6b4391333` |
| End pulse, no legacy (`dead-ball-specialist-audio-normalized-v4-no-legacy`) | `53aee0feac37df93ccdfbf1bbd985a63a24f45f9e4a753dc89f7ad2bfd78b6a3` | `7f799073bb9374ceb1f614438b056a4ae90ff2863ae1b6ef56988923e8752857` | `5a94a3a7d924c67b47fb2c8c82df5b3f0764125d8d21454f0cd07330da1272c5` |
| End pulse, new only (`dead-ball-specialist-audio-normalized-v5-new-only`) | `09d68be8f17a574ee9c201f7ec25b429a29574f3a9b0ef8d394699da665d4406` | `c572943c6c15c72cd456e2ca25a4f9d65c97007c0cab298dd4016cadc26cf588` | `a26a3ee2a7fa64ce6469eb9e6d47a5d218498907b01236037aa467a040c82cfd` |
| Local transition, visual + legacy (`dead-state-transition-audio-normalized-v3-legacy-only-final`) | `a53ca50890997d15d17f9b43a12a1a9fd7c7f90a187b7ae4b9fb0c4e8b3d3b0b` | `f4840323228f154a2c3be60bddf1b6c8e549dab305871207a42c012dcecfbd70` | `669036b89aac121f696013b1be95a685e6118eeaf89ee593a1b258042389f91b` |
| Local transition, full (`dead-state-transition-audio-normalized-v4-full-final`) | `cbb1125014c1f28ee1f8f38eb2da6fba3ac6e8c853e6a879039e37e67da2c0e5` | `c34a60c257fc65684ef869391d7b9f0621539381eb57e1cb0ab76f7cd182bd1f` | `9cf7797913fe7f6b24f40a88b041708ce271dc06a54c6d25ede6ebbbaebfce05` |
| Local transition, no legacy (`dead-state-transition-audio-normalized-v5-no-legacy-final`) | `f70f934a27753573ca2e5c33ba9da0000758f547f5c09aa5cbcd353f9aba697e` | `db75a4f6d38feada372e361edb31e6184c04765da6aba899eae6f5ee7f97b7d4` | `35e468da68002ba5cc6fad7ccc33f342dd132bf425e2550ff5a0366cf68d3d7a` |
| Global dead, full (`dead-state-global-audio-normalized-v2-full-final`) | `bba9eec387298b8e155ff9d15d91bc7af26277533e7b83f00ef6a1a3a08d4976` | `ff908b174eeda6d77eb5fdecc56f8980cb02e141fdbce1c290a1fc03a507c80a` | `2bdfffd546eea476535c318ac706aab5097ab75fd0016bd5b1b99d54b1b3657f` |
