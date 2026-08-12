# Noise-normalized frequency-band serve audio — 2026-08-11

## Outcome

The existing audiovisual serve specialist already uses audio: 13 base audio channels become 65
contextual inputs, and audio carries 14.1% of the frozen v4 head's absolute coefficient mass. Its
noise floor is only a scalar RMS feature, however; it does not remove background energy from the
spectrum or expose frequency bands.

This experiment adds causal, label-free spectral background subtraction and six broad frequency
bands. Using old and new audio together is not better than v4 at the one-second tolerance, but the
requested ablation finds a cleaner serve head: **visual inputs plus the new normalized-band audio,
with all 65 legacy audio inputs removed**. Its validation contact precision/recall/F1 is
0.723/0.895/0.800, and the already-inspected test is 0.688/0.846/0.759. Both exceed the full
old+new head and v4 at that tolerance. The full head is slightly better at two seconds on the
retrospective test, so v6 is specifically the best one-second contact candidate.

The new audio alone fails badly, so this is not an audio-only detector. The v6 paired composition
also does not improve overall interval F1 because its matched rally model is materially worse and
the fixed fallback still produces false intervals. Preserve v6 as the best research contact head,
but do not replace the existing cutting path with this pair.

## Added signal

The legacy 90 base features are unchanged. The opt-in
`noise-normalized-bands-v3` profile appends 14 signals, producing 104 base and 520 contextual
features:

- a broadband noise-removed log-SNR and a combined noise-normalized flux;
- log-SNR and positive log-SNR flux in 80–250, 250–500, 500–1000, 1000–2000,
  2000–4000, and 4000–7800 Hz.

Audio remains 16 kHz mono in non-overlapping 50 ms Hann windows. At every frame and in every band,
the implementation estimates the causal trailing 10-second 20th-percentile power floor, subtracts
that floor, divides excess power by the floor, and applies `log1p`. Positive changes form the
noise-normalized flux. Values are pooled to 4 Hz, percentile-ranked within each recording, and
expanded over the existing -2/-1/0/+1/+2-second context.

The old audio path remains untouched and legacy artifacts still load. New configs use feature
version `audiovisual-noise-normalized-audio-v3`; cache identity includes both that version and the
explicit feature set. The cache augmentation utility reuses the frozen v2 visual matrix and only
decodes audio, avoiding a second video-feature extraction.

## Frequency discovery

Frequency discovery used only the six train recordings and then checked the frozen validation
recordings. The nearest 4 Hz sample to each gold serve was compared with preparation audio one to
four seconds before serve and with in-rally audio starting two seconds after serve. These are harder
negatives than random quiet frames. AUC is descriptive feature separation, not model AP.

| Feature | Train contact-vs-hard-negative AUC | Minimum train-group AUC | Validation AUC |
|---|---:|---:|---:|
| 80–250 Hz noise-normalized flux | **0.737** | **0.716** | 0.693 |
| Combined noise-normalized flux | 0.732 | 0.696 | 0.718 |
| Existing RMS novelty | 0.708 | 0.710 | 0.683 |
| 4–7.8 kHz noise-normalized flux | 0.722 | 0.650 | **0.719** |
| Existing scalar SNR | 0.568 | 0.581 | 0.605 |
| Existing normalized spectral flux | 0.552 | 0.455 | 0.528 |

There is no single stable “serve frequency.” The consistent pattern is a low 80–250 Hz body/thump
plus a high 4–8 kHz attack, with useful energy between them. The experimental feature therefore uses
a coarse band bank rather than a narrow band-pass. FFmpeg also provides FFT denoising and standard
high/low-pass filters, but the implemented NumPy path keeps the old waveform intact and makes each
derived band independently ablatable. See the official
[FFmpeg filter documentation](https://ffmpeg.org/ffmpeg-filters.html).

## Model result

The enhanced rally and serve heads use the same split, learning rate, L2 penalty, batch size, seed,
and serve-positive radius as the v4 study. Training schedules differ. Decoder and composition
choices use validation. The test is retrospective because it was already inspected in prior
experiments.

| Split | Model | Contact at +/-1 s (P / R / F1) | Composed rallies / strict matches | Event F1 | Strict short | Strict fault |
|---|---|---:|---:|---:|---:|---:|
| Validation | v4 pair | 0.680 / 0.895 / **0.773** | 112 / 56 | 0.596 | 6/21 | 3/13 |
| Validation | normalized-band pair | 0.670 / 0.829 / 0.741 | 110 / 57 | **0.613** | **8/21** | **6/13** |
| Retrospective test | v4 pair | 0.681 / 0.821 / **0.744** | 59 / **31** | **0.633** | 2/10 | 0/7 |
| Retrospective test | normalized-band pair | 0.667 / 0.821 / 0.736 | 60 / 30 | 0.606 | **5/10** | **3/7** |

The enhanced pair covers all 10 test short events and all 7 faults at least partially. Strict
fault recovery changes from zero to three, so the frequency features are useful to a downstream
short-event path. However, the separately retrained enhanced rally baseline falls from v2's 29
strict test matches and 0.690 F1 to 25 and 0.588. The comparison confounds the new serve head with
that changed primary head and decoder.

The serve head assigns 12.8% of its total absolute coefficient mass to the 70 new contextual
inputs. Its largest new coefficient is `t+1s/audio_band_80_250_snr` (+0.312); high-band
`t+1s/audio_band_4000_7800_snr` is also positive (+0.150). These are associations among correlated
inputs, not causal importances.

Post-fit coefficient controls reinforce the non-promotion decision. Zeroing all new channels
improves validation contact F1 from 0.741 to 0.776 but loses two strict test matches and one strict
fault. Zeroing only the 80–250 Hz channels improves retrospective contact F1 to 0.800 without
changing its five strict short matches. These controls were inspected after the test and cannot be
used for selection; they show that the full six-band bank needs grouped pruning on new data.

## Legacy-audio ablation

Three serve heads share the exact enhanced rally model, samples, targets, optimizer, seed, and
520-input artifact signature. The ablated columns are set to a constant during fitting and persist
with exactly zero weight, so inference remains compatible with the paired rally extractor:

- v5 full: 385 visual + 65 legacy audio + 70 normalized-band inputs;
- v6 no legacy: 385 visual + 70 normalized-band inputs; 65 removed;
- v7 new audio only: 70 normalized-band inputs; all 385 visual and 65 legacy-audio inputs removed.

V5 and v6 independently select the same serve decoder: threshold 0.90, 10-second separation, and
+0.25-second time offset. Their contact comparison therefore isolates the fitted inputs rather
than a different operating point.

Both ablation profiles also remove the legacy `audio_available` indicator. Every recording in this
corpus has audio, so audio-missing behavior was not tested; v6/v7 must not be promoted for silent
inputs without a separate fallback check.

| Split | Serve inputs | +/-0.5 s P/R/F1 | +/-1 s P/R/F1 | +/-2 s P/R/F1 |
|---|---|---:|---:|---:|
| Validation | Full old + new | .351 / .434 / .388 | .670 / .829 / .741 | .734 / .908 / .812 |
| Validation | Visual + new, no legacy | **.394 / .487 / .435** | **.723 / .895 / .800** | **.745 / .921 / .824** |
| Validation | New audio only | .091 / .105 / .098 | .114 / .132 / .122 | .284 / .329 / .305 |
| Retrospective test | Full old + new | .417 / .513 / .460 | .667 / .821 / .736 | **.792 / .974 / .874** |
| Retrospective test | Visual + new, no legacy | .417 / .513 / .460 | **.688 / .846 / .759** | .771 / .949 / .851 |
| Retrospective test | New audio only | .104 / .128 / .115 | .208 / .256 / .230 | .250 / .308 / .276 |

Removing legacy audio recovers five validation contacts at one second with no added detections
(68/76 from 94 rather than 63/76 from 94). It also recovers one retrospective-test contact
(33/39 rather than 32/39). At two seconds the full head retains one more test contact, so the
ablation improves localization around one second rather than every tolerance.

This does not translate directly into better intervals. V6 validation composition ties v5 at
57 strict matches, 8/21 short rallies, 6/13 faults, and 0.613 F1. On test, v6 has 29 strict matches,
4/10 shorts, 2/7 faults, and 0.574 F1, versus v5's 30, 5/10, 3/7, and 0.606. V7 adds no strict test
rallies over the primary head. The serve representation improved, but composition/end-boundary
logic remains the bottleneck.

## Decision

Keep `full-audiovisual-v2-final` plus `serve-specialist-audiovisual-v4` as the current cutting
reference. Use `serve-specialist-audio-normalized-v6-no-legacy` as the best isolated one-second
contact-head research candidate, not as a promoted pair. Before another candidate:

1. confirm the v6 legacy-audio ablation in source-group folds and on new recordings;
2. allow the enhanced serve head to pair with the frozen v2 rally head so the contact contribution
   is isolated from a changed rally decoder;
3. use the learned end/dead-ball path to test whether the additional fault evidence yields correct
   boundaries rather than fixed fallback windows;
4. confirm on untouched indoor, grass, and beach source groups.

## Reproduction and immutable artifacts

```bash
PYTHONPATH=. /home/developer/volleycut/.venv/bin/python scripts/augment-audio-feature-cache.py \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --base-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-v2-final \
  --source-cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2 \
  --output-cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3

PYTHONPATH=. /home/developer/volleycut/.venv/bin/python -m analysis train \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-audio-normalized-v3 \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3 \
  --audio-feature-set noise-normalized-bands-v3 \
  --epochs 180 --batch-size 2048 --learning-rate 0.02 --seed 7

PYTHONPATH=. /home/developer/volleycut/.venv/bin/python scripts/analyze-serve-audio-features.py \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-audio-normalized-v3 \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/noise-normalized-audio-feature-auc-v1-audited.json

PYTHONPATH=. /home/developer/volleycut/.venv/bin/python -m analysis train-serve \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-audio-normalized-v3 \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-audio-normalized-v5 \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serve-specialist-audio-normalized-v5-validation.json \
  --target-radius 1 --epochs 180 --batch-size 2048 --learning-rate 0.02 --seed 7

PYTHONPATH=. /home/developer/volleycut/.venv/bin/python -m analysis train-serve \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-audio-normalized-v3 \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-audio-normalized-v6-no-legacy \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serve-specialist-audio-normalized-v6-no-legacy-validation.json \
  --serve-input-profile visual-plus-normalized-band-audio \
  --target-radius 1 --epochs 180 --batch-size 2048 --learning-rate 0.02 --seed 7

PYTHONPATH=. /home/developer/volleycut/.venv/bin/python -m analysis train-serve \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-audio-normalized-v3 \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/serve-specialist-audio-normalized-v7-new-only \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3 \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serve-specialist-audio-normalized-v7-new-only-validation.json \
  --serve-input-profile normalized-band-audio-only \
  --target-radius 1 --epochs 180 --batch-size 2048 --learning-rate 0.02 --seed 7

PYTHONPATH=. /home/developer/volleycut/.venv/bin/python -m analysis evaluate \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-audio-normalized-v3 \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3 \
  --split validation \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-audiovisual-audio-normalized-v3-final-validation.json

PYTHONPATH=. /home/developer/volleycut/.venv/bin/python -m analysis evaluate \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  --model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-audio-normalized-v3 \
  --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3 \
  --split test --retrospective \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/full-audiovisual-audio-normalized-v3-final-retrospective-test.json

for SERVE_VERSION in \
  serve-specialist-audio-normalized-v5 \
  serve-specialist-audio-normalized-v6-no-legacy \
  serve-specialist-audio-normalized-v7-new-only
do
  PYTHONPATH=. /home/developer/volleycut/.venv/bin/python -m analysis evaluate-serve \
    --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
    --rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-audio-normalized-v3 \
    --serve-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/$SERVE_VERSION \
    --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3 \
    --split validation \
    --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/$SERVE_VERSION-final-ablation-validation.json

  PYTHONPATH=. /home/developer/volleycut/.venv/bin/python -m analysis evaluate-serve \
    --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
    --rally-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/full-audiovisual-audio-normalized-v3 \
    --serve-model /mnt/freenas/volleycut/labeling-v1-2026-08-09/models/$SERVE_VERSION \
    --cache-dir /mnt/freenas/volleycut/labeling-v1-2026-08-09/features/audiovisual-audio-normalized-v3 \
    --split test --retrospective \
    --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/$SERVE_VERSION-final-ablation-retrospective-test.json
done
```

- Enhanced rally artifact SHA-256: `6ed21cff27eef53990f72ab26ebac17d910d3fa41d50e7818359f2e8e09f234f`
- Enhanced serve artifact SHA-256: `951689b0f1146aba02f4074affaf95a355f162baa81e222f7f811de108992979`
- No-legacy serve artifact SHA-256: `70350d1417dc1a0c7eb5c6445b26542ce495e7ed4f25fe8f11e2816e4a5138d7`
- New-audio-only serve artifact SHA-256: `271241dbc2979688643e1e207c521d7f4cafee2434cab73d87e06d40b77abcc2`
- Frequency AUC report SHA-256: `858e7e05c089f44f2e43c3ec33274d77af64eb6c32355c8775103ef7b8b598b9`
- Enhanced rally validation report SHA-256: `2b5147e417cd5765724fd7ed3b08a94d723b73fd1880ad217846a28ee3e15217`
- Enhanced rally retrospective-test report SHA-256: `e948b76974e7c8310affbb6d9cddd5cbeecfddd9f399edc49ac56590b0b59f52`
- V5 ablation validation report SHA-256: `e59560c29d8efd2d1372bd795e51a4974e80d7ab3791f764fa8c56f9f490e9e7`
- V5 ablation retrospective-test report SHA-256: `ad15d8698842cc54091cd05e7df41af1dd5a96fc929b150de9b940cf6b7f8fa1`
- V6 ablation validation report SHA-256: `85dbebc4b6ebc779a1add11f846969662c9d359b8d1888018a00d73dde9fc65f`
- V6 ablation retrospective-test report SHA-256: `975de98440ee67fc3bf122922d9aca2a22c3b3cbdb6d2646c9ea301ac01eeb8f`
- V7 ablation validation report SHA-256: `b5648a941dfe846711b9e122d0ef93e91f6369a9018ffd167f35c115f62f4b91`
- V7 ablation retrospective-test report SHA-256: `aca65ec9458f126d840598a18d9eec17a970f72b024e913adc508944a6bd9362`

The enhanced cache contains nine immutable NPZ files under
`features/audiovisual-audio-normalized-v3`. Training reused validation for early stopping, rally
decoder selection, serve early stopping, and the serve-composition grid. Test is a one-source,
previously inspected regression set; none of these numbers establish cross-environment
generalization.
