# Side-switch specialist v4 — 2026-08-20

## Decision

V4 validates multi-frame side identity and court-distance normalization as a useful
feature direction, but it is **not promoted** to browser or Android inference. On the
retrospective raw-phone evaluation, exact-gap F1 improves from v3's 7.59% to 16.67% and
±2-rally F1 improves from 32.91% to 41.67%. Absolute accuracy remains inadequate: only
6 of 37 exact predictions match the 35 reviewed switch gaps.

This iteration deliberately changes only requested steps 3 and 4: multi-frame visual
identity and camera-distance normalization. Labels, source split, seven-point cadence,
candidate margins, re-anchoring, six-opportunity cap, and decoder selection remain the
frozen v3 contract. Rally order is still a noisy scored-point proxy.

## Frozen data

V4 inherits the immutable v3 6/4/11 recording split and reviewed gap labels. The blurry
`beach-source-02` recording remains excluded from fitting, preprocessing,
threshold selection, and decoder selection.

| Role | Recordings | Reviewed gaps | Switch gaps |
| --- | ---: | ---: | ---: |
| Train | 6 | 234 | 29 |
| Validation | 4 | 143 | 17 |
| Retrospective evaluation | 11 | 352 | 35 |

All 729 rows extracted successfully, all 19 features are finite, and all 21 recording
calibrations detected horizontal court geometry in each of their 49 sampled frames.
The raw-phone evaluation labels were already opened by historical v1/v2/v3 research,
so this remains confirmation evidence rather than a pristine promotion test.

## `MULTIFRAME-NORMALIZED19`

Each rally is represented by seven 256×144 ROI frames sampled from 8% through 92% of
its duration. V4 then:

1. estimates a recording-level foreground-net height from long horizontal lines in the
   first seven rallies;
2. applies a piecewise vertical warp that maps the detected net to normalized `y=0.5`;
3. estimates and removes global translation using upper-frame phase correlation;
4. weights pixels by temporal deviation and motion over all seven frames;
5. creates joint hue/saturation plus value palettes in broad and tight court-relative
   near/far regions; and
6. compares the same-side and swapped-side assignments across adjacent rallies.

The 19 ordered model inputs contain broad/tight same and swapped costs, swap margins,
orientation-flip evidence, mean and cross-scale swap behavior, side separation and
instability, global appearance change, foreground coverage, and camera-alignment
quality. No person detector or neural feature extractor is used.

The fixed class-balanced linear head selected L2 10.0 by recording-held-out training AP.
Validation selected threshold `0.477614905849774`, candidate margin ±1 rally,
distance penalty 0.25, re-anchoring, and the six-opportunity cap. The deployable
classifier/decoder fingerprint is
`54447d6dbb3db1c2ab281f3cfef0e5ef05e4cfa374cae21aa6f287508e9a98ae`.

## Results

### Development validation

| Metric | V3 | V4 |
| --- | ---: | ---: |
| Visual row AP | 27.66% | **34.69%** |
| Exact precision | 77.78% | **82.35%** |
| Exact recall | 82.35% | 82.35% |
| Exact F1 | 80.00% | **82.35%** |
| ±1-rally F1 | 91.43% | **94.12%** |

### Retrospective raw-phone evaluation

| Scoring tolerance | Model | Predictions | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Exact | V3 | 44 | 6.82% | 8.57% | 7.59% |
| Exact | V4 | 37 | **16.22%** | **17.14%** | **16.67%** |
| ±1 rally | V3 | 44 | 18.18% | 22.86% | 20.25% |
| ±1 rally | V4 | 37 | **27.03%** | **28.57%** | **27.78%** |
| ±2 rallies | V3 | 44 | 29.55% | 37.14% | 32.91% |
| ±2 rallies | V4 | 37 | **40.54%** | **42.86%** | **41.67%** |

Raw-phone row AP improves from 18.41% to 26.92%. The selected V4 threshold yields the
same 37 predictions when the candidate margin is replayed at ±1, ±2, ±3, or ±4; wider
windows add no above-threshold visual selections. The cadence-only ±4 control remains
unchanged at 9.38% exact precision, 17.14% exact recall, and 12.12% exact F1.

The model predicts the exact number of switches in only 2 of 11 evaluation recordings,
unchanged from v3. The visual improvements therefore reduce false positives and recover
three additional exact events, but do not solve the rally-count/cadence alignment error.

## Artifacts

- Feature artifact:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v4-multiframe-normalized-features.json`
  (`4d9ae424a48b81b630c72ad8650fbc2cf68be41b350b593339fe774564717069`)
- Model:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v4-multiframe-normalized/model.json`
  (`951dae585e61d00e5d65f16e85e18f0e5f16516a9c819f6765c6f9e3d63cac29`)
- Development dataset:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/models/side-switch-specialist-v4-multiframe-normalized/dataset-development.json`
  (`73e4674b6029284ec183f618c315f5426224475d4d71af45d7549723356d65ba`)
- Evaluation:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-specialist-v4-multiframe-normalized-evaluation.json`
  (`7ebb29e2a0aa3b815caefe351be028c0e1b36f7ac78638154ff9d3f72c2f74a5`)
- Provenance:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v4-provenance.json`
  (`bd7841b73427f292c864ae5e070bf54a94a276114e93ac335c4949cd8e0e90d1`)

The feature artifact was hashed and frozen before model fitting. A separate provenance
artifact binds the implementation revision, transitive full-file hashes for all 21
videos, inherited v3 label identity, and every selected v4 artifact.
The bound implementation revision is
`c53a0159e39c9f8f4bdb5b0fb9a2e593f340fa64`.

## Next dependency

V4 shows that better visual evidence helps, but its remaining failures should not be
addressed by widening the rally margin again. The next independent change should be the
point-advance/redo state model described in the v3 follow-up: reconstruct scored-point
progression, then use V4 visual evidence to confirm or place each seven-point switch.
