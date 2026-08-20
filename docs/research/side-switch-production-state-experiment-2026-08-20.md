# Side-switch production-state and serve-grounding experiment — 2026-08-20

## Decision

Reuse the production ensemble's rally/serve/dead-state outputs as **soft V5
features**, but do not use them as hard gates and do not promote a V6 variant.

The validation-selected V5 candidate keeps the original V5 appearance frames and
adds ten production-state/gap features. On the frozen 11-recording retrospective
scope it reduces proposals from 44 to 38 while retaining 11 exact matches:

| V5 result | P | R | F1 | Proposals | Row AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen V5 | 25.00% | 31.43% | 27.85% | 44 | 43.40% |
| V5 + state gate | **28.95%** | **31.43%** | **30.14%** | **38** | **44.30%** |

This is a useful research improvement, not an automatic-use result. Exact
per-recording switch-count accuracy falls from 3/11 to 0/11, tolerant recall moves
in both directions, the labels are candidate-conditioned rather than exhaustive,
and the raw recordings were opened by earlier V1–V6 research. Keep frozen V5 and
V6 available in review; treat the new V5 output as another review-ranking layer
until the candidate gaps receive exhaustive adjudication.

The V6 candidate selected on validation (`serve-grounded:combined`) regresses on
retrospective data. Frozen V6 therefore remains unchanged.

## Question and constraints

The experiment tested whether outputs already computed by the on-device production
pipeline can:

1. down-rank gaps next to weak or false rally detections;
2. identify dead-time regions that look compatible with a side switch; and
3. anchor V5/V6 appearance comparisons near serve contact, while teams are still
   in stable formations.

No new large model was introduced. The features reuse the two shipped three-head
linear bundles over the existing F104 matrix at 4 Hz:

- `model-1ca43e38eefc` (all-labels v2);
- `model-9c92b8e9333f` (previous production); and
- production `overlap-union-disagreement-v1` components.

The frozen 6-train/4-validation/11-retrospective split, seven-point cadence,
start-at-zero assumption, six-opportunity cap, excluded blurry
`beach-source-02`, and stable source event IDs are unchanged. Both V5 and V6
had independently selected the same decoder geometry, so this comparison holds
margin `±1`, distance penalty `0.25`, and orientation weight `0` fixed. Only the
classifier threshold is reselected on validation for each feature view. The full
`±1` through `±4` margin sensitivity remains in the evaluation artifacts.

## Production replay and coverage

The extractor reuses cached F104 matrices for the ten historical sets and the F104
matrices embedded in the 11 model-feedback bundles. For every raw recording it
recomputes both production bundles, then checks the all-labels-v2 trace and merged
ranges against the inference frozen in the feedback file.

- maximum primary-head probability replay error: `3.1591e-6`;
- maximum ensemble range-boundary replay error: `0.0 s`;
- 729 candidate gaps and 1,024 source rally intervals covered;
- train: 232 both-model, 6 one-model, 4 unmatched source rallies;
- validation: 144 both-model, 3 one-model, 0 unmatched; and
- retrospective: 464 both-model, 171 one-model, 0 unmatched.

The four unmatched historical rallies use an explicit source-start fallback for
appearance sampling and carry zero production support. Every raw on-device rally
matches a production component. This support-rate shift is why agreement is a soft
input rather than a hard requirement.

## Feature views

`PRODUCTION-STATE20` contains two ten-input banks.

The state/gating bank records before/after component support, minimum adjacent
rally peak, union live fraction inside the gap, ensemble-max rally and dead-state
mean/peak values, and gap duration.

The serve/anchor bank records before/after serve-head support and confidence,
anchor error relative to the source rally start, and elapsed time from the preceding
decoded end to the next serve anchor.

The appearance ablation has two modes:

- `original`: frozen V5 or V6 whole-rally sampling; and
- `serve-grounded`: a two-second comparison window from 1.25 seconds before through
  0.75 seconds after the best production serve anchor. Two-head contacts within one
  second are confidence-weighted; fallback order is one-head serve, component start,
  then source start.

For each family, training compares `base`, `state-gate`, `serve-anchor`, and
`combined` under both appearance modes. L2 is selected by recording-held-out
training AP. Candidate and threshold selection use only validation.

## V5 result

All values below use the candidate's validation-selected classifier and threshold.

| Candidate | Validation AP | Retrospective AP | Exact F1 | ±2-rally F1 | Proposals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original/base control | 53.46% | 43.40% | 27.85% | 45.57% | 44 |
| **Original/state gate (selected)** | **65.91%** | **44.30%** | **30.14%** | 46.58% | **38** |
| Original/serve anchor bank | 57.38% | 36.88% | 25.58% | 48.84% | 51 |
| Original/combined | 63.07% | 44.21% | 28.21% | 43.59% | 43 |
| Serve-grounded/base | 50.12% | 42.02% | 12.70% | 38.10% | 28 |
| Serve-grounded/state gate | 55.59% | 43.57% | 15.69% | 31.37% | 16 |
| Serve-grounded/serve bank | 51.44% | 42.23% | 13.33% | 36.67% | 25 |
| Serve-grounded/combined | 57.04% | 41.88% | 33.77% | 49.35% | 42 |

The last row looks strongest retrospectively, but its validation exact F1 was only
91.43% versus 97.14% for the selected original/state candidate. It cannot replace
the validation winner after retrospective labels are opened.

Selected V5 tolerance detail:

| Tolerance | Frozen V5 P / R / F1 | V5 state P / R / F1 |
| --- | --- | --- |
| Exact | 25.00 / 31.43 / 27.85% | **28.95 / 31.43 / 30.14%** |
| ±1 rally | 34.09 / 42.86 / 37.97% | **39.47 / 42.86 / 41.10%** |
| ±2 rallies | 40.91 / **51.43** / 45.57% | **44.74** / 48.57 / **46.58%** |

The separately requested candidate-margin sensitivity is also positive for V5.
These are retrospective diagnostics with the classifier and threshold frozen; margin
1 remains the validation-selected setting and the wider rows cannot be promoted from
this result.

| Candidate margin | Frozen V5 P / R / exact F1 | V5 state P / R / exact F1 |
| --- | --- | --- |
| ±1 rally | 25.00 / 31.43 / 27.85% | **28.95 / 31.43 / 30.14%** |
| ±2 rallies | 28.00 / 40.00 / 32.94% | **33.33 / 40.00 / 36.36%** |
| ±3 rallies | 27.45 / 40.00 / 32.56% | **32.56 / 40.00 / 35.90%** |
| ±4 rallies | 28.85 / 42.86 / 34.48% | **34.88 / 42.86 / 38.46%** |

## Attribution

Because gap duration is not a production-head output, a post-hoc attribution run
separates it from the actual production scores. These rows were generated after the
retrospective result was opened and cannot change selection.

| V5 attribution | Validation AP | Retrospective AP | Exact F1 | ±2 F1 |
| --- | ---: | ---: | ---: | ---: |
| Base | 53.46% | 43.40% | 27.85% | 45.57% |
| Gap duration only | 51.82% | 25.44% | 20.78% | 46.75% |
| Production head state only | 55.40% | **46.51%** | **29.63%** | 44.44% |
| Head state + duration | **65.91%** | 44.30% | **30.14%** | 46.58% |

The production heads contribute genuine exact-event value; the improvement is not
just a long-gap shortcut. Duration alone harms exact F1, while head outputs alone
raise exact F1 and row AP. Their combination removes more proposals but loses one
±2 match relative to frozen V5.

## V6 result

All eight V6 candidates tied at 88.89% validation exact F1. Validation AP selected
serve-grounded/combined at 62.54%, ahead of the original/base control at 58.51%.
That ranking did not transfer:

| V6 result | P | R | Exact F1 | ±1 F1 | ±2 F1 | Proposals | Row AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen V6 | 20.83% | 28.57% | **24.10%** | **33.73%** | **40.96%** | 48 | **45.47%** |
| Grounded + combined | 18.75% | 17.14% | 17.91% | 29.85% | 38.81% | 32 | 36.23% |

The V6 attribution likewise finds no event-level gain from head-only state: it
leaves exact/±2 F1 at 24.10%/40.96% despite a small AP increase to 46.21%. V6 is not
promoted. The selected experimental V6 candidate emits the same 32 decisions at
candidate margins ±1 through ±4, so its exact F1 remains 17.91% throughout.

## Hard-gate result

Hard gates were deliberately diagnostics, not selectable candidates. On selected
V5, requiring both adjacent rallies to have two-model support drops exact recall from
31.43% to 11.43% and F1 from 30.14% to 15.09%. Requiring a serve detection on both
sides drops exact F1 to 21.54%; requiring both conditions drops it to 12.00%.

The raw recordings contain far more valid one-model intervals than historical
development data. Production support must remain graded evidence.

## Suppression decision

The suppression head is not a valid model input for this experiment:

- its positive definition explicitly contains side-switch intervals;
- its fitting data overlap 3/6 training, 3/4 validation, and 5/11 retrospective
  side-switch recordings; and
- even on the six recording-disjoint retrospective files, gap-mean suppression AP
  is 10.53% and gap-peak AP is 9.32%, with switch gaps scoring lower than unmarked
  gaps on average.

Scores are retained under a quarantined diagnostic key with
`eligibleForModelInput: false`. They are absent from every classifier signature.

## On-device implication

The selected V5 variant adds only ten scalar reductions and a 32-weight logistic
head after production analysis has already run. It requires no new video decode,
neural model, or F104 extraction. The serve-grounded appearance path would also be
on-device feasible, but it did not pass this comparison and should not be ported.

No TypeScript/Java runtime or review-UI layer is added in this experiment. That work
should wait for exhaustive review of the current V5/V6 proposal union and the new
V5-state proposals.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-production-state.py prepare

PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-v5.py \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-production-grounded-corpus-v1.json \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v5-serve-grounded-base-features.json \
  --no-enforce-source-hash

PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-v6.py \
  --manifest /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-production-grounded-corpus-v1.json \
  --output /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-v6-serve-grounded-base-features.json \
  --no-enforce-source-hash

# Join the state artifact to original and grounded V5/V6 appearance artifacts.
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-production-state.py augment \
  --family v5 --appearance-mode original --base-features <v5-original> --output <v5-original-state>
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-production-state.py augment \
  --family v5 --appearance-mode serve-grounded --base-features <v5-grounded> --output <v5-grounded-state>
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-production-state.py augment \
  --family v6 --appearance-mode original --base-features <v6-original> --output <v6-original-state>
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-production-state.py augment \
  --family v6 --appearance-mode serve-grounded --base-features <v6-grounded> --output <v6-grounded-state>

PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py freeze --family v5
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py evaluate --family v5
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py attribute --family v5
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py freeze --family v6
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py evaluate --family v6
PYTHONPATH=. .venv/bin/python scripts/train-side-switch-production-state.py attribute --family v6
```

Every destination refuses overwrite.

## Immutable artifacts

All paths are under
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/`.

| Artifact | SHA-256 |
| --- | --- |
| `reports/side-switch/side-switch-production-state-v1.json` | `a350fc11ecf875dd40b2b30a8abbf66ff8cce13684d2aad8f9433713f148c44a` |
| `reports/side-switch/side-switch-production-grounded-corpus-v1.json` | `bc6bb6cfdeae154fd6399f6a813b920f325a4704049b6d03c8ead879aeb4df8e` |
| `reports/side-switch/side-switch-v5-production-state-v1-features.json` | `c86b8ef7427dd8e1347d726c7d2326f18f22a6eb9fb16a0eec685c0fde0c9f36` |
| `reports/side-switch/side-switch-v5-serve-grounded-base-features.json` | `cfc01f852467b0b2c30ef1dc78ea0806bdcd51a931b62c5dfde6892b1193a1a6` |
| `reports/side-switch/side-switch-v5-serve-grounded-production-state-v1-features.json` | `6e75f2de2936917ea3f97a5e95c29924d4b4b5807363a9df035bb125875bb06e` |
| `models/side-switch-v5-production-state-v1/model.json` | `0fcbde8ed4deb5e3d805a336861918f9f52d5ad753a0f6ef1f6ee1bdda25a7c0` |
| `models/side-switch-v5-production-state-v1/dataset-development.json` | `b28a1f4e7674c1def377b598f1cbd8ac1f86df6420a7e27062eaad9d9b9a3dfc` |
| `reports/side-switch/side-switch-v5-production-state-v1-evaluation.json` | `7b243b5140fdbf08fb476d0679af314196744137588fd2e30f2f92e3d9bf3391` |
| `reports/side-switch/side-switch-v5-production-state-v1-attribution.json` | `d8c190c648509f6088d6b77cd916b0e09ed09b46d1c5cbc1b9d4f371ba9a05e2` |
| `reports/side-switch/side-switch-v6-production-state-v1-features.json` | `74bf411b583f896b41a70d3d635b864c33f42d6436ffcf505753c7bb050c511c` |
| `reports/side-switch/side-switch-v6-serve-grounded-base-features.json` | `a845b93d16286e73b8c67ae699abca45acf2ea82fd1c9c4dae9d0a7f8d711f0c` |
| `reports/side-switch/side-switch-v6-serve-grounded-production-state-v1-features.json` | `4da7b8f7c695e1576970008fb61268c16daa1a437743b1774a72738f0c4a035b` |
| `models/side-switch-v6-production-state-v1/model.json` | `f6d34eba0eed340b92deab609488dad7df9046da8d832f13cb6ec2f068609207` |
| `models/side-switch-v6-production-state-v1/dataset-development.json` | `b4e8cb441f30a6df6498032a62e618240e1b15bfb660e9e4583114c8279fafd7` |
| `reports/side-switch/side-switch-v6-production-state-v1-evaluation.json` | `a49249e6696f7eaee3528ffe6925ea43a1f28e3cb37dbcb8168814735b33b7e0` |
| `reports/side-switch/side-switch-v6-production-state-v1-attribution.json` | `80122a69ddfb386b7381f697fb5f5a0d8c664a9b0f57bbe5c86d2d3bd069484f` |

- selected V5 fingerprint:
  `dc5ea15a4455ac06af9e1d71d81dbd0fbe1886e2b2d0eb708d60f1ab9647e5dd`;
- selected V6 experimental fingerprint:
  `5909cfb20b975f416973b42bd7426e7b4c5fa42e4441caa2dd83c76d50937032`;
- implementation revision: `5221c74f51356e993ebfcad8326469fd6bb80124`; and
- provenance:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-production-state-v1-provenance.json`,
  SHA-256 `174fe3980cdd55fa14dda00c7e27d1b01882f303ec3ece6b6d709276e88643e7`.
