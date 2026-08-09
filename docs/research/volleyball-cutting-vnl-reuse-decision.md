# Beach cutting and VNL-STES reuse decision

**Status:** Adopt the research concepts; do not depend on either research artifact as a drop-in model.

**Decision date:** 2026-08-07

## Decision in one page

Build and evaluate our own rally-cutting model on our stationary end-line recordings. Use the 2024 beach-volleyball thesis as the main design and evaluation blueprint. Treat the 2025 CVPR VNL-STES work as an optional later source of action-spotting code and labels, not as the primary rally cutter.

The concrete decisions are:

1. **Proceed with a clean-room v0 now.** The audited public beach-thesis record contains no task code, trained model, or training data that can be executed as delivered. VNL-STES has public BSD-3-Clause code and a dataset that is anonymously downloadable in practice, but no released weights, no stated dataset license, and multiple data/code reproducibility defects.
2. **Decision — use a fixed, elevated, landscape camera behind an end line.** Keep the full court, both sidelines, and both service areas in frame with margin; do not pan, zoom, or cut. Record at 1080p, 30 or 60 fps. The end-line/elevated choice is informed by beach-paper figures, not prescribed by its text; neither paper reports an exact camera height or distance.
3. **Train rally/dead-time segmentation on our own full-match videos.** VNL's public claim concerns already-extracted rally clips, so it lacks the between-rally negative footage needed for our primary task.
4. **Reuse ideas, metrics, and selected BSD code—not reported accuracy.** The beach results are for professional outdoor 2v2 beach footage; VNL results are for professional indoor broadcast footage. Neither number is a zero-shot estimate for a fixed phone in a gym.
5. **Do not count the 24-match BMVC dataset as available.** It is a later, larger work that says the authors *plan* to release data. It is not the same artifact as the public-claim, 8-match CVPR VNL-STES dataset.

## Evidence labels used below

- **Fact** means the source states it or it was directly verified in the public artifact.
- **Visual inference** means it was inferred from a paper figure or released demonstration frame, not specified as capture geometry by the authors.
- **Decision** is our engineering choice based on those facts.

## Do not conflate the two VNL works

| Work | Data described | Labels | Method | Availability conclusion |
|---|---:|---|---|---|
| **CVPR Workshops 2025, VNL-STES** | 8 matches, 1,028 rally clips, 251,110 frames, 6,137 events | Serve, receive, set, spike, block, score | RegNet-Y + GSM, bidirectional GRU, MLP location head | Paper and BSD-3-Clause code are public. The resized archive is anonymously downloadable through SharePoint with redirect/cookie handling, but has no stated data license and does not match several paper/metadata counts. No trained weights were found. |
| **BMVC 2025, 3D heatmaps with DSGK** | 24 matches, 3,518 rally clips, 579.5 minutes, 23,638 events | The six above plus dig and pass | Modified 3D U-Net trained on dynamically shifted 3D heatmaps | Paper and demonstration supplement are public. The paper says the dataset will be released in the future; no code, dataset, weights, or license were found. |

The CVPR paper's dataset link redirects to a file named `vnl_1.0.zip`, whose root directory is `vnl_1.5/`. Current GitHub scripts also refer to unavailable `vnl_2.0` data, without documenting its relationship to the download or the BMVC expansion. We therefore do not infer that `vnl_2.0` means the 24-match BMVC data.

Primary sources: [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Nguyen_VNL-STES_A_Benchmark_Dataset_and_Model_for_Spatiotemporal_Event_Spotting_CVPRW_2025_paper.pdf), [CVPR project page](https://hoangqnguyen.github.io/stes/), [current code](https://github.com/hoangqnguyen/spot), [BMVC paper and supplement page](https://bmvc2025.bmva.org/proceedings/987/).

## Availability and licensing

| Artifact | Paper | Dataset | Model code | Trained weights | Rights that are actually stated |
|---|---|---|---|---|---|
| Beach thesis | Publicly readable | No public release found in the audited record/search as of 2026-08-07 | No public release found in the audited record/search as of 2026-08-07 | No public release found in the audited record/search as of 2026-08-07 | The CTU record says the thesis is copyright-protected and permits extracts/copies for personal use under Czech law. It is not an open-source license. |
| CVPR VNL-STES | Public CVF open-access paper | Resized 398×224 archive was anonymously downloadable on 2026-08-07 (13,024,307,624 bytes); full resolution is explicitly unavailable. Redirect/cookie handling is tooling-sensitive. | Public repository, BSD-3-Clause | None found in Git history, tags, releases, or project links | BSD-3-Clause covers the repository code. No separate dataset license or commercial-use grant is stated on the paper, project page, download link, archive, or repository. |
| BMVC 24-match VNL | Public BMVA paper | Not released; paper says the authors plan to release it | None found | None found | No data/model license found. The supplement contains five rendered prediction videos, not source data or code. |

### Beach artifact audit

**Fact — source and rights.** The [CTU repository record](https://hdl.handle.net/10467/114627) exposes the thesis and two reviews. Its `ORIGINAL` bundle contains only those three PDFs. The repository rights statement is not an open-source or dataset license. The thesis's attachment inventory on printed p. 66 lists a `README.md`, `data/`, `demo/`, `results/`, and `src/` tree, showing that an implementation package existed, but that package is not present in the public DSpace bundle. See the thesis [full text](https://dspace.cvut.cz/server/api/core/bitstreams/39ffe6fe-d5aa-4ef9-b1d4-40be5beae72e/content), attachment inventory p. 66.

**Fact — data provenance.** The videos came from an official FIVB database and rally timestamps came from BEACH-DATA's product and scouts. The thesis does not offer either source as a public dataset. It describes 324 matches from 2023 (about 250 hours) plus 88 matches from two 2024 tournaments, more than 300 hours in total. See §3.1, printed pp. 19–24.

**Conclusion.** We may learn from the disclosed method and independently implement it. We should not copy an unavailable implementation, assume rights to the FIVB/BEACH-DATA footage, or describe its dataset/model as open source.

### CVPR VNL-STES artifact audit

**Fact — public claim.** The paper says the resized dataset with annotations is public and full-resolution data is unavailable due to size. It contains 8 matches from the 2022–2023 VNL seasons, 1,028 rally clips at 25 fps, and 6,137 point events. Original match videos were full HD at 24–50 fps and were collected with permission from Volleyball World. See §§3.1–3.5, PDF pp. 2–4 / proceedings pp. 5862–5864.

**Fact — current access.** On 2026-08-07, the project page's [Data link](https://bit.ly/vnlvolley1) redirected to a university SharePoint file named `vnl_1.0.zip`. An anonymous download succeeded when the SharePoint cookies established during the redirect were retained; the payload was 13,024,307,624 bytes. Some browser/safety tooling surfaces an authentication page or blocks the redirect, so this is a public but tooling-sensitive distribution path—not evidence of a private dataset. No dataset license was found in the archive or linked materials.

**Fact — downloaded archive audit.** The archive is useful but not internally self-consistent:

- The file is named `vnl_1.0.zip`, but its root directory is `vnl_1.5/`; repository scripts additionally reference an unavailable `vnl_2.0`.
- The paper reports 6,137 events. Event arrays in the archived split JSON files contain 6,489 (`serve` 1,063; `receive` 1,699; `set` 1,375; `spike` 1,329; `block` 513; `score` 510), while `all.json` contains 6,623. Summed `num_events` fields give 5,987 and are stale in 382 records.
- The archive contains 251,803 JPEGs versus the paper/metadata count of 251,110. This includes one additional unlabeled 657-frame rally and 36 rallies with one extra frame.
- It contains only 329 per-rally CSV files for 1,028 reported rallies and no MP4 files. Code paths that expect video therefore need a conversion/export step.
- Archived events and the loader use `xy: [x, y]`; the README's example instead documents separate `x` and `y` fields.

**Fact — code and weights.** The [repository](https://github.com/hoangqnguyen/spot) is public under [BSD-3-Clause](https://github.com/hoangqnguyen/spot/blob/bbaa249c285c6bc133b091e50e6da2bf4ed42a39/LICENSE). It includes training, temporal/spatial evaluation internals, MP4 inference, and a Gradio demo. It has no tags, GitHub releases, checkpoints, ONNX files, released annotation/data payload, or frozen paper commit. The current head is `bbaa249c285c6bc133b091e50e6da2bf4ed42a39`, dated 2026-03-18—well after the CVPR paper.

**Conclusion.** The code is reusable under its license. The dataset is publicly retrievable, but its rights are not established, its link is unsuitable as an unattended bootstrap dependency without robust redirect/cookie handling, and its count/schema drift must be repaired and versioned locally. There is no public trained model to evaluate zero-shot.

## Camera evidence and fit

### What the beach recordings used

The following are **facts** from §3.1, printed pp. 19–24:

- Each recorded court had at least one camera covering the whole playing area.
- Every recording was 1920×1080 at 50 fps.
- The view stayed constant throughout a match and contained no cuts.
- There was no generally defined camera rig; angle and perspective varied by court.
- Court-corner image coordinates were manually collected for each court and used to calculate geometry-based grouping features (§3.1.3, printed pp. 22–23).
- Optical-flow preprocessing masks/crops non-court pixels (§3.2.1, p. 24; §4.1, p. 31). The public text does not establish that this mask was generated directly from the manually recorded corner points.
- Videos were grouped using court perspective: the lower-corner angle and court-area/image-area ratio.

**Assessment.** Because perspective is an explicit grouping variable and the viewpoints vary, camera geometry must be treated as a generalization variable in our data collection and evaluation.

The following is **visual inference**, not a stated physical specification:

- Figure 3.1 (p. 20) and the camera mosaics in Appendix A (pp. 55–61) predominantly show wide, elevated views from behind an end line, with the net approximately horizontal. Some are centered and some are oblique.
- The figures do not establish camera height, distance behind the line, focal length, or required clearance.

**Assessment.** This is a good geometric match for our capture contract and a domain mismatch in other respects: the thesis uses professional outdoor sand, 2v2 play, FIVB venues, and 50 fps.

### What VNL-STES used and assumed

The following are **facts** from §§3.1–3.5, PDF pp. 2–4 / proceedings pp. 5862–5864:

- The source is professional VNL broadcast video, originally full HD at 24–50 fps.
- A court-keypoint detector looks for 24 court keypoints; clips are created when at least 10 are detected in consecutive frames.
- Extracted rally clips are saved at 25 fps.
- Splits are match-disjoint: 811 train, 102 validation, and 115 test rallies.
- The paper says the split includes variation in camera angles but gives no physical camera coordinates, height, distance, or placement requirement.

The following is **visual inference**:

- Figure 1 (PDF p. 3 / proceedings p. 5863), the annotation screenshot in Figure 3 (p. 4 / 5864), and the public BMVC VNL demonstration clips show a high, centered **sideline broadcast** view with the net approximately vertical in the image.
- These examples are not evidence that every clip uses that exact shot or that a fixed end-line camera is supported zero-shot.

**Assessment.** VNL is closer to our indoor 6v6 sport domain, but farther from our fixed-phone camera geometry and from the full-match cutting task.

### Camera compatibility decision

| Requirement | Beach thesis | CVPR VNL-STES | Fit for us |
|---|---|---|---|
| One locked camera | Explicit | Not specified; source is broadcast | Use the beach assumption. |
| Full court visible | Explicit | Court keypoint visibility drives extraction | Required. Include service zones and a safety margin. |
| Behind end line | Common in figures, not mandated in text | Example frames are sideline broadcast | Our end-line view aligns much more closely with beach. |
| Elevation | Visually elevated, no number | Visually elevated, no number | Required enough to reduce player/net occlusion, but the papers do not justify an exact height. |
| Resolution and fps | 1080p50 | Full-HD 24–50; clips normalized to 25 | 1080p30 works for cutting; 1080p60 preserves options for later touch/event work. |
| Panning/zoom/cuts | Fixed and no cuts | No fixed-camera contract stated | Reject or flag interrupted/panning footage for v0. |

**Decision — recording contract.** Put the phone in landscape, centered behind either end line, elevated, and locked. Frame the complete court, sidelines, both service areas, antennas, and several feet/metres of margin. Keep that framing for the full set. Retain audio. There is no research-supported exact height/distance, so acceptance should be coverage-based: all court corners and a server at the farthest service position must remain visible without zoom or operator movement.

## What is technically reusable

### From the beach thesis: reuse now

1. **The task definition.** Learn a dense binary state—rally versus non-rally—over the full match, then convert that state into intervals. This directly serves cutting. See §3.2 and §§4.2–4.3, printed pp. 23–35.
2. **Low-rate temporal context.** The best system used two EfficientNetV2-B3 2.5D classifiers, each taking five frames at 1 fps, with steps `k=1` and `k=3`; predictions were ensembled 0.6:0.4. This creates short- and longer-context views cheaply. See §5.3.3, pp. 43–46.
3. **Temporal decoding.** The thesis applies median smoothing and a minimum interval length. We should preserve the concept but tune the thresholds on our video. See §4.3, pp. 34–35.
4. **Camera-aware splitting.** Group or stratify by court geometry, venue, team, device, and recording session; split whole matches, never random frames. The thesis explicitly groups by perspective, while VNL explicitly splits by match.
5. **Full-match metrics.** Report interval IoU, false/missed rallies per match, start/end mean absolute error, precision/recall, and the percentage of matches with the exact rally count. Frame accuracy alone hides cutting failures.
6. **Conservative editing.** Human scout timestamps themselves vary, especially at rally start (§6, pp. 52–53). Define our label contract precisely and favor extra context over removing live play.

The best beach test result was 93.0% frame accuracy, 77.2% interval IoU, 99.6% rally precision, 99.7% recall, 1.84 s start error, 1.44 s end error, and 64% of 45 test matches with an exact rally count; 98% were within two count errors. These are useful target metrics, not expected performance on our domain. See §5.4, pp. 47–49, Tables 5.17–5.18.

Those boundary errors are also **not directly comparable to our v0**. The thesis ground truth starts at the referee's whistle and ends when the ball touches the court or the referee otherwise terminates play (§3.1.2, printed p. 21). Our v0 intentionally defines the unpadded core as serve-ball contact through dead ball, with edit padding applied afterward. The thesis itself reports inconsistent scout conventions at rally start, so adopting its MAE as our acceptance threshold would mix two different label policies with annotator noise.

The thesis's failure cases should become explicit tests. It separately specifies a greater-than-four-second output rule (§4.3, p. 34) and reports false negatives on short, roughly 5–10 s ace/service-fault rallies (§5.4, p. 47); the thesis does not prove that the rule caused those misses, so we should test that hypothesis rather than state it as fact. Reported false positives included a referee whistle for a new ball—after which the interrupted rally was replayed on court—and cheerleader activity, not replay footage. Section 6 suggests fine-tuning as a possible mitigation for a materially different tournament viewpoint. See §§5.4–5.6, pp. 47–51 and §6, pp. 52–53.

### From CVPR VNL-STES: reuse later or selectively

1. **BSD model building blocks.** RegNet-Y/GSM feature extraction, a bidirectional GRU temporal head, and an event-location head can seed an event-spotting branch after the rally cutter works.
2. **Serve as a supporting boundary cue.** Serve spotting is the strongest reported class at 79.73 mTAP@0–4F. It can eventually reinforce candidate rally starts.
3. **Normalized event locations.** Frame-normalized `(x,y)` labels are useful for an annotation UI and later associating events with court/player tracks.
4. **Match-disjoint evaluation.** Preserve its split discipline.
5. **Manual correction after pseudolabeling.** The acquisition pipeline is a practical pattern for expanding labels once a baseline exists.

VNL-STES should not be the primary cutting model:

- The downloadable unit is an already-extracted rally. Between-rally dead time—the hard negative class for our task—is absent.
- Its labels are instantaneous actions, not rally intervals.
- The paper/project page reports only 244 score events across 1,028 rallies; the archived split JSON instead contains 510, part of the count drift documented below. Score was still the paper's weakest STES class at 43.05 mTAP@0–4F, so it is not a dependable end-of-rally proxy.
- Its spatial score is measured in pixels after image resizing, not in calibrated court distance.
- Camera and competition domain differ from a stationary youth/amateur end-line phone.

See CVPR §3, pp. 2–4 / 5862–5864 and §6, pp. 6–8 / 5866–5868.

### Pieces not reusable as-is

- Beach dataset, source code, trained weights, exact label exports, and FIVB/BEACH-DATA footage.
- Beach accuracy claims as an estimate for indoor 6v6 footage.
- VNL full-resolution footage, trained weights, dataset rights, or a stable one-request download suitable for unattended builds. The resized archive is anonymously downloadable with SharePoint redirect/cookie handling.
- VNL's rally-extraction stage as a complete cutter: it detects court-keypoint presence in a broadcast, not live/dead play in a locked full-match shot.
- The current VNL demo/CLI without repair.
- Any BMVC 24-match data, model implementation, or weights until an actual licensed release exists.

## Known reproducibility defects and gaps

### Beach thesis

| Gap | Evidence | Consequence |
|---|---|---|
| Public attachment is missing | Appendix inventory p. 66 lists code/data directories; the DSpace `ORIGINAL` bundle has only three PDFs. | No source-level reproduction or licensed implementation reuse. |
| Dataset and labels are not publicly released and are third-party controlled | §3.1 cites FIVB and BEACH-DATA sources; no public reuse license was found. | Cannot recreate the reported split without new data/permission. |
| Test-size inconsistency | §4.3 says 46 test videos; §5.4 reports results on 45. The reason is not stated there. | Exact sample accounting is ambiguous. |
| Label policy is noisy | Scouts cut at up to 4× speed, and §6 reports inconsistent rally-start conventions. | Boundary error mixes model error with annotator-policy variance. |
| Physical capture geometry omitted | The paper states coverage/resolution/fps but no camera height, distance, intrinsics, or extrinsics. | We must define and validate our own capture geometry. |
| Training artifact absent | No public environment lock, random seeds, checkpoints, or experiment logs. | Results can guide design but are not independently repeatable from the public record. |

### Current VNL-STES repository at `bbaa249c`

These are source-audit facts, not complaints about the paper's conceptual contribution:

| Defect or drift | Direct evidence | Required repair |
|---|---|---|
| Paper uses a 3-layer bidirectional GRU; published commands instantiate one layer | In [`train_e2e_spatial.py`](https://github.com/hoangqnguyen/spot/blob/bbaa249c285c6bc133b091e50e6da2bf4ed42a39/train_e2e_spatial.py#L367-L380), `gru` means 1 layer and `deeper_gru` means 3. The [README command](https://github.com/hoangqnguyen/spot/blob/bbaa249c285c6bc133b091e50e6da2bf4ed42a39/README.md#L115-L123) and `run.sh` use `-t gru`. | Use and verify `-t deeper_gru`; freeze the corrected config. |
| Inference command has the wrong flag | README uses `--video`; parser requires [`--video_path`](https://github.com/hoangqnguyen/spot/blob/bbaa249c285c6bc133b091e50e6da2bf4ed42a39/inference_on_mp4.py#L495-L503). | Fix documentation or CLI alias. |
| Default inference taxonomy matches neither paper | [`inference_on_mp4.py`](https://github.com/hoangqnguyen/spot/blob/bbaa249c285c6bc133b091e50e6da2bf4ed42a39/inference_on_mp4.py#L30-L47) defaults to 9 actions, including `net`; CVPR has 6 and BMVC has 8 without `net`. | Load an explicit ordered class list from the checkpoint/dataset. |
| Saved config omits the ordered class list | [`store_config`](https://github.com/hoangqnguyen/spot/blob/bbaa249c285c6bc133b091e50e6da2bf4ed42a39/train_e2e_spatial.py#L984-L1004) stores only `num_classes`; the demo relies on defaults. | Persist exact class names/order and all model-head options. |
| README schema disagrees with archive and loader | README shows separate `x` and `y`; archived events and training/evaluation use `event["xy"]`, with the loader silently substituting `[0,0]` when it is absent. | Correct the documentation and add schema validation/conversion before training. |
| No paper weights or frozen release | GitHub has no tags/releases and `.gitignore` excludes `exp/` and `data/`. | Ask authors for the exact paper commit/config/checkpoint, or retrain and publish our own provenance. |
| Dataset identifiers are unexplained | Download is named `vnl_1.0.zip`, its root is `vnl_1.5/`, and scripts also refer to unavailable `vnl_2.0`. | Record checksums and explicitly map every local dataset version. |
| Event accounting has four conflicting totals | Paper: 6,137; split JSON arrays: 6,489; `all.json`: 6,623; summed `num_events`: 5,987, with stale fields in 382 records. | Recompute one canonical manifest from event arrays, document exclusions, and fail validation on stale counts. |
| Frame accounting and coverage drift | Archive: 251,803 JPEGs versus 251,110 reported, including one unlabeled 657-frame rally and 36 one-frame overages; only 329 per-rally CSVs and no MP4s are present. | Validate every rally/frame against the split, quarantine extras, define whether CSVs are optional, and generate MP4s only as a derived artifact if needed. |
| Background class can be duplicated | Archived `class.txt` includes `background`, while the loader/model also add background, producing a duplicate or unused output. | Canonicalize the six foreground classes and create background exactly once. |
| Standalone evaluator is temporal-only | `eval.py` calls temporal `compute_mAPs`, while spatial metrics live elsewhere in the training code. | Build one versioned evaluation entry point for both mTAP and mSAP. |
| Experiment settings drift | Paper/README say batch 16; runnable shell scripts use batch 8. Dependencies have lower bounds but no lock file; cuDNN benchmark is enabled and no global NumPy/Torch seed is set. | Lock environment/hardware/config and add deterministic smoke/regression tests. |

The repository compiles syntactically at the audited commit, but no published checkpoint exists and the full training run was not repeated during this audit, so the paper's numbers were not independently reproduced.

## Concrete adoption plan

### V0 — our rally cutter

This contract is now implemented under [`analysis/`](../../analysis/README.md) as a CPU-first evaluation baseline: constant-frame-rate normalization with provenance, a leakage-safe annotation manifest, court appearance/motion/optical-flow features, a class-weighted temporal logistic model, validation-only decoder selection, immutable held-out evaluation, and `analysis.json` inference output. A synthetic-video CLI run exercised normalization-independent train → validation tuning → held-out evaluation → inference; it validates the plumbing, not accuracy on real volleyball. A deliberately shifted synthetic capture failed to transfer until the training set included compatible variants, reinforcing the domain-shift risk described above.

Keep this contract intact before adding action labels:

1. **Input:** a full fixed-camera match plus a manifest of match/venue/device/source-group metadata.
2. **Ground truth:** half-open rally intervals `[start, end)` under one written policy. Use serve-ball-contact-to-dead-ball as the objective core; padding is a separate product setting. Do not compare its boundary MAE directly with the beach thesis's whistle-to-termination labels.
3. **Features/model:** start with cheap frame/motion features and a deterministic temporal classifier. Add the beach-style five-frame 2.5D visual model only after this baseline is measurable.
4. **Decoder:** smooth probabilities, bridge short gaps, remove implausibly short intervals, then apply configurable pre-roll/post-roll. Never bake padding into model labels.
5. **Splits:** isolate whole source groups across train/validation/test. At minimum group by match, venue/court, device, and recording session.
6. **Metrics:** rally precision/recall, missed and extra rallies per set, interval IoU, start/end MAE, exact rally-count rate, live-play seconds removed, and review time.
7. **Product behavior:** surface confidence and require fast human review; bias toward retaining extra seconds.

### V1 — stronger visual temporal model

- Use a pretrained 2D backbone with short and long temporal windows inspired by the beach thesis.
- Automate court detection/homography so manual corners are optional.
- Train on our end-line footage with hard negatives: warmups, timeouts, between-set activity, adjacent courts, camera bumps, service faults, replayed balls, and celebrations.
- Fine-tune and revalidate for each materially new tournament/viewpoint until held-out evidence shows that the geometry transfers.
- Compare against the deterministic v0 on an untouched match-level test set.

### V2 — optional action spotting

Only after the cutter is useful:

- Ask the VNL authors for a stable anonymous mirror, explicit dataset license, exact annotation/count rules, paper commit, and trained CVPR weights.
- If rights permit, fork the BSD code at a pinned commit, fix the defects above, and pretrain/benchmark serve/receive/set/spike/block event spotting.
- Fine-tune on our fixed end-line recordings; do not assume broadcast-view transfer.
- Use events as supporting signals and user-visible suggestions, not as sole cut boundaries.

### Go/no-go gates once videos arrive

Green-light a learned beta only if, on source-group-held-out videos:

- it removes less live play than the agreed safety threshold;
- rally precision and recall both beat the deterministic baseline;
- review time is materially lower than manual cutting;
- performance does not collapse on a new gym/device/team; and
- all training footage has explicit product-consistent consent and retention terms.

## Questions to send the authors

1. Can they provide a stable anonymous mirror for `vnl_1.0.zip` and publish its SHA-256?
2. What license governs video, annotations, and commercial model training separately?
3. Which commit, dataset version, class order, command, and checkpoint produced CVPR Tables 4–8?
4. Are full-resolution clips available under a research or commercial agreement?
5. Will the BMVC 24-match/eight-class dataset, DSGK code, and weights be released, and under what license?
6. Why is the download named `vnl_1.0.zip` with a `vnl_1.5/` root, and does the repository's unavailable `vnl_2.0` correspond to either published dataset?
7. Which event/frame count is canonical, and what generates the stale `num_events` fields, extra frames/rally, and partial CSV coverage?

## Primary references

- Justýna Frommová, *Automated cutting of beach volleyball match videos*, CTU, 2024: [record](https://hdl.handle.net/10467/114627), [full text](https://dspace.cvut.cz/server/api/core/bitstreams/39ffe6fe-d5aa-4ef9-b1d4-40be5beae72e/content). Relevant: §3.1 pp. 19–24; §§4.2–4.3 pp. 31–35; §§5.3–5.6 pp. 42–51; §6 pp. 52–53; Appendix A pp. 55–61; attachment inventory p. 66.
- Nguyen et al., *VNL-STES: A Benchmark Dataset and Model for Spatiotemporal Event Spotting in Volleyball Analytics*, CVPR Workshops 2025: [paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Nguyen_VNL-STES_A_Benchmark_Dataset_and_Model_for_Spatiotemporal_Event_Spotting_CVPRW_2025_paper.pdf), [project](https://hoangqnguyen.github.io/stes/), [code](https://github.com/hoangqnguyen/spot). Relevant: §§3.1–3.5, PDF pp. 2–4 / proceedings pp. 5862–5864; §4.2, pp. 4–5 / 5864–5865; §5.1, p. 6 / 5866; §6, pp. 6–8 / 5866–5868.
- Jamsrandorj et al., *Spatiotemporal Event Spotting via 3D Heatmaps with Dynamically Shifted Gaussian Kernels*, BMVC 2025: [proceedings page](https://bmvc2025.bmva.org/proceedings/987/), [paper](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_987/paper.pdf), [supplement](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_987/supplementary.zip). Relevant: §3.1 dataset and the release statement, paper pp. 4–5.
