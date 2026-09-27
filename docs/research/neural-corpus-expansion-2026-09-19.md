**Neural training corpus expansion — 2026-09-19**

**The next useful step is more of the existing labeled footage, before a larger network.** The first grass/indoor comparison used six recordings from three source groups. A fuller provenance audit finds eight recordings with completed exact labels, and eleven if three historically trained human-reviewed drafts are retained as a separate quality tier. Those eleven double the source-group count to six. That offers more variation in camera, court and play conditions, although improved held-group performance must still be measured rather than assumed.

The six-recording limit was too conservative. Five later additions already have affirmative training consent in the historical all-labels-v2 training manifest; their earlier intake-plan entries did not contain those fields. The first experiment's results remain valid for its frozen scope, but that scope was not the full available training corpus. Missing fields in one intake document were not enough to establish a lack of authorization elsewhere in the provenance chain.

The affirmative source is `private-reference-0121`, SHA-256 `152e62ac25538c1f0d63f4b540fc9253656fb517011e1c89b1e4e153a36d96ff`. It explicitly records `train=true` for Bauos, jA3, ugSoh, PKK and zxtl.

| Cohort, all excluding beach/protected sources | Videos | Labeled intervals | Source groups | Video hours |
|---|---:|---:|---:|---:|
| Completed small-scope experiment | 6 | 235 | 3 | 1.701 |
| Completed exact labels | **8** | **322** | **4** | **2.318** |
| Exact labels plus three historically trained reviewed drafts | **11** | **445** | **6** | **3.193** |
| Including three more continuously reviewed drafts, not yet qualified for this run | 14 | 560 | 6 | 3.987 |

The strict expansion adds `grass-source-03` (31 rallies, existing source-group-007 group) and `indoor-source-08` (56, new source-group-012 group). The next three are `grass-source-05` (43), `indoor-source-06` (39) and `indoor-source-04` (41). They have affirmative historical training evidence and continuous human-review provenance, but their label documents remain in progress. Keep those quality distinctions visible; do not automatically promote them to exact evaluation gold because an earlier model trained on them.

All eleven have reusable current **104-channel audiovisual feature caches**, so the next linear/MLP/TCN experiment needs no new AV feature extraction. Their raw/proxy content identities are distinct and do not intersect the protected source lineage. Frozen DINO features exist only for the original six; the five additions need extraction before repeating the DINO arm. Additional available videos therefore fit the existing 3080 workflow without requiring a larger network or a new input pipeline for the AV models.

**Project exports and raw recordings add a different kind of supervision.** The audit identifies eighteen matched raw/project pairs: eleven from August 16 and seven from August 29. Pair identities pass filename, file-size and sampled-fingerprint checks. A raw file, normalized proxy, project and exported concatenation are one source family, not independent examples. Keep every derivative in the same source-group fold.

The seven August 29 grass recordings already supply 275 usable retained-coverage ranges, 263 approximate serve markers, 33 side switches and three ignored spans. They form one source group. Current AV caches are available for all seven under the exported-project dataset's feature root. The August 16 projects contain corrected inclusion/exclusion decisions and a few manually added ranges; they do not independently certify exact serve/dead-ball boundaries. Their environment is still unknown in the metadata and must be confirmed before inclusion in a grass/indoor-only study. Duplicate revisions have different manual-addition counts, so each revision must be frozen explicitly.

Useful source-clock information is preserved:

- `coreStart/coreEnd`: editable cut boundaries before export padding; they may still be inferred or approximate.
- `keepStart/keepEnd`: padded keep ranges.
- `finalExportIntervals`: final merged/suppressed export coverage, including retained gaps.
- Serve/switch markers: separate event supervision with their own timing provenance.

Do not assume `confirmedModelRanges`, inclusion, a changed cut or a complete status certifies exact event endpoints. In particular, two newer files marked complete retain explicit weak-export provenance. Some feedback bundles also omit the editor's `reviewedCutIds`. The existing importer intentionally keeps coverage, approximate serves, switches and ignored spans separate.

When project JSON is present, concatenated export time can be mapped back to the raw video: within export segment i, source time equals that segment's raw start plus elapsed time since its output start. Split annotations crossing an export seam. Prefer training on the original raw timeline; exported concatenations remove dead time and introduce artificial transitions. A pair of MP4s without the project mapping would need alignment rather than this direct transform.

**Recommended execution sequence.** Freeze the eight completed exact-label recordings as the expanded boundary cohort. Add the three historically reviewed drafts in a declared training-quality tier, with reviewed coverage and uncertain endpoints represented explicitly; keep strict evaluation on qualified exact labels. Compare the unchanged compact models on the same held-source evaluation scope so a different evaluation population cannot masquerade as a data gain. Source-family exclusion applies to every auxiliary row whenever its group is held out. Record this as further development on already inspected data, not a new untouched test.

The user subsequently confirmed that the kept and discarded sections of all project exports were manually reviewed, and authorized continued execution with more videos. The [expanded experiment](./neural-expanded-execution-2026-09-19.md) now includes seven August 29 raw/project pairs through a separate keep-coverage objective. It uses the 252 actual final export intervals without further padding; the 275 association ranges and 263 approximate serves remain separate metadata. Unknown endpoints stay masked. This confirms keep/drop supervision without upgrading padded cuts to exact live/dead boundaries.

For the model-size question, the current compact TCN has **64 hidden channels in five temporal blocks, three output heads and 29,635 trainable parameters**. It reuses the same weights across timestamps. More independent training conditions are a more compelling next experiment than simply widening this network; neither approach guarantees an improvement.

**Inventory limits and artifacts.** The current inventory reconciles 43 unique source recording IDs, including six August 25 intake sources absent from the old 37-record catalog. That is an inventory count, not 43 independent labeled matches. The September A09 directories contain snapshots and predictions over existing labels, not new independent gold. Twenty non-protected label-source hashes were checked against their referenced originals. Candidate-only, partially reviewed, protected and beach material remain separately identified.

The detailed inventory (ledger `private-reference-0122`) and JSON audit (ledger `private-reference-0123`) preserve source groups, label tiers, provenance and hashes. JSON SHA-256: `02bc294d5ececa9e39e0045548ac7c680a7b25e2a9e382fc278d13ba0cccacde`. The v2 table's default-root cache lookup misses five August 29 caches; the paragraph above uses the subsequent verification of all seven in the exported-project feature root. V1 is superseded because its cache lookup used integer ROI coordinates instead of the loader's normalized floats. These are cache-inventory corrections, not changes to source data or labels.

The cache and source-identity supplement (ledger `private-reference-0124`) verifies all seven August 29 caches by full file hash, exact key, 104 feature names, tensor shape and timestamps, and all eleven proposed exact/draft source families against duplicate/protected lineage. Supplemental JSON SHA-256: `5aa69b50c3022cf807491e3c744988f6eca188b30f39f3dfd452baa8050a4713`.

Reproducible metadata audit: [audit-neural-corpus-expansion.py](../../scripts/audit-neural-corpus-expansion.py). Export/raw semantics: [dataset importer](../../analysis/exported_project_dataset.py:484), [cut draft schema](../../prod/src/lib/cut-draft.ts:29), and [source-to-export mapping](../../prod/src/lib/on-device/export.ts:321). The [completed beach-free experiment](./neural-nonbeach-execution-2026-09-19.md) supplies the baseline and remaining failure examples. This corpus audit did not train another model or change annotations.
