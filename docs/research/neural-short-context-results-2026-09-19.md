# Shorter temporal context: execution and results

Reducing the temporal receptive field from 125 to 33 ticks is rejected for the tested reviewed-export/short-boost configuration. Compact TCN mean `F1_padP_coreR` fell from 0.903077 to 0.856701, and DINO+TCN fell from 0.925127 to 0.894519. F1 declined in every seed and every source group for both architectures, and at every required padding case. Both standard F1 and retention-recovery screens failed, as did cross-architecture replication. All 24 new inner selections were feasible; this was a quality failure after successful execution and independent audits.

Compact short-rally complete losses improved from 21.67 to 18.67 per seed on average, but overall complete losses rose from 33.00 to 34.33 and incomplete losses from 75.00 to 79.00. Longer-rally retained core time fell from 93.82% to 91.45%. DINO worsened on both short and overall complete losses: 11.00 to 18.33 and 13.67 to 26.67 respectively, with incomplete losses rising from 43.00 to 56.33. Its longer-rally recall fell from 97.65% to 95.64%. These recall values already include the product's two-second padding and strict-under-three-second gap joining.

The smaller context also increased mean export duration by 243.14 seconds for compact and 60.41 seconds for DINO across the eight evaluation recordings. Padded precision and event F1 fell in both. Retaining more output time therefore did not recover more useful play overall. Incomplete losses include completely and partially lost rallies; partial-loss counts alone should not be interpreted as total missed rallies.

This was the first of two separately registered follow-ups selected after the [completed short-rally weighting comparison](./neural-short-boost-transfer-results-2026-09-19.md). The [prospective protocol](./neural-short-context-protocol-2026-09-19.md) fixed the reviewed-export cohort, short-boost loss arm, both architectures and three seeds before this intervention's outcomes. The separate [keep-head rescue experiment](./neural-keep-rescue-results-2026-09-19.md) passed recovery for both architectures in the reviewed-export cohort. Its decoder intervention was evaluated against baseline-loss models; these studies do not test stacking keep rescue with short-boost weighting or shorter context.

The sole model change reduces the temporal receptive field from 125 to 33 ticks at 4 Hz by changing dilation values. Inputs, labels, trainable parameter counts, initialization, training chunks and optimizer remain fixed. The experiment compares each shorter-context model with its own immutable original-context result. It does not combine the separate keep-head rescue decoder with shorter context.

The actual-data GPU preflight passed for both architectures. The original profile reproduced predetermined historical checkpoint/prediction/history prefixes, and the new profile passed union/single/repeated validation identity, immutable-resume and saved-checkpoint replay checks. Preflight report SHA-256 is `7b1f491226ff1567771706502290ad61d5a850e19008c1967a37ea020174f224`; prospective protocol snapshot is `e4b288741ccd028c547913c5941abd8a4e5c4c70ac3084bd50c3a1c136bff782`.

The 19 analysis sources, exact source/feature/label revision, preflight and reference audits were registered under contract `d3115b34d6a6e5b5447ddde6b6ddc4435ae68d704d1cdc0914d73ac481fe530a` (registration-file SHA-256 `1f17002e3010a38d04bc77bc8fcb68777d6556b13aa9d8aed14dcfdc604b3a35`). Training began at 20:35:01 UTC on 2026-09-19 with one worker. It requires 60 physical fits, corresponding to 96 logical fits, and six new result cells alongside six original-context references.

Execution checklist (Tasks tool unavailable):

- [x] Choose the intervention only after the preceding full audits.
- [x] Freeze the prospective cohort, loss arm, architecture change and comparison rules.
- [x] Pass actual-data engineering qualification and register source/input hashes.
- [x] Finish all 60 physical fits and six candidate cells.
- [x] Pass independent context tensor/scaler/exposure and full selection/interval audits.
- [x] Report paired results, all four padding cases and recovery/transfer checks; bind the final source archive through its verification receipt.
- [x] Restore the temporary WSL configuration without overwriting independent user edits.

Rank at fixed two-second symmetric padding using pooled `F1_padP_coreR`, strict-under-three-second gap joining and ignored-time subtraction. All zero/one/two/three-second padding cases remain required. The protected test remains unopened; no production promotion or phone/browser latency claim is authorized by a development result.

Artifacts: `private-reference-0187`. The preceding verified source archive is `private-reference-0188`, SHA-256 `40b8ff7948c1b287ce5c18df3c89901d778e1207f39fe534227fd129b45dbfb5`.

The compact model has 29,700 trainable parameters and the DINO temporal model has 46,868, each including four output heads; the frozen DINO encoder is additional. Both retain 64 hidden channels and five kernel-five temporal blocks. Dilations change from `(1,2,4,8,16)` to `(1,1,2,2,2)`. This changes neither parameter count nor training chunk size, and does not shorten the precomputed audiovisual feature windows. It is not a phone/browser latency benchmark.

Each architecture uses seeds 3407, 1729 and 20260918 on the same four development source groups. Within each seed, six physical inner fits supply twelve isolated validation views by excluding both associated source groups, and four separate outer refits supply held-out predictions. Epoch and decoder selection stay inside the inner folds. The original-context reference cells are immutable copies of the completed short-boost comparison; no protected-test predictions are generated.

The context tensor auditor's low-level dependency pin required an evidence-only update after the preceding study's independently reviewed identity fix. Its pin changed from the archived pre-fix helper to SHA-256 `da773784aea271f1f19c67f02fa6246c6dc5557d628ef53334ada892f24df8b4`; no audit calculation or registered analysis source changed. Before/after sources and the 13 passing focused tests are preserved under `audit-stage/dependency-repair/`. The context auditor SHA-256 is `06aec8f98164ea8a64a274999dba56de3c2b37921ed35f48cc933b32a5f2fb33`.

Continuous native-host and WSL resource observers began at 20:36:39 UTC, after the first two physical fits. They cover the remaining run; this is not claimed as full coverage from the first fit. Their local NTFS heartbeat avoids the stale Windows NAS-read behavior found during the preceding study. The final source snapshot binds the closed monitoring and configuration-restoration evidence.

All 60 physical fits completed successfully in 3,080.41 seconds (51.34 minutes), with six candidate cells and six immutable references. The original WSL configuration was restored byte-for-byte at 21:26:49 UTC after confirming that the user had not independently edited it. No WSL/service restart was performed; the restored configuration takes effect at its next startup. The restoration receipt and closed inventory are preserved under `runtime-restoration/`.

The observers exited successfully after 100 WSL and 99 native-host samples, covering 20:36:39 through 21:26:23 UTC. They recorded no resource/transport alerts, zero WSL swap use, a minimum 6.697 GiB available in WSL and 3.652 GiB physical memory free on Windows, and maximum GPU memory use of 3,300 MiB. Maximum heartbeat age was 27.36 seconds. This describes the observed run and establishes no cause for preceding interruptions. Closed observer inventory SHA-256: `780295340b45fa433ecf0e902c5dc90ef270a741b47f17a8c094a12719237744`. Restoration receipt SHA-256: `4e3800a08751cb0e95503aadc8f0f9063d4be544cda7a48e8544eb5f621ed194`.

**Primary results and required padding sensitivity**

Primary scores use fixed 2s symmetric padding, strictly <3s gap joining and ignored subtraction. Each seed pools recordings before scoring; displayed means average the three seed scores. The original profile reuses audited reference results. No protected test or production promotion.

| Architecture | Context | F1_padP_coreR | R_core | P_pad | Event F1 |
|---|---|---:|---:|---:|---:|
| tcn | original | 0.903077 | 0.924092 | 0.883046 | 0.694493 |
| tcn | short | 0.856701 | 0.904306 | 0.815767 | 0.577791 |
| dino_tcn | original | 0.925127 | 0.969545 | 0.884748 | 0.741651 |
| dino_tcn | short | 0.894519 | 0.945339 | 0.849909 | 0.679347 |

| Architecture | F1 change | Recall change | F1 screen | Recovery screen |
|---|---:|---:|---|---|
| tcn | -0.046376 | -0.019787 | False | False |
| dino_tcn | -0.030608 | -0.024207 | False | False |

| Architecture | Context | Slice | Complete losses | Partial losses | Core recall |
|---|---|---|---:|---:|---:|
| tcn | original | all | 33.00 | 42.00 | 0.924092 |
| tcn | original | duration_le_3s | 21.67 | 0.67 | 0.688409 |
| tcn | original | duration_gt_3s | 11.33 | 41.33 | 0.938176 |
| tcn | original | ace | 7.00 | 1.00 | 0.703609 |
| tcn | original | service_fault | 12.00 | 0.33 | 0.625392 |
| tcn | short | all | 34.33 | 44.67 | 0.904306 |
| tcn | short | duration_le_3s | 18.67 | 2.67 | 0.734080 |
| tcn | short | duration_gt_3s | 15.67 | 42.00 | 0.914478 |
| tcn | short | ace | 3.00 | 1.67 | 0.832491 |
| tcn | short | service_fault | 11.33 | 1.33 | 0.655564 |
| dino_tcn | original | all | 13.67 | 29.33 | 0.969545 |
| dino_tcn | original | duration_le_3s | 11.00 | 0.67 | 0.853280 |
| dino_tcn | original | duration_gt_3s | 2.67 | 28.67 | 0.976493 |
| dino_tcn | original | ace | 2.67 | 0.67 | 0.894691 |
| dino_tcn | original | service_fault | 6.67 | 0.67 | 0.796644 |
| dino_tcn | short | all | 26.67 | 29.67 | 0.945339 |
| dino_tcn | short | duration_le_3s | 18.33 | 1.33 | 0.760977 |
| dino_tcn | short | duration_gt_3s | 8.33 | 28.33 | 0.956355 |
| dino_tcn | short | ace | 3.67 | 1.00 | 0.842831 |
| dino_tcn | short | service_fault | 11.67 | 0.67 | 0.654121 |

| Architecture | Context | Padding | P_pad | R_core | F1_padP_coreR | Model seconds | Human seconds | Difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tcn | original | 0 | 0.842414 | 0.819957 | 0.830962 | 2324.050 | 2387.151 | -63.101 |
| tcn | original | 1 | 0.864670 | 0.896008 | 0.880023 | 2943.533 | 3031.151 | -87.618 |
| tcn | original | 2 | 0.883046 | 0.924092 | 0.903077 | 3543.467 | 3675.151 | -131.684 |
| tcn | original | 3 | 0.897107 | 0.938826 | 0.917483 | 4145.108 | 4323.263 | -178.155 |
| tcn | short | 0 | 0.781829 | 0.783724 | 0.780754 | 2401.708 | 2387.151 | +14.557 |
| tcn | short | 1 | 0.799294 | 0.873021 | 0.833169 | 3125.014 | 3031.151 | +93.863 |
| tcn | short | 2 | 0.815767 | 0.904306 | 0.856701 | 3786.603 | 3675.151 | +111.452 |
| tcn | short | 3 | 0.834404 | 0.920194 | 0.874296 | 4408.997 | 4323.263 | +85.734 |
| dino_tcn | original | 0 | 0.843366 | 0.862223 | 0.852536 | 2441.886 | 2387.151 | +54.735 |
| dino_tcn | original | 1 | 0.866891 | 0.946347 | 0.904761 | 3101.642 | 3031.151 | +70.491 |
| dino_tcn | original | 2 | 0.884748 | 0.969545 | 0.925127 | 3739.831 | 3675.151 | +64.680 |
| dino_tcn | original | 3 | 0.900763 | 0.978352 | 0.937896 | 4364.456 | 4323.263 | +41.193 |
| dino_tcn | short | 0 | 0.821443 | 0.828893 | 0.823389 | 2414.521 | 2387.151 | +27.370 |
| dino_tcn | short | 1 | 0.835740 | 0.917554 | 0.873840 | 3137.560 | 3031.151 | +106.409 |
| dino_tcn | short | 2 | 0.849909 | 0.945339 | 0.894519 | 3800.238 | 3675.151 | +125.087 |
| dino_tcn | short | 3 | 0.867929 | 0.955106 | 0.908965 | 4413.118 | 4323.263 | +89.855 |

Per-seed and source-group effects, inner-selection feasibility, loss identities and paired difference-in-differences are retained in the JSON artifacts. Context refers to the temporal network only; existing audiovisual feature windows are unchanged.

**Acceptance and consistency checks**

Both architectures fail all four standard F1 conditions: the +0.02 mean gain, at least two positive seeds, a majority of improving source groups, and the overall recall regression limit. The ten recovery checks are reported individually below; no outer absolute recall gate was added after seeing the results.

| Recovery condition | Compact TCN | DINO+TCN |
|---|---|---|
| Mean F1 delta >= -0.005 | fail | fail |
| Mean R_core delta >= -0.005 | fail | fail |
| Overall complete losses reduced >=20% | fail | fail |
| Short complete losses reduced >=20% | fail | fail |
| Overall incomplete losses do not increase | fail | fail |
| Short incomplete losses do not increase | pass | fail |
| Overall complete losses improve in >=2 seeds | pass | fail |
| Short complete losses improve in >=2 seeds | pass | fail |
| Long-rally R_core delta >= -0.005 | fail | fail |
| Mean event-F1 delta >= -0.01 | fail | fail |

| Architecture | Long-rally recall delta | Event-F1 delta | Incomplete-loss delta | Short incomplete-loss delta | Recovery checks passed |
|---|---:|---:|---:|---:|---:|
| tcn | -0.023698 | -0.116703 | +4.00 | -1.00 | 3/10 |
| dino_tcn | -0.020138 | -0.062304 | +13.33 | +8.00 | 0/10 |

All 24 candidate inner choices met the 0.95 recall floor without a fallback. All three compact outer recalls and two of three DINO outer recalls remained below 0.95. These outer values are diagnostics, not additional acceptance rules.

| Architecture | Seed | F1 delta | Short-context R_core | Complete losses, original -> short | Short complete losses, original -> short | Incomplete losses, original -> short |
|---|---:|---:|---:|---|---|---|
| tcn | 3407 | -0.049099 | 0.932941 | 32 -> 27 | 22 -> 16 | 79 -> 63 |
| tcn | 1729 | -0.040912 | 0.927732 | 36 -> 26 | 23 -> 16 | 74 -> 70 |
| tcn | 20260918 | -0.049115 | 0.852245 | 31 -> 50 | 20 -> 24 | 72 -> 104 |
| dino_tcn | 3407 | -0.019065 | 0.980581 | 15 -> 14 | 11 -> 11 | 48 -> 25 |
| dino_tcn | 1729 | -0.027236 | 0.933896 | 18 -> 32 | 16 -> 23 | 46 -> 70 |
| dino_tcn | 20260918 | -0.045523 | 0.921539 | 8 -> 34 | 6 -> 21 | 35 -> 74 |

| Architecture | Source group | Mean F1 delta | Mean R_core delta |
|---|---|---:|---:|
| tcn | source-group-005 | -0.092429 | -0.088878 |
| tcn | source-group-007 | -0.042042 | -0.003941 |
| tcn | source-group-009 | -0.020874 | -0.000156 |
| tcn | source-group-012 | -0.038660 | +0.016286 |
| dino_tcn | source-group-005 | -0.013811 | -0.058320 |
| dino_tcn | source-group-007 | -0.040635 | -0.026832 |
| dino_tcn | source-group-009 | -0.031885 | -0.006164 |
| dino_tcn | source-group-012 | -0.042171 | -0.005706 |

The F1 difference-in-differences is +0.015768 (DINO effect minus compact effect), meaning DINO degraded less. Both within-architecture effects are negative, so this is not successful transfer. Both replication screens fail. The tested long context is retained for further research; the experiment does not establish an optimal receptive field or rule out different multiscale architectures.

**Verification and source archive**

The independent tensor audit verified 60 fresh physical fits: 36 inner owners and 24 outer refits, with 72 isolated logical inner views and 96 logical fresh fits. It checked 168 checkpoint pairs (336 NPZ artifacts), reconstructed scalers/masks/exposure and verified all six immutable reference payloads. The independent summary then replayed all 12 canonical result evaluations, 24 complete inner-grid choices and 24 held-out refit decodes, plus 624 endpoint-sweep padding/scope rows. All 19 registered analysis sources remain unchanged. The earlier auditor dependency-pin repair changes no model or numerical evaluation rule.

The context source archiver was extended only to include completed observer, dependency-repair and configuration-restoration evidence. Its 17 focused tests passed in Windows and WSL; it rejects active observers, changed file inventories, symlinks, model arrays and arbitrary evidence directories. The registered analysis sources were not edited. The source snapshot is under `private-reference-0189`; `verification.json` records its manifest/ZIP hashes, completed-audit bindings, source stability and ZIP round-trip checks. The archive hash is recorded outside this document to avoid a self-referential checksum.

| Evidence | SHA-256 |
|---|---|
| study/report.json | c1210af7b3c018c1be1ac4f65b866a2a0a0286d9f6bc59923ff021e9bc66e6cd |
| study/summary.json | ba1d435cb86c8c3e92e6165c3ad6f412eb708512ef846fcb33385229943db505 |
| study/summary.md | 0cf5e32432e8e2e68b77b2c497303303b51ff58c4c359350b602918c4f7e791d |
| study/loss-identities.json | 85008473718ff19dfb2e4ef81dae2f43faebde9cab46d6ab355b73690673b55a |
| study/tensor-audit-v1.json | 4661cb607b9ecd493eb189729ea6c978c90bf89fcf903b5bfc86cfbb31b55cc3 |
| audit-stage/audits-completed.json | 6bbb2c89020de226edf4b3a402a79c556a7dea5005b5b117b23fae3db3ffa11c |

Three seeds and four previously inspected development source groups support this rejection in the tested setting. They do not establish population significance or a production performance guarantee. Both architectures retain their original parameter counts; target phone/browser latency was not measured. Beach and the protected test remain excluded, and no model was promoted to production.
