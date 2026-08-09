# Source-access log — 2026-08-07

**Local date/time zone:** 2026-08-07, America/Los_Angeles

This log records what was actually reachable during the audit. It separates author claims from successful artifact access.

## Access results

| Source | URL | Result |
|---|---|---|
| CTU thesis record | <https://hdl.handle.net/10467/114627> | HTTP 200 after redirect to DSpace. Metadata and rights statement readable. |
| CTU thesis PDF | <https://dspace.cvut.cz/server/api/core/bitstreams/39ffe6fe-d5aa-4ef9-b1d4-40be5beae72e/content> | Downloaded successfully through byte-range requests; 6,841,992 bytes. |
| CTU original bundle API | <https://dspace.cvut.cz/server/api/core/items/644392f7-0059-4ed1-90a9-8d7255f74c20/bundles?size=9999> | Only the thesis PDF and two review PDFs were listed. No source/data archive was exposed. |
| VNL-STES CVPR paper | <https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Nguyen_VNL-STES_A_Benchmark_Dataset_and_Model_for_Spatiotemporal_Event_Spotting_CVPRW_2025_paper.pdf> | HTTP 200; downloaded successfully; 3,301,790 bytes. |
| VNL-STES project page | <https://hoangqnguyen.github.io/stes/> | HTTP 200. Code, Data, and Paper links present. |
| VNL-STES data link | <https://bit.ly/vnlvolley1> | Redirected to a UST SharePoint item named `vnl_1.0.zip`. Anonymous download succeeded after retaining the SharePoint cookies established during redirect; 13,024,307,624 bytes. Simple clients/browser safety tooling may instead stop at an authentication or redirect-interstitial page. No dataset license was present in the archive. |
| VNL-STES code | <https://github.com/hoangqnguyen/spot> | Cloned successfully. Audited `main` at `bbaa249c285c6bc133b091e50e6da2bf4ed42a39` (commit date 2026-03-18). GitHub reported BSD-3-Clause. No tags or releases existed. |
| BMVC proceedings | <https://bmvc2025.bmva.org/proceedings/987/> | HTTP 200. Paper, poster, video, and supplement links present; no code/data link present. |
| BMVC paper | <https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_987/paper.pdf> | Downloaded successfully; 20,003,553 bytes. The data section says the authors plan to release the dataset. |
| BMVC supplement | <https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_987/supplementary.zip> | Downloaded successfully; 66,630,916 bytes. It contains five rendered prediction MP4s plus a 620-byte README, not code, raw annotations, weights, or the full dataset. |

Exact-title and author-repository searches found no separate public implementation or dataset for the 24-match BMVC work on this date. Absence from search is not proof that no private or unindexed artifact exists.

## VNL archive inspection

The downloaded file was named `vnl_1.0.zip`, but its root directory was `vnl_1.5/`. Its contents did not reconcile to one canonical inventory:

| Check | Observed result |
|---|---|
| Event totals | Paper: 6,137; archived split JSON event arrays: 6,489; `all.json`: 6,623; summed `num_events` fields: 5,987. The field was stale in 382 records. |
| Split-array class counts | Serve 1,063; receive 1,699; set 1,375; spike 1,329; block 513; score 510. |
| Frames | 251,803 JPEGs versus the reported/metadata 251,110; one extra unlabeled 657-frame rally and 36 labeled rallies with one additional frame. |
| Other media/metadata | No MP4s; only 329 per-rally CSVs for 1,028 reported rallies. |
| Event schema | Archive and loader use `xy: [x, y]`; repository README example uses separate `x` and `y`. |
| Classes | `class.txt` includes `background`, while the loader/model add background separately. |

These findings show that the archive is genuinely downloadable, but do not establish permission for redistribution or commercial training. Any local import must retain provenance, validate and canonicalize counts/schema, and await a rights decision.

## Immutable identifiers and checksums

```text
Beach thesis PDF
sha256 68fbbec521f9b3bf0695affacfc289d30f8e1b6d664cf652ca6a66c2a7368ba8

CVPR VNL-STES PDF
sha256 5f0a9ba541e12bf591f5a9bd4dd3b650fa275efd361d4eedc5302de51b78c92c

BMVC DSGK PDF
sha256 1df8bdd7451db7d16a9e79202a6ddf2eb6c24ca37f734af55e12cd93c16fde6f

BMVC supplementary ZIP
sha256 abe747de060ed174ddf2654cdede36d0b3ac87ba947fdb3314bd5997630a4510

VNL-STES Git commit
bbaa249c285c6bc133b091e50e6da2bf4ed42a39

VNL-STES vnl_1.0.zip
sha256 not recorded during the 13 GB access audit; reacquire and hash before any local import
```

## Scope and limitations

- No research videos, datasets, or papers were copied into this repository.
- The 13 GB CVPR dataset was anonymously downloadable only with SharePoint redirect/cookie handling. Its layout and schema were inspected, but no dataset license was found and the archive was not copied into this repository.
- No VNL checkpoint was available, so the repository audit was static plus Python syntax compilation; reported paper metrics were not rerun.
- Camera placement statements marked as visual inference in the decision record came from the thesis figures, CVPR figures, and BMVC rendered demonstration clips. They are not measurements of camera height, distance, or lens parameters.
