# Dataset and production exposure inventory — 2026-09-23

The current catalogs, label workspaces and feedback repositories contain **44
distinct full/raw recordings**. Excluding two beach recordings leaves **42
recordings, 12.869 hours**. Of these, 34 have usable supervision in separate quality
tiers; eight support inference and inspection without an accuracy claim. Raw,
normalized, pilot, project and cut-export derivatives are not additional independent
recordings.

| Non-beach label tier | Videos | Groups | Intervals | Hours | Permitted interpretation |
|---|---:|---:|---:|---:|---|
| Completed exact labels | 9 | 5 | 361 | 2.625 | Human-reviewed semantic rally boundaries |
| Continuously reviewed drafts | 6 | 4 | 238 | 1.669 | Separate draft-boundary evaluation |
| Reviewed project exports | 19 | 3 | 715 | 5.988 | Coverage and export membership; approximate endpoints |
| Partially reviewed draft | 1 | 1 | 40 | 0.244 | Inspection only; incomplete negatives |
| Unvalidated candidates | 5 | 2 | 282 | 1.436 | Predictions/prelabels, not human gold |
| Unlabeled | 2 | 2 | 0 | 0.907 | Inference only |

Groups overlap across tiers, so the group column must not be summed. The protected
indoor source group contains one completed 39-rally recording and one unvalidated candidate
recording. It remains forbidden for fitting, calibration and selection. The user's
current request authorizes its final evaluation after the operating points are
frozen; the resulting opened evaluation cannot be called an untouched future test.

The existing eight exact recordings are unchanged. The three additional continuous
drafts are source-group-005/source-group-007 recordings within existing source groups. There is **no new
independent exact training group**. The additional August16 project recordings add
one coverage-supervised group. The September17 project is coverage-supervised and
reserved for external evaluation by the experiment protocol.

## Authority and media lineage

The authoritative label order is task template, editable full draft, completed
full-v1, editable full-v3, completed full-v3. All competing revisions and their
hashes are retained. A completed export-derived document still has a coverage tier;
its status does not certify serve-contact/dead-ball endpoints. The September17
import explicitly disclaims independent endpoint review.

Feedback authority uses the latest `corrections.updatedAt`, then `generatedAt`.
Equal-time contradictory target revisions fail rather than silently selecting one.
Raw and copied project bundles are deduplicated by source identity. Every one of
the 19 selected feedback sources and all revision aliases matched the raw file's
size and first/last1MiB fingerprint. Full media bytes are not rehashed by this
inventory; new feature preparation must supply that stronger binding.

All eleven August16 recordings visibly show grass courts in retained fixed60s
frames. Several orientations/courts appear, but the entire capture session stays in
`source-group-006`. The seven August29 files remain
`source-group-004`; September17 remains `source-group-001`. Export-to-source maps
preserve every edit seam. Cut video bytes themselves are not verified against those
maps, so exports are recorded as derivatives and are not separately scored.

The inventory distinguishes OpenCV AV104, native Android DSP AV104 and the
September17 WASM feedback features. Equal dimensionality does not make them
interchangeable. The original18 numerical feature caches remain frozen; new
label-blind preparation has its own PTS feature contract.

## Production exposure

The audit binds the current shipped two ensemble bundles, all six rally/serve/dead
heads, the suppression specialist, serving-side and side-switch assets to their
NAS training metadata. It separates direct fitting, same raw/project source fitting,
same-session fitting, calibration/validation exposure and unknown lineage. Unknown
scope is never classified clean. A separate whole-product classification includes
the serving-side and side-switch heads, which do not determine rally export
membership.

The all-labels-v2 heads fit 11 recordings across six source groups. Suppression
additionally fits five August16 project aliases; therefore all eleven August16
recordings are fitting-related under the conservative session grouping. The old
ensemble's validation/decoder scope is inherited by v2. The suppression decoder
also used `indoor-source-03` for selection: this is the **same protected indoor source
group** as `indoor-source-05`. The latter is fitting-clean for production,
but calibration-related. This historical fact does not authorize current fitting.

| Scorable production comparison panel | Exact | Draft | Coverage | Total |
|---|---:|---:|---:|---:|
| Exclude fitting and same-source/session fitting | 1 | 0 | 8 | 9 |
| Also exclude known calibration/validation exposure | 0 | 0 | 8 | 8 |

The eight coverage recordings are the seven August29 sources and September17.
There is **no completed exact recording clean of both production fitting and
calibration-related exposure**. These facts constrain the claim available from a
fair production comparison. Historical diagnostic viewing is additionally possible;
the clean status describes the pinned fit/calibration lineage, not a guarantee of
never having inspected the footage.

## Separate approximate rally experiment

The user's subsequent request explicitly authorizes trying export corrections as
approximate rally labels for training and selection. A separate proxy manifest
contains **674 individual saved cores:399 August16 and275 August29**. It takes
`corrections.correctedRanges.coreStart/coreEnd` with original cut IDs, requires
included status plus final-export membership, and retains ignored/game-window masks.
It never creates a serve/end boundary by subtracting an ignored span or by splitting
a joined export. Four existing micro artifacts below0.25s are excluded with IDs,
durations and source provenance; all other discarded cores have explicit reasons.

`continuousVideoReviewed=true` in the derived coverage/proxy contract means the
user-confirmed review of all kept/discarded export coverage. It does not mean every
endpoint was independently relabeled. `independentEndpointGold=false` remains
explicit. Protected and September17 sources are rejected by the proxy builder.
Final evaluation uses the original exact/draft/coverage tiers, irrespective of the
experimental training interpretation.

## Reconciliation and verification

The audit rehashed210 dependencies. All eight original exact and three original
draft label hashes, endpoints and ignored spans match the frozen prior manifest.
All seven August29 literal export targets also match. The broader inventory retains
four raw micro-core artifacts omitted by the earlier coverage manifest, plus a
0.0002s game-window tail on one recording; neither difference changes the original18
fitting rows. Fourteen pilot/v0 derivative aliases are bound to their full sources.
All674 proxy endpoint pairs were independently checked against saved feedback.
Eleven focused tests cover tier preservation, protected exclusion, source aliases,
unknown exposure, revision conflicts, edit seams and ignored-boundary preservation.

Artifacts are under
`private-reference-0198`:

| Artifact | SHA-256 |
|---|---|
| `inventory-v2.json` | `fcb6263fbc2b1ce8374d0ca411e25eece6ee029ffebac134b5cc087e69293a85` |
| `export-rally-proxies-v1.json` | `f7d9b515080ae3ac49d2e28363ccf5861b7249e407568f90e8531ac2fee71884` |
| `audit-v2.json` | `352b4b328df61ee1e6c55ab4e22f6bd9f9a9cc2b75e892befecf75c4ae111e12` |

The first inventory and its source snapshot remain preserved. Revision2 fixes the
August16 group identifier inherited from the old generic catalog and makes the
coverage-review scope explicit; no numerical experiment used the first inventory.
No training, new prediction scoring, or source-label modification was performed by
this inventory work.
