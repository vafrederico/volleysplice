# Project interoperability audit

September 11, 2026. This is a compatibility audit, not a claim that all exchanges
are lossless. No application implementation was changed during this audit.

All three platforms export `volleycut-model-feedback` schema v3 as their shared
project exchange format. The iOS `volleycut-ios-project` file in Documents is a
separate local format, not the interchange file. Source media is not embedded;
the receiving device must select/re-link the original recording. Photos asset
identifiers and Files bookmarks are device-local and deliberately not exported.

## Confirmed coverage

| Direction | Evidence | Limit |
| --- | --- | --- |
| iPad → Android | Actual file-picker import of the full iPad export, source re-link, 59 suggested ranges / 52 included logical clips / 688.375s final union | Does not establish lossless reexport or every preference |
| iPad → browser | Historical full-file parser acceptance; current PhotoKit projects also accepted by `importModelFeedbackProject`, with retained features/cuts/score markers | Importer execution, not a new browser file-picker/source-relink UI run; chapter extension is lost |
| Android → browser | Historical actual Android feedback parser acceptance, `docs/android-rally-start-marker-validation.md` | Parser acceptance does not verify full editor/reexport preservation |
| Android → iPad | Shared-v3 implementation and Swift contract tests | Complete real producer-to-consumer UI and reexport cycle not established by reviewed evidence |
| browser → iPad | Shared-v3 implementation and Swift contract tests | Complete real producer-to-consumer UI and reexport cycle not established by reviewed evidence |
| browser → Android | Shared-v3 implementation and Android importer tests | Complete real producer-to-consumer UI and reexport cycle not established by reviewed evidence |

Current focused run: 17/17 tests pass across browser model-feedback project import
and project store suites. iPad feedback from projects DE210934 and 11F6841F imports
through the actual browser project importer. A current 50x104-feature iPad payload
with valid `corrections.ui.preferences.chapterOptions` passes Swift
`VolleyProjectCheck`, preserving the options and immutable features/inference.
Browser validation/import accepts the same file but discards `corrections.ui`.
Reproduction inputs/results: `${VOLLEYCUT_IOS_LAB_ROOT}/artifacts/project-interop-audit/`.

## Confirmed gaps

1. **Android drops imported raw side-switch inference.**
   `ModelFeedbackImporter.kt:133` checks only whether the payload exists to set
   `sideSwitchEnabled`; its NativeProject construction does not restore the
   side-switch output. `ModelFeedbackExporter.kt:268` then emits null. A subsequent
   Android import can set the feature disabled and remove model-origin switch
   markers during seeding (`ScoreTrackingModels.kt:132–137`). This chain was found
   by source inspection, not a new device reproduction. Existing import/cache
   instrumentation does not include this payload.
2. **Chapter preferences are lost in browser and Android.** iOS reads/writes
   `corrections.ui.preferences.chapterOptions`; the browser parser omits it and
   Android rebuilds corrections without it. Browser loss was reproduced with a
   valid Swift-accepted fixture. Android loss is source-confirmed.
3. **Overlay toggles reset differently.** Shared feedback does not preserve the
   score/point-timeline render flags. Android imports both as false
   (`ModelFeedbackImporter.kt:453–454`); iOS imports into defaults of true
   (`ProjectDocument.swift:101–102`). A readable file can therefore produce
   different video-export settings.
4. **Review completion and UI preferences reset.** Shared export/import omits
   reviewed cut IDs, confidence-review threshold, playback rate and final-preview
   preferences. Local native project persistence is separate from this exchange.
5. **Legacy suppression behavior can change.** Android exports
   `suppressionInitialBehavior` but does not restore it on import, defaulting to
   disable-initially. iOS restores it. Importing highlight-only behavior can change
   the retained union for untouched active suppression suggestions. Nested policy,
   explicit decisions, scope overrides and touched cut IDs are imported.

Cross-platform acceptance must compare
features, immutable predictions, cuts/padding/gaps, ignored intervals, suppression
decisions, scoring/marker tombstones, review completion and export preferences
after import **and reexport**, with original media re-linked. Merely accepting the
JSON or passing each platform's self-roundtrip is insufficient.
