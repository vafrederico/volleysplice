# Web project side-switch evidence

Feedback v1–v3 may include `initialInference.sideSwitch`. Missing or null output
means no raw result was saved. A valid object with zero candidates is a completed
result, including a zero-row matrix. It is not a missing result.

The web parser retains the model ID, fingerprint, feature version, candidate
contract, full Float64 feature matrix, and selected candidate IDs, source
timestamps, probabilities, kinds and source-range IDs. The matrix describes all
proposals; selected candidates can be fewer than matrix rows. Validation rejects
unsupported feature/candidate contracts, wrong dimensions, nonfinite features,
duplicate event IDs, invalid kinds, probabilities or source timestamps, and
malformed numeric encodings with a field-specific import error. Imported model
identity is retained independently of whether the current runtime can reuse it.

`source.generateSideSwitchMarkers` is the original analysis preference. It maps
to the local project's `sideSwitchEnabled` and to the export analysis's same
property. It is a boolean independent of raw output presence. An explicit false
survives import, persistence and re-export even if raw evidence is present. An
absent field uses the historical web default, true; presence of output never
overrides an explicit value. This uses the existing native feedback field name.

Review restoration keeps corrected automatic timestamps, manual markers and
removed-model-marker tombstones. Reset project changes rebuilds markers from raw
inference, restores original automatic timestamps and provenance, clears manual
edits and tombstones, and recalculates scoring. A disabled preference prevents
automatic switch generation on restoration/reset but leaves saved corrections
intact until reset. Legacy bundles without raw output cannot reconstruct missing
automatic events; their existing corrections remain editable.

`tests/fixtures/web-side-switch-project.ts` is synthetic and includes three
proposal rows, two selected switches of different kinds, a moved automatic
marker, a removed automatic marker and a manual marker. The production parser
tests and `prod`'s `npm run test:side-switch-import` cover import, project download,
reimport, reset, undo/redo, scoring and IndexedDB reopen. No source media or model
execution is required by these regressions.

Integration with specialist-recovery work must retain the preference when
calling `withModelScoreMarkers`/`createModelReviewDraft`; raw result presence is
not the enable flag. In particular, its added `scoreRecovery` refresh path in
`useDesignReview` also needs to pass the project's `sideSwitchEnabled`, as the
initial model-draft construction does here. Portable-review-preference changes may overlap the exporter,
parser and importer, but this field belongs under `source`, independently of
`corrections.ui`. Only the three corresponding lab helper/type/exporter copies
and their LF-normalized hash entries are updated; older frozen lab differences
remain intentional.
