# Interactive generalization report renderer

`scripts/render-neural-generalization-report.py` turns an audited flattened result
file into one standalone HTML artifact. It performs no metric aggregation, fitting,
threshold selection or model ranking. The producer must calculate pooled metrics
and declare each complete recording scope before rendering.

The report offers scenario, evaluation-panel, label-policy, production-exposure,
padding, draw/seed and model-precision filters. It plots precision, recall and F1
against the90–100% calibration floors, with model toggles and point tooltips.
Infeasible/incomplete points appear as gaps and explicit crosses without fabricated
metric values. Different completed recording scopes trigger a visible warning.
The table contains export/error/removal durations and exact-rally loss/event metrics
when those quantities exist. Inventory rows expose label quality, production
fitting/calibration lineage and each saved task's fit/calibrate/evaluate/infer roles.

A default-on production overlay includes the fixed shipped comparator only when
panel, label policy, exposure filter, padding and recording IDs exactly match the
chosen neural scope. It ignores neural draw/dtype selectors and explicitly states
that the production configuration is not recalibrated across floors. Unmatched
production scopes never appear as comparison lines.

A notice immediately below the filters distinguishes inference context. Historical
nested evaluation used the original label-derived valid segments for context and
decoding. New deployment and source-variant runs use full-timeline inference with
labels hidden, then remove ignored time only when scoring. Selecting the historical
scenario raises the notice prominently: cross-scenario differences cannot be
attributed to model or training changes alone, even on identical recordings.

The JSON download contains every input row, including infeasible cases and records
beyond the600-row display cap. It is not a filtered or aggregated export. All code,
styles, plots and data are inline; the page makes no network requests and uses no
external dependencies. Values embedded in HTML/JSON are escaped independently.

## Input contract

The top-level object contains `metadata`, `inventory`, `series` and `tasks`.
`inventory` accepts either the inventory object with `records`, or the row array.
Optional `scopes` maps a scope ID to its recording-ID array. Rows may supply
`scopeId` instead of repeating `recordingIds`; the browser shares that array without
aggregating metrics. Unknown scope IDs fail rendering. The JSON download preserves
the original compact representation.

Each series row has:

```json
{
  "model": "mobile_tcn",
  "variant": "baseline",
  "scenario": "baseline",
  "draw": 3407,
  "precision": "fp32",
  "floorPercent": 99,
  "panelId": "matched-external-exact",
  "labelPolicy": "exact-core",
  "productionFilter": "training-clean",
  "paddingSeconds": 2,
  "status": "infeasible",
  "precisionValue": null,
  "recallValue": null,
  "f1Value": null,
  "exportSeconds": null,
  "humanExportSeconds": null,
  "correctlyRemovedSeconds": null,
  "incorrectExportSeconds": null,
  "wantedExportOmittedSeconds": null,
  "missedCoreSeconds": null,
  "completeRallyLosses": null,
  "eventF1": null,
  "recordingIds": [],
  "aggregationScope": "complete prespecified panel",
  "reason": "No calibration candidate met the declared floor"
}
```

The example describes schema only, not an experimental result. Metrics are fractions
between0 and1; durations are seconds. `precision` is the model dtype label, whereas
`precisionValue` is the precision metric. Completed plot points accept status
`available`, `complete`, `feasible` or `passed`, and require all three finite metric
values. All other statuses remain visible gaps. Reviewed export/coverage plots are
titled `Coverage F1`; reviewed drafts use `F1_reviewed`; completed exact labels use
`F1_padP_coreR`.

Task rows should provide
`{id, model, variant, draw, precision, memberships: {fit: [], calibrate: [], evaluate: [], infer: []}}`
using recording IDs. The renderer also recognizes the corresponding
`trainIds`/`trainingIds`, `calibrationIds`, `evaluationIds` and `inferenceIds` aliases.

Optional metadata defaults are `defaultScenario`, `defaultPanelId`,
`defaultLabelPolicy`, `defaultProductionFilter`, `defaultPaddingSeconds`,
`defaultDraw` and `defaultPrecision`. The default padding is2s. Real results require
`metadata.auditPassed=true`; the metadata should also bind the relevant audit and
source artifacts. Synthetic input must use `metadata.syntheticFixture=true` and the
CLI flag `--allow-synthetic`, and is prominently labeled as invented QA data.

```sh
PYTHONDONTWRITEBYTECODE=1 python scripts/render-neural-generalization-report.py \
  --input private-reference-0144 \
  --output private-reference-0149
```

The output is exclusive-create. Each changed rendering uses a new filename, so
previous artifacts stay reviewable.

## Browser verification and publication

The synthetic fixture was exercised in desktop Chrome153 at1440px and a390px mobile
viewport. Tests verified filtering, line toggles,100% infeasibility without zero
metrics, separate coverage warnings, source/exposure lookup, JSON round-trip,
HTML/script escaping, no horizontal page overflow, no page errors and no external
requests. Screenshots were visually inspected. This is desktop responsive-layout
QA, not a physical-phone inference measurement.

The latest synthetic-only artifacts live at
`private-reference-0150`.
The HTML SHA-256 is
`9b48d86a44eb114fe359c2a32232b61f639d1b93adc0c25f98fa9e1ce70acb41`.
`browser-qa.json` binds the artifact and QA script. The fixture has not been
published and must never be presented as model evidence.
This revision also checks compact scopes, the producer's `available` status,
the distinct reviewed-export and reviewed-draft F1 names, and the production
overlay's checkbox and rejection of a mismatched video scope. The historical
context warning and its scenario-dependent emphasis also passed browser checks.

After the final audited data is available, render and verify that real artifact
separately. Publish only the finished artifact through the shared
Publish Review Artifacts skill (ledger `private-reference-0147`)
and retain the service's artifact ID and `viewUrl` in the final report receipt.
