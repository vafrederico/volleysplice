# Suppression product implementation plan

Status: production web implementation completed; native Android remains planned.

## Outcome

Add the corrected suppression specialist to the existing production ensemble as an
optional, review-first export feature. The base rally predictions remain the current
previous-production + all-labels-v2 ensemble. Neither v3 candidate is part of this
work.

Both clients will offer the same four choices:

| Product label | Recorded policy | Agreement construction | Relative effect |
| --- | --- | --- | --- |
| No suppression | None | No suppression candidate is applied | Existing production output |
| Conservative | Zero non-exempt misses | 2.0 s agreement padding; join positive gaps strictly below 0.5 s | Fewest suggestions |
| Balanced | Aggressive intermediate | 1.5 s agreement padding; join positive gaps strictly below 0.5 s | More suggestions |
| Aggressive | Raw connected | No agreement padding or joining | Most suggestions |

The historical policy names stay in diagnostics, feedback JSON, and help text. The
shorter product labels are easier to scan in the editor. If all three policies produce
identical normalized suggestion intervals for a recording, show one **Suppression
suggestions** choice instead of a redundant aggressiveness selector and resolve it
internally with the conservative policy.

## Fixed model and interval contract

The implementation must reproduce the held decision rather than reselecting a model
or decoder:

- Keep the current production ensemble and preserve both models' raw, source-tagged
  rally intervals.
- Add the corrected overlap-safe suppression artifact
  `suppression-overlap-exclusion-retrained`, artifact SHA
  `39eddf58163901930434ea422a802686ae921ae1e8fe23c20a3c12e5f453da93`
  and weights SHA
  `a943749b69c98a1bc926f8efc9fe60c67c1226534fe09a892c632519217aa3bb`.
- Hold the previous decoder: 1.0 s smoothing, 0.75 enter threshold, 0.65 exit
  threshold, 0.5 s minimum live duration, 0.5 s bridge gap, 0.25 s short-event
  minimum, and 0.9 short-event threshold.
- Do not use the later re-tuned decoder or either v3 candidate.
- Keep the one-model-only eligibility gate for every suppression policy. The
  [gate ablation](research/suppression-eligibility-gate-ablation-2026-08-18.md)
  found that removing it collapses all policies to the same output and causes 75
  complete plus 101 partial non-exempt rally losses at the held decoder.

For each policy, build agreement components from the source-tagged raw production
intervals. A component is protected as supported by both models when any interval
from each model occurs anywhere in that connected component. That protection covers
the complete component, including non-overlapping heads and tails.

Let:

- `P` be the raw union of both production models;
- `S` be decoded suppression time, clipped to `P`;
- `E_policy` be raw production time in one-model-only agreement components for the
  selected policy; and
- `C_policy = P intersect S intersect E_policy` be the suggested removal spans.

All time comparisons use integer milliseconds at the persisted boundary. A positive
gap exactly equal to a threshold is not joined.

## Product behavior

### Default and selection behavior

1. A new or migrated project opens with **No suppression** selected. Its export is
   byte-for-byte equivalent at the interval-list level to today's production path.
2. Selecting Conservative, Balanced, or Aggressive shows that policy's exact
   `C_policy` spans in red on both timelines.
3. An untouched suggestion is initially applied: that raw span is pre-disabled and
   removed from the derived export. This is the requested pre-disable behavior.
4. A user can change each suggestion between **Suppress** and **Keep**. This writes an
   explicit override; it is not inferred later from the current export shape.
5. Selecting **No suppression** makes every suggestion dormant without deleting its
   review state. Returning to a suppression policy restores the user's prior choices.
6. Switching aggressiveness preserves overrides for the same logical suppression
   event. A newly eligible, untouched suggestion defaults to Suppress. A suggestion
   absent from the new policy becomes dormant, not deleted.
7. If a user already changed an overlapping inferred rally before suppression was
   enabled, default suppression must not override that edit. Show the red suggestion
   as **Kept because this rally was already edited** until the user explicitly chooses
   Suppress.
8. Manual ranges always win over automatic suppression. They are unioned back into
   the raw keep set and can only be removed through the normal manual-cut controls.

“Already interacted” means either an explicit Keep/Suppress choice for the logical
suggestion or a recorded user edit to an overlapping inferred rally: enable/disable,
boundary change, or manual replacement. Merely opening, seeking, previewing, or
selecting a range does not count as interaction.

For legacy drafts, mark only materially changed inferred ranges as touched by
comparing them with the seed range and current global padding. Unchanged legacy
ranges remain eligible for the new default. This is the safest migration that still
allows the feature to work on existing projects.

### Timeline presentation

Red is reserved for suppression suggestions; ignored-source styling must be changed
to a neutral charcoal crosshatch so the two meanings cannot be confused.

- An applied suggestion uses a solid red fill plus diagonal hatching and a
  **Suppressed** label/icon.
- A suggestion kept by the user uses a red outline with lighter hatching and a
  **Suggestion kept** label/icon. It stays visible because it still needs to be
  recognizable as model advice.
- A suggestion protected by an earlier rally edit uses the outlined treatment and an
  **Edited rally—kept** badge.
- The exact suggestion span overlays the production timeline. If it covers only part
  of a rally, only that part is red; the parent rally card receives a red suggestion
  badge.
- The whole-game and focused timelines use the same semantics. Clicking/tapping a red
  span selects the suggestion rather than the underlying rally.
- Color is never the only signal: patterns, labels, state text, accessible names, and
  a legend distinguish applied, kept, ignored, and ordinary removed ranges.

The editor summary should show the active policy, suggestion count, applied count,
and derived export time change. The time value must come from the final interval
materializer, because final padding and short-gap joining can retain some time inside
a raw red span.

### Suppression-only navigation

When a suppression policy is active and has at least one suggestion, show a prominent
**Next suppression** button with a `current / total` counter. Also expose Previous in
the focused suggestion controls; the one-button compact/mobile layout may put Previous
in the overflow menu.

Pressing Next suppression will:

1. traverse only active-policy suppression suggestions in source-time order;
2. wrap from the last suggestion to the first;
3. select the red span and center the focused timeline on it;
4. pause playback and seek to `max(gameStart, suggestionStart - 2 seconds)` for
   review context; and
5. leave its Keep/Suppress state unchanged.

The queue includes every active-policy suggestion, including suggestions already kept
or removed by another edit, so a full pass really reviews the entire suppression set.
The focused card explains its current effective state. If there are no active
suggestions, show **No suppression suggestions for this game** and hide the navigation
controls. Selecting No suppression also hides them.

## Shared data contract

Implement the policy algebra twice, but freeze it with one language-neutral golden
fixture consumed by TypeScript and JVM tests. Do not make the UI responsible for
interval math.

The persisted analysis needs these additional fields:

```text
productionComponents:
  allLabelsV2: scored raw intervals
  previousProduction: scored raw intervals
suppression:
  modelId, artifactSha256, weightsSha256, decoderVersion
  probabilities aligned to inferenceTimes
  decoded scored intervals
  suggestions[]:
    logicalId, start, end, score, sourceProductionIds, eligiblePolicyIds
policyContractVersion
```

`logicalId` must be independent of the selected policy. Derive it from the decoded
suppression event identity plus intersecting raw production component IDs, using
millisecond-normalized boundaries. This lets the same user decision survive a policy
switch even when the displayed suggestion is clipped into different fragments.

The draft needs a separate suppression layer instead of changing `EditableCut`:

```text
selectedSuppressionPolicy: none | conservative | balanced | aggressive
suppressionDecisionOverrides: logicalId -> keep | suppress
userTouchedCutIds: set of inferred cut IDs
suppressionContractVersion
```

Absence from `suppressionDecisionOverrides` means “use the default”: suppress when an
active policy makes the event eligible, except when it overlaps a touched inferred
cut. This tri-state representation is necessary; a plain Boolean cannot distinguish
a default from a deliberate user choice.

Do not add the suppression artifact to the base production model ID. Store a separate
suppression identity so old production analyses remain recognizable and can be
augmented from cached features without pretending their base predictions changed.

## Final interval materialization

Refactor web `buildFinalCutIntervals` and Android `EditorMath.finalIntervals` around
the same ordered operations:

1. Start from enabled inferred **raw core** intervals.
2. If a suppression policy is active, subtract applied `C_policy` fragments.
3. Apply current before/after padding to every remaining inferred fragment, clip to
   the game window, and merge overlapping or touching ranges.
4. Union enabled manual ranges. Automatic suppression never subtracts them.
5. Join positive gaps strictly below the user's export join threshold (3 seconds by
   default). A gap of exactly 3 seconds remains a cut.
6. Subtract ignored intervals and never rejoin across an ignored interval.

This order preserves the evaluated product contract: agreement padding determines
whether suppression is eligible, while export padding is applied only after the raw
candidate is suppressed. Existing per-rally padding edits remain outer-boundary
overrides. If suppression splits a rally, new internal fragment edges receive the
current global padding; the original custom start/end applies only to the surviving
outer fragment on that side.

The materializer should also return provenance segments so timelines and JSON exports
can explain which output time came from an inferred core, padding, joined gap, manual
range, or suppression override.

## Production web work

### Inference and storage

1. Convert the held suppression artifact to a deterministic browser asset under
   `prod/public/runtime/`; record the source and emitted SHA-256 values in its manifest.
2. Add a single-head loader/runner beside `prod/src/lib/on-device/model.ts`. Reuse the
   existing logistic scoring and probability decoder math, but validate the exact
   feature signature and held decoder constants.
3. Extend `prod/src/lib/on-device/pipeline.ts` to retain each production model's raw
   intervals, run the suppression head on the already contextualized matrix, and call
   a new pure `suppression-policy.ts` module to build all three policies.
4. Extend `OnDeviceAnalysis`, `ProductAnalysis`, and validation in
   `prod/src/lib/on-device/types.ts`, `prod/src/lib/product-analysis.ts`, and
   `prod/src/lib/project-store.ts`. Bump the IndexedDB record schema only if needed;
   keep the base production analysis ID stable.
5. For an old project with retained base features, contextualize those cached values
   and backfill the two raw component predictions plus suppression output without
   decoding media again. If neither retained features nor a valid feature cache is
   available, keep the project usable with No suppression and ask for the source only
   when the user requests suppression augmentation.

### Draft, editor, and export

1. Bump `CUT_DRAFT_VERSION` in `prod/src/lib/cut-draft.ts`; migrate the prior draft,
   add the tri-state suppression fields, and include the suppression contract revision
   in the seed without invalidating unrelated user edits.
2. Split final interval construction into a pure raw materializer and a presentation
   provenance result. All video export, final-cut preview, duration totals, and edit
   list generation must consume that one result.
3. Add the four-choice control and compact identical-policy form to
   `prod/src/components/CutEditor.tsx`. Keep it near the existing padding/join output
   controls so the export effect is clear.
4. Extend the whole and focused timeline markup in `CutEditor.tsx` and styles in
   `CutEditor.module.css` with the red overlays, patterns, legend, suggestion
   selection, and accessible names.
5. Add a suppression-specific selection/navigation cursor. Reuse the existing exact
   seek path, but do not mix this queue with the current low-confidence/disagreement
   review queue.
6. Add Keep/Suppress actions to the focused suggestion card. Updating one suggestion
   records an explicit override and updates preview/export immediately.
7. Include policy, artifact/decoder identity, all candidate spans, applied decisions,
   dormant decisions, and final derived intervals in downloaded edit-list and model
   feedback JSON. Preserve the untouched base production intervals.

## Native Android work

### Inference and project storage

1. Package the same converted suppression weights in
   `android/app/src/main/assets/`, verify its hash at test/build time, and add its
   identity to `FeatureSchema.java` without replacing `FeatureSchema.MODEL_ID`.
2. Add a focused single-head suppression runner or a validated single-head mode to
   `ModelRunner.java`. The held decoder must share golden probability and interval
   fixtures with the browser implementation.
3. Extend `AnalysisEngine.java` and `AnalysisTypes.AnalysisResult` to retain raw ranges
   from both production runners, suppression probabilities/intervals, and the three
   derived policy suggestion sets before creating the merged production display
   ranges.
4. Version `NativeProjectStore.kt` records to persist the added analysis. Existing
   projects remain ready. When a suppression option is requested, use the saved
   contextual feature cache to augment them; do not re-run video/audio extraction on
   a cache hit.
5. Extend `EditorSeed` and the Activity intent/project resume path with suppression
   analysis references. Keep the seed's base-rally revision stable so augmentation
   does not discard an existing editor draft.

### Draft, Compose UI, and export

1. Bump `EDITOR_DRAFT_VERSION` in `EditorModels.kt` and migrate through
   `EditorDraftStore.kt`. Persist selected policy, decision overrides, and touched
   inferred cut IDs atomically with the existing draft.
2. Implement the shared ordered materialization in `EditorMath`; `ExportService`,
   final-cut-only playback, duration summaries, edit-list JSON, and model feedback
   must all consume it.
3. Add the segmented/dropdown suppression control to the OUTPUT card in
   `EditorActivity.kt`. Use a compact single toggle when policy results are equal.
4. Extend `WholeTimeline` and `FocusTimeline` to draw the same red solid/outline
   states and non-color patterns as web. Add Compose semantics describing start, end,
   score, policy, and current Keep/Suppress state.
5. Add a suppression selection type alongside the current `selectedId`. The Next
   suppression action seeks with 2 seconds of pre-roll, focuses the suggestion, and
   does not modify its state. Preserve the existing rally Previous/Next and confidence
   review controls as separate queues.
6. Add Keep/Suppress buttons to the focused suggestion card and update export preview
   immediately. Back handling, rotation, process death, and project relinking must
   restore the same active policy and decisions.
7. Extend `ModelFeedbackExporter.kt` and edit-list JSON with the same fields and wire
   names as web.

## Cross-platform verification

Create one repository fixture containing source-tagged production intervals,
suppression probabilities and decoded events, ignored intervals, touched-cut states,
and expected results for all policies. It must cover:

- partial overlap, containment, and overlap-connected non-overlapping heads/tails;
- positive gaps of 0, 0.499, 0.5, 2.999, and 3.0 seconds;
- suggestions that split an inferred rally;
- suggestions rescued or partly covered by final padding/joining;
- a touched rally, explicit Keep, explicit Suppress, and policy switching;
- manual cuts overlapping a suppression span;
- ignored intervals that must not be rejoined; and
- identical policy sets, empty suggestions, and a single suggestion.

Required automated coverage:

- Web unit tests for artifact parsing, probabilities, held decoding, policy algebra,
  draft migration, final materialization, feedback/edit-list serialization, and the
  suppression navigation selector.
- JVM tests for the same golden fixture in `ProductionEnsembleTest` and
  `EditorMathTest`, plus `EditorDraftStore` migration/round-trip tests.
- Web component tests and Android Compose tests for default No suppression,
  pre-disabled untouched suggestions, preservation of touched decisions, red timeline
  semantics, identical-policy compaction, and Next suppression wrapping.
- An end-to-end parity run on at least one recording where the three policies differ.
  Web and Android must emit identical millisecond-normalized suggestions and final
  interval lists for the same draft state.
- Regression tests proving No suppression produces today's interval list for saved
  fixtures and that MP4 export and final-cut preview use the same list.

## Rollout

1. Land the shared fixture and pure policy/materialization functions first. No UI or
   export behavior changes in this phase.
2. Add artifact packaging, inference augmentation, and persisted schemas behind a
   disabled feature flag on both clients.
3. Add the review UI and navigation, still internal-only. Compare web/Android JSON
   on the inspection corpus and manually inspect red spans.
4. Enable for internal builds with No suppression as the default. Record local
   diagnostic counters in feedback JSON; do not add network telemetry to the offline
   products.
5. Promote to production only after parity, migration, no-suppression regression, and
   export-preview consistency gates pass. Keep an emergency flag that hides the
   suppression controls while leaving stored decisions intact.

## Definition of done

- Web and Android use the same held model, decoder, policy names, interval math, and
  serialization contract.
- New and migrated projects default to No suppression.
- Selecting a suppression level pre-disables untouched suggestions, paints the exact
  spans red, and never overwrites prior rally/suggestion interaction.
- Policy switching and app/browser restart preserve explicit decisions.
- Next suppression visits only active suggestions with context and does not change
  output by itself.
- No suppression exactly reproduces the current production export intervals.
- Preview, duration, edit-list JSON, feedback JSON, and MP4 export all use the same
  final materialized intervals.
- The v3 candidates and re-tuned decoder are absent from production assets and code
  paths.

The canonical model/policy decision and evaluated metrics remain in
`/mnt/freenas/llmvault/memories/VolleyCut Suppression Product Options.md`; this plan is
the implementation handoff for that decision.
