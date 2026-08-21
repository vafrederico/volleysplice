# Side-switch V5/V5-state/V6 proposal review UI — 2026-08-20

## Purpose

The `/side-switch-review` development UI treats frozen V5, production-state V5, and
V6 outputs as independent proposal layers. Earlier `sideSwitches` markers were produced
through heuristic-guided review; they are seed evidence, not an exhaustive inventory
of every physical switch. An unmarked model selection must therefore remain reviewable
rather than being presented as a confirmed false positive.

The existing appearance report is not rewritten. The server reads each frozen
evaluation/feature pair, plus the model artifact required by production-state V5,
resolves every prediction through each feature row's `sourceEventIds`, and attaches it
to the stable appearance-report event ID at request time. This preserves the appearance
report identity and all 1,096 existing NAS-backed review decisions.

## Attached scope

| Layer | Evaluated gaps | Selected proposals | Attached proposals |
| --- | ---: | ---: | ---: |
| V5 `PLAYER-ORIENTATION22` | 352 | 44 | 44 |
| V5 + state `PLAYER-ORIENTATION22+STATE10` | 352 | 38 | 38 |
| V6 `DETECTED-ADAPTIVE29` | 352 | 48 | 48 |
| Three-variant union | — | 61 unique gaps | 61 |
| All three variants | — | 28 unique gaps | 28 |
| At least two variants | — | 41 unique gaps | 41 |
| V5-state only | — | 2 unique gaps | 2 |

All 352 prediction rows from every variant map to report events; there are no orphaned
selected predictions. The model scope is the 11-set retrospective raw-phone group, not
all 30 recordings in the appearance report.

## Review behavior

- The default queue is the 61-gap three-variant union.
- Filters expose each variant, selections shared by at least two variants,
  disagreement, and the full report.
- Every selected recording renders three aligned timelines: V5, V5 + state, and V6.
  Thin ticks retain every review gap for temporal alignment; solid variant-colored
  ticks show that variant's selected proposals.
- Queue badges identify every variant that selected a gap.
- The inspector shows each model's ranker score, frozen threshold, rally-gap order, and
  selected cadence margin.
- The saved decision remains visible and editable. Navigation advances through the
  filtered proposal queue even when every event already has a prior decision.
- Opening or filtering the page does not rewrite the NAS file; autosave starts only
  after the decision map actually changes (or the reviewer presses the save button).
- UI wording distinguishes prior switch markers, previously unmarked heuristic gaps,
  and target-free candidate gaps. It does not call an unmarked gap a gold negative.

The UI asks whether a physical side switch occurs in the video. Model selection,
appearance distance, and earlier markers are evidence only.

## Full-video coverage extension — 2026-08-21

The 11 raw-phone model-feedback bundles contain no explicit `sideSwitches` point
markers. Their 354 saved candidate decisions (35 switch, 317 no-switch, and 2 unclear)
are conditioned on rally-gap candidate generation and therefore cannot measure switches
outside that candidate universe.

The same review page now supports a separate exhaustive pass:

- a human-coverage timeline distinguishes explicit source markers, candidate decisions
  reviewed as switches, and newly placed full-video markers;
- `M` or the visible add button places a millisecond point marker at the video playhead;
- markers can be sought and removed, and each recording can be marked fully reviewed so
  an empty marker list is distinguishable from an unreviewed video;
- the recording picker identifies recordings that still need a continuous pass;
- each selected V5/V5-state/V6 proposal displays its model name and timestamp directly
  on its aligned rail;
- raw model-feedback recordings add production model-label, corrected editor-cut, and
  final-export rails. Corrected cuts retain their range IDs, core/padding structure,
  included/removed state, and ignored-time overlays.

Full-video markers and completion state autosave atomically to
`full-video-side-switch-markers-full-nas-v1.json` beside the appearance report, or to
`VOLLEYCUT_SIDE_SWITCH_MARKERS` when configured. They are deliberately separate from
`appearance-review-decisions-full-nas-v1.json`, whose hash is already bound by frozen
model artifacts. Opening or filtering the page writes neither file.

For a recording marked fully reviewed, the exhaustive human point set is the union of
its existing candidate decisions whose value is `switch` (at the report transition
time) and its new full-video markers. The new artifact stores additions rather than
copying or silently revising the candidate-decision artifact.

## Artifact binding and overrides

The default layers are loaded from immutable V5, V5-state, and V6 paths recorded in
their decision documents. Alternate artifacts can be supplied with:

```text
VOLLEYCUT_SIDE_SWITCH_V5_EVALUATION
VOLLEYCUT_SIDE_SWITCH_V5_FEATURES
VOLLEYCUT_SIDE_SWITCH_V5_STATE_EVALUATION
VOLLEYCUT_SIDE_SWITCH_V5_STATE_FEATURES
VOLLEYCUT_SIDE_SWITCH_V5_STATE_MODEL
VOLLEYCUT_SIDE_SWITCH_V6_EVALUATION
VOLLEYCUT_SIDE_SWITCH_V6_FEATURES
```

Each layer is schema-checked independently, and the feature bytes must match the SHA-256
bound by that evaluation artifact. Production-state V5 additionally requires the
evaluation to bind the model SHA-256 and the model to bind the same feature bytes. If
one artifact set is missing, mismatched, or invalid, the other layers remain available
and the UI displays the layer error. Duplicate mappings are rejected rather than
silently overwriting a proposal.

## Evaluation interpretation

Existing variant precision, recall, and F1 values measure agreement inside the reviewed
rally-gap candidate universe. They do not prove exhaustive full-video precision or
recall because rally proposals and the earlier marker inventory can omit events. The
models remain out of production while the proposal audit is reviewable; unmatched
events should be described as review-decision mismatches unless independently verified,
not as confirmed physical false positives.
