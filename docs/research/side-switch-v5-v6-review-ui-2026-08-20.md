# Side-switch V5/V6 proposal review UI — 2026-08-20

## Purpose

The `/side-switch-review` development UI now treats V5 and V6 outputs as proposal
layers. Earlier `sideSwitches` markers were produced through heuristic-guided review;
they are seed evidence, not an exhaustive inventory of every physical switch. An
unmarked V5/V6 selection must therefore remain reviewable rather than being presented
as a confirmed false positive.

The existing appearance report is not rewritten. The server reads the frozen V5/V6
evaluation and feature artifacts, resolves every prediction through each feature row's
`sourceEventIds`, and attaches it to the stable appearance-report event ID at request
time. This preserves the appearance report identity and all 1,096 existing NAS-backed
review decisions.

## Attached scope

| Layer | Evaluated gaps | Selected proposals | Attached proposals |
| --- | ---: | ---: | ---: |
| V5 `PLAYER-ORIENTATION22` | 352 | 44 | 44 |
| V6 `DETECTED-ADAPTIVE29` | 352 | 48 | 48 |
| V5 ∪ V6 | — | 59 unique gaps | 59 |
| V5 ∩ V6 | — | 33 unique gaps | 33 |
| V5/V6 disagreement | — | 26 unique gaps | 26 |

All 352 prediction rows from each model map to report events; there are no orphaned
selected predictions. The model scope is the 11-set retrospective raw-phone group, not
all 30 recordings in the appearance report.

## Review behavior

- The default queue is the 59-gap V5/V6 union.
- Filters expose V5, V6, intersection, disagreement, and the full report.
- Queue badges and timeline outlines identify which model selected a gap.
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

## Artifact binding and overrides

The default layers are loaded from the immutable V5/V6 feature and evaluation paths
recorded in their decision documents. Alternate artifacts can be supplied with:

```text
VOLLEYCUT_SIDE_SWITCH_V5_EVALUATION
VOLLEYCUT_SIDE_SWITCH_V5_FEATURES
VOLLEYCUT_SIDE_SWITCH_V6_EVALUATION
VOLLEYCUT_SIDE_SWITCH_V6_FEATURES
```

Each layer is schema-checked independently, and the feature bytes must match the SHA-256
bound by that evaluation artifact. If one artifact pair is missing, mismatched, or
invalid, the other remains available and the UI displays the layer error. Duplicate
mappings are rejected rather than silently overwriting a proposal.

## Evaluation interpretation

Existing V5/V6 precision, recall, and F1 values measure agreement inside the reviewed
rally-gap candidate universe. They do not prove exhaustive full-video precision or
recall because rally proposals and the earlier marker inventory can omit events. The
models remain out of production while the proposal audit is reviewable; unmatched
events should be described as review-decision mismatches unless independently verified,
not as confirmed physical false positives.
