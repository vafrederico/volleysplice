# Portable review preferences (UI extension v1)

Feedback/project JSON remains `volleycut-model-feedback` schema version **3**.
This additive extension lives at **`corrections.ui`**, matching the existing iOS
`corrections.ui.preferences.chapterOptions` layout. Web local drafts remain version
**16** and optionally retain this envelope in `portableUi`. No suppression, marker,
scoring, padding, or export-union semantics change.

```json
{
  "corrections": {
    "ui": {
      "schemaVersion": 1,
      "preferences": {
        "cutPreviewEnabled": false,
        "renderScoreOnPreviewVideo": true,
        "renderScoreOnFinalVideo": true,
        "renderScoreTimeline": false,
        "fadeScoreOverlay": false,
        "playbackRate": 1,
        "confidenceReviewThreshold": 0.7,
        "reviewedCutIds": []
      }
    }
  }
}
```

## Known fields and precedence

Every preference is optional on input. Emit all known fields when writing v1.
The defaults below are shared defaults for a v1 envelope; explicitly stored values
always win over consumer/device defaults. For a legacy document with no versioned
UI envelope, retain historical consumer defaults: web preview/final score on and
timeline off; Android final score/timeline off; iOS final score/timeline on.
An unversioned iOS UI object is accepted with its existing preference layout;
absent fields keep the legacy consumer defaults. Web exports upgrade it to v1.

| Key under `preferences` | Type, units, allowed values | v1 default | Meaning |
| --- | --- | --- | --- |
| `cutPreviewEnabled` | boolean | `false` | Play the final cut rather than the full source in the editor. |
| `renderScoreOnPreviewVideo` | boolean | `true` | Editor video preview score visibility, independent of output. |
| `renderScoreOnFinalVideo` | boolean | `true` | Burn the score into final output video. |
| `renderScoreTimeline` | boolean | `false` | Include the point timeline in rendered score overlays (preview and final). |
| `fadeScoreOverlay` | boolean | `false` | Fade score visibility around point-timeline events instead of keeping it visible. |
| `playbackRate` | finite number, speed multiplier, inclusive `[0.25, 8]` | `1` | `1.5` means 1.5 times source speed; never percent or milliseconds. |
| `confidenceReviewThreshold` | finite number, probability, inclusive `[0, 1]` | `0.7` | Confidence below which an inferred rally needs review. |
| `reviewedCutIds` | unique nonempty strings; at most 2048 IDs, each at most 256 UTF-16 code units | `[]` | Reviewed inferred/cached cut identities from `correctedRanges`, not display labels or marker IDs. Each must identify an inferred/cached cut. |

Web maps `renderScoreOnPreviewVideo` to local `draft.renderScoreOverlay`. Native
ports must map **`renderScoreOnFinalVideo`** to their native `renderScoreOverlay`
output switch; they must not map the web preview flag to that switch. Score
tracking `enabled` remains a separate gate: disabling it hides output without
erasing preferences. Timeline/fade choices remain stored when output is disabled.

The PR8 root field `editorPreferences.renderScoreOnFinalVideo` is a compatibility
alias. A supported UI preference wins when both are present, even if they disagree.
If the canonical final flag is absent, use the alias before the default. Validate
the alias even when shadowed. Web writes the alias from the same active final-score
value as the canonical field; it is not a second authoritative flag.

Web supports every playback multiplier in the stated range exactly and adds an
imported rate to the speed selector. The standard menu is `1, 1.5, 2, 4, 8`.
A native consumer unable to activate a valid multiplier must retain its exact wire
value independently of its active rate, disclose any nearest-rate substitution,
and re-export the original until the user explicitly changes that preference.
Never silently rewrite `1.5` to `1`.

## Unknown extensions and versions

`ui` and `ui.preferences` must be JSON objects, never arrays or null. Preserve all
unknown members at either level, recursively, including arrays and null nested
inside extension values. In particular, `preferences.chapterOptions` remains
opaque to web; its existing iOS fields and any nested future options survive.
Web's separate chapter-export controls do not reinterpret or overwrite that blob.

The active web draft fields own supported preferences. `portableUi` is the retained
transport envelope, not another live settings object. On re-export, copy the
envelope and overlay current known values; preserve unknown siblings. Keep this
envelope through source reconnection, draft persistence, project reset, and review
history. Reset returns supported settings to model defaults while retaining the
transport envelope; unknown members and opaque future versions survive unchanged.
Project fade wins over the device-local fade preference on opening/reopening;
undo/redo restores fade from the
review snapshot. Playback/final-cut preview remain transport preferences excluded
from review undo, so undo/redo keeps their current values.

`ui.schemaVersion` must be a positive safe integer. A missing version uses the
existing unversioned layout. An unsupported newer version is retained **whole and
opaque** and re-exported without downgrade or edits. Do not activate its fields;
use legacy consumer defaults and the PR8 alias where available. Web edits to these
preferences remain local for such a document until that UI version is supported.
Reserved known field types/ranges and envelope safety constraints are still
validated; incompatible data is rejected rather than interpreted speculatively.

## Validation and fixtures

Reject invalid recognized values before creating/persisting a project. Do not
coerce strings, numbers, null, or invalid arrays into defaults. Reject duplicate or
missing inferred review identities. The retained UI JSON has these bounds:

- At most 65,536 bytes in compact UTF-8 JSON, including keys and punctuation.
- At most 4096 value nodes, counting the UI object itself; maximum nesting depth
  16 with the UI root at depth zero.
- Each string value has at most 16,384 UTF-16 code units.
- Only JSON null, booleans, finite numbers, strings, arrays, and plain objects.
  Reject keys `__proto__`, `constructor`, and `prototype` at every depth.

Export validates the composed envelope against the same bounds and fails rather
than truncating extension data. No spreading/merging of unvalidated input into
application objects is allowed.

The shared synthetic fixture is
[`tests/fixtures/portable-review-preferences.json`](../tests/fixtures/portable-review-preferences.json).
It contains UI fragments, not media or a complete project. Native port tests can
overlay each fragment on their synthetic feedback-v3 project with inferred cut
`R001`: `portableUi` activates all known nondefault values and preserves nested
chapter options; `legacyIosUi` retains chapters with legacy defaults; `futureUi`
stays opaque; every `invalidUi` entry must fail import. Native owners should add
their own activation/default/persistence tests when adopting the contract.

The web full export/import/re-export regression is in
[`tests/web/prod-model-feedback-project-import.test.mts`](../tests/web/prod-model-feedback-project-import.test.mts).
The real RallyDesk persistence harness is
[`prod/scripts/test-review-persistence-ui.mjs`](../prod/scripts/test-review-persistence-ui.mjs).
Physical browser playback and encoded-video checks remain release checks; these
synthetic tests neither rerun inference nor encode media.
