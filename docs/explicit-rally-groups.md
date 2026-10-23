# Explicit rally groups

Feedback/project JSON stays `volleycut-model-feedback` **version 3**. The additive,
optional field **`corrections.correctedRanges[].editableGroupId`** records explicit
editing identity. It is a core range extension, independent of `corrections.ui`,
reviewed IDs, confidence, scores, suppression touch tracking, and export geometry.
Selecting a rally clears cleanup selection. Even a selected overlapping cleanup
suggestion cannot redirect a split part's Keep/Remove control to a shared override.
Android local draft version **10** stores the same optional field on `cuts[]`.
Web local draft version remains unchanged; its range parser, local persistence,
history, editor reconstruction, and feedback writer relay the field.

The shared synthetic port fixture is
[`tests/fixtures/explicit-rally-groups.json`](../tests/fixtures/explicit-rally-groups.json).
It contains no source media. Overlay `correctedRanges` on a synthetic v3 archive;
all four ranges must remain independently selectable even when their padded output
is one interval. The invalid values must fail import, including explicit null.

## Wire rules

- Absence means legacy grouping. Never infer explicit boundaries from touched,
  reviewed, excluded, trimmed, or manually created cuts in old drafts.
- A present value must be a string matching
  `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}` (ASCII, 1–128 characters). It is an opaque
  group token, never a cut ID, display number, timestamp, source ID, or path.
- At most 2048 corrected ranges may carry the field. A participating cut must
  have a nonblank ID of at most 256 UTF-16 code units. Cut IDs must be unique.
  Existing range origins remain strictly `cached-label` or `manual`; Android maps
  `cached-label` to `INFERRED`. Mixed-origin members of one explicit group are valid.
  Web draft parsing requires uniqueness across **all cuts and ignored intervals**.
  Split allocation and preflight use that complete namespace, including unassigned
  and excluded cuts, and skip occupied numbered IDs without a timestamp fallback.
  Score-marker IDs (including their removal tombstones) have their own namespace;
  they and opaque group tokens do not reserve cut/ignored-range IDs.
  Web split actions check the resulting member count and participating IDs before
  changing working state, history or persistence, and explain a rejected split.
  Splitting an assigned cut adds one member; an unassigned cut adds two. Excluded
  members still count. Unassigned legacy cuts may exceed these bounds until an
  action would assign them explicit identity; their IDs must never be shortened
  or renumbered to make a split fit.
- Same nonempty group token means the same editable group, irrespective of current
  inclusion, padding, trimmed position, confidence, review state, or score markers.
  Different tokens, or explicit versus absent tokens, must never auto-merge.
  Unassigned cuts retain the existing connectivity/serve/confidence grouping rules.
  A paused player retains the explicitly selected group instead of selecting again
  from overlapping padding. Playing video continues to follow the playhead.
- Splitting partitions all cuts in the selected editable group by core start at
  the split position. The split target becomes left/right geometry as before.
  Every left member receives one fresh token and every right member another.
  The left cut keeps its immutable ID; only the new right cut gets a new unused
  ID. Repeated splits partition the selected explicit group, never its siblings.
  Android uses `split:` plus a random UUID. Consumers must not parse that prefix.
- Removing a part from output only changes its inclusion. Deleting a cut removes
  its inline membership naturally; no stale ID references remain. Do not transfer
  its metadata to a replacement or renumber existing IDs. Singleton groups remain
  explicit, including when their sibling was deleted.
- Full user-confirmed **Reset project changes / Start this review over** restores
  the saved model geometry and original model order. It clears explicit groups,
  trims, keep/remove choices, split children, and all manual additions, whether
  those additions were split or unsplit. It does not rerun inference. Ordinary
  edits, persistence, relink and archive round trips retain explicit identity.
  Android has no review undo/redo; web history restores split metadata when undoing
  the reset and clears it again on redo. The `fullReset` fixture exercises trimmed
  `[12, 18]` split parts returning to model `[10, 20]`, an unrelated model range,
  and a manual explicit addition disappearing.


Final export materialization does **not** read this field. Existing padding,
overlap, ignored-range subtraction, suppression, and strictly-less-than join-gap
semantics continue to apply. With both split halves included the export union is
unchanged. Removing one exports only the retained half's padded geometry; its
padding can still include time near the split and can join other retained ranges.
Do not trim padding at explicit group boundaries.

## UI relay and integration

This extension does not live inside the UI envelope, so creating a split never
modifies a future UI schema or overwrites chapter options. PR19's Android opaque
UI preservation is retained. The reviewed web PR14 contract and fixture at
`1d079cab58607aec3fcaa2456e77727cbf59c872` were read; this PR does not merge that
branch. Its UI relay and this inline range relay are additive. When combining the
reset handlers, retain PR14's `portableUi` envelope while resetting supported edits;
`resetProjectReview` already carries that optional opaque envelope. Never replace
the envelope with model defaults. Unsupported UI versions stay whole and
opaque. Unknown UI extension objects, arrays and nulls remain unchanged.

Consumers predating this extension may drop it. Web in this change preserves it
through corrected-range import/export and production editor reconstruction; it
does not adopt Android's connectivity-based editor grouping. The shared fixture
and production round-trip tests are the port contract. Physical-device playback,
source-provider permission/relink, and encoded MP4 verification remain
integration/release checks.

## iOS relay

iOS local draft version 8 stores the optional inline field without changing old
draft defaults: absence remains legacy grouping, and a present invalid value
(including null) fails decoding. Project import/export uses the same strict token,
member-count and UTF-16 ID limits. Splitting allocates new clip IDs against both
cuts and ignored intervals. Model-range restore clears explicit groups and manual
additions while retaining immutable original IDs and existing score corrections.

Paused explicit selection survives overlapping padding; selecting a rally clears
cleanup selection. Automatic cleanup suggestions do not replace explicit rally
Keep/Remove controls. Choosing Review cleanup explicitly still opens its separately
labelled cleanup controls. Portable core tests cover grouping, repeated splits,
removal, trimming, archive and local persistence, chapters, reset and limits.
Native Xcode compilation and the SwiftUI interaction require separate Apple
runtime qualification when the Mac lab is available.
