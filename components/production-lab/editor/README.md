# Production editor snapshot for the lab

This is a copy of the `RallyDesk` editor used by `prod/src/App.tsx`. The production
application does not import these files. `snapshot.json` records the original
files, source hashes, copied hashes, and deliberate adaptations.

The lab supplies a `ReadyDesignReview` object rather than invoking the production
project-loading hook. The copied `project-store.ts` and `useDesignReview.ts` contain
types only. Drafts, histories, layout preferences, and tours use lab storage keys.

Wrap the editor in `.production-lab-root`. Give every immutable recording/model
configuration a distinct project and analysis identity, and key the `RallyDesk`
component by that identity so each configuration keeps its own edits. Keep the
review object and `saveDraft` callback stable while that configuration is mounted.

`RallyDesk` accepts a `labTools` render callback. It receives the current draft,
`applyDraft`, `seek`, and selected cut ID. Applying a draft uses the editor's normal
state and undo history. The extension is rendered above the video in the review
workspace; the original trim, split, keep/remove, serve, score, and preview controls
remain available.

The NAS video URL supports playback without loading the entire source as a browser
`File`. The production MP4 export path still requires an attached local source.

`node scripts/snapshot-production-lab-editor.mjs` refreshes the snapshot and the two
logo assets. This overwrites copied files, so update its adaptations when changing
the lab editor itself. It never changes production files.
