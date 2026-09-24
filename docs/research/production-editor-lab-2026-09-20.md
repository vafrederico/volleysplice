# Production editor model lab

Purpose: try the production Rally Desk review experience with frozen production and compact model configurations on NAS footage.

## Work checklist

- [x] Inspect the production editor, lab routes and existing model artifacts.
- [x] Copy the production Rally Desk into a lab-owned workspace with isolated persistence.
- [x] Prepare label-independent model configurations, boundary proposals and removal review queues.
- [x] Add a configuration switch and actionable review flows with seek, apply, restore, adjustment and undo.
- [x] Verify model counts, edit isolation, review behavior, desktop/mobile rendering, lab build and production-site static build.
- [x] Record the usable lab URL and implementation limitations.

The first prepared recording is `recording-044`. Human answers never seed a model configuration. A separately requested **Human export** reference loads the saved human label document into its own editable lab draft.

## Try it

Open the editor lab (resolve the recording index through the private ledger), or use **Editor lab** in this recording's labeling header. The existing lab server serves the NAS video; no production application server is needed.

| Configuration | Starting rallies outside ignored footage | Review behavior |
| --- | ---: | --- |
| Production ensemble | 52 | Original ensemble and production serve predictions; suppression off. |
| Suppression · conservative | 46 | Native production suppression; six removed parents can be restored. Balanced is identical here and shares this choice. |
| Suppression · aggressive | 40 | Native production suppression; twelve removed parents can be restored. |
| Compact review | 40 | Aggressive production cores initially; nine recommended parent proposals, with all 27 flagged parents available. Apply a proposal or keep the original boundaries. |
| Boundary edits + removal review | 42 | All frozen compact boundary edits already applied; review 60 removed core fragments and two explicit split decisions. Restore footage or undo a split when it is harmful. |
| Compact standalone | 34 | Compact's own events, with fresh serving-side inference at their starts. |
| Human export | 37 | Saved human regions and 36 saved serve markers, preserving their original timestamps and manual/model provenance. Lab edits never write back to the labels. |

The copied production controls support boundary adjustment, manual rallies, serves and sides, score tracking, padding, final-cut preview, and undo/redo. Review decisions participate in the same undo history as geometry. **Play with context** plays original source footage, including removed sections, then stops at the context end. Choosing a queue region seeks its source context. Four compact signals can be expanded and clicked to seek.

The review-region bar and expanded signal graph both show a red video playhead and current source timestamp. Both follow playback and seeking; signal percentages follow the nearest native sample. When playback leaves the displayed region, the position label says so and the marker is hidden instead of being pinned to a misleading edge.

Each configuration has its own browser-local draft, keyed by recording, frozen source revision, and configuration. Switching modes or reloading preserves edits without applying them to another mode. **Download lab edits** saves the current editable draft and review decisions as JSON. The labeling tool's human labels are never written by this route.

The human reference uses the saved draft when present, otherwise completed labels; unsaved labeling-UI edits are not included. A model/prelabel fallback is never called a human export. Its own label revision and ignored intervals are used without resetting model drafts. Human coverage regions are shown as human regions, without invented model-confidence percentages. These imported human endpoints have not been independently relabeled as exact serve/dead-ball gold.

## Serving-side inference at neural starts

The frozen production `serving-side-fixed-flight-v3` model was run on the neural events themselves: all 46 boundary-edited events and all 34 compact standalone events. Its production browser sequential decoder sampled 1,248 native timestamps from the original 4,373,996,025-byte video, using the original full-frame ROI. Production OpenCV feature extraction computes 237 features per event. Each complete candidate population is ranked separately, including ignored candidates before display exclusion. The original two production serve-head outputs provide the gate evidence at each new start. Boundary events retain their production-parent agreement lineage; compact standalone events do not invent that agreement.

| Mode, outside ignored footage | Near | Far | Model requests review | Serve gate rejected |
| --- | ---: | ---: | ---: | ---: |
| Boundary edits + removal review | 20 | 17 | 4 | 1 |
| Compact standalone | 15 | 17 | 0 | 2 |

Rejected serves create no score marker and do not remove the rally. Review decisions remain when the frozen model requests them; unresolved starts are not arbitrarily assigned a side. Compact guidance retains production serve predictions before applying a proposal, then uses the corresponding frozen boundary-population prediction at the proposed start. Manually creating or moving a start later still requires review; inference is not rerun automatically after every edit.

Inference scripts: `scripts/prepare-editor-lab-neural-serves.mjs`, `scripts/editor-lab-neural-serves-browser.ts`, and `scripts/run-editor-lab-neural-serves.mjs`. The last script launches an isolated Chrome context and temporary inference harness, not the production application. Model features and verdicts are preserved in `production-editor-lab-v1/neural-serving-results.json` on the NAS. No human labels or correction fields were used for inference, and no training was performed.

## Boundaries and export behavior

The **Compared with human labels** rail below the editor timeline reuses the labeling tool's `RallyTimeline`. It compares the current editable draft with the saved human reference, updates with edits/restoration/undo and tracks the playhead. Clicking a segment seeks the source video. Matching padded export, extra exported footage, joined gaps, missed human core and ignored time remain separately visible; phones scroll the rail horizontally inside its panel.

Live metrics are `P_pad`, `R_core`, `F1_padP_coreR`, actual export duration, its difference from padded human duration, extra exported footage and omitted human core. These are time-coverage diagnostics, not event recall. The actual production materializer supplies model exports, preserving manual keep edges, suppression overrides and barriers. Only saved human ignored intervals define the scoring universe: manually excluding wanted footage still lowers recall. Human labels remain fixed even while editing the Human export mode. An expandable sensitivity table reports precision, recall, F1, model/human durations and their difference for each symmetric 0/1/2/3-second padding case at the same gap threshold. The primary live case is the editor's current padding (initially 2/2 seconds).

The default product padding is two seconds before and after. The copied production materializer retains its strict less-than-three-second gap joining, ignored-interval handling and native suppression barriers. Exact exports for production, conservative and aggressive suppression match the frozen production replay at all four symmetric padding cases: zero, one, two and three seconds.

The removal queue compares **raw rally cores**, before padding or joining. A trimmed start/end or internal gap remains reviewable even if padding preserves its footage in the final cut. The two split decisions remain separate from keep/remove decisions, so export joining does not conceal event identity or the need for a new serve. A restored gap can merge linked child events; obsolete linked serve markers are removed. Moving a start or creating a new rally does not inherit a serving-side answer from a different anchor.

## Implementation and source identity

- Lab-owned editor snapshot: `components/production-lab/editor/`, copied from the actual production Rally Desk in `prod/src/designs/taste/index.tsx`. Regenerate with `node scripts/snapshot-production-lab-editor.mjs`; the generator includes the lab adapters. Production source files are unchanged.
- Route and wrapper: `app/editor-lab/`, `components/production-lab/`.
- Validated model API and geometry: `app/api/editor-lab/tasks/[id]/route.ts`, `lib/production-editor-lab.ts`, `lib/production-editor-lab-draft.ts`, `lib/server/production-editor-lab.ts`.
- Manifest preparation: `scripts/prepare-production-editor-lab.py`; copies only model outputs, existing ignored spans, serving predictions and native signal timestamps. No simulated human decisions are imported.
- Active manifest: `private-reference-0195`.
- Manifest SHA-256: `d525bbdb967662cc878ee3dcd5da1244d1f532e9f65d521cf6cd1af1a2b744de`.
- Previous geometry manifest `manifest-guidance-v3.json` is preserved. Its revision remains the storage identity for unchanged production/suppression modes; neural modes get a new revision for the fresh serving predictions.
- Server environment: `VOLLEYCUT_EDITOR_LAB_MANIFEST_PATH`, configured outside Git in the root `.env.local` using the WSL NAS path.

## Validation and limitations

All 31 focused tests passed, including frozen NAS fixtures, suppression export parity, stale-edit guards, ignored spans, manual-rally identity, native split serving markers, human-reference preservation, neural serving-anchor validation, live comparison exclusions/strict gap joining and isolated persistence. Full root TypeScript validation passed. The lab's Next.js production build and the separate production-site static build passed. `git diff --check` passed.

Real Chrome interaction checks passed for NAS playback, separate mode drafts, reload persistence, proposal application, removal restoration, native suppression overrides, split undo, source-context stopping, keyboard undo, desktop/mobile layout and no browser page errors. Reproduce with:

```powershell
node scripts/test-production-editor-lab-browser.mjs --harness private-reference-0109 --output artifacts/private-media/editor-lab
```

The harness uses an existing Playwright installation and a fresh browser context. Its verification receipt and desktop/mobile screenshots are ignored local artifacts in `artifacts/private-media/editor-lab/`.

The human label file was unchanged: SHA-256 `97ab7adce70f57ad521add3a18f18765abec61d36b2ba31cf0c739638164d94c`.

This first UX prototype uses frozen predictions for this recording-044 recording. The editor does not rerun inference during editing, render an MP4, or save a production project archive. Those unavailable export actions are explicitly disabled; lab JSON and YouTube chapter downloads are available. Browser drafts are local to the browser and origin.

The initial production-site build attempt was blocked by disk exhaustion while installing dependencies. After the user freed space, C: had about 9.5 GiB available. Retrying `npm ci --no-audit --no-fund` and `npm run build` in `prod/` succeeded, including TypeScript, Vite, and postbuild checks for portable assets, integrity hashes, relative paths and Cloudflare asset limits. No production source or lockfile changed. The earlier disk-space blocker is resolved.
