import { parseCutDraft, type CutDraftSeed } from "../components/production-lab/editor/lib/cut-draft.ts";
import { editHistoryKey, readEditHistory, writeEditHistory } from "../components/production-lab/editor/designs/taste/edit-history.ts";
import { addMissingLabServeMarkers, type LabDraft } from "./production-editor-lab-draft.ts";

type LabStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;
export type LabDraftRestore = { status: "ready"; draft: LabDraft }
  | { status: "blocked"; raw: string | null; message: string };

/** Never mount the autosaving editor after an unreadable or failed storage read. */
export function restoreLabDraft(storage: LabStorage, key: string, seed: CutDraftSeed, model: LabDraft): LabDraftRestore {
  let raw: string | null;
  try { raw = storage.getItem(key); }
  catch { return { status: "blocked", raw: null, message: "Browser storage could not be read. Your saved edits have not been changed. Allow storage access and try again." }; }
  if (raw === null) return { status: "ready", draft: model };
  const saved = parseCutDraft(raw, seed);
  if (!saved) return { status: "blocked", raw, message: "The saved edits could not be restored. They have been preserved. Download them before starting a new review, or try again in a version that can read them." };
  const merged = addMissingLabServeMarkers(saved, model);
  if (JSON.stringify(saved.scoreTracking.serveMarkers) !== JSON.stringify(merged.scoreTracking.serveMarkers)) {
    try {
      const historyKey = editHistoryKey(key);
      const rawHistory = storage.getItem(historyKey);
      // React can repeat initialization before the refreshed draft is saved.
      const migrated = readEditHistory(rawHistory, merged, seed);
      if (!migrated.past.length && !migrated.future.length) {
        const history = readEditHistory(rawHistory, saved, seed);
        const refresh = (draft: LabDraft) => addMissingLabServeMarkers(draft, model);
        writeEditHistory(storage, historyKey, { past: history.past.map(refresh), present: merged, future: history.future.map(refresh) });
      }
    } catch { /* A usable draft remains available when history storage is full. */ }
  }
  return { status: "ready", draft: merged };
}

/** Explicit recovery preserves both old records before replacing either one. */
export function startNewLabReview(storage: LabStorage, key: string, model: LabDraft): LabDraftRestore {
  try {
    const raw = storage.getItem(key);
    const history = storage.getItem(editHistoryKey(key));
    if (raw !== null || history !== null) {
      const backupKey = `${key}:unrestored:${crypto.randomUUID()}`;
      storage.setItem(backupKey, JSON.stringify({ draft: raw, history }));
    }
    // Clear history first so it cannot resurrect the old draft on editor mount.
    // The preserved backup remains available even if a later write fails.
    storage.removeItem(editHistoryKey(key));
    storage.setItem(key, JSON.stringify(model));
    return { status: "ready", draft: model };
  } catch {
    return { status: "blocked", raw: null, message: "A new review could not be saved. Your previous data has been preserved. Free browser storage and try again." };
  }
}
