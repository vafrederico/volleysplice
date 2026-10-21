import {
  createCutDraft,
  cutDraftStorageKey,
  cutDraftStorageKeys,
  parseCutDraft,
  type CutDraft,
  type CutDraftSeed,
} from "./cut-draft.ts";
import { withModelScoreMarkers } from "./model-review-draft.ts";
import {
  putProjectReviewDraft,
  type VolleySpliceProject,
} from "./project-store.ts";

export type DraftSaveResult = { saved: boolean; message: string };

// Accessing localStorage itself can throw when browser storage is restricted.
export function browserDraftStorage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function restoreProjectReviewDraft(
  project: VolleySpliceProject,
  seed: CutDraftSeed,
  storage: Pick<Storage, "getItem"> | null,
): CutDraft {
  const candidates: string[] = [];
  for (const key of cutDraftStorageKeys(seed.analysisId)) {
    try {
      const raw = storage?.getItem(key);
      if (raw !== null && raw !== undefined) candidates.push(raw);
    } catch {
      // The project record still provides a restore path.
      break;
    }
  }
  if (project.reviewDraft !== undefined) {
    candidates.push(JSON.stringify(project.reviewDraft));
  }
  const parsed = candidates.map((raw) => ({
    raw,
    draft: parseCutDraft(raw, seed),
  }));
  const drafts = parsed
    .map(({ draft }) => draft)
    .filter((draft): draft is CutDraft => draft !== null)
    .sort(
      (left, right) =>
        (Date.parse(right.updatedAt) || 0) - (Date.parse(left.updatedAt) || 0),
    );
  const latest = drafts[0];
  // Mounting a fresh editor autosaves immediately. Never let an unreadable
  // review be replaced by initial results or an older draft during that mount.
  const unreadableLatest = parsed.some(({ raw, draft }) => {
    if (draft) return false;
    try {
      const updatedAt = Date.parse(JSON.parse(raw)?.updatedAt);
      return (
        !latest ||
        !Number.isFinite(updatedAt) ||
        updatedAt >= (Date.parse(latest.updatedAt) || 0)
      );
    } catch {
      return true;
    }
  });
  if (unreadableLatest) {
    throw new Error(
      "The saved edits could not be restored. They have been preserved. Open an exported project backup to recover your review.",
    );
  }
  if (latest) return withModelScoreMarkers(latest, project.analysis!);
  if (project.importedFeedback) {
    const imported = parseCutDraft(
      JSON.stringify(project.importedFeedback.initialDraft),
      seed,
    );
    if (!imported)
      throw new Error(
        "The imported review could not be restored. Its saved data has been preserved.",
      );
    return withModelScoreMarkers(imported, project.analysis!);
  }
  return withModelScoreMarkers(createCutDraft(seed), project.analysis!);
}

export async function saveProjectReviewDraft(
  projectId: string,
  seed: CutDraftSeed,
  draft: CutDraft,
  storage: Pick<Storage, "setItem"> | null,
): Promise<DraftSaveResult> {
  const persisted = { ...draft, updatedAt: new Date().toISOString() };
  const raw = JSON.stringify(persisted);
  if (!parseCutDraft(raw, seed)) {
    return {
      saved: false,
      message:
        "These edits could not be saved. Export a project backup before closing this tab.",
    };
  }
  let localSaved = false;
  try {
    if (storage) {
      storage.setItem(cutDraftStorageKey(seed.analysisId), raw);
      localSaved = true;
    }
  } catch {
    // The independent project-store write below can still save the review.
  }
  let projectSaved = false;
  try {
    await putProjectReviewDraft(projectId, persisted);
    projectSaved = true;
  } catch {
    // A successful synchronous write is sufficient to restore after reload.
  }
  return localSaved || projectSaved
    ? { saved: true, message: "Saved on this device" }
    : {
        saved: false,
        message:
          "Edits are not saved. Browser storage is unavailable or full. Export a project backup before closing this tab.",
      };
}
