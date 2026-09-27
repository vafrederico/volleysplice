import { parseCutDraft, type CutDraft, type CutDraftSeed } from "../../lib/cut-draft.ts";

export const HISTORY_LIMIT = 30;
export type EditHistory = { past: CutDraft[]; present: CutDraft; future: CutDraft[] };
export const editHistoryKey = (projectId: string) => `volleycut:production-lab:edit-history:v1:${encodeURIComponent(projectId)}`;

// Playback preferences and save timestamps are not review edits.
export function editFingerprint(draft: CutDraft): string {
  return JSON.stringify({ ...draft, updatedAt: "", playbackRate: 1, cutPreviewEnabled: false },
    (_key, value) => value && typeof value === "object" && !Array.isArray(value)
      ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b))) : value);
}

export function recordEdit(history: EditHistory, draft: CutDraft): EditHistory {
  if (editFingerprint(history.present) === editFingerprint(draft)) return history;
  return { past: [...history.past, history.present].slice(-HISTORY_LIMIT), present: draft, future: [] };
}

export function moveHistory(history: EditHistory, direction: "undo" | "redo"): EditHistory {
  if (direction === "undo") {
    const previous = history.past.at(-1);
    return previous ? {
      past: history.past.slice(0, -1), present: previous,
      future: [history.present, ...history.future].slice(0, HISTORY_LIMIT),
    } : history;
  }
  const next = history.future[0];
  return next ? {
    past: [...history.past, history.present].slice(-HISTORY_LIMIT), present: next,
    future: history.future.slice(1),
  } : history;
}

export function readEditHistory(raw: string | null, current: CutDraft, seed: CutDraftSeed, normalize: (draft: CutDraft) => CutDraft = (draft) => draft): EditHistory {
  const fresh = { past: [], present: current, future: [] };
  if (!raw) return fresh;
  try {
    const value = JSON.parse(raw);
    if (value?.version !== 1 || !Array.isArray(value.past) || !Array.isArray(value.future) ||
      value.past.length + value.future.length > HISTORY_LIMIT) return fresh;
    const parse = (draft: unknown) => {
      const parsed = parseCutDraft(JSON.stringify(draft), seed);
      return parsed ? normalize(parsed) : null;
    };
    const present = parse(value.present);
    const past = value.past.map(parse);
    const future = value.future.map(parse);
    if (!present || past.some((draft: CutDraft | null) => !draft) || future.some((draft: CutDraft | null) => !draft) ||
      editFingerprint(present) !== editFingerprint(current)) return fresh;
    return { past, present: current, future };
  } catch { return fresh; }
}

export function writeEditHistory(storage: Pick<Storage, "setItem">, key: string, history: EditHistory): boolean {
  try {
    storage.setItem(key, JSON.stringify({ version: 1, ...history }));
    return true;
  } catch { return false; }
}
