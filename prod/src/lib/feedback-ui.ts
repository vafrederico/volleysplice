import type { CutDraft } from "./cut-draft.ts";

export const FEEDBACK_UI_SCHEMA_VERSION = 1;
export const MIN_PLAYBACK_RATE = 0.25;
export const MAX_PLAYBACK_RATE = 8;
export type UiJson = null | boolean | number | string | UiJson[] | { [key: string]: UiJson };
export type FeedbackUi = { [key: string]: UiJson };
export type PortablePreferences = Pick<CutDraft,
  "cutPreviewEnabled" | "renderScoreOnFinalVideo" | "renderScoreTimeline" |
  "fadeScoreOverlay" | "playbackRate" | "confidenceReviewThreshold" | "reviewedCutIds"
> & { renderScoreOnPreviewVideo: boolean };

function object(value: UiJson, field: string): FeedbackUi {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${field} must be an object`);
  }
  return value;
}

/** Copy bounded JSON only; never spread an unvalidated extension into app state. */
export function parseFeedbackUi(value: unknown): FeedbackUi {
  let nodes = 0;
  function copy(item: unknown, depth: number): UiJson {
    if (++nodes > 4096 || depth > 16) throw new Error("corrections.ui exceeds JSON limits");
    if (item === null || typeof item === "boolean") return item;
    if (typeof item === "number" && Number.isFinite(item)) return item;
    if (typeof item === "string" && item.length <= 16384) return item;
    if (Array.isArray(item)) return item.map((entry) => copy(entry, depth + 1));
    if (typeof item !== "object" || !item ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(item))) {
      throw new Error("corrections.ui must contain JSON values only");
    }
    const result: FeedbackUi = {};
    for (const [key, entry] of Object.entries(item)) {
      if (["__proto__", "constructor", "prototype"].includes(key)) {
        throw new Error("corrections.ui contains an unsafe key");
      }
      result[key] = copy(entry, depth + 1);
    }
    return result;
  }
  const ui = object(copy(value, 0), "corrections.ui");
  if (new TextEncoder().encode(JSON.stringify(ui)).length > 65536) {
    throw new Error("corrections.ui exceeds 64 KiB");
  }
  if (ui.schemaVersion !== undefined &&
    (typeof ui.schemaVersion !== "number" || !Number.isSafeInteger(ui.schemaVersion) || ui.schemaVersion < 1)) {
    throw new Error("corrections.ui.schemaVersion must be a positive safe integer");
  }
  const preferences = ui.preferences === undefined ? {} : object(ui.preferences, "corrections.ui.preferences");
  for (const key of ["cutPreviewEnabled", "renderScoreOnPreviewVideo", "renderScoreOnFinalVideo", "renderScoreTimeline", "fadeScoreOverlay"]) {
    if (preferences[key] !== undefined && typeof preferences[key] !== "boolean") {
      throw new Error(`corrections.ui.preferences.${key} must be a boolean`);
    }
  }
  for (const [key, min, max] of [
    ["playbackRate", MIN_PLAYBACK_RATE, MAX_PLAYBACK_RATE],
    ["confidenceReviewThreshold", 0, 1],
  ] as const) {
    const number = preferences[key];
    if (number !== undefined && (typeof number !== "number" || number < min || number > max)) {
      throw new Error(`corrections.ui.preferences.${key} must be between ${min} and ${max}`);
    }
  }
  const ids = preferences.reviewedCutIds;
  if (ids !== undefined && (!Array.isArray(ids) || ids.length > 2048 ||
    !ids.every((id) => typeof id === "string" && id.length > 0 && id.length <= 256) ||
    new Set(ids).size !== ids.length)) {
    throw new Error("corrections.ui.preferences.reviewedCutIds must contain unique nonempty IDs");
  }
  return ui;
}

/** Unversioned iOS extensions have v1 layout; future envelopes remain opaque. */
export function portablePreferences(ui?: FeedbackUi): Partial<PortablePreferences> {
  if (!ui || (typeof ui.schemaVersion === "number" && ui.schemaVersion > FEEDBACK_UI_SCHEMA_VERSION)) return {};
  return (ui.preferences ?? {}) as Partial<PortablePreferences>;
}

export function feedbackUiFromDraft(draft: CutDraft): FeedbackUi {
  const retained = draft.portableUi ? parseFeedbackUi(draft.portableUi) : {};
  if (typeof retained.schemaVersion === "number" && retained.schemaVersion > FEEDBACK_UI_SCHEMA_VERSION) return retained;
  return parseFeedbackUi({
    ...retained,
    schemaVersion: FEEDBACK_UI_SCHEMA_VERSION,
    preferences: {
      ...object(retained.preferences ?? {}, "corrections.ui.preferences"),
      cutPreviewEnabled: draft.cutPreviewEnabled,
      renderScoreOnPreviewVideo: draft.renderScoreOverlay,
      renderScoreOnFinalVideo: draft.renderScoreOnFinalVideo,
      renderScoreTimeline: draft.renderScoreTimeline,
      fadeScoreOverlay: draft.fadeScoreOverlay,
      playbackRate: draft.playbackRate,
      confidenceReviewThreshold: draft.confidenceReviewThreshold,
      reviewedCutIds: [...draft.reviewedCutIds],
    },
  });
}
