export const SCORE_VISIBILITY_STORAGE_KEY = "volleycut:production-lab:export:fade-scores";

export function readScoreVisibilityPreference(fallback = false): boolean {
  try {
    const value = localStorage.getItem(SCORE_VISIBILITY_STORAGE_KEY);
    return value === null ? fallback : value === "true";
  } catch {
    return fallback;
  }
}

export function saveScoreVisibilityPreference(fade: boolean): void {
  try {
    localStorage.setItem(SCORE_VISIBILITY_STORAGE_KEY, String(fade));
  } catch {
    // Keep the current export usable when browser storage is unavailable.
  }
}
