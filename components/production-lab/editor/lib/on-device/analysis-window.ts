export type AnalysisWindow = {
  start: number;
  end: number;
};

export const MIN_ANALYSIS_WINDOW_SECONDS = 1;

export function fullAnalysisWindow(duration: number): AnalysisWindow {
  return { start: 0, end: Math.max(0, duration) };
}

export function normalizeAnalysisWindow(
  window: AnalysisWindow | null | undefined,
  duration: number,
): AnalysisWindow {
  const safeDuration = Math.max(0, Number.isFinite(duration) ? duration : 0);
  if (!window || !Number.isFinite(window.start) || !Number.isFinite(window.end)) {
    return fullAnalysisWindow(safeDuration);
  }
  const start = Math.max(0, Math.min(safeDuration, window.start));
  const end = Math.max(start, Math.min(safeDuration, window.end));
  return { start, end };
}

export function isFullAnalysisWindow(
  window: AnalysisWindow,
  duration: number,
): boolean {
  const normalized = normalizeAnalysisWindow(window, duration);
  return normalized.start <= 1e-9 && Math.abs(normalized.end - duration) <= 1e-9;
}
