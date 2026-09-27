import { applyPaddingToCachedCuts, materializeFinalCutIntervals, type CutDraft } from "../components/production-lab/editor/lib/cut-draft.ts";
import { buildLiveTimeComparisonSegments, calculateF1, calculateLiveTimeMetrics, excludeIgnoredTime, padAndMergeRallies, totalRallySeconds } from "./timeline-comparison.ts";
import type { Rally } from "./edit-list.ts";
import type { LabConfiguration, ProductionEditorLabTask } from "./production-editor-lab.ts";

export function compareLabDraft(task: ProductionEditorLabTask, configuration: LabConfiguration, draft: CutDraft) {
  const human = task.configurations.find(row => row.humanReference);
  if (!human) return null;
  // Only the saved human ignored spans define the evaluation universe. User-removed
  // footage is still scored, so excluding wanted footage cannot improve recall.
  const ignored = human.ignoredIntervals ?? task.ignoredIntervals;
  const core: Rally[] = human.events.map(event => ({ ...event, confidence: 1, included: true }));
  const humanCore = excludeIgnoredTime(core, ignored);
  const humanPadded = excludeIgnoredTime(padAndMergeRallies(core, draft.beforePaddingSeconds, draft.afterPaddingSeconds,
    task.durationSeconds, draft.joinGapSeconds), ignored);
  const materialized = materializeFinalCutIntervals(draft, configuration.suppression);
  const model = excludeIgnoredTime(materialized.intervals.map((row, index) => ({ ...row, id: `export-${index}`, confidence: 1, included: true })), ignored);
  const precision = calculateLiveTimeMetrics(model, humanPadded).precision;
  const recall = calculateLiveTimeMetrics(model, humanCore).recall;
  const segments = buildLiveTimeComparisonSegments(model, humanPadded).filter(row => row.kind !== "missed");
  const missing = buildLiveTimeComparisonSegments(model, humanCore).filter(row => row.kind === "missed");
  const exportSeconds = totalRallySeconds(model);
  const humanExportSeconds = totalRallySeconds(humanPadded);
  const joinedGaps = excludeIgnoredTime(materialized.intervals.flatMap(row => row.joinedGaps ?? []).map((row, index) => ({ ...row, id: `gap-${index}`, confidence: 1, included: true })), ignored);
  return { precision, recall, F1_padP_coreR: calculateF1(precision, recall), exportSeconds, humanExportSeconds,
    deltaSeconds: exportSeconds - humanExportSeconds, extraSeconds: segments.filter(row => row.kind === "added").reduce((sum, row) => sum + row.end - row.start, 0),
    missedSeconds: missing.reduce((sum, row) => sum + row.end - row.start, 0), model, humanCore, humanPadded, segments, missing, joinedGaps, ignored,
    before: draft.beforePaddingSeconds, after: draft.afterPaddingSeconds, joinGapSeconds: draft.joinGapSeconds,
    hasHumanCore: totalRallySeconds(humanCore) > 0 };
}

export function labPaddingSensitivity(task: ProductionEditorLabTask, configuration: LabConfiguration, draft: CutDraft) {
  return [0, 1, 2, 3].map(padding => ({ padding,
    comparison: compareLabDraft(task, configuration, applyPaddingToCachedCuts(draft, padding, padding, task.durationSeconds)) }));
}
