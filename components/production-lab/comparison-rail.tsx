"use client";

import { useMemo } from "react";
import { RallyTimeline, type TimelineTrack } from "@/components/rally-timeline";
import { compareLabDraft, labPaddingSensitivity } from "@/lib/production-editor-lab-comparison";
import type { LabConfiguration, ProductionEditorLabTask } from "@/lib/production-editor-lab";
import type { RallyDeskLabTools } from "./editor/designs/taste";
import styles from "./lab.module.css";
import { initialLabDraft } from "@/lib/production-editor-lab-draft";
import { applyPaddingToCachedCuts } from "./editor/lib/cut-draft";

const percent = (value: number) => `${(100 * value).toFixed(1)}%`;
const time = (value: number) => `${value < 0 ? "−" : ""}${Math.floor(Math.abs(value) / 60)}:${(Math.abs(value) % 60).toFixed(1).padStart(4, "0")}`;

export function ComparisonRail({ task, configuration, tools }: { task: ProductionEditorLabTask; configuration: LabConfiguration; tools: RallyDeskLabTools }) {
  const comparison = useMemo(() => compareLabDraft(task, configuration, tools.draft), [task, configuration, tools.draft]);
  const sensitivity = useMemo(() => labPaddingSensitivity(task, configuration, tools.draft), [task, configuration, tools.draft]);
  const baseline = useMemo(() => {
    const base = task.configurations.find(c => c.id === configuration.suppressionBaseId);
    if (!base) return null;
    const draft = applyPaddingToCachedCuts(initialLabDraft(task, base), tools.draft.beforePaddingSeconds, tools.draft.afterPaddingSeconds, task.durationSeconds);
    draft.joinGapSeconds = tools.draft.joinGapSeconds;
    return compareLabDraft(task, base, draft);
  }, [task, configuration.suppressionBaseId, tools.draft.beforePaddingSeconds, tools.draft.afterPaddingSeconds, tools.draft.joinGapSeconds]);
  if (!comparison) return <p className={styles.notice}>Save human labels to enable the comparison rail.</p>;
  const rows = (ranges: { start: number; end: number }[], prefix: string, tone?: "gold" | "ignored") => ranges.map((row, index) => ({ ...row, id: `${prefix}-${index}`, tone,
    title: `${prefix} · ${time(row.start)}–${time(row.end)}` }));
  const tracks: TimelineTrack[] = [
    { id: "saved-human", label: "Saved human", detail: `Core boundaries · export ${time(comparison.humanExportSeconds)}`,
      intervals: rows(comparison.humanCore, "Human core", "gold"), exportIntervals: rows(comparison.humanPadded, "Padded human export") },
    { id: "current-edit", label: "Current edit", detail: `${configuration.label} · export ${time(comparison.exportSeconds)}`, active: true,
      intervals: comparison.segments.map(row => ({ ...row, tone: `model-${row.kind}` as "model-match" | "model-added", selectionId: null,
        title: `${row.kind === "match" ? "Matches padded human export" : "Extra exported footage"} · ${time(row.start)}–${time(row.end)}` })),
      exportIntervals: rows(comparison.model, "Current export"), joinedGapIntervals: rows(comparison.joinedGaps, "Joined short gap"),
      missingHumanIntervals: rows(comparison.missing, "Missed human core") },
    ...(comparison.ignored.length ? [{ id: "human-ignored", label: "Ignored", detail: "Outside evaluation", intervals: rows(comparison.ignored, "Ignored", "ignored") }] : []),
  ];
  return <section className={styles.comparison} aria-label="Performance against saved human labels">
    <h3>Compared with human labels</h3>
    <p>Live export coverage · {comparison.before}s before / {comparison.after}s after · joins under {comparison.joinGapSeconds}s. Click the rail to seek.</p>
    <div className={styles.comparisonMetrics} data-testid="lab-comparison-metrics">
      <span>P_pad <strong>{comparison.hasHumanCore ? percent(comparison.precision) : "—"}</strong></span>
      <span>R_core <strong>{comparison.hasHumanCore ? percent(comparison.recall) : "—"}</strong></span>
      <span>F1_padP_coreR <strong>{comparison.hasHumanCore ? percent(comparison.F1_padP_coreR) : "—"}</strong></span>
      <span>Export <strong>{time(comparison.exportSeconds)}</strong></span>
      <span>vs human <strong>{comparison.deltaSeconds > 0 ? "+" : ""}{time(comparison.deltaSeconds)}</strong></span>
      <span>Extra footage <strong>{time(comparison.extraSeconds)}</strong></span>
      <span>Missed core <strong>{time(comparison.missedSeconds)}</strong></span>
    </div>
    {baseline && <div className={styles.comparisonTable} data-testid="suppression-baseline-comparison">
      <p>Original model versus this combination, including your review edits. Both use the current padding and gap joining.</p>
      <table><thead><tr><th>Version</th><th>P_pad</th><th>R_core</th><th>F1_padP_coreR</th><th>Export</th><th>Extra footage</th><th>Missed core</th></tr></thead>
        <tbody>{[["Original model (unedited)", baseline], ["Current combination / edits", comparison]].map(([label, value]) => {
          const row = value as NonNullable<typeof baseline>;
          return <tr key={String(label)}><td>{String(label)}</td><td>{row.hasHumanCore ? percent(row.precision) : "—"}</td>
            <td>{row.hasHumanCore ? percent(row.recall) : "—"}</td><td>{row.hasHumanCore ? percent(row.F1_padP_coreR) : "—"}</td>
            <td>{time(row.exportSeconds)}</td><td>{time(row.extraSeconds)}</td><td>{time(row.missedSeconds)}</td></tr>;
        })}</tbody></table>
      <p>{time(baseline.exportSeconds - comparison.exportSeconds)} less export · {time(baseline.extraSeconds - comparison.extraSeconds)} less extra footage · {time(comparison.missedSeconds - baseline.missedSeconds)} additional missed human play. Negative values indicate the opposite change.</p>
    </div>}
    <div className={styles.comparisonLegend}><span>Green: matching export</span><span className={styles.extraFootageLegend}>Purple: extra footage vs human labels</span><span>Red marks: missed human core</span><span>Gold: joined gap</span></div>
    <div className={styles.comparisonRail}><RallyTimeline duration={task.durationSeconds} currentTime={tools.currentTime} tracks={tracks} ariaLabel="Human comparison rail"
      onSeek={time => tools.seek(time)} /></div>
    <p>Precision uses padded human export; recall measures retained human core time. This is not rally-event recall. Saved human labels stay fixed while you edit.</p>
    <details><summary>Padding sensitivity: 0, 1, 2 and 3 seconds</summary>
      <div className={styles.comparisonTable}><table><thead><tr><th>Before / after</th><th>P_pad</th><th>R_core</th><th>F1_padP_coreR</th><th>Export</th><th>Human export</th><th>Difference</th></tr></thead>
        <tbody>{sensitivity.map(({ padding, comparison: row }) => row && <tr key={padding}><td>{padding}s / {padding}s</td>
          <td>{row.hasHumanCore ? percent(row.precision) : "—"}</td><td>{row.hasHumanCore ? percent(row.recall) : "—"}</td><td>{row.hasHumanCore ? percent(row.F1_padP_coreR) : "—"}</td>
          <td>{time(row.exportSeconds)}</td><td>{time(row.humanExportSeconds)}</td><td>{row.deltaSeconds > 0 ? "+" : ""}{time(row.deltaSeconds)}</td></tr>)}</tbody></table></div>
    </details>
  </section>;
}
