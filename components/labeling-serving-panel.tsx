"use client";

import { useState } from "react";
import { formatPreciseTime, type IgnoredInterval } from "@/lib/annotations";
import { servingDecisionExplanation, type ServingPrediction } from "@/lib/labeling-serving";
import styles from "./labeling-serving-panel.module.css";

const verdictLabel = { near: "Near serve", far: "Far serve", review: "Needs review", "not-serve": "Rejected by serve gate" };

export function LabelingServingPanel({ predictions, modelLabel, ignoredIntervals, onSeek }: {
  predictions: ServingPrediction[];
  modelLabel: string;
  ignoredIntervals: IgnoredInterval[];
  onSeek: (time: number) => void;
}) {
  const [showIgnored, setShowIgnored] = useState(false);
  const isIgnored = (row: ServingPrediction) => ignoredIntervals.some(r => row.anchor >= r.start && row.anchor < r.end);
  const rows = predictions.filter(row => showIgnored || !isIgnored(row));
  const count = (verdict: ServingPrediction["verdict"]) => rows.filter(row => row.verdict === verdict).length;
  const ignoredCount = predictions.filter(isIgnored).length;
  return <section className={styles.panel} aria-label="Production serve predictions">
    <h3>Production serve predictions</h3>
    <p>{modelLabel} · {count("near")} near · {count("far")} far · {count("review")} need review · {count("not-serve")} rejected by serve gate</p>
    <p>These predictions use production rally starts as anchors, not confirmed serve-contact times. “Needs review” preserves the predicted side as a suggestion. Rejected candidates do not become serve markers or remove rallies. Scores are model outputs, not calibrated confidence.</p>
    {ignoredCount > 0 && <label className={styles.toggle}><input type="checkbox" checked={showIgnored} onChange={event => setShowIgnored(event.target.checked)} /> Include {ignoredCount} candidates in ignored footage</label>}
    <div className={styles.scroll}>
      <table><thead><tr><th>Candidate / anchor</th><th>Decision</th><th>Side scores</th><th>Why / serve evidence</th></tr></thead>
        <tbody>{rows.map(row => <tr key={row.id} data-verdict={row.verdict}>
          <td><button type="button" onClick={() => onSeek(Math.max(0, row.anchor - 2))}>{row.id} · {formatPreciseTime(row.anchor)}</button>{isIgnored(row) && <small>Ignored footage</small>}</td>
          <td><strong>{verdictLabel[row.verdict]}</strong>{row.verdict === "review" && <small>Side suggestion: {row.side}</small>}</td>
          <td>Near {row.nearProbability.toFixed(3)}<br />Far {(1 - row.nearProbability).toFixed(3)}</td>
          <td>{servingDecisionExplanation(row)}{row.evidence.map(head => <small key={head.label}>{head.label}: peak {head.peakProbability.toFixed(3)} at {formatPreciseTime(head.peakTime)}; threshold {head.threshold.toFixed(3)} ({head.crossesThreshold ? "passed" : "below threshold"}).</small>)}</td>
        </tr>)}</tbody>
      </table>
    </div>
  </section>;
}
