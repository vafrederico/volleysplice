"use client";

import { useId, useMemo, useState } from "react";
import { formatPreciseTime } from "@/lib/annotations";
import { nearestSignalIndex, type LabelingResearch, type ResearchBoundaryFlag, type ResearchExportPolicy } from "@/lib/labeling-research";
import styles from "./labeling-research-panel.module.css";

const heads = [
  { key: "live", label: "Live play", color: "#186d50" },
  { key: "serve", label: "Serve start", color: "#275ca7" },
  { key: "end", label: "Rally end", color: "#a44918" },
  { key: "keep", label: "Keep", color: "#754997" },
] as const;

function boundaryLabel(kind: ResearchBoundaryFlag["kind"]): string {
  return kind === "initial_start" ? "Initial start" : kind === "additional_start" ? "Additional rally" : "Rally end";
}

function reasonLabel(reason: string): string {
  return reason.replaceAll("_", " ").replaceAll("-", " ");
}

function boundaryTitle(flag: ResearchBoundaryFlag): string {
  if (flag.kind === "initial_start") return "Initial start correction";
  if (flag.kind === "additional_start") return "Additional rally start";
  return flag.details?.comparison === "new_boundary" ? "Separate rally end" : "Final end correction";
}

function signedSeconds(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)} s`;
}

function BoundaryExplanation({ flag, onSeek }: { flag: ResearchBoundaryFlag; onSeek: (time: number) => void }) {
  const details = flag.details;
  const source = details?.source === "head-serve" ? "Serve-start head refinement"
    : details?.source === "head-end" ? "Rally-end head refinement"
      : details?.source === "compact-start" ? "Compact event start; no head refinement"
        : details?.source === "compact-end" ? "Compact event end; no head refinement" : details?.source;
  return <li className={styles.flagDetail}>
    <div className={styles.flagTitle}><strong>{boundaryTitle(flag)}</strong><button type="button" onClick={() => onSeek(flag.time)}>Proposed {formatPreciseTime(flag.time)}</button></div>
    {details ? <>
      <p className={styles.boundaryChange}>{details.previousTime !== null && details.shiftSeconds !== null
        ? <>Original {details.comparison === "original_parent_start" ? "start" : "end"} <button type="button" onClick={() => onSeek(details.previousTime ?? flag.time)}>{formatPreciseTime(details.previousTime)}</button> → proposed {formatPreciseTime(flag.time)} <b>({signedSeconds(details.shiftSeconds)})</b></>
        : <>New boundary inside the original production region. No existing {flag.kind === "end" ? "separate rally end" : "additional rally start"} to move.</>}</p>
      <p className={styles.candidateRange}>Proposed rally: {formatPreciseTime(details.candidateStart)}–{formatPreciseTime(details.candidateEnd)}
        {details.previousProposedEnd !== undefined && <> · Previous proposed end: {formatPreciseTime(details.previousProposedEnd)}</>}
        {details.nextProposedStart !== undefined && <> · Next proposed start: {formatPreciseTime(details.nextProposedStart)}</>}</p>
      <p className={styles.whatToCheck}><b>Check:</b> {details.reviewInstruction}</p>
      <div className={styles.evidence}>
        <span>{source}. {details.observed ? "Model evidence; not human verified." : "Inherited or clipped model boundary; not human verified."}</span>
        {details.source.startsWith("head-") && <span>Compact event boundary before refinement: {formatPreciseTime(details.originalNeuralTime)} ({signedSeconds(details.headShiftSeconds)} refinement).</span>}
        <span>Native sample at {formatPreciseTime(details.sample.time)} · {signedSeconds(details.sample.time - flag.time)} from proposal</span>
        <div className={styles.sampleScores}>{heads.map(head => <span key={head.key} data-relevant={head.key === (flag.kind === "end" ? "end" : "serve")}>{head.label} <b>{details.sample[head.key].toFixed(3)}</b></span>)}</div>
      </div>
    </> : <p className={styles.whatToCheck}>Inspect the full parent region and confirm this proposed boundary against the video.</p>}
  </li>;
}

export function LabelingResearchPanel({ modelLabel, research, duration, currentTime, onSeek, exportPolicy = "fixed-production" }: {
  modelLabel: string;
  research: LabelingResearch;
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
  exportPolicy?: ResearchExportPolicy;
}) {
  const standalone = exportPolicy === "model-predictions";
  const headingId = useId();
  const [windowSeconds, setWindowSeconds] = useState(30);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);
  const windowSize = windowSeconds || duration;
  const start = Math.max(0, Math.min(currentTime - windowSize / 2, duration - windowSize));
  const end = Math.min(duration, start + windowSize);
  const span = Math.max(.001, end - start);
  const x = (time: number) => 1000 * (time - start) / span;
  const sampleIndex = nearestSignalIndex(research.signals.times, hoverTime ?? currentTime);
  const sampleTime = research.signals.times[sampleIndex];
  const paths = useMemo(() => {
    const { times } = research.signals;
    const first = Math.max(0, nearestSignalIndex(times, start) - 1);
    const last = Math.min(times.length - 1, nearestSignalIndex(times, end) + 1);
    return heads.map((head, lane) => {
      const points: string[] = [];
      for (let i = first; i <= last; i++) {
        points.push(`${i === first ? "M" : "L"}${(1000 * (times[i] - start) / span).toFixed(2)},${(lane * 48 + 42 - research.signals[head.key][i] * 36).toFixed(2)}`);
      }
      return points.join(" ");
    });
  }, [research.signals, start, end, span]);
  const regions = research.reviewRegions.filter(region => showAll || region.recommended)
    .sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
  const visibleRegions = research.reviewRegions.filter(region => region.end >= start && region.start <= end);
  const visibleFlags = research.boundaryFlags.filter(flag => flag.time >= start && flag.time <= end);
  const activeRegions = research.reviewRegions.filter(region => region.start <= currentTime && currentTime <= region.end);
  const flagCounts = {
    initial: research.boundaryFlags.filter(flag => flag.kind === "initial_start").length,
    additional: research.boundaryFlags.filter(flag => flag.kind === "additional_start").length,
    end: research.boundaryFlags.filter(flag => flag.kind === "end").length,
  };

  function pointerTime(event: { currentTarget: HTMLDivElement; clientX: number }): number {
    const bounds = event.currentTarget.getBoundingClientRect();
    return Math.max(start, Math.min(end, start + (event.clientX - bounds.left) / bounds.width * span));
  }

  return (
    <section className={styles.panel} aria-labelledby={headingId}>
      <div className={styles.heading}>
        <div><h3 id={headingId}>Model signals{!standalone && " & review"}</h3><p>{modelLabel}</p></div>
        <label>Signal window <select value={windowSeconds} onChange={event => { setWindowSeconds(Number(event.target.value)); setHoverTime(null); }}>
          <option value={30}>30 seconds</option><option value={90}>90 seconds</option><option value={0}>Full video</option>
        </select></label>
      </div>
      <p className={styles.explanation}>{standalone
        ? "Standalone model predictions. Main blocks keep each decoded start and end; the export strip applies padding and gap joining to these model rallies. These predictions do not edit human labels. Serve-start evidence does not predict serving side."
        : "Production export stays unchanged. Review flagged regions in full to correct rally starts, ends and separation. Proposed boundaries are provisional; they do not edit human labels. Serve-start evidence does not predict serving side."}</p>
      <p className={styles.recommendation}>{research.recommendation}</p>
      <div className={styles.signalChart}>
        <div className={styles.signalLabels}>
          {heads.map(head => <div key={head.key}><span style={{ color: head.color }}>{head.label}</span><strong>{sampleIndex < 0 ? "—" : research.signals[head.key][sampleIndex].toFixed(3)}</strong></div>)}
        </div>
        <div className={styles.plot} role="slider" tabIndex={0} aria-label="Seek video using model signals"
          aria-valuemin={0} aria-valuemax={duration} aria-valuenow={currentTime} aria-valuetext={formatPreciseTime(currentTime)}
          onPointerMove={event => setHoverTime(pointerTime(event))} onPointerLeave={() => setHoverTime(null)}
          onClick={event => onSeek(pointerTime(event))}
          onKeyDown={event => {
            if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
              event.preventDefault(); event.stopPropagation();
              onSeek(Math.max(0, Math.min(duration, currentTime + (event.key === "ArrowRight" ? .25 : -.25))));
            }
          }}>
          <svg viewBox="0 0 1000 192" preserveAspectRatio="none" aria-hidden="true">
            {visibleRegions.map(region => <rect key={region.id} x={Math.max(0, x(region.start))} y={0} width={Math.max(0, Math.min(1000, x(region.end)) - Math.max(0, x(region.start)))} height={192} fill={region.recommended ? "#dfb13d" : "#a8aba7"} opacity={region.recommended ? .16 : .09} />)}
            {heads.map((head, lane) => <g key={head.key}>
              <line x1={0} x2={1000} y1={lane * 48 + 42} y2={lane * 48 + 42} stroke="#d6d7ce" />
              <line x1={0} x2={1000} y1={lane * 48 + 24} y2={lane * 48 + 24} stroke="#e2e3da" strokeDasharray="3 5" />
              <path d={paths[lane]} fill="none" stroke={head.color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
            </g>)}
            {visibleFlags.map(flag => <line key={flag.id} x1={x(flag.time)} x2={x(flag.time)} y1={0} y2={192} stroke={flag.kind === "end" ? "#a44918" : "#275ca7"} strokeDasharray="2 4" opacity={.5} />)}
            <line x1={x(currentTime)} x2={x(currentTime)} y1={0} y2={192} stroke="#171914" strokeWidth={2} vectorEffect="non-scaling-stroke" />
            {hoverTime !== null && <line x1={x(hoverTime)} x2={x(hoverTime)} y1={0} y2={192} stroke="#72756c" vectorEffect="non-scaling-stroke" />}
          </svg>
        </div>
      </div>
      <div className={styles.axis}><span>{formatPreciseTime(start)}</span><span>{formatPreciseTime(end)}</span></div>
      <p className={styles.chartHint}>Scores 0–1 at {sampleTime === undefined ? "no sample" : formatPreciseTime(sampleTime)} ({hoverTime === null ? "nearest playhead sample" : "hover sample"}). Scores are model signals, not calibrated confidence. Actual video timestamps. Hover to inspect; click to seek.{!standalone && " Gold shading: recommended review. Dashed lines: proposed boundaries."}</p>
      {!standalone && <>
      <div className={styles.queueHeading}>
        <h4>Review queue <span>{research.queue.selectedParentCount} recommended regions · {formatPreciseTime(research.queue.reviewSeconds)} playback · {Math.round(research.queue.budgetFraction * 100)}% budget · Regions shown in video order</span></h4>
        <label><input type="checkbox" checked={showAll} onChange={event => setShowAll(event.target.checked)} /> Show all {research.reviewRegions.length} flagged regions</label>
      </div>
      <div className={styles.flagSummary} role="group" aria-label="All boundary review counts">
        <strong>{research.boundaryFlags.length} boundary flags across {research.reviewRegions.length} regions</strong>
        <span>{flagCounts.initial} initial start corrections</span><span>{flagCounts.additional} additional rally starts</span><span>{flagCounts.end} end boundaries</span>
      </div>
      <p className={styles.chartHint}>The {Math.round(research.queue.budgetFraction * 100)}% budget chooses a limited review workload. Regions outside it can still need correction. Expand any region below for every proposed change and its evidence.</p>
      <div className={styles.queue}>
        {regions.length ? regions.map(region => {
          const flags = research.boundaryFlags.filter(flag => flag.parentId === region.parentId).sort((a, b) => a.time - b.time || a.id.localeCompare(b.id));
          return <article key={region.id} className={styles.regionCard} data-active={region.start <= currentTime && currentTime <= region.end}>
            <div className={styles.regionHeading}><button type="button" onClick={() => onSeek(Math.max(0, region.start - 2))}>Review {region.parentId} · {formatPreciseTime(region.start)}–{formatPreciseTime(region.end)}</button>
              <span className={styles.budgetBadge} data-recommended={region.recommended}>{region.recommended ? `Recommended · within ${Math.round(research.queue.budgetFraction * 100)}% budget` : `Outside ${Math.round(research.queue.budgetFraction * 100)}% budget`}</span></div>
            <p className={styles.parentDescription}>Original production parent {region.parentId}. Whole region review; playback begins 2 seconds before it.</p>
            <details className={styles.regionDetails}>
              <summary>{flags.length} boundary {flags.length === 1 ? "suggestion" : "suggestions"} · {flags.filter(flag => flag.kind === "initial_start").length} initial / {flags.filter(flag => flag.kind === "additional_start").length} additional / {flags.filter(flag => flag.kind === "end").length} end</summary>
              <ul>{flags.map(flag => <BoundaryExplanation key={flag.id} flag={flag} onSeek={onSeek} />)}</ul>
              {!flags.length && <p>{region.reasons.map(reasonLabel).join(" · ") || "Check rally boundaries and cleanup"}</p>}
            </details>
          </article>;
        }) : <p>No regions in this review selection.</p>}
      </div>
      <div className={styles.boundaries}><strong>Proposals in the current region</strong>
        {activeRegions.length ? research.boundaryFlags.filter(flag => activeRegions.some(region => region.parentId === flag.parentId)).map(flag => <button key={flag.id} type="button" onClick={() => onSeek(flag.time)}>{boundaryLabel(flag.kind)} · {formatPreciseTime(flag.time)}</button>) : <span>Seek to a flagged region to inspect its start and end proposals.</span>}
      </div>
      </>}
    </section>
  );
}
