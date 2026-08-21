"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import { formatTime } from "@/lib/edit-list";

import styles from "../side-switch-review/side-switch-review.module.css";
import {
  SERVING_VARIANTS,
  type ServingDecision,
  type ServingMetric,
  type ServingRally,
  type ServingRecording,
  type ServingReport,
  type ServingVariant,
  type ServingVariantEvidence,
} from "./types";

type Props = {
  report: ServingReport | null;
  recordings: ServingRecording[];
  reportPath: string;
  initialDecisions: Record<string, ServingDecision>;
  initialSavedAt: string | null;
  decisionLoadError?: string;
  loadError?: string;
};

type RallyFilter = "all" | "known" | "near" | "far" | "unknown" | "frame-error";
type SaveStatus = "idle" | "saving" | "saved" | "error";
type RallyKind = "near" | "far" | "unknown";

const VARIANT_DETAILS: Record<
  ServingVariant,
  { label: string; detail: string }
> = {
  pixelMotion: { label: "Pixel motion", detail: "Whole-half grayscale change" },
  paletteChange: { label: "Palette change", detail: "HSV palette distance" },
  hogAreaChange: { label: "HOG area change", detail: "Proposal area change" },
  motionPalette: {
    label: "Motion + palette",
    detail: "Weighted whole-half bundle",
  },
  motionPaletteHog: {
    label: "Motion + palette + HOG",
    detail: "Fixed three-channel bundle",
  },
  baselineMotion: {
    label: "Baseline motion",
    detail: "Outer baseline-band motion",
  },
  baselinePalette: {
    label: "Baseline palette",
    detail: "Outer baseline-band palette",
  },
  baselineHogArea: {
    label: "Baseline HOG area",
    detail: "Outer baseline proposal area",
  },
  baselineMotionPalette: {
    label: "Baseline motion + palette",
    detail: "Weighted baseline bundle",
  },
};

const RALLY_FILTERS: Array<{ value: RallyFilter; label: string }> = [
  { value: "all", label: "All rallies" },
  { value: "known", label: "Weak targets" },
  { value: "near", label: "Near targets" },
  { value: "far", label: "Far targets" },
  { value: "unknown", label: "Unknown / candidate" },
  { value: "frame-error", label: "Frame errors" },
];

const OVERVIEW_TICK_RATIOS = [0, 0.2, 0.4, 0.6, 0.8, 1];
const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 2, 4];

function compactNumber(value: number): string {
  return value.toLocaleString("en-US");
}

function decimal(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

function percentage(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : "—";
}

function rallyKind(rally: ServingRally): RallyKind {
  if (rally.weakTarget.side === "near") return "near";
  if (rally.weakTarget.side === "far") return "far";
  return "unknown";
}

function rallyKindLabel(rally: ServingRally): string {
  if (rallyKind(rally) === "near") return "Weak near target";
  if (rallyKind(rally) === "far") return "Weak far target";
  return rally.targetStatus === "candidate-only"
    ? "Candidate-only rally"
    : "Unknown target";
}

function filterMatches(rally: ServingRally, filter: RallyFilter): boolean {
  if (filter === "known") return rallyKind(rally) !== "unknown";
  if (filter === "near" || filter === "far") return rallyKind(rally) === filter;
  if (filter === "unknown") return rallyKind(rally) === "unknown";
  if (filter === "frame-error") return rally.status === "frame-error";
  return true;
}

function evidenceValue(
  rally: ServingRally,
  variant: ServingVariant,
): number | null {
  const value = rally.variants[variant]?.score;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function metricFor(
  report: ServingReport,
  variant: ServingVariant,
  scope = "pooled",
): ServingMetric | null {
  return report.summary.metrics[scope]?.variants[variant] ?? null;
}

function boundedTime(value: number, duration: number): number {
  return Math.max(0, Math.min(duration, Number.isFinite(value) ? value : 0));
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    target.tagName === "INPUT" ||
    target.tagName === "SELECT" ||
    target.tagName === "TEXTAREA"
  );
}

function percentageAt(value: number, start: number, end: number): number {
  if (!Number.isFinite(value) || end <= start) return 0;
  return Math.max(0, Math.min(100, ((value - start) / (end - start)) * 100));
}

function recordingDuration(
  recording: ServingRecording | undefined,
  rallies: ServingRally[],
): number {
  if (recording && recording.durationSeconds > 0)
    return recording.durationSeconds;
  return Math.max(1, ...rallies.map((rally) => rally.end + 5));
}

function savedTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString();
}

function videoUrl(recordingId: string): string {
  return `/api/review-media/serving-side/${encodeURIComponent(recordingId)}`;
}

function sourceSummary(rally: ServingRally): string {
  const candidateSource = rally.candidateSource;
  if (candidateSource && typeof candidateSource.modelId === "string") {
    return `candidate model ${candidateSource.modelId}`;
  }
  return rally.sourceType ?? "label-derived row";
}

function OverviewTimeline({
  rallies,
  duration,
  selectedRallyId,
  currentTime,
  onSelect,
  onSeek,
}: {
  rallies: ServingRally[];
  duration: number;
  selectedRallyId: string;
  currentTime: number;
  onSelect: (rallyId: string) => void;
  onSeek: (time: number) => void;
}) {
  return (
    <div className={styles.timelineBlock}>
      <div className={styles.timelineHeading}>
        <div>
          <strong>Recording overview</strong>
          <span>Every serve/rally candidate in this recording</span>
        </div>
        <span>{formatTime(duration)}</span>
      </div>
      <div className={styles.overviewAxis}>
        {OVERVIEW_TICK_RATIOS.map((ratio) => (
          <span key={`overview-tick-${ratio}`}>
            {formatTime(duration * ratio)}
          </span>
        ))}
      </div>
      <div
        className={styles.overviewRail}
        onClick={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect();
          onSeek(
            boundedTime(
              ((event.clientX - bounds.left) / bounds.width) * duration,
              duration,
            ),
          );
        }}
        role="presentation"
      >
        {rallies.map((rally) => (
          <button
            type="button"
            className={styles.overviewEvent}
            data-kind={rallyKind(rally)}
            data-status={rally.status}
            data-selected={rally.rallyId === selectedRallyId ? "true" : "false"}
            style={{ left: `${(rally.start / duration) * 100}%` }}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(rally.rallyId);
            }}
            title={`${rallyKindLabel(rally)} · ${formatTime(rally.start)}`}
            aria-label={`Select ${rally.rallyId}`}
            key={rally.rallyId}
          />
        ))}
        <span
          className={styles.overviewPlayhead}
          style={{ left: `${(currentTime / duration) * 100}%` }}
        />
      </div>
      <div className={styles.timelineLegend}>
        <span data-tone="switch">near target</span>
        <span data-tone="control">far target</span>
        <span data-tone="selected">selected rally</span>
      </div>
    </div>
  );
}

function RallyWindowTimeline({
  rally,
  duration,
  currentTime,
  onSeek,
}: {
  rally: ServingRally;
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}) {
  const start = Math.max(0, rally.start - 3);
  const end = Math.min(duration, rally.start + 4);
  const sampleTimes = [
    ...rally.preTimes.map((time) => ({ time, side: "before" })),
    ...rally.actionTimes.map((time) => ({ time, side: "after" })),
  ];
  return (
    <div className={styles.windowBlock}>
      <div className={styles.timelineHeading}>
        <div>
          <strong>Serve evidence window</strong>
          <span>
            {formatTime(rally.start)} anchor · pre-serve and immediate action
            frames
          </span>
        </div>
        <span>{sampleTimes.length} frames</span>
      </div>
      <div className={styles.windowAxis}>
        <span>{formatTime(start)}</span>
        <span>{formatTime(rally.start)}</span>
        <span>{formatTime(end)}</span>
      </div>
      <div
        className={styles.windowRail}
        onClick={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect();
          onSeek(
            boundedTime(
              start +
                ((event.clientX - bounds.left) / bounds.width) * (end - start),
              duration,
            ),
          );
        }}
        role="presentation"
      >
        <span
          className={styles.gapBand}
          style={{
            left: `${percentageAt(rally.start, start, end)}%`,
            width: "1.5%",
          }}
        />
        {sampleTimes.map(({ time, side }) => (
          <button
            type="button"
            className={styles.sampleMarker}
            data-side={side}
            style={{ left: `${percentageAt(time, start, end)}%` }}
            onClick={(event) => {
              event.stopPropagation();
              onSeek(time);
            }}
            title={`${side} sample at ${formatTime(time)}`}
            aria-label={`Seek to ${side} sample at ${formatTime(time)}`}
            key={`${side}-${time}`}
          />
        ))}
        <span
          className={styles.transitionMarker}
          style={{ left: `${percentageAt(rally.start, start, end)}%` }}
        />
        {currentTime >= start && currentTime <= end && (
          <span
            className={styles.windowPlayhead}
            style={{ left: `${percentageAt(currentTime, start, end)}%` }}
          />
        )}
      </div>
      <div className={styles.windowLegend}>
        <span data-tone="before">pre-serve samples</span>
        <span data-tone="after">action samples</span>
        <span data-tone="transition">serve anchor</span>
      </div>
    </div>
  );
}

function VariantCard({
  variant,
  rally,
  metric,
}: {
  variant: ServingVariant;
  rally: ServingRally;
  metric: ServingMetric | null;
}) {
  const evidence: ServingVariantEvidence | undefined = rally.variants[variant];
  return (
    <article className={styles.featureCard}>
      <div className={styles.featureCardHeading}>
        <strong>{VARIANT_DETAILS[variant].label}</strong>
        <span>{VARIANT_DETAILS[variant].detail}</span>
      </div>
      <div className={styles.featureScore}>
        <strong>{decimal(evidence?.score ?? null)}</strong>
        <span>{evidence?.predictedSide ?? "abstain"}</span>
      </div>
      <dl>
        <div>
          <dt>Directional accuracy</dt>
          <dd>{percentage(metric?.directionalAccuracy)}</dd>
        </div>
        <div>
          <dt>Decision accuracy</dt>
          <dd>{percentage(metric?.decisionAccuracy)}</dd>
        </div>
        <div>
          <dt>Decision coverage</dt>
          <dd>{percentage(metric?.decisionCoverage)}</dd>
        </div>
      </dl>
    </article>
  );
}

function unavailable(reportPath: string, loadError?: string) {
  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Brand className={styles.brand} label="Serving-side review" priority />
        <nav>
          <Link href="/">Rally model review ↗</Link>
          <Link href="/serving-side-results">Model results ↗</Link>
          <Link href="/serving-side-flight-review">Flight error review ↗</Link>
          <Link href="/side-switch-review">Side-switch review ↗</Link>
        </nav>
      </header>
      <section className={styles.unavailable}>
        <p className={styles.eyebrow}>SERVING-SIDE EVIDENCE · DEV REVIEW</p>
        <h1>
          Report not <em>available.</em>
        </h1>
        <p>
          Run the full-NAS serving-side diagnostic first, or set{" "}
          <code>VOLLEYCUT_SERVING_SIDE_REPORT</code>.
        </p>
        <code className={styles.path}>{reportPath}</code>
        {loadError && <small>{loadError}</small>}
      </section>
    </main>
  );
}

function LoadedServingSideReview({
  report,
  recordings,
  reportPath,
  initialDecisions,
  initialSavedAt,
  decisionLoadError,
}: Omit<Props, "report" | "loadError"> & {
  report: ServingReport;
  decisionLoadError?: string;
}) {
  const allRallies = report.rallies;
  const [environment, setEnvironment] = useState("all");
  const [recordingId, setRecordingId] = useState("all");
  const [rallyFilter, setRallyFilter] = useState<RallyFilter>("all");
  const [selectedRallyId, setSelectedRallyId] = useState(
    () =>
      allRallies.find((rally) => rally.weakTarget.side !== null)?.rallyId ??
      allRallies[0]?.rallyId ??
      "",
  );
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [decisions, setDecisions] =
    useState<Record<string, ServingDecision>>(initialDecisions);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>(
    decisionLoadError ? "error" : initialSavedAt ? "saved" : "idle",
  );
  const [saveError, setSaveError] = useState<string | null>(
    decisionLoadError ?? null,
  );
  const [lastSavedAt, setLastSavedAt] = useState(initialSavedAt);
  const videoRef = useRef<HTMLVideoElement>(null);
  const skipInitialDecisionSave = useRef(true);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const environments = useMemo(
    () =>
      [...new Set(recordings.map((recording) => recording.environment))].sort(),
    [recordings],
  );
  const filteredRallies = useMemo(
    () =>
      allRallies.filter(
        (rally) =>
          (environment === "all" || rally.environment === environment) &&
          (recordingId === "all" || rally.recordingId === recordingId) &&
          filterMatches(rally, rallyFilter),
      ),
    [allRallies, environment, rallyFilter, recordingId],
  );
  const selectedRally =
    filteredRallies.find((rally) => rally.rallyId === selectedRallyId) ??
    filteredRallies[0] ??
    null;
  const selectedRecording = recordings.find(
    (recording) => recording.recordingId === selectedRally?.recordingId,
  );
  const selectedRecordingRallies = useMemo(
    () =>
      selectedRally
        ? allRallies.filter(
            (rally) => rally.recordingId === selectedRally.recordingId,
          )
        : [],
    [allRallies, selectedRally],
  );
  const duration = recordingDuration(
    selectedRecording,
    selectedRecordingRallies,
  );
  const selectedIndex = selectedRally
    ? filteredRallies.findIndex(
        (rally) => rally.rallyId === selectedRally.rallyId,
      )
    : -1;
  const selectedDecision = selectedRally
    ? decisions[selectedRally.rallyId]
    : undefined;
  const reviewedCount = Object.keys(decisions).length;
  const pooledMetric = metricFor(report, "motionPaletteHog");
  const scopeLabel =
    selectedRally &&
    report.summary.metrics[`environment:${selectedRally.environment}`]
      ? `environment:${selectedRally.environment}`
      : "pooled";

  const nextUnreviewedRally = useMemo(() => {
    if (!filteredRallies.length) return null;
    const startIndex = selectedIndex >= 0 ? selectedIndex : -1;
    for (let offset = 1; offset <= filteredRallies.length; offset += 1) {
      const candidate =
        filteredRallies[(startIndex + offset) % filteredRallies.length];
      if (
        candidate &&
        candidate.rallyId !== selectedRally?.rallyId &&
        decisions[candidate.rallyId] === undefined
      ) {
        return candidate;
      }
    }
    return null;
  }, [decisions, filteredRallies, selectedIndex, selectedRally?.rallyId]);

  const moveToNextUnreviewed = useCallback(() => {
    if (nextUnreviewedRally) {
      setSelectedRallyId(nextUnreviewedRally.rallyId);
    }
  }, [nextUnreviewedRally]);

  const moveToPrevious = useCallback(() => {
    if (selectedIndex <= 0) return;
    const previousRally = filteredRallies[selectedIndex - 1];
    if (previousRally) setSelectedRallyId(previousRally.rallyId);
  }, [filteredRallies, selectedIndex]);

  const setDecision = useCallback(
    (decision: ServingDecision) => {
      if (!selectedRally) return;
      setDecisions((current) => ({
        ...current,
        [selectedRally.rallyId]: decision,
      }));
    },
    [selectedRally],
  );

  const saveDecisionsToNas = useCallback(
    async (values: Record<string, ServingDecision>) => {
      setSaveStatus("saving");
      setSaveError(null);
      try {
        const response = await fetch("/api/serving-side-review/decisions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            schemaVersion: 1,
            reportKind: report.kind,
            reportCreatedAt: report.createdAt,
            decisions: values,
          }),
        });
        const payload = (await response.json().catch(() => null)) as {
          error?: unknown;
          savedAt?: unknown;
        } | null;
        if (!response.ok)
          throw new Error(
            typeof payload?.error === "string"
              ? payload.error
              : `Save failed with HTTP ${response.status}`,
          );
        const savedAt =
          typeof payload?.savedAt === "string"
            ? payload.savedAt
            : new Date().toISOString();
        setLastSavedAt(savedAt);
        setSaveStatus("saved");
      } catch (error) {
        setSaveStatus("error");
        setSaveError(error instanceof Error ? error.message : String(error));
      }
    },
    [report.createdAt, report.kind],
  );

  useEffect(() => {
    if (skipInitialDecisionSave.current) {
      skipInitialDecisionSave.current = false;
      return;
    }
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    setSaveStatus("saving");
    setSaveError(null);
    saveTimerRef.current = setTimeout(
      () => void saveDecisionsToNas(decisions),
      350,
    );
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, [decisions, saveDecisionsToNas]);

  useEffect(
    () => () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    },
    [],
  );

  useEffect(() => {
    if (filteredRallies.some((rally) => rally.rallyId === selectedRallyId))
      return;
    setSelectedRallyId(filteredRallies[0]?.rallyId ?? "");
  }, [filteredRallies, selectedRallyId]);

  useEffect(() => {
    if (!selectedRally) return;
    const target = boundedTime(selectedRally.start, duration);
    setCurrentTime(target);
    if (videoRef.current && videoRef.current.readyState >= 1)
      videoRef.current.currentTime = target;
  }, [duration, selectedRally]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    function handleShortcut(event: KeyboardEvent) {
      if (
        event.defaultPrevented ||
        event.repeat ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        isEditableTarget(event.target)
      ) {
        return;
      }

      const key = event.key.toLowerCase();
      if (key === "j") {
        event.preventDefault();
        moveToNextUnreviewed();
      } else if (key === "p") {
        event.preventDefault();
        moveToPrevious();
      } else if (key === "f") {
        event.preventDefault();
        setDecision("far");
      } else if (key === "n") {
        event.preventDefault();
        setDecision("near");
      } else if (key === "i") {
        event.preventDefault();
        setDecision("unclear");
      }
    }

    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [moveToNextUnreviewed, moveToPrevious, setDecision]);

  function seek(time: number) {
    const target = boundedTime(time, duration);
    setCurrentTime(target);
    if (videoRef.current) videoRef.current.currentTime = target;
  }

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Brand className={styles.brand} label="Serving-side review" priority />
        <nav>
          <Link href="/">Rally model review ↗</Link>
          <Link href="/serving-side-results">Model results ↗</Link>
          <Link href="/serving-side-flight-review">Flight error review ↗</Link>
          <Link href="/side-switch-review">Side-switch review ↗</Link>
          <Link href="/suppression-review">Suppression review ↗</Link>
        </nav>
      </header>

      <header className={styles.hero}>
        <div>
          <p className={styles.eyebrow}>
            NAS-WIDE DIAGNOSTIC · {report.protocol.hogProposal}
          </p>
          <h1>
            Find the <em>serving side.</em>
          </h1>
          <p className={styles.intro}>
            Review the fixed camera-space near/far evidence around every serve
            candidate. The report includes non-training videos, but
            candidate-only rows are validation evidence—not gold targets.
          </p>
          <p className={styles.sourceLine}>
            {report.summary.recordings} recordings · {report.summary.rallies}{" "}
            rally windows · report {new Date(report.createdAt).toLocaleString()}
          </p>
        </div>
        <div className={styles.heroMetric}>
          <span>Motion + palette + HOG</span>
          <strong>{percentage(pooledMetric?.directionalAccuracy)}</strong>
          <small>weak-target directional accuracy</small>
          <b>
            {percentage(pooledMetric?.decisionCoverage)} decision coverage ·
            diagnostic only
          </b>
        </div>
      </header>

      <section
        className={styles.summaryStrip}
        aria-label="Serving-side summary"
      >
        <div>
          <span>Recordings</span>
          <strong>{compactNumber(report.summary.recordings)}</strong>
        </div>
        <div>
          <span>Rally windows</span>
          <strong>{compactNumber(report.summary.rallies)}</strong>
        </div>
        <div>
          <span>Weak targets</span>
          <strong data-tone="switch">
            {compactNumber(report.summary.targetableRallies)}
          </strong>
        </div>
        <div>
          <span>Candidate-only</span>
          <strong data-tone="warning">
            {compactNumber(
              report.summary.targetStatusCounts?.["candidate-only"] ?? 0,
            )}
          </strong>
        </div>
        <div>
          <span>Reviewed decisions</span>
          <strong>{compactNumber(reviewedCount)}</strong>
        </div>
        <div>
          <span>Frame errors</span>
          <strong data-tone="warning">
            {compactNumber(report.summary.statuses["frame-error"] ?? 0)}
          </strong>
        </div>
      </section>

      <section className={styles.protocolPanel}>
        <div>
          <span className={styles.panelKicker}>EXPERIMENT CONTRACT</span>
          <strong>
            {report.protocol.nearSideDefinition} ·{" "}
            {report.protocol.farSideDefinition}
          </strong>
          <p>
            Positive evidence is signed near-minus-far. Existing note phrases
            are weak targets; raw and blind-source rows remain candidate-only.
            The score tracker must abstain when this review is unclear.
          </p>
        </div>
        <dl>
          <div>
            <dt>Corpus</dt>
            <dd>{report.protocol.corpusScope ?? "NAS manifest"}</dd>
          </div>
          <div>
            <dt>Sampling</dt>
            <dd>
              {report.protocol.preOffsetsSeconds.join(" · ")} /{" "}
              {report.protocol.actionOffsetsSeconds.join(" · ")}
            </dd>
          </div>
          <div>
            <dt>Targets</dt>
            <dd>
              {(report.protocol.evaluationTargetStatuses ?? ["gold"]).join(
                " · ",
              )}
            </dd>
          </div>
        </dl>
      </section>

      <section className={styles.filterPanel} aria-label="Review filters">
        <div className={styles.filterHeading}>
          <span className={styles.panelKicker}>01 / QUEUE</span>
          <strong>Choose what to inspect</strong>
          <small>{filteredRallies.length} rallies in view</small>
        </div>
        <label>
          <span>Environment</span>
          <select
            value={environment}
            onChange={(event) => {
              setEnvironment(event.target.value);
              setRecordingId("all");
            }}
          >
            <option value="all">All environments</option>
            {environments.map((item) => (
              <option value={item} key={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Recording</span>
          <select
            value={recordingId}
            onChange={(event) => setRecordingId(event.target.value)}
          >
            <option value="all">All recordings</option>
            {recordings
              .filter(
                (recording) =>
                  environment === "all" ||
                  recording.environment === environment,
              )
              .map((recording) => (
                <option
                  value={recording.recordingId}
                  key={recording.recordingId}
                >
                  {recording.environment} · {recording.recordingId}
                </option>
              ))}
          </select>
        </label>
        <div className={styles.filterChoices}>
          <span>Target type</span>
          <div>
            {RALLY_FILTERS.map((item) => (
              <button
                type="button"
                data-active={rallyFilter === item.value ? "true" : "false"}
                onClick={() => setRallyFilter(item.value)}
                key={item.value}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.workbench}>
        <aside className={styles.queue}>
          <header>
            <div>
              <span className={styles.panelKicker}>RALLY QUEUE</span>
              <strong>{filteredRallies.length} candidates</strong>
            </div>
            <small>{scopeLabel} metrics</small>
          </header>
          <div className={styles.queueList}>
            {filteredRallies.map((rally, index) => (
              <button
                type="button"
                className={styles.queueItem}
                data-active={
                  rally.rallyId === selectedRally?.rallyId ? "true" : "false"
                }
                data-kind={rallyKind(rally)}
                onClick={() => setSelectedRallyId(rally.rallyId)}
                key={rally.rallyId}
              >
                <span className={styles.queueIndex}>
                  {String(index + 1).padStart(3, "0")}
                </span>
                <span className={styles.queueMain}>
                  <strong>{rally.recordingId}</strong>
                  <small>
                    {formatTime(rally.start)} · {rallyKindLabel(rally)} ·{" "}
                    {rally.status}
                  </small>
                </span>
                <span className={styles.queueScore}>
                  {decimal(evidenceValue(rally, "motionPaletteHog"))}
                  {decisions[rally.rallyId] && (
                    <i data-decision={decisions[rally.rallyId]} />
                  )}
                </span>
              </button>
            ))}
            {!filteredRallies.length && (
              <p className={styles.emptyQueue}>
                No rallies match these filters.
              </p>
            )}
          </div>
        </aside>

        <section
          className={styles.inspector}
          aria-label="Selected serving-side rally inspector"
        >
          {selectedRally ? (
            <>
              <header className={styles.inspectorHeader}>
                <div>
                  <p className={styles.eyebrow}>
                    {selectedRally.environment} ·{" "}
                    {rallyKindLabel(selectedRally)} ·{" "}
                    {selectedRally.targetStatus ?? "gold"} ·{" "}
                    {selectedRally.status}
                  </p>
                  <h2>{selectedRally.rallyId}</h2>
                </div>
                <div className={styles.navigationButtons}>
                  <button
                    type="button"
                    disabled={selectedIndex <= 0}
                    onClick={moveToPrevious}
                  >
                    ← Previous <kbd>P</kbd>
                  </button>
                  <span>
                    {selectedIndex + 1} / {filteredRallies.length}
                  </span>
                  <button
                    type="button"
                    disabled={!nextUnreviewedRally}
                    onClick={moveToNextUnreviewed}
                    title="Next unreviewed rally (J)"
                  >
                    Next unreviewed <kbd>J</kbd> →
                  </button>
                </div>
              </header>

              <div
                className={styles.eventSummary}
                data-kind={rallyKind(selectedRally)}
              >
                <div>
                  <span>Serve anchor</span>
                  <strong>{formatTime(selectedRally.start)}</strong>
                </div>
                <div>
                  <span>Weak note target</span>
                  <strong>{selectedRally.weakTarget.side ?? "unknown"}</strong>
                </div>
                <div>
                  <span>Evidence status</span>
                  <strong>{selectedRally.status}</strong>
                </div>
                <div>
                  <span>Bundle score</span>
                  <strong>
                    {decimal(evidenceValue(selectedRally, "motionPaletteHog"))}
                  </strong>
                </div>
              </div>

              <div className={styles.videoStage}>
                <video
                  key={selectedRally.recordingId}
                  ref={videoRef}
                  controls
                  playsInline
                  preload="metadata"
                  src={videoUrl(selectedRally.recordingId)}
                  onLoadedMetadata={(event) => {
                    event.currentTarget.currentTime = boundedTime(
                      selectedRally.start,
                      duration,
                    );
                    event.currentTarget.playbackRate = playbackRate;
                    setCurrentTime(boundedTime(selectedRally.start, duration));
                  }}
                  onTimeUpdate={(event) =>
                    setCurrentTime(event.currentTarget.currentTime)
                  }
                  aria-label={`Video review for ${selectedRally.recordingId}`}
                >
                  Your browser does not support video playback.
                </video>
                <span className={styles.videoTime}>
                  {formatTime(currentTime)}
                </span>
                <span className={styles.videoLabel}>
                  {selectedRecording?.videoFilename ??
                    selectedRally.recordingId}
                </span>
              </div>

              <div className={styles.reviewToolbar}>
                <label className={styles.speedControl}>
                  <span>Playback speed</span>
                  <select
                    value={playbackRate}
                    onChange={(event) =>
                      setPlaybackRate(Number(event.target.value))
                    }
                    aria-label="Video playback speed"
                  >
                    {PLAYBACK_RATES.map((rate) => (
                      <option value={rate} key={rate}>
                        {rate}×
                      </option>
                    ))}
                  </select>
                </label>
                <div
                  className={styles.shortcutLegend}
                  role="group"
                  aria-label="Keyboard shortcuts"
                >
                  <span>
                    <kbd>J</kbd> next unreviewed
                  </span>
                  <span>
                    <kbd>P</kbd> previous
                  </span>
                  <span>
                    <kbd>F</kbd> far
                  </span>
                  <span>
                    <kbd>N</kbd> near
                  </span>
                  <span>
                    <kbd>I</kbd> ignore
                  </span>
                </div>
              </div>

              <OverviewTimeline
                rallies={selectedRecordingRallies}
                duration={duration}
                selectedRallyId={selectedRally.rallyId}
                currentTime={currentTime}
                onSelect={setSelectedRallyId}
                onSeek={seek}
              />
              <RallyWindowTimeline
                rally={selectedRally}
                duration={duration}
                currentTime={currentTime}
                onSeek={seek}
              />

              <section className={styles.decisionPanel}>
                <div>
                  <span className={styles.panelKicker}>
                    NAS-BACKED REVIEW · NOT GOLD LABELING
                  </span>
                  <strong>Which side appears to be serving?</strong>
                  <small>
                    Near/far is camera-space only. Decisions are automatically
                    saved to the NAS and can be used to validate candidate-only
                    recordings.
                  </small>
                </div>
                <div className={styles.decisionButtons}>
                  {(["near", "far", "unclear"] as ServingDecision[]).map(
                    (decision) => (
                      <button
                        type="button"
                        data-active={
                          selectedDecision === decision ? "true" : "false"
                        }
                        data-decision={decision}
                        onClick={() => setDecision(decision)}
                        key={decision}
                      >
                        {decision === "near"
                          ? "Near side"
                          : decision === "far"
                            ? "Far side"
                            : "Ignore"}
                        <kbd>
                          {decision === "near"
                            ? "N"
                            : decision === "far"
                              ? "F"
                              : "I"}
                        </kbd>
                      </button>
                    ),
                  )}
                  <button
                    type="button"
                    className={styles.saveButton}
                    onClick={() => void saveDecisionsToNas(decisions)}
                    disabled={saveStatus === "saving"}
                  >
                    {saveStatus === "saving" ? "Saving…" : "Save to NAS"}
                  </button>
                  <span className={styles.saveStatus} data-status={saveStatus}>
                    {saveStatus === "saving" && "Saving to NAS…"}
                    {saveStatus === "saved" &&
                      `Saved ${savedTime(lastSavedAt)}`}
                    {saveStatus === "error" &&
                      `Save failed: ${saveError ?? "unknown error"}`}
                    {saveStatus === "idle" && "Not saved yet"}
                  </span>
                </div>
              </section>

              <div className={styles.featureGrid}>
                {SERVING_VARIANTS.map((variant) => (
                  <VariantCard
                    key={variant}
                    variant={variant}
                    rally={selectedRally}
                    metric={metricFor(report, variant, scopeLabel)}
                  />
                ))}
              </div>

              <div className={styles.aggregateGrid}>
                <div className={styles.aggregateCard}>
                  <span>Target provenance</span>
                  <strong>{rallyKindLabel(selectedRally)}</strong>
                  <small>
                    {selectedRally.targetStatus ?? "gold"} ·{" "}
                    {sourceSummary(selectedRally)}
                  </small>
                  <small>{selectedRally.weakTarget.reason}</small>
                </div>
                <div className={styles.aggregateCard}>
                  <span>Weak cue</span>
                  <strong>{selectedRally.weakTarget.side ?? "unknown"}</strong>
                  <small>
                    {selectedRally.weakTarget.strength} ·{" "}
                    {selectedRally.weakTarget.matches.join(" · ") ||
                      "no phrase"}
                  </small>
                  <small>
                    Do not treat a note cue as a final serving-side label.
                  </small>
                </div>
                <div className={styles.notesCard}>
                  <span>Interpretation guardrail</span>
                  <strong>
                    {selectedRally.status === "ok"
                      ? "Compare the video with the nine scalar variants. A signed score is evidence, not a score-map update."
                      : (selectedRally.error ??
                        "No valid frame evidence was available.")}
                  </strong>
                  <small>
                    Near is the lower/foreground ROI half; far is the
                    upper/background ROI half. Camera orientation and side
                    switching still require separate state.
                  </small>
                </div>
              </div>
            </>
          ) : (
            <div className={styles.emptyInspector}>
              <span>No rally selected</span>
              <p>Adjust the queue filters to bring a rally into view.</p>
            </div>
          )}
        </section>
      </section>

      <footer className={styles.footer}>
        <span>Source report</span>
        <code>{reportPath}</code>
        <span>
          All rows are reviewable; only non-candidate target statuses contribute
          to the diagnostic accuracy summaries.
        </span>
      </footer>
    </main>
  );
}

export function ServingSideReviewClient({
  report,
  recordings,
  reportPath,
  initialDecisions,
  initialSavedAt,
  decisionLoadError,
  loadError,
}: Props) {
  if (!report) return unavailable(reportPath, loadError);
  return (
    <LoadedServingSideReview
      report={report}
      recordings={recordings}
      reportPath={reportPath}
      initialDecisions={initialDecisions}
      initialSavedAt={initialSavedAt}
      decisionLoadError={decisionLoadError}
    />
  );
}
