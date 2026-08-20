"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import { formatTime } from "@/lib/edit-list";

import styles from "./side-switch-review.module.css";
import {
  APPEARANCE_FEATURES,
  type AppearanceEvent,
  type AppearanceFeature,
  type AppearanceMetric,
  type AppearanceReport,
  type ReviewDecision,
  type SideSwitchRecording,
} from "./types";

type SideSwitchReviewClientProps = {
  report: AppearanceReport | null;
  recordings: SideSwitchRecording[];
  reportPath: string;
  loadError?: string;
};

type EventFilter = "all" | "switch" | "control" | "insufficient";

const FEATURE_DEFINITIONS: Array<{
  key: AppearanceFeature;
  label: string;
  detail: string;
}> = [
  {
    key: "fullFrameControl",
    label: "Full-frame control",
    detail: "Color change without player proposals",
  },
  {
    key: "playerPaletteEqual",
    label: "Player palette · equal",
    detail: "Each detected torso contributes equally",
  },
  {
    key: "playerPaletteArea",
    label: "Player palette · area",
    detail: "Torso palettes weighted by proposal area",
  },
  {
    key: "playerPaletteAreaPlusGeometry",
    label: "Area palette + geometry",
    detail: "Area palette plus count, box-area, and height change",
  },
];

const EVENT_FILTERS: Array<{ value: EventFilter; label: string }> = [
  { value: "all", label: "All events" },
  { value: "switch", label: "Labeled switches" },
  { value: "control", label: "No-switch controls" },
  { value: "insufficient", label: "Insufficient windows" },
];

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

function eventKind(event: AppearanceEvent): "switch" | "control" {
  return event.label === 1 ? "switch" : "control";
}

function eventKindLabel(event: AppearanceEvent): string {
  return event.label === 1 ? "Labeled switch" : "No-switch control";
}

function statusLabel(status: AppearanceEvent["status"]): string {
  if (status === "ok") return "Analyzed";
  if (status === "insufficient-window") return "Insufficient window";
  if (status === "frame-error") return "Frame error";
  return status;
}

function featureValue(
  event: AppearanceEvent,
  key: AppearanceFeature,
): number | null {
  const value = event.features[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function boundedTime(value: number, duration: number): number {
  return Math.max(0, Math.min(duration, Number.isFinite(value) ? value : 0));
}

function percentageAt(value: number, start: number, end: number): number {
  if (!Number.isFinite(value) || end <= start) return 0;
  return Math.max(0, Math.min(100, ((value - start) / (end - start)) * 100));
}

function recordingDuration(
  recording: SideSwitchRecording | undefined,
  events: AppearanceEvent[],
): number {
  if (recording && recording.durationSeconds > 0) return recording.durationSeconds;
  return Math.max(1, ...events.map((event) => event.gapEnd + 5));
}

function eventFilterMatches(event: AppearanceEvent, filter: EventFilter): boolean {
  if (filter === "switch") return event.label === 1;
  if (filter === "control") return event.label === 0;
  if (filter === "insufficient") return event.status !== "ok";
  return true;
}

function featureMetric(
  report: AppearanceReport,
  key: AppearanceFeature,
  scope = "pooled",
): AppearanceMetric | null {
  return report.summary.metrics[scope]?.features[key] ?? null;
}

function eventVideoUrl(recordingId: string): string {
  return `/api/labeling/tasks/${encodeURIComponent(recordingId)}/video`;
}

function OverviewTimeline({
  events,
  duration,
  selectedEventId,
  currentTime,
  onSelect,
  onSeek,
}: {
  events: AppearanceEvent[];
  duration: number;
  selectedEventId: string;
  currentTime: number;
  onSelect: (eventId: string) => void;
  onSeek: (time: number) => void;
}) {
  return (
    <div className={styles.timelineBlock}>
      <div className={styles.timelineHeading}>
        <div>
          <strong>Recording overview</strong>
          <span>Every labeled transition and sampled control in this recording</span>
        </div>
        <span>{formatTime(duration)}</span>
      </div>
      <div className={styles.overviewAxis}>
        {Array.from({ length: 6 }, (_, index) => (
          <span key={`overview-tick-${index}`}>
            {formatTime((duration * index) / 5)}
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
        {events.map((item) => (
          <button
            type="button"
            className={styles.overviewEvent}
            data-kind={eventKind(item)}
            data-status={item.status}
            data-selected={item.eventId === selectedEventId ? "true" : "false"}
            style={{ left: `${(item.transitionTime / duration) * 100}%` }}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(item.eventId);
            }}
            title={`${eventKindLabel(item)} · ${formatTime(item.transitionTime)}`}
            aria-label={`Select ${item.eventId}`}
            key={item.eventId}
          />
        ))}
        <span
          className={styles.overviewPlayhead}
          style={{ left: `${(currentTime / duration) * 100}%` }}
        />
      </div>
      <div className={styles.timelineLegend}>
        <span data-tone="switch">labeled switch</span>
        <span data-tone="control">no-switch control</span>
        <span data-tone="selected">selected event</span>
      </div>
    </div>
  );
}

function WindowTimeline({
  event,
  duration,
  currentTime,
  onSeek,
}: {
  event: AppearanceEvent;
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}) {
  const start = Math.max(0, event.gapStart - 3);
  const end = Math.min(duration, event.gapEnd + 3);
  const sampleTimes = [
    ...event.beforeTimes.map((time) => ({ time, side: "before" })),
    ...event.afterTimes.map((time) => ({ time, side: "after" })),
  ];
  return (
    <div className={styles.windowBlock}>
      <div className={styles.timelineHeading}>
        <div>
          <strong>Sampled transition window</strong>
          <span>
            {formatTime(event.gapStart)}–{formatTime(event.gapEnd)} gap · {decimal(event.gapSeconds, 1)}s
          </span>
        </div>
        <span>{event.beforeTimes.length + event.afterTimes.length} frames</span>
      </div>
      <div className={styles.windowAxis}>
        <span>{formatTime(start)}</span>
        <span>{formatTime((start + end) / 2)}</span>
        <span>{formatTime(end)}</span>
      </div>
      <div
        className={styles.windowRail}
        onClick={(click) => {
          const bounds = click.currentTarget.getBoundingClientRect();
          onSeek(
            boundedTime(
              start + ((click.clientX - bounds.left) / bounds.width) * (end - start),
              duration,
            ),
          );
        }}
        role="presentation"
      >
        <span
          className={styles.gapBand}
          style={{
            left: `${percentageAt(event.gapStart, start, end)}%`,
            width: `${percentageAt(event.gapEnd, start, end) - percentageAt(event.gapStart, start, end)}%`,
          }}
        />
        {sampleTimes.map(({ time, side }) => (
          <button
            type="button"
            className={styles.sampleMarker}
            data-side={side}
            style={{ left: `${percentageAt(time, start, end)}%` }}
            onClick={(click) => {
              click.stopPropagation();
              onSeek(time);
            }}
            title={`${side} sample at ${formatTime(time)}`}
            aria-label={`Seek to ${side} sample at ${formatTime(time)}`}
            key={`${side}-${time}`}
          />
        ))}
        <span
          className={styles.transitionMarker}
          style={{ left: `${percentageAt(event.transitionTime, start, end)}%` }}
        />
        {currentTime >= start && currentTime <= end && (
          <span
            className={styles.windowPlayhead}
            style={{ left: `${percentageAt(currentTime, start, end)}%` }}
          />
        )}
      </div>
      <div className={styles.windowLegend}>
        <span data-tone="before">before samples</span>
        <span data-tone="after">after samples</span>
        <span data-tone="transition">labeled transition</span>
      </div>
    </div>
  );
}

function FeatureCard({
  definition,
  event,
  metric,
}: {
  definition: (typeof FEATURE_DEFINITIONS)[number];
  event: AppearanceEvent;
  metric: AppearanceMetric | null;
}) {
  return (
    <article className={styles.featureCard}>
      <div className={styles.featureCardHeading}>
        <strong>{definition.label}</strong>
        <span>{definition.detail}</span>
      </div>
      <div className={styles.featureScore}>
        <strong>{decimal(featureValue(event, definition.key))}</strong>
        <span>selected distance</span>
      </div>
      <dl>
        <div>
          <dt>Pooled ROC AUC</dt>
          <dd>{decimal(metric?.rocAuc, 4)}</dd>
        </div>
        <div>
          <dt>Average precision</dt>
          <dd>{decimal(metric?.averagePrecision, 4)}</dd>
        </div>
        <div>
          <dt>Usable coverage</dt>
          <dd>{percentage(metric?.coverage)}</dd>
        </div>
      </dl>
    </article>
  );
}

function AggregateCard({
  label,
  aggregate,
}: {
  label: string;
  aggregate: AppearanceEvent["before"];
}) {
  return (
    <div className={styles.aggregateCard}>
      <span>{label}</span>
      <strong>
        {aggregate ? decimal(aggregate.meanDetectionCount, 1) : "—"} people
      </strong>
      <small>
        box area {aggregate ? percentage(aggregate.meanTotalBoxAreaFraction, 2) : "—"}
        {" · "}
        median height {aggregate ? percentage(aggregate.meanMedianBoxHeightFraction, 1) : "—"}
      </small>
      <small>
        usable proposals {aggregate?.usableFrameCount ?? "—"} · HOG score {decimal(aggregate?.meanDetectionScore, 2)}
      </small>
    </div>
  );
}

function UnavailableReview({
  reportPath,
  loadError,
}: {
  reportPath: string;
  loadError?: string;
}) {
  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Brand className={styles.brand} label="Side-switch review" priority />
        <nav>
          <Link href="/">Rally model review ↗</Link>
        </nav>
      </header>
      <section className={styles.unavailable}>
        <p className={styles.eyebrow}>SIDE-SWITCH APPEARANCE · DEV REVIEW</p>
        <h1>Report not <em>available.</em></h1>
        <p>
          Run the all-video diagnostic first, or set
          <code>VOLLEYCUT_SIDE_SWITCH_REPORT</code> to an existing report path.
        </p>
        <code className={styles.path}>{reportPath}</code>
        {loadError && <small>{loadError}</small>}
      </section>
    </main>
  );
}

function LoadedSideSwitchReview({
  report,
  recordings,
  reportPath,
}: {
  report: AppearanceReport;
  recordings: SideSwitchRecording[];
  reportPath: string;
}) {
  const allEvents = report.events;
  const [environment, setEnvironment] = useState("all");
  const [recordingId, setRecordingId] = useState("all");
  const [eventFilter, setEventFilter] = useState<EventFilter>("all");
  const [selectedEventId, setSelectedEventId] = useState(
    () => allEvents.find((event) => event.label === 1)?.eventId ?? allEvents[0]?.eventId ?? "",
  );
  const [currentTime, setCurrentTime] = useState(0);
  const [decisions, setDecisions] = useState<Record<string, ReviewDecision>>({});
  const [storageReady, setStorageReady] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const environments = useMemo(
    () => [...new Set(recordings.map((recording) => recording.environment))].sort(),
    [recordings],
  );

  const filteredEvents = useMemo(
    () =>
      allEvents.filter(
        (event) =>
          (environment === "all" || event.environment === environment) &&
          (recordingId === "all" || event.recordingId === recordingId) &&
          eventFilterMatches(event, eventFilter),
      ),
    [allEvents, environment, eventFilter, recordingId],
  );

  const selectedEvent =
    filteredEvents.find((event) => event.eventId === selectedEventId) ??
    filteredEvents[0] ??
    null;
  const selectedRecording = recordings.find(
    (recording) => recording.recordingId === selectedEvent?.recordingId,
  );
  const selectedRecordingEvents = useMemo(
    () =>
      selectedEvent
        ? allEvents.filter((event) => event.recordingId === selectedEvent.recordingId)
        : [],
    [allEvents, selectedEvent],
  );
  const duration = recordingDuration(selectedRecording, selectedRecordingEvents);
  const selectedIndex = selectedEvent
    ? filteredEvents.findIndex((event) => event.eventId === selectedEvent.eventId)
    : -1;
  const selectedDecision = selectedEvent ? decisions[selectedEvent.eventId] : undefined;
  const reviewedCount = Object.keys(decisions).length;
  const pooledAreaMetric = featureMetric(report, "playerPaletteArea");

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem("volleycut-side-switch-review-v1");
      if (stored) {
        const parsed = JSON.parse(stored) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          setDecisions(parsed as Record<string, ReviewDecision>);
        }
      }
    } catch {
      // Local review state is optional and should never block video inspection.
    } finally {
      setStorageReady(true);
    }
  }, []);

  useEffect(() => {
    if (!storageReady) return;
    window.localStorage.setItem(
      "volleycut-side-switch-review-v1",
      JSON.stringify(decisions),
    );
  }, [decisions, storageReady]);

  useEffect(() => {
    if (filteredEvents.some((event) => event.eventId === selectedEventId)) return;
    setSelectedEventId(filteredEvents[0]?.eventId ?? "");
  }, [filteredEvents, selectedEventId]);

  useEffect(() => {
    if (!selectedEvent) return;
    const target = boundedTime(selectedEvent.transitionTime, duration);
    setCurrentTime(target);
    if (videoRef.current && videoRef.current.readyState >= 1) {
      videoRef.current.currentTime = target;
    }
  }, [duration, selectedEvent]);

  function seek(time: number) {
    const target = boundedTime(time, duration);
    setCurrentTime(target);
    if (videoRef.current) videoRef.current.currentTime = target;
  }

  function selectEvent(eventId: string) {
    setSelectedEventId(eventId);
  }

  function moveSelection(direction: -1 | 1) {
    if (!filteredEvents.length) return;
    const nextIndex = Math.max(
      0,
      Math.min(filteredEvents.length - 1, selectedIndex + direction),
    );
    setSelectedEventId(filteredEvents[nextIndex]?.eventId ?? "");
  }

  function setDecision(decision: ReviewDecision) {
    if (!selectedEvent) return;
    setDecisions((current) => ({ ...current, [selectedEvent.eventId]: decision }));
  }

  function exportDecisions() {
    const reviewed = allEvents
      .filter((event) => decisions[event.eventId])
      .map((event) => ({
        eventId: event.eventId,
        recordingId: event.recordingId,
        environment: event.environment,
        transitionTime: event.transitionTime,
        sourceLabel: event.label === 1 ? "switch" : "no-switch-control",
        decision: decisions[event.eventId],
      }));
    const payload = {
      schemaVersion: 1,
      reportKind: report.kind,
      reportCreatedAt: report.createdAt,
      sourceReport: reportPath,
      decisions: reviewed,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `volleycut-side-switch-review-${new Date().toISOString().replaceAll(":", "-")}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const currentScope = selectedEvent?.environment ?? "pooled";
  const scopeLabel = report.summary.metrics[currentScope]
    ? currentScope
    : "pooled";

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Brand className={styles.brand} label="Side-switch review" priority />
        <nav>
          <Link href="/">Rally model review ↗</Link>
          <Link href="/suppression-review">Suppression review ↗</Link>
        </nav>
      </header>

      <header className={styles.hero}>
        <div>
          <p className={styles.eyebrow}>
            LABEL-ONLY DIAGNOSTIC · {report.protocol.personProposal}
          </p>
          <h1>
            Find the <em>side flip.</em>
          </h1>
          <p className={styles.intro}>
            Inspect whether cheap before/after appearance changes can identify
            beach and grass side switches. This is a research review of color
            distance and proposal stability, not a production threshold.
          </p>
          <p className={styles.sourceLine}>
            {report.summary.recordings} recordings · {report.summary.events} sampled gaps ·
            {" "}{report.protocol.samplesPerSide} frames per side · report {new Date(report.createdAt).toLocaleString()}
          </p>
        </div>
        <div className={styles.heroMetric}>
          <span>Best pooled diagnostic</span>
          <strong>{decimal(pooledAreaMetric?.rocAuc, 3)}</strong>
          <small>ROC AUC · area-weighted player palette</small>
          <b>{percentage(pooledAreaMetric?.coverage)} usable event coverage</b>
        </div>
      </header>

      <section className={styles.summaryStrip} aria-label="Diagnostic summary">
        <div>
          <span>Recordings</span>
          <strong>{compactNumber(report.summary.recordings)}</strong>
        </div>
        <div>
          <span>Switch markers</span>
          <strong data-tone="switch">{compactNumber(report.summary.positives)}</strong>
        </div>
        <div>
          <span>No-switch controls</span>
          <strong>{compactNumber(report.summary.negatives)}</strong>
        </div>
        <div>
          <span>Insufficient windows</span>
          <strong data-tone="warning">
            {compactNumber(report.summary.statuses["insufficient-window"] ?? 0)}
          </strong>
        </div>
        <div>
          <span>Local decisions</span>
          <strong>{compactNumber(reviewedCount)}</strong>
        </div>
      </section>

      <section className={styles.protocolPanel}>
        <div>
          <span className={styles.panelKicker}>EXPERIMENT CONTRACT</span>
          <strong>{report.protocol.colorRepresentation}</strong>
          <p>
            Positive rows come from completed <code>sideSwitches</code> markers;
            negatives are unmarked inter-rally gaps of at least {report.protocol.minimumGapSeconds}s.
            HOG boxes are proposal evidence only.
          </p>
        </div>
        <dl>
          <div>
            <dt>Scope</dt>
            <dd>{report.protocol.environments.join(" · ")}</dd>
          </div>
          <div>
            <dt>Sampling</dt>
            <dd>±{report.protocol.flankSeconds}s · {report.protocol.edgeMarginSeconds}s edge margin</dd>
          </div>
          <div>
            <dt>Thresholds</dt>
            <dd>{report.protocol.thresholdsAreDiagnostic ? "Diagnostic only" : "Production candidate"}</dd>
          </div>
        </dl>
      </section>

      <section className={styles.filterPanel} aria-label="Review filters">
        <div className={styles.filterHeading}>
          <span className={styles.panelKicker}>01 / QUEUE</span>
          <strong>Choose what to inspect</strong>
          <small>{filteredEvents.length} events in view</small>
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
              <option value={item} key={item}>{item}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Recording</span>
          <select value={recordingId} onChange={(event) => setRecordingId(event.target.value)}>
            <option value="all">All recordings</option>
            {recordings
              .filter((recording) => environment === "all" || recording.environment === environment)
              .map((recording) => (
                <option value={recording.recordingId} key={recording.recordingId}>
                  {recording.environment} · {recording.recordingId}
                </option>
              ))}
          </select>
        </label>
        <div className={styles.filterChoices}>
          <span>Event type</span>
          <div>
            {EVENT_FILTERS.map((item) => (
              <button
                type="button"
                data-active={eventFilter === item.value ? "true" : "false"}
                onClick={() => setEventFilter(item.value)}
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
              <span className={styles.panelKicker}>EVENT QUEUE</span>
              <strong>{filteredEvents.length} candidates</strong>
            </div>
            <small>{scopeLabel} metrics</small>
          </header>
          <div className={styles.queueList}>
            {filteredEvents.map((event, index) => (
              <button
                type="button"
                className={styles.queueItem}
                data-active={event.eventId === selectedEvent?.eventId ? "true" : "false"}
                data-kind={eventKind(event)}
                onClick={() => selectEvent(event.eventId)}
                key={event.eventId}
              >
                <span className={styles.queueIndex}>{String(index + 1).padStart(3, "0")}</span>
                <span className={styles.queueMain}>
                  <strong>{event.recordingId}</strong>
                  <small>{formatTime(event.transitionTime)} · {eventKindLabel(event)}</small>
                </span>
                <span className={styles.queueScore}>
                  {decimal(featureValue(event, "playerPaletteArea"))}
                  {decisions[event.eventId] && <i data-decision={decisions[event.eventId]} />}
                </span>
              </button>
            ))}
            {!filteredEvents.length && <p className={styles.emptyQueue}>No events match these filters.</p>}
          </div>
        </aside>

        <section className={styles.inspector} aria-label="Selected event inspector">
          {selectedEvent ? (
            <>
              <header className={styles.inspectorHeader}>
                <div>
                  <p className={styles.eyebrow}>
                    {selectedEvent.environment} · {eventKindLabel(selectedEvent)} · {statusLabel(selectedEvent.status)}
                  </p>
                  <h2>{selectedEvent.eventId}</h2>
                </div>
                <div className={styles.navigationButtons}>
                  <button type="button" disabled={selectedIndex <= 0} onClick={() => moveSelection(-1)}>← Previous</button>
                  <span>{selectedIndex + 1} / {filteredEvents.length}</span>
                  <button type="button" disabled={selectedIndex < 0 || selectedIndex >= filteredEvents.length - 1} onClick={() => moveSelection(1)}>Next →</button>
                </div>
              </header>

              <div className={styles.eventSummary} data-kind={eventKind(selectedEvent)}>
                <div>
                  <span>Transition point</span>
                  <strong>{formatTime(selectedEvent.transitionTime)}</strong>
                </div>
                <div>
                  <span>Gap window</span>
                  <strong>{formatTime(selectedEvent.gapStart)}–{formatTime(selectedEvent.gapEnd)}</strong>
                </div>
                <div>
                  <span>Sample status</span>
                  <strong>{statusLabel(selectedEvent.status)}</strong>
                </div>
                <div>
                  <span>Appearance distance</span>
                  <strong>{decimal(featureValue(selectedEvent, "playerPaletteArea"))}</strong>
                </div>
              </div>

              <div className={styles.videoStage}>
                <video
                  key={selectedEvent.recordingId}
                  ref={videoRef}
                  controls
                  playsInline
                  preload="metadata"
                  src={eventVideoUrl(selectedEvent.recordingId)}
                  onLoadedMetadata={(event) => {
                    event.currentTarget.currentTime = boundedTime(selectedEvent.transitionTime, duration);
                    setCurrentTime(boundedTime(selectedEvent.transitionTime, duration));
                  }}
                  onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                  aria-label={`Video review for ${selectedEvent.recordingId}`}
                >
                  Your browser does not support video playback.
                </video>
                <span className={styles.videoTime}>{formatTime(currentTime)}</span>
                <span className={styles.videoLabel}>{selectedRecording?.videoFilename ?? selectedEvent.recordingId}</span>
              </div>

              <OverviewTimeline
                events={selectedRecordingEvents}
                duration={duration}
                selectedEventId={selectedEvent.eventId}
                currentTime={currentTime}
                onSelect={selectEvent}
                onSeek={seek}
              />
              <WindowTimeline
                event={selectedEvent}
                duration={duration}
                currentTime={currentTime}
                onSeek={seek}
              />

              <section className={styles.decisionPanel}>
                <div>
                  <span className={styles.panelKicker}>LOCAL REVIEW · NOT GOLD LABELING</span>
                  <strong>Does the visible team appearance change across this gap?</strong>
                  <small>Saved in this browser only. Export the reviewed rows when ready for a labeling pass.</small>
                </div>
                <div className={styles.decisionButtons}>
                  {(["switch", "no-switch", "unclear"] as ReviewDecision[]).map((decision) => (
                    <button
                      type="button"
                      data-active={selectedDecision === decision ? "true" : "false"}
                      data-decision={decision}
                      onClick={() => setDecision(decision)}
                      key={decision}
                    >
                      {decision === "switch" ? "Visible switch" : decision === "no-switch" ? "No switch" : "Unclear"}
                    </button>
                  ))}
                  <button type="button" className={styles.exportButton} onClick={exportDecisions}>
                    Export {reviewedCount} decisions
                  </button>
                </div>
              </section>

              <div className={styles.featureGrid}>
                {FEATURE_DEFINITIONS.map((definition) => (
                  <FeatureCard
                    definition={definition}
                    event={selectedEvent}
                    metric={featureMetric(report, definition.key, scopeLabel)}
                    key={definition.key}
                  />
                ))}
              </div>

              <div className={styles.aggregateGrid}>
                <AggregateCard label="Before window" aggregate={selectedEvent.before} />
                <AggregateCard label="After window" aggregate={selectedEvent.after} />
                <div className={styles.notesCard}>
                  <span>Interpretation guardrail</span>
                  <strong>
                    {selectedEvent.status === "ok"
                      ? "Compare the video with the scalar distance. A high score is evidence, not a decision."
                      : selectedEvent.error ?? "No valid before/after feature pair was available."}
                  </strong>
                  <small>
                    The current report stores aggregate proposal statistics, not player identities or bounding-box tracks.
                  </small>
                </div>
              </div>
            </>
          ) : (
            <div className={styles.emptyInspector}>
              <span>No event selected</span>
              <p>Adjust the queue filters to bring an event into view.</p>
            </div>
          )}
        </section>
      </section>

      <footer className={styles.footer}>
        <span>Source report</span>
        <code>{reportPath}</code>
        <span>Scroll the queue to inspect every sampled gap; indoor recordings are included as controls.</span>
      </footer>
    </main>
  );
}

export function SideSwitchReviewClient({
  report,
  recordings,
  reportPath,
  loadError,
}: SideSwitchReviewClientProps) {
  if (!report) return <UnavailableReview reportPath={reportPath} loadError={loadError} />;
  return (
    <LoadedSideSwitchReview
      report={report}
      recordings={recordings}
      reportPath={reportPath}
    />
  );
}
