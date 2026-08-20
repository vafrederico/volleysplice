"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import { formatTime } from "@/lib/edit-list";
import {
  SIDE_SWITCH_PROPOSAL_MODELS,
  type SideSwitchModelProposal,
  type SideSwitchProposalBundle,
  type SideSwitchProposalModel,
} from "@/lib/side-switch-review-proposals";

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
  proposalBundle: SideSwitchProposalBundle;
  reportPath: string;
  initialDecisions: Record<string, ReviewDecision>;
  initialSavedAt: string | null;
  decisionLoadError?: string;
  loadError?: string;
};

type EventFilter = "all" | "switch" | "unmarked" | "candidate" | "insufficient";
type ProposalFilter = "all" | "any" | "v5" | "v6" | "both" | "disagreement";
type SaveStatus = "idle" | "saving" | "saved" | "error";

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
  { value: "all", label: "All source gaps" },
  { value: "switch", label: "Prior switch markers" },
  { value: "unmarked", label: "Previously unmarked" },
  { value: "candidate", label: "No seed target" },
  { value: "insufficient", label: "Insufficient windows" },
];

const PROPOSAL_FILTERS: Array<{ value: ProposalFilter; label: string }> = [
  { value: "any", label: "V5 or V6" },
  { value: "v5", label: "V5 proposals" },
  { value: "v6", label: "V6 proposals" },
  { value: "both", label: "Both models" },
  { value: "disagreement", label: "Model disagreement" },
  { value: "all", label: "All gaps" },
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

function eventKind(
  event: AppearanceEvent,
): "switch" | "unmarked" | "candidate" {
  if (event.label === 1) return "switch";
  if (event.label === 0) return "unmarked";
  return "candidate";
}

function eventKindLabel(event: AppearanceEvent): string {
  if (event.label === 1) return "Prior switch marker";
  if (event.label === 0) return "Previously unmarked gap";
  return "Gap with no seed target";
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

function reviewStartTime(event: AppearanceEvent): number {
  const beforeTimes = event.beforeTimes.filter((time) => Number.isFinite(time));
  return beforeTimes.length > 0
    ? Math.min(...beforeTimes)
    : event.transitionTime;
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
  recording: SideSwitchRecording | undefined,
  events: AppearanceEvent[],
): number {
  if (recording && recording.durationSeconds > 0)
    return recording.durationSeconds;
  return Math.max(1, ...events.map((event) => event.gapEnd + 5));
}

function eventFilterMatches(
  event: AppearanceEvent,
  filter: EventFilter,
): boolean {
  if (filter === "switch") return event.label === 1;
  if (filter === "unmarked") return event.label === 0;
  if (filter === "candidate") return event.label === null;
  if (filter === "insufficient") return event.status !== "ok";
  return true;
}

function eventProposals(
  event: AppearanceEvent,
  proposalBundle: SideSwitchProposalBundle,
): Partial<Record<SideSwitchProposalModel, SideSwitchModelProposal>> {
  return proposalBundle.byEventId[event.eventId] ?? {};
}

function proposalFilterMatches(
  event: AppearanceEvent,
  filter: ProposalFilter,
  proposalBundle: SideSwitchProposalBundle,
): boolean {
  if (filter === "all") return true;
  const proposals = eventProposals(event, proposalBundle);
  const v5 = proposals.v5?.selected === true;
  const v6 = proposals.v6?.selected === true;
  if (filter === "v5") return v5;
  if (filter === "v6") return v6;
  if (filter === "both") return v5 && v6;
  if (filter === "disagreement") return v5 !== v6;
  return v5 || v6;
}

function proposalTone(
  event: AppearanceEvent,
  proposalBundle: SideSwitchProposalBundle,
): "both" | "v5" | "v6" | "none" {
  const proposals = eventProposals(event, proposalBundle);
  const v5 = proposals.v5?.selected === true;
  const v6 = proposals.v6?.selected === true;
  if (v5 && v6) return "both";
  if (v5) return "v5";
  if (v6) return "v6";
  return "none";
}

function featureMetric(
  report: AppearanceReport,
  key: AppearanceFeature,
  scope = "pooled",
): AppearanceMetric | null {
  return report.summary.metrics[scope]?.features[key] ?? null;
}

function eventVideoUrl(recordingId: string): string {
  return `/api/review-media/side-switch/${encodeURIComponent(recordingId)}`;
}

function savedTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString();
}

function OverviewTimeline({
  events,
  proposalBundle,
  duration,
  selectedEventId,
  currentTime,
  onSelect,
  onSeek,
}: {
  events: AppearanceEvent[];
  proposalBundle: SideSwitchProposalBundle;
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
          <span>
            Every review gap in this recording, including V5/V6 proposals
          </span>
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
        {events.map((item) => (
          <button
            type="button"
            className={styles.overviewEvent}
            data-kind={eventKind(item)}
            data-proposal={proposalTone(item, proposalBundle)}
            data-status={item.status}
            data-selected={item.eventId === selectedEventId ? "true" : "false"}
            style={{ left: `${(item.transitionTime / duration) * 100}%` }}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(item.eventId);
            }}
            title={`${eventKindLabel(item)} · ${formatTime(item.transitionTime)} · ${proposalTone(item, proposalBundle) === "none" ? "no selected model proposal" : `${proposalTone(item, proposalBundle).toUpperCase()} proposal`}`}
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
        <span data-tone="switch">prior switch marker</span>
        <span data-tone="control">other review gap</span>
        <span data-tone="proposal">V5/V6 proposal outline</span>
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
            {formatTime(event.gapStart)}–{formatTime(event.gapEnd)} gap ·{" "}
            {decimal(event.gapSeconds, 1)}s
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
              start +
                ((click.clientX - bounds.left) / bounds.width) * (end - start),
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
        <span data-tone="transition">candidate transition point</span>
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
          <dt>Seed ROC AUC</dt>
          <dd>{decimal(metric?.rocAuc, 4)}</dd>
        </div>
        <div>
          <dt>Seed AP</dt>
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
        box area{" "}
        {aggregate ? percentage(aggregate.meanTotalBoxAreaFraction, 2) : "—"}
        {" · "}
        median height{" "}
        {aggregate ? percentage(aggregate.meanMedianBoxHeightFraction, 1) : "—"}
      </small>
      <small>
        usable proposals {aggregate?.usableFrameCount ?? "—"} · HOG score{" "}
        {decimal(aggregate?.meanDetectionScore, 2)}
      </small>
    </div>
  );
}

function ProposalBadges({
  event,
  proposalBundle,
}: {
  event: AppearanceEvent;
  proposalBundle: SideSwitchProposalBundle;
}) {
  const proposals = eventProposals(event, proposalBundle);
  return (
    <span
      className={styles.proposalBadges}
      role="group"
      aria-label="Selected model proposals"
    >
      {SIDE_SWITCH_PROPOSAL_MODELS.map((modelId) =>
        proposals[modelId]?.selected ? (
          <b data-model={modelId} key={modelId}>
            {modelId.toUpperCase()}
          </b>
        ) : null,
      )}
    </span>
  );
}

function ModelProposalPanel({
  event,
  proposalBundle,
}: {
  event: AppearanceEvent;
  proposalBundle: SideSwitchProposalBundle;
}) {
  const proposals = eventProposals(event, proposalBundle);
  return (
    <section
      className={styles.proposalPanel}
      aria-label="V5 and V6 model proposals"
    >
      <header>
        <div>
          <span className={styles.panelKicker}>
            MODEL PROPOSAL LAYERS · REVIEW EVIDENCE
          </span>
          <strong>Independent suggestions, not ground truth</strong>
        </div>
        <small>
          A model can surface a real switch even when the earlier heuristic seed
          did not.
        </small>
      </header>
      <div className={styles.proposalGrid}>
        {proposalBundle.layers.map((layer) => {
          const proposal = proposals[layer.modelId];
          return (
            <article
              className={styles.proposalCard}
              data-model={layer.modelId}
              data-selected={proposal?.selected === true ? "true" : "false"}
              key={layer.modelId}
            >
              <div>
                <span>{layer.label}</span>
                <strong>
                  {proposal
                    ? proposal.selected
                      ? "Proposes a switch"
                      : "Did not select this gap"
                    : "Outside this model scope"}
                </strong>
                <small>{layer.detail}</small>
              </div>
              <dl>
                <div>
                  <dt>Ranker score</dt>
                  <dd>{decimal(proposal?.score, 4)}</dd>
                </div>
                <div>
                  <dt>Selected threshold</dt>
                  <dd>{decimal(layer.threshold, 4)}</dd>
                </div>
                <div>
                  <dt>Rally gap</dt>
                  <dd>{proposal ? `#${proposal.gapOrder}` : "—"}</dd>
                </div>
                <div>
                  <dt>Cadence margin</dt>
                  <dd>±{layer.candidateMargin} rally</dd>
                </div>
              </dl>
            </article>
          );
        })}
        {proposalBundle.errors.map((error) => (
          <article className={styles.proposalError} key={error.modelId}>
            <strong>{error.modelId.toUpperCase()} layer unavailable</strong>
            <small>{error.message}</small>
          </article>
        ))}
      </div>
    </section>
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
        <p className={styles.eyebrow}>
          SIDE-SWITCH MODEL PROPOSALS · DEV REVIEW
        </p>
        <h1>
          Report not <em>available.</em>
        </h1>
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
  proposalBundle,
  reportPath,
  initialDecisions,
  initialSavedAt,
  decisionLoadError,
}: {
  report: AppearanceReport;
  recordings: SideSwitchRecording[];
  proposalBundle: SideSwitchProposalBundle;
  reportPath: string;
  initialDecisions: Record<string, ReviewDecision>;
  initialSavedAt: string | null;
  decisionLoadError?: string;
}) {
  const allEvents = report.events;
  const [environment, setEnvironment] = useState("all");
  const [recordingId, setRecordingId] = useState("all");
  const [eventFilter, setEventFilter] = useState<EventFilter>("all");
  const [proposalFilter, setProposalFilter] = useState<ProposalFilter>(() =>
    proposalBundle.layers.length > 0 ? "any" : "all",
  );
  const [selectedEventId, setSelectedEventId] = useState(
    () =>
      allEvents.find((event) => event.label !== null)?.eventId ??
      allEvents[0]?.eventId ??
      "",
  );
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [decisions, setDecisions] =
    useState<Record<string, ReviewDecision>>(initialDecisions);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>(
    decisionLoadError ? "error" : initialSavedAt ? "saved" : "idle",
  );
  const [saveError, setSaveError] = useState<string | null>(
    decisionLoadError ?? null,
  );
  const [lastSavedAt, setLastSavedAt] = useState(initialSavedAt);
  const videoRef = useRef<HTMLVideoElement>(null);
  const persistedDecisionSignature = useRef(JSON.stringify(initialDecisions));
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const environments = useMemo(
    () =>
      [...new Set(recordings.map((recording) => recording.environment))].sort(),
    [recordings],
  );

  const filteredEvents = useMemo(
    () =>
      allEvents.filter(
        (event) =>
          (environment === "all" || event.environment === environment) &&
          (recordingId === "all" || event.recordingId === recordingId) &&
          eventFilterMatches(event, eventFilter) &&
          proposalFilterMatches(event, proposalFilter, proposalBundle),
      ),
    [
      allEvents,
      environment,
      eventFilter,
      proposalBundle,
      proposalFilter,
      recordingId,
    ],
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
        ? allEvents.filter(
            (event) => event.recordingId === selectedEvent.recordingId,
          )
        : [],
    [allEvents, selectedEvent],
  );
  const duration = recordingDuration(
    selectedRecording,
    selectedRecordingEvents,
  );
  const selectedIndex = selectedEvent
    ? filteredEvents.findIndex(
        (event) => event.eventId === selectedEvent.eventId,
      )
    : -1;
  const selectedDecision = selectedEvent
    ? decisions[selectedEvent.eventId]
    : undefined;
  const reviewedCount = Object.keys(decisions).length;
  const proposedEventCount = useMemo(
    () =>
      allEvents.filter((event) =>
        proposalFilterMatches(event, "any", proposalBundle),
      ).length,
    [allEvents, proposalBundle],
  );
  const layerByModel = useMemo(
    () =>
      Object.fromEntries(
        proposalBundle.layers.map((layer) => [layer.modelId, layer]),
      ) as Partial<
        Record<SideSwitchProposalModel, (typeof proposalBundle.layers)[number]>
      >,
    [proposalBundle.layers],
  );

  const moveToNext = useCallback(() => {
    if (!filteredEvents.length) return;
    const nextIndex =
      selectedIndex < 0 ? 0 : (selectedIndex + 1) % filteredEvents.length;
    const nextEvent = filteredEvents[nextIndex];
    if (nextEvent) setSelectedEventId(nextEvent.eventId);
  }, [filteredEvents, selectedIndex]);

  const moveToPrevious = useCallback(() => {
    if (selectedIndex <= 0) return;
    const previousEvent = filteredEvents[selectedIndex - 1];
    if (previousEvent) setSelectedEventId(previousEvent.eventId);
  }, [filteredEvents, selectedIndex]);

  const setDecision = useCallback(
    (decision: ReviewDecision) => {
      if (!selectedEvent) return;
      setDecisions((current) => ({
        ...current,
        [selectedEvent.eventId]: decision,
      }));
    },
    [selectedEvent],
  );

  const saveDecisionsToNas = useCallback(
    async (values: Record<string, ReviewDecision>) => {
      setSaveStatus("saving");
      setSaveError(null);
      try {
        const response = await fetch("/api/side-switch-review/decisions", {
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
        if (!response.ok) {
          throw new Error(
            typeof payload?.error === "string"
              ? payload.error
              : `Save failed with HTTP ${response.status}`,
          );
        }
        const savedAt =
          typeof payload?.savedAt === "string"
            ? payload.savedAt
            : new Date().toISOString();
        persistedDecisionSignature.current = JSON.stringify(values);
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
    if (JSON.stringify(decisions) === persistedDecisionSignature.current)
      return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    setSaveStatus("saving");
    setSaveError(null);
    saveTimerRef.current = setTimeout(() => {
      void saveDecisionsToNas(decisions);
    }, 350);
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
    if (filteredEvents.some((event) => event.eventId === selectedEventId))
      return;
    setSelectedEventId(filteredEvents[0]?.eventId ?? "");
  }, [filteredEvents, selectedEventId]);

  useEffect(() => {
    if (!selectedEvent) return;
    const target = boundedTime(reviewStartTime(selectedEvent), duration);
    setCurrentTime(target);
    if (videoRef.current && videoRef.current.readyState >= 1) {
      videoRef.current.currentTime = target;
    }
  }, [duration, selectedEvent]);

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
        moveToNext();
      } else if (key === "p") {
        event.preventDefault();
        moveToPrevious();
      } else if (key === "v") {
        event.preventDefault();
        setDecision("switch");
      } else if (key === "n") {
        event.preventDefault();
        setDecision("no-switch");
      } else if (key === "u") {
        event.preventDefault();
        setDecision("unclear");
      }
    }

    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [moveToNext, moveToPrevious, setDecision]);

  function seek(time: number) {
    const target = boundedTime(time, duration);
    setCurrentTime(target);
    if (videoRef.current) videoRef.current.currentTime = target;
  }

  function selectEvent(eventId: string) {
    setSelectedEventId(eventId);
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
          <Link href="/serving-side-review">Serving-side review ↗</Link>
          <Link href="/suppression-review">Suppression review ↗</Link>
        </nav>
      </header>

      <header className={styles.hero}>
        <div>
          <p className={styles.eyebrow}>
            V5 + V6 PROPOSAL AUDIT · {report.protocol.personProposal}
          </p>
          <h1>
            Find the <em>side flip.</em>
          </h1>
          <p className={styles.intro}>
            Review every switch proposed by the V5 and V6 specialists against
            the source video. Earlier switch markers came from heuristic review;
            they are useful seeds, not an exhaustive truth set, so an unmarked
            proposal can still be a real switch.
          </p>
          <p className={styles.sourceLine}>
            {report.summary.recordings} recordings · {report.summary.events}{" "}
            sampled gaps · {report.protocol.samplesPerSide} frames per side ·
            report {new Date(report.createdAt).toLocaleString()}
          </p>
        </div>
        <div className={styles.heroMetric}>
          <span>V5/V6 proposal union</span>
          <strong>{compactNumber(proposedEventCount)}</strong>
          <small>unique gaps selected by either specialist</small>
          <b>
            {compactNumber(layerByModel.v5?.selectedAttachedEvents ?? 0)} V5 ·{" "}
            {compactNumber(layerByModel.v6?.selectedAttachedEvents ?? 0)} V6
          </b>
        </div>
      </header>

      <section className={styles.summaryStrip} aria-label="Diagnostic summary">
        <div>
          <span>Recordings</span>
          <strong>{compactNumber(report.summary.recordings)}</strong>
        </div>
        <div>
          <span>Prior switch markers</span>
          <strong data-tone="switch">
            {compactNumber(report.summary.positives)}
          </strong>
        </div>
        <div>
          <span>Previously unmarked</span>
          <strong>{compactNumber(report.summary.negatives)}</strong>
        </div>
        <div>
          <span>V5 proposals</span>
          <strong data-tone="model-v5">
            {compactNumber(layerByModel.v5?.selectedAttachedEvents ?? 0)}
          </strong>
        </div>
        <div>
          <span>V6 proposals</span>
          <strong data-tone="model-v6">
            {compactNumber(layerByModel.v6?.selectedAttachedEvents ?? 0)}
          </strong>
        </div>
        <div>
          <span>Reviewed decisions</span>
          <strong>{compactNumber(reviewedCount)}</strong>
        </div>
      </section>

      <section className={styles.protocolPanel}>
        <div>
          <span className={styles.panelKicker}>REVIEW CONTRACT</span>
          <strong>Model proposals are discovery candidates</strong>
          <p>
            Prior <code>sideSwitches</code> markers and unmarked heuristic gaps
            are context, not exhaustive positive/negative labels. V5 and V6 are
            shown as independent layers; decide from the video whether a
            physical side switch occurs.
          </p>
        </div>
        <dl>
          <div>
            <dt>Scope</dt>
            <dd>{report.protocol.environments.join(" · ")}</dd>
          </div>
          <div>
            <dt>Model scope</dt>
            <dd>
              {layerByModel.v5?.attachedEvents ?? 0} V5 ·{" "}
              {layerByModel.v6?.attachedEvents ?? 0} V6 gaps
            </dd>
          </div>
          <div>
            <dt>Review state</dt>
            <dd>{reviewedCount} saved · decisions remain editable</dd>
          </div>
        </dl>
      </section>

      {proposalBundle.errors.length > 0 && (
        <section className={styles.layerWarning} role="status">
          {proposalBundle.errors.map((error) => (
            <span key={error.modelId}>
              {error.modelId.toUpperCase()} could not be loaded: {error.message}
            </span>
          ))}
        </section>
      )}

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
          <span>Seed context</span>
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
        <div className={styles.filterChoices}>
          <span>Model proposal layer</span>
          <div>
            {PROPOSAL_FILTERS.map((item) => (
              <button
                type="button"
                data-active={proposalFilter === item.value ? "true" : "false"}
                onClick={() => setProposalFilter(item.value)}
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
            <small>{scopeLabel} appearance context</small>
          </header>
          <div className={styles.queueList}>
            {filteredEvents.map((event, index) => (
              <button
                type="button"
                className={styles.queueItem}
                data-active={
                  event.eventId === selectedEvent?.eventId ? "true" : "false"
                }
                data-kind={eventKind(event)}
                data-proposal={proposalTone(event, proposalBundle)}
                onClick={() => selectEvent(event.eventId)}
                key={event.eventId}
              >
                <span className={styles.queueIndex}>
                  {String(index + 1).padStart(3, "0")}
                </span>
                <span className={styles.queueMain}>
                  <strong>{event.recordingId}</strong>
                  <small>
                    {formatTime(event.transitionTime)} · {eventKindLabel(event)}
                  </small>
                </span>
                <span className={styles.queueSignals}>
                  <ProposalBadges
                    event={event}
                    proposalBundle={proposalBundle}
                  />
                  <span className={styles.queueScore}>
                    {decimal(featureValue(event, "playerPaletteArea"))}
                    {decisions[event.eventId] && (
                      <i data-decision={decisions[event.eventId]} />
                    )}
                  </span>
                </span>
              </button>
            ))}
            {!filteredEvents.length && (
              <p className={styles.emptyQueue}>
                No events match these filters.
              </p>
            )}
          </div>
        </aside>

        <section
          className={styles.inspector}
          aria-label="Selected event inspector"
        >
          {selectedEvent ? (
            <>
              <header className={styles.inspectorHeader}>
                <div>
                  <p className={styles.eyebrow}>
                    {selectedEvent.environment} ·{" "}
                    {eventKindLabel(selectedEvent)} ·{" "}
                    {selectedEvent.targetStatus} ·{" "}
                    {statusLabel(selectedEvent.status)}
                  </p>
                  <h2>{selectedEvent.eventId}</h2>
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
                    {selectedIndex + 1} / {filteredEvents.length}
                  </span>
                  <button
                    type="button"
                    disabled={filteredEvents.length < 2}
                    onClick={moveToNext}
                    title="Next event in the filtered queue (J)"
                  >
                    Next candidate <kbd>J</kbd> →
                  </button>
                </div>
              </header>

              <div
                className={styles.eventSummary}
                data-kind={eventKind(selectedEvent)}
              >
                <div>
                  <span>Transition point</span>
                  <strong>{formatTime(selectedEvent.transitionTime)}</strong>
                </div>
                <div>
                  <span>Gap window</span>
                  <strong>
                    {formatTime(selectedEvent.gapStart)}–
                    {formatTime(selectedEvent.gapEnd)}
                  </strong>
                </div>
                <div>
                  <span>Source status</span>
                  <strong>{selectedEvent.targetStatus}</strong>
                </div>
                <div>
                  <span>Appearance distance</span>
                  <strong>
                    {decimal(featureValue(selectedEvent, "playerPaletteArea"))}
                  </strong>
                </div>
              </div>

              <ModelProposalPanel
                event={selectedEvent}
                proposalBundle={proposalBundle}
              />

              <div className={styles.videoStage}>
                <video
                  key={selectedEvent.recordingId}
                  ref={videoRef}
                  controls
                  playsInline
                  preload="metadata"
                  src={eventVideoUrl(selectedEvent.recordingId)}
                  onLoadedMetadata={(event) => {
                    const startTime = boundedTime(
                      reviewStartTime(selectedEvent),
                      duration,
                    );
                    event.currentTarget.currentTime = startTime;
                    event.currentTarget.playbackRate = playbackRate;
                    setCurrentTime(startTime);
                  }}
                  onTimeUpdate={(event) =>
                    setCurrentTime(event.currentTarget.currentTime)
                  }
                  aria-label={`Video review for ${selectedEvent.recordingId}`}
                >
                  Your browser does not support video playback.
                </video>
                <span className={styles.videoTime}>
                  {formatTime(currentTime)}
                </span>
                <span className={styles.videoLabel}>
                  {selectedRecording?.videoFilename ??
                    selectedEvent.recordingId}
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
                    <kbd>J</kbd> next candidate
                  </span>
                  <span>
                    <kbd>P</kbd> previous
                  </span>
                  <span>
                    <kbd>V</kbd> visible switch
                  </span>
                  <span>
                    <kbd>N</kbd> no switch
                  </span>
                  <span>
                    <kbd>U</kbd> unclear
                  </span>
                </div>
              </div>

              <OverviewTimeline
                events={selectedRecordingEvents}
                proposalBundle={proposalBundle}
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
                  <span className={styles.panelKicker}>
                    MODEL PROPOSAL AUDIT · NAS-BACKED REVIEW
                  </span>
                  <strong>
                    Does a physical side switch occur across this gap?
                  </strong>
                  <small>
                    Judge the video, not the earlier marker or model badge.
                    Prior markers were heuristic review seeds and may omit real
                    switches. Decisions autosave to NAS.
                  </small>
                </div>
                <div className={styles.decisionButtons}>
                  {(["switch", "no-switch", "unclear"] as ReviewDecision[]).map(
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
                        {decision === "switch"
                          ? "Switch occurs"
                          : decision === "no-switch"
                            ? "No switch"
                            : "Unclear"}
                        <kbd>
                          {decision === "switch"
                            ? "V"
                            : decision === "no-switch"
                              ? "N"
                              : "U"}
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
                <AggregateCard
                  label="Before window"
                  aggregate={selectedEvent.before}
                />
                <AggregateCard
                  label="After window"
                  aggregate={selectedEvent.after}
                />
                <div className={styles.notesCard}>
                  <span>Interpretation guardrail</span>
                  <strong>
                    {selectedEvent.status === "ok"
                      ? "Compare the video with both model layers. A selected proposal or high score is evidence, not a decision."
                      : (selectedEvent.error ??
                        "No valid before/after feature pair was available.")}
                  </strong>
                  <small>
                    The current report stores aggregate proposal statistics, not
                    player identities or bounding-box tracks.
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
        <span>
          V5/V6 predictions are attached without changing the report identity,
          so existing review decisions remain intact.
        </span>
      </footer>
    </main>
  );
}

export function SideSwitchReviewClient({
  report,
  recordings,
  proposalBundle,
  reportPath,
  initialDecisions,
  initialSavedAt,
  decisionLoadError,
  loadError,
}: SideSwitchReviewClientProps) {
  if (!report)
    return <UnavailableReview reportPath={reportPath} loadError={loadError} />;
  return (
    <LoadedSideSwitchReview
      report={report}
      recordings={recordings}
      proposalBundle={proposalBundle}
      reportPath={reportPath}
      initialDecisions={initialDecisions}
      initialSavedAt={initialSavedAt}
      decisionLoadError={decisionLoadError}
    />
  );
}
