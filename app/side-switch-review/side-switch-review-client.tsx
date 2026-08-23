"use client";

import Link from "next/link";
import {
  type MouseEvent as ReactMouseEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

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
  type FullVideoSideSwitchMarker,
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
  initialMarkers: FullVideoSideSwitchMarker[];
  initialReviewedRecordingIds: string[];
  initialMarkersSavedAt: string | null;
  markerLoadError?: string;
  loadError?: string;
};

type EventFilter = "all" | "switch" | "unmarked" | "candidate" | "insufficient";
type ProposalFilter =
  | "all"
  | "any"
  | "multiple"
  | "disagreement"
  | SideSwitchProposalModel;
type SaveStatus = "idle" | "saving" | "saved" | "error";

const MODEL_SHORT_LABELS: Record<SideSwitchProposalModel, string> = {
  v5: "V5",
  "v5-state": "V5 + state",
  v6: "V6",
};

const MODEL_BADGE_LABELS: Record<SideSwitchProposalModel, string> = {
  v5: "V5",
  "v5-state": "V5+S",
  v6: "V6",
};

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
  { value: "any", label: "Any variant" },
  { value: "v5", label: "V5 proposals" },
  { value: "v5-state", label: "V5 + state proposals" },
  { value: "v6", label: "V6 proposals" },
  { value: "multiple", label: "2+ variants" },
  { value: "disagreement", label: "Variant disagreement" },
  { value: "all", label: "All gaps" },
];

const OVERVIEW_TICK_RATIOS = [0, 0.2, 0.4, 0.6, 0.8, 1];
const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 2, 4];
const FULL_VIDEO_MARKER_KIND = "volleycut-full-video-side-switch-markers-v1";

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
  const loadedModels = proposalBundle.layers.map((layer) => layer.modelId);
  const selectedCount = loadedModels.filter(
    (modelId) => proposals[modelId]?.selected === true,
  ).length;
  if (filter === "any") return selectedCount > 0;
  if (filter === "multiple") return selectedCount >= 2;
  if (filter === "disagreement") {
    return (
      loadedModels.length >= 2 &&
      selectedCount > 0 &&
      selectedCount < loadedModels.length
    );
  }
  return proposals[filter]?.selected === true;
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

function preciseTime(value: number): string {
  const safe = Math.max(0, Number.isFinite(value) ? value : 0);
  const minutes = Math.floor(safe / 60);
  const seconds = safe - minutes * 60;
  return `${minutes}:${seconds.toFixed(3).padStart(6, "0")}`;
}

function VariantTimelines({
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
    <section
      className={styles.timelineBlock}
      aria-label="Side-switch variant timelines"
    >
      <div className={styles.timelineHeading}>
        <div>
          <strong>Variant timelines</strong>
          <span>
            One aligned proposal rail per model variant for this recording
          </span>
        </div>
        <span>{formatTime(duration)}</span>
      </div>
      <div className={styles.variantTimelineAxis}>
        <span aria-hidden="true" />
        <div className={styles.overviewAxis}>
          {OVERVIEW_TICK_RATIOS.map((ratio) => (
            <span key={`overview-tick-${ratio}`}>
              {formatTime(duration * ratio)}
            </span>
          ))}
        </div>
      </div>
      <div className={styles.variantTimelineRows}>
        {proposalBundle.layers.map((layer) => {
          const scopedCount = events.filter(
            (item) => eventProposals(item, proposalBundle)[layer.modelId],
          ).length;
          const selectedCount = events.filter(
            (item) =>
              eventProposals(item, proposalBundle)[layer.modelId]?.selected ===
              true,
          ).length;
          return (
            <div
              className={styles.variantTimelineRow}
              data-model={layer.modelId}
              key={layer.modelId}
            >
              <div className={styles.variantTimelineLabel}>
                <strong>{layer.label}</strong>
                <span>
                  {selectedCount} proposals · {scopedCount} scored gaps
                </span>
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
                {events.map((item, eventIndex) => {
                  const proposal = eventProposals(item, proposalBundle)[
                    layer.modelId
                  ];
                  return (
                    <button
                      type="button"
                      className={styles.variantEvent}
                      data-kind={eventKind(item)}
                      data-proposed={proposal?.selected ? "true" : "false"}
                      data-scoped={proposal ? "true" : "false"}
                      data-label-row={eventIndex % 2}
                      data-status={item.status}
                      data-selected={
                        item.eventId === selectedEventId ? "true" : "false"
                      }
                      style={{
                        left: `${percentageAt(item.transitionTime, 0, duration)}%`,
                      }}
                      onClick={(event) => {
                        event.stopPropagation();
                        onSelect(item.eventId);
                      }}
                      title={`${layer.label} · ${proposal?.selected ? "proposal" : proposal ? "not selected" : "outside model scope"} · ${eventKindLabel(item)} · ${formatTime(item.transitionTime)}`}
                      aria-label={`Select ${item.eventId} on the ${layer.label} timeline`}
                      key={item.eventId}
                    >
                      {proposal?.selected && (
                        <span className={styles.variantEventLabel}>
                          {MODEL_BADGE_LABELS[layer.modelId]} ·{" "}
                          {formatTime(item.transitionTime)}
                        </span>
                      )}
                    </button>
                  );
                })}
                <span
                  className={styles.overviewPlayhead}
                  style={{ left: `${percentageAt(currentTime, 0, duration)}%` }}
                />
              </div>
            </div>
          );
        })}
        {proposalBundle.layers.length === 0 && (
          <p className={styles.emptyVariantTimelines}>
            No proposal variant could be loaded for this recording.
          </p>
        )}
      </div>
      <div className={styles.timelineLegend}>
        <span data-tone="proposal">solid variant proposal</span>
        <span data-tone="switch">thin prior switch marker</span>
        <span data-tone="control">other scored gap</span>
        <span data-tone="selected">selected event</span>
      </div>
    </section>
  );
}

function FullVideoCoverageTimeline({
  recording,
  events,
  decisions,
  markers,
  reviewedComplete,
  duration,
  currentTime,
  markerSaveStatus,
  markerSavedAt,
  markerSaveError,
  onSelect,
  onSeek,
  onAddMarker,
  onRemoveMarker,
  onSetReviewedComplete,
}: {
  recording: SideSwitchRecording;
  events: AppearanceEvent[];
  decisions: Record<string, ReviewDecision>;
  markers: FullVideoSideSwitchMarker[];
  reviewedComplete: boolean;
  duration: number;
  currentTime: number;
  markerSaveStatus: SaveStatus;
  markerSavedAt: string | null;
  markerSaveError: string | null;
  onSelect: (eventId: string) => void;
  onSeek: (time: number) => void;
  onAddMarker: () => void;
  onRemoveMarker: (markerId: string) => void;
  onSetReviewedComplete: (reviewed: boolean) => void;
}) {
  const reviewedSwitches = events.filter(
    (event) => decisions[event.eventId] === "switch",
  );
  const completeFromSource = recording.continuousVideoReviewed;
  const complete = completeFromSource || reviewedComplete;
  const seekFromRail = (event: ReactMouseEvent<HTMLDivElement>): void => {
    const bounds = event.currentTarget.getBoundingClientRect();
    onSeek(
      boundedTime(
        ((event.clientX - bounds.left) / bounds.width) * duration,
        duration,
      ),
    );
  };

  return (
    <section
      className={`${styles.timelineBlock} ${styles.coverageTimeline}`}
      data-complete={complete ? "true" : "false"}
      aria-label="Full-video human side-switch coverage"
    >
      <div className={styles.timelineHeading}>
        <div>
          <strong>Human switch coverage</strong>
          <span>
            {complete
              ? completeFromSource
                ? "Source label document records continuous-video review"
                : "This recording is marked as fully reviewed"
              : "Candidate reviews are not exhaustive—scan the full video and add every switch"}
          </span>
        </div>
        <span>{formatTime(duration)}</span>
      </div>
      <div className={styles.coverageActions}>
        <button type="button" onClick={onAddMarker}>
          + Add switch at {preciseTime(currentTime)} <kbd>M</kbd>
        </button>
        {!completeFromSource && (
          <button
            type="button"
            data-complete={reviewedComplete ? "true" : "false"}
            onClick={() => onSetReviewedComplete(!reviewedComplete)}
          >
            {reviewedComplete
              ? "✓ Full video reviewed"
              : "Mark full video reviewed"}
          </button>
        )}
        <span className={styles.saveStatus} data-status={markerSaveStatus}>
          {markerSaveStatus === "saving" && "Saving marker coverage…"}
          {markerSaveStatus === "saved" &&
            `Marker coverage saved ${savedTime(markerSavedAt)}`}
          {markerSaveStatus === "error" &&
            `Marker save failed: ${markerSaveError ?? "unknown error"}`}
          {markerSaveStatus === "idle" && "No marker edits yet"}
        </span>
      </div>
      <div className={styles.variantTimelineAxis}>
        <span aria-hidden="true" />
        <div className={styles.overviewAxis}>
          {OVERVIEW_TICK_RATIOS.map((ratio) => (
            <span key={`coverage-tick-${ratio}`}>
              {formatTime(duration * ratio)}
            </span>
          ))}
        </div>
      </div>
      <div className={styles.coverageTimelineRows}>
        <div className={styles.coverageTimelineRow}>
          <div className={styles.variantTimelineLabel} data-tone="source">
            <strong>Explicit source markers</strong>
            <span>{recording.sourceSideSwitches.length} point labels</span>
          </div>
          <div
            className={styles.overviewRail}
            onClick={seekFromRail}
            role="presentation"
          >
            {recording.sourceSideSwitches.map((marker) => (
              <button
                type="button"
                className={styles.coverageMarker}
                data-tone="source"
                style={{ left: `${percentageAt(marker.time, 0, duration)}%` }}
                onClick={(event) => {
                  event.stopPropagation();
                  onSeek(marker.time);
                }}
                title={`Source side-switch marker · ${preciseTime(marker.time)}${marker.notes ? ` · ${marker.notes}` : ""}`}
                aria-label={`Seek to source side-switch marker at ${preciseTime(marker.time)}`}
                key={marker.time}
              >
                <span>{formatTime(marker.time)}</span>
              </button>
            ))}
            <span
              className={styles.overviewPlayhead}
              style={{ left: `${percentageAt(currentTime, 0, duration)}%` }}
            />
          </div>
        </div>
        <div className={styles.coverageTimelineRow}>
          <div className={styles.variantTimelineLabel} data-tone="reviewed">
            <strong>Candidate review labels</strong>
            <span>{reviewedSwitches.length} reviewed as switch</span>
          </div>
          <div
            className={styles.overviewRail}
            onClick={seekFromRail}
            role="presentation"
          >
            {reviewedSwitches.map((event) => (
              <button
                type="button"
                className={styles.coverageMarker}
                data-tone="reviewed"
                style={{
                  left: `${percentageAt(event.transitionTime, 0, duration)}%`,
                }}
                onClick={(click) => {
                  click.stopPropagation();
                  onSelect(event.eventId);
                }}
                title={`Human-reviewed candidate · ${event.eventId} · ${preciseTime(event.transitionTime)}`}
                aria-label={`Select reviewed switch candidate at ${preciseTime(event.transitionTime)}`}
                key={event.eventId}
              >
                <span>{formatTime(event.transitionTime)}</span>
              </button>
            ))}
            <span
              className={styles.overviewPlayhead}
              style={{ left: `${percentageAt(currentTime, 0, duration)}%` }}
            />
          </div>
        </div>
        <div className={styles.coverageTimelineRow}>
          <div className={styles.variantTimelineLabel} data-tone="manual">
            <strong>Full-video markers</strong>
            <span>{markers.length} manually placed points</span>
          </div>
          <div
            className={styles.overviewRail}
            onClick={seekFromRail}
            role="presentation"
          >
            {markers.map((marker) => (
              <button
                type="button"
                className={styles.coverageMarker}
                data-tone="manual"
                style={{ left: `${percentageAt(marker.time, 0, duration)}%` }}
                onClick={(event) => {
                  event.stopPropagation();
                  onSeek(marker.time);
                }}
                title={`Full-video switch marker · ${preciseTime(marker.time)}`}
                aria-label={`Seek to full-video switch marker at ${preciseTime(marker.time)}`}
                key={marker.id}
              >
                <span>{formatTime(marker.time)}</span>
              </button>
            ))}
            <span
              className={styles.overviewPlayhead}
              style={{ left: `${percentageAt(currentTime, 0, duration)}%` }}
            />
          </div>
        </div>
      </div>
      {markers.length > 0 && (
        <div
          className={styles.markerList}
          role="list"
          aria-label="Editable full-video markers"
        >
          {markers.map((marker) => (
            <span role="listitem" key={marker.id}>
              <button type="button" onClick={() => onSeek(marker.time)}>
                {preciseTime(marker.time)}
              </button>
              <button
                type="button"
                onClick={() => onRemoveMarker(marker.id)}
                aria-label={`Remove full-video marker at ${preciseTime(marker.time)}`}
                title="Remove marker"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      <div className={styles.timelineLegend}>
        <span data-tone="source">explicit source marker</span>
        <span data-tone="reviewed">reviewed candidate switch</span>
        <span data-tone="manual">full-video marker</span>
        <span data-tone="selected">playhead</span>
      </div>
    </section>
  );
}

function ProductionContextTimelines({
  recording,
  duration,
  currentTime,
  onSeek,
}: {
  recording: SideSwitchRecording;
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}) {
  const hasProductionBundle =
    recording.productionModelRanges.length > 0 ||
    recording.productionEditorRanges.length > 0 ||
    recording.productionFinalIntervals.length > 0;
  const producerLabel =
    recording.feedbackProducer === "android"
      ? "saved Android feedback"
      : recording.feedbackProducer === "production-web"
        ? "saved production-web feedback"
        : "saved model feedback";
  const seekFromRail = (event: ReactMouseEvent<HTMLDivElement>): void => {
    const bounds = event.currentTarget.getBoundingClientRect();
    onSeek(
      boundedTime(
        ((event.clientX - bounds.left) / bounds.width) * duration,
        duration,
      ),
    );
  };
  const intervalStyle = (start: number, end: number) => ({
    left: `${percentageAt(start, 0, duration)}%`,
    width: `${percentageAt(end, 0, duration) - percentageAt(start, 0, duration)}%`,
  });

  return (
    <section
      className={`${styles.timelineBlock} ${styles.productionTimelines}`}
      aria-label="Production web app model and editor timelines"
    >
      <div className={styles.timelineHeading}>
        <div>
          <strong>Production web context</strong>
          <span>
            Production model labels plus the web-editor-equivalent cut timeline
            reconstructed from {producerLabel}
          </span>
        </div>
        <span>{formatTime(duration)}</span>
      </div>
      <div className={styles.variantTimelineAxis}>
        <span aria-hidden="true" />
        <div className={styles.overviewAxis}>
          {OVERVIEW_TICK_RATIOS.map((ratio) => (
            <span key={`production-tick-${ratio}`}>
              {formatTime(duration * ratio)}
            </span>
          ))}
        </div>
      </div>
      {hasProductionBundle ? (
        <div className={styles.productionTimelineRows}>
          {recording.gameWindow && (
            <div className={styles.productionTimelineRow} data-track="window">
              <div className={styles.variantTimelineLabel}>
                <strong>Game window</strong>
                <span>
                  {formatTime(recording.gameWindow.start)}–
                  {formatTime(recording.gameWindow.end)}
                </span>
              </div>
              <div
                className={styles.overviewRail}
                onClick={seekFromRail}
                role="presentation"
              >
                <button
                  type="button"
                  className={styles.productionRange}
                  data-tone="window"
                  style={intervalStyle(
                    recording.gameWindow.start,
                    recording.gameWindow.end,
                  )}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSeek(recording.gameWindow?.start ?? 0);
                  }}
                  title={`Production analysis game window · ${preciseTime(recording.gameWindow.start)}–${preciseTime(recording.gameWindow.end)}`}
                >
                  <span>game</span>
                </button>
                <span
                  className={styles.overviewPlayhead}
                  style={{
                    left: `${percentageAt(currentTime, 0, duration)}%`,
                  }}
                />
              </div>
            </div>
          )}
          <div className={styles.productionTimelineRow} data-track="model">
            <div className={styles.variantTimelineLabel}>
              <strong>Production model labels</strong>
              <span>
                {recording.productionModelRanges.length} initial inference
                ranges
              </span>
            </div>
            <div
              className={styles.overviewRail}
              onClick={seekFromRail}
              role="presentation"
            >
              {recording.productionModelRanges.map((range) => (
                <button
                  type="button"
                  className={styles.productionRange}
                  data-tone="model"
                  style={intervalStyle(range.start, range.end)}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSeek(range.start);
                  }}
                  title={`${range.id} · initial production model label · ${preciseTime(range.start)}–${preciseTime(range.end)}${range.confidence === null ? "" : ` · ${Math.round(range.confidence * 100)}%`}${range.agreement ? ` · ${range.agreement}` : ""}`}
                  aria-label={`Seek to production model label ${range.id}`}
                  key={range.id}
                >
                  <span>{range.id}</span>
                </button>
              ))}
              <span
                className={styles.overviewPlayhead}
                style={{ left: `${percentageAt(currentTime, 0, duration)}%` }}
              />
            </div>
          </div>
          <div className={styles.productionTimelineRow} data-track="editor">
            <div className={styles.variantTimelineLabel}>
              <strong>Production web editor view</strong>
              <span>
                {
                  recording.productionEditorRanges.filter(
                    (range) => range.included,
                  ).length
                }{" "}
                kept ·{" "}
                {
                  recording.productionEditorRanges.filter(
                    (range) => !range.included,
                  ).length
                }{" "}
                removed
              </span>
            </div>
            <div
              className={styles.overviewRail}
              onClick={seekFromRail}
              role="presentation"
            >
              {recording.productionEditorRanges.map((range) => (
                <button
                  type="button"
                  className={styles.productionRange}
                  data-tone="editor"
                  data-included={range.included ? "true" : "false"}
                  style={intervalStyle(range.keepStart, range.keepEnd)}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSeek(range.keepStart);
                  }}
                  title={`${range.id} · ${range.included ? "kept" : "removed"} in production editor · ${preciseTime(range.keepStart)}–${preciseTime(range.keepEnd)}${range.agreement ? ` · ${range.agreement}` : ""}`}
                  aria-label={`Seek to ${range.included ? "kept" : "removed"} production editor cut ${range.id}`}
                  key={range.id}
                >
                  {range.included && range.coreStart > range.keepStart && (
                    <i
                      data-padding="before"
                      style={{
                        left: 0,
                        width: `${percentageAt(range.coreStart, range.keepStart, range.keepEnd)}%`,
                      }}
                    />
                  )}
                  <b
                    style={{
                      left: `${percentageAt(range.coreStart, range.keepStart, range.keepEnd)}%`,
                      width: `${percentageAt(range.coreEnd, range.keepStart, range.keepEnd) - percentageAt(range.coreStart, range.keepStart, range.keepEnd)}%`,
                    }}
                  />
                  {range.included && range.keepEnd > range.coreEnd && (
                    <i
                      data-padding="after"
                      style={{
                        left: `${percentageAt(range.coreEnd, range.keepStart, range.keepEnd)}%`,
                        width: `${100 - percentageAt(range.coreEnd, range.keepStart, range.keepEnd)}%`,
                      }}
                    />
                  )}
                  <span>{range.id}</span>
                </button>
              ))}
              {recording.productionIgnoredIntervals.map((interval) => (
                <span
                  className={styles.productionIgnored}
                  style={intervalStyle(interval.start, interval.end)}
                  title={`${interval.id} · ignored${interval.reason ? ` · ${interval.reason}` : ""}`}
                  key={interval.id}
                />
              ))}
              <span
                className={styles.overviewPlayhead}
                style={{ left: `${percentageAt(currentTime, 0, duration)}%` }}
              />
            </div>
          </div>
          <div className={styles.productionTimelineRow} data-track="export">
            <div className={styles.variantTimelineLabel}>
              <strong>Production final export</strong>
              <span>
                {recording.productionFinalIntervals.length} materialized
                intervals
              </span>
            </div>
            <div
              className={styles.overviewRail}
              onClick={seekFromRail}
              role="presentation"
            >
              {recording.productionFinalIntervals.map((range, index) => (
                <button
                  type="button"
                  className={styles.productionRange}
                  data-tone="export"
                  style={intervalStyle(range.start, range.end)}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSeek(range.start);
                  }}
                  title={`Export ${index + 1} · ${preciseTime(range.start)}–${preciseTime(range.end)} · ${range.cutIds.join(", ")}`}
                  aria-label={`Seek to production final export interval ${index + 1}`}
                  key={`${range.start}-${range.end}`}
                >
                  <span>E{index + 1}</span>
                </button>
              ))}
              {recording.productionIgnoredIntervals.map((interval) => (
                <span
                  className={styles.productionIgnored}
                  style={intervalStyle(interval.start, interval.end)}
                  title={`${interval.id} · ignored${interval.reason ? ` · ${interval.reason}` : ""}`}
                  key={interval.id}
                />
              ))}
              <span
                className={styles.overviewPlayhead}
                style={{ left: `${percentageAt(currentTime, 0, duration)}%` }}
              />
            </div>
          </div>
        </div>
      ) : (
        <p className={styles.emptyVariantTimelines}>
          {recording.timelineLoadError
            ? `The source timeline could not be loaded: ${recording.timelineLoadError}`
            : "No production model-feedback bundle is attached to this recording."}
        </p>
      )}
      <div className={styles.timelineLegend}>
        <span data-tone="model-label">initial model label</span>
        <span data-tone="editor-kept">editor kept</span>
        <span data-tone="editor-removed">editor removed</span>
        <span data-tone="export">final export</span>
        <span data-tone="ignored">ignored time</span>
      </div>
    </section>
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
            {MODEL_BADGE_LABELS[modelId]}
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
      aria-label="Side-switch model variant proposals"
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
            <strong>
              {MODEL_SHORT_LABELS[error.modelId]} layer unavailable
            </strong>
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
  initialMarkers,
  initialReviewedRecordingIds,
  initialMarkersSavedAt,
  markerLoadError,
}: {
  report: AppearanceReport;
  recordings: SideSwitchRecording[];
  proposalBundle: SideSwitchProposalBundle;
  reportPath: string;
  initialDecisions: Record<string, ReviewDecision>;
  initialSavedAt: string | null;
  decisionLoadError?: string;
  initialMarkers: FullVideoSideSwitchMarker[];
  initialReviewedRecordingIds: string[];
  initialMarkersSavedAt: string | null;
  markerLoadError?: string;
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
  const [markers, setMarkers] =
    useState<FullVideoSideSwitchMarker[]>(initialMarkers);
  const [reviewedRecordingIds, setReviewedRecordingIds] = useState(
    initialReviewedRecordingIds,
  );
  const [markerSaveStatus, setMarkerSaveStatus] = useState<SaveStatus>(
    markerLoadError ? "error" : initialMarkersSavedAt ? "saved" : "idle",
  );
  const [markerSaveError, setMarkerSaveError] = useState<string | null>(
    markerLoadError ?? null,
  );
  const [markersLastSavedAt, setMarkersLastSavedAt] = useState(
    initialMarkersSavedAt,
  );
  const videoRef = useRef<HTMLVideoElement>(null);
  const persistedDecisionSignature = useRef(JSON.stringify(initialDecisions));
  const persistedMarkerSignature = useRef(
    JSON.stringify({
      markers: initialMarkers,
      reviewedRecordingIds: initialReviewedRecordingIds,
    }),
  );
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const markerSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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
  const markerCount = markers.length;
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

  const selectedRecordingMarkers = useMemo(
    () =>
      selectedRecording
        ? markers
            .filter(
              (marker) => marker.recordingId === selectedRecording.recordingId,
            )
            .sort((left, right) => left.time - right.time)
        : [],
    [markers, selectedRecording],
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

  const saveMarkersToNas = useCallback(
    async (values: FullVideoSideSwitchMarker[], reviewedIds: string[]) => {
      setMarkerSaveStatus("saving");
      setMarkerSaveError(null);
      try {
        const response = await fetch("/api/side-switch-review/markers", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            schemaVersion: 1,
            kind: FULL_VIDEO_MARKER_KIND,
            reportKind: report.kind,
            reportCreatedAt: report.createdAt,
            markers: values,
            reviewedRecordingIds: reviewedIds,
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
        persistedMarkerSignature.current = JSON.stringify({
          markers: values,
          reviewedRecordingIds: reviewedIds,
        });
        setMarkersLastSavedAt(savedAt);
        setMarkerSaveStatus("saved");
      } catch (error) {
        setMarkerSaveStatus("error");
        setMarkerSaveError(
          error instanceof Error ? error.message : String(error),
        );
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

  useEffect(() => {
    const signature = JSON.stringify({ markers, reviewedRecordingIds });
    if (signature === persistedMarkerSignature.current) return;
    if (markerSaveTimerRef.current) clearTimeout(markerSaveTimerRef.current);
    setMarkerSaveStatus("saving");
    setMarkerSaveError(null);
    markerSaveTimerRef.current = setTimeout(() => {
      void saveMarkersToNas(markers, reviewedRecordingIds);
    }, 350);
    return () => {
      if (markerSaveTimerRef.current) clearTimeout(markerSaveTimerRef.current);
    };
  }, [markers, reviewedRecordingIds, saveMarkersToNas]);

  useEffect(
    () => () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      if (markerSaveTimerRef.current) clearTimeout(markerSaveTimerRef.current);
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

  const addMarkerAtCurrentTime = useCallback(() => {
    if (!selectedRecording) return;
    const time = Math.round(boundedTime(currentTime, duration) * 1000) / 1000;
    const suffix =
      typeof globalThis.crypto?.randomUUID === "function"
        ? globalThis.crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const marker: FullVideoSideSwitchMarker = {
      id: `manual:${selectedRecording.recordingId}:${suffix}`,
      recordingId: selectedRecording.recordingId,
      time,
      createdAt: new Date().toISOString(),
    };
    setMarkers((current) =>
      [...current, marker].sort(
        (left, right) =>
          left.recordingId.localeCompare(right.recordingId) ||
          left.time - right.time ||
          left.id.localeCompare(right.id),
      ),
    );
  }, [currentTime, duration, selectedRecording]);

  const removeMarker = useCallback((markerId: string) => {
    setMarkers((current) => current.filter((marker) => marker.id !== markerId));
  }, []);

  const setRecordingReviewedComplete = useCallback(
    (reviewed: boolean) => {
      if (!selectedRecording) return;
      setReviewedRecordingIds((current) => {
        const next = new Set(current);
        if (reviewed) next.add(selectedRecording.recordingId);
        else next.delete(selectedRecording.recordingId);
        return [...next].sort((left, right) => left.localeCompare(right));
      });
    },
    [selectedRecording],
  );

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
      } else if (key === "m") {
        event.preventDefault();
        addMarkerAtCurrentTime();
      }
    }

    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [addMarkerAtCurrentTime, moveToNext, moveToPrevious, setDecision]);

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
            V5 + V5 STATE + V6 PROPOSAL AUDIT · {report.protocol.personProposal}
          </p>
          <h1>
            Find the <em>side flip.</em>
          </h1>
          <p className={styles.intro}>
            Review every switch proposed by V5, the production-state V5 variant,
            and V6 against the source video. Earlier switch markers came from
            heuristic review; they are useful seeds, not an exhaustive truth
            set, so an unmarked proposal can still be a real switch.
          </p>
          <p className={styles.sourceLine}>
            {report.summary.recordings} recordings · {report.summary.events}{" "}
            sampled gaps · {report.protocol.samplesPerSide} frames per side ·
            report {new Date(report.createdAt).toLocaleString()}
          </p>
        </div>
        <div className={styles.heroMetric}>
          <span>Three-variant proposal union</span>
          <strong>{compactNumber(proposedEventCount)}</strong>
          <small>unique gaps selected by any loaded variant</small>
          <b>
            {proposalBundle.layers
              .map(
                (layer) =>
                  `${compactNumber(layer.selectedAttachedEvents)} ${MODEL_SHORT_LABELS[layer.modelId]}`,
              )
              .join(" · ")}
          </b>
        </div>
      </header>

      <section
        className={`${styles.summaryStrip} ${styles.variantSummaryStrip}`}
        aria-label="Diagnostic summary"
      >
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
          <span>V5 + state proposals</span>
          <strong data-tone="model-v5-state">
            {compactNumber(
              layerByModel["v5-state"]?.selectedAttachedEvents ?? 0,
            )}
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
        <div>
          <span>Full-video markers</span>
          <strong data-tone="switch">{compactNumber(markerCount)}</strong>
        </div>
        <div>
          <span>Full videos reviewed</span>
          <strong>
            {compactNumber(
              recordings.filter(
                (recording) =>
                  recording.continuousVideoReviewed ||
                  reviewedRecordingIds.includes(recording.recordingId),
              ).length,
            )}
          </strong>
        </div>
      </section>

      <section className={styles.protocolPanel}>
        <div>
          <span className={styles.panelKicker}>REVIEW CONTRACT</span>
          <strong>Model proposals are discovery candidates</strong>
          <p>
            Prior <code>sideSwitches</code> markers and unmarked heuristic gaps
            are context, not exhaustive positive/negative labels. Every model
            variant has an independent timeline; decide from the video whether a
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
              {proposalBundle.layers
                .map(
                  (layer) =>
                    `${layer.attachedEvents} ${MODEL_SHORT_LABELS[layer.modelId]}`,
                )
                .join(" · ")}{" "}
              gaps
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
              {MODEL_SHORT_LABELS[error.modelId]} could not be loaded:{" "}
              {error.message}
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
            onChange={(event) => {
              setRecordingId(event.target.value);
              setEventFilter("all");
              setProposalFilter("all");
            }}
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
                  {recording.environment} · {recording.recordingId} ·{" "}
                  {recording.continuousVideoReviewed ||
                  reviewedRecordingIds.includes(recording.recordingId)
                    ? "covered"
                    : "needs full review"}
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
                  <span>
                    <kbd>M</kbd> add full-video marker
                  </span>
                </div>
              </div>

              {selectedRecording && (
                <FullVideoCoverageTimeline
                  recording={selectedRecording}
                  events={selectedRecordingEvents}
                  decisions={decisions}
                  markers={selectedRecordingMarkers}
                  reviewedComplete={reviewedRecordingIds.includes(
                    selectedRecording.recordingId,
                  )}
                  duration={duration}
                  currentTime={currentTime}
                  markerSaveStatus={markerSaveStatus}
                  markerSavedAt={markersLastSavedAt}
                  markerSaveError={markerSaveError}
                  onSelect={selectEvent}
                  onSeek={seek}
                  onAddMarker={addMarkerAtCurrentTime}
                  onRemoveMarker={removeMarker}
                  onSetReviewedComplete={setRecordingReviewedComplete}
                />
              )}
              <VariantTimelines
                events={selectedRecordingEvents}
                proposalBundle={proposalBundle}
                duration={duration}
                selectedEventId={selectedEvent.eventId}
                currentTime={currentTime}
                onSelect={selectEvent}
                onSeek={seek}
              />
              {selectedRecording && (
                <ProductionContextTimelines
                  recording={selectedRecording}
                  duration={duration}
                  currentTime={currentTime}
                  onSeek={seek}
                />
              )}
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
                      ? "Compare the video across all model variant layers. A selected proposal or high score is evidence, not a decision."
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
          V5, V5 + state, and V6 predictions are attached without changing the
          report identity, so existing review decisions remain intact.
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
  initialMarkers,
  initialReviewedRecordingIds,
  initialMarkersSavedAt,
  markerLoadError,
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
      initialMarkers={initialMarkers}
      initialReviewedRecordingIds={initialReviewedRecordingIds}
      initialMarkersSavedAt={initialMarkersSavedAt}
      markerLoadError={markerLoadError}
    />
  );
}
