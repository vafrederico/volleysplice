import { type PointerEvent as ReactPointerEvent, useRef } from "react";

import { formatTime, timelinePercent, timelineTicks } from "@/lib/edit-list";
import styles from "./rally-timeline.module.css";

export type TimelineInterval = {
  id: string;
  start: number;
  end: number;
  confidence?: number;
  title?: string;
  selectionId?: string | null;
  paddingOrigin?: "before" | "after" | "both";
  tone?:
    | "model"
    | "model-no-beach"
    | "model-match"
    | "model-added"
    | "model-missed"
    | "model-disagreement"
    | "gold-padding"
    | "sol-padding"
    | "heuristic"
    | "sol"
    | "gold"
    | "ignored"
    | "negative";
};

export type TimelineTrack = {
  id: string;
  label: string;
  title?: string;
  detail?: string;
  summary?: {
    exportTime: string;
    exportDelta?: string;
    metricsLabel?: string;
    coreMetrics?: string;
    paddedMetrics?: string;
    hybridF1?: string;
  };
  active?: boolean;
  intervals: TimelineInterval[];
  exportIntervals?: TimelineInterval[];
  joinedGapIntervals?: TimelineInterval[];
  missingHumanIntervals?: TimelineInterval[];
  modelOnlyIntervals?: TimelineInterval[];
  suppressedIntervals?: TimelineInterval[];
};

export type TimelineMarker = {
  id: string;
  time: number;
  title: string;
  label?: string;
  trackId?: string;
  disagrees?: boolean;
  tone?:
    | "side-switch"
    | "model-side-switch"
    | "serve-near"
    | "serve-far"
    | "serve-review";
};

type RallyTimelineProps = {
  duration: number;
  currentTime: number;
  tracks: TimelineTrack[];
  markers?: TimelineMarker[];
  markerTrackId?: string;
  selectedTrackId?: string;
  selectedIntervalId?: string;
  selectedIntervalIds?: readonly string[];
  selectedMarkerId?: string;
  ariaLabel?: string;
  onSeek?: (
    time: number,
    trackId: string,
    intervalId?: string,
    interaction?: { shiftKey: boolean },
  ) => void;
  onMarkerSeek?: (time: number, trackId: string, markerId: string) => void;
  onTrackSelect?: (trackId: string) => void;
};

type TimelineSeekDrag = {
  pointerId: number;
  left: number;
  width: number;
  startX: number;
  startY: number;
  moved: boolean;
  shiftKey: boolean;
  tapTarget:
    | { kind: "marker"; id: string; time: number }
    | { kind: "seek"; intervalId?: string; time: number }
    | null;
  trackId: string;
};

export function RallyTimeline({
  duration,
  currentTime,
  tracks,
  markers = [],
  markerTrackId,
  selectedTrackId,
  selectedIntervalId,
  selectedIntervalIds = [],
  selectedMarkerId,
  ariaLabel = "Rally timeline",
  onSeek,
  onMarkerSeek,
  onTrackSelect,
}: RallyTimelineProps) {
  const ticks = timelineTicks(duration);
  const hasSummaries = tracks.some((track) => track.summary !== undefined);
  const seekDragRef = useRef<TimelineSeekDrag | null>(null);
  const suppressClickUntilRef = useRef(0);
  const markersForTrack = (trackId: string) =>
    markers.filter((marker) =>
      marker.trackId !== undefined
        ? marker.trackId === trackId
        : markerTrackId === undefined || markerTrackId === trackId,
    );

  function seekFromPointer(clientX: number, drag: TimelineSeekDrag) {
    const ratio = Math.max(
      0,
      Math.min(1, (clientX - drag.left) / Math.max(1, drag.width)),
    );
    onSeek?.(ratio * duration, drag.trackId);
  }

  function beginSeek(
    event: ReactPointerEvent<HTMLDivElement>,
    trackId: string,
  ) {
    if (!onSeek || (event.pointerType === "mouse" && event.button !== 0))
      return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const target = event.target instanceof Element ? event.target : null;
    const action = target?.closest<HTMLButtonElement>(
      "[data-timeline-seek-time]",
    );
    const actionTime = Number(action?.dataset.timelineSeekTime);
    const markerId = action?.dataset.timelineMarkerId;
    const intervalId = action?.dataset.timelineIntervalId;
    seekDragRef.current = {
      pointerId: event.pointerId,
      left: bounds.left,
      width: bounds.width,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
      shiftKey: event.shiftKey,
      tapTarget:
        action && Number.isFinite(actionTime)
          ? markerId
            ? { kind: "marker", id: markerId, time: actionTime }
            : {
                kind: "seek",
                time: actionTime,
                ...(intervalId ? { intervalId } : {}),
              }
          : null,
      trackId,
    };
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Drag seeking still works while the pointer remains over the rail.
    }
  }

  function moveSeek(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = seekDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (!drag.moved) {
      const horizontalDistance = Math.abs(event.clientX - drag.startX);
      const verticalDistance = Math.abs(event.clientY - drag.startY);
      if (horizontalDistance < 4 || verticalDistance > horizontalDistance)
        return;
      drag.moved = true;
    }
    event.preventDefault();
    seekFromPointer(event.clientX, drag);
  }

  function endSeek(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = seekDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (drag.moved) {
      seekFromPointer(event.clientX, drag);
      suppressClickUntilRef.current = Date.now() + 800;
    } else if (drag.tapTarget) {
      suppressClickUntilRef.current = Date.now() + 800;
      if (drag.tapTarget.kind === "marker") {
        if (onMarkerSeek) {
          onMarkerSeek(drag.tapTarget.time, drag.trackId, drag.tapTarget.id);
        } else {
          onSeek?.(drag.tapTarget.time, drag.trackId);
        }
      } else {
        onSeek?.(drag.tapTarget.time, drag.trackId, drag.tapTarget.intervalId, {
          shiftKey: drag.shiftKey,
        });
      }
    } else {
      seekFromPointer(event.clientX, drag);
    }
    seekDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function cancelSeek(event: ReactPointerEvent<HTMLDivElement>) {
    if (seekDragRef.current?.pointerId === event.pointerId) {
      seekDragRef.current = null;
    }
  }

  return (
    <div
      className={`${styles.timeline} ${hasSummaries ? "" : styles.timelineCompact}`}
      role="group"
      aria-label={ariaLabel}
    >
      <div className={styles.ticks} aria-hidden="true">
        <span />
        <div className={styles.tickRail}>
          {ticks.map((tick) => (
            <span key={tick}>{formatTime(tick)}</span>
          ))}
        </div>
        <span className={styles.summaryHeading}>Output summary</span>
      </div>
      {tracks.map((track) => (
        <div
          className={`${styles.track} ${track.active ? styles.trackActive : ""} ${markersForTrack(track.id).length > 0 ? styles.trackWithMarkers : ""}`}
          key={track.id}
          aria-current={track.id === selectedTrackId ? "true" : undefined}
        >
          {onTrackSelect ? (
            <button
              type="button"
              className={styles.trackLabel}
              title={track.title}
              onClick={() => onTrackSelect(track.id)}
            >
              <strong>{track.label}</strong>
              {track.detail && <small>{track.detail}</small>}
            </button>
          ) : (
            <div className={styles.trackLabel} title={track.title}>
              <strong>{track.label}</strong>
              {track.detail && <small>{track.detail}</small>}
            </div>
          )}
          <div
            className={styles.rail}
            onPointerDown={(event) => beginSeek(event, track.id)}
            onPointerMove={moveSeek}
            onPointerUp={endSeek}
            onPointerCancel={cancelSeek}
            onLostPointerCapture={cancelSeek}
            onClickCapture={(event) => {
              if (Date.now() > suppressClickUntilRef.current) return;
              event.preventDefault();
              event.stopPropagation();
              suppressClickUntilRef.current = 0;
            }}
          >
            {track.intervals.map((interval) => (
              <button
                type="button"
                key={interval.id}
                className={styles.interval}
                data-tone={interval.tone}
                data-uncertain={
                  interval.confidence !== undefined && interval.confidence < 0.7
                    ? "true"
                    : undefined
                }
                data-selected={
                  track.id === selectedTrackId &&
                  interval.selectionId !== null &&
                  ((interval.selectionId ?? interval.id) ===
                    selectedIntervalId ||
                    selectedIntervalIds.includes(
                      interval.selectionId ?? interval.id,
                    ))
                    ? "true"
                    : undefined
                }
                data-padding-origin={interval.paddingOrigin}
                data-timeline-seek-time={interval.start}
                data-timeline-interval-id={
                  interval.selectionId === null
                    ? undefined
                    : (interval.selectionId ?? interval.id)
                }
                style={{
                  left: `${timelinePercent(interval.start, duration)}%`,
                  width: `${timelinePercent(interval.end - interval.start, duration)}%`,
                }}
                onClick={(event) =>
                  onSeek?.(
                    interval.start,
                    track.id,
                    interval.selectionId === null
                      ? undefined
                      : (interval.selectionId ?? interval.id),
                    { shiftKey: event.shiftKey },
                  )
                }
                title={
                  interval.title ??
                  `${interval.id}: ${formatTime(interval.start)}–${formatTime(interval.end)}`
                }
                aria-label={`Seek to ${interval.id} at ${formatTime(interval.start)}`}
              />
            ))}
            {track.intervals.length === 0 && (
              <span className={styles.empty}>No intervals</span>
            )}
            {track.modelOnlyIntervals && (
              <div
                className={styles.modelOnlyRail}
                role="group"
                aria-label={`${track.label} model-only rallies`}
              >
                {track.modelOnlyIntervals.map((interval) => (
                  <button
                    type="button"
                    key={interval.id}
                    className={styles.modelOnlyInterval}
                    data-timeline-seek-time={interval.start}
                    style={{
                      left: `${timelinePercent(interval.start, duration)}%`,
                      width: `${timelinePercent(interval.end - interval.start, duration)}%`,
                    }}
                    onClick={() => onSeek?.(interval.start, track.id)}
                    title={interval.title}
                    aria-label={`Seek to model-only rally at ${formatTime(interval.start)}`}
                  />
                ))}
              </div>
            )}
            {markersForTrack(track.id).map((marker) => (
              <button
                type="button"
                key={marker.id}
                className={styles.marker}
                data-tone={marker.tone}
                data-disagrees={marker.disagrees || undefined}
                data-selected={marker.id === selectedMarkerId || undefined}
                data-timeline-seek-time={marker.time}
                data-timeline-marker-id={marker.id}
                style={{ left: `${timelinePercent(marker.time, duration)}%` }}
                onClick={() =>
                  onMarkerSeek
                    ? onMarkerSeek(marker.time, track.id, marker.id)
                    : onSeek?.(marker.time, track.id)
                }
                title={marker.title}
                aria-label={marker.title}
              >
                {marker.label && <span aria-hidden="true">{marker.label}</span>}
              </button>
            ))}
            {track.exportIntervals && (
              <div
                className={styles.exportRail}
                role="group"
                aria-label={`${track.label} padded export`}
              >
                {track.exportIntervals.map((interval) => (
                  <button
                    type="button"
                    key={interval.id}
                    className={styles.exportInterval}
                    data-timeline-seek-time={interval.start}
                    style={{
                      left: `${timelinePercent(interval.start, duration)}%`,
                      width: `${timelinePercent(interval.end - interval.start, duration)}%`,
                    }}
                    onClick={() => onSeek?.(interval.start, track.id)}
                    title={
                      interval.title ??
                      `Exported: ${formatTime(interval.start)}–${formatTime(interval.end)}`
                    }
                    aria-label={`Seek to exported segment at ${formatTime(interval.start)}`}
                  />
                ))}
                {track.joinedGapIntervals?.map((interval) => (
                  <button
                    type="button"
                    key={interval.id}
                    className={`${styles.exportInterval} ${styles.joinedGapInterval}`}
                    data-timeline-seek-time={interval.start}
                    style={{
                      left: `${timelinePercent(interval.start, duration)}%`,
                      width: `${timelinePercent(interval.end - interval.start, duration)}%`,
                    }}
                    onClick={() => onSeek?.(interval.start, track.id)}
                    title={
                      interval.title ??
                      `Joined short gap: ${formatTime(interval.start)}–${formatTime(interval.end)}`
                    }
                    aria-label={`Seek to joined short gap at ${formatTime(interval.start)}`}
                  />
                ))}
                {track.missingHumanIntervals?.map((interval) => (
                  <button
                    type="button"
                    key={interval.id}
                    className={`${styles.exportInterval} ${styles.missingHumanInterval}`}
                    data-timeline-seek-time={interval.start}
                    style={{
                      left: `${timelinePercent(interval.start, duration)}%`,
                      width: `${timelinePercent(interval.end - interval.start, duration)}%`,
                    }}
                    onClick={() => onSeek?.(interval.start, track.id)}
                    title={
                      interval.title ??
                      `Missed human rally: ${formatTime(interval.start)}–${formatTime(interval.end)}`
                    }
                    aria-label={`Seek to missed human rally at ${formatTime(interval.start)}`}
                  />
                ))}
                {track.suppressedIntervals?.map((interval) => (
                  <button
                    type="button"
                    key={interval.id}
                    className={`${styles.exportInterval} ${styles.suppressedInterval}`}
                    data-timeline-seek-time={interval.start}
                    style={{
                      left: `${timelinePercent(interval.start, duration)}%`,
                      width: `${timelinePercent(interval.end - interval.start, duration)}%`,
                    }}
                    onClick={() => onSeek?.(interval.start, track.id)}
                    title={
                      interval.title ??
                      `Suppressed: ${formatTime(interval.start)}–${formatTime(interval.end)}`
                    }
                    aria-label={`Seek to suppressed model range at ${formatTime(interval.start)}`}
                  />
                ))}
              </div>
            )}
            <span
              className={styles.playhead}
              style={{ left: `${timelinePercent(currentTime, duration)}%` }}
              aria-hidden="true"
            />
          </div>
          <div className={styles.trackSummary}>
            {track.summary && (
              <>
                <small>
                  <b>Export</b>
                  <span>
                    {track.summary.exportTime}
                    {track.summary.exportDelta && (
                      <> · {track.summary.exportDelta}</>
                    )}
                  </span>
                </small>
                {track.summary.coreMetrics && (
                  <small>
                    <b>{track.summary.metricsLabel ?? "Core P/R/F1"}</b>
                    <span>{track.summary.coreMetrics}</span>
                  </small>
                )}
                {track.summary.paddedMetrics && (
                  <small>
                    <b>Padded P/R/F1</b>
                    <span>{track.summary.paddedMetrics}</span>
                  </small>
                )}
                {track.summary.hybridF1 && (
                  <small>
                    <b>Padded P/Core R F1</b>
                    <span>{track.summary.hybridF1}</span>
                  </small>
                )}
              </>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
