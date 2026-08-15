import styles from "./rally-timeline.module.css";
import { formatTime, timelinePercent, timelineTicks } from "@/lib/edit-list";

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
};

export type TimelineMarker = {
  id: string;
  time: number;
  title: string;
};

type RallyTimelineProps = {
  duration: number;
  currentTime: number;
  tracks: TimelineTrack[];
  markers?: TimelineMarker[];
  selectedTrackId?: string;
  selectedIntervalId?: string;
  ariaLabel?: string;
  onSeek?: (time: number, trackId: string, intervalId?: string) => void;
  onTrackSelect?: (trackId: string) => void;
};

export function RallyTimeline({
  duration,
  currentTime,
  tracks,
  markers = [],
  selectedTrackId,
  selectedIntervalId,
  ariaLabel = "Rally timeline",
  onSeek,
  onTrackSelect,
}: RallyTimelineProps) {
  const ticks = timelineTicks(duration);
  return (
    <div className={styles.timeline} aria-label={ariaLabel}>
      <div className={styles.ticks} aria-hidden="true">
        <span />
        <div className={styles.tickRail}>
          {ticks.map((tick) => <span key={tick}>{formatTime(tick)}</span>)}
        </div>
        <span className={styles.summaryHeading}>Output summary</span>
      </div>
      {tracks.map((track) => (
        <div
          className={`${styles.track} ${track.active ? styles.trackActive : ""}`}
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
          <div className={styles.rail}>
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
                  (interval.selectionId ?? interval.id) === selectedIntervalId
                    ? "true"
                    : undefined
                }
                data-padding-origin={interval.paddingOrigin}
                style={{
                  left: `${timelinePercent(interval.start, duration)}%`,
                  width: `${timelinePercent(interval.end - interval.start, duration)}%`,
                }}
                onClick={() => onSeek?.(
                  interval.start,
                  track.id,
                  interval.selectionId === null
                    ? undefined
                    : interval.selectionId ?? interval.id,
                )}
                title={interval.title ?? `${interval.id}: ${formatTime(interval.start)}–${formatTime(interval.end)}`}
                aria-label={`Seek to ${interval.id} at ${formatTime(interval.start)}`}
              />
            ))}
            {track.intervals.length === 0 && <span className={styles.empty}>No intervals</span>}
            {markers.map((marker) => (
              <button
                type="button"
                key={marker.id}
                className={styles.marker}
                style={{ left: `${timelinePercent(marker.time, duration)}%` }}
                onClick={() => onSeek?.(marker.time, track.id)}
                title={marker.title}
                aria-label={marker.title}
              />
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
                    style={{
                      left: `${timelinePercent(interval.start, duration)}%`,
                      width: `${timelinePercent(interval.end - interval.start, duration)}%`,
                    }}
                    onClick={() => onSeek?.(interval.start, track.id)}
                    title={interval.title ?? `Exported: ${formatTime(interval.start)}–${formatTime(interval.end)}`}
                    aria-label={`Seek to exported segment at ${formatTime(interval.start)}`}
                  />
                ))}
                {track.joinedGapIntervals?.map((interval) => (
                  <button
                    type="button"
                    key={interval.id}
                    className={`${styles.exportInterval} ${styles.joinedGapInterval}`}
                    style={{
                      left: `${timelinePercent(interval.start, duration)}%`,
                      width: `${timelinePercent(interval.end - interval.start, duration)}%`,
                    }}
                    onClick={() => onSeek?.(interval.start, track.id)}
                    title={interval.title ?? `Joined short gap: ${formatTime(interval.start)}–${formatTime(interval.end)}`}
                    aria-label={`Seek to joined short gap at ${formatTime(interval.start)}`}
                  />
                ))}
                {track.missingHumanIntervals?.map((interval) => (
                  <button
                    type="button"
                    key={interval.id}
                    className={`${styles.exportInterval} ${styles.missingHumanInterval}`}
                    style={{
                      left: `${timelinePercent(interval.start, duration)}%`,
                      width: `${timelinePercent(interval.end - interval.start, duration)}%`,
                    }}
                    onClick={() => onSeek?.(interval.start, track.id)}
                    title={interval.title ?? `Missed human rally: ${formatTime(interval.start)}–${formatTime(interval.end)}`}
                    aria-label={`Seek to missed human rally at ${formatTime(interval.start)}`}
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
                    {track.summary.exportDelta && <> · {track.summary.exportDelta}</>}
                  </span>
                </small>
                {track.summary.coreMetrics && (
                  <small>
                    <b>{track.summary.metricsLabel ?? "Core P/R/F1"}</b>
                    <span>{track.summary.coreMetrics}</span>
                  </small>
                )}
                {track.summary.paddedMetrics && (
                  <small><b>Padded P/R/F1</b><span>{track.summary.paddedMetrics}</span></small>
                )}
                {track.summary.hybridF1 && (
                  <small><b>Padded P/Core R F1</b><span>{track.summary.hybridF1}</span></small>
                )}
              </>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
