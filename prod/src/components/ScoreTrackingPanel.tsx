import { useEffect, useMemo, useRef, useState } from "react";

import type { InferenceProgressStep } from "@/lib/inference-progress";
import {
  addServeMarker,
  addSideSwitchMarker,
  deriveScoreAt,
  nextReviewServeMarker,
  orderedServeMarkers,
  orderedSideSwitchMarkers,
  removeServeMarker,
  removeSideSwitchMarker,
  type ScoreIgnoredInterval,
  type ScoreTracking,
  type ServingSide,
  scoreTrackingOutsideExcludedRallies,
  scoreTrackingOutsideIgnoredIntervals,
  setPreviousPointIgnored,
  setServeMarkerSide,
} from "@/lib/score-tracking";
import { scrollElementIntoContainer } from "@/lib/scroll-container";

import styles from "./CutEditor.module.css";

type ScoreTrackingPanelProps = {
  tracking: ScoreTracking;
  ignoredIntervals: readonly ScoreIgnoredInterval[];
  excludedRallyIds: ReadonlySet<string>;
  playbackTime: number;
  scoreBoundaryTime: number;
  selectedServeMarkerId: string;
  inferenceStatus: "idle" | "running" | "done" | "error";
  inferenceMessage: string | null;
  inferenceSteps: readonly InferenceProgressStep[];
  canRunInference: boolean;
  onChange: (tracking: ScoreTracking) => void;
  onSelectServeMarker: (id: string, timestamp: number) => void;
  onSelectSideSwitchMarker: (id: string, timestamp: number) => void;
  onRunInference: () => void;
};

function preciseTime(seconds: number): string {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = safe % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`
    : `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function sideLabel(side: ServingSide): string {
  return side === "review" ? "Review" : side === "near" ? "Near" : "Far";
}

export function ScoreTrackingPanel({
  tracking,
  ignoredIntervals,
  excludedRallyIds,
  playbackTime,
  scoreBoundaryTime,
  selectedServeMarkerId,
  inferenceStatus,
  inferenceMessage,
  inferenceSteps: _inferenceSteps,
  canRunInference,
  onChange,
  onSelectServeMarker,
  onSelectSideSwitchMarker,
  onRunInference,
}: ScoreTrackingPanelProps) {
  const [manualSide, setManualSide] = useState<ServingSide>("near");
  const markerRowsRef = useRef(new Map<string, HTMLElement>());
  const markerListRef = useRef<HTMLDivElement>(null);
  const activeTracking = useMemo(
    () =>
      scoreTrackingOutsideExcludedRallies(
        scoreTrackingOutsideIgnoredIntervals(tracking, ignoredIntervals),
        excludedRallyIds,
      ),
    [excludedRallyIds, ignoredIntervals, tracking],
  );
  const serves = useMemo(
    () => orderedServeMarkers(activeTracking),
    [activeTracking],
  );
  const switches = useMemo(
    () => orderedSideSwitchMarkers(activeTracking),
    [activeTracking],
  );
  const score = useMemo(
    () => deriveScoreAt(activeTracking, scoreBoundaryTime),
    [activeTracking, scoreBoundaryTime],
  );
  const finalScore = useMemo(
    () => deriveScoreAt(activeTracking),
    [activeTracking],
  );
  const pointTimeline = useMemo(
    () =>
      score.points.flatMap((point, rallyIndex) =>
        point.status === "counted" && point.winnerTeamId
          ? [
              {
                id: point.serveMarkerId,
                timestamp: point.timestamp,
                rallyNumber: rallyIndex + 1,
                winnerTeamId: point.winnerTeamId,
                teamPointNumber:
                  point.winnerTeamId === "team-1"
                    ? point.team1ScoreAfter
                    : point.team2ScoreAfter,
              },
            ]
          : [],
      ),
    [score.points],
  );
  const ignoredServeCount = tracking.serveMarkers.length - serves.length;
  const reviewMarkers = serves.filter(
    (marker) => marker.side === "review",
  );
  const selected =
    serves.find((marker) => marker.id === selectedServeMarkerId) ??
    null;
  const selectedIndex = selected
    ? serves.findIndex((marker) => marker.id === selected.id)
    : -1;
  const reviewSummary =
    reviewMarkers.length === 0
      ? "No serve markers need review"
      : reviewMarkers.length === 1
        ? "1 serve marker needs review"
        : `${reviewMarkers.length} serve markers need review`;
  const displayedInferenceMessage = inferenceMessage
    ? /\d+ serve (?:verdicts?|markers?) need review/i.test(inferenceMessage)
      ? inferenceMessage.replace(
          /\d+ serve (?:verdicts?|markers?) need review/i,
          reviewSummary.toLowerCase(),
        )
      : inferenceStatus === "done"
        ? `${inferenceMessage.replace(/[.\s]+$/, "")} · ${reviewSummary.toLowerCase()}.`
        : inferenceMessage
    : "Check any marker marked for review, then add missing serves or team side switches below.";
  const servingName =
    score.servingTeamId === "team-1"
      ? tracking.team1Name
      : score.servingTeamId === "team-2"
        ? tracking.team2Name
        : "Review needed";

  useEffect(() => {
    if (!selectedServeMarkerId) return;
    const frame = requestAnimationFrame(() => {
      const markerList = markerListRef.current;
      const markerRow = markerRowsRef.current.get(selectedServeMarkerId);
      if (markerList && markerRow) {
        scrollElementIntoContainer(markerList, markerRow);
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [selectedServeMarkerId, serves, switches]);

  return (
    <aside
      className={styles.scorePanel}
      aria-label="Score tracking controls"
      data-tour="editor-score-panel"
    >
      <header className={styles.scorePanelHeader}>
        <div>
          <span>OPTIONAL SCOREBOARD</span>
          <strong>Score and serve markers</strong>
        </div>
        <small>
          {serves.length} serve markers · {switches.length} side switches
          {reviewMarkers.length > 0
            ? ` · ${reviewMarkers.length} need review`
            : ""}
          {ignoredServeCount > 0 ? ` · ${ignoredServeCount} ignored` : ""}
        </small>
      </header>

      <section className={styles.scoreGuide} aria-label="How score tracking works">
        <div>
          <span>HOW THE SCORE IS BUILT</span>
          <strong>Serve markers connect each rally to a team.</strong>
        </div>
        <p>
          The first serve sets the starting server. Every serve after that awards
          the previous rally to the team now serving.
        </p>
        <p>
          Add a side-switch marker whenever teams change court sides so Near and Far
          still point to the correct team. Add any missed serves—including one after
          the final rally if its point has not been counted.
        </p>
      </section>

      <div className={styles.scoreboard} aria-live="polite">
        <label>
          <span>Team 1 · starts near</span>
          <input
            aria-label="Team 1 name"
            value={tracking.team1Name}
            onChange={(event) =>
              onChange({
                ...tracking,
                team1Name: event.currentTarget.value || "Team 1",
              })
            }
          />
          <output>{score.team1Score}</output>
        </label>
        <label>
          <span>Team 2 · starts far</span>
          <input
            aria-label="Team 2 name"
            value={tracking.team2Name}
            onChange={(event) =>
              onChange({
                ...tracking,
                team2Name: event.currentTarget.value || "Team 2",
              })
            }
          />
          <output>{score.team2Score}</output>
        </label>
        <p>
          Serving now: <strong>{servingName}</strong>
          {score.servingSide ? ` · ${sideLabel(score.servingSide)} side` : ""}
        </p>
        {(finalScore.reviewPointCount > 0 ||
          finalScore.ignoredPointCount > 0) && (
          <small>
            {finalScore.reviewPointCount} need review ·{" "}
            {finalScore.ignoredPointCount} ignored or replayed
          </small>
        )}
      </div>

      <div className={styles.scoreInference} data-state={inferenceStatus}>
        <div>
          <strong>
            {inferenceStatus === "running"
              ? "Finding serve markers…"
              : inferenceStatus === "error"
                ? "Score markers need attention"
                : serves.length > 0
                  ? "Serve markers ready"
                  : "No serve markers yet"}
          </strong>
          <small>{displayedInferenceMessage}</small>
        </div>
        {(reviewMarkers.length > 0 ||
          canRunInference ||
          inferenceStatus === "running") && (
          <div className={styles.scoreInferenceActions}>
            {reviewMarkers.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  const next = nextReviewServeMarker(
                    serves,
                    selectedServeMarkerId,
                    playbackTime,
                  );
                  if (next) onSelectServeMarker(next.id, next.timestamp);
                }}
              >
                Check next · {reviewMarkers.length}
              </button>
            )}
            {(canRunInference || inferenceStatus === "running") && (
              <button
                type="button"
                disabled={!canRunInference || inferenceStatus === "running"}
                onClick={onRunInference}
              >
                Find score markers
              </button>
            )}
          </div>
        )}
      </div>

      {inferenceStatus === "running" && (
        <div className={styles.scoreInferenceProgress}>
          Finding the score markers in your video…
        </div>
      )}

      <section className={styles.selectedServeEditor}>
        <div>
          <span>SELECTED SERVE</span>
          <strong>
            {selected
              ? `${preciseTime(selected.timestamp)} · ${sideLabel(selected.side)}`
              : "Select a ball marker on the timeline"}
          </strong>
          {selected?.modelSide && selected.side !== selected.modelSide && (
            <small>
              Changed from {sideLabel(selected.modelSide)}
            </small>
          )}
        </div>
        <fieldset disabled={!selected}>
          <legend>Which side is serving?</legend>
          <div className={styles.sideVerdictButtons}>
            {(["near", "far"] as const).map((side) => (
              <button
                type="button"
                key={side}
                aria-pressed={selected?.side === side}
                data-active={selected?.side === side || undefined}
                onClick={() =>
                  selected &&
                  onChange(setServeMarkerSide(tracking, selected.id, side))
                }
              >
                {sideLabel(side)}
              </button>
            ))}
          </div>
        </fieldset>
        <small className={styles.sideChoiceHelp}>
          Near means closest to the camera; Far means across the court. For every
          serve after the first, this choice decides which team won the previous rally.
        </small>
        <label
          className={styles.replayToggle}
          data-disabled={selectedIndex <= 0 || undefined}
        >
          <input
            type="checkbox"
            disabled={!selected || selectedIndex <= 0}
            checked={selected?.ignorePreviousPoint ?? false}
            onChange={(event) =>
              selected &&
              onChange(
                setPreviousPointIgnored(
                  tracking,
                  selected.id,
                  event.currentTarget.checked,
                ),
              )
            }
          />
          <span>
            <strong>Ignore/replay previous point</strong>
            <small>Do not change the score at this serve.</small>
          </span>
        </label>
      </section>

      <section className={styles.markerAddTools}>
        <p className={styles.markerHelp}>
          <strong>Missing a serve?</strong> Move the video to that serve, choose the
          server’s court side, then add the marker.
        </p>
        <div>
          <label>
            <span>Missing serve at {preciseTime(playbackTime)}</span>
            <select
              aria-label="Side for missing serve marker"
              value={manualSide}
              onChange={(event) =>
                setManualSide(event.currentTarget.value as ServingSide)
              }
            >
              <option value="near">Near</option>
              <option value="far">Far</option>
              <option value="review">Review</option>
            </select>
          </label>
          <button
            type="button"
            onClick={() => {
              const next = addServeMarker(tracking, playbackTime, manualSide);
              onChange(next);
              const added = next.serveMarkers.find(
                (marker) =>
                  !tracking.serveMarkers.some((old) => old.id === marker.id),
              );
              if (added) onSelectServeMarker(added.id, added.timestamp);
            }}
          >
            + Add serve
          </button>
        </div>
        <p className={styles.markerHelp}>
          <strong>Teams switched ends?</strong> Move to the switch and add it here.
          Without this marker, later points can be assigned to the wrong team.
        </p>
        <button
          type="button"
          className={styles.addSwitchButton}
          onClick={() => onChange(addSideSwitchMarker(tracking, playbackTime))}
        >
          ⇄ Add team side switch · {preciseTime(playbackTime)}
        </button>
      </section>

      <section
        className={styles.pointTimeline}
        aria-label={`${pointTimeline.length} awarded points. ${tracking.team1Name}: ${score.team1Score}. ${tracking.team2Name}: ${score.team2Score}.`}
      >
        <div className={styles.pointTimelineHeader}>
          <span>POINT TIMELINE</span>
          <small>{pointTimeline.length} awarded</small>
        </div>
        {pointTimeline.length > 0 ? (
          <div className={styles.pointTimelineScroller}>
            <div
              className={`${styles.pointTimelineRow} ${styles.pointTimelineRallyRow}`}
              style={{
                gridTemplateColumns: `78px repeat(${pointTimeline.length}, 22px)`,
              }}
            >
              <strong>RALLY</strong>
              {pointTimeline.map((point) => (
                <span
                  className={styles.pointTimelineUnit}
                  key={`rally-${point.id}`}
                  title={`Rally ${point.rallyNumber}`}
                >
                  <b>{point.rallyNumber}</b>
                </span>
              ))}
            </div>
            {(
              [
                ["team-1", tracking.team1Name],
                ["team-2", tracking.team2Name],
              ] as const
            ).map(([teamId, teamName]) => (
              <div
                className={styles.pointTimelineRow}
                data-team={teamId}
                key={teamId}
                style={{
                  gridTemplateColumns: `78px repeat(${pointTimeline.length}, 22px)`,
                }}
              >
                <strong title={teamName}>{teamName}</strong>
                {pointTimeline.map((point) => (
                  <span
                    className={styles.pointTimelineUnit}
                    data-won={point.winnerTeamId === teamId || undefined}
                    key={`${teamId}-${point.id}`}
                    title={`Rally ${point.rallyNumber}${point.winnerTeamId === teamId ? ` · ${teamName} point ${point.teamPointNumber}` : ""}`}
                  >
                    {point.winnerTeamId === teamId && (
                      <button
                        type="button"
                        aria-label={`Seek video to when ${teamName} reached ${point.teamPointNumber}, at ${preciseTime(point.timestamp)}`}
                        data-selected={point.id === selectedServeMarkerId || undefined}
                        onClick={() =>
                          onSelectServeMarker(point.id, point.timestamp)
                        }
                      >
                        {point.teamPointNumber}
                      </button>
                    )}
                  </span>
                ))}
              </div>
            ))}
          </div>
        ) : (
          <small className={styles.pointTimelineEmpty}>
            No points awarded yet
          </small>
        )}
      </section>

      <section
        className={styles.scoreMarkerList}
        aria-label="Score marker list"
      >
        <div>
          <span>MARKERS</span>
          <small>
            {reviewMarkers.length > 0
              ? `${reviewMarkers.length} need review · `
              : ""}
            Markers can be removed
          </small>
        </div>
        <div ref={markerListRef}>
          {serves.map((marker, index) => (
            <article
              key={marker.id}
              ref={(node) => {
                if (node) markerRowsRef.current.set(marker.id, node);
                else markerRowsRef.current.delete(marker.id);
              }}
              data-selected={marker.id === selected?.id || undefined}
              data-review={marker.side === "review" || undefined}
            >
              <button
                type="button"
                aria-label={`Serve marker at ${preciseTime(marker.timestamp)}${marker.side === "review" ? ", needs review" : ""}`}
                onClick={() => onSelectServeMarker(marker.id, marker.timestamp)}
              >
                <span aria-hidden="true">🏐</span>
                <strong>{preciseTime(marker.timestamp)}</strong>
                <small>
                  {marker.side === "review"
                    ? "Needs review"
                    : index === 0
                      ? "First serve"
                      : sideLabel(marker.side)}{" "}
                  ·{" "}
                  {marker.origin === "model" ? "Found automatically" : "Added by you"}
                </small>
              </button>
              <button
                type="button"
                className={styles.removeMarker}
                aria-label={`Remove ${marker.origin === "model" ? "automatically found" : "added"} serve at ${preciseTime(marker.timestamp)}`}
                onClick={() => onChange(removeServeMarker(tracking, marker.id))}
              >
                Remove
              </button>
            </article>
          ))}
          {switches.map((marker) => (
            <article
              key={marker.id}
              ref={(node) => {
                if (node) markerRowsRef.current.set(marker.id, node);
                else markerRowsRef.current.delete(marker.id);
              }}
              data-selected={
                marker.id === selectedServeMarkerId || undefined
              }
            >
              <button
                type="button"
                onClick={() =>
                  onSelectSideSwitchMarker(marker.id, marker.timestamp)
                }
              >
                <span aria-hidden="true">⇄</span>
                <strong>{preciseTime(marker.timestamp)}</strong>
                <small>
                  Team side switch · {marker.origin === "model" ? "found automatically" : "added by you"}
                </small>
              </button>
              <button
                type="button"
                className={styles.removeMarker}
                aria-label={`Remove ${marker.origin === "model" ? "automatically found" : "added"} side switch at ${preciseTime(marker.timestamp)}`}
                onClick={() =>
                  onChange(removeSideSwitchMarker(tracking, marker.id))
                }
              >
                Remove
              </button>
            </article>
          ))}
          {serves.length === 0 && switches.length === 0 && (
            <p>No score markers have been added.</p>
          )}
        </div>
      </section>
    </aside>
  );
}
