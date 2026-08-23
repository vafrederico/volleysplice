import { useMemo, useState } from "react";

import {
  addServeMarker,
  addSideSwitchMarker,
  deriveScoreAt,
  orderedServeMarkers,
  orderedSideSwitchMarkers,
  removeServeMarker,
  removeSideSwitchMarker,
  scoreTrackingOutsideExcludedRallies,
  scoreTrackingOutsideIgnoredIntervals,
  setPreviousPointIgnored,
  setServeMarkerSide,
  type ScoreTracking,
  type ScoreIgnoredInterval,
  type ServingSide,
} from "@/lib/score-tracking";

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
  canRunInference: boolean;
  onChange: (tracking: ScoreTracking) => void;
  onSelectServeMarker: (id: string, timestamp: number) => void;
  onSeek: (timestamp: number) => void;
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
  canRunInference,
  onChange,
  onSelectServeMarker,
  onSeek,
  onRunInference,
}: ScoreTrackingPanelProps) {
  const [manualSide, setManualSide] = useState<ServingSide>("near");
  const activeTracking = useMemo(
    () => scoreTrackingOutsideExcludedRallies(
      scoreTrackingOutsideIgnoredIntervals(tracking, ignoredIntervals),
      excludedRallyIds,
    ),
    [excludedRallyIds, ignoredIntervals, tracking],
  );
  const serves = useMemo(() => orderedServeMarkers(activeTracking), [activeTracking]);
  const switches = useMemo(
    () => orderedSideSwitchMarkers(activeTracking),
    [activeTracking],
  );
  const score = useMemo(
    () => deriveScoreAt(activeTracking, scoreBoundaryTime),
    [activeTracking, scoreBoundaryTime],
  );
  const finalScore = useMemo(() => deriveScoreAt(activeTracking), [activeTracking]);
  const pointTimeline = useMemo(
    () => score.points.flatMap((point, rallyIndex) =>
      point.status === "counted" && point.winnerTeamId
        ? [{
            id: point.serveMarkerId,
            rallyNumber: rallyIndex + 1,
            winnerTeamId: point.winnerTeamId,
            teamPointNumber: point.winnerTeamId === "team-1"
              ? point.team1ScoreAfter
              : point.team2ScoreAfter,
          }]
        : []
    ),
    [score.points],
  );
  const ignoredServeCount = tracking.serveMarkers.length - serves.length;
  const selected = serves.find((marker) => marker.id === selectedServeMarkerId) ?? null;
  const selectedIndex = selected
    ? serves.findIndex((marker) => marker.id === selected.id)
    : -1;
  const servingName = score.servingTeamId === "team-1"
    ? tracking.team1Name
    : score.servingTeamId === "team-2"
      ? tracking.team2Name
      : "Review needed";

  return (
    <aside
      className={styles.scorePanel}
      aria-label="Score tracking controls"
      data-tour="editor-score-panel"
    >
      <header className={styles.scorePanelHeader}>
        <div>
          <span>SCORE TRACKING · BETA</span>
          <strong>Serving-side point history</strong>
        </div>
        <small>
          {serves.length} serves · {switches.length} switches
          {ignoredServeCount > 0 ? ` · ${ignoredServeCount} ignored` : ""}
        </small>
      </header>

      <div className={styles.scoreboard} aria-live="polite">
        <label>
          <span>Team 1 · starts near</span>
          <input
            aria-label="Team 1 name"
            value={tracking.team1Name}
            onChange={(event) => onChange({
              ...tracking,
              team1Name: event.currentTarget.value || "Team 1",
            })}
          />
          <output>{score.team1Score}</output>
        </label>
        <label>
          <span>Team 2 · starts far</span>
          <input
            aria-label="Team 2 name"
            value={tracking.team2Name}
            onChange={(event) => onChange({
              ...tracking,
              team2Name: event.currentTarget.value || "Team 2",
            })}
          />
          <output>{score.team2Score}</output>
        </label>
        <p>
          Serving now: <strong>{servingName}</strong>
          {score.servingSide ? ` · ${sideLabel(score.servingSide)} side` : ""}
        </p>
        {(finalScore.reviewPointCount > 0 || finalScore.ignoredPointCount > 0) && (
          <small>
            {finalScore.reviewPointCount} unresolved · {finalScore.ignoredPointCount} replayed/ignored
          </small>
        )}
      </div>

      <div className={styles.scoreInference} data-state={inferenceStatus}>
        <div>
          <strong>
            {inferenceStatus === "running"
              ? "Analyzing serving sides…"
              : inferenceStatus === "error"
                ? "Serving-side analysis needs attention"
                : serves.length > 0
                  ? "Serve markers ready"
                  : "No serve markers yet"}
          </strong>
          <small>
            {inferenceMessage ??
              "The first serve sets the initial server. Each later serve identifies the previous rally winner; the last rally needs a later or manually added serve to be counted."}
          </small>
        </div>
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

      <section className={styles.selectedServeEditor}>
        <div>
          <span>SELECTED SERVE</span>
          <strong>
            {selected
              ? `${preciseTime(selected.timestamp)} · ${sideLabel(selected.side)}`
              : "Select a ball marker on the timeline"}
          </strong>
          {selected?.modelSide && (
            <small>
              Model: {sideLabel(selected.modelSide)}
              {selected.side !== selected.modelSide ? " · corrected" : ""}
            </small>
          )}
        </div>
        <fieldset disabled={!selected}>
          <legend>Serving side verdict</legend>
          <div className={styles.sideVerdictButtons}>
            {(["near", "far"] as const).map((side) => (
              <button
                type="button"
                key={side}
                aria-pressed={selected?.side === side}
                data-active={selected?.side === side || undefined}
                onClick={() => selected && onChange(
                  setServeMarkerSide(tracking, selected.id, side),
                )}
              >
                {sideLabel(side)}
              </button>
            ))}
          </div>
        </fieldset>
        <label className={styles.replayToggle} data-disabled={selectedIndex <= 0 || undefined}>
          <input
            type="checkbox"
            disabled={!selected || selectedIndex <= 0}
            checked={selected?.ignorePreviousPoint ?? false}
            onChange={(event) => selected && onChange(
              setPreviousPointIgnored(tracking, selected.id, event.currentTarget.checked),
            )}
          />
          <span>
            <strong>Ignore/replay previous point</strong>
            <small>Do not change the score at this serve.</small>
          </span>
        </label>
      </section>

      <section className={styles.markerAddTools}>
        <div>
          <label>
            <span>Missing serve at {preciseTime(playbackTime)}</span>
            <select
              aria-label="Side for missing serve marker"
              value={manualSide}
              onChange={(event) => setManualSide(event.currentTarget.value as ServingSide)}
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
                (marker) => !tracking.serveMarkers.some((old) => old.id === marker.id),
              );
              if (added) onSelectServeMarker(added.id, added.timestamp);
            }}
          >
            + Add serve
          </button>
        </div>
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
            {([
              ["team-1", tracking.team1Name],
              ["team-2", tracking.team2Name],
            ] as const).map(([teamId, teamName]) => (
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
                    {point.winnerTeamId === teamId && <i>{point.teamPointNumber}</i>}
                  </span>
                ))}
              </div>
            ))}
          </div>
        ) : (
          <small className={styles.pointTimelineEmpty}>No points awarded yet</small>
        )}
      </section>

      <section className={styles.scoreMarkerList} aria-label="Score marker list">
        <div>
          <span>MARKERS</span>
          <small>Markers can be removed</small>
        </div>
        <div>
          {serves.map((marker, index) => (
            <article key={marker.id} data-selected={marker.id === selected?.id || undefined}>
              <button
                type="button"
                onClick={() => onSelectServeMarker(marker.id, marker.timestamp)}
              >
                <span aria-hidden="true">🏐</span>
                <strong>{preciseTime(marker.timestamp)}</strong>
                <small>
                  {index === 0 ? "First serve" : sideLabel(marker.side)} · {marker.origin}
                </small>
              </button>
              <button
                type="button"
                className={styles.removeMarker}
                aria-label={`Remove ${marker.origin === "model" ? "predicted" : "added"} serve at ${preciseTime(marker.timestamp)}`}
                onClick={() => onChange(removeServeMarker(tracking, marker.id))}
              >
                Remove
              </button>
            </article>
          ))}
          {switches.map((marker) => (
            <article key={marker.id}>
              <button type="button" onClick={() => onSeek(marker.timestamp)}>
                <span aria-hidden="true">⇄</span>
                <strong>{preciseTime(marker.timestamp)}</strong>
                <small>
                  Team side switch · {marker.origin}
                  {marker.modelConfidence === undefined
                    ? ""
                    : ` · ${Math.round(marker.modelConfidence * 100)}%`}
                </small>
              </button>
              <button
                type="button"
                className={styles.removeMarker}
                aria-label={`Remove ${marker.origin === "model" ? "predicted" : "added"} side switch at ${preciseTime(marker.timestamp)}`}
                onClick={() => onChange(removeSideSwitchMarker(tracking, marker.id))}
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
