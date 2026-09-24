import { useEffect, useMemo, useRef } from "react";

import {
  drawScoreOverlayScoreboard,
  drawScorePointTimeline,
  type PreparedScoreOverlay,
  scoreOverlayLayout,
  scoreOverlaySnapshot,
  scorePointTimelineSnapshot,
} from "../lib/score-overlay";

import styles from "./ScoreOverlay.module.css";

type ScoreOverlayProps = {
  prepared: PreparedScoreOverlay;
  timestamp: number;
  videoWidth: number;
  videoHeight: number;
  className?: string;
};

export function ScoreOverlay({
  prepared,
  timestamp,
  videoWidth,
  videoHeight,
  className,
}: ScoreOverlayProps) {
  const scoreboardCanvas = useRef<HTMLCanvasElement>(null);
  const timelineCanvas = useRef<HTMLCanvasElement>(null);
  const score = useMemo(
    () => scoreOverlaySnapshot(prepared, timestamp),
    [prepared, timestamp],
  );
  const timeline = useMemo(
    () =>
      prepared.renderPointTimeline || prepared.fadeScoreOverlay
        ? scorePointTimelineSnapshot(prepared, timestamp)
        : { points: [], opacity: 0 },
    [prepared, timestamp],
  );
  const width = Math.max(1, Math.round(videoWidth));
  const height = Math.max(1, Math.round(videoHeight));

  useEffect(() => {
    const context = scoreboardCanvas.current?.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    drawScoreOverlayScoreboard(context, width, height, score);
  }, [height, score, width]);

  useEffect(() => {
    const context = timelineCanvas.current?.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    const shortestEdge = Math.max(1, Math.min(width, height));
    const provisionalHeight = Math.round(
      Math.min(76, Math.max(36, shortestEdge * 0.064)),
    );
    context.font = `700 ${Math.round(provisionalHeight * 0.39)}px sans-serif`;
    const layout = scoreOverlayLayout(context, width, height, score);
    drawScorePointTimeline(context, width, height, layout, timeline, 1);
  }, [height, score, timeline, width]);

  const servingTeamName =
    score.servingTeamId === "team-1"
      ? score.team1Name
      : score.servingTeamId === "team-2"
        ? score.team2Name
        : null;
  const label = `${score.team1Name} ${score.team1ScoreLabel}, ${score.team2Name} ${score.team2ScoreLabel}${servingTeamName ? `, ${servingTeamName} serving` : ""}`;

  return (
    <div
      className={[styles.overlay, className].filter(Boolean).join(" ")}
      aria-label={label}
      role="img"
    >
      <canvas
        ref={scoreboardCanvas}
        style={{ opacity: prepared.fadeScoreOverlay ? timeline.opacity : 1 }}
        width={width}
        height={height}
        aria-hidden="true"
      />
      <canvas
        ref={timelineCanvas}
        width={width}
        height={height}
        aria-hidden="true"
        style={{ opacity: prepared.renderPointTimeline ? timeline.opacity : 0 }}
      />
    </div>
  );
}
