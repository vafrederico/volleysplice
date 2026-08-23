import { type CSSProperties, useMemo } from "react";

import { formatOverlayScore, SCORE_OVERLAY_COLORS } from "@/lib/score-overlay";
import { deriveScoreAt, type ScoreTracking } from "@/lib/score-tracking";

import styles from "./ScoreOverlay.module.css";

type ScoreOverlayProps = {
  tracking: ScoreTracking;
  timestamp: number;
  className?: string;
};

export function ScoreOverlay({
  tracking,
  timestamp,
  className,
}: ScoreOverlayProps) {
  const score = useMemo(
    () => deriveScoreAt(tracking, timestamp),
    [timestamp, tracking],
  );
  const team1Score = formatOverlayScore(score.team1Score);
  const team2Score = formatOverlayScore(score.team2Score);
  const servingTeamName =
    score.servingTeamId === "team-1"
      ? tracking.team1Name
      : score.servingTeamId === "team-2"
        ? tracking.team2Name
        : null;
  const label = `${tracking.team1Name} ${team1Score}, ${tracking.team2Name} ${team2Score}${servingTeamName ? `, ${servingTeamName} serving` : ""}`;
  const colors = {
    "--score-overlay-border": SCORE_OVERLAY_COLORS.border,
    "--score-overlay-team-1": SCORE_OVERLAY_COLORS.team1,
    "--score-overlay-team-2": SCORE_OVERLAY_COLORS.team2,
    "--score-overlay-team-text": SCORE_OVERLAY_COLORS.teamText,
    "--score-overlay-score-background": SCORE_OVERLAY_COLORS.scoreBackground,
    "--score-overlay-score-text": SCORE_OVERLAY_COLORS.scoreText,
  } as CSSProperties;

  return (
    <div
      className={[styles.overlay, className].filter(Boolean).join(" ")}
      aria-label={label}
      role="img"
      style={colors}
    >
      <span className={styles.team1Name}>
        <span className={styles.teamLabel}>{tracking.team1Name}</span>
        {score.servingTeamId === "team-1" && (
          <span className={styles.servingIcon} aria-hidden="true">
            🏐
          </span>
        )}
      </span>
      <strong className={styles.score}>{team1Score}</strong>
      <span className={styles.team2Name}>
        <span className={styles.teamLabel}>{tracking.team2Name}</span>
        {score.servingTeamId === "team-2" && (
          <span className={styles.servingIcon} aria-hidden="true">
            🏐
          </span>
        )}
      </span>
      <strong className={styles.score}>{team2Score}</strong>
    </div>
  );
}
