import {
  deriveScoreAt,
  scoreBoundaryTimestamp,
  scoreTrackingOutsideExcludedRallies,
  scoreTrackingOutsideIgnoredIntervals,
  type ScoreIgnoredInterval,
  type ScoreRallyRange,
  type ScoreTracking,
} from "./score-tracking.ts";

export const SCORE_OVERLAY_COLORS = {
  border: "#000000",
  team1: "#d9342b",
  team2: "#2367c9",
  teamText: "#ffffff",
  scoreBackground: "#ffffff",
  scoreText: "#000000",
} as const;

export type ScoreOverlayOptions = {
  scoreTracking: ScoreTracking;
  excludedRallyIds?: readonly string[];
  ignoredIntervals?: readonly ScoreIgnoredInterval[];
  rallyRanges?: readonly ScoreRallyRange[];
};

export type PreparedScoreOverlay = {
  scoreTracking: ScoreTracking;
  rallyRanges: readonly ScoreRallyRange[];
};

export type ScoreOverlaySnapshot = {
  team1Name: string;
  team1Score: number;
  team1ScoreLabel: string;
  team2Name: string;
  team2Score: number;
  team2ScoreLabel: string;
};

export type ScoreOverlayLayout = {
  width: number;
  height: number;
  team1Width: number;
  team2Width: number;
  scoreWidth: number;
  borderWidth: number;
  radius: number;
  fontSize: number;
  horizontalPadding: number;
};

export function formatOverlayScore(score: number): string {
  return String(Math.max(0, Math.trunc(score))).padStart(2, "0");
}

export function prepareScoreOverlay(
  options: ScoreOverlayOptions,
): PreparedScoreOverlay {
  const outsideIgnored = scoreTrackingOutsideIgnoredIntervals(
    options.scoreTracking,
    options.ignoredIntervals ?? [],
  );
  const scoreTracking = scoreTrackingOutsideExcludedRallies(
    outsideIgnored,
    new Set(options.excludedRallyIds ?? []),
  );
  return {
    scoreTracking,
    rallyRanges: options.rallyRanges ?? [],
  };
}

export function scoreOverlaySnapshot(
  prepared: PreparedScoreOverlay,
  sourceTimestamp: number,
): ScoreOverlaySnapshot {
  const boundaryTimestamp = scoreBoundaryTimestamp(
    sourceTimestamp,
    prepared.rallyRanges,
    prepared.scoreTracking,
  );
  const score = deriveScoreAt(prepared.scoreTracking, boundaryTimestamp);
  return {
    team1Name: prepared.scoreTracking.team1Name,
    team1Score: score.team1Score,
    team1ScoreLabel: formatOverlayScore(score.team1Score),
    team2Name: prepared.scoreTracking.team2Name,
    team2Score: score.team2Score,
    team2ScoreLabel: formatOverlayScore(score.team2Score),
  };
}

type TextMeasurer = Pick<CanvasRenderingContext2D, "measureText">;

/**
 * Produces a compact layout that remains readable from SD through 4K while
 * allowing team names to grow independently up to a sensible maximum.
 */
export function scoreOverlayLayout(
  context: TextMeasurer,
  videoWidth: number,
  videoHeight: number,
  snapshot: ScoreOverlaySnapshot,
): ScoreOverlayLayout {
  const shortestEdge = Math.max(1, Math.min(videoWidth, videoHeight));
  const height = Math.round(Math.min(76, Math.max(36, shortestEdge * 0.064)));
  const borderWidth = shortestEdge >= 720 ? 2 : 1;
  const fontSize = Math.round(height * 0.39);
  const horizontalPadding = Math.round(height * 0.24);
  const scoreWidth = Math.round(height * 1.3);
  const minimumTeamWidth = Math.round(height * 2.25);
  const measuredTeamWidth = (name: string) =>
    Math.max(
      minimumTeamWidth,
      Math.ceil(context.measureText(name).width) + horizontalPadding * 2,
    );
  const desiredTeam1Width = measuredTeamWidth(snapshot.team1Name);
  const desiredTeam2Width = measuredTeamWidth(snapshot.team2Name);
  const maximumOverlayWidth = Math.max(
    minimumTeamWidth * 2 + scoreWidth * 2,
    Math.floor(videoWidth * 0.96),
  );
  const availableTeamWidth = maximumOverlayWidth - scoreWidth * 2;
  const desiredTeamWidth = desiredTeam1Width + desiredTeam2Width;
  let team1Width = desiredTeam1Width;
  let team2Width = desiredTeam2Width;
  if (desiredTeamWidth > availableTeamWidth) {
    const flexibleTeam1Width = desiredTeam1Width - minimumTeamWidth;
    const flexibleTeam2Width = desiredTeam2Width - minimumTeamWidth;
    const flexibleWidth = flexibleTeam1Width + flexibleTeam2Width;
    const availableFlexibleWidth = Math.max(
      0,
      availableTeamWidth - minimumTeamWidth * 2,
    );
    const scale = flexibleWidth > 0
      ? Math.min(1, availableFlexibleWidth / flexibleWidth)
      : 0;
    team1Width = Math.round(minimumTeamWidth + flexibleTeam1Width * scale);
    team2Width = Math.round(minimumTeamWidth + flexibleTeam2Width * scale);
  }

  return {
    width: team1Width + scoreWidth + team2Width + scoreWidth,
    height,
    team1Width,
    team2Width,
    scoreWidth,
    borderWidth,
    radius: Math.round(height * 0.24),
    fontSize,
    horizontalPadding,
  };
}
