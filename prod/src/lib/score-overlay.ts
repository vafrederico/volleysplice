import {
  deriveScoreAt,
  type ScoreIgnoredInterval,
  type ScoreMergedRange,
  type ScoreRallyRange,
  type ScoreTeamId,
  type ScoreTracking,
  scoreBoundaryTimestamp,
  scoreTrackingOutsideExcludedRallies,
  scoreTrackingOutsideIgnoredIntervals,
} from "./score-tracking.ts";

export const SCORE_OVERLAY_COLORS = {
  border: "#000000",
  team1: "#d9342b",
  team2: "#2367c9",
  teamText: "#ffffff",
  scoreBackground: "#ffffff",
  scoreText: "#000000",
  pointText: "#ffffff",
} as const;

export const SCORE_POINT_TIMELINE_FADE_IN_SECONDS = 0.25;
export const SCORE_POINT_TIMELINE_HOLD_SECONDS = 2;
export const SCORE_POINT_TIMELINE_FADE_OUT_SECONDS = 0.35;

export type ScoreOverlayOptions = {
  scoreTracking: ScoreTracking;
  fadeScoreOverlay?: boolean;
  renderPointTimeline?: boolean;
  excludedRallyIds?: readonly string[];
  ignoredIntervals?: readonly ScoreIgnoredInterval[];
  rallyRanges?: readonly ScoreRallyRange[];
  mergedRanges?: readonly ScoreMergedRange[];
};

export type PreparedScoreOverlay = {
  scoreTracking: ScoreTracking;
  fadeScoreOverlay: boolean;
  renderPointTimeline: boolean;
  rallyRanges: readonly ScoreRallyRange[];
  mergedRanges: readonly ScoreMergedRange[];
};

export type ScoreOverlaySnapshot = {
  team1Name: string;
  team1Score: number;
  team1ScoreLabel: string;
  team2Name: string;
  team2Score: number;
  team2ScoreLabel: string;
  servingTeamId: ScoreTeamId | null;
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

export type ScorePointTimelineEntry = {
  serveMarkerId: string;
  winnerTeamId: ScoreTeamId;
  teamPointNumber: number;
};

export type ScorePointTimelineSnapshot = {
  points: readonly ScorePointTimelineEntry[];
  opacity: number;
};

export type ScorePointTimelineLayout = {
  startX: number;
  team1CenterY: number;
  team2CenterY: number;
  columnSpacing: number;
  circleRadius: number;
  lineWidth: number;
  fontSize: number;
};

export function formatOverlayScore(score: number): string {
  return String(Math.max(0, Math.trunc(score))).padStart(2, "0");
}

export function formatOverlayTeamLabel(
  teamName: string,
  isServing: boolean,
): string {
  return isServing ? `${teamName} 🏐` : teamName;
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
    fadeScoreOverlay: options.fadeScoreOverlay ?? false,
    renderPointTimeline: options.renderPointTimeline ?? true,
    rallyRanges: options.rallyRanges ?? [],
    mergedRanges: options.mergedRanges ?? [],
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
    prepared.mergedRanges,
  );
  const score = deriveScoreAt(prepared.scoreTracking, boundaryTimestamp);
  return {
    team1Name: prepared.scoreTracking.team1Name,
    team1Score: score.team1Score,
    team1ScoreLabel: formatOverlayScore(score.team1Score),
    team2Name: prepared.scoreTracking.team2Name,
    team2Score: score.team2Score,
    team2ScoreLabel: formatOverlayScore(score.team2Score),
    servingTeamId: score.servingTeamId,
  };
}

function pointRevealTimestamp(
  prepared: PreparedScoreOverlay,
  pointTimestamp: number,
): number {
  const containingRange = [...prepared.rallyRanges]
    .filter(
      (range) =>
        pointTimestamp >= range.keepStart && pointTimestamp < range.keepEnd,
    )
    .sort((left, right) => {
      const leftContainsCore =
        pointTimestamp >= left.coreStart && pointTimestamp < left.coreEnd;
      const rightContainsCore =
        pointTimestamp >= right.coreStart && pointTimestamp < right.coreEnd;
      if (leftContainsCore !== rightContainsCore)
        return leftContainsCore ? -1 : 1;
      return (
        Math.abs(left.coreStart - pointTimestamp) -
        Math.abs(right.coreStart - pointTimestamp)
      );
    })[0];
  if (!containingRange) return pointTimestamp;
  const candidate = containingRange.keepStart;
  const boundary = scoreBoundaryTimestamp(
    candidate,
    prepared.rallyRanges,
    prepared.scoreTracking,
    prepared.mergedRanges,
  );
  return boundary >= pointTimestamp ? candidate : pointTimestamp;
}

export function scorePointTimelineSnapshot(
  prepared: PreparedScoreOverlay,
  sourceTimestamp: number,
): ScorePointTimelineSnapshot {
  const boundaryTimestamp = scoreBoundaryTimestamp(
    sourceTimestamp,
    prepared.rallyRanges,
    prepared.scoreTracking,
    prepared.mergedRanges,
  );
  const score = deriveScoreAt(prepared.scoreTracking, boundaryTimestamp);
  const points = score.points.flatMap((point) =>
    point.status === "counted" && point.winnerTeamId
      ? [
          {
            serveMarkerId: point.serveMarkerId,
            winnerTeamId: point.winnerTeamId,
            teamPointNumber:
              point.winnerTeamId === "team-1"
                ? point.team1ScoreAfter
                : point.team2ScoreAfter,
          },
        ]
      : [],
  );
  const latest = [...score.points]
    .reverse()
    .find((point) => point.status === "counted" && point.winnerTeamId);
  if (!latest || points.length === 0) return { points, opacity: 0 };

  const age =
    sourceTimestamp - pointRevealTimestamp(prepared, latest.timestamp);
  const fadeInEnd = SCORE_POINT_TIMELINE_FADE_IN_SECONDS;
  const holdEnd = fadeInEnd + SCORE_POINT_TIMELINE_HOLD_SECONDS;
  const fadeOutEnd = holdEnd + SCORE_POINT_TIMELINE_FADE_OUT_SECONDS;
  const opacity =
    age < 0 || age >= fadeOutEnd
      ? 0
      : age < fadeInEnd
        ? age / SCORE_POINT_TIMELINE_FADE_IN_SECONDS
        : age < holdEnd
          ? 1
          : 1 - (age - holdEnd) / SCORE_POINT_TIMELINE_FADE_OUT_SECONDS;
  const boundedOpacity = Math.max(0, Math.min(1, opacity));
  return {
    points,
    opacity:
      boundedOpacity < 1e-6
        ? 0
        : boundedOpacity > 1 - 1e-6
          ? 1
          : boundedOpacity,
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
  const measuredTeamWidth = (name: string, teamId: ScoreTeamId) =>
    Math.max(
      minimumTeamWidth,
      Math.ceil(
        context.measureText(
          formatOverlayTeamLabel(name, snapshot.servingTeamId === teamId),
        ).width,
      ) +
        horizontalPadding * 2,
    );
  const desiredTeam1Width = measuredTeamWidth(snapshot.team1Name, "team-1");
  const desiredTeam2Width = measuredTeamWidth(snapshot.team2Name, "team-2");
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
    const scale =
      flexibleWidth > 0
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

export function scorePointTimelineLayout(
  videoWidth: number,
  videoHeight: number,
  scoreLayout: ScoreOverlayLayout,
  pointCount: number,
): ScorePointTimelineLayout {
  const availableWidth = Math.max(0, videoWidth - scoreLayout.width);
  const normalColumnSpacing = scoreLayout.height * 0.58;
  const columnSpacing =
    pointCount > 0 && availableWidth > 0
      ? Math.min(normalColumnSpacing, availableWidth)
      : 0;
  return {
    startX: scoreLayout.width,
    team1CenterY: scoreLayout.height * 0.28,
    team2CenterY: scoreLayout.height * 0.72,
    columnSpacing,
    circleRadius: Math.min(
      scoreLayout.height * 0.18,
      columnSpacing * 0.36,
      videoHeight * 0.04,
    ),
    lineWidth: Math.max(2, scoreLayout.borderWidth * 1.5),
    fontSize: Math.max(
      1,
      Math.round(Math.min(scoreLayout.height * 0.18, columnSpacing * 0.52)),
    ),
  };
}

export function visibleScorePointTimelineEntries(
  videoWidth: number,
  scoreLayout: ScoreOverlayLayout,
  points: readonly ScorePointTimelineEntry[],
): readonly ScorePointTimelineEntry[] {
  const availableWidth = Math.max(0, videoWidth - scoreLayout.width);
  if (availableWidth <= 0 || points.length === 0) return [];
  const normalColumnSpacing = scoreLayout.height * 0.58;
  const maximumVisiblePoints = Math.max(
    1,
    Math.floor(availableWidth / normalColumnSpacing),
  );
  return points.length <= maximumVisiblePoints
    ? points
    : points.slice(points.length - maximumVisiblePoints);
}

export type ScoreOverlayCanvasContext =
  | CanvasRenderingContext2D
  | OffscreenCanvasRenderingContext2D;

function scoreOverlayPath(
  context: ScoreOverlayCanvasContext,
  width: number,
  height: number,
  borderWidth: number,
  radius: number,
): void {
  const inset = borderWidth / 2;
  const right = width - inset;
  const bottom = height - inset;
  context.beginPath();
  context.moveTo(inset, inset);
  context.lineTo(right, inset);
  context.lineTo(right, bottom - radius);
  context.quadraticCurveTo(right, bottom, right - radius, bottom);
  context.lineTo(inset, bottom);
  context.closePath();
}

export function drawScoreOverlayScoreboard(
  context: ScoreOverlayCanvasContext,
  videoWidth: number,
  videoHeight: number,
  snapshot: ScoreOverlaySnapshot,
): ScoreOverlayLayout {
  const shortestEdge = Math.max(1, Math.min(videoWidth, videoHeight));
  const provisionalHeight = Math.round(
    Math.min(76, Math.max(36, shortestEdge * 0.064)),
  );
  context.font = `700 ${Math.round(provisionalHeight * 0.39)}px sans-serif`;
  const layout = scoreOverlayLayout(context, videoWidth, videoHeight, snapshot);
  const firstScoreX = layout.team1Width;
  const team2X = firstScoreX + layout.scoreWidth;
  const secondScoreX = team2X + layout.team2Width;

  context.save();
  scoreOverlayPath(
    context,
    layout.width,
    layout.height,
    layout.borderWidth,
    layout.radius,
  );
  context.clip();
  context.fillStyle = SCORE_OVERLAY_COLORS.team1;
  context.fillRect(0, 0, layout.team1Width, layout.height);
  context.fillStyle = SCORE_OVERLAY_COLORS.scoreBackground;
  context.fillRect(firstScoreX, 0, layout.scoreWidth, layout.height);
  context.fillStyle = SCORE_OVERLAY_COLORS.team2;
  context.fillRect(team2X, 0, layout.team2Width, layout.height);
  context.fillStyle = SCORE_OVERLAY_COLORS.scoreBackground;
  context.fillRect(secondScoreX, 0, layout.scoreWidth, layout.height);

  context.font = `700 ${layout.fontSize}px sans-serif`;
  context.textBaseline = "middle";
  context.fillStyle = SCORE_OVERLAY_COLORS.teamText;
  context.textAlign = "center";
  context.fillText(
    formatOverlayTeamLabel(
      snapshot.team1Name,
      snapshot.servingTeamId === "team-1",
    ),
    layout.team1Width / 2,
    layout.height / 2,
    layout.team1Width - layout.horizontalPadding * 2,
  );
  context.fillText(
    formatOverlayTeamLabel(
      snapshot.team2Name,
      snapshot.servingTeamId === "team-2",
    ),
    team2X + layout.team2Width / 2,
    layout.height / 2,
    layout.team2Width - layout.horizontalPadding * 2,
  );
  context.fillStyle = SCORE_OVERLAY_COLORS.scoreText;
  context.fillText(
    snapshot.team1ScoreLabel,
    firstScoreX + layout.scoreWidth / 2,
    layout.height / 2,
  );
  context.fillText(
    snapshot.team2ScoreLabel,
    secondScoreX + layout.scoreWidth / 2,
    layout.height / 2,
  );

  context.strokeStyle = SCORE_OVERLAY_COLORS.border;
  context.lineWidth = layout.borderWidth;
  for (const x of [firstScoreX, team2X, secondScoreX]) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, layout.height);
    context.stroke();
  }
  scoreOverlayPath(
    context,
    layout.width,
    layout.height,
    layout.borderWidth,
    layout.radius,
  );
  context.stroke();
  context.restore();
  return layout;
}

export function drawScorePointTimeline(
  context: ScoreOverlayCanvasContext,
  videoWidth: number,
  videoHeight: number,
  scoreLayout: ScoreOverlayLayout,
  timeline: ScorePointTimelineSnapshot,
  opacity = timeline.opacity,
): void {
  if (timeline.points.length === 0 || opacity <= 0) return;
  const visiblePoints = visibleScorePointTimelineEntries(
    videoWidth,
    scoreLayout,
    timeline.points,
  );
  if (visiblePoints.length === 0) return;
  const layout = scorePointTimelineLayout(
    videoWidth,
    videoHeight,
    scoreLayout,
    visiblePoints.length,
  );
  if (layout.columnSpacing <= 0 || layout.circleRadius <= 0) return;
  context.save();
  context.globalAlpha *= Math.max(0, Math.min(1, opacity));
  context.lineWidth = layout.lineWidth;
  for (const [teamId, y, color] of [
    ["team-1", layout.team1CenterY, SCORE_OVERLAY_COLORS.team1],
    ["team-2", layout.team2CenterY, SCORE_OVERLAY_COLORS.team2],
  ] as const) {
    let lastPointIndex = -1;
    visiblePoints.forEach((point, index) => {
      if (point.winnerTeamId === teamId) lastPointIndex = index;
    });
    if (lastPointIndex < 0) continue;
    const lastX = layout.startX + layout.columnSpacing * (lastPointIndex + 0.5);
    context.strokeStyle = color;
    context.beginPath();
    context.moveTo(layout.startX, y);
    context.lineTo(lastX, y);
    context.stroke();
  }
  context.font = `800 ${layout.fontSize}px sans-serif`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  visiblePoints.forEach((point, index) => {
    const x = layout.startX + layout.columnSpacing * (index + 0.5);
    const y =
      point.winnerTeamId === "team-1"
        ? layout.team1CenterY
        : layout.team2CenterY;
    context.beginPath();
    context.arc(x, y, layout.circleRadius, 0, Math.PI * 2);
    context.fillStyle =
      point.winnerTeamId === "team-1"
        ? SCORE_OVERLAY_COLORS.team1
        : SCORE_OVERLAY_COLORS.team2;
    context.fill();
    context.strokeStyle = SCORE_OVERLAY_COLORS.border;
    context.lineWidth = layout.lineWidth;
    context.stroke();
    context.fillStyle = SCORE_OVERLAY_COLORS.pointText;
    context.fillText(String(point.teamPointNumber), x, y);
  });
  context.restore();
}

export function drawScoreOverlay(
  context: ScoreOverlayCanvasContext,
  videoWidth: number,
  videoHeight: number,
  snapshot: ScoreOverlaySnapshot,
  timeline: ScorePointTimelineSnapshot,
  renderPointTimeline = true,
  fadeScoreOverlay = false,
): void {
  context.save();
  context.globalAlpha *= fadeScoreOverlay ? timeline.opacity : 1;
  const layout = drawScoreOverlayScoreboard(
    context,
    videoWidth,
    videoHeight,
    snapshot,
  );
  context.restore();
  if (renderPointTimeline) {
    drawScorePointTimeline(context, videoWidth, videoHeight, layout, timeline);
  }
}
