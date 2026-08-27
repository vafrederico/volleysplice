import type { EditableCut, FinalCutInterval } from "./cut-draft.ts";
import {
  deriveScoreAt,
  orderedServeMarkers,
  type ScoreTracking,
  type ServeMarker,
} from "./score-tracking.ts";

export type YouTubeChapterOptions = {
  includeRallyNumber: boolean;
  includeServeNumber: boolean;
  includeScore: boolean;
  includeServingTeam: boolean;
  includeSideSwitches: boolean;
};

export type YouTubeChapter = {
  kind: "rally" | "side-switch";
  outputSeconds: number;
  sourceTimestamp: number;
  title: string;
};

export function defaultYouTubeChapterOptions(
  hasScoreTracking: boolean,
  hasSideSwitches: boolean,
): YouTubeChapterOptions {
  return {
    includeRallyNumber: !hasScoreTracking,
    includeServeNumber: false,
    includeScore: hasScoreTracking,
    includeServingTeam: hasScoreTracking,
    includeSideSwitches: hasSideSwitches,
  };
}

type ChapterInput = {
  intervals: readonly FinalCutInterval[];
  cuts: readonly EditableCut[];
  scoreTracking: ScoreTracking | null;
  options: YouTubeChapterOptions;
};

type OutputInterval = FinalCutInterval & { outputStart: number };

function outputIntervals(
  intervals: readonly FinalCutInterval[],
): OutputInterval[] {
  let outputStart = 0;
  return intervals.map((interval) => {
    const mapped = { ...interval, outputStart };
    outputStart += Math.max(0, interval.end - interval.start);
    return mapped;
  });
}

function outputTimeAt(
  interval: OutputInterval,
  sourceTimestamp: number,
): number {
  return (
    interval.outputStart +
    Math.max(0, Math.min(interval.end, sourceTimestamp) - interval.start)
  );
}

function firstVisiblePointForCut(
  cut: EditableCut,
  intervals: readonly OutputInterval[],
): { outputSeconds: number; sourceTimestamp: number } | null {
  const interval = intervals.find((candidate) =>
    candidate.cutIds.includes(cut.id),
  );
  if (!interval) return null;
  const sourceTimestamp = Math.max(
    interval.start,
    Math.min(interval.end, cut.keepStart),
  );
  return {
    outputSeconds: outputTimeAt(interval, sourceTimestamp),
    sourceTimestamp,
  };
}

function outputPointAtOrAfter(
  sourceTimestamp: number,
  intervals: readonly OutputInterval[],
): { outputSeconds: number; sourceTimestamp: number } | null {
  const interval = intervals.find(
    (candidate) =>
      candidate.start <= sourceTimestamp && sourceTimestamp < candidate.end,
  );
  if (interval) {
    return {
      outputSeconds: outputTimeAt(interval, sourceTimestamp),
      sourceTimestamp,
    };
  }
  const nextInterval = intervals.find(
    (candidate) => candidate.start > sourceTimestamp,
  );
  return nextInterval
    ? {
        outputSeconds: nextInterval.outputStart,
        sourceTimestamp: nextInterval.start,
      }
    : null;
}

function markerForCut(
  cut: EditableCut,
  markers: readonly ServeMarker[],
  claimedMarkerIds: Set<string>,
): ServeMarker | null {
  const available = markers.filter(
    (marker) => !claimedMarkerIds.has(marker.id),
  );
  const linked = available.find((marker) => marker.rallyId === cut.id);
  if (linked) return linked;

  const insideCore = available.filter(
    (marker) =>
      cut.coreStart <= marker.timestamp && marker.timestamp < cut.coreEnd,
  );
  const insideKept = available.filter(
    (marker) =>
      cut.keepStart <= marker.timestamp && marker.timestamp < cut.keepEnd,
  );
  const candidates = insideCore.length > 0 ? insideCore : insideKept;
  return (
    [...candidates].sort(
      (left, right) =>
        Math.abs(left.timestamp - cut.coreStart) -
          Math.abs(right.timestamp - cut.coreStart) ||
        left.timestamp - right.timestamp ||
        left.id.localeCompare(right.id),
    )[0] ?? null
  );
}

function rallyTitle(
  rallyNumber: number,
  marker: ServeMarker | null,
  isRedo: boolean,
  serveNumbers: ReadonlyMap<string, number>,
  scoreTracking: ScoreTracking | null,
  options: YouTubeChapterOptions,
): string {
  const parts: string[] = [];
  if (options.includeRallyNumber) parts.push(`Rally ${rallyNumber}`);
  if (options.includeServeNumber && marker) {
    const serveNumber = serveNumbers.get(marker.id);
    if (serveNumber !== undefined) parts.push(`Serve ${serveNumber}`);
  }
  if (
    marker &&
    scoreTracking &&
    (options.includeScore || options.includeServingTeam)
  ) {
    const score = deriveScoreAt(scoreTracking, marker.timestamp);
    if (options.includeScore) {
      parts.push(`${score.team1Score}–${score.team2Score}`);
    }
    if (options.includeServingTeam) {
      const servingTeam =
        score.servingTeamId === "team-1"
          ? scoreTracking.team1Name
          : score.servingTeamId === "team-2"
            ? scoreTracking.team2Name
            : null;
      if (servingTeam) parts.push(`${servingTeam} serving`);
    }
  }
  if (isRedo) parts.push("Re-do");
  return parts.length > 0 ? parts.join(" - ") : `Rally ${rallyNumber}`;
}

function mergeSameSecondChapters(
  chapters: readonly YouTubeChapter[],
): YouTubeChapter[] {
  return chapters.reduce<YouTubeChapter[]>((merged, chapter) => {
    const outputSeconds = Math.max(0, Math.floor(chapter.outputSeconds));
    const previous = merged.at(-1);
    if (previous && previous.outputSeconds === outputSeconds) {
      if (!previous.title.split(" / ").includes(chapter.title)) {
        previous.title = `${previous.title} / ${chapter.title}`;
      }
      return merged;
    }
    merged.push({ ...chapter, outputSeconds });
    return merged;
  }, []);
}

export function buildYouTubeChapters({
  intervals,
  cuts,
  scoreTracking,
  options,
}: ChapterInput): YouTubeChapter[] {
  const mappedIntervals = outputIntervals(intervals);
  const visibleCuts = cuts
    .flatMap((cut) => {
      const point = firstVisiblePointForCut(cut, mappedIntervals);
      return point ? [{ cut, ...point }] : [];
    })
    .sort(
      (left, right) =>
        left.outputSeconds - right.outputSeconds ||
        left.cut.coreStart - right.cut.coreStart ||
        left.cut.id.localeCompare(right.cut.id),
    );
  const markers = scoreTracking ? orderedServeMarkers(scoreTracking) : [];
  const serveNumbers = new Map(
    markers.map((marker, index) => [marker.id, index + 1]),
  );
  const redoMarkerIds = new Set(
    markers.flatMap((marker, index) =>
      markers[index + 1]?.ignorePreviousPoint ? [marker.id] : [],
    ),
  );
  const claimedMarkerIds = new Set<string>();
  const rallyChapters = visibleCuts.map((visible, index): YouTubeChapter => {
    const marker = markerForCut(visible.cut, markers, claimedMarkerIds);
    if (marker) claimedMarkerIds.add(marker.id);
    return {
      kind: "rally",
      outputSeconds: visible.outputSeconds,
      sourceTimestamp: visible.sourceTimestamp,
      title: rallyTitle(
        index + 1,
        marker,
        marker ? redoMarkerIds.has(marker.id) : false,
        serveNumbers,
        scoreTracking,
        options,
      ),
    };
  });

  const switchChapters =
    options.includeSideSwitches && scoreTracking
      ? scoreTracking.sideSwitchMarkers
          .flatMap((marker) => {
            const point = outputPointAtOrAfter(
              marker.timestamp,
              mappedIntervals,
            );
            return point ? [point] : [];
          })
          .map(
            (point, index): YouTubeChapter => ({
              kind: "side-switch",
              ...point,
              title: `Side switch ${index + 1}`,
            }),
          )
      : [];

  return mergeSameSecondChapters(
    [...rallyChapters, ...switchChapters].sort(
      (left, right) =>
        left.outputSeconds - right.outputSeconds ||
        (left.kind === "rally" ? -1 : 1),
    ),
  );
}

export function formatYouTubeChapterTimestamp(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds)
    ? Math.max(0, Math.floor(seconds))
    : 0;
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const remainder = safeSeconds % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

export function youtubeChaptersText(
  chapters: readonly YouTubeChapter[],
): string {
  return chapters
    .map(
      (chapter) =>
        `${formatYouTubeChapterTimestamp(chapter.outputSeconds)} ${chapter.title}`,
    )
    .join("\n");
}

export function youtubeChaptersFilename(sourceFilename: string): string {
  const base =
    sourceFilename
      .replace(/\.[^.]+$/, "")
      .replace(/[^a-zA-Z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "volleycut";
  return `${base}-youtube-chapters.txt`;
}
