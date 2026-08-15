"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Brand } from "@/components/brand";
import { RallyTimeline, type TimelineTrack } from "@/components/rally-timeline";
import {
  downloadLabels,
  endObservabilityValues,
  formatPreciseTime,
  hardNegativeCategories,
  parseLabelDocument,
  playerCourtSideValues,
  playerStateValues,
  playerTeamValues,
  playerTrackletWindowValues,
  roundTime,
  terminalCueValues,
  type CourtGeometry,
  type HardNegative,
  type IgnoredInterval,
  type LabelDocument,
  type NormalizedBox,
  type NormalizedPoint,
  type PlayerTracklet,
  type PlayerTrackletObservation,
  type RallyLabel,
  type SideSwitch,
} from "@/lib/annotations";
import {
  DEFAULT_JOIN_GAP_SECONDS,
  MAX_JOIN_GAP_SECONDS,
  type Rally,
} from "@/lib/edit-list";
import {
  buildLiveTimeComparisonSegments,
  calculateF1,
  calculateLiveTimeMetrics,
  excludeIgnoredTime,
  markModelPaddingOrigins,
  padAndMergeRallies,
  totalRallySeconds,
} from "@/lib/timeline-comparison";
import styles from "./labeling-editor.module.css";

type IntervalKind = "rally" | "ignored" | "negative";
type LabelingBatch = "full" | "pilot";
type TrackletCaptureMode = "footpoint" | "box";
type TrackletBoxDrag = { start: NormalizedPoint; current: NormalizedPoint };
type ModelReference = {
  modelId: string;
  modelLabel: string;
  description?: string;
  rallies: RallyLabel[];
};
type CourtAnchorId =
  | "nearLeft"
  | "nearRight"
  | "farLeft"
  | "farRight"
  | "netLeft"
  | "netRight"
  | "serviceNear"
  | "serviceFar";

const courtAnchorSpecs: Array<{ id: CourtAnchorId; label: string; optional: boolean }> = [
  { id: "nearLeft", label: "Near-left corner", optional: false },
  { id: "nearRight", label: "Near-right corner", optional: false },
  { id: "farLeft", label: "Far-left corner", optional: false },
  { id: "farRight", label: "Far-right corner", optional: false },
  { id: "netLeft", label: "Net-left", optional: true },
  { id: "netRight", label: "Net-right", optional: true },
  { id: "serviceNear", label: "Near service zone", optional: true },
  { id: "serviceFar", label: "Far service zone", optional: true },
];

function courtPoint(geometry: CourtGeometry | undefined, id: CourtAnchorId): NormalizedPoint | undefined {
  if (!geometry) return undefined;
  if (id === "netLeft") return geometry.netAnchors?.left;
  if (id === "netRight") return geometry.netAnchors?.right;
  if (id === "serviceNear") return geometry.serviceZoneAnchors?.near;
  if (id === "serviceFar") return geometry.serviceZoneAnchors?.far;
  return geometry.corners[id];
}

function setCourtPoint(
  geometry: CourtGeometry | undefined,
  id: CourtAnchorId,
  point: NormalizedPoint | undefined,
): CourtGeometry {
  const next: CourtGeometry = {
    corners: { ...(geometry?.corners ?? {}) },
    ...(geometry?.netAnchors ? { netAnchors: { ...geometry.netAnchors } } : {}),
    ...(geometry?.serviceZoneAnchors
      ? { serviceZoneAnchors: { ...geometry.serviceZoneAnchors } }
      : {}),
  };
  if (id === "netLeft" || id === "netRight") {
    next.netAnchors = { ...(next.netAnchors ?? {}), [id === "netLeft" ? "left" : "right"]: point };
  } else if (id === "serviceNear" || id === "serviceFar") {
    next.serviceZoneAnchors = {
      ...(next.serviceZoneAnchors ?? {}),
      [id === "serviceNear" ? "near" : "far"]: point,
    };
  } else {
    next.corners = { ...next.corners, [id]: point };
  }
  return next;
}

type PreparedTaskSummary = {
  id: string;
  batch: LabelingBatch;
  priority: number;
  environment: LabelDocument["recording"]["environment"];
  split: LabelDocument["recording"]["split"];
  durationSeconds: number;
  originalFilename: string;
  videoFilename: string;
  documentSource: "draft" | "completed" | "production-model" | "prelabel" | "task";
  savedAt: string | null;
  annotationStatus: LabelDocument["annotation"]["status"];
  rallyCount: number;
};

type BatchSummary = Record<
  LabelingBatch,
  { ready: number; total: number; saved: number; prelabeled: number }
>;

const emptyBatchSummary: BatchSummary = {
  full: { ready: 0, total: 0, saved: 0, prelabeled: 0 },
  pilot: { ready: 0, total: 0, saved: 0, prelabeled: 0 },
};

const editableRallyTags = new Set(["service-fault", "ace", "interrupted-replay"]);
const playbackResumeKey = "volleycut.labeling.playback.v1";
const timestampEpsilon = 0.0005;
const trackletFrameEpsilon = 0.001;
const comparisonPaddingCases = [2, 3] as const;
const activityPaddingStorageKey = "volleycut:activity-padding:v1";
const activityPaddingEvent = "volleycut:activity-padding";

function metricPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function comparableRallies(rows: RallyLabel[], prefix: string): Rally[] {
  return rows.map((row, index) => ({
    id: `${prefix}-${index + 1}`,
    start: row.start,
    end: row.end,
    confidence: 1,
    included: true,
  }));
}

function playerWindowBounds(
  rally: RallyLabel,
  window: PlayerTracklet["window"],
  duration: number,
): { start: number; end: number } {
  return window === "serve"
    ? { start: Math.max(0, rally.start - 2), end: Math.min(duration, rally.start + 3) }
    : { start: Math.max(0, rally.end - 3), end: Math.min(duration, rally.end + 2) };
}

function trackletsInsideRallyBounds(
  tracklets: PlayerTracklet[] | undefined,
  rally: RallyLabel,
  duration: number,
): PlayerTracklet[] | undefined {
  if (!tracklets) return undefined;
  const retained = tracklets.flatMap((tracklet) => {
    const bounds = playerWindowBounds(rally, tracklet.window, duration);
    const observations = tracklet.observations.filter(
      (observation) => observation.time >= bounds.start && observation.time <= bounds.end,
    );
    return observations.length > 0 ? [{ ...tracklet, observations }] : [];
  });
  return retained.length > 0 ? retained : undefined;
}

function normalizedPointer(
  event: React.PointerEvent<HTMLButtonElement> | React.MouseEvent<HTMLButtonElement>,
): NormalizedPoint {
  const bounds = event.currentTarget.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
    y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
  };
}

type PlaybackResume = {
  version: 1;
  taskId: string;
  time: number;
};

function readPlaybackResume(): PlaybackResume | null {
  try {
    const raw = window.localStorage.getItem(playbackResumeKey);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<PlaybackResume>;
    if (
      value.version !== 1 ||
      typeof value.taskId !== "string" ||
      !/^[A-Za-z0-9_-]+$/.test(value.taskId) ||
      typeof value.time !== "number" ||
      !Number.isFinite(value.time) ||
      value.time < 0
    ) {
      window.localStorage.removeItem(playbackResumeKey);
      return null;
    }
    return value as PlaybackResume;
  } catch {
    try {
      window.localStorage.removeItem(playbackResumeKey);
    } catch {
      // Storage can be unavailable in privacy-restricted browser contexts.
    }
    return null;
  }
}

function overlaps(start: number, end: number, rows: Array<{ start: number; end: number }>): boolean {
  return rows.some((row) => start < row.end && row.start < end);
}

function totalSeconds(rows: Array<{ start: number; end: number }>): number {
  return rows.reduce((total, row) => total + row.end - row.start, 0);
}

function hardNegativeLabel(value: (typeof hardNegativeCategories)[number]): string {
  const labels: Partial<Record<(typeof hardNegativeCategories)[number], string>> = {
    "adjacent-court": "Adjacent-court play",
    "celebration-huddle": "Celebration / huddle",
    celebration: "Celebration (legacy)",
    "foreground-crossing": "Foreground crossing",
    "model-false-positive": "Model false positive",
    "random-dead-control": "Random dead-time control",
    "setup-between-points": "Setup between points",
    "walking-ball-retrieval": "Walking / ball retrieval",
  };
  return labels[value] ?? value.replaceAll("-", " ");
}

export function LabelingEditor() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const preparedRequestRef = useRef<AbortController | null>(null);
  const pendingResumeSecondsRef = useRef<number | null>(null);
  const resumeAttemptedRef = useRef(false);
  const lastPersistedPlaybackRef = useRef<{ taskId: string; time: number } | null>(null);
  const [labels, setLabels] = useState<LabelDocument | null>(null);
  const [productionReference, setProductionReference] =
    useState<ModelReference | null>(null);
  const [experimentReferences, setExperimentReferences] = useState<ModelReference[]>([]);
  const [solReferenceRallies, setSolReferenceRallies] = useState<RallyLabel[]>([]);
  const [preparedTasks, setPreparedTasks] = useState<PreparedTaskSummary[]>([]);
  const [batchSummary, setBatchSummary] = useState<BatchSummary>(emptyBatchSummary);
  const [selectedBatch, setSelectedBatch] = useState<LabelingBatch>("full");
  const [selectedPreparedTask, setSelectedPreparedTask] = useState("");
  const [preparedTasksLoading, setPreparedTasksLoading] = useState(true);
  const [savingDraft, setSavingDraft] = useState(false);
  const [lastSavedAt, setLastSavedAt] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoFilename, setVideoFilename] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [rallyStart, setRallyStart] = useState<number | null>(null);
  const [ignoredStart, setIgnoredStart] = useState<number | null>(null);
  const [negativeStart, setNegativeStart] = useState<number | null>(null);
  const [negativeCategory, setNegativeCategory] = useState("foreground-crossing");
  const [courtAnchor, setCourtAnchor] = useState<CourtAnchorId | null>(null);
  const [trackletRallyIndex, setTrackletRallyIndex] = useState<number | null>(null);
  const [trackletWindow, setTrackletWindow] =
    useState<PlayerTracklet["window"]>("serve");
  const [trackletId, setTrackletId] = useState("P1");
  const [trackletTeam, setTrackletTeam] = useState<PlayerTracklet["team"]>("team-a");
  const [trackletCourtSide, setTrackletCourtSide] =
    useState<PlayerTracklet["courtSide"]>("near");
  const [trackletState, setTrackletState] =
    useState<PlayerTrackletObservation["state"]>("ready");
  const [trackletCapture, setTrackletCapture] = useState<TrackletCaptureMode | null>(null);
  const [trackletBoxDrag, setTrackletBoxDrag] = useState<TrackletBoxDrag | null>(null);
  const [message, setMessage] = useState(
    "Choose a prepared full-corpus task, or use the local fallback files.",
  );
  const [error, setError] = useState<string | null>(null);
  const [joinGapSeconds, setJoinGapSeconds] = useState(DEFAULT_JOIN_GAP_SECONDS);

  useEffect(() => {
    const syncJoinGap = () => {
      try {
        const raw = window.localStorage.getItem(activityPaddingStorageKey);
        if (!raw) return;
        const value = JSON.parse(raw) as { joinGap?: unknown } | null;
        if (typeof value?.joinGap === "number" && Number.isFinite(value.joinGap)) {
          setJoinGapSeconds(Math.max(0, Math.min(MAX_JOIN_GAP_SECONDS, value.joinGap)));
        }
      } catch {
        // Storage can be unavailable or contain a legacy value without joinGap.
      }
    };
    syncJoinGap();
    window.addEventListener("storage", syncJoinGap);
    window.addEventListener(activityPaddingEvent, syncJoinGap);
    return () => {
      window.removeEventListener("storage", syncJoinGap);
      window.removeEventListener(activityPaddingEvent, syncJoinGap);
    };
  }, []);

  function updateJoinGapSeconds(seconds: number) {
    if (!Number.isFinite(seconds)) return;
    const next = Math.max(0, Math.min(MAX_JOIN_GAP_SECONDS, seconds));
    setJoinGapSeconds(next);
    try {
      const raw = window.localStorage.getItem(activityPaddingStorageKey);
      const parsed = raw ? JSON.parse(raw) as unknown : null;
      const persisted = parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? parsed as Record<string, unknown>
        : {};
      const before = typeof persisted.before === "number" && Number.isFinite(persisted.before)
        ? persisted.before
        : 3;
      const after = typeof persisted.after === "number" && Number.isFinite(persisted.after)
        ? persisted.after
        : 2;
      window.localStorage.setItem(
        activityPaddingStorageKey,
        JSON.stringify({ before, after, joinGap: next }),
      );
      window.dispatchEvent(new Event(activityPaddingEvent));
    } catch {
      // The in-memory setting still works when browser storage is unavailable.
    }
  }

  useEffect(() => {
    return () => {
      if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => {
    const controller = new AbortController();
    let firstLoad = true;
    async function loadPreparedTasks() {
      try {
        const response = await fetch("/api/labeling/tasks", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Prepared labeling tasks are unavailable");
        const payload = (await response.json()) as {
          batches?: BatchSummary;
          tasks?: PreparedTaskSummary[];
        };
        if (!Array.isArray(payload.tasks) || !payload.batches) {
          throw new Error("Prepared task catalog is invalid");
        }
        setPreparedTasks(payload.tasks);
        setBatchSummary(payload.batches);
        if (!resumeAttemptedRef.current) {
          resumeAttemptedRef.current = true;
          const resume = readPlaybackResume();
          const resumedTask = resume
            ? payload.tasks.find((task) => task.id === resume.taskId)
            : undefined;
          if (resume && resumedTask) {
            setSelectedBatch(resumedTask.batch);
            void loadPreparedTask(resume.taskId, resume.time);
          }
        }
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(loadError instanceof Error ? loadError.message : "Could not list labeling tasks");
        }
      } finally {
        if (!controller.signal.aborted && firstLoad) {
          setPreparedTasksLoading(false);
          firstLoad = false;
        }
      }
    }
    void loadPreparedTasks();
    const refresh = window.setInterval(() => void loadPreparedTasks(), 15_000);
    return () => {
      window.clearInterval(refresh);
      controller.abort();
    };
  }, []);

  const allRows = useMemo(() => {
    if (!labels) return [];
    return [...labels.rallies, ...labels.ignoredIntervals, ...labels.hardNegatives];
  }, [labels]);

  const selectedRallyIndex = useMemo(
    () =>
      labels?.rallies.findIndex(
        (row) => row.start < currentTime && currentTime < row.end,
      ) ?? -1,
    [currentTime, labels],
  );

  const previousRallyIndex = useMemo(() => {
    if (!labels) return -1;
    let previous = -1;
    labels.rallies.forEach((row, index) => {
      if (row.end <= currentTime + timestampEpsilon) previous = index;
    });
    return previous;
  }, [currentTime, labels]);

  const playheadInsideRally = selectedRallyIndex >= 0;
  const startingRallyIndex = useMemo(
    () =>
      labels?.rallies.findIndex(
        (row) => Math.abs(row.start - currentTime) < timestampEpsilon,
      ) ?? -1,
    [currentTime, labels],
  );
  const sidebarRallyIndex =
    selectedRallyIndex >= 0
      ? selectedRallyIndex
      : startingRallyIndex >= 0
        ? startingRallyIndex
        : previousRallyIndex;
  const sidebarRally =
    labels && sidebarRallyIndex >= 0 ? labels.rallies[sidebarRallyIndex] : null;
  const activeTrackletRallyIndex =
    labels && trackletRallyIndex !== null && labels.rallies[trackletRallyIndex]
      ? trackletRallyIndex
      : sidebarRallyIndex;
  const activeTrackletRally =
    labels && activeTrackletRallyIndex >= 0
      ? labels.rallies[activeTrackletRallyIndex]
      : null;
  const currentPlayerObservations = useMemo(() => {
    if (!activeTrackletRally) return [];
    return (activeTrackletRally.playerTracklets ?? []).flatMap((tracklet) =>
      tracklet.observations
        .filter((observation) => Math.abs(observation.time - currentTime) <= 0.02)
        .map((observation) => ({ tracklet, observation })),
    );
  }, [activeTrackletRally, currentTime]);

  const selectedPreparedSummary = useMemo(
    () => preparedTasks.find((task) => task.id === selectedPreparedTask) ?? null,
    [preparedTasks, selectedPreparedTask],
  );

  const tasksForSelectedBatch = useMemo(
    () => preparedTasks.filter((task) => task.batch === selectedBatch),
    [preparedTasks, selectedBatch],
  );

  const courtPolygon = useMemo(() => {
    const geometry = labels?.recording.courtGeometry;
    const points = [
      geometry?.corners.nearLeft,
      geometry?.corners.nearRight,
      geometry?.corners.farRight,
      geometry?.corners.farLeft,
    ];
    return points.every((point) => point !== undefined)
      ? [...points, points[0]]
          .map((point) => `${(point?.x ?? 0) * 100},${(point?.y ?? 0) * 100}`)
          .join(" ")
      : null;
  }, [labels?.recording.courtGeometry]);

  const touchingRallyIndexes = useMemo(() => {
    const indexes = new Set<number>();
    labels?.rallies.forEach((row, index) => {
      if (index > 0 && row.start <= labels.rallies[index - 1].end) {
        indexes.add(index - 1);
        indexes.add(index);
      }
    });
    return indexes;
  }, [labels]);

  const referenceComparisons = useMemo(() => {
    if (!labels) return [];
    const duration = labels.recording.durationSeconds;
    const humanCore = comparableRallies(labels.rallies, "editable");
    const ignored = labels.ignoredIntervals;
    const references = [
      ...(productionReference ? [{ ...productionReference, baseline: true }] : []),
      ...experimentReferences.map((reference) => ({ ...reference, baseline: false })),
    ];
    return references.flatMap((reference) => {
      const modelCore = comparableRallies(reference.rallies, reference.modelId);
      return comparisonPaddingCases.map((paddingSeconds) => {
        // Padding is merged before any duration or metric calculation, so
        // overlapping/touching model exports contribute to the union only once.
        const paddedModel = padAndMergeRallies(
          modelCore,
          paddingSeconds,
          paddingSeconds,
          duration,
          joinGapSeconds,
        );
        const paddedHuman = padAndMergeRallies(
          humanCore,
          paddingSeconds,
          paddingSeconds,
          duration,
          joinGapSeconds,
        );
        const paddedHumanMetrics = calculateLiveTimeMetrics(
          paddedModel,
          paddedHuman,
          ignored,
        );
        const coreHumanMetrics = calculateLiveTimeMetrics(
          paddedModel,
          humanCore,
          ignored,
        );
        const precision = paddedHumanMetrics.precision;
        const recall = coreHumanMetrics.recall;
        const segments = markModelPaddingOrigins(
          buildLiveTimeComparisonSegments(paddedModel, humanCore, ignored),
          modelCore,
          paddingSeconds,
          paddingSeconds,
          duration,
        );
        return {
          reference,
          paddingSeconds,
          precision,
          recall,
          f1: calculateF1(precision, recall),
          exportSeconds: totalRallySeconds(excludeIgnoredTime(paddedModel, ignored)),
          exportRallies: excludeIgnoredTime(paddedModel, ignored),
          joinedGapRallies: excludeIgnoredTime(
            paddedModel.flatMap((rally, rallyIndex) =>
              rally.joinedGaps.map((gap, gapIndex) => ({
                id: `${reference.modelId}-${paddingSeconds}s-gap-${rallyIndex + 1}-${gapIndex + 1}`,
                start: gap.start,
                end: gap.end,
                confidence: 1,
                included: true,
              }))),
            ignored,
          ),
          missingHumanSegments: segments.filter((segment) => segment.kind === "missed"),
          segments,
        };
      });
    });
  }, [experimentReferences, joinGapSeconds, labels, productionReference]);

  const completionIssues = useMemo(() => {
    if (!labels) return ["Load a label task"];
    const issues: string[] = [];
    if (!videoUrl) issues.push("Select the matching proxy video");
    if (videoFilename && videoFilename !== labels.recording.videoFilename) {
      issues.push(`Selected video must be ${labels.recording.videoFilename}`);
    }
    if (
      videoDuration !== null &&
      Math.abs(videoDuration - labels.recording.durationSeconds) > Math.max(0.1, 1 / 24)
    ) {
      issues.push("Selected video duration does not match the task");
    }
    if (!labels.annotation.annotator.trim()) issues.push("Enter the annotator name");
    if (!labels.annotation.continuousVideoReviewed) issues.push("Confirm the complete video was reviewed");
    if (labels.recording.game.playersPerTeam === null) issues.push("Set players per team");
    if (labels.rallies.length === 0) issues.push("Label at least one rally");
    if (rallyStart !== null || ignoredStart !== null || negativeStart !== null) {
      issues.push("Finish or cancel the open interval marker");
    }
    const orderedGroups = [labels.rallies, labels.ignoredIntervals, labels.hardNegatives];
    orderedGroups.forEach((rows) => {
      rows.forEach((row, index) => {
        if (row.start < 0 || row.end <= row.start || row.end > labels.recording.durationSeconds) {
          issues.push(`Fix invalid interval at ${formatPreciseTime(row.start)} (#${index + 1})`);
        }
        if (index > 0 && row.start < rows[index - 1].end) issues.push("Intervals overlap or are unordered");
      });
    });
    if (touchingRallyIndexes.size > 0) {
      issues.push("Separate touching rallies with a positive dead-time gap");
    }
    labels.rallies.forEach((row) => {
      if (overlaps(row.start, row.end, labels.ignoredIntervals)) issues.push("A rally overlaps ignored time");
      if (overlaps(row.start, row.end, labels.hardNegatives)) issues.push("A rally overlaps a hard negative");
    });
    labels.ignoredIntervals.forEach((row) => {
      if (overlaps(row.start, row.end, labels.hardNegatives)) issues.push("Ignored time overlaps a hard negative");
    });
    labels.sideSwitches.forEach((marker, index) => {
      if (
        !Number.isFinite(marker.time) ||
        marker.time < 0 ||
        marker.time > labels.recording.durationSeconds ||
        (index > 0 && marker.time <= labels.sideSwitches[index - 1].time)
      ) {
        issues.push("Side-switch points must be in range and strictly ordered");
      }
    });
    const geometry = labels.recording.courtGeometry;
    if (geometry) {
      const missingCorners = ["nearLeft", "nearRight", "farLeft", "farRight"].filter(
        (id) => !courtPoint(geometry, id as CourtAnchorId),
      );
      if (missingCorners.length > 0) issues.push("Finish all four named court corners");
      if (!!geometry.netAnchors?.left !== !!geometry.netAnchors?.right) {
        issues.push("Mark both net anchors or clear the partial pair");
      }
      if (!!geometry.serviceZoneAnchors?.near !== !!geometry.serviceZoneAnchors?.far) {
        issues.push("Mark both service-zone anchors or clear the partial pair");
      }
    }
    labels.rallies.forEach((row, index) => {
      if (
        row.receiverReactionTime !== undefined &&
        (row.receiverReactionTime < row.start ||
          row.receiverReactionTime > Math.min(row.end, row.start + 5))
      ) {
        issues.push(`Rally ${index + 1} receiver reaction must be within 5s of its start`);
      }
      if (
        row.collectiveStandDownTime !== undefined &&
        (row.collectiveStandDownTime < Math.max(row.start, row.end - 5) ||
          row.collectiveStandDownTime >
            Math.min(labels.recording.durationSeconds, row.end + 5))
      ) {
        issues.push(`Rally ${index + 1} stand-down must be within 5s of its end`);
      }
      for (const confidence of [row.startConfidence, row.endConfidence]) {
        if (confidence !== undefined && (confidence < 0 || confidence > 1)) {
          issues.push(`Rally ${index + 1} confidence must be between 0 and 1`);
        }
      }
      const seenTracklets = new Set<string>();
      (row.playerTracklets ?? []).forEach((tracklet) => {
        const key = `${tracklet.window}:${tracklet.trackId}`;
        if (seenTracklets.has(key)) {
          issues.push(`Rally ${index + 1} has duplicate player track ${key}`);
        }
        seenTracklets.add(key);
        const bounds = playerWindowBounds(
          row,
          tracklet.window,
          labels.recording.durationSeconds,
        );
        tracklet.observations.forEach((observation, observationIndex) => {
          if (
            observation.time < bounds.start ||
            observation.time > bounds.end ||
            (observationIndex > 0 &&
              observation.time <= tracklet.observations[observationIndex - 1].time)
          ) {
            issues.push(
              `Rally ${index + 1} player track ${tracklet.trackId} has an out-of-window or unordered frame`,
            );
          }
        });
      });
    });
    return [...new Set(issues)];
  }, [
    ignoredStart,
    labels,
    negativeStart,
    rallyStart,
    touchingRallyIndexes,
    videoDuration,
    videoFilename,
    videoUrl,
  ]);

  function markChanged(document: LabelDocument): LabelDocument {
    return {
      ...document,
      annotation: {
        ...document.annotation,
        status: "in-progress",
        reviewedAt: null,
      },
    };
  }

  async function loadTask(file: File | undefined) {
    if (!file) return;
    preparedRequestRef.current?.abort();
    pendingResumeSecondsRef.current = null;
    setSelectedPreparedTask("");
    setLastSavedAt(null);
    setError(null);
    try {
      const document = parseLabelDocument(JSON.parse(await file.text()));
      setLabels(document);
      setProductionReference(null);
      setExperimentReferences([]);
      setSolReferenceRallies([]);
      setRallyStart(null);
      setIgnoredStart(null);
      setNegativeStart(null);
      setCourtAnchor(null);
      setTrackletRallyIndex(null);
      setTrackletCapture(null);
      setTrackletBoxDrag(null);
      setCurrentTime(0);
      setMessage(
        document.annotation.status === "not-started"
          ? "Task loaded. Select the matching proxy video."
          : `Resumed ${document.rallies.length} rally labels.`,
      );
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load label document");
    }
  }

  function loadVideo(file: File | undefined) {
    if (!file) return;
    pendingResumeSecondsRef.current = null;
    setError(null);
    const nextUrl = URL.createObjectURL(file);
    setVideoUrl(nextUrl);
    setVideoFilename(file.name);
    setVideoDuration(null);
    setMessage(`Loaded local video ${file.name}. Nothing is uploaded.`);
  }

  async function loadPreparedTask(id: string, resumeSeconds: number | null = null) {
    setSelectedPreparedTask(id);
    if (!id) return;
    pendingResumeSecondsRef.current =
      resumeSeconds !== null && Number.isFinite(resumeSeconds) && resumeSeconds >= 0
        ? resumeSeconds
        : null;
    preparedRequestRef.current?.abort();
    const controller = new AbortController();
    preparedRequestRef.current = controller;
    setError(null);
    setPreparedTasksLoading(true);
    try {
      const taskUrl = `/api/labeling/tasks/${encodeURIComponent(id)}`;
      const [response, referencesResponse] = await Promise.all([
        fetch(taskUrl, {
          cache: "no-store",
          signal: controller.signal,
        }),
        fetch(`${taskUrl}/references`, {
          cache: "no-store",
          signal: controller.signal,
        }).catch(() => null),
      ]);
      if (!response.ok) throw new Error("Could not load the selected prepared task");
      const batch = response.headers.get("X-VolleyCut-Batch");
      const documentSource = response.headers.get("X-VolleyCut-Document-Source");
      const savedAt = response.headers.get("X-VolleyCut-Saved-At");
      const document = parseLabelDocument(await response.json());
      let referenceRallies: RallyLabel[] = [];
      let nextProductionReference: ModelReference | null = null;
      let nextExperimentReferences: ModelReference[] = [];
      if (referencesResponse?.ok) {
        const references = (await referencesResponse.json()) as {
          production?: Partial<ModelReference> | null;
          experiments?: Array<Partial<ModelReference>>;
          sol?: { rallies?: RallyLabel[] } | null;
        };
        if (
          typeof references.production?.modelId === "string" &&
          typeof references.production.modelLabel === "string" &&
          Array.isArray(references.production.rallies)
        ) {
          nextProductionReference = references.production as ModelReference;
        }
        if (Array.isArray(references.experiments)) {
          nextExperimentReferences = references.experiments.filter(
            (reference): reference is ModelReference =>
              typeof reference.modelId === "string" &&
              typeof reference.modelLabel === "string" &&
              Array.isArray(reference.rallies),
          );
        }
        if (Array.isArray(references.sol?.rallies)) {
          referenceRallies = references.sol.rallies;
        }
      }
      setLabels(document);
      setProductionReference(nextProductionReference);
      setExperimentReferences(nextExperimentReferences);
      setSolReferenceRallies(referenceRallies);
      setVideoUrl(`/api/labeling/tasks/${encodeURIComponent(id)}/video`);
      setVideoFilename(document.recording.videoFilename);
      setVideoDuration(null);
      setCurrentTime(pendingResumeSecondsRef.current ?? 0);
      setRallyStart(null);
      setIgnoredStart(null);
      setNegativeStart(null);
      setCourtAnchor(null);
      setTrackletRallyIndex(null);
      setTrackletCapture(null);
      setTrackletBoxDrag(null);
      setLastSavedAt(savedAt);
      if (batch === "full" || batch === "pilot") setSelectedBatch(batch);
      setMessage(
        documentSource === "draft"
          ? `Resumed the NAS draft for ${document.recording.id} with ${document.rallies.length} rallies.`
          : documentSource === "completed"
            ? `Loaded the completed human labels for ${document.recording.id} with ${document.rallies.length} rallies. Any new save creates an editable NAS draft without changing the completed source.`
          : documentSource === "production-model"
            ? `Loaded ${document.rallies.length} editable predictions from the production model for ${document.recording.id}. Sol is shown below as a read-only reference.`
          : documentSource === "prelabel"
            ? `Loaded ${document.rallies.length} unvalidated GPT-5.6 Sol rally candidates for ${document.recording.id}. Review every boundary before completing.`
          : `Loaded ${document.recording.id} and its matching NAS proxy. No local file selection needed.`,
      );
    } catch (loadError) {
      if (!controller.signal.aborted) {
        pendingResumeSecondsRef.current = null;
        setError(loadError instanceof Error ? loadError.message : "Could not load prepared task");
      }
    } finally {
      if (!controller.signal.aborted) setPreparedTasksLoading(false);
    }
  }

  function persistPlaybackPosition(video: HTMLVideoElement, force = false) {
    if (
      !selectedPreparedTask ||
      labels?.recording.id !== selectedPreparedTask ||
      !videoUrl?.startsWith("/api/labeling/tasks/") ||
      !Number.isFinite(video.currentTime) ||
      video.currentTime < 0
    ) {
      return;
    }
    const previous = lastPersistedPlaybackRef.current;
    if (
      !force &&
      previous?.taskId === selectedPreparedTask &&
      Math.abs(previous.time - video.currentTime) < 0.25
    ) {
      return;
    }
    const resume: PlaybackResume = {
      version: 1,
      taskId: selectedPreparedTask,
      time: roundTime(video.currentTime),
    };
    try {
      window.localStorage.setItem(playbackResumeKey, JSON.stringify(resume));
      lastPersistedPlaybackRef.current = resume;
    } catch {
      // Browsers can deny local storage; labeling and NAS draft saves still work.
    }
  }

  function handleLoadedMetadata(video: HTMLVideoElement) {
    setVideoDuration(video.duration);
    const resumeSeconds = pendingResumeSecondsRef.current;
    pendingResumeSecondsRef.current = null;
    if (resumeSeconds !== null) {
      video.currentTime = Math.max(0, Math.min(video.duration || Infinity, resumeSeconds));
      setMessage(
        `Resumed ${labels?.recording.id ?? "prepared task"} at ${formatPreciseTime(video.currentTime)}.`,
      );
    }
    setCurrentTime(video.currentTime);
    persistPlaybackPosition(video, true);
  }

  async function saveDraftDirectly() {
    if (!labels) return;
    const preparedTask = preparedTasks.find((task) => task.id === labels.recording.id);
    if (!preparedTask) {
      setError("Direct save is available only for a prepared NAS task.");
      return;
    }
    const draft: LabelDocument = {
      ...labels,
      annotation: {
        ...labels.annotation,
        status: "in-progress",
        reviewedAt: null,
      },
    };
    setError(null);
    setSavingDraft(true);
    try {
      const response = await fetch(
        `/api/labeling/tasks/${encodeURIComponent(preparedTask.id)}/draft`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(draft),
        },
      );
      const result = (await response.json()) as {
        batch?: LabelingBatch;
        error?: string;
        savedAt?: string;
      };
      if (!response.ok || !result.savedAt) {
        throw new Error(result.error ?? "The draft could not be saved");
      }
      setLabels(draft);
      setLastSavedAt(result.savedAt);
      setPreparedTasks((current) =>
        current.map((task) =>
          task.id === preparedTask.id
            ? {
                ...task,
                annotationStatus: "in-progress",
                documentSource: "draft",
                rallyCount: draft.rallies.length,
                savedAt: result.savedAt ?? null,
              }
            : task,
        ),
      );
      if (!preparedTask.savedAt) {
        setBatchSummary((current) => ({
          ...current,
          [preparedTask.batch]: {
            ...current[preparedTask.batch],
            saved: current[preparedTask.batch].saved + 1,
          },
        }));
      }
      setMessage(
        `${preparedTask.batch === "full" ? "Full-corpus" : "Pilot"} draft saved directly to the NAS at ${new Date(result.savedAt).toLocaleTimeString()}.`,
      );
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "The draft could not be saved");
    } finally {
      setSavingDraft(false);
    }
  }

  function seek(seconds: number) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.min(video.duration || Infinity, Math.max(0, video.currentTime + seconds));
    setCurrentTime(video.currentTime);
    persistPlaybackPosition(video, true);
  }

  function seekTo(seconds: number) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, Math.min(video.duration || Infinity, seconds));
    setCurrentTime(video.currentTime);
    persistPlaybackPosition(video, true);
    video.focus();
  }

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play();
    else video.pause();
  }

  function selectCourtAnchor(id: CourtAnchorId) {
    videoRef.current?.pause();
    setCourtAnchor(id);
    setMessage(`Court mode: click ${courtAnchorSpecs.find((item) => item.id === id)?.label.toLowerCase()} on the paused frame.`);
  }

  function updateCourtPoint(id: CourtAnchorId, point: NormalizedPoint | undefined) {
    if (!labels) return;
    const courtGeometry = setCourtPoint(labels.recording.courtGeometry, id, point);
    setLabels(
      markChanged({
        ...labels,
        recording: { ...labels.recording, courtGeometry },
      }),
    );
  }

  function captureCourtPoint(event: React.MouseEvent<HTMLButtonElement>) {
    if (!labels || !courtAnchor) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const point = {
      x: Math.round(((event.clientX - bounds.left) / bounds.width) * 1_000_000) / 1_000_000,
      y: Math.round(((event.clientY - bounds.top) / bounds.height) * 1_000_000) / 1_000_000,
    };
    updateCourtPoint(courtAnchor, point);
    const currentIndex = courtAnchorSpecs.findIndex((item) => item.id === courtAnchor);
    const next = courtAnchorSpecs
      .slice(currentIndex + 1)
      .find((item) => !courtPoint(labels.recording.courtGeometry, item.id));
    setCourtAnchor(next?.id ?? null);
    setMessage(
      next
        ? `Saved ${courtAnchorSpecs[currentIndex].label}. Click ${next.label.toLowerCase()}, or finish after the four corners.`
        : "Saved all eight court anchors.",
    );
  }

  function startTrackletCapture(mode: TrackletCaptureMode) {
    if (!labels || !activeTrackletRally || activeTrackletRallyIndex < 0) {
      setError("Choose a labeled rally before adding anonymous player tracks.");
      return;
    }
    const normalizedId = trackletId.trim();
    if (!/^[A-Z]{0,2}[0-9]{1,3}$/.test(normalizedId)) {
      setError("Use an anonymous track token such as P1 or A02—not a name.");
      return;
    }
    setTrackletId(normalizedId);
    setTrackletRallyIndex(activeTrackletRallyIndex);
    setCourtAnchor(null);
    videoRef.current?.pause();
    setTrackletCapture(mode);
    setTrackletBoxDrag(null);
    setError(null);
    setMessage(
      mode === "footpoint"
        ? `Player mode: click ${normalizedId}'s feet at each chosen frame.`
        : `Player mode: drag a box around ${normalizedId} at each chosen frame.`,
    );
  }

  function addPlayerObservation(
    geometry: Pick<PlayerTrackletObservation, "footpoint" | "box">,
  ) {
    if (!labels || !activeTrackletRally || activeTrackletRallyIndex < 0) return;
    const time = roundTime(videoRef.current?.currentTime ?? currentTime);
    const bounds = playerWindowBounds(
      activeTrackletRally,
      trackletWindow,
      labels.recording.durationSeconds,
    );
    if (time < bounds.start || time > bounds.end) {
      setError(
        `${trackletWindow === "serve" ? "Serve" : "Rally-end"} observations for rally ${activeTrackletRallyIndex + 1} must stay between ${formatPreciseTime(bounds.start)} and ${formatPreciseTime(bounds.end)}.`,
      );
      return;
    }
    const normalizedId = trackletId.trim();
    const observation: PlayerTrackletObservation = {
      time,
      ...geometry,
      ...(trackletState ? { state: trackletState } : {}),
    };
    const tracklets = [...(activeTrackletRally.playerTracklets ?? [])];
    const trackletIndex = tracklets.findIndex(
      (tracklet) =>
        tracklet.window === trackletWindow && tracklet.trackId === normalizedId,
    );
    if (trackletIndex >= 0) {
      const existing = tracklets[trackletIndex];
      const observationIndex = existing.observations.findIndex(
        (row) => Math.abs(row.time - time) <= trackletFrameEpsilon,
      );
      const observations = [...existing.observations];
      if (observationIndex >= 0) {
        observations[observationIndex] = {
          ...observations[observationIndex],
          ...observation,
        };
      } else {
        observations.push(observation);
      }
      observations.sort((left, right) => left.time - right.time);
      tracklets[trackletIndex] = {
        ...existing,
        team: trackletTeam,
        courtSide: trackletCourtSide,
        observations,
      };
    } else {
      tracklets.push({
        trackId: normalizedId,
        window: trackletWindow,
        team: trackletTeam,
        courtSide: trackletCourtSide,
        observations: [observation],
      });
    }
    updateRally(activeTrackletRallyIndex, { playerTracklets: tracklets });
    setError(null);
    setMessage(
      `Saved ${normalizedId} at ${formatPreciseTime(time)} for rally ${activeTrackletRallyIndex + 1}'s ${trackletWindow} window.`,
    );
  }

  function captureTrackletClick(event: React.MouseEvent<HTMLButtonElement>) {
    if (trackletCapture !== "footpoint") return;
    addPlayerObservation({ footpoint: normalizedPointer(event) });
  }

  function beginTrackletBox(event: React.PointerEvent<HTMLButtonElement>) {
    if (trackletCapture !== "box") return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = normalizedPointer(event);
    setTrackletBoxDrag({ start: point, current: point });
  }

  function moveTrackletBox(event: React.PointerEvent<HTMLButtonElement>) {
    if (trackletCapture !== "box" || !trackletBoxDrag) return;
    setTrackletBoxDrag({ ...trackletBoxDrag, current: normalizedPointer(event) });
  }

  function finishTrackletBox(event: React.PointerEvent<HTMLButtonElement>) {
    if (trackletCapture !== "box" || !trackletBoxDrag) return;
    event.preventDefault();
    const end = normalizedPointer(event);
    const box: NormalizedBox = {
      x: Math.min(trackletBoxDrag.start.x, end.x),
      y: Math.min(trackletBoxDrag.start.y, end.y),
      width: Math.abs(trackletBoxDrag.start.x - end.x),
      height: Math.abs(trackletBoxDrag.start.y - end.y),
    };
    setTrackletBoxDrag(null);
    if (box.width < 0.003 || box.height < 0.003) {
      setError("Drag a visible player box rather than clicking a single point.");
      return;
    }
    addPlayerObservation({ box });
  }

  function removePlayerTracklet(rallyIndex: number, trackletIndex: number) {
    const rally = labels?.rallies[rallyIndex];
    if (!rally) return;
    const next = (rally.playerTracklets ?? []).filter((_, index) => index !== trackletIndex);
    updateRally(rallyIndex, { playerTracklets: next.length > 0 ? next : undefined });
  }

  function removePlayerObservation(
    rallyIndex: number,
    trackletIndex: number,
    observationIndex: number,
  ) {
    const rally = labels?.rallies[rallyIndex];
    const tracklet = rally?.playerTracklets?.[trackletIndex];
    if (!rally || !tracklet) return;
    const observations = tracklet.observations.filter((_, index) => index !== observationIndex);
    if (observations.length === 0) {
      removePlayerTracklet(rallyIndex, trackletIndex);
      return;
    }
    const tracklets = [...(rally.playerTracklets ?? [])];
    tracklets[trackletIndex] = { ...tracklet, observations };
    updateRally(rallyIndex, { playerTracklets: tracklets });
  }

  function beginRally() {
    if (!labels || !videoRef.current) return;
    const time = roundTime(videoRef.current.currentTime);
    const existingIndex = labels.rallies.findIndex(
      (row) => row.start < time && time < row.end,
    );
    if (existingIndex >= 0) {
      updateRally(existingIndex, { start: time });
      setMessage(
        `Moved rally ${existingIndex + 1} start to ${formatPreciseTime(time)}.`,
      );
      return;
    }
    setRallyStart(time);
    setMessage("Rally start marked. Seek to the first instant live play has ended, then press E.");
  }

  function moveRallyEnd(index: number, end: number): boolean {
    if (!labels) return false;
    const rally = labels.rallies[index];
    if (!rally || end <= rally.start) {
      setError("Rally end must be after its serve contact.");
      return false;
    }
    if (end > labels.recording.durationSeconds) {
      setError("Rally end cannot exceed the video duration.");
      return false;
    }
    const otherRallies = labels.rallies.filter((_, rowIndex) => rowIndex !== index);
    if (
      overlaps(rally.start, end, otherRallies) ||
      overlaps(rally.start, end, labels.ignoredIntervals) ||
      overlaps(rally.start, end, labels.hardNegatives)
    ) {
      setError(
        "That end would overlap the next rally, an ignored span, or a hard negative.",
      );
      return false;
    }
    setError(null);
    updateRally(index, { end });
    setMessage(`Moved rally ${index + 1} end to ${formatPreciseTime(end)}.`);
    return true;
  }

  function splitRallyFromOpenStart(index: number, start: number, split: number): boolean {
    if (!labels) return false;
    const existing = labels.rallies[index];
    if (!existing || split <= start || split >= existing.end) {
      setError("The split must be after the new start and before the existing rally end.");
      return false;
    }
    const otherRallies = labels.rallies.filter((_, rowIndex) => rowIndex !== index);
    if (
      overlaps(start, existing.end, otherRallies) ||
      overlaps(start, existing.end, labels.ignoredIntervals) ||
      overlaps(start, existing.end, labels.hardNegatives)
    ) {
      setError(
        "The split rally would overlap another rally, an ignored span, or a hard negative.",
      );
      return false;
    }
    const firstBase: RallyLabel = { ...existing, start, end: split };
    const remainderBase: RallyLabel = { ...existing, start: split, end: existing.end };
    const first: RallyLabel = {
      ...firstBase,
      playerTracklets: trackletsInsideRallyBounds(
        existing.playerTracklets,
        firstBase,
        labels.recording.durationSeconds,
      ),
    };
    const remainder: RallyLabel = {
      ...remainderBase,
      playerTracklets: trackletsInsideRallyBounds(
        existing.playerTracklets,
        remainderBase,
        labels.recording.durationSeconds,
      ),
    };
    const rallies = [...otherRallies, first, remainder].sort(
      (left, right) => left.start - right.start,
    );
    setError(null);
    setLabels(markChanged({ ...labels, rallies }));
    setRallyStart(null);
    setMessage(
      `Split rally ${index + 1} at ${formatPreciseTime(split)}: the first part now starts at ${formatPreciseTime(start)}, and the remainder ends at ${formatPreciseTime(existing.end)}.`,
    );
    return true;
  }

  function finishRally() {
    if (!labels || !videoRef.current) return;
    const end = roundTime(videoRef.current.currentTime);
    if (rallyStart === null) {
      if (selectedRallyIndex >= 0) {
        moveRallyEnd(selectedRallyIndex, end);
        return;
      }
      if (previousRallyIndex >= 0) moveRallyEnd(previousRallyIndex, end);
      return;
    }
    const containingRallyIndex = labels.rallies.findIndex(
      (row) => row.start < end && end < row.end,
    );
    if (containingRallyIndex >= 0) {
      splitRallyFromOpenStart(containingRallyIndex, rallyStart, end);
      return;
    }
    if (!addInterval(rallyStart, end, "rally")) return;
    setRallyStart(null);
  }

  function toggleIgnored() {
    if (!labels || !videoRef.current) return;
    const time = roundTime(videoRef.current.currentTime);
    if (ignoredStart === null) {
      setIgnoredStart(time);
      setMessage("Ignore start marked. Seek to the end of the ambiguous/censored span and press ].");
      return;
    }
    if (addInterval(ignoredStart, time, "ignored")) setIgnoredStart(null);
  }

  function toggleNegative() {
    if (!labels || !videoRef.current) return;
    const time = roundTime(videoRef.current.currentTime);
    if (negativeStart === null) {
      setNegativeStart(time);
      setMessage("Hard-negative start marked. Seek to its end and press H again.");
      return;
    }
    if (addInterval(negativeStart, time, "negative")) setNegativeStart(null);
  }

  function addSideSwitch() {
    if (!labels || !videoRef.current) return;
    const time = roundTime(videoRef.current.currentTime);
    if (labels.sideSwitches.some((marker) => marker.time === time)) {
      setError("A side switch is already marked at this timestamp.");
      return;
    }
    if (time > labels.recording.durationSeconds) {
      setError("A side switch cannot be marked beyond the task duration.");
      return;
    }
    const sideSwitches = [...labels.sideSwitches, { time }].sort(
      (left, right) => left.time - right.time,
    );
    setError(null);
    setLabels(markChanged({ ...labels, sideSwitches }));
    setMessage(`Marked a side switch at ${formatPreciseTime(time)}.`);
  }

  function addInterval(start: number, end: number, kind: IntervalKind): boolean {
    if (!labels) return false;
    setError(null);
    if (end <= start) {
      setError("Interval end must be after its start.");
      return false;
    }
    if (overlaps(start, end, allRows)) {
      setError("That interval overlaps an existing rally, ignored span, or hard negative.");
      return false;
    }
    const sortRows = <T extends { start: number }>(rows: T[]) =>
      [...rows].sort((left, right) => left.start - right.start);
    if (kind === "rally") {
      const rally: RallyLabel = { start, end, tags: [] };
      setLabels(markChanged({ ...labels, rallies: sortRows([...labels.rallies, rally]) }));
      setMessage(`Added rally ${formatPreciseTime(start)}–${formatPreciseTime(end)}.`);
    } else if (kind === "ignored") {
      const ignored: IgnoredInterval = { start, end, reason: "partial-rally" };
      setLabels(
        markChanged({ ...labels, ignoredIntervals: sortRows([...labels.ignoredIntervals, ignored]) }),
      );
      setMessage("Added an ignored interval. These samples will not be fitted or scored.");
    } else {
      const negative: HardNegative = { start, end, category: negativeCategory };
      setLabels(
        markChanged({ ...labels, hardNegatives: sortRows([...labels.hardNegatives, negative]) }),
      );
      setMessage("Added an optional hard-negative example.");
    }
    return true;
  }

  function cancelMarker() {
    setRallyStart(null);
    setIgnoredStart(null);
    setNegativeStart(null);
    setMessage("Open marker cancelled.");
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, button, [contenteditable='true']")) return;
      const key = event.key.toLowerCase();
      if (key === " ") {
        event.preventDefault();
        togglePlayback();
      } else if (key === "s") beginRally();
      else if (key === "e") finishRally();
      else if (key === "[") toggleIgnored();
      else if (key === "]" && ignoredStart !== null) toggleIgnored();
      else if (key === "h") toggleNegative();
      else if (key === "x") addSideSwitch();
      else if (key === "escape") cancelMarker();
      else if ((key === "delete" || key === "backspace") && selectedRallyIndex >= 0) {
        event.preventDefault();
        removeSelectedRally();
      }
      else if (key === "j") seek(event.shiftKey ? -1 : -0.1);
      else if (key === "k") seek(event.shiftKey ? 1 : 0.1);
      else if (key === "arrowleft") {
        event.preventDefault();
        seek(-1);
      } else if (key === "arrowright") {
        event.preventDefault();
        seek(1);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  function updateRally(index: number, patch: Partial<RallyLabel>) {
    if (!labels) return;
    const existing = labels.rallies[index];
    if (!existing) return;
    const candidate = { ...existing, ...patch };
    if (patch.start !== undefined || patch.end !== undefined) {
      const invalidTracklet = (candidate.playerTracklets ?? []).find((tracklet) => {
        const bounds = playerWindowBounds(
          candidate,
          tracklet.window,
          labels.recording.durationSeconds,
        );
        return tracklet.observations.some(
          (observation) => observation.time < bounds.start || observation.time > bounds.end,
        );
      });
      if (invalidTracklet) {
        setError(
          `Move or delete ${invalidTracklet.trackId}'s ${invalidTracklet.window} observations before changing this boundary.`,
        );
        return;
      }
    }
    const rallies = labels.rallies.map((row, rowIndex) =>
      rowIndex === index ? candidate : row,
    );
    setLabels(markChanged({ ...labels, rallies }));
  }

  function updateRallyClassification(index: number, classification: string) {
    const rally = labels?.rallies[index];
    if (!rally) return;
    updateRally(index, {
      tags: [
        ...rally.tags.filter((tag) => !editableRallyTags.has(tag)),
        ...(classification ? [classification] : []),
      ],
    });
  }

  function updateIgnored(index: number, patch: Partial<IgnoredInterval>) {
    if (!labels) return;
    const ignoredIntervals = labels.ignoredIntervals.map((row, rowIndex) =>
      rowIndex === index ? { ...row, ...patch } : row,
    );
    setLabels(markChanged({ ...labels, ignoredIntervals }));
  }

  function updateNegative(index: number, patch: Partial<HardNegative>) {
    if (!labels) return;
    const hardNegatives = labels.hardNegatives.map((row, rowIndex) =>
      rowIndex === index ? { ...row, ...patch } : row,
    );
    setLabels(markChanged({ ...labels, hardNegatives }));
  }

  function updateSideSwitch(index: number, patch: Partial<SideSwitch>) {
    if (!labels) return;
    const sideSwitches = labels.sideSwitches.map((marker, markerIndex) =>
      markerIndex === index ? { ...marker, ...patch } : marker,
    );
    setLabels(markChanged({ ...labels, sideSwitches }));
  }

  function removeSideSwitch(index: number) {
    if (!labels) return;
    setLabels(
      markChanged({
        ...labels,
        sideSwitches: labels.sideSwitches.filter((_, markerIndex) => markerIndex !== index),
      }),
    );
    setMessage(`Deleted side-switch marker ${index + 1}.`);
  }

  function removeSelectedRally() {
    if (!labels || selectedRallyIndex < 0) return;
    const rally = labels.rallies[selectedRallyIndex];
    setLabels(
      markChanged({
        ...labels,
        rallies: labels.rallies.filter((_, index) => index !== selectedRallyIndex),
      }),
    );
    setError(null);
    setMessage(
      `Deleted rally ${selectedRallyIndex + 1} (${formatPreciseTime(rally.start)}–${formatPreciseTime(rally.end)}).`,
    );
  }

  function removeRow(kind: IntervalKind, index: number) {
    if (!labels) return;
    if (kind === "rally") {
      setLabels(markChanged({ ...labels, rallies: labels.rallies.filter((_, row) => row !== index) }));
    } else if (kind === "ignored") {
      setLabels(
        markChanged({
          ...labels,
          ignoredIntervals: labels.ignoredIntervals.filter((_, row) => row !== index),
        }),
      );
    } else {
      setLabels(
        markChanged({
          ...labels,
          hardNegatives: labels.hardNegatives.filter((_, row) => row !== index),
        }),
      );
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <Brand className={styles.brand} label="LABEL" priority />
        <div className={styles.local}>Local workspace · no cloud upload</div>
      </header>

      <section className={styles.intro}>
        <div>
          <p className={styles.eyebrow}>GOLD-LABEL WORKSTATION</p>
          <h1>Mark the ball live.<br /><em>Teach the cut.</em></h1>
        </div>
        <div className={styles.policy}>
          <strong>Boundary contract</strong>
          <p><b>Start:</b> the instant the server contacts the ball.</p>
          <p><b>End:</b> the first instant live play has ended—not the celebration or walk back.</p>
          <p>Short aces and service faults still count as rallies. Mark uncertain/partial footage as ignored.</p>
        </div>
      </section>

      <section className={styles.loaders}>
        <label className={styles.preparedTask}>
          <span>Prepared NAS task · recommended</span>
          <select
            aria-label="Labeling batch"
            value={selectedBatch}
            disabled={preparedTasksLoading}
            onChange={(event) => {
              setSelectedBatch(event.target.value as LabelingBatch);
              setSelectedPreparedTask("");
            }}
          >
            <option value="full">
              Full corpus · {batchSummary.full.ready}/{batchSummary.full.total} ready · {batchSummary.full.saved} saved
              {batchSummary.full.prelabeled > 0 ? ` · ${batchSummary.full.prelabeled} AI starting points` : ""}
            </option>
            <option value="pilot">
              Pilot · {batchSummary.pilot.ready}/{batchSummary.pilot.total} ready · {batchSummary.pilot.saved} saved
            </option>
          </select>
          <select
            aria-label="Prepared labeling task"
            value={selectedPreparedTask}
            disabled={preparedTasksLoading || tasksForSelectedBatch.length === 0}
            onChange={(event) => void loadPreparedTask(event.target.value)}
          >
            <option value="">
              {preparedTasksLoading
                ? "Loading prepared tasks…"
                : tasksForSelectedBatch.length === 0
                  ? "Waiting for this batch to be prepared…"
                  : "Choose a task and video…"}
            </option>
            {tasksForSelectedBatch.map((task) => (
              <option key={task.id} value={task.id}>
                {task.priority}. {task.environment} · {task.originalFilename} · {formatPreciseTime(task.durationSeconds)} · {task.savedAt ? `${task.rallyCount} rallies saved` : task.documentSource === "production-model" ? `${task.rallyCount} production-model rallies to review` : task.documentSource === "prelabel" ? `${task.rallyCount} Sol rallies to review` : "not started"}
              </option>
            ))}
          </select>
          <small>Loads both files from the NAS. The list refreshes as full proxies finish.</small>
        </label>
        <label>
          <span>Local fallback · task or saved draft</span>
          <input type="file" accept="application/json,.json" onChange={(event) => void loadTask(event.target.files?.[0])} />
        </label>
        <label>
          <span>Local fallback · matching proxy</span>
          <input type="file" accept="video/mp4,video/*" onChange={(event) => loadVideo(event.target.files?.[0])} />
        </label>
        <div className={styles.loaded}>
          <span>Batch</span>
          <strong>{selectedPreparedSummary?.batch ?? "Local fallback"}</strong>
          <span>Starting point</span>
          <strong>
            {selectedPreparedSummary?.documentSource === "production-model"
              ? "Production model · editable labels"
              : selectedPreparedSummary?.documentSource === "prelabel"
              ? `Unvalidated GPT-5.6 Sol prelabel · ${labels?.prelabel?.ambiguities.length ?? 0} ambiguities`
              : selectedPreparedSummary?.documentSource === "completed"
                ? "Completed human labels · editable copy"
              : selectedPreparedSummary?.documentSource === "draft"
                ? labels?.prelabel?.candidateFile.startsWith("model-")
                  ? "Human-saved NAS draft · started from production model"
                  : labels?.prelabel
                    ? "Human-saved NAS draft · started from Sol prelabel"
                  : "Human-saved NAS draft"
                : "Blank task"}
          </strong>
          <span>Original source</span>
          <strong>{selectedPreparedSummary?.originalFilename ?? "Local task or draft"}</strong>
          <span>Annotation proxy</span>
          <strong>{labels?.recording.videoFilename ?? "Load a task first"}</strong>
        </div>
      </section>

      {error && <div className={styles.error}>{error}</div>}
      <div className={styles.message}>{message}</div>

      <section className={styles.workbench}>
        <div className={styles.videoColumn}>
          <div className={styles.videoWrap}>
            {videoUrl ? (
              <div className={styles.videoStage}>
                <video
                  key={videoUrl}
                  ref={videoRef}
                  src={videoUrl}
                  controls={courtAnchor === null && trackletCapture === null}
                  preload="metadata"
                  onTimeUpdate={(event) => {
                    setCurrentTime(event.currentTarget.currentTime);
                    persistPlaybackPosition(event.currentTarget);
                  }}
                  onSeeked={(event) => persistPlaybackPosition(event.currentTarget, true)}
                  onPlay={(event) => persistPlaybackPosition(event.currentTarget, true)}
                  onPause={(event) => persistPlaybackPosition(event.currentTarget, true)}
                  onLoadedMetadata={(event) => handleLoadedMetadata(event.currentTarget)}
                />
                {labels?.recording.courtGeometry && (
                  <svg
                    className={styles.courtDrawing}
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                  >
                    {courtPolygon && <polyline points={courtPolygon} />}
                    {labels.recording.courtGeometry.netAnchors?.left &&
                      labels.recording.courtGeometry.netAnchors?.right && (
                        <line
                          x1={labels.recording.courtGeometry.netAnchors.left.x * 100}
                          y1={labels.recording.courtGeometry.netAnchors.left.y * 100}
                          x2={labels.recording.courtGeometry.netAnchors.right.x * 100}
                          y2={labels.recording.courtGeometry.netAnchors.right.y * 100}
                        />
                      )}
                    {courtAnchorSpecs.map((anchor) => {
                      const point = courtPoint(labels.recording.courtGeometry, anchor.id);
                      return point ? (
                        <circle
                          key={anchor.id}
                          cx={point.x * 100}
                          cy={point.y * 100}
                          r={anchor.optional ? 0.9 : 1.2}
                          className={anchor.optional ? styles.optionalCourtPoint : undefined}
                        />
                      ) : null;
                    })}
                  </svg>
                )}
                {(currentPlayerObservations.length > 0 || trackletBoxDrag) && (
                  <svg
                    className={styles.trackletDrawing}
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                  >
                    {currentPlayerObservations.map(({ tracklet, observation }) => (
                      <g
                        key={`${tracklet.window}-${tracklet.trackId}-${observation.time}`}
                        className={
                          tracklet.team === "team-a"
                            ? styles.trackletTeamA
                            : tracklet.team === "team-b"
                              ? styles.trackletTeamB
                              : styles.trackletUnknownTeam
                        }
                      >
                        {observation.box && (
                          <rect
                            x={observation.box.x * 100}
                            y={observation.box.y * 100}
                            width={observation.box.width * 100}
                            height={observation.box.height * 100}
                          />
                        )}
                        {observation.footpoint && (
                          <circle
                            cx={observation.footpoint.x * 100}
                            cy={observation.footpoint.y * 100}
                            r="1.1"
                          />
                        )}
                        <text
                          x={(observation.footpoint?.x ?? observation.box?.x ?? 0) * 100}
                          y={(observation.footpoint?.y ?? observation.box?.y ?? 0) * 100 - 1}
                        >
                          {tracklet.trackId}
                        </text>
                      </g>
                    ))}
                    {trackletBoxDrag && (
                      <rect
                        className={styles.trackletDraftBox}
                        x={Math.min(trackletBoxDrag.start.x, trackletBoxDrag.current.x) * 100}
                        y={Math.min(trackletBoxDrag.start.y, trackletBoxDrag.current.y) * 100}
                        width={Math.abs(trackletBoxDrag.start.x - trackletBoxDrag.current.x) * 100}
                        height={Math.abs(trackletBoxDrag.start.y - trackletBoxDrag.current.y) * 100}
                      />
                    )}
                  </svg>
                )}
                {courtAnchor && (
                  <button
                    type="button"
                    className={styles.courtClickLayer}
                    aria-label={`Click ${courtAnchorSpecs.find((item) => item.id === courtAnchor)?.label}`}
                    onClick={captureCourtPoint}
                  />
                )}
                {trackletCapture && (
                  <button
                    type="button"
                    className={styles.trackletClickLayer}
                    aria-label={
                      trackletCapture === "footpoint"
                        ? `Click ${trackletId} footpoint`
                        : `Draw ${trackletId} player box`
                    }
                    onClick={captureTrackletClick}
                    onPointerDown={beginTrackletBox}
                    onPointerMove={moveTrackletBox}
                    onPointerUp={finishTrackletBox}
                    onPointerCancel={() => setTrackletBoxDrag(null)}
                  />
                )}
              </div>
            ) : (
              <div className={styles.videoEmpty}>Select the proxy listed by the task.</div>
            )}
            <div className={styles.timecode}>{formatPreciseTime(currentTime)}</div>
          </div>

          <div className={styles.transport}>
            <button onClick={() => seek(-1)}>−1s <kbd>←</kbd></button>
            <button onClick={() => seek(-0.1)}>−0.1s <kbd>J</kbd></button>
            <button className={styles.playButton} onClick={togglePlayback}>Play / pause <kbd>Space</kbd></button>
            <button onClick={() => seek(0.1)}>+0.1s <kbd>K</kbd></button>
            <button onClick={() => seek(1)}>+1s <kbd>→</kbd></button>
          </div>

          <div className={styles.markers}>
            <button className={styles.start} onClick={beginRally} disabled={!labels || !videoUrl || rallyStart !== null}>
              {playheadInsideRally ? "Move rally start" : "Mark serve contact"} <kbd>S</kbd>
            </button>
            <button
              className={styles.end}
              onClick={finishRally}
              disabled={
                !labels ||
                !videoUrl ||
                (rallyStart === null && selectedRallyIndex < 0 && previousRallyIndex < 0)
              }
            >
              {rallyStart !== null
                ? selectedRallyIndex >= 0
                  ? "Split rally here"
                  : "Mark end of play"
                : selectedRallyIndex >= 0
                  ? "Move rally end"
                  : "Extend previous rally"} <kbd>E</kbd>
            </button>
            <button
              className={styles.deleteSelected}
              onClick={removeSelectedRally}
              disabled={selectedRallyIndex < 0}
            >
              Delete selected <kbd>Del</kbd>
            </button>
            <button onClick={toggleIgnored} disabled={!labels || !videoUrl}>
              {ignoredStart === null ? "Start ignored span" : "Finish ignored span"} <kbd>[ ]</kbd>
            </button>
            <button onClick={addSideSwitch} disabled={!labels || !videoUrl}>
              Mark side switch <kbd>X</kbd>
            </button>
            <button onClick={cancelMarker} disabled={rallyStart === null && ignoredStart === null && negativeStart === null}>
              Cancel <kbd>Esc</kbd>
            </button>
          </div>

          {labels && (
            <>
              <div className={styles.comparisonRailControls}>
                <div className={styles.comparisonRailLegend} aria-label="Final export rail legend">
                  <span data-tone="export">Final padded export</span>
                  <span data-tone="joined-gap">Joined short gap</span>
                  <span data-tone="missing">Missed human core</span>
                </div>
                <label htmlFor="label-join-gap">
                  Join gaps under
                  <output>{joinGapSeconds.toFixed(1)}s</output>
                  <input
                    id="label-join-gap"
                    aria-label="Join evaluation and export gaps shorter than"
                    type="range"
                    min="0"
                    max={MAX_JOIN_GAP_SECONDS}
                    step="0.5"
                    value={joinGapSeconds}
                    onChange={(event) => updateJoinGapSeconds(Number(event.currentTarget.value))}
                  />
                </label>
              </div>
              <RallyTimeline
              duration={labels.recording.durationSeconds}
              currentTime={currentTime}
              tracks={[
                {
                  id: "editable-rallies",
                  label: "Editable labels",
                  detail: `${labels.rallies.length} rallies · working set`,
                  active: true,
                  intervals: labels.rallies.map((row, index) => ({
                    id: `rally-${index}`,
                    start: row.start,
                    end: row.end,
                    title: `Rally ${index + 1}`,
                    tone: labels.prelabel?.candidateFile.startsWith("model-")
                      ? ("model" as const)
                      : ("gold" as const),
                    })),
                },
                ...referenceComparisons.map((comparison) => ({
                  id: `${comparison.reference.modelId}-${comparison.paddingSeconds}s`,
                  label: `${comparison.reference.baseline ? "Production" : comparison.reference.modelLabel} · ${comparison.paddingSeconds}s`,
                  detail: `${comparison.reference.modelId} · ${comparison.reference.rallies.length} core rallies · ${comparison.paddingSeconds}s pad · < ${joinGapSeconds}s joins`,
                  title: `${comparison.reference.modelLabel}. ${comparison.reference.description ?? "Read-only model inference"} Compared live with the editable labels at ${comparison.paddingSeconds} seconds before and after. Padded ranges with gaps strictly under ${joinGapSeconds} seconds are joined before scoring and export measurement.`,
                  summary: {
                    exportTime: formatPreciseTime(comparison.exportSeconds),
                    metricsLabel: `${comparison.paddingSeconds}s P_pad/R_core/F1`,
                    coreMetrics: `P ${metricPercent(comparison.precision)} · R ${metricPercent(comparison.recall)} · F1 ${metricPercent(comparison.f1)}`,
                  },
                  exportIntervals: comparison.exportRallies.map((rally, index) => ({
                    id: `${comparison.reference.modelId}-${comparison.paddingSeconds}s-export-${index + 1}`,
                    start: rally.start,
                    end: rally.end,
                    title: `${comparison.reference.modelLabel} · final ${comparison.paddingSeconds}s padded export · ${formatPreciseTime(rally.start)}–${formatPreciseTime(rally.end)}`,
                  })),
                  joinedGapIntervals: comparison.joinedGapRallies.map((gap, index) => ({
                    id: `${comparison.reference.modelId}-${comparison.paddingSeconds}s-visible-gap-${index + 1}`,
                    start: gap.start,
                    end: gap.end,
                    title: `${comparison.reference.modelLabel} · retained gap under ${joinGapSeconds}s · ${formatPreciseTime(gap.start)}–${formatPreciseTime(gap.end)}`,
                  })),
                  missingHumanIntervals: comparison.missingHumanSegments.map((segment, index) => ({
                    id: `${comparison.reference.modelId}-${comparison.paddingSeconds}s-missing-human-${index + 1}`,
                    start: segment.start,
                    end: segment.end,
                    title: `${comparison.reference.modelLabel} · missed unpadded human rally time · ${formatPreciseTime(segment.start)}–${formatPreciseTime(segment.end)}`,
                  })),
                  intervals: comparison.segments.map((segment) => ({
                    id: `${comparison.reference.modelId}-${comparison.paddingSeconds}s-${segment.id}`,
                    selectionId: null,
                    start: segment.start,
                    end: segment.end,
                    tone: `model-${segment.kind}` as const,
                    paddingOrigin: segment.paddingOrigin,
                    title: `${comparison.reference.modelLabel} · ${comparison.paddingSeconds}s padding · ${
                      segment.kind === "match"
                        ? "matches editable human live time"
                        : segment.kind === "added"
                          ? "predicted outside editable human live time"
                          : "editable human live time missed by the model"
                    } · ${formatPreciseTime(segment.start)}–${formatPreciseTime(segment.end)}`,
                  })),
                } satisfies TimelineTrack)),
                ...(solReferenceRallies.length
                  ? [{
                      id: "sol-reference",
                      label: "Sol reference",
                      detail: `${solReferenceRallies.length} rallies · read only`,
                      title:
                        "Blind Sol labels for reference only; edits apply to the working set above.",
                      intervals: solReferenceRallies.map((row, index) => ({
                        id: `sol-reference-${index}`,
                        selectionId: null,
                        start: row.start,
                        end: row.end,
                        title: `Sol reference ${index + 1}`,
                        tone: "sol" as const,
                      })),
                    } satisfies TimelineTrack]
                  : []),
                ...(labels.ignoredIntervals.length
                  ? [{
                      id: "ignored",
                      label: "Ignored",
                      detail: `${labels.ignoredIntervals.length} spans`,
                      intervals: labels.ignoredIntervals.map((row, index) => ({
                        id: `ignored-${index}`,
                        start: row.start,
                        end: row.end,
                        title: `Ignored ${index + 1}`,
                        tone: "ignored" as const,
                      })),
                    } satisfies TimelineTrack]
                  : []),
                ...(labels.hardNegatives.length
                  ? [{
                      id: "hard-negatives",
                      label: "Hard negatives",
                      detail: `${labels.hardNegatives.length} spans`,
                      intervals: labels.hardNegatives.map((row, index) => ({
                        id: `negative-${index}`,
                        start: row.start,
                        end: row.end,
                        title: `Hard negative ${index + 1}`,
                        tone: "negative" as const,
                      })),
                    } satisfies TimelineTrack]
                  : []),
              ]}
              markers={labels.sideSwitches.map((marker, index) => ({
                id: `side-switch-${index}`,
                time: marker.time,
                title: `Side switch ${index + 1}${marker.notes ? ` · ${marker.notes}` : ""}`,
              }))}
              selectedTrackId="editable-rallies"
              selectedIntervalId={
                selectedRallyIndex >= 0 ? `rally-${selectedRallyIndex}` : undefined
              }
              onSeek={(time) => seekTo(time)}
              ariaLabel="Editable rally labels with read-only model and Sol references"
            />
            </>
          )}
        </div>

        <aside className={styles.metadata}>
          <div className={styles.currentRallyCard}>
            <p className={styles.eyebrow}>
              {selectedRallyIndex >= 0 || startingRallyIndex >= 0
                ? "CURRENT RALLY"
                : sidebarRally
                  ? "PREVIOUS RALLY"
                  : "CURRENT RALLY"}
            </p>
            {sidebarRally ? (
              <>
                <div className={styles.currentRallyHeading}>
                  <strong>
                    R{String(sidebarRallyIndex + 1).padStart(3, "0")}
                    {sidebarRally.tags.includes("ai-prelabel") ? " AI" : ""}
                  </strong>
                  <span>{(sidebarRally.end - sidebarRally.start).toFixed(3)}s</span>
                </div>
                <div className={styles.currentRallyTimes}>
                  <button onClick={() => seekTo(sidebarRally.start)}>
                    {formatPreciseTime(sidebarRally.start)}
                  </button>
                  <span aria-hidden="true">→</span>
                  <button onClick={() => seekTo(sidebarRally.end)}>
                    {formatPreciseTime(sidebarRally.end)}
                  </button>
                </div>
                <label>
                  Classification
                  <select
                    aria-label="Current rally classification"
                    value={
                      sidebarRally.tags.find((tag) => editableRallyTags.has(tag)) ?? ""
                    }
                    onChange={(event) =>
                      updateRallyClassification(sidebarRallyIndex, event.target.value)
                    }
                  >
                    <option value="">Normal rally</option>
                    <option value="service-fault">Service fault</option>
                    <option value="ace">Ace / very short</option>
                    <option value="interrupted-replay">Interrupted / replayed</option>
                  </select>
                </label>
                <div className={styles.transitionPanel}>
                  <strong>Optional transition cues</strong>
                  <p>Use absolute video times. Reaction stays within 5s of serve; stand-down stays within 5s of the end.</p>
                  <label>
                    Receiver reaction time
                    <div className={styles.cueTime}>
                      <input
                        aria-label="Receiver reaction time"
                        type="number"
                        step="0.001"
                        value={sidebarRally.receiverReactionTime ?? ""}
                        onChange={(event) =>
                          updateRally(sidebarRallyIndex, {
                            receiverReactionTime: event.target.value
                              ? Number(event.target.value)
                              : undefined,
                          })
                        }
                      />
                      <button
                        type="button"
                        onClick={() => updateRally(sidebarRallyIndex, {
                          receiverReactionTime: roundTime(currentTime),
                        })}
                      >
                        Use playhead
                      </button>
                    </div>
                  </label>
                  <label>
                    Collective stand-down time
                    <div className={styles.cueTime}>
                      <input
                        aria-label="Collective stand-down time"
                        type="number"
                        step="0.001"
                        value={sidebarRally.collectiveStandDownTime ?? ""}
                        onChange={(event) =>
                          updateRally(sidebarRallyIndex, {
                            collectiveStandDownTime: event.target.value
                              ? Number(event.target.value)
                              : undefined,
                          })
                        }
                      />
                      <button
                        type="button"
                        onClick={() => updateRally(sidebarRallyIndex, {
                          collectiveStandDownTime: roundTime(currentTime),
                        })}
                      >
                        Use playhead
                      </button>
                    </div>
                  </label>
                  <label>
                    Terminal cue
                    <select
                      value={sidebarRally.terminalCue ?? ""}
                      onChange={(event) => updateRally(sidebarRallyIndex, {
                        terminalCue: (event.target.value || undefined) as RallyLabel["terminalCue"],
                      })}
                    >
                      <option value="">Not labeled</option>
                      {terminalCueValues.map((value) => (
                        <option key={value} value={value}>{value.replaceAll("-", " ")}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    End observability
                    <select
                      value={sidebarRally.endObservability ?? ""}
                      onChange={(event) => updateRally(sidebarRallyIndex, {
                        endObservability: (event.target.value || undefined) as RallyLabel["endObservability"],
                      })}
                    >
                      <option value="">Not labeled</option>
                      {endObservabilityValues.map((value) => (
                        <option key={value} value={value}>{value.replaceAll("-", " ")}</option>
                      ))}
                    </select>
                  </label>
                  <div className={styles.confidenceGrid}>
                    <label>
                      Start confidence
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        value={sidebarRally.startConfidence ?? ""}
                        onChange={(event) => updateRally(sidebarRallyIndex, {
                          startConfidence: event.target.value ? Number(event.target.value) : undefined,
                        })}
                      />
                    </label>
                    <label>
                      End confidence
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        value={sidebarRally.endConfidence ?? ""}
                        onChange={(event) => updateRally(sidebarRallyIndex, {
                          endConfidence: event.target.value ? Number(event.target.value) : undefined,
                        })}
                      />
                    </label>
                  </div>
                  <label>
                    Immediate result verified
                    <select
                      value={
                        sidebarRally.verifiedImmediateResult === undefined
                          ? ""
                          : sidebarRally.verifiedImmediateResult
                            ? "yes"
                            : "no"
                      }
                      onChange={(event) => updateRally(sidebarRallyIndex, {
                        verifiedImmediateResult:
                          event.target.value === "" ? undefined : event.target.value === "yes",
                      })}
                    >
                      <option value="">Not labeled</option>
                      <option value="yes">Yes</option>
                      <option value="no">No</option>
                    </select>
                  </label>
                </div>
                <label>
                  Rally notes
                  <textarea
                    rows={3}
                    placeholder="Optional note"
                    value={sidebarRally.notes ?? ""}
                    onChange={(event) =>
                      updateRally(sidebarRallyIndex, {
                        notes: event.target.value || undefined,
                      })
                    }
                  />
                </label>
                {sidebarRally.tags.some(
                  (tag) => tag.startsWith("ai-") || tag.includes("confidence:"),
                ) && (
                  <p className={styles.currentRallyEvidence}>
                    {sidebarRally.tags
                      .filter((tag) => tag.startsWith("ai-") || tag.includes("confidence:"))
                      .join(" · ")}
                  </p>
                )}
              </>
            ) : (
              <p className={styles.currentRallyEmpty}>
                Move the playhead inside a labeled rally to classify it here.
              </p>
            )}
          </div>

          <p className={styles.eyebrow}>TASK METADATA</p>
          <strong className={styles.taskId}>{labels?.recording.id ?? "No task"}</strong>
          {labels && (
            <>
              <dl>
                <div><dt>Environment</dt><dd>{labels.recording.environment}</dd></div>
                <div><dt>Split</dt><dd>{labels.recording.split}</dd></div>
                <div><dt>Duration</dt><dd>{formatPreciseTime(labels.recording.durationSeconds)}</dd></div>
                <div><dt>Rallies</dt><dd>{labels.rallies.length}</dd></div>
                <div><dt>Live time</dt><dd>{formatPreciseTime(totalSeconds(labels.rallies))}</dd></div>
              </dl>
              <label>Annotator
                <input value={labels.annotation.annotator} onChange={(event) => setLabels(markChanged({ ...labels, annotation: { ...labels.annotation, annotator: event.target.value } }))} />
              </label>
              <label>Players per team
                <select value={labels.recording.game.playersPerTeam ?? ""} onChange={(event) => setLabels(markChanged({ ...labels, recording: { ...labels.recording, game: { ...labels.recording.game, playersPerTeam: event.target.value ? Number(event.target.value) : null } } }))}>
                  <option value="">Confirm…</option>
                  {[1, 2, 3, 4, 5, 6].map((count) => <option key={count} value={count}>{count}</option>)}
                </select>
              </label>
              <label>Target points
                <input type="number" min="1" max="100" placeholder="e.g. 21" value={labels.recording.game.targetPoints ?? ""} onChange={(event) => setLabels(markChanged({ ...labels, recording: { ...labels.recording, game: { ...labels.recording.game, targetPoints: event.target.value ? Number(event.target.value) : null } } }))} />
              </label>
              <label>Format / scoring notes
                <input value={labels.recording.game.format ?? ""} onChange={(event) => setLabels(markChanged({ ...labels, recording: { ...labels.recording, game: { ...labels.recording.game, format: event.target.value || null } } }))} />
              </label>
              <label>Annotation notes
                <textarea rows={3} value={labels.annotation.notes} onChange={(event) => setLabels(markChanged({ ...labels, annotation: { ...labels.annotation, notes: event.target.value } }))} />
              </label>
              <label className={styles.reviewed}>
                <input type="checkbox" checked={labels.annotation.continuousVideoReviewed} onChange={(event) => setLabels(markChanged({ ...labels, annotation: { ...labels.annotation, continuousVideoReviewed: event.target.checked } }))} />
                I reviewed the entire continuous video, including short service faults.
              </label>
            </>
          )}
        </aside>
      </section>

      {labels && (
        <section className={styles.tables}>
          <div className={styles.tableHeading}>
            <div><p className={styles.eyebrow}>REQUIRED</p><h2>Rally intervals</h2></div>
            <span>{labels.rallies.length} rallies · click a time to seek</span>
          </div>
          <div className={styles.rows}>
            {labels.rallies.map((row, index) => (
              <div
                className={`${styles.row} ${selectedRallyIndex === index ? styles.selectedRow : ""} ${touchingRallyIndexes.has(index) ? styles.invalidRow : ""}`}
                key={`rally-row-${index}`}
                aria-current={selectedRallyIndex === index ? "true" : undefined}
                aria-invalid={touchingRallyIndexes.has(index) ? "true" : undefined}
                title={
                  touchingRallyIndexes.has(index)
                    ? "This rally touches an adjacent rally. Add a positive dead-time gap."
                    : undefined
                }
              >
                <strong
                  title={[
                    ...row.tags.filter((tag) => tag.startsWith("ai-") || tag.includes("confidence:")),
                    ...(row.notes ? [row.notes] : []),
                  ].join(" · ")}
                >
                  R{String(index + 1).padStart(3, "0")}{row.tags.includes("ai-prelabel") ? " AI" : ""}
                </strong>
                <button onClick={() => seekTo(row.start)}>{formatPreciseTime(row.start)}</button>
                <span>→</span>
                <button onClick={() => seekTo(row.end)}>{formatPreciseTime(row.end)}</button>
                <input aria-label="Rally start seconds" type="number" step="0.001" value={row.start} onChange={(event) => updateRally(index, { start: Number(event.target.value) })} />
                <input aria-label="Rally end seconds" type="number" step="0.001" value={row.end} onChange={(event) => updateRally(index, { end: Number(event.target.value) })} />
                <select
                  className={selectedRallyIndex === index ? styles.selectedClassification : undefined}
                  aria-label="Rally tag"
                  value={row.tags.find((tag) => editableRallyTags.has(tag)) ?? ""}
                  onChange={(event) => updateRallyClassification(index, event.target.value)}
                >
                  <option value="">Normal rally</option>
                  <option value="service-fault">Service fault</option>
                  <option value="ace">Ace / very short</option>
                  <option value="interrupted-replay">Interrupted / replayed</option>
                </select>
                <button className={styles.delete} onClick={() => removeRow("rally", index)}>Delete</button>
              </div>
            ))}
            {labels.rallies.length === 0 && <p className={styles.empty}>No rallies yet. Play to serve contact and press S.</p>}
          </div>

          <div className={styles.pointSection}>
            <div className={styles.tableHeading}>
              <div><p className={styles.eyebrow}>OPTIONAL · WHEN PRESENT</p><h2>Side switches</h2></div>
              <span>{labels.sideSwitches.length} point markers · press X at the switch</span>
            </div>
            <p className={styles.help}>
              Mark the moment teams switch court sides when the recording format includes it. Add a note if the exact transition is obscured.
            </p>
            <div className={styles.pointRows}>
              {labels.sideSwitches.map((marker, index) => (
                <div className={styles.pointRow} key={`side-switch-row-${index}`}>
                  <strong>SW{String(index + 1).padStart(2, "0")}</strong>
                  <button onClick={() => seekTo(marker.time)}>{formatPreciseTime(marker.time)}</button>
                  <input
                    aria-label={`Side switch ${index + 1} seconds`}
                    type="number"
                    step="0.001"
                    value={marker.time}
                    onChange={(event) => updateSideSwitch(index, { time: Number(event.target.value) })}
                  />
                  <input
                    aria-label={`Side switch ${index + 1} notes`}
                    placeholder="Optional note"
                    value={marker.notes ?? ""}
                    onChange={(event) => updateSideSwitch(index, {
                      notes: event.target.value || undefined,
                    })}
                  />
                  <button className={styles.delete} onClick={() => removeSideSwitch(index)}>Delete</button>
                </div>
              ))}
              {labels.sideSwitches.length === 0 && (
                <p className={styles.empty}>No side switches marked for this video.</p>
              )}
            </div>
          </div>

          <div className={styles.secondaryGrid}>
            <div>
              <div className={styles.tableHeading}><div><p className={styles.eyebrow}>WHEN NEEDED</p><h2>Ignored spans</h2></div></div>
              <p className={styles.help}>Use for a partial rally at a file edge, camera gap, or truly unresolvable boundary. Do not use for ordinary dead time.</p>
              {labels.ignoredIntervals.map((row, index) => (
                <div className={styles.smallRow} key={`ignored-row-${index}`}>
                  <button onClick={() => seekTo(row.start)}>{formatPreciseTime(row.start)}–{formatPreciseTime(row.end)}</button>
                  <select value={row.reason} onChange={(event) => updateIgnored(index, { reason: event.target.value })}>
                    <option value="partial-rally">Partial rally</option>
                    <option value="camera-gap">Camera gap</option>
                    <option value="boundary-ambiguous">Boundary ambiguous</option>
                    <option value="non-game-content">Non-game content</option>
                  </select>
                  <button className={styles.delete} onClick={() => removeRow("ignored", index)}>Delete</button>
                </div>
              ))}
            </div>

            <div>
              <div className={styles.tableHeading}><div><p className={styles.eyebrow}>OPTIONAL · 3–5 PER VIDEO</p><h2>Hard negatives</h2></div></div>
              <p className={styles.help}>
                Tag 3–5 confusing dead-time examples per video. Prioritize walking/ball retrieval and celebration/huddle, then add model false positives plus random dead-time controls for an unbiased comparison.
              </p>
              <div className={styles.negativeMarker}>
                <select value={negativeCategory} onChange={(event) => setNegativeCategory(event.target.value)}>
                  {hardNegativeCategories.map((value) => (
                    <option key={value} value={value}>{hardNegativeLabel(value)}</option>
                  ))}
                </select>
                <button onClick={toggleNegative}>{negativeStart === null ? "Start hard negative" : "Finish hard negative"} <kbd>H</kbd></button>
              </div>
              {labels.hardNegatives.map((row, index) => (
                <div className={styles.smallRow} key={`negative-row-${index}`}>
                  <button onClick={() => seekTo(row.start)}>{formatPreciseTime(row.start)}–{formatPreciseTime(row.end)}</button>
                  <select value={row.category} onChange={(event) => updateNegative(index, { category: event.target.value })}>
                    {hardNegativeCategories.map((value) => (
                      <option key={value} value={value}>{hardNegativeLabel(value)}</option>
                    ))}
                  </select>
                  <button className={styles.delete} onClick={() => removeRow("negative", index)}>Delete</button>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.exportPanel}>
            <div>
              <p className={styles.eyebrow}>SAVE OFTEN</p>
              <h2>Save your progress directly.</h2>
              <p>
                Prepared-task drafts save directly to the NAS and resume from the selector.
                Downloads remain available as backups. Completed files are validated before training.
              </p>
              {lastSavedAt && <p>Last direct save: {new Date(lastSavedAt).toLocaleString()}</p>}
            </div>
            <div className={styles.issueList}>
              {completionIssues.length > 0 ? completionIssues.map((issue) => <span key={issue}>• {issue}</span>) : <strong>Ready to export complete labels.</strong>}
            </div>
            <div className={styles.exportButtons}>
              <button
                className={styles.directSave}
                disabled={savingDraft || !preparedTasks.some((task) => task.id === labels.recording.id)}
                onClick={() => void saveDraftDirectly()}
              >
                {savingDraft ? "Saving…" : "Save draft to NAS"}
              </button>
              <button onClick={() => downloadLabels(labels, false)}>Download backup JSON</button>
              <button className={styles.complete} disabled={completionIssues.length > 0} onClick={() => downloadLabels(labels, true)}>Export completed labels</button>
            </div>
          </div>
        </section>
      )}

      {labels && (
        <section className={styles.supplementalTools} aria-label="Optional court and player labels">
          <div className={styles.courtPanel}>
            <div className={styles.courtPanelHeading}>
              <div>
                <strong>Court geometry · 4–8 clicks</strong>
                <span>Near is the camera side. Pause on a clear full-court frame.</span>
              </div>
              {courtAnchor && (
                <button type="button" onClick={() => setCourtAnchor(null)}>Finish court clicks</button>
              )}
            </div>
            <div className={styles.courtAnchors}>
              {courtAnchorSpecs.map((anchor) => {
                const point = courtPoint(labels.recording.courtGeometry, anchor.id);
                return (
                  <div
                    className={`${styles.courtAnchor} ${courtAnchor === anchor.id ? styles.activeCourtAnchor : ""}`}
                    key={anchor.id}
                  >
                    <button type="button" onClick={() => selectCourtAnchor(anchor.id)}>
                      {point ? "✓ " : ""}{anchor.label}{anchor.optional ? " · optional" : ""}
                    </button>
                    {point && (
                      <button
                        type="button"
                        aria-label={`Clear ${anchor.label}`}
                        onClick={() => updateCourtPoint(anchor.id, undefined)}
                      >
                        ×
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className={styles.playerTrackletPanel}>
            <div className={styles.courtPanelHeading}>
              <div>
                <strong>Sparse anonymous player tracks · optional</strong>
                <span>Label short serve/end windows only. IDs are temporary within one rally window—never enter names.</span>
              </div>
              {trackletCapture && (
                <button
                  type="button"
                  onClick={() => {
                    setTrackletCapture(null);
                    setTrackletBoxDrag(null);
                  }}
                >
                  Finish player clicks
                </button>
              )}
            </div>
            <div className={styles.trackletControls}>
              <label>
                Rally
                <select
                  value={activeTrackletRallyIndex >= 0 ? activeTrackletRallyIndex : ""}
                  onChange={(event) =>
                    setTrackletRallyIndex(
                      event.target.value === "" ? null : Number(event.target.value),
                    )
                  }
                >
                  <option value="">Choose…</option>
                  {labels.rallies.map((rally, index) => (
                    <option key={`tracklet-rally-${index}`} value={index}>
                      R{String(index + 1).padStart(3, "0")} · {formatPreciseTime(rally.start)}–{formatPreciseTime(rally.end)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Boundary window
                <select
                  value={trackletWindow}
                  onChange={(event) =>
                    setTrackletWindow(event.target.value as PlayerTracklet["window"])
                  }
                >
                  {playerTrackletWindowValues.map((value) => (
                    <option key={value} value={value}>{value.replaceAll("-", " ")}</option>
                  ))}
                </select>
              </label>
              <label>
                Short track ID
                <input
                  value={trackletId}
                  maxLength={5}
                  placeholder="P1"
                  onChange={(event) => setTrackletId(event.target.value.toUpperCase())}
                />
              </label>
              <label>
                Anonymous team
                <select
                  value={trackletTeam}
                  onChange={(event) =>
                    setTrackletTeam(event.target.value as PlayerTracklet["team"])
                  }
                >
                  {playerTeamValues.map((value) => (
                    <option key={value} value={value}>{value.replaceAll("-", " ")}</option>
                  ))}
                </select>
              </label>
              <label>
                Court side
                <select
                  value={trackletCourtSide}
                  onChange={(event) =>
                    setTrackletCourtSide(event.target.value as PlayerTracklet["courtSide"])
                  }
                >
                  {playerCourtSideValues.map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </select>
              </label>
              <label>
                Coarse state
                <select
                  value={trackletState ?? ""}
                  onChange={(event) =>
                    setTrackletState(
                      (event.target.value || undefined) as PlayerTrackletObservation["state"],
                    )
                  }
                >
                  <option value="">Not labeled</option>
                  {playerStateValues.map((value) => (
                    <option key={value} value={value}>{value.replaceAll("-", " ")}</option>
                  ))}
                </select>
              </label>
            </div>
            <div className={styles.trackletActions}>
              <button type="button" onClick={() => startTrackletCapture("footpoint")}>
                Click footpoints
              </button>
              <button type="button" onClick={() => startTrackletCapture("box")}>
                Draw boxes
              </button>
              <span>
                {activeTrackletRally
                  ? (() => {
                      const bounds = playerWindowBounds(
                        activeTrackletRally,
                        trackletWindow,
                        labels.recording.durationSeconds,
                      );
                      return `${formatPreciseTime(bounds.start)}–${formatPreciseTime(bounds.end)} · seek, then click or drag at 2+ frames`;
                    })()
                  : "Choose a rally first"}
              </span>
            </div>
            {activeTrackletRally && (activeTrackletRally.playerTracklets ?? []).length > 0 ? (
              <div className={styles.trackletRows}>
                {(activeTrackletRally.playerTracklets ?? []).map((tracklet, trackletIndex) => (
                  <div
                    className={styles.trackletRow}
                    key={`${tracklet.window}-${tracklet.trackId}`}
                  >
                    <div className={styles.trackletRowHeading}>
                      <button
                        type="button"
                        onClick={() => {
                          setTrackletId(tracklet.trackId);
                          setTrackletWindow(tracklet.window);
                          setTrackletTeam(tracklet.team);
                          setTrackletCourtSide(tracklet.courtSide);
                          seekTo(tracklet.observations[0].time);
                        }}
                      >
                        {tracklet.trackId} · {tracklet.window} · {tracklet.team} · {tracklet.courtSide}
                      </button>
                      <button
                        type="button"
                        className={styles.delete}
                        onClick={() =>
                          removePlayerTracklet(activeTrackletRallyIndex, trackletIndex)
                        }
                      >
                        Delete track
                      </button>
                    </div>
                    <div className={styles.trackletObservations}>
                      {tracklet.observations.map((observation, observationIndex) => (
                        <div key={`${tracklet.trackId}-${observation.time}`}>
                          <button type="button" onClick={() => seekTo(observation.time)}>
                            {formatPreciseTime(observation.time)}
                          </button>
                          <span>
                            {observation.box && observation.footpoint
                              ? "box + feet"
                              : observation.box
                                ? "box"
                                : "feet"}
                            {observation.state ? ` · ${observation.state}` : ""}
                          </span>
                          <button
                            type="button"
                            aria-label={`Delete ${tracklet.trackId} observation at ${formatPreciseTime(observation.time)}`}
                            onClick={() =>
                              removePlayerObservation(
                                activeTrackletRallyIndex,
                                trackletIndex,
                                observationIndex,
                              )
                            }
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className={styles.trackletEmpty}>
                No player tracks for the selected rally. A usable track has at least two frames.
              </p>
            )}
          </div>
        </section>
      )}
    </main>
  );
}
