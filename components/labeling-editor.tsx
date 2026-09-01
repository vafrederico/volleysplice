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
  servingSideValues,
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
  type ServeMarker,
  type SideSwitch,
} from "@/lib/annotations";
import {
  DEFAULT_JOIN_GAP_SECONDS,
  MAX_JOIN_GAP_SECONDS,
  type ModelAgreement,
  type Rally,
} from "@/lib/edit-list";
import {
  isProductionModelDisagreement,
  productionModelAgreementLabel,
} from "@/lib/production-ensemble";
import {
  findClosestNextRallyIndex,
  findServeMarkerIndexForRally,
  findServeMarkerRallyIndex,
  mergeSelectedRallies,
} from "@/lib/rally-label-editing";
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
import v2 from "./labeling-editor-v2.module.css";

type IntervalKind = "rally" | "ignored" | "negative";
type LabelingBatch = "full" | "pilot";
type TrackletCaptureMode = "footpoint" | "box";
type TrackletBoxDrag = { start: NormalizedPoint; current: NormalizedPoint };
type ModelReference = {
  modelId: string;
  modelLabel: string;
  description?: string;
  rallies: RallyLabel[];
  serveMarkers?: ServeMarker[];
  humanServeMarkers?: ServeMarker[];
  serveModelLabel?: string;
  sideSwitches?: SideSwitch[];
  sideSwitchModelLabel?: string;
  suppressedRanges?: RallyLabel[];
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
  modelSeeded: boolean;
  sourceType: string | null;
  targetStatus: string | null;
};

function preparedTaskStateLabel(task: PreparedTaskSummary): string {
  if (task.savedAt) return `${task.rallyCount} saved`;
  if (task.targetStatus === "reviewed-export-coverage") {
    return `${task.rallyCount} reviewed import`;
  }
  if (task.sourceType === "raw-no-backup-model-feedback") {
    return `${task.rallyCount} model candidates`;
  }
  return task.modelSeeded ? "model ready" : "unlabeled";
}

type BatchSummary = Record<
  LabelingBatch,
  { ready: number; total: number; saved: number; prelabeled: number }
>;

type WorkspaceLayoutDimension =
  | "videoHeight"
  | "leftSidebarWidth"
  | "rightSidebarWidth";
type WorkspaceLayout = Partial<Record<WorkspaceLayoutDimension, number>>;
type WorkspaceResizeDrag = {
  dimension: WorkspaceLayoutDimension;
  direction: 1 | -1;
  pointerId: number;
  startCoordinate: number;
  startSize: number;
};

const emptyBatchSummary: BatchSummary = {
  full: { ready: 0, total: 0, saved: 0, prelabeled: 0 },
  pilot: { ready: 0, total: 0, saved: 0, prelabeled: 0 },
};

const editableRallyTags = new Set(["service-fault", "ace", "interrupted-replay"]);
const playbackResumeKey = "volleycut.labeling.playback.v1";
const timestampEpsilon = 0.0005;
const trackletFrameEpsilon = 0.001;
const comparisonPaddingCases = [0, 1, 2, 3] as const;
const activityPaddingStorageKey = "volleycut:activity-padding:v1";
const activityPaddingEvent = "volleycut:activity-padding";
const workspaceLayoutStorageKey = "volleycut.labelv2.workspace-layout.v1";
const workspaceLayoutBounds: Record<
  WorkspaceLayoutDimension,
  { min: number; max: number; fallback: number }
> = {
  videoHeight: { min: 320, max: 1200, fallback: 900 },
  leftSidebarWidth: { min: 260, max: 520, fallback: 330 },
  rightSidebarWidth: { min: 220, max: 520, fallback: 290 },
};

function clampWorkspaceLayoutDimension(
  dimension: WorkspaceLayoutDimension,
  value: number,
): number {
  const bounds = workspaceLayoutBounds[dimension];
  return Math.round(Math.max(bounds.min, Math.min(bounds.max, value)));
}

function readWorkspaceLayout(): WorkspaceLayout {
  try {
    const raw = window.localStorage.getItem(workspaceLayoutStorageKey);
    if (!raw) return {};
    const value = JSON.parse(raw) as Record<string, unknown> | null;
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const layout: WorkspaceLayout = {};
    for (const dimension of Object.keys(workspaceLayoutBounds) as WorkspaceLayoutDimension[]) {
      const candidate = value[dimension];
      if (typeof candidate === "number" && Number.isFinite(candidate)) {
        layout[dimension] = clampWorkspaceLayoutDimension(dimension, candidate);
      }
    }
    return layout;
  } catch {
    return {};
  }
}

function metricPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function modelAgreementFromTags(tags: string[]): ModelAgreement | undefined {
  const value = tags
    .find((tag) => tag.startsWith("model-agreement:"))
    ?.slice("model-agreement:".length);
  return value === "both-models" ||
    value === "all-labels-v2-only" ||
    value === "previous-production-only"
    ? value
    : undefined;
}

function modelConfidenceFromTags(tags: string[]): number {
  const value = Number(
    tags
      .find((tag) => tag.startsWith("model-confidence:"))
      ?.slice("model-confidence:".length),
  );
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 1;
}

function comparableRallies(rows: RallyLabel[], prefix: string): Rally[] {
  return rows.map((row, index) => ({
    id: `${prefix}-${index + 1}`,
    start: row.start,
    end: row.end,
    confidence: modelConfidenceFromTags(row.tags),
    included: true,
    agreement: modelAgreementFromTags(row.tags),
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

type LabelingEditorProps = {
  variant?: "legacy" | "v2";
};

export function LabelingEditor({ variant = "legacy" }: LabelingEditorProps = {}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoStageRef = useRef<HTMLDivElement>(null);
  const leftSidebarRef = useRef<HTMLElement>(null);
  const rightSidebarRef = useRef<HTMLElement>(null);
  const workspaceLayoutRef = useRef<WorkspaceLayout>({});
  const workspaceResizeRef = useRef<WorkspaceResizeDrag | null>(null);
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
  const [publishingLabels, setPublishingLabels] = useState(false);
  const [lastSavedAt, setLastSavedAt] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoFilename, setVideoFilename] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [rallyStart, setRallyStart] = useState<number | null>(null);
  const [mergeRallyIndexes, setMergeRallyIndexes] = useState<number[]>([]);
  const [focusedServeMarkerIndex, setFocusedServeMarkerIndex] =
    useState<number | null>(null);
  const [focusedSideSwitchIndex, setFocusedSideSwitchIndex] =
    useState<number | null>(null);
  const [ignoredStart, setIgnoredStart] = useState<number | null>(null);
  const [negativeStart, setNegativeStart] = useState<number | null>(null);
  const [negativeCategory, setNegativeCategory] = useState("foreground-crossing");
  const [newServeSide, setNewServeSide] = useState<ServeMarker["side"]>("review");
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
  const [referenceLayer, setReferenceLayer] = useState("production");
  const [timelinePaddingSeconds, setTimelinePaddingSeconds] = useState(2);
  const [modelServeVisibility, setModelServeVisibility] = useState<
    "disagreements" | "all" | "none"
  >("disagreements");
  const [modelSideSwitchVisibility, setModelSideSwitchVisibility] = useState<
    "all" | "disagreements"
  >("all");
  const [workspaceLayout, setWorkspaceLayout] = useState<WorkspaceLayout>({});

  useEffect(() => {
    if (variant !== "v2") return;
    const syncWorkspaceLayout = () => {
      const next = readWorkspaceLayout();
      workspaceLayoutRef.current = next;
      setWorkspaceLayout(next);
    };
    syncWorkspaceLayout();
    window.addEventListener("storage", syncWorkspaceLayout);
    return () => window.removeEventListener("storage", syncWorkspaceLayout);
  }, [variant]);

  function persistWorkspaceLayout(layout: WorkspaceLayout) {
    try {
      window.localStorage.setItem(workspaceLayoutStorageKey, JSON.stringify(layout));
    } catch {
      // Resizing still works for the current session if browser storage is unavailable.
    }
  }

  function updateWorkspaceLayoutDimension(
    dimension: WorkspaceLayoutDimension,
    value: number,
    persist = false,
  ) {
    const next = {
      ...workspaceLayoutRef.current,
      [dimension]: clampWorkspaceLayoutDimension(dimension, value),
    };
    workspaceLayoutRef.current = next;
    setWorkspaceLayout(next);
    if (persist) persistWorkspaceLayout(next);
  }

  function beginWorkspaceResize(
    event: React.PointerEvent<HTMLDivElement>,
    dimension: WorkspaceLayoutDimension,
    element: HTMLElement | null,
    direction: 1 | -1,
  ) {
    if (!element) return;
    event.preventDefault();
    event.currentTarget.focus();
    const bounds = element.getBoundingClientRect();
    workspaceResizeRef.current = {
      dimension,
      direction,
      pointerId: event.pointerId,
      startCoordinate: dimension === "videoHeight" ? event.clientY : event.clientX,
      startSize: dimension === "videoHeight" ? bounds.height : bounds.width,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function moveWorkspaceResize(event: React.PointerEvent<HTMLDivElement>) {
    const drag = workspaceResizeRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const coordinate = drag.dimension === "videoHeight" ? event.clientY : event.clientX;
    updateWorkspaceLayoutDimension(
      drag.dimension,
      drag.startSize + (coordinate - drag.startCoordinate) * drag.direction,
    );
  }

  function finishWorkspaceResize(event: React.PointerEvent<HTMLDivElement>) {
    const drag = workspaceResizeRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const coordinate = drag.dimension === "videoHeight" ? event.clientY : event.clientX;
    updateWorkspaceLayoutDimension(
      drag.dimension,
      drag.startSize + (coordinate - drag.startCoordinate) * drag.direction,
      true,
    );
    workspaceResizeRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function cancelWorkspaceResize(event: React.PointerEvent<HTMLDivElement>) {
    const drag = workspaceResizeRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    persistWorkspaceLayout(workspaceLayoutRef.current);
    workspaceResizeRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function resizeWorkspaceWithKeyboard(
    event: React.KeyboardEvent<HTMLDivElement>,
    dimension: WorkspaceLayoutDimension,
    element: HTMLElement | null,
    direction: 1 | -1,
  ) {
    if (!element) return;
    const vertical = dimension === "videoHeight";
    const decreaseKey = vertical ? "ArrowUp" : "ArrowLeft";
    const increaseKey = vertical ? "ArrowDown" : "ArrowRight";
    const bounds = workspaceLayoutBounds[dimension];
    const currentBounds = element.getBoundingClientRect();
    const current = vertical ? currentBounds.height : currentBounds.width;
    let next: number | null = null;
    if (event.key === decreaseKey) next = current - (event.shiftKey ? 40 : 10) * direction;
    else if (event.key === increaseKey) next = current + (event.shiftKey ? 40 : 10) * direction;
    else if (event.key === "Home") next = bounds.min;
    else if (event.key === "End") next = bounds.max;
    if (next === null) return;
    event.preventDefault();
    event.stopPropagation();
    updateWorkspaceLayoutDimension(dimension, next, true);
  }

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
  const sidebarServeMarkerIndex = labels && sidebarRally
    ? findServeMarkerIndexForRally(
        labels.rallies,
        labels.serveMarkers,
        sidebarRallyIndex,
      )
    : -1;
  const sidebarServeMarker =
    labels && sidebarServeMarkerIndex >= 0
      ? labels.serveMarkers[sidebarServeMarkerIndex]
      : null;
  const focusedServeMarker =
    labels && focusedServeMarkerIndex !== null
      ? labels.serveMarkers[focusedServeMarkerIndex] ?? null
      : null;
  const focusedSideSwitch =
    labels && focusedSideSwitchIndex !== null
      ? labels.sideSwitches[focusedSideSwitchIndex] ?? null
      : null;
  const serveControlRallyIndex = focusedServeMarker && labels
    ? findServeMarkerRallyIndex(labels.rallies, focusedServeMarker.time)
    : sidebarRallyIndex;
  const serveControlRally =
    labels && serveControlRallyIndex >= 0
      ? labels.rallies[serveControlRallyIndex]
      : null;
  const serveControlMarkerIndex = focusedServeMarker && focusedServeMarkerIndex !== null
    ? focusedServeMarkerIndex
    : sidebarServeMarkerIndex;
  const serveControlMarker =
    labels && serveControlMarkerIndex >= 0
      ? labels.serveMarkers[serveControlMarkerIndex]
      : null;
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
          disagreementRallies: modelCore.filter(isProductionModelDisagreement),
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
      if (overlaps(row.start, row.end, labels.hardNegatives)) issues.push("A rally overlaps a hard negative");
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
    labels.serveMarkers.forEach((marker, index) => {
      if (
        !Number.isFinite(marker.time) ||
        marker.time < 0 ||
        marker.time > labels.recording.durationSeconds ||
        (index > 0 && marker.time <= labels.serveMarkers[index - 1].time) ||
        !servingSideValues.includes(marker.side)
      ) {
        issues.push("Serving-side points must be in range, strictly ordered, and labeled");
      }
    });
    if (labels.serveMarkers.some((marker) => marker.side === "review")) {
      issues.push("Resolve every needs-review serving-side marker to near or far");
    }
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

  const publishIssues = useMemo(
    () =>
      completionIssues.filter(
        (issue) =>
          issue !== "Enter the annotator name" &&
          issue !== "Confirm the complete video was reviewed" &&
          issue !== "Set players per team",
      ),
    [completionIssues],
  );

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
      setMergeRallyIndexes([]);
      setFocusedServeMarkerIndex(null);
      setFocusedSideSwitchIndex(null);
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
      let document = parseLabelDocument(await response.json());
      let referenceRallies: RallyLabel[] = [];
      let nextProductionReference: ModelReference | null = null;
      let nextExperimentReferences: ModelReference[] = [];
      if (referencesResponse?.ok) {
        const references = (await referencesResponse.json()) as {
          production?: Partial<ModelReference> | null;
          models?: Array<Partial<ModelReference>>;
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
        const modelReferences = [
          ...(Array.isArray(references.models) ? references.models : []),
          ...(Array.isArray(references.experiments) ? references.experiments : []),
        ];
        if (modelReferences.length > 0) {
          nextExperimentReferences = modelReferences.filter(
            (reference): reference is ModelReference =>
              typeof reference.modelId === "string" &&
              typeof reference.modelLabel === "string" &&
              Array.isArray(reference.rallies),
          );
          nextExperimentReferences = nextExperimentReferences.filter(
            (reference, index, rows) =>
              rows.findIndex((candidate) => candidate.modelId === reference.modelId) === index,
          );
        }
        if (Array.isArray(references.sol?.rallies)) {
          referenceRallies = references.sol.rallies;
        }
      }
      if (
        document.serveMarkers.length === 0 &&
        nextProductionReference?.humanServeMarkers?.length
      ) {
        document = {
          ...document,
          serveMarkers: nextProductionReference.humanServeMarkers,
        };
      }
      setLabels(document);
      setMergeRallyIndexes([]);
      setFocusedServeMarkerIndex(null);
      setFocusedSideSwitchIndex(null);
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
            ? `Loaded ${document.rallies.length} editable predictions from the production ensemble for ${document.recording.id}. Yellow ranges are model disagreements; Sol is shown below as a read-only reference.`
          : documentSource === "prelabel"
            ? `Loaded ${document.rallies.length} unvalidated GPT-5.6 Sol rally candidates for ${document.recording.id}. Review every boundary before completing.`
          : document.prelabel?.analysisMethod ===
              "frozen-production-rally-and-score-specialists-v1"
            ? `Loaded ${document.rallies.length} editable frozen-model rallies, ${document.serveMarkers.length} serving-side markers, and ${document.sideSwitches.length} side switches for ${document.recording.id}. Review every prediction before completing.`
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

  async function publishLabelsDirectly() {
    if (!labels || publishIssues.length > 0) return;
    const preparedTask = preparedTasks.find((task) => task.id === labels.recording.id);
    if (!preparedTask) {
      setError("Publishing is available only for a prepared NAS dataset video.");
      return;
    }
    const completed: LabelDocument = {
      ...labels,
      annotation: {
        ...labels.annotation,
        status: "complete",
        continuousVideoReviewed: true,
        reviewedAt: new Date().toISOString(),
      },
    };
    setError(null);
    setPublishingLabels(true);
    try {
      const response = await fetch(
        `/api/labeling/tasks/${encodeURIComponent(preparedTask.id)}/complete`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(completed),
        },
      );
      const result = (await response.json()) as { error?: string; savedAt?: string };
      if (!response.ok || !result.savedAt) {
        throw new Error(result.error ?? "The completed labels could not be published");
      }
      setLabels(completed);
      setLastSavedAt(result.savedAt);
      setPreparedTasks((current) =>
        current.map((task) =>
          task.id === preparedTask.id
            ? {
                ...task,
                annotationStatus: "complete",
                documentSource: "completed",
                rallyCount: completed.rallies.length,
                savedAt: result.savedAt ?? null,
              }
            : task,
        ),
      );
      setMessage(
        `Published ${completed.rallies.length} human rallies to the NAS at ${new Date(result.savedAt).toLocaleTimeString()}.`,
      );
    } catch (publishError) {
      setError(
        publishError instanceof Error
          ? publishError.message
          : "The completed labels could not be published",
      );
    } finally {
      setPublishingLabels(false);
    }
  }

  function seek(seconds: number) {
    const video = videoRef.current;
    if (!video) return;
    setFocusedServeMarkerIndex(null);
    video.currentTime = Math.min(video.duration || Infinity, Math.max(0, video.currentTime + seconds));
    setCurrentTime(video.currentTime);
    persistPlaybackPosition(video, true);
  }

  function seekTo(seconds: number, serveMarkerIndex: number | null = null) {
    const video = videoRef.current;
    if (!video) return;
    setFocusedServeMarkerIndex(serveMarkerIndex);
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
      if (updateRally(existingIndex, { start: time })) {
        setMessage(
          `Moved rally ${existingIndex + 1} start to ${formatPreciseTime(time)}.`,
        );
      }
      return;
    }
    setRallyStart(time);
    setMessage("Rally start marked. Seek to the first instant live play has ended, then press E.");
  }

  function moveClosestNextRallyStart() {
    if (!labels || !videoRef.current) return;
    const time = roundTime(videoRef.current.currentTime);
    const rallyIndex = findClosestNextRallyIndex(labels.rallies, time, timestampEpsilon);
    if (rallyIndex < 0) {
      setError("There is no later rally start to move.");
      return;
    }
    const rally = labels.rallies[rallyIndex];
    const otherRallies = labels.rallies.filter((_, index) => index !== rallyIndex);
    if (
      overlaps(time, rally.end, otherRallies) ||
      overlaps(time, rally.end, labels.hardNegatives)
    ) {
      setError(
        "Moving that start here would overlap another rally or a hard negative.",
      );
      return;
    }
    const serveMarkerIndex = findServeMarkerIndexForRally(
      labels.rallies,
      labels.serveMarkers,
      rallyIndex,
    );
    if (
      serveMarkerIndex >= 0 &&
      labels.serveMarkers.some(
        (marker, markerIndex) =>
          markerIndex !== serveMarkerIndex &&
          Math.abs(marker.time - time) <= timestampEpsilon,
      )
    ) {
      setError("Another serve marker is already present at the target timestamp.");
      return;
    }
    if (
      updateRally(
        rallyIndex,
        { start: time },
        serveMarkerIndex >= 0 ? { index: serveMarkerIndex, time } : undefined,
      )
    ) {
      setMergeRallyIndexes([]);
      setFocusedServeMarkerIndex(null);
      setMessage(
        `Moved rally ${rallyIndex + 1} start from ${formatPreciseTime(rally.start)} to ${formatPreciseTime(time)}${serveMarkerIndex >= 0 ? " and moved its serve marker with it" : ""}.`,
      );
    }
  }

  function toggleMergeRally(index: number) {
    if (!labels?.rallies[index]) return;
    setMergeRallyIndexes((current) =>
      current.includes(index)
        ? current.filter((selectedIndex) => selectedIndex !== index)
        : [...current, index].sort((left, right) => left - right),
    );
    setError(null);
    setMessage("Updated the rally merge selection. Shift-click a rally to toggle it.");
  }

  function mergeRallySelection() {
    if (!labels) return;
    const result = mergeSelectedRallies({
      rallies: labels.rallies,
      serveMarkers: labels.serveMarkers,
      hardNegatives: labels.hardNegatives,
      selectedIndexes: mergeRallyIndexes,
    });
    if (!result.ok) {
      setError(result.error);
      return;
    }
    setLabels(
      markChanged({
        ...labels,
        rallies: result.rallies,
        serveMarkers: result.serveMarkers,
      }),
    );
    setMergeRallyIndexes([]);
    setFocusedServeMarkerIndex(null);
    setError(null);
    seekTo(
      Math.min(
        result.mergedRally.end - timestampEpsilon,
        result.mergedRally.start + 0.01,
      ),
    );
    setMessage(
      `Merged ${mergeRallyIndexes.length} rallies into rally ${result.mergedIndex + 1}${result.removedServeMarkerCount > 0 ? ` and removed ${result.removedServeMarkerCount} later serve ${result.removedServeMarkerCount === 1 ? "marker" : "markers"}` : ""}. The earliest serve marker was kept${result.movedServeMarkerToStart ? " and moved to the merged rally start" : ""}.`,
    );
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
      overlaps(rally.start, end, labels.hardNegatives)
    ) {
      setError(
        "That end would overlap the next rally or a hard negative.",
      );
      return false;
    }
    if (!updateRally(index, { end })) return false;
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
      overlaps(start, existing.end, labels.hardNegatives)
    ) {
      setError(
        "The split rally would overlap another rally or a hard negative.",
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
    setMergeRallyIndexes([]);
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
    setFocusedServeMarkerIndex(null);
    setFocusedSideSwitchIndex(sideSwitches.findIndex((marker) => marker.time === time));
    setError(null);
    setLabels(markChanged({ ...labels, sideSwitches }));
    setMessage(`Marked a side switch at ${formatPreciseTime(time)}.`);
  }

  function addServeMarker(sideOverride?: ServeMarker["side"]) {
    if (!labels || !videoRef.current) return;
    const time = roundTime(videoRef.current.currentTime);
    if (labels.serveMarkers.some((marker) => marker.time === time)) {
      setError("A serving-side marker is already present at this timestamp.");
      return;
    }
    if (time > labels.recording.durationSeconds) {
      setError("A serving-side marker cannot be placed beyond the task duration.");
      return;
    }
    const side = sideOverride ?? newServeSide;
    const serveMarkers = [
      ...labels.serveMarkers,
      { time, side, origin: "manual" as const },
    ].sort((left, right) => left.time - right.time);
    setFocusedSideSwitchIndex(null);
    setFocusedServeMarkerIndex(serveMarkers.findIndex((marker) => marker.time === time));
    setError(null);
    setLabels(markChanged({ ...labels, serveMarkers }));
    setMessage(
      `Marked a ${side === "review" ? "needs-review" : side} serve at ${formatPreciseTime(time)}.`,
    );
  }

  function setCurrentRallyServeSide(side: ServeMarker["side"]) {
    if (!labels || (!serveControlRally && !serveControlMarker)) {
      setError("Seek into or immediately after a rally before setting its serving side.");
      return;
    }
    if (serveControlMarkerIndex >= 0) {
      updateServeMarker(serveControlMarkerIndex, { side });
    } else if (serveControlRally) {
      const serveMarkers = [
        ...labels.serveMarkers,
        { time: serveControlRally.start, side, origin: "manual" as const },
      ].sort((left, right) => left.time - right.time);
      setLabels(markChanged({ ...labels, serveMarkers }));
    }
    setError(null);
    setMessage(
      `Set ${serveControlRallyIndex >= 0 ? `rally ${serveControlRallyIndex + 1}` : `serve marker ${serveControlMarkerIndex + 1}`} serving side to ${side === "review" ? "needs review" : side}.`,
    );
  }

  function removeCurrentRallyServeMarker() {
    if (!labels || serveControlMarkerIndex < 0) {
      setError("The current rally does not have a serve marker to remove.");
      return;
    }
    const marker = labels.serveMarkers[serveControlMarkerIndex];
    setLabels(
      markChanged({
        ...labels,
        serveMarkers: labels.serveMarkers.filter(
          (_, markerIndex) => markerIndex !== serveControlMarkerIndex,
        ),
      }),
    );
    setFocusedServeMarkerIndex(null);
    setError(null);
    setMessage(
      `Removed the serve marker at ${formatPreciseTime(marker.time)}.`,
    );
  }

  function addInterval(start: number, end: number, kind: IntervalKind): boolean {
    if (!labels) return false;
    setError(null);
    if (end <= start) {
      setError("Interval end must be after its start.");
      return false;
    }
    const conflictingRows = kind === "ignored"
      ? labels.ignoredIntervals
      : [...labels.rallies, ...labels.hardNegatives];
    if (overlaps(start, end, conflictingRows)) {
      setError(
        kind === "ignored"
          ? "That interval overlaps an existing ignored span."
          : "That interval overlaps an existing rally or hard negative.",
      );
      return false;
    }
    const sortRows = <T extends { start: number }>(rows: T[]) =>
      [...rows].sort((left, right) => left.start - right.start);
    if (kind === "rally") {
      const rally: RallyLabel = { start, end, tags: [] };
      setLabels(markChanged({ ...labels, rallies: sortRows([...labels.rallies, rally]) }));
      setMergeRallyIndexes([]);
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
      const key = event.key.toLowerCase();
      if (
        target?.closest(
          variant === "v2"
            ? "input, textarea, select, [contenteditable='true']"
            : "input, textarea, select, button, [contenteditable='true']",
        ) ||
        (variant === "v2" && target?.closest("button") && [" ", "enter"].includes(key))
      ) {
        return;
      }
      if (key === "s" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        void saveDraftDirectly();
      } else if (key === " ") {
        event.preventDefault();
        togglePlayback();
      } else if (key === "s" && event.shiftKey) {
        event.preventDefault();
        moveClosestNextRallyStart();
      } else if (key === "s") beginRally();
      else if (key === "e") finishRally();
      else if (key === "[") toggleIgnored();
      else if (key === "]" && ignoredStart !== null) toggleIgnored();
      else if (key === "h") toggleNegative();
      else if (key === "n") addServeMarker("near");
      else if (key === "f") addServeMarker("far");
      else if (key === "v") addServeMarker();
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

  function updateRally(
    index: number,
    patch: Partial<RallyLabel>,
    linkedServeMarker?: { index: number; time: number },
  ): boolean {
    if (!labels) return false;
    const existing = labels.rallies[index];
    if (!existing) return false;
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
        return false;
      }
    }
    const rallies = labels.rallies.map((row, rowIndex) =>
      rowIndex === index ? candidate : row,
    );
    const serveMarkers = linkedServeMarker
      ? labels.serveMarkers
          .map((marker, markerIndex) =>
            markerIndex === linkedServeMarker.index
              ? { ...marker, time: linkedServeMarker.time }
              : marker,
          )
          .sort((left, right) => left.time - right.time)
      : labels.serveMarkers;
    setError(null);
    setLabels(markChanged({ ...labels, rallies, serveMarkers }));
    return true;
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
    const updated = { ...labels.sideSwitches[index], ...patch };
    const sideSwitches = labels.sideSwitches
      .map((marker, markerIndex) => (markerIndex === index ? updated : marker))
      .sort((left, right) => left.time - right.time);
    setLabels(markChanged({ ...labels, sideSwitches }));
    if (focusedSideSwitchIndex === index) {
      setFocusedSideSwitchIndex(sideSwitches.indexOf(updated));
    }
  }

  function updateServeMarker(index: number, patch: Partial<ServeMarker>) {
    if (!labels) return;
    const updated = { ...labels.serveMarkers[index], ...patch };
    const serveMarkers = labels.serveMarkers
      .map((marker, markerIndex) => (markerIndex === index ? updated : marker))
      .sort((left, right) => left.time - right.time);
    setLabels(markChanged({ ...labels, serveMarkers }));
    if (focusedServeMarkerIndex === index) {
      setFocusedServeMarkerIndex(serveMarkers.indexOf(updated));
    }
  }

  function removeServeMarker(index: number) {
    if (!labels) return;
    setLabels(
      markChanged({
        ...labels,
        serveMarkers: labels.serveMarkers.filter(
          (_, markerIndex) => markerIndex !== index,
        ),
      }),
    );
    setFocusedServeMarkerIndex(null);
    setMessage(`Deleted serving-side marker ${index + 1}.`);
  }

  function removeSideSwitch(index: number) {
    if (!labels) return;
    setLabels(
      markChanged({
        ...labels,
        sideSwitches: labels.sideSwitches.filter((_, markerIndex) => markerIndex !== index),
      }),
    );
    setFocusedSideSwitchIndex(null);
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
    setMergeRallyIndexes([]);
    setError(null);
    setMessage(
      `Deleted rally ${selectedRallyIndex + 1} (${formatPreciseTime(rally.start)}–${formatPreciseTime(rally.end)}).`,
    );
  }

  function removeRow(kind: IntervalKind, index: number) {
    if (!labels) return;
    if (kind === "rally") {
      setLabels(markChanged({ ...labels, rallies: labels.rallies.filter((_, row) => row !== index) }));
      setMergeRallyIndexes([]);
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

  if (variant === "v2") {
    const preparedTaskIndex = tasksForSelectedBatch.findIndex(
      (task) => task.id === selectedPreparedTask,
    );
    const visibleModelReferences = [
      ...(productionReference &&
      (referenceLayer === "production" || referenceLayer === "all")
        ? [productionReference]
        : []),
      ...experimentReferences.filter((reference) =>
        referenceLayer === "all" || referenceLayer === `model:${reference.modelId}`,
      ),
    ];
    const referenceTracks: TimelineTrack[] = visibleModelReferences.map((reference) => {
      const comparison = referenceComparisons.find(
        (candidate) =>
          candidate.reference.modelId === reference.modelId &&
          candidate.paddingSeconds === timelinePaddingSeconds,
      );
      const isProduction = reference === productionReference;
      return {
        id: isProduction ? "production-reference" : reference.modelId,
        label: isProduction ? "Production ensemble" : reference.modelLabel,
        detail: `${reference.rallies.length} core rallies · ${timelinePaddingSeconds}s pad · < ${joinGapSeconds}s joins`,
        title: reference.description,
        ...(comparison
          ? {
              summary: {
                exportTime: formatPreciseTime(comparison.exportSeconds),
                metricsLabel: `${timelinePaddingSeconds}s P_pad/R_core/F1`,
                coreMetrics: `P ${metricPercent(comparison.precision)} · R ${metricPercent(comparison.recall)} · F1 ${metricPercent(comparison.f1)}`,
              },
              exportIntervals: comparison.exportRallies.map((rally, index) => ({
                id: `${reference.modelId}-export-${index}`,
                start: rally.start,
                end: rally.end,
                title: `${reference.modelLabel} · final padded export · ${formatPreciseTime(rally.start)}–${formatPreciseTime(rally.end)}`,
              })),
              joinedGapIntervals: comparison.joinedGapRallies.map((gap, index) => ({
                id: `${reference.modelId}-gap-${index}`,
                start: gap.start,
                end: gap.end,
                title: `${reference.modelLabel} · retained gap under ${joinGapSeconds}s`,
              })),
              missingHumanIntervals: comparison.missingHumanSegments.map((segment, index) => ({
                id: `${reference.modelId}-miss-${index}`,
                start: segment.start,
                end: segment.end,
                title: `${reference.modelLabel} · missed human live time`,
              })),
              intervals: comparison.segments.map((segment) => {
                const midpoint = segment.start + (segment.end - segment.start) / 2;
                const disagreement = comparison.disagreementRallies.find(
                  (rally) => rally.start <= midpoint && midpoint < rally.end,
                );
                return {
                  id: `${reference.modelId}-${segment.id}`,
                  selectionId: null,
                  start: segment.start,
                  end: segment.end,
                  tone: disagreement
                    ? "model-disagreement" as const
                    : `model-${segment.kind}` as const,
                  paddingOrigin: segment.paddingOrigin,
                  title: `${reference.modelLabel} · ${disagreement
                    ? productionModelAgreementLabel(disagreement.agreement)
                    : segment.kind === "match"
                      ? "matches human live time"
                      : segment.kind === "added"
                        ? "predicted outside human live time"
                        : "human live time missed by model"
                  }${segment.paddingOrigin ? ` · ${segment.paddingOrigin} padding` : " · model core"}`,
                };
              }),
            }
          : {
              intervals: reference.rallies.map((row, index) => ({
                id: `${reference.modelId}-${index}`,
                selectionId: null,
                start: row.start,
                end: row.end,
                confidence: modelConfidenceFromTags(row.tags),
                tone: "model" as const,
              })),
            }),
        ...(reference.suppressedRanges?.length
          ? {
              suppressedIntervals: reference.suppressedRanges.map((range, index) => ({
                id: `${reference.modelId}-suppressed-${index}`,
                start: range.start,
                end: range.end,
                title: `Removed by suppression v3 · ${formatPreciseTime(range.start)}–${formatPreciseTime(range.end)}`,
              })),
            }
          : {}),
      } satisfies TimelineTrack;
    });
    if ((referenceLayer === "all" || referenceLayer === "sol") && solReferenceRallies.length > 0) {
      referenceTracks.push({
        id: "sol-reference",
        label: "Sol reference",
        detail: `${solReferenceRallies.length} rallies · read only`,
        intervals: solReferenceRallies.map((row, index) => ({
          id: `sol-${index}`,
          selectionId: null,
          start: row.start,
          end: row.end,
          tone: "sol",
          title: `Sol rally ${index + 1} · ${formatPreciseTime(row.start)}–${formatPreciseTime(row.end)}`,
        })),
      });
    }
    const timelineTracks: TimelineTrack[] = labels
      ? [
          {
            id: "human-labels",
            label: "Human labels",
            detail: `${labels.rallies.length} rallies · editable`,
            active: true,
            intervals: labels.rallies.map((row, index) => ({
              id: `rally-${index}`,
              start: row.start,
              end: row.end,
              tone: "gold",
              title: `Human rally ${index + 1} · ${formatPreciseTime(row.start)}–${formatPreciseTime(row.end)}`,
            })),
          },
          ...referenceTracks,
          ...(labels.ignoredIntervals.length > 0
            ? [
                {
                  id: "ignored",
                  label: "Ignored footage",
                  detail: `${labels.ignoredIntervals.length} spans · outside evaluation`,
                  intervals: labels.ignoredIntervals.map((row, index) => ({
                    id: `ignored-${index}`,
                    selectionId: null,
                    start: row.start,
                    end: row.end,
                    tone: "ignored" as const,
                    title: `${row.reason} · ${formatPreciseTime(row.start)}–${formatPreciseTime(row.end)}`,
                  })),
                } satisfies TimelineTrack,
              ]
            : []),
        ]
      : [];
    const productionServeMarkers = labels && productionReference?.serveMarkers
      ? productionReference.serveMarkers.map((marker, index) => {
          const rallyIndex = findServeMarkerRallyIndex(labels.rallies, marker.time);
          const humanIndex = rallyIndex >= 0
            ? findServeMarkerIndexForRally(labels.rallies, labels.serveMarkers, rallyIndex)
            : -1;
          const humanMarker = humanIndex >= 0 ? labels.serveMarkers[humanIndex] : null;
          const disagrees = humanMarker === null || humanMarker.side !== marker.side;
          return {
            id: `production-serve-marker-${index}`,
            trackId: "production-reference",
            time: marker.time,
            tone: `serve-${marker.side}` as const,
            label: marker.side === "near" ? "N" : "F",
            disagrees,
            title: `${productionReference.serveModelLabel ?? "Serving-side model"} · ${marker.side} · ${formatPreciseTime(marker.time)}${marker.modelConfidence !== undefined ? ` · ${(marker.modelConfidence * 100).toFixed(1)}% confidence` : ""}${humanMarker ? ` · human: ${humanMarker.side}${disagrees ? " (DISAGREES)" : " (agrees)"}` : " · no matching human serve (DISAGREES)"}`,
          };
        })
        .filter((marker) =>
          modelServeVisibility === "all" ||
          (modelServeVisibility === "disagreements" && marker.disagrees),
        )
      : [];
    const humanServeMarkers = labels
      ? labels.serveMarkers.map((marker, index) => {
          const rallyIndex = findServeMarkerRallyIndex(labels.rallies, marker.time);
          const modelMarker = rallyIndex >= 0
            ? productionReference?.serveMarkers?.find(
                (candidate) =>
                  findServeMarkerRallyIndex(labels.rallies, candidate.time) === rallyIndex,
              )
            : undefined;
          return {
            id: `serve-marker-${index}`,
            trackId: "human-labels",
            time: marker.time,
            tone: `serve-${marker.side}` as const,
            label: marker.side === "near" ? "N" : marker.side === "far" ? "F" : "?",
            disagrees: modelMarker !== undefined && modelMarker.side !== marker.side,
            title: `Human serve ${index + 1} · ${marker.side} · ${formatPreciseTime(marker.time)}${modelMarker ? ` · model: ${modelMarker.side}${modelMarker.side !== marker.side ? " (DISAGREES)" : " (agrees)"}` : ""}`,
          };
        })
      : [];
    const productionSideSwitchMarkers = labels && productionReference?.sideSwitches
      ? productionReference.sideSwitches
          .map((marker, index) => {
            const matchingHuman = labels.sideSwitches.find(
              (candidate) => Math.abs(candidate.time - marker.time) <= 4,
            );
            const disagrees = matchingHuman === undefined;
            return {
              id: `production-side-switch-${index}`,
              trackId: "production-reference",
              time: marker.time,
              tone: "model-side-switch" as const,
              label: "X",
              disagrees,
              title: `${productionReference.sideSwitchModelLabel ?? "Side-switch model"} · ${formatPreciseTime(marker.time)}${marker.modelConfidence !== undefined ? ` · ${(marker.modelConfidence * 100).toFixed(1)}% confidence` : ""}${matchingHuman ? ` · human switch at ${formatPreciseTime(matchingHuman.time)} (agrees)` : " · no human switch within 4s (DISAGREES)"}`,
            };
          })
          .filter(
            (marker) =>
              modelSideSwitchVisibility === "all" || marker.disagrees,
          )
      : [];
    const workspaceStyle = {
      ...(workspaceLayout.leftSidebarWidth
        ? { "--v2-left-sidebar-width": `${workspaceLayout.leftSidebarWidth}px` }
        : {}),
      ...(workspaceLayout.rightSidebarWidth
        ? { "--v2-right-sidebar-width": `${workspaceLayout.rightSidebarWidth}px` }
        : {}),
    } as React.CSSProperties;

    return (
      <main className={v2.page}>
        <header className={v2.commandBar} aria-label="Labeling workspace controls">
          <Brand className={v2.brand} label="R&D LABELS" priority />
          <span className={v2.stepBadge}><b>{labels ? 2 : 1}</b>{labels ? "Human labels" : "Dataset video"}</span>
          <label className={v2.compactField}>
            <span>Dataset</span>
            <select
              aria-label="Labeling dataset"
              value={selectedBatch}
              disabled={preparedTasksLoading}
              onChange={(event) => {
                setSelectedBatch(event.target.value as LabelingBatch);
                setSelectedPreparedTask("");
              }}
            >
              <option value="full">Full corpus · {batchSummary.full.ready} ready</option>
              <option value="pilot">Pilot · {batchSummary.pilot.ready} ready</option>
            </select>
          </label>
          <label className={`${v2.compactField} ${v2.videoSelect}`}>
            <span>Video</span>
            <select
              aria-label="Prepared dataset video"
              value={selectedPreparedTask}
              disabled={preparedTasksLoading || tasksForSelectedBatch.length === 0}
              onChange={(event) => void loadPreparedTask(event.target.value)}
            >
              <option value="">
                {preparedTasksLoading ? "Loading videos…" : "Choose a video…"}
              </option>
              {tasksForSelectedBatch.map((task) => (
                <option key={task.id} value={task.id}>
                  {task.priority}. {task.originalFilename} · {formatPreciseTime(task.durationSeconds)} · {preparedTaskStateLabel(task)}
                </option>
              ))}
            </select>
          </label>
          <div className={v2.sourceNavigation}>
            <button
              type="button"
              disabled={preparedTaskIndex <= 0}
              onClick={() => {
                const task = tasksForSelectedBatch[preparedTaskIndex - 1];
                if (task) void loadPreparedTask(task.id);
              }}
              aria-label="Previous dataset video"
            >
              ←
            </button>
            <button
              type="button"
              disabled={preparedTaskIndex < 0 || preparedTaskIndex >= tasksForSelectedBatch.length - 1}
              onClick={() => {
                const task = tasksForSelectedBatch[preparedTaskIndex + 1];
                if (task) void loadPreparedTask(task.id);
              }}
              aria-label="Next dataset video"
            >
              →
            </button>
          </div>
          <div className={v2.sourceState} data-saved={lastSavedAt ? "true" : undefined}>
            <i data-ready={labels ? "true" : undefined} />
            <span>
              {!labels
                ? "Choose a video"
                : lastSavedAt
                  ? `Connected · saved ${new Date(lastSavedAt).toLocaleTimeString()}`
                  : selectedPreparedSummary
                    ? "Connected · not saved"
                    : "Dataset connected"}
            </span>
          </div>
        </header>

        {error && <p className={v2.error} role="alert">{error}</p>}

        <section className={v2.editorShell} style={workspaceStyle}>
          <aside ref={leftSidebarRef} className={v2.summaryCard} aria-label="Human label summary and save actions">
            <div className={v2.summaryStats}>
              <span>HUMAN LABEL SET</span>
              <strong>{labels ? `${labels.rallies.length} rallies` : "No video"}</strong>
              <div>
                <p>{labels?.serveMarkers.length ?? 0} serve markers</p>
                <p>{labels?.sideSwitches.length ?? 0} side switches</p>
                <p>{labels?.ignoredIntervals.length ?? 0} ignored spans</p>
              </div>
            </div>

            <div className={v2.boundaryContract}>
              <span>BOUNDARY CONTRACT</span>
              <p><b>Start</b> at serve contact.</p>
              <p><b>End</b> when live play ends.</p>
              <p>Use ignored spans only for footage that cannot be judged.</p>
            </div>

            <details className={v2.serveMarkerEditor} open>
              <summary>
                <span>Serve markers</span>
                <small>{labels?.serveMarkers.length ?? 0}</small>
              </summary>
              <div className={v2.serveAddButtons} role="group" aria-label="Add serve marker at playhead">
                <button type="button" data-side="near" disabled={!labels || !videoUrl} onClick={() => addServeMarker("near")}>+ Near <kbd>N</kbd></button>
                <button type="button" data-side="far" disabled={!labels || !videoUrl} onClick={() => addServeMarker("far")}>+ Far <kbd>F</kbd></button>
                <button type="button" data-side="review" disabled={!labels || !videoUrl} onClick={() => addServeMarker("review")}>+ Review</button>
              </div>
              {focusedServeMarker && focusedServeMarkerIndex !== null && (
                <div className={v2.serveMarkerInspector}>
                  <div>
                    <strong>Serve {focusedServeMarkerIndex + 1}</strong>
                    <button type="button" onClick={() => seekTo(focusedServeMarker.time, focusedServeMarkerIndex)}>{formatPreciseTime(focusedServeMarker.time)}</button>
                  </div>
                  <div className={v2.serveChoices} role="group" aria-label="Selected serving side">
                    {(["near", "far", "review"] as const).map((side) => (
                      <button
                        type="button"
                        key={side}
                        data-side={side}
                        data-selected={focusedServeMarker.side === side || undefined}
                        onClick={() => updateServeMarker(focusedServeMarkerIndex, { side })}
                      >
                        {side === "review" ? "Review" : side}
                      </button>
                    ))}
                  </div>
                  <div className={v2.serveMarkerActions}>
                    <button type="button" onClick={() => updateServeMarker(focusedServeMarkerIndex, { time: roundTime(currentTime) })}>Move to playhead</button>
                    <button type="button" data-danger="true" onClick={() => removeServeMarker(focusedServeMarkerIndex)}>Delete</button>
                  </div>
                </div>
              )}
              <div className={v2.serveMarkerList} role="region" aria-label="Human serve markers">
                {labels?.serveMarkers.map((marker, index) => {
                  const rallyIndex = findServeMarkerRallyIndex(labels.rallies, marker.time);
                  const modelMarker = rallyIndex >= 0
                    ? productionReference?.serveMarkers?.find(
                        (candidate) => findServeMarkerRallyIndex(labels.rallies, candidate.time) === rallyIndex,
                      )
                    : undefined;
                  const disagrees = modelMarker !== undefined && modelMarker.side !== marker.side;
                  return (
                    <button
                      type="button"
                      key={`serve-${marker.time}`}
                      data-selected={focusedServeMarkerIndex === index || undefined}
                      data-disagrees={disagrees || undefined}
                      onClick={() => {
                        setFocusedSideSwitchIndex(null);
                        setFocusedServeMarkerIndex(index);
                        seekTo(marker.time, index);
                      }}
                    >
                      <i data-side={marker.side}>{marker.side === "near" ? "N" : marker.side === "far" ? "F" : "?"}</i>
                      <span>
                        <strong>Serve {index + 1}</strong>
                        <small>{formatPreciseTime(marker.time)}</small>
                      </span>
                      <em>{modelMarker ? (disagrees ? `Model ${modelMarker.side} ≠` : "Agrees") : "No model"}</em>
                    </button>
                  );
                })}
                {labels?.serveMarkers.length === 0 && <p>No human serve markers yet.</p>}
              </div>
            </details>

            <details className={v2.courtGeometry}>
              <summary>
                <span>Court geometry</span>
                <small>
                  {labels
                    ? `${courtAnchorSpecs.filter((anchor) => courtPoint(labels.recording.courtGeometry, anchor.id)).length}/${courtAnchorSpecs.length}`
                    : "optional"}
                </small>
              </summary>
              <p>Pause on a clear frame, choose an anchor, then click it in the video.</p>
              <div className={v2.courtAnchorGrid}>
                {courtAnchorSpecs.map((anchor) => {
                  const point = courtPoint(labels?.recording.courtGeometry, anchor.id);
                  return (
                    <button
                      type="button"
                      key={anchor.id}
                      disabled={!labels || !videoUrl}
                      data-active={courtAnchor === anchor.id || undefined}
                      data-saved={point ? "true" : undefined}
                      onClick={() => selectCourtAnchor(anchor.id)}
                      title={point ? `${point.x.toFixed(3)}, ${point.y.toFixed(3)}` : undefined}
                    >
                      <i />
                      <span>{anchor.label}</span>
                      <small>{anchor.optional ? "Optional" : point ? "Saved" : "Required"}</small>
                    </button>
                  );
                })}
              </div>
              {courtAnchor && (
                <button type="button" className={v2.finishCourtMode} onClick={() => setCourtAnchor(null)}>
                  Finish court mode
                </button>
              )}
            </details>

            <div className={v2.saveActions}>
              <button
                type="button"
                className={v2.saveDraft}
                disabled={savingDraft || !selectedPreparedSummary || !labels}
                onClick={() => void saveDraftDirectly()}
              >
                {savingDraft ? "Saving…" : "Save draft to NAS"}
              </button>
              <button
                type="button"
                className={v2.publish}
                disabled={publishingLabels || publishIssues.length > 0 || !selectedPreparedSummary}
                onClick={() => void publishLabelsDirectly()}
              >
                {publishingLabels ? "Publishing…" : "Publish human labels"}
              </button>
              {publishIssues.length > 0 && labels && (
                <details className={v2.publishIssues}>
                  <summary>{publishIssues.length} item{publishIssues.length === 1 ? "" : "s"} before publish</summary>
                  <ul>{publishIssues.map((issue) => <li key={issue}>{issue}</li>)}</ul>
                </details>
              )}
            </div>

            <details className={v2.shortcutPanel} open>
              <summary>Keyboard shortcuts</summary>
              <dl>
                <div><dt><kbd>Space</kbd></dt><dd>Play / pause</dd></div>
                <div><dt><kbd>S</kbd> <kbd>E</kbd></dt><dd>Rally start / end</dd></div>
                <div><dt><kbd>N</kbd> <kbd>F</kbd></dt><dd>Near / far serve</dd></div>
                <div><dt><kbd>X</kbd></dt><dd>Side switch</dd></div>
                <div><dt><kbd>[</kbd> <kbd>]</kbd></dt><dd>Ignored span</dd></div>
                <div><dt><kbd>J</kbd> <kbd>K</kbd></dt><dd>Step 0.1 sec</dd></div>
                <div><dt><kbd>⌘/Ctrl S</kbd></dt><dd>Save draft</dd></div>
              </dl>
            </details>

            <details className={v2.localFallback}>
              <summary>Local files</summary>
              <label>
                Open label JSON
                <input type="file" accept="application/json,.json" onChange={(event) => void loadTask(event.target.files?.[0])} />
              </label>
              <label>
                Open matching video
                <input type="file" accept="video/mp4,video/*" onChange={(event) => loadVideo(event.target.files?.[0])} />
              </label>
              {labels && (
                <button type="button" onClick={() => downloadLabels(labels, false)}>
                  Download JSON copy
                </button>
              )}
            </details>
          </aside>

          <div
            className={v2.sidebarResizeHandle}
            role="separator"
            aria-label="Resize labeling sidebar"
            aria-orientation="vertical"
            aria-valuemin={workspaceLayoutBounds.leftSidebarWidth.min}
            aria-valuemax={workspaceLayoutBounds.leftSidebarWidth.max}
            aria-valuenow={workspaceLayout.leftSidebarWidth ?? workspaceLayoutBounds.leftSidebarWidth.fallback}
            tabIndex={0}
            title="Drag to resize the labeling sidebar"
            onPointerDown={(event) => beginWorkspaceResize(event, "leftSidebarWidth", leftSidebarRef.current, 1)}
            onPointerMove={moveWorkspaceResize}
            onPointerUp={finishWorkspaceResize}
            onPointerCancel={cancelWorkspaceResize}
            onKeyDown={(event) => resizeWorkspaceWithKeyboard(event, "leftSidebarWidth", leftSidebarRef.current, 1)}
          />

          <div className={v2.playerColumn}>
            <div className={v2.playerWorkspace}>
              <div className={v2.videoColumn}>
                <div
                  ref={videoStageRef}
                  className={v2.videoStage}
                  style={workspaceLayout.videoHeight ? { height: `${workspaceLayout.videoHeight}px` } : undefined}
                >
                  {videoUrl ? (
                    <>
                      <video
                        key={videoUrl}
                        ref={videoRef}
                        src={videoUrl}
                        controls={courtAnchor === null}
                        playsInline
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
                          className={v2.courtDrawing}
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
                                data-optional={anchor.optional || undefined}
                              />
                            ) : null;
                          })}
                        </svg>
                      )}
                      {courtAnchor && (
                        <button
                          type="button"
                          className={v2.courtClickLayer}
                          aria-label={`Click ${courtAnchorSpecs.find((item) => item.id === courtAnchor)?.label}`}
                          onClick={captureCourtPoint}
                        />
                      )}
                    </>
                  ) : (
                    <div className={v2.noVideo}>
                      <span>Choose a dataset video</span>
                      <p>The proxy, current human labels, and model reference will load together.</p>
                    </div>
                  )}
                </div>
                <div
                  className={v2.videoResizeHandle}
                  role="separator"
                  aria-label="Resize video height"
                  aria-orientation="horizontal"
                  aria-valuemin={workspaceLayoutBounds.videoHeight.min}
                  aria-valuemax={workspaceLayoutBounds.videoHeight.max}
                  aria-valuenow={workspaceLayout.videoHeight ?? workspaceLayoutBounds.videoHeight.fallback}
                  tabIndex={0}
                  title="Drag to resize the video height"
                  onPointerDown={(event) => beginWorkspaceResize(event, "videoHeight", videoStageRef.current, 1)}
                  onPointerMove={moveWorkspaceResize}
                  onPointerUp={finishWorkspaceResize}
                  onPointerCancel={cancelWorkspaceResize}
                  onKeyDown={(event) => resizeWorkspaceWithKeyboard(event, "videoHeight", videoStageRef.current, 1)}
                />
                <div className={v2.playerControls}>
                  <span className={v2.timecode}>{formatPreciseTime(currentTime)}</span>
                  <div className={v2.transport}>
                    <button type="button" onClick={() => seek(-1)} aria-label="Back one second">−1s</button>
                    <button type="button" onClick={() => seek(-0.1)} aria-label="Back one tenth second">−.1</button>
                    <button type="button" className={v2.playButton} onClick={togglePlayback}>Play / pause</button>
                    <button type="button" onClick={() => seek(0.1)} aria-label="Forward one tenth second">+.1</button>
                    <button type="button" onClick={() => seek(1)} aria-label="Forward one second">+1s</button>
                  </div>
                </div>
              </div>

              <div
                className={v2.sidebarResizeHandle}
                role="separator"
                aria-label="Resize selected-label sidebar"
                aria-orientation="vertical"
                aria-valuemin={workspaceLayoutBounds.rightSidebarWidth.min}
                aria-valuemax={workspaceLayoutBounds.rightSidebarWidth.max}
                aria-valuenow={workspaceLayout.rightSidebarWidth ?? workspaceLayoutBounds.rightSidebarWidth.fallback}
                tabIndex={0}
                title="Drag to resize the selected-label sidebar"
                onPointerDown={(event) => beginWorkspaceResize(event, "rightSidebarWidth", rightSidebarRef.current, -1)}
                onPointerMove={moveWorkspaceResize}
                onPointerUp={finishWorkspaceResize}
                onPointerCancel={cancelWorkspaceResize}
                onKeyDown={(event) => resizeWorkspaceWithKeyboard(event, "rightSidebarWidth", rightSidebarRef.current, -1)}
              />

              <aside ref={rightSidebarRef} className={v2.selectionCard} aria-label="Selected human label">
                <span className={v2.selectionEyebrow}>SELECTED LABEL</span>
                {focusedServeMarker && focusedServeMarkerIndex !== null ? (
                  <>
                    <strong>Serve {focusedServeMarkerIndex + 1}</strong>
                    <button type="button" className={v2.selectionTime} onClick={() => seekTo(focusedServeMarker.time, focusedServeMarkerIndex)}>
                      {formatPreciseTime(focusedServeMarker.time)}
                    </button>
                    <div className={v2.serveChoices} role="group" aria-label="Serving side">
                      {(["near", "far", "review"] as const).map((side) => (
                        <button
                          type="button"
                          key={side}
                          data-side={side}
                          data-selected={focusedServeMarker.side === side || undefined}
                          onClick={() => updateServeMarker(focusedServeMarkerIndex, { side })}
                        >
                          {side === "review" ? "Review" : side}
                        </button>
                      ))}
                    </div>
                    <button type="button" onClick={() => updateServeMarker(focusedServeMarkerIndex, { time: roundTime(currentTime) })}>Move to playhead</button>
                    <button type="button" className={v2.danger} onClick={() => removeServeMarker(focusedServeMarkerIndex)}>Delete serve</button>
                  </>
                ) : focusedSideSwitch && focusedSideSwitchIndex !== null ? (
                  <>
                    <strong>Side switch {focusedSideSwitchIndex + 1}</strong>
                    <button type="button" className={v2.selectionTime} onClick={() => seekTo(focusedSideSwitch.time)}>
                      {formatPreciseTime(focusedSideSwitch.time)}
                    </button>
                    <p>Applies to the next serve marker.</p>
                    <button type="button" onClick={() => updateSideSwitch(focusedSideSwitchIndex, { time: roundTime(currentTime) })}>Move to playhead</button>
                    <button type="button" className={v2.danger} onClick={() => removeSideSwitch(focusedSideSwitchIndex)}>Delete switch</button>
                  </>
                ) : sidebarRally ? (
                  <>
                    <strong>Rally {sidebarRallyIndex + 1}</strong>
                    <div className={v2.rallyTimes}>
                      <button type="button" onClick={() => seekTo(sidebarRally.start)}>{formatPreciseTime(sidebarRally.start)}</button>
                      <span>→</span>
                      <button type="button" onClick={() => seekTo(sidebarRally.end)}>{formatPreciseTime(sidebarRally.end)}</button>
                    </div>
                    <p>{(sidebarRally.end - sidebarRally.start).toFixed(2)} seconds of live play</p>
                    <button type="button" onClick={beginRally}>Move start to playhead</button>
                    <button type="button" onClick={finishRally}>Move end to playhead</button>
                    <button type="button" className={v2.danger} onClick={removeSelectedRally} disabled={selectedRallyIndex < 0}>Delete rally</button>
                  </>
                ) : (
                  <div className={v2.noSelection}>
                    <strong>No label selected</strong>
                    <p>Click a human rally or marker in the timeline.</p>
                  </div>
                )}
              </aside>
            </div>

            <section className={v2.timelineSection} aria-label="Human and model label timeline">
              <div className={v2.timelineControlBar}>
                <div className={v2.markingToolbar} role="toolbar" aria-label="Labeling tools">
                  <button type="button" data-tone="start" onClick={beginRally} disabled={!labels || !videoUrl || rallyStart !== null}>
                    {playheadInsideRally ? "Move rally start" : "Rally start"} <kbd>S</kbd>
                  </button>
                  <button type="button" data-tone="end" onClick={finishRally} disabled={!labels || !videoUrl || (rallyStart === null && selectedRallyIndex < 0 && previousRallyIndex < 0)}>
                    {rallyStart !== null ? "Rally end" : playheadInsideRally ? "Move rally end" : "Extend previous"} <kbd>E</kbd>
                  </button>
                  <button type="button" data-tone="near" onClick={() => addServeMarker("near")} disabled={!labels || !videoUrl}>Near serve <kbd>N</kbd></button>
                  <button type="button" data-tone="far" onClick={() => addServeMarker("far")} disabled={!labels || !videoUrl}>Far serve <kbd>F</kbd></button>
                  <button type="button" data-tone="switch" onClick={addSideSwitch} disabled={!labels || !videoUrl}>Side switch <kbd>X</kbd></button>
                  <button type="button" onClick={toggleIgnored} disabled={!labels || !videoUrl}>{ignoredStart === null ? "Ignored start" : "Ignored end"} <kbd>[</kbd></button>
                  <button type="button" onClick={mergeRallySelection} disabled={mergeRallyIndexes.length < 2}>Merge {mergeRallyIndexes.length > 0 ? mergeRallyIndexes.length : "rallies"}</button>
                  <button type="button" onClick={cancelMarker} disabled={rallyStart === null && ignoredStart === null}>Cancel <kbd>Esc</kbd></button>
                </div>

                <div className={v2.timelineOptions}>
                  <label>
                    Model breakdown
                    <select value={referenceLayer} onChange={(event) => setReferenceLayer(event.target.value)}>
                      <option value="production">Production ensemble</option>
                      {experimentReferences.map((reference) => (
                        <option key={reference.modelId} value={`model:${reference.modelId}`}>
                          {reference.modelLabel}
                        </option>
                      ))}
                      {solReferenceRallies.length > 0 && <option value="sol">Sol reference</option>}
                      <option value="all">All model rails</option>
                      <option value="none">Human only</option>
                    </select>
                  </label>
                  <label>
                    Rail padding
                    <select value={timelinePaddingSeconds} onChange={(event) => setTimelinePaddingSeconds(Number(event.target.value))}>
                      {comparisonPaddingCases.map((seconds) => (
                        <option key={seconds} value={seconds}>{seconds}s before / after</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Model serves
                    <select
                      value={modelServeVisibility}
                      onChange={(event) =>
                        setModelServeVisibility(
                          event.target.value as typeof modelServeVisibility,
                        )
                      }
                    >
                      <option value="disagreements">Disagreements only</option>
                      <option value="all">All predictions</option>
                      <option value="none">Hidden</option>
                    </select>
                  </label>
                  <label>
                    Model side switches
                    <select
                      value={modelSideSwitchVisibility}
                      onChange={(event) =>
                        setModelSideSwitchVisibility(
                          event.target.value as typeof modelSideSwitchVisibility,
                        )
                      }
                    >
                      <option value="all">All predictions</option>
                      <option value="disagreements">Disagreements only</option>
                    </select>
                  </label>
                </div>
              </div>

              <div className={v2.timelineLegend}>
                <span data-tone="human"><i /> Human</span>
                <span data-tone="model"><i /> Model</span>
                <span data-tone="disagreement"><i /> Model disagreement</span>
                <span data-tone="near"><i /> Near serve</span>
                <span data-tone="far"><i /> Far serve</span>
                <span data-tone="switch"><i /> Human side switch</span>
                <span data-tone="model-switch"><i /> Model side switch</span>
                <span data-tone="suppressed"><i /> Suppressed / vetoed</span>
                <span data-tone="miss"><i /> Missed human time</span>
              </div>

              <div className={v2.timelineScroll}>
                {labels ? (
                  <RallyTimeline
                    duration={labels.recording.durationSeconds}
                    currentTime={currentTime}
                    tracks={timelineTracks}
                    markers={[
                      ...humanServeMarkers,
                      ...productionServeMarkers,
                      ...productionSideSwitchMarkers,
                      ...labels.sideSwitches.map((marker, index) => ({
                        id: `side-switch-${index}`,
                        trackId: "human-labels",
                        time: marker.time,
                        tone: "side-switch" as const,
                        title: `Side switch ${index + 1} · ${formatPreciseTime(marker.time)}`,
                      })),
                    ]}
                    selectedTrackId="human-labels"
                    selectedMarkerId={focusedServeMarkerIndex !== null ? `serve-marker-${focusedServeMarkerIndex}` : undefined}
                    selectedIntervalId={sidebarRallyIndex >= 0 ? `rally-${sidebarRallyIndex}` : undefined}
                    selectedIntervalIds={mergeRallyIndexes.map((index) => `rally-${index}`)}
                    onSeek={(time, trackId, intervalId, interaction) => {
                      if (interaction?.shiftKey && trackId === "human-labels" && intervalId?.startsWith("rally-")) {
                        const index = Number(intervalId.slice("rally-".length));
                        if (Number.isInteger(index)) toggleMergeRally(index);
                        return;
                      }
                      setMergeRallyIndexes([]);
                      setFocusedServeMarkerIndex(null);
                      setFocusedSideSwitchIndex(null);
                      seekTo(time);
                    }}
                    onMarkerSeek={(time, _trackId, markerId) => {
                      setMergeRallyIndexes([]);
                      if (markerId.startsWith("serve-marker-")) {
                        const index = Number(markerId.slice("serve-marker-".length));
                        setFocusedSideSwitchIndex(null);
                        setFocusedServeMarkerIndex(index);
                        seekTo(time, index);
                      } else if (markerId.startsWith("production-serve-marker-")) {
                        setFocusedServeMarkerIndex(null);
                        setFocusedSideSwitchIndex(null);
                        seekTo(time);
                        setMessage(`Selected the production serving-side prediction at ${formatPreciseTime(time)}.`);
                      } else if (markerId.startsWith("production-side-switch-")) {
                        setFocusedServeMarkerIndex(null);
                        setFocusedSideSwitchIndex(null);
                        seekTo(time);
                        setMessage(`Selected the production side-switch prediction at ${formatPreciseTime(time)}.`);
                      } else {
                        const index = Number(markerId.slice("side-switch-".length));
                        setFocusedServeMarkerIndex(null);
                        setFocusedSideSwitchIndex(index);
                        seekTo(time);
                      }
                    }}
                    ariaLabel="Editable human labels and read-only model references"
                  />
                ) : (
                  <div className={v2.emptyTimeline}>Choose a dataset video to load human and model label layers.</div>
                )}
              </div>
              <p className={v2.timelineHint}>Click to seek. Shift-click consecutive human rallies to merge them. Model rows are read-only.</p>
            </section>

            {message && <p className={v2.editorMessage} aria-live="polite">{message}</p>}
          </div>
        </section>
      </main>
    );
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
                {task.priority}. {task.environment} · {task.originalFilename} · {formatPreciseTime(task.durationSeconds)} · {preparedTaskStateLabel(task)}
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
              onClick={moveClosestNextRallyStart}
              disabled={!labels || !videoUrl}
              title="Move the first rally start after the playhead to the current timestamp"
            >
              Move next start <kbd>Shift+S</kbd>
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
            <button
              className={styles.mergeSelected}
              onClick={mergeRallySelection}
              disabled={mergeRallyIndexes.length < 2}
              title="Shift-click consecutive rally bars or rows, then merge them"
            >
              Merge selected ({mergeRallyIndexes.length})
            </button>
            <button onClick={toggleIgnored} disabled={!labels || !videoUrl}>
              {ignoredStart === null ? "Start ignored span" : "Finish ignored span"} <kbd>[ ]</kbd>
            </button>
            <label className={styles.markerChoice}>
              Serving side
              <select
                aria-label="New serving-side marker label"
                value={newServeSide}
                onChange={(event) => setNewServeSide(event.target.value as ServeMarker["side"])}
              >
                <option value="review">Needs review</option>
                <option value="near">Near side</option>
                <option value="far">Far side</option>
              </select>
            </label>
            <button onClick={() => addServeMarker()} disabled={!labels || !videoUrl}>
              Mark serve side <kbd>V</kbd>
            </button>
            <button onClick={addSideSwitch} disabled={!labels || !videoUrl}>
              Mark side switch <kbd>X</kbd>
            </button>
            <button onClick={cancelMarker} disabled={rallyStart === null && ignoredStart === null && negativeStart === null}>
              Cancel <kbd>Esc</kbd>
            </button>
          </div>

          {(serveControlRally || serveControlMarker) && (
            <div className={styles.serveDecisionPanel}>
              <span>
                <b>
                  {serveControlRally
                    ? `R${String(serveControlRallyIndex + 1).padStart(3, "0")} server`
                    : `SV${String(serveControlMarkerIndex + 1).padStart(3, "0")} marker`}
                </b>
                <small>
                  {serveControlMarker
                    ? serveControlMarker.side === "review"
                      ? "Needs review"
                      : `${serveControlMarker.side} side`
                    : "No marker"}
                </small>
              </span>
              <div
                role="group"
                aria-label={`${serveControlRally ? `R${String(serveControlRallyIndex + 1).padStart(3, "0")} server` : `SV${String(serveControlMarkerIndex + 1).padStart(3, "0")} marker`} decision`}
              >
                {(["near", "far", "review"] as const).map((side) => (
                  <button
                    type="button"
                    key={side}
                    data-side={side}
                    data-selected={serveControlMarker?.side === side ? "true" : undefined}
                    aria-pressed={serveControlMarker?.side === side}
                    onClick={() => setCurrentRallyServeSide(side)}
                  >
                    {side === "review" ? "Review" : side[0].toUpperCase() + side.slice(1)}
                  </button>
                ))}
                <button
                  type="button"
                  className={styles.removeServeDecision}
                  disabled={!serveControlMarker}
                  onClick={removeCurrentRallyServeMarker}
                >
                  Remove
                </button>
              </div>
            </div>
          )}

          {labels && (
            <>
              <div className={styles.comparisonRailControls}>
                <div className={styles.comparisonRailLegend} aria-label="Final export rail legend">
                  <span data-tone="export">Final padded export</span>
                  <span data-tone="joined-gap">Joined short gap</span>
                  <span data-tone="missing">Missed human core</span>
                  <span data-tone="disagreement">Model disagreement</span>
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
                    tone: modelAgreementFromTags(row.tags) &&
                      modelAgreementFromTags(row.tags) !== "both-models"
                      ? ("model-disagreement" as const)
                      : labels.prelabel?.candidateFile.startsWith("model-")
                        ? ("model" as const)
                        : ("gold" as const),
                    })),
                },
                ...referenceComparisons.map((comparison) => ({
                  id: `${comparison.reference.modelId}-${comparison.paddingSeconds}s`,
                  label: `${comparison.reference.modelLabel} · ${comparison.paddingSeconds}s`,
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
                  intervals: comparison.segments.map((segment) => {
                    const midpoint = segment.start + (segment.end - segment.start) / 2;
                    const disagreement = comparison.disagreementRallies.find(
                      (rally) => rally.start <= midpoint && midpoint < rally.end,
                    );
                    return {
                      id: `${comparison.reference.modelId}-${comparison.paddingSeconds}s-${segment.id}`,
                      selectionId: null,
                      start: segment.start,
                      end: segment.end,
                      tone: disagreement
                        ? "model-disagreement" as const
                        : `model-${segment.kind}` as const,
                      paddingOrigin: segment.paddingOrigin,
                      title: `${comparison.reference.modelLabel} · ${comparison.paddingSeconds}s padding · ${disagreement
                        ? productionModelAgreementLabel(disagreement.agreement)
                        : segment.kind === "match"
                          ? "matches editable human live time"
                          : segment.kind === "added"
                            ? "predicted outside editable human live time"
                            : "editable human live time missed by the model"
                      } · ${formatPreciseTime(segment.start)}–${formatPreciseTime(segment.end)}`,
                    };
                  }),
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
              markers={[
                ...labels.serveMarkers.map((marker, index) => ({
                  id: `serve-marker-${index}`,
                  time: marker.time,
                  tone: `serve-${marker.side}` as const,
                  title: `${marker.side === "review" ? "Serving side needs review" : `${marker.side} side serves`}${marker.origin === "model" ? " · model" : " · manual"}${marker.modelConfidence !== undefined ? ` · ${(marker.modelConfidence * 100).toFixed(1)}% near-side probability` : ""}${marker.notes ? ` · ${marker.notes}` : ""}`,
                })),
                ...labels.sideSwitches.map((marker, index) => ({
                  id: `side-switch-${index}`,
                  time: marker.time,
                  tone: "side-switch" as const,
                  title: `Side switch ${index + 1}${marker.origin === "model" ? " · model" : " · manual"}${marker.modelConfidence !== undefined ? ` · ${(marker.modelConfidence * 100).toFixed(1)}% confidence` : ""}${marker.notes ? ` · ${marker.notes}` : ""}`,
                })),
              ]}
              selectedTrackId="editable-rallies"
              selectedIntervalId={
                selectedRallyIndex >= 0 ? `rally-${selectedRallyIndex}` : undefined
              }
              selectedIntervalIds={mergeRallyIndexes.map((index) => `rally-${index}`)}
              onSeek={(time, trackId, intervalId, interaction) => {
                if (
                  interaction?.shiftKey &&
                  trackId === "editable-rallies" &&
                  intervalId?.startsWith("rally-")
                ) {
                  const rallyIndex = Number(intervalId.slice("rally-".length));
                  if (Number.isInteger(rallyIndex)) toggleMergeRally(rallyIndex);
                  return;
                }
                setMergeRallyIndexes([]);
                seekTo(time);
              }}
              onMarkerSeek={(time, _trackId, markerId) => {
                setMergeRallyIndexes([]);
                if (markerId.startsWith("serve-marker-")) {
                  const markerIndex = Number(markerId.slice("serve-marker-".length));
                  if (Number.isInteger(markerIndex) && labels.serveMarkers[markerIndex]) {
                    seekTo(time, markerIndex);
                    setMessage(
                      `Selected serve marker ${markerIndex + 1} at ${formatPreciseTime(time)}.`,
                    );
                    return;
                  }
                }
                seekTo(time);
              }}
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
                  <span>
                    {sidebarServeMarker
                      ? `${sidebarServeMarker.side === "review" ? "serve side: review" : `${sidebarServeMarker.side} serves`} · `
                      : ""}
                    {(sidebarRally.end - sidebarRally.start).toFixed(3)}s
                  </span>
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
            <span>{labels.rallies.length} rallies · Shift-click consecutive rows to select a merge</span>
          </div>
          <div className={styles.rows}>
            {labels.rallies.map((row, index) => (
              <div
                className={`${styles.row} ${selectedRallyIndex === index ? styles.selectedRow : ""} ${mergeRallyIndexes.includes(index) ? styles.mergeSelectedRow : ""} ${touchingRallyIndexes.has(index) ? styles.invalidRow : ""}`}
                key={`rally-row-${index}`}
                aria-current={selectedRallyIndex === index ? "true" : undefined}
                aria-invalid={touchingRallyIndexes.has(index) ? "true" : undefined}
                data-merge-selected={mergeRallyIndexes.includes(index) ? "true" : undefined}
                onClickCapture={(event) => {
                  if (!event.shiftKey) {
                    setMergeRallyIndexes([]);
                    return;
                  }
                  event.preventDefault();
                  event.stopPropagation();
                  toggleMergeRally(index);
                }}
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
              <div><p className={styles.eyebrow}>MODEL-SEEDED · FULLY EDITABLE</p><h2>Serving side</h2></div>
              <span>{labels.serveMarkers.length} serve markers · press V to add at the playhead</span>
            </div>
            <p className={styles.help}>
              Near and far are camera-relative court sides. Correct the model label, move the timestamp, add a missing serve, or delete a false marker; every change is saved in the label document.
            </p>
            <div className={styles.pointRows}>
              {labels.serveMarkers.map((marker, index) => (
                <div className={`${styles.pointRow} ${styles.servePointRow}`} key={`serve-marker-row-${index}`}>
                  <strong title={marker.modelId ?? undefined}>
                    SV{String(index + 1).padStart(3, "0")}{marker.origin === "model" ? " AI" : ""}
                  </strong>
                  <button onClick={() => seekTo(marker.time)}>{formatPreciseTime(marker.time)}</button>
                  <input
                    aria-label={`Serve marker ${index + 1} seconds`}
                    type="number"
                    step="0.001"
                    value={marker.time}
                    onChange={(event) => updateServeMarker(index, { time: Number(event.target.value) })}
                  />
                  <select
                    aria-label={`Serve marker ${index + 1} serving side`}
                    value={marker.side}
                    onChange={(event) => updateServeMarker(index, {
                      side: event.target.value as ServeMarker["side"],
                    })}
                  >
                    <option value="near">Near side</option>
                    <option value="far">Far side</option>
                    <option value="review">Needs review</option>
                  </select>
                  <span className={styles.modelEvidence}>
                    {marker.origin === "model"
                      ? `Model ${marker.modelSide ?? marker.side}${marker.modelConfidence !== undefined ? ` · ${(marker.modelConfidence * 100).toFixed(1)}% near` : ""}`
                      : "Manual marker"}
                  </span>
                  <input
                    aria-label={`Serve marker ${index + 1} notes`}
                    placeholder="Optional correction note"
                    value={marker.notes ?? ""}
                    onChange={(event) => updateServeMarker(index, {
                      notes: event.target.value || undefined,
                    })}
                  />
                  <button className={styles.delete} onClick={() => removeServeMarker(index)}>Delete</button>
                </div>
              ))}
              {labels.serveMarkers.length === 0 && (
                <p className={styles.empty}>No serving-side markers yet. Choose a side and press V at serve contact.</p>
              )}
            </div>
          </div>

          <div className={styles.pointSection}>
            <div className={styles.tableHeading}>
              <div><p className={styles.eyebrow}>MODEL-SEEDED · WHEN PRESENT</p><h2>Side switches</h2></div>
              <span>{labels.sideSwitches.length} point markers · press X at the switch</span>
            </div>
            <p className={styles.help}>
              Model switches are starting suggestions. Move an incorrect timestamp, add a missing switch, or delete a false marker. Add a note if the exact transition is obscured.
            </p>
            <div className={styles.pointRows}>
              {labels.sideSwitches.map((marker, index) => (
                <div className={styles.pointRow} key={`side-switch-row-${index}`}>
                  <strong title={marker.modelId ?? undefined}>
                    SW{String(index + 1).padStart(2, "0")}{marker.origin === "model" ? " AI" : ""}
                  </strong>
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
                  <span className={styles.modelEvidence}>
                    {marker.origin === "model"
                      ? `Model${marker.modelConfidence !== undefined ? ` · ${(marker.modelConfidence * 100).toFixed(1)}%` : ""}`
                      : "Manual marker"}
                  </span>
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
