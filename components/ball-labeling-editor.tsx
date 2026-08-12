"use client";

import Link from "next/link";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
} from "react";

import {
  ballObjectRoles,
  ballObjectVisibilities,
  copyBallFrameLabelPreservingExposure,
  parseBallReviewDocument,
  reviewedFrameCount,
  type BallFrame,
  type BallFrameAnnotation,
  type BallComparisonLayers,
  type BallObject,
  type BallObjectRole,
  type BallObjectVisibility,
  type BallReviewDocument,
  type NormalizedBox,
  type PrimaryBallState,
} from "@/lib/ball-annotations";
import styles from "./ball-labeling-editor.module.css";

type TaskSummary = {
  id: string;
  taskId: string;
  split: "train" | "validation";
  environment: string;
  frameCount: number;
  reviewedFrameCount: number;
  windowCount: number;
  reviewStatus: "unreviewed" | "in_progress" | "complete";
  assistedFrameCount: number;
  savedAt: string | null;
};

type Drawing = {
  role: BallObjectRole;
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
};

const roleLabels: Record<BallObjectRole, string> = {
  "primary-court": "Primary ball",
  "other-court": "Other-court ball",
  unknown: "Unknown-role ball",
};

const visibilityLabels: Record<BallObjectVisibility, string> = {
  clear: "Clear",
  "motion-blurred": "Motion blurred",
  "partially-occluded": "Partially occluded",
};

const stateLabels: Record<PrimaryBallState, string> = {
  localizable: "Primary ball boxed",
  fully_occluded: "Fully occluded",
  out_of_frame: "Out of frame",
  indeterminate: "Indeterminate",
};

const solBoxPalette = [
  { stroke: "#e53935", fill: "rgba(229, 57, 53, 0.16)", surface: "#fff0ef", ink: "#7d1714" },
  { stroke: "#1565c0", fill: "rgba(21, 101, 192, 0.16)", surface: "#edf5ff", ink: "#0c3d74" },
  { stroke: "#2e7d32", fill: "rgba(46, 125, 50, 0.16)", surface: "#effaf0", ink: "#174a1a" },
  { stroke: "#f57c00", fill: "rgba(245, 124, 0, 0.17)", surface: "#fff5e8", ink: "#834300" },
  { stroke: "#8e24aa", fill: "rgba(142, 36, 170, 0.15)", surface: "#fbf0ff", ink: "#541365" },
  { stroke: "#00838f", fill: "rgba(0, 131, 143, 0.16)", surface: "#eaf9fa", ink: "#00515a" },
] as const;

function solBoxColors(index: number) {
  return solBoxPalette[index % solBoxPalette.length];
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function boxIoU(first: NormalizedBox, second: NormalizedBox): number {
  const left = Math.max(first.x, second.x);
  const top = Math.max(first.y, second.y);
  const right = Math.min(first.x + first.width, second.x + second.width);
  const bottom = Math.min(first.y + first.height, second.y + second.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = first.width * first.height + second.width * second.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function sameBox(first: NormalizedBox, second: NormalizedBox): boolean {
  return (["x", "y", "width", "height"] as const).every(
    (key) => Math.abs(first[key] - second[key]) <= 1e-9,
  );
}

function boxFromDrawing(drawing: Drawing): NormalizedBox {
  const x = Math.min(drawing.startX, drawing.currentX);
  const y = Math.min(drawing.startY, drawing.currentY);
  return {
    x,
    y,
    width: Math.max(drawing.startX, drawing.currentX) - x,
    height: Math.max(drawing.startY, drawing.currentY) - y,
  };
}

function cloneAnnotation(annotation: BallFrameAnnotation): BallFrameAnnotation {
  return {
    ...annotation,
    proposalSources: [...annotation.proposalSources],
    objects: annotation.objects.map((object) => ({
      ...object,
      bbox: { ...object.bbox },
    })),
  };
}

function frameImageUrl(taskId: string, frameId: string): string {
  return `/api/ball-labeling/tasks/${encodeURIComponent(taskId)}/frames/${encodeURIComponent(frameId)}`;
}

function pointerPosition(event: ReactPointerEvent<SVGSVGElement>): { x: number; y: number } {
  const bounds = event.currentTarget.getBoundingClientRect();
  return {
    x: clamp((event.clientX - bounds.left) / bounds.width),
    y: clamp((event.clientY - bounds.top) / bounds.height),
  };
}

function nextObjectId(
  annotation: BallFrameAnnotation,
  role: BallObjectRole,
  previous: BallFrameAnnotation | null,
): string {
  const occupied = new Set(annotation.objects.map((object) => object.id));
  const prior = previous?.objects.find((object) => object.role === role && !occupied.has(object.id));
  if (prior) return prior.id;
  const prefix = role === "primary-court" ? "primary-ball" : role === "other-court" ? "other-ball" : "unknown-ball";
  let sequence = 1;
  while (occupied.has(`${prefix}-${sequence}`)) sequence += 1;
  return `${prefix}-${sequence}`;
}

function taskCanBeSaved(document: BallReviewDocument): boolean {
  const reviewed = reviewedFrameCount(document);
  const total = document.immutable.frames.length;
  const review = document.annotations.review;
  return (
    (review.status === "in_progress" && reviewed > 0 && reviewed < total) ||
    (review.status === "complete" && reviewed === total)
  );
}

export function BallLabelingEditor() {
  const [catalog, setCatalog] = useState<TaskSummary[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [task, setTask] = useState<BallReviewDocument | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [selectedObjectIndex, setSelectedObjectIndex] = useState<number | null>(null);
  const [drawRole, setDrawRole] = useState<BallObjectRole | null>(null);
  const [drawing, setDrawing] = useState<Drawing | null>(null);
  const drawingRef = useRef<Drawing | null>(null);
  const [zoom, setZoom] = useState<1 | 2 | 4>(1);
  const [playing, setPlaying] = useState(false);
  const [assistedMode, setAssistedMode] = useState(true);
  const [comparisonByFrame, setComparisonByFrame] = useState<Record<string, BallComparisonLayers>>({});
  const [comparisonAttempted, setComparisonAttempted] = useState<Record<string, boolean>>({});
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [highlightedSolObjectIndex, setHighlightedSolObjectIndex] = useState<number | null>(null);
  const [visibleLayers, setVisibleLayers] = useState({ human: true, sol: true, detector: true });
  const [imageLoaded, setImageLoaded] = useState(false);
  const [annotator, setAnnotator] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const saveInFlightRef = useRef(false);
  const revisionRef = useRef(0);
  const [message, setMessage] = useState("Choose a development task to begin review.");
  const [error, setError] = useState<string | null>(null);

  const frames = task?.immutable.frames ?? [];
  const currentFrame = frames[frameIndex] ?? null;
  const currentAnnotation = currentFrame ? task?.annotations.frames[currentFrame.id] ?? null : null;
  const selectedObject =
    currentAnnotation && selectedObjectIndex !== null
      ? currentAnnotation.objects[selectedObjectIndex] ?? null
      : null;
  const reviewLocked = task?.annotations.review.status === "complete";
  const reviewed = task ? reviewedFrameCount(task) : 0;
  const totalFrames = frames.length;
  const progressPercent = totalFrames ? (reviewed / totalFrames) * 100 : 0;
  const currentComparison = currentFrame ? comparisonByFrame[currentFrame.id] ?? null : null;

  const currentWindowIndex = useMemo(() => {
    if (!task || !currentFrame) return -1;
    return task.immutable.windows.findIndex((window) => window.id === currentFrame.windowId);
  }, [currentFrame, task]);

  const currentWindowFrames = useMemo(
    () =>
      task && currentFrame
        ? task.immutable.frames.filter((frame) => frame.windowId === currentFrame.windowId)
        : [],
    [currentFrame, task],
  );

  const currentWindowFrameIndex = currentFrame
    ? currentWindowFrames.findIndex((frame) => frame.id === currentFrame.id)
    : -1;

  const persist = useCallback(async (document: BallReviewDocument, automatic: boolean): Promise<boolean> => {
    if (saveInFlightRef.current) return false;
    if (!taskCanBeSaved(document)) {
      if (!automatic) {
        setError(
          reviewedFrameCount(document) === document.immutable.frames.length
            ? "All frames are reviewed. Use Complete review to save the final frame."
            : "Review at least one frame before saving.",
        );
      }
      return false;
    }
    const recordingId = document.immutable.recording.id;
    const requestedRevision = revisionRef.current;
    saveInFlightRef.current = true;
    setSaving(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/ball-labeling/tasks/${encodeURIComponent(recordingId)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(document),
        },
      );
      const payload = (await response.json()) as {
        error?: string;
        savedAt?: string;
        reviewStatus?: TaskSummary["reviewStatus"];
        assistedFrameCount?: number;
        proposalExposureByFrame?: Record<
          string,
          BallFrameAnnotation["proposalExposure"]
        >;
      };
      if (!response.ok) throw new Error(payload.error ?? "Review could not be saved");
      if (revisionRef.current === requestedRevision) setDirty(false);
      if (payload.proposalExposureByFrame) {
        setTask((current) => {
          if (!current || current.immutable.recording.id !== recordingId) return current;
          return {
            ...current,
            annotations: {
              ...current.annotations,
              frames: Object.fromEntries(
                Object.entries(current.annotations.frames).map(([frameId, frame]) => [
                  frameId,
                  payload.proposalExposureByFrame?.[frameId] ===
                  "shown_before_label_finalized"
                    ? { ...frame, proposalExposure: "shown_before_label_finalized" as const }
                    : frame,
                ]),
              ),
            },
          };
        });
      }
      setCatalog((current) =>
        current.map((item) =>
          item.id === recordingId
            ? {
                ...item,
                reviewedFrameCount: reviewedFrameCount(document),
                reviewStatus: payload.reviewStatus ?? document.annotations.review.status,
                assistedFrameCount: payload.assistedFrameCount ?? item.assistedFrameCount,
                savedAt: payload.savedAt ?? item.savedAt,
              }
            : item,
        ),
      );
      setMessage(
        automatic
          ? `Autosaved ${reviewedFrameCount(document)}/${document.immutable.frames.length} frames.`
          : `Saved ${reviewedFrameCount(document)}/${document.immutable.frames.length} frames.`,
      );
      return true;
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Review could not be saved");
      return false;
    } finally {
      saveInFlightRef.current = false;
      setSaving(false);
    }
  }, []);

  const requestComparison = useCallback(async (
    mode: "post-decision" | "assisted",
    sources: Array<"sol" | "detector"> = ["sol"],
  ): Promise<BallComparisonLayers | null> => {
    if (!task || !currentFrame || comparisonLoading) return null;
    if (mode === "post-decision" && currentAnnotation?.status !== "reviewed") {
      setError("Make and save the human decision before revealing comparison layers.");
      return null;
    }
    if (dirty && !(await persist(task, false))) return null;
    setComparisonAttempted((current) => ({ ...current, [currentFrame.id]: true }));
    setComparisonLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/ball-labeling/tasks/${encodeURIComponent(task.immutable.recording.id)}/comparisons/${encodeURIComponent(currentFrame.id)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode, sources }),
        },
      );
      const payload = (await response.json()) as BallComparisonLayers & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "Comparison layers are unavailable");
      if (
        payload.schemaVersion !== 1 ||
        payload.taskId !== task.immutable.taskId ||
        payload.frameId !== currentFrame.id ||
        payload.accessMode !== mode
      ) {
        throw new Error("Comparison response does not match the current source frame");
      }
      setComparisonByFrame((current) => {
        const previous = current[currentFrame.id];
        return {
          ...current,
          [currentFrame.id]: {
            ...payload,
            layers: {
              sol: payload.layers.sol ?? previous?.layers.sol ?? null,
              detector: payload.layers.detector ?? previous?.layers.detector ?? null,
            },
          },
        };
      });
      if (mode === "assisted" && payload.proposalExposure !== "blind") {
        setTask((current) => {
          if (!current) return current;
          return {
            ...current,
            annotations: {
              ...current.annotations,
              frames: {
                ...current.annotations.frames,
                [currentFrame.id]: {
                  ...current.annotations.frames[currentFrame.id],
                  proposalExposure: "shown_before_label_finalized",
                  proposalSources: Array.from(
                    new Set([
                      ...current.annotations.frames[currentFrame.id].proposalSources,
                      ...sources,
                    ]),
                  ).filter(
                    (source): source is "sol" | "detector" =>
                      source === "sol" || source === "detector",
                  ),
                },
              },
            },
          };
        });
        if (currentAnnotation?.proposalExposure !== "shown_before_label_finalized") {
          setCatalog((current) =>
            current.map((item) =>
              item.id === task.immutable.recording.id
                ? { ...item, assistedFrameCount: item.assistedFrameCount + 1 }
                : item,
            ),
          );
        }
      }
      setMessage(
        mode === "assisted"
          ? "Sol pre-label loaded. Human verification will be reported as assisted."
          : "Post-decision comparison loaded; the saved human decision remains independent.",
      );
      return payload;
    } catch (comparisonError) {
      setError(
        comparisonError instanceof Error
          ? comparisonError.message
          : "Comparison layers are unavailable",
      );
      return null;
    } finally {
      setComparisonLoading(false);
    }
  }, [comparisonLoading, currentAnnotation, currentFrame, dirty, persist, task]);

  async function loadTask(id: string): Promise<void> {
    if (dirty && !window.confirm("Discard changes that have not been autosaved?")) return;
    setSelectedTaskId(id);
    setTask(null);
    setFrameIndex(0);
    setHighlightedSolObjectIndex(null);
    setPlaying(false);
    setAssistedMode(true);
    setComparisonByFrame({});
    setComparisonAttempted({});
    setSelectedObjectIndex(null);
    setDrawRole(null);
    setImageLoaded(false);
    setDirty(false);
    setError(null);
    setMessage("Loading review task…");
    try {
      const response = await fetch(`/api/ball-labeling/tasks/${encodeURIComponent(id)}`, {
        cache: "no-store",
      });
      const payload = (await response.json()) as unknown;
      if (!response.ok) {
        const detail = payload as { error?: string };
        throw new Error(detail.error ?? "Task could not be loaded");
      }
      const document = parseBallReviewDocument(payload);
      setTask(document);
      setAnnotator(document.annotations.review.annotator ?? "");
      const firstUnreviewed = document.immutable.frames.findIndex(
        (frame) => document.annotations.frames[frame.id].status === "unreviewed",
      );
      setFrameIndex(firstUnreviewed >= 0 ? firstUnreviewed : 0);
      setMessage(
        document.annotations.review.status === "complete"
          ? "Completed human review loaded read-only."
          : "Task loaded without detector suggestions.",
      );
    } catch (loadError) {
      setSelectedTaskId("");
      setError(loadError instanceof Error ? loadError.message : "Task could not be loaded");
      setMessage("Choose another task.");
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    async function loadCatalog() {
      try {
        const response = await fetch("/api/ball-labeling/tasks", {
          cache: "no-store",
          signal: controller.signal,
        });
        const payload = (await response.json()) as { tasks?: TaskSummary[]; error?: string };
        if (!response.ok || !Array.isArray(payload.tasks)) {
          throw new Error(payload.error ?? "Ball-labeling catalog is unavailable");
        }
        setCatalog(payload.tasks);
        const first = payload.tasks.find((item) => item.reviewStatus !== "complete") ?? payload.tasks[0];
        if (first) void loadTask(first.id);
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(loadError instanceof Error ? loadError.message : "Catalog is unavailable");
        }
      } finally {
        if (!controller.signal.aborted) setCatalogLoading(false);
      }
    }
    void loadCatalog();
    return () => controller.abort();
    // The initial catalog load deliberately runs once; subsequent progress updates are local.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!dirty || saving || !task || !taskCanBeSaved(task)) return;
    const timeout = window.setTimeout(() => void persist(task, true), 1200);
    return () => window.clearTimeout(timeout);
  }, [dirty, persist, saving, task]);

  useEffect(() => {
    function warnBeforeUnload(event: BeforeUnloadEvent) {
      if (!dirty) return;
      event.preventDefault();
    }
    window.addEventListener("beforeunload", warnBeforeUnload);
    return () => window.removeEventListener("beforeunload", warnBeforeUnload);
  }, [dirty]);

  useEffect(() => {
    if (!task || !currentFrame || currentWindowFrameIndex < 0) return;
    for (let offset = -3; offset <= 3; offset += 1) {
      if (offset === 0) continue;
      const candidate = currentWindowFrames[currentWindowFrameIndex + offset];
      if (!candidate) continue;
      const image = new Image();
      image.src = frameImageUrl(task.immutable.recording.id, candidate.id);
    }
  }, [currentFrame, currentWindowFrameIndex, currentWindowFrames, task]);

  useEffect(() => {
    if (!playing || currentWindowFrameIndex < 0) return;
    const interval = window.setInterval(() => {
      const next = currentWindowFrames[currentWindowFrameIndex + 1];
      if (!next || !task) {
        setPlaying(false);
        return;
      }
      const nextIndex = task.immutable.frames.findIndex((frame) => frame.id === next.id);
      if (nextIndex >= 0) {
        setSelectedObjectIndex(null);
        setDrawing(null);
        drawingRef.current = null;
        setHighlightedSolObjectIndex(null);
        setImageLoaded(false);
        setFrameIndex(nextIndex);
      }
    }, 125);
    return () => window.clearInterval(interval);
  }, [currentWindowFrameIndex, currentWindowFrames, playing, task]);

  useEffect(() => {
    if (
      !assistedMode ||
      !currentFrame ||
      currentComparison ||
      comparisonAttempted[currentFrame.id] ||
      comparisonLoading
    ) {
      return;
    }
    const mode = currentAnnotation?.status === "reviewed" ? "post-decision" : "assisted";
    const timeout = window.setTimeout(() => void requestComparison(mode), 0);
    return () => window.clearTimeout(timeout);
  }, [
    assistedMode,
    comparisonAttempted,
    comparisonLoading,
    currentComparison,
    currentAnnotation?.status,
    currentFrame,
    requestComparison,
  ]);

  function ensureEditable(): boolean {
    if (!task || !currentFrame || reviewLocked) return false;
    return true;
  }

  function updateFrameAnnotation(
    frame: BallFrame,
    transform: (annotation: BallFrameAnnotation) => BallFrameAnnotation,
  ): void {
    if (!ensureEditable() || !task) return;
    setTask((current) => {
      if (!current) return current;
      const nextAnnotation = transform(cloneAnnotation(current.annotations.frames[frame.id]));
      const nextFrames = { ...current.annotations.frames, [frame.id]: nextAnnotation };
      const nextReviewed = Object.values(nextFrames).filter((item) => item.status === "reviewed").length;
      return {
        ...current,
        annotations: {
          review: {
            ...current.annotations.review,
            status:
              current.annotations.review.status === "complete"
                ? "complete"
                : nextReviewed === 0
                  ? "unreviewed"
                  : "in_progress",
            annotator: nextReviewed === 0 ? null : annotator.trim() || null,
            reviewedAt: current.annotations.review.status === "complete" ? current.annotations.review.reviewedAt : null,
            notes: current.annotations.review.notes,
          },
          frames: nextFrames,
        },
      };
    });
    revisionRef.current += 1;
    setDirty(true);
    setError(null);
  }

  function goToFrame(nextIndex: number): void {
    if (!task) return;
    setPlaying(false);
    setSelectedObjectIndex(null);
    setDrawing(null);
    drawingRef.current = null;
    setHighlightedSolObjectIndex(null);
    setImageLoaded(false);
    setFrameIndex(Math.max(0, Math.min(task.immutable.frames.length - 1, nextIndex)));
  }

  function stepWithinWindow(delta: number): void {
    if (!task || currentWindowFrameIndex < 0) return;
    const next = currentWindowFrames[currentWindowFrameIndex + delta];
    if (!next) return;
    const nextIndex = task.immutable.frames.findIndex((frame) => frame.id === next.id);
    if (nextIndex >= 0) goToFrame(nextIndex);
  }

  function advanceAfterLabel(): void {
    if (!task) return;
    goToFrame(Math.min(task.immutable.frames.length - 1, frameIndex + 1));
  }

  function markNonlocalizable(state: Exclude<PrimaryBallState, "localizable">): void {
    if (!currentFrame || !ensureEditable()) return;
    updateFrameAnnotation(currentFrame, (annotation) => ({
      ...annotation,
      status: "reviewed",
      primaryBallState: state,
      objects: annotation.objects.filter((object) => object.role !== "primary-court"),
    }));
    advanceAfterLabel();
  }

  function previousAnnotation(): BallFrameAnnotation | null {
    if (!task || !currentFrame) return null;
    const previous = task.immutable.frames[frameIndex - 1];
    if (!previous || previous.windowId !== currentFrame.windowId) return null;
    const annotation = task.annotations.frames[previous.id];
    return annotation.status === "reviewed" ? annotation : null;
  }

  function copyPrevious(): void {
    if (!currentFrame) return;
    const previous = previousAnnotation();
    if (!previous) {
      setError("The previous frame in this window has not been reviewed.");
      return;
    }
    updateFrameAnnotation(currentFrame, (current) =>
      copyBallFrameLabelPreservingExposure(previous, current),
    );
    setSelectedObjectIndex(previous.objects.length ? 0 : null);
  }

  async function copySolComparison(): Promise<void> {
    if (!currentFrame || reviewLocked || !ensureEditable()) return;
    if (!window.confirm("Replace this human frame label with the Sol label and mark it assisted?")) {
      return;
    }
    const comparison = await requestComparison("assisted", ["sol"]);
    const sol = comparison?.layers.sol;
    if (!sol) {
      setError("No complete Sol label is available for this frame.");
      return;
    }
    updateFrameAnnotation(currentFrame, () => ({
      ...cloneAnnotation(sol.annotation),
      proposalExposure: "shown_before_label_finalized",
      proposalSources: ["sol"],
    }));
    setSelectedObjectIndex(sol.annotation.objects.length ? 0 : null);
    setMessage("Sol label copied explicitly; this frame is marked assisted.");
  }

  function acceptSolObject(objectIndex: number): void {
    if (!currentFrame || !currentComparison?.layers.sol || !ensureEditable()) return;
    const solAnnotation = currentComparison.layers.sol.annotation;
    const solObject = solAnnotation.objects[objectIndex];
    if (!solObject) return;
    updateFrameAnnotation(currentFrame, (annotation) => {
      const retained = annotation.objects.filter(
        (object) =>
          object.id !== solObject.id &&
          (solObject.role !== "primary-court" || object.role !== "primary-court"),
      );
      return {
        ...annotation,
        status: "reviewed",
        primaryBallState:
          solObject.role === "primary-court"
            ? "localizable"
            : annotation.primaryBallState ??
              (solAnnotation.primaryBallState === "localizable"
                ? "indeterminate"
                : solAnnotation.primaryBallState),
        objects: [
          ...retained,
          { ...solObject, bbox: { ...solObject.bbox } },
        ],
        proposalExposure: "shown_before_label_finalized",
        proposalSources: Array.from(new Set([...annotation.proposalSources, "sol" as const])),
      };
    });
    setSelectedObjectIndex(null);
    setMessage(
      `Sol box ${objectIndex + 1} accepted as correct; advanced to the next frame.`,
    );
    advanceAfterLabel();
  }

  function redrawSolObject(objectIndex: number): void {
    if (!currentFrame || !currentComparison?.layers.sol || !ensureEditable()) return;
    const solObject = currentComparison.layers.sol.annotation.objects[objectIndex];
    if (!solObject) return;
    updateFrameAnnotation(currentFrame, (annotation) => {
      const objects = annotation.objects.filter(
        (object) =>
          solObject.role === "primary-court"
            ? object.role !== "primary-court"
            : !(
                object.role === solObject.role &&
                (object.id === solObject.id || boxIoU(object.bbox, solObject.bbox) > 0.1)
              ),
      );
      const remainingPrimary = objects.some((object) => object.role === "primary-court");
      return {
        ...annotation,
        objects,
        status: objects.length ? annotation.status : "unreviewed",
        primaryBallState:
          solObject.role === "primary-court"
            ? remainingPrimary
              ? "localizable"
              : objects.length
                ? "indeterminate"
                : null
            : annotation.primaryBallState,
        proposalExposure: "shown_before_label_finalized",
        proposalSources: Array.from(new Set([...annotation.proposalSources, "sol" as const])),
      };
    });
    setDrawRole(solObject.role);
    setSelectedObjectIndex(null);
    setMessage(`Draw the improved ${roleLabels[solObject.role].toLowerCase()} box on the image.`);
  }

  function acceptSolState(): void {
    if (!currentFrame || !currentComparison?.layers.sol || !ensureEditable()) return;
    const solAnnotation = currentComparison.layers.sol.annotation;
    updateFrameAnnotation(currentFrame, (annotation) => ({
      ...annotation,
      status: "reviewed",
      primaryBallState: solAnnotation.primaryBallState,
      objects: solAnnotation.objects.map((object) => ({
        ...object,
        bbox: { ...object.bbox },
      })),
      proposalExposure: "shown_before_label_finalized",
      proposalSources: Array.from(new Set([...annotation.proposalSources, "sol" as const])),
    }));
    setMessage("Sol state accepted as correct; it remains classified as human-verified assisted data.");
  }

  async function copyDetectorComparison(): Promise<void> {
    if (!currentFrame || !currentAnnotation || reviewLocked || !ensureEditable()) return;
    const role = drawRole ?? "primary-court";
    if (
      role !== "primary-court" &&
      currentAnnotation.primaryBallState === null
    ) {
      setError("Set the primary-ball state before copying a non-primary detector box.");
      return;
    }
    if (!window.confirm(`Copy the detector's best box as ${roleLabels[role]} and mark it assisted?`)) {
      return;
    }
    const comparison = await requestComparison("assisted", ["detector"]);
    const detector = comparison?.layers.detector;
    const best = detector?.detections.reduce<null | { confidence: number; bbox: NormalizedBox }>(
      (current, candidate) =>
        current === null || candidate.confidence > current.confidence ? candidate : current,
      null,
    );
    if (!best) {
      setError("The detector has no box for this frame.");
      return;
    }
    updateFrameAnnotation(currentFrame, (annotation) => {
      const retained =
        role === "primary-court"
          ? annotation.objects.filter((object) => object.role !== "primary-court")
          : annotation.objects;
      const object: BallObject = {
        id: nextObjectId(annotation, role, previousAnnotation()),
        category: "volleyball",
        role,
        bbox: { ...best.bbox },
        visibility: "clear",
        truncated: false,
      };
      return {
        ...annotation,
        status: "reviewed",
        primaryBallState: role === "primary-court" ? "localizable" : annotation.primaryBallState,
        objects: [...retained, object],
        proposalExposure: "shown_before_label_finalized",
        proposalSources: Array.from(new Set([...annotation.proposalSources, "detector" as const])),
      };
    });
    setMessage("Detector box copied explicitly; this frame is marked assisted.");
  }

  function beginDrawing(event: ReactPointerEvent<SVGSVGElement>): void {
    if (!drawRole || !ensureEditable()) return;
    const point = pointerPosition(event);
    const next = {
      role: drawRole,
      startX: point.x,
      startY: point.y,
      currentX: point.x,
      currentY: point.y,
    };
    drawingRef.current = next;
    setDrawing(next);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function moveDrawing(event: ReactPointerEvent<SVGSVGElement>): void {
    if (!drawingRef.current) return;
    const point = pointerPosition(event);
    const next = { ...drawingRef.current, currentX: point.x, currentY: point.y };
    drawingRef.current = next;
    setDrawing(next);
  }

  function finishDrawing(event: ReactPointerEvent<SVGSVGElement>): void {
    const completed = drawingRef.current;
    drawingRef.current = null;
    setDrawing(null);
    if (!completed || !currentFrame || !currentAnnotation) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const box = boxFromDrawing(completed);
    const minimumWidth = 2 / currentFrame.image.width;
    const minimumHeight = 2 / currentFrame.image.height;
    if (box.width < minimumWidth || box.height < minimumHeight) {
      setError("Draw at least a 2×2 pixel box.");
      return;
    }
    if (completed.role !== "primary-court" && currentAnnotation.primaryBallState === null) {
      setError("Set the primary-ball state before adding an other-court or unknown-role ball.");
      return;
    }
    const prior = previousAnnotation();
    const id = nextObjectId(currentAnnotation, completed.role, prior);
    updateFrameAnnotation(currentFrame, (annotation) => {
      const retained =
        completed.role === "primary-court"
          ? annotation.objects.filter((object) => object.role !== "primary-court")
          : annotation.objects;
      const object: BallObject = {
        id,
        category: "volleyball",
        role: completed.role,
        bbox: box,
        visibility: "clear",
        truncated: false,
      };
      return {
        ...annotation,
        status: "reviewed",
        primaryBallState:
          completed.role === "primary-court" ? "localizable" : annotation.primaryBallState,
        objects: [...retained, object],
      };
    });
    const nextIndex =
      completed.role === "primary-court"
        ? currentAnnotation.objects.filter((object) => object.role !== "primary-court").length
        : currentAnnotation.objects.length;
    setSelectedObjectIndex(nextIndex);
  }

  function updateSelectedObject(transform: (object: BallObject) => BallObject): void {
    if (!currentFrame || selectedObjectIndex === null) return;
    updateFrameAnnotation(currentFrame, (annotation) => ({
      ...annotation,
      objects: annotation.objects.map((object, index) =>
        index === selectedObjectIndex ? transform(object) : object,
      ),
    }));
  }

  function deleteSelectedObject(): void {
    if (!currentFrame || selectedObjectIndex === null || !selectedObject) return;
    const removedPrimary = selectedObject.role === "primary-court";
    updateFrameAnnotation(currentFrame, (annotation) => {
      const objects = annotation.objects.filter((_, index) => index !== selectedObjectIndex);
      return removedPrimary
        ? {
            ...annotation,
            status: objects.length ? "reviewed" : "unreviewed",
            primaryBallState: objects.length ? "indeterminate" : null,
            objects,
          }
        : { ...annotation, objects };
    });
    setSelectedObjectIndex(null);
    if (removedPrimary) setMessage("Primary box deleted; state changed to indeterminate when other objects remain.");
  }

  function chooseWindow(windowIndex: number): void {
    if (!task) return;
    const windowId = task.immutable.windows[windowIndex]?.id;
    if (!windowId) return;
    const windowFrames = task.immutable.frames.filter((frame) => frame.windowId === windowId);
    const firstUnreviewed = windowFrames.find(
      (frame) => task.annotations.frames[frame.id].status === "unreviewed",
    );
    const target = firstUnreviewed ?? windowFrames[0];
    const index = task.immutable.frames.findIndex((frame) => frame.id === target?.id);
    if (index >= 0) goToFrame(index);
  }

  async function completeReview(): Promise<void> {
    if (!task || reviewed !== totalFrames) {
      setError("Review every frame before completing the task.");
      return;
    }
    const completed: BallReviewDocument = {
      ...task,
      annotations: {
        ...task.annotations,
        review: {
          ...task.annotations.review,
          status: "complete",
          annotator: annotator.trim() || null,
          reviewedAt: new Date().toISOString(),
        },
      },
    };
    setTask(completed);
    revisionRef.current += 1;
    setDirty(true);
    await persist(completed, false);
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (
        event.metaKey ||
        event.ctrlKey ||
        event.altKey ||
        target?.matches("input, textarea, select, button")
      ) {
        return;
      }
      const key = event.key.toLowerCase();
      const solShortcutIndex = /^[1-9]$/.test(key) ? Number(key) - 1 : null;
      if (
        ["arrowleft", "arrowright", " ", "backspace", "delete"].includes(key) ||
        solShortcutIndex !== null
      ) {
        event.preventDefault();
      }
      if (
        solShortcutIndex !== null &&
        visibleLayers.sol &&
        currentComparison?.layers.sol?.annotation.objects[solShortcutIndex]
      ) {
        acceptSolObject(solShortcutIndex);
      } else if (key === "arrowleft") stepWithinWindow(-1);
      else if (key === "arrowright" || key === "enter") stepWithinWindow(1);
      else if (key === " ") setPlaying((current) => !current);
      else if (key === "p") setDrawRole("primary-court");
      else if (key === "a") setDrawRole("other-court");
      else if (key === "u") setDrawRole("unknown");
      else if (key === "o") markNonlocalizable("out_of_frame");
      else if (key === "c") markNonlocalizable("fully_occluded");
      else if (key === "i") markNonlocalizable("indeterminate");
      else if (key === "v") copyPrevious();
      else if (key === "backspace" || key === "delete") deleteSelectedObject();
      else if (key === "escape") {
        setDrawRole(null);
        setDrawing(null);
        drawingRef.current = null;
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  const drawingBox = drawing ? boxFromDrawing(drawing) : null;

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>HUMAN-VERIFIED BALL-PRESENCE PILOT</p>
          <h1>Frame review</h1>
          <p>Verify Sol pre-labels quickly, redraw only when needed, and track each source.</p>
        </div>
        <nav>
          <Link href="/label/ball/benchmark">Effort benchmark</Link>
          <Link href="/label">Rally labels</Link>
          <Link href="/">Review dashboard</Link>
        </nav>
      </header>

      <section className={styles.taskBar}>
        <label>
          Development task
          <select
            value={selectedTaskId}
            disabled={catalogLoading || saving}
            onChange={(event) => void loadTask(event.target.value)}
          >
            <option value="">{catalogLoading ? "Loading tasks…" : "Choose a task"}</option>
            {catalog.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id} · {item.reviewedFrameCount}/{item.frameCount}
                {item.assistedFrameCount ? ` · ${item.assistedFrameCount} assisted` : ""}
                {item.reviewStatus === "complete" ? " · complete" : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          Annotator (optional)
          <input
            value={annotator}
            disabled={!task || reviewLocked}
            placeholder="Optional name"
            onChange={(event) => setAnnotator(event.target.value)}
          />
        </label>
        <div className={styles.taskFacts}>
          <span>{task?.immutable.recording.environment ?? "—"}</span>
          <span>{task?.immutable.recording.split ?? "—"}</span>
          <strong>{reviewed}/{totalFrames || "—"}</strong>
        </div>
      </section>

      <div className={styles.progress} aria-label={`${reviewed} of ${totalFrames} frames reviewed`}>
        <span style={{ width: `${progressPercent}%` }} />
      </div>

      {(message || error) && (
        <section className={error ? styles.error : styles.message} aria-live="polite">
          {error ?? message}
        </section>
      )}

      {task && currentFrame && currentAnnotation ? (
        <section className={styles.workspace}>
          <div className={styles.viewerColumn}>
            <div className={styles.windowTabs} aria-label="Sample windows">
              {task.immutable.windows.map((window, index) => {
                const windowFrames = task.immutable.frames.filter((frame) => frame.windowId === window.id);
                const complete = windowFrames.every(
                  (frame) => task.annotations.frames[frame.id].status === "reviewed",
                );
                return (
                  <button
                    key={window.id}
                    className={index === currentWindowIndex ? styles.activeWindow : ""}
                    data-complete={complete}
                    onClick={() => chooseWindow(index)}
                  >
                    Window {index + 1}<small>{complete ? "complete" : "review"}</small>
                  </button>
                );
              })}
            </div>

            <div className={styles.frameMeta}>
              <div>
                <strong>Frame {currentWindowFrameIndex + 1}</strong>
                <span> / {currentWindowFrames.length} in window {currentWindowIndex + 1}</span>
              </div>
              <div>
                <span>{currentFrame.sourceTimestampSeconds.toFixed(3)}s</span>
                <span>{currentAnnotation.status === "reviewed" ? "reviewed" : "unreviewed"}</span>
              </div>
            </div>

            <div className={styles.scroller}>
              <div
                className={styles.stage}
                style={{
                  aspectRatio: `${currentFrame.image.width} / ${currentFrame.image.height}`,
                  width: `${zoom * 100}%`,
                }}
              >
                {/* The native image preserves the exact SHA-pinned PNG pixels used for annotation. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  key={currentFrame.id}
                  src={frameImageUrl(task.immutable.recording.id, currentFrame.id)}
                  alt={`Annotation frame ${currentWindowFrameIndex + 1}`}
                  draggable={false}
                  onLoad={() => setImageLoaded(true)}
                />
                {!imageLoaded && <div className={styles.imageLoading}>Loading exact frame…</div>}
                <svg
                  viewBox="0 0 1 1"
                  preserveAspectRatio="none"
                  aria-label="Ball bounding-box canvas"
                  onPointerDown={beginDrawing}
                  onPointerMove={moveDrawing}
                  onPointerUp={finishDrawing}
                  onPointerCancel={() => {
                    drawingRef.current = null;
                    setDrawing(null);
                  }}
                >
                  {visibleLayers.sol && currentComparison?.layers.sol?.annotation.objects.map((object, index) => (
                    <rect
                      key={`sol-${object.id}-${index}`}
                      className={styles.solBox}
                      data-highlighted={highlightedSolObjectIndex === index}
                      data-muted={highlightedSolObjectIndex !== null && highlightedSolObjectIndex !== index}
                      x={object.bbox.x}
                      y={object.bbox.y}
                      width={object.bbox.width}
                      height={object.bbox.height}
                      style={{
                        "--sol-box-stroke": solBoxColors(index).stroke,
                        "--sol-box-fill": solBoxColors(index).fill,
                      } as CSSProperties}
                      vectorEffect="non-scaling-stroke"
                    />
                  ))}
                  {visibleLayers.detector && currentComparison?.layers.detector?.detections.map((detection, index) => (
                    <rect
                      key={`detector-${index}`}
                      className={styles.detectorBox}
                      x={detection.bbox.x}
                      y={detection.bbox.y}
                      width={detection.bbox.width}
                      height={detection.bbox.height}
                      vectorEffect="non-scaling-stroke"
                    />
                  ))}
                  {visibleLayers.human && currentAnnotation.objects.map((object, index) => (
                    <rect
                      key={`${object.id}-${index}`}
                      className={styles.objectBox}
                      data-role={object.role}
                      data-selected={index === selectedObjectIndex}
                      x={object.bbox.x}
                      y={object.bbox.y}
                      width={object.bbox.width}
                      height={object.bbox.height}
                      vectorEffect="non-scaling-stroke"
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        setSelectedObjectIndex(index);
                      }}
                    />
                  ))}
                  {drawingBox && (
                    <rect
                      className={styles.drawingBox}
                      data-role={drawing?.role}
                      x={drawingBox.x}
                      y={drawingBox.y}
                      width={drawingBox.width}
                      height={drawingBox.height}
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                </svg>
              </div>
            </div>

            <div className={styles.transport}>
              <button onClick={() => stepWithinWindow(-1)} disabled={currentWindowFrameIndex <= 0}>← Previous</button>
              <button onClick={() => setPlaying((current) => !current)}>{playing ? "Pause" : "Play window"}</button>
              <button
                onClick={() => stepWithinWindow(1)}
                disabled={currentWindowFrameIndex >= currentWindowFrames.length - 1}
              >
                Next →
              </button>
              <button onClick={copyPrevious} disabled={currentWindowFrameIndex <= 0 || reviewLocked}>Copy previous <kbd>V</kbd></button>
              <label>
                Zoom
                <select value={zoom} onChange={(event) => setZoom(Number(event.target.value) as 1 | 2 | 4)}>
                  <option value={1}>1×</option>
                  <option value={2}>2×</option>
                  <option value={4}>4×</option>
                </select>
              </label>
            </div>

            <input
              className={styles.scrubber}
              type="range"
              aria-label="Frame within current window"
              min={0}
              max={Math.max(0, currentWindowFrames.length - 1)}
              value={Math.max(0, currentWindowFrameIndex)}
              onChange={(event) => {
                const frame = currentWindowFrames[Number(event.target.value)];
                const index = task.immutable.frames.findIndex((candidate) => candidate.id === frame?.id);
                if (index >= 0) goToFrame(index);
              }}
            />

            <div className={styles.drawTools}>
              <span>Draw rectangle</span>
              {ballObjectRoles.map((role) => (
                <button
                  key={role}
                  className={drawRole === role ? styles.activeTool : ""}
                  data-role={role}
                  disabled={reviewLocked}
                  onClick={() => setDrawRole((current) => (current === role ? null : role))}
                >
                  {roleLabels[role]}
                  <kbd>{role === "primary-court" ? "P" : role === "other-court" ? "A" : "U"}</kbd>
                </button>
              ))}
            </div>
          </div>

          <aside className={styles.sidebar}>
            <section className={styles.comparisonPanel}>
              <p className={styles.sectionLabel}>COMPARISON LAYERS</p>
              <strong
                className={
                  currentAnnotation.proposalExposure === "shown_before_label_finalized"
                    ? styles.assistedBadge
                    : styles.blindBadge
                }
              >
                {currentAnnotation.proposalExposure === "shown_before_label_finalized"
                  ? `${currentAnnotation.proposalSources.length ? currentAnnotation.proposalSources.join(" + ").toUpperCase() : "PROPOSAL"}-ASSISTED · INCLUDED AS HUMAN VERIFIED`
                  : "BLIND HUMAN LABEL"}
              </strong>
              <label className={styles.assistedToggle}>
                <input
                  type="checkbox"
                  checked={assistedMode}
                  disabled={reviewLocked}
                  onChange={(event) => {
                    if (
                      event.target.checked &&
                      !window.confirm(
                        "Enable assisted mode? Sol labels will be shown as pre-labels. Accepted or corrected frames remain in assisted human-verified metrics and are also reported separately from independent labels.",
                      )
                    ) {
                      return;
                    }
                    setAssistedMode(event.target.checked);
                    setMessage(
                      event.target.checked
                        ? "Sol-assisted mode enabled; verify each proposed box or draw a better one."
                        : "Blind mode restored for frames not previously exposed.",
                    );
                  }}
                />
                Assisted pre-label mode
              </label>
              {!currentComparison ? (
                <button
                  className={styles.reveal}
                  disabled={
                    comparisonLoading ||
                    currentAnnotation.status !== "reviewed" ||
                    assistedMode
                  }
                  onClick={() => void requestComparison("post-decision")}
                >
                  {comparisonLoading ? "Loading layers…" : "Reveal after human decision"}
                </button>
              ) : (
                <>
                  <div className={styles.layerToggles}>
                    {(["human", "sol", "detector"] as const).map((layer) => (
                      <label key={layer} data-layer={layer}>
                        <input
                          type="checkbox"
                          checked={visibleLayers[layer]}
                          onChange={(event) =>
                            setVisibleLayers((current) => ({
                              ...current,
                              [layer]: event.target.checked,
                            }))
                          }
                        />
                        {layer}
                      </label>
                    ))}
                  </div>
                  <dl className={styles.layerFacts}>
                    <div>
                      <dt>Sol</dt>
                      <dd>
                        {currentComparison.layers.sol?.annotation.primaryBallState
                          ? stateLabels[currentComparison.layers.sol.annotation.primaryBallState]
                          : "unavailable"}
                      </dd>
                    </div>
                    <div>
                      <dt>Detector</dt>
                      <dd>
                        {currentComparison.layers.detector
                          ? `${currentComparison.layers.detector.detections.length} box(es) · ${Math.round(currentComparison.layers.detector.ballPresenceProbability * 100)}%`
                          : "unavailable"}
                      </dd>
                    </div>
                  </dl>
                  <div className={styles.copyActions}>
                    {currentComparison.layers.sol?.annotation.objects.map(
                      (solObject, objectIndex) => {
                        const candidates = currentAnnotation.objects.filter(
                          (humanObject) => humanObject.role === solObject.role,
                        );
                        const exact = candidates.some(
                          (humanObject) =>
                            sameBox(humanObject.bbox, solObject.bbox) &&
                            humanObject.visibility === solObject.visibility &&
                            humanObject.truncated === solObject.truncated,
                        );
                        const adjusted =
                          !exact &&
                          candidates.some(
                            (humanObject) => boxIoU(humanObject.bbox, solObject.bbox) > 0.1,
                          );
                        return (
                          <div
                            key={`${solObject.id}-${objectIndex}`}
                            className={styles.solBoxDecision}
                            data-highlighted={highlightedSolObjectIndex === objectIndex}
                            style={{
                              "--sol-box-color": solBoxColors(objectIndex).stroke,
                              "--sol-box-surface": solBoxColors(objectIndex).surface,
                              "--sol-box-ink": solBoxColors(objectIndex).ink,
                            } as CSSProperties}
                            onMouseEnter={() => setHighlightedSolObjectIndex(objectIndex)}
                            onMouseLeave={() => setHighlightedSolObjectIndex(null)}
                            onFocus={() => setHighlightedSolObjectIndex(objectIndex)}
                            onBlur={(event) => {
                              if (!event.currentTarget.contains(event.relatedTarget)) {
                                setHighlightedSolObjectIndex(null);
                              }
                            }}
                          >
                            <span>
                              <b>{objectIndex + 1}</b>
                              Sol box {objectIndex + 1} · {roleLabels[solObject.role]} ·{
                              exact ? " accepted" : adjusted ? " adjusted" : " pending"
                              }
                            </span>
                            <button
                              disabled={reviewLocked || exact}
                              aria-label={`Accept Sol box ${objectIndex + 1} as correct and go to the next frame`}
                              title={`Shortcut: ${objectIndex + 1}`}
                              onClick={() => acceptSolObject(objectIndex)}
                            >
                              {exact ? "Correct ✓" : "Correct · accept"}
                            </button>
                            <button
                              disabled={reviewLocked}
                              onClick={() => redrawSolObject(objectIndex)}
                            >
                              Draw better
                            </button>
                          </div>
                        );
                      },
                    )}
                    {currentComparison.layers.sol &&
                      currentComparison.layers.sol.annotation.objects.length === 0 && (
                        <button disabled={reviewLocked} onClick={acceptSolState}>
                          Accept Sol state: {stateLabels[currentComparison.layers.sol.annotation.primaryBallState!]}
                        </button>
                      )}
                    <button
                      disabled={reviewLocked || !currentComparison.layers.sol}
                      onClick={() => void copySolComparison()}
                    >
                      Accept entire Sol label
                    </button>
                    <button
                      disabled={reviewLocked || comparisonLoading}
                      onClick={() => void requestComparison("assisted", ["detector"])}
                    >
                      {currentComparison.layers.detector ? "Detector loaded" : "Load detector separately"}
                    </button>
                    {currentComparison.layers.detector?.detections.length ? (
                      <button disabled={reviewLocked} onClick={() => void copyDetectorComparison()}>
                        Use best detector box
                      </button>
                    ) : null}
                  </div>
                  <p className={styles.hint}>
                    Verify Sol boxes one by one. Accept correct boxes directly; use Draw better
                    only when a box is wrong or can be improved. Assisted and independent metrics
                    are reported separately.
                  </p>
                </>
              )}
            </section>

            <section>
              <p className={styles.sectionLabel}>PRIMARY BALL STATE</p>
              <strong className={styles.stateValue}>
                {currentAnnotation.primaryBallState
                  ? stateLabels[currentAnnotation.primaryBallState]
                  : "Not reviewed"}
              </strong>
              <div className={styles.stateButtons}>
                <button
                  data-active={currentAnnotation.primaryBallState === "out_of_frame"}
                  disabled={reviewLocked}
                  onClick={() => markNonlocalizable("out_of_frame")}
                >
                  Out of frame <kbd>O</kbd>
                </button>
                <button
                  data-active={currentAnnotation.primaryBallState === "fully_occluded"}
                  disabled={reviewLocked}
                  onClick={() => markNonlocalizable("fully_occluded")}
                >
                  Fully occluded <kbd>C</kbd>
                </button>
                <button
                  data-active={currentAnnotation.primaryBallState === "indeterminate"}
                  disabled={reviewLocked}
                  onClick={() => markNonlocalizable("indeterminate")}
                >
                  Indeterminate <kbd>I</kbd>
                </button>
              </div>
              <p className={styles.hint}>Drawing a primary-court box sets the state to localizable.</p>
            </section>

            <section>
              <p className={styles.sectionLabel}>OBJECTS · {currentAnnotation.objects.length}</p>
              {currentAnnotation.objects.length ? (
                <div className={styles.objectList}>
                  {currentAnnotation.objects.map((object, index) => (
                    <button
                      key={`${object.id}-${index}`}
                      data-active={index === selectedObjectIndex}
                      onClick={() => setSelectedObjectIndex(index)}
                    >
                      <span data-role={object.role} />
                      {object.id}
                      <small>{roleLabels[object.role]}</small>
                    </button>
                  ))}
                </div>
              ) : (
                <p className={styles.empty}>No volleyball boxes on this frame.</p>
              )}
            </section>

            {selectedObject && selectedObjectIndex !== null && (
              <section className={styles.objectEditor}>
                <p className={styles.sectionLabel}>SELECTED BOX</p>
                <label>
                  Track ID
                  <input
                    value={selectedObject.id}
                    disabled={reviewLocked}
                    onChange={(event) => {
                      const nextId = event.target.value.trim();
                      if (!nextId || currentAnnotation.objects.some((object, index) => index !== selectedObjectIndex && object.id === nextId)) return;
                      updateSelectedObject((object) => ({ ...object, id: nextId }));
                    }}
                  />
                </label>
                <label>
                  Role
                  <select
                    value={selectedObject.role}
                    disabled={reviewLocked || selectedObject.role === "primary-court"}
                    onChange={(event) =>
                      updateSelectedObject((object) => ({ ...object, role: event.target.value as BallObjectRole }))
                    }
                  >
                    {ballObjectRoles.filter((role) => role !== "primary-court" || selectedObject.role === role).map((role) => (
                      <option key={role} value={role}>{roleLabels[role]}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Visibility
                  <select
                    value={selectedObject.visibility}
                    disabled={reviewLocked}
                    onChange={(event) =>
                      updateSelectedObject((object) => ({
                        ...object,
                        visibility: event.target.value as BallObjectVisibility,
                      }))
                    }
                  >
                    {ballObjectVisibilities.map((visibility) => (
                      <option key={visibility} value={visibility}>{visibilityLabels[visibility]}</option>
                    ))}
                  </select>
                </label>
                <label className={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={selectedObject.truncated}
                    disabled={reviewLocked}
                    onChange={(event) =>
                      updateSelectedObject((object) => ({ ...object, truncated: event.target.checked }))
                    }
                  />
                  Cropped by the image edge
                </label>
                <button className={styles.delete} disabled={reviewLocked} onClick={deleteSelectedObject}>
                  Delete selected box
                </button>
              </section>
            )}

            <section>
              <label className={styles.notes}>
                Frame notes
                <textarea
                  value={currentAnnotation.notes}
                  disabled={reviewLocked || currentAnnotation.status !== "reviewed"}
                  placeholder="Optional ambiguity note"
                  onChange={(event) => {
                    const notes = event.target.value;
                    updateFrameAnnotation(currentFrame, (annotation) => ({ ...annotation, notes }));
                  }}
                />
              </label>
            </section>

            <section className={styles.shortcutHelp}>
              <p className={styles.sectionLabel}>KEYBOARD</p>
              <p><kbd>←</kbd><kbd>→</kbd> step · <kbd>Space</kbd> play · <kbd>Enter</kbd> next</p>
              <p><kbd>1</kbd>–<kbd>9</kbd> accept matching Sol box + next frame</p>
              <p><kbd>P</kbd><kbd>A</kbd><kbd>U</kbd> draw roles · <kbd>V</kbd> copy previous</p>
              <p><kbd>Delete</kbd> remove selected · <kbd>Esc</kbd> cancel tool</p>
            </section>

            <section className={styles.savePanel}>
              <button
                onClick={() => task && void persist(task, false)}
                disabled={!task || saving || !dirty || !taskCanBeSaved(task)}
              >
                {saving ? "Saving…" : "Save now"}
              </button>
              <button
                className={styles.complete}
                onClick={() => void completeReview()}
                disabled={reviewLocked || saving || reviewed !== totalFrames}
              >
                {reviewLocked ? "Review complete" : "Complete review"}
              </button>
              {reviewed === totalFrames && !reviewLocked && (
                <p>All frames are labeled. Complete the review to persist the final frame.</p>
              )}
            </section>
          </aside>
        </section>
      ) : (
        <section className={styles.emptyWorkspace}>
          <strong>{catalogLoading ? "Loading pilot…" : "No task selected"}</strong>
          <p>The workstation loads only pristine human-review tasks and saved human drafts.</p>
        </section>
      )}
    </main>
  );
}
