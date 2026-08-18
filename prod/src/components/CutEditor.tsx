import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";

import {
  activeSuppressionSuggestions,
  applyPaddingToCachedCuts,
  buildFinalCutIntervals,
  createCutDraft,
  cutDraftStorageKey,
  cutDraftStorageKeys,
  effectiveKeptCutIds,
  materializeFinalCutIntervals,
  nextFinalCutTime,
  parseCutDraft,
  playbackFocusCut,
  PLAYBACK_RATES,
  totalFinalCutSeconds,
  suppressionSuggestionState,
  type CutDraft,
  type CutDraftSeed,
  type EditableCut,
} from "@/lib/cut-draft";
import { formatTime, timelinePercent } from "@/lib/edit-list";
import {
  createModelFeedbackBundle,
  modelFeedbackBlob,
  modelFeedbackFilename,
} from "@/lib/model-feedback";
import { isChromeOnIosBrowser } from "@/lib/on-device/browser-support";
import {
  deliverPreparedVideoExport,
  supportsOpfsExport,
  type PreparedVideoExport,
} from "@/lib/on-device/export-delivery";
import type { ExportProgress, VideoExportMode } from "@/lib/on-device/export";
import { modelDisplayName } from "@/lib/on-device/ensemble";
import { requestPlayingSeek } from "@/lib/on-device/player";
import {
  nextSuppressionAfterTime,
  nextSuppressionSuggestion,
  SUPPRESSION_POLICY_DIAGNOSTIC_NAMES,
  SUPPRESSION_POLICY_LABELS,
  type SuppressionPolicyId,
  type SuppressionSuggestion,
} from "@/lib/on-device/suppression-policy";
import { prepareServiceWorkerStreamDownload } from "@/lib/on-device/stream-download";
import type { WakeLockState } from "@/lib/on-device/wake-lock";
import type { ProductAnalysis } from "@/lib/product-analysis";

import styles from "./CutEditor.module.css";

type CutEditorProps = {
  header: ReactNode;
  initialAnalysis: ProductAnalysis;
  sourceFile: File | null;
  sourceError: string | null;
  onAttachSource: (file: File | null) => void;
  onRequestSuppression: () => void;
};

type ExportState = "idle" | "exporting" | "done" | "error";

type BoundarySide = "start" | "end";

type BoundaryDrag = {
  pointerId: number;
  side: BoundarySide;
  left: number;
  width: number;
  windowStart: number;
  windowEnd: number;
};

type TimelineDrag = {
  pointerId: number;
  left: number;
  width: number;
  windowStart: number;
  windowEnd: number;
  startX: number;
  startY: number;
  moved: boolean;
  seekOnTap: boolean;
  tapCutId: string | null;
};

function preciseTime(seconds: number): string {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = safe % 60;
  const prefix = hours
    ? `${hours}:${minutes.toString().padStart(2, "0")}`
    : `${minutes}`;
  return `${prefix}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function roundTime(seconds: number): number {
  return Math.round(seconds * 1000) / 1000;
}

function nextId(prefix: string, ids: string[]): string {
  const used = new Set(ids);
  for (let index = 1; index < 10_000; index += 1) {
    const candidate = `${prefix}${String(index).padStart(3, "0")}`;
    if (!used.has(candidate)) return candidate;
  }
  return `${prefix}${Date.now()}`;
}

function detailWindow(
  cut: EditableCut | null,
  playbackTime: number,
  minimum: number,
  maximum: number,
): { start: number; end: number } {
  const duration = Math.max(0, maximum - minimum);
  const minimumSpan = Math.min(24, duration || 24);
  const center = cut
    ? (cut.keepStart + cut.keepEnd) / 2
    : playbackTime;
  const contentSpan = cut ? cut.keepEnd - cut.keepStart + 10 : minimumSpan;
  const span = Math.min(duration || minimumSpan, Math.max(minimumSpan, contentSpan));
  let start = Math.max(minimum, center - span / 2);
  let end = Math.min(maximum, start + span);
  start = Math.max(minimum, end - span);
  if (end <= start) end = start + 1;
  return { start, end };
}

function downloadFilename(value: string): string {
  const safe = value.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  return `${safe || "volleycut"}.edit-list.json`;
}

function isModelDisagreement(cut: EditableCut): boolean {
  return cut.agreement === "all-labels-v2-only" ||
    cut.agreement === "previous-production-only";
}

function modelAgreementLabel(cut: EditableCut): string {
  if (cut.agreement === "both-models") return "Both models agree";
  if (cut.agreement === "all-labels-v2-only") return "Disagreement · all-labels v2 only";
  if (cut.agreement === "previous-production-only")
    return "Disagreement · previous production only";
  return "Model prediction";
}

export function CutEditor({
  header,
  initialAnalysis,
  sourceFile,
  sourceError,
  onAttachSource,
  onRequestSuppression,
}: CutEditorProps) {
  const analysisStart = initialAnalysis.analysisWindow.start;
  const analysisEnd = initialAnalysis.analysisWindow.end;
  const analysisDuration = analysisEnd - analysisStart;
  const videoRef = useRef<HTMLVideoElement>(null);
  const detailRailRef = useRef<HTMLDivElement>(null);
  const boundaryDragRef = useRef<BoundaryDrag | null>(null);
  const timelineDragRef = useRef<TimelineDrag | null>(null);
  const suppressTimelineClickUntilRef = useRef(0);
  const previewEndRef = useRef<number | null>(null);
  const previewStopRef = useRef<number | null>(null);
  const resumeAfterSeekRef = useRef(false);
  const selectedSeekRef = useRef<{
    cutId: string;
    target: number;
    arrived: boolean;
  } | null>(null);
  const seed = useMemo<CutDraftSeed>(
    () => ({
      analysisId: initialAnalysis.id,
      recordingId: initialAnalysis.recordingId,
      duration: initialAnalysis.duration,
      analysisStart,
      analysisEnd,
      rallies: initialAnalysis.rallies,
      ignoredIntervals: initialAnalysis.ignoredIntervals,
      suppressionContractVersion: initialAnalysis.suppression?.policyContractVersion,
    }),
    [initialAnalysis, analysisStart, analysisEnd],
  );
  const initialDraft = useMemo(() => createCutDraft(seed), [seed]);
  const [draft, setDraft] = useState<CutDraft>(initialDraft);
  const [storageReady, setStorageReady] = useState(false);
  const [storageMessage, setStorageMessage] = useState("Loading on-device draft…");
  const [selectedId, setSelectedId] = useState(initialDraft.cuts[0]?.id ?? "");
  const [selectedSuppressionId, setSelectedSuppressionId] = useState("");
  const [focusLocked, setFocusLocked] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(analysisStart);
  const [isPlaying, setIsPlaying] = useState(false);
  const [editorMessage, setEditorMessage] = useState<string | null>(null);
  const [exportState, setExportState] = useState<ExportState>("idle");
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);
  const [preparedExport, setPreparedExport] = useState<PreparedVideoExport | null>(null);
  const [exportMode, setExportMode] = useState<VideoExportMode>(() =>
    isChromeOnIosBrowser() ? "stream-download" : "compatible",
  );
  const [streamFallbackReason, setStreamFallbackReason] = useState<string | null>(null);
  const [streamDownloadReady, setStreamDownloadReady] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportWakeLock, setExportWakeLock] = useState<WakeLockState>("idle");
  const manualStart = draft.pendingManualStart;
  const ignoreStart = draft.pendingIgnoreStart;
  const ignoreReason = draft.ignoreReason;
  const cutPreviewEnabled = draft.cutPreviewEnabled;

  useEffect(() => {
    const timer = window.setTimeout(() => {
      let restored: CutDraft | null = null;
      try {
        for (const key of cutDraftStorageKeys(seed.analysisId)) {
          const raw = window.localStorage.getItem(key);
          restored = raw ? parseCutDraft(raw, seed) : null;
          if (restored) break;
        }
      } catch {
        // Privacy-restricted browsers can deny storage; editing still works in memory.
      }
      const next = restored ?? initialDraft;
      setDraft(next);
      setSelectedId((current) =>
        next.cuts.some((cut) => cut.id === current) ? current : next.cuts[0]?.id ?? "",
      );
      setStorageMessage(
        restored ? "Restored cached edits on this device" : "New on-device draft",
      );
      setStorageReady(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [initialDraft, seed]);

  useEffect(() => {
    if (!storageReady) return;
    let message = "Saved on this device";
    try {
      window.localStorage.setItem(cutDraftStorageKey(seed.analysisId), JSON.stringify(draft));
    } catch {
      message = "Browser storage unavailable · changes live in this tab";
    }
    const timer = window.setTimeout(() => setStorageMessage(message), 0);
    return () => window.clearTimeout(timer);
  }, [draft, seed.analysisId, storageReady]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = draft.playbackRate;
  }, [draft.playbackRate]);

  useEffect(() => {
    const chromeOnIos = isChromeOnIosBrowser();
    if (!chromeOnIos) return;
    let active = true;
    const workerUrl = new URL(
      `${import.meta.env.BASE_URL}volleycut-export-sw.js`,
      window.location.href,
    ).href;
    void prepareServiceWorkerStreamDownload(workerUrl).then((readiness) => {
      if (!active) return;
      setStreamDownloadReady(readiness.ready);
      if (readiness.ready) {
        setExportMode("stream-download");
        setStreamFallbackReason(null);
      } else {
        setExportMode("opfs");
        setStreamFallbackReason(readiness.reason ?? "Direct download is unavailable.");
      }
    });
    return () => {
      active = false;
    };
  }, []);

  const sortedCuts = useMemo(
    () => [...draft.cuts].sort(
      (left, right) => left.keepStart - right.keepStart || left.keepEnd - right.keepEnd,
    ),
    [draft.cuts],
  );
  const selected = draft.cuts.find((cut) => cut.id === selectedId) ?? sortedCuts[0] ?? null;
  const selectedIndex = selected
    ? sortedCuts.findIndex((cut) => cut.id === selected.id)
    : -1;
  const materialized = useMemo(
    () => materializeFinalCutIntervals(draft, initialAnalysis.suppression),
    [draft, initialAnalysis.suppression],
  );
  const finalIntervals = materialized.intervals;
  const keptSeconds = totalFinalCutSeconds(finalIntervals);
  const noSuppressionSeconds = useMemo(
    () => totalFinalCutSeconds(buildFinalCutIntervals(
      { ...draft, selectedSuppressionPolicy: "none" },
      initialAnalysis.suppression,
    )),
    [draft, initialAnalysis.suppression],
  );
  const effectiveKeptIds = useMemo(
    () => new Set(effectiveKeptCutIds(draft, initialAnalysis.suppression)),
    [draft, initialAnalysis.suppression],
  );
  const suppressionSuggestions = useMemo(
    () => activeSuppressionSuggestions(draft, initialAnalysis.suppression)
      .sort((left, right) => left.start - right.start || left.end - right.end),
    [draft, initialAnalysis.suppression],
  );
  const selectedSuppression = suppressionSuggestions.find(
    (suggestion) => suggestion.id === selectedSuppressionId,
  ) ?? null;
  const selectedSuppressionIndex = selectedSuppression
    ? suppressionSuggestions.findIndex((suggestion) => suggestion.id === selectedSuppression.id)
    : -1;
  const appliedSuppressionCount = suppressionSuggestions.filter(
    (suggestion) => suppressionSuggestionState(suggestion, draft) === "suppressed",
  ).length;
  const keptCount = effectiveKeptIds.size;
  const removedCount = draft.cuts.filter((cut) => !cut.included).length;
  const fullyIgnoredCount = draft.cuts.filter(
    (cut) => cut.included && !effectiveKeptIds.has(cut.id),
  ).length;
  const focus = detailWindow(
    selectedSuppression
      ? {
          id: selectedSuppression.id,
          coreStart: selectedSuppression.start,
          coreEnd: selectedSuppression.end,
          keepStart: selectedSuppression.start,
          keepEnd: selectedSuppression.end,
          confidence: selectedSuppression.score,
          included: true,
          origin: "cached-label",
        }
      : selected,
    playbackTime,
    analysisStart,
    analysisEnd,
  );
  const activeMarkStart = manualStart ?? ignoreStart;
  const overviewCuts = activeMarkStart === null
    ? sortedCuts
    : sortedCuts.filter((cut) => cut.keepEnd >= activeMarkStart);
  const visibleIgnoredIntervals = draft.ignoredIntervals.filter(
    (interval) => interval.end > analysisStart && interval.start < analysisEnd,
  );
  const overviewIgnoredIntervals = activeMarkStart === null
    ? visibleIgnoredIntervals
    : visibleIgnoredIntervals.filter((interval) => interval.end >= activeMarkStart);
  const manualCuts = sortedCuts.filter((cut) => cut.origin === "manual");
  const lowConfidenceCuts = sortedCuts.filter(
    (cut) => cut.origin === "cached-label" &&
      cut.included &&
      effectiveKeptIds.has(cut.id) &&
      (isModelDisagreement(cut) || cut.confidence < draft.confidenceReviewThreshold),
  );
  const disagreementCount = lowConfidenceCuts.filter(isModelDisagreement).length;
  const reviewedCutIds = useMemo(() => new Set(draft.reviewedCutIds), [draft.reviewedCutIds]);
  const unreviewedLowConfidenceCuts = lowConfidenceCuts.filter(
    (cut) => !reviewedCutIds.has(cut.id),
  );
  const reviewedLowConfidenceCount = lowConfidenceCuts.length -
    unreviewedLowConfidenceCuts.length;
  const selectedReviewCandidate = selected
    ? lowConfidenceCuts.find((cut) => cut.id === selected.id) ?? null
    : null;
  const selectedIsReviewed = Boolean(selected && reviewedCutIds.has(selected.id));
  const exportPercent = exportProgress && exportProgress.totalSeconds > 0
    ? Math.min(100, Math.max(0, exportProgress.completedSeconds / exportProgress.totalSeconds * 100))
    : 0;
  const exportRate = exportProgress && exportProgress.elapsedSeconds > 0
    ? exportProgress.completedSeconds / exportProgress.elapsedSeconds
    : null;
  const exportEta = exportProgress && exportRate && exportRate > 0
    ? Math.max(0, exportProgress.totalSeconds - exportProgress.completedSeconds) / exportRate
    : null;
  const directDiskSupported = "showSaveFilePicker" in window;
  const opfsSupported = supportsOpfsExport();
  const chromeOnIos = isChromeOnIosBrowser();
  const encodingSupported = "VideoEncoder" in window && "AudioEncoder" in window;
  const exportStorageReady = exportMode === "stream-download"
    ? streamDownloadReady
    : exportMode === "opfs"
      ? opfsSupported
      : directDiskSupported || opfsSupported;
  const localExportSupported = Boolean(sourceFile) && encodingSupported && exportStorageReady;

  function updateDraft(mutate: (current: CutDraft) => CutDraft) {
    if (exportState !== "exporting") {
      setPreparedExport(null);
      setExportState("idle");
      setExportProgress(null);
      setExportError(null);
    }
    setDraft((current) => ({
      ...mutate(current),
      updatedAt: new Date().toISOString(),
    }));
  }

  function updateCut(id: string, mutate: (cut: EditableCut) => EditableCut) {
    updateDraft((current) => ({
      ...current,
      cuts: current.cuts.map((cut) => (cut.id === id ? mutate(cut) : cut)),
      userTouchedCutIds: current.cuts.some(
        (cut) => cut.id === id && cut.origin === "cached-label",
      )
        ? [...new Set([...current.userTouchedCutIds, id])]
        : current.userTouchedCutIds,
    }));
  }

  function setPlaybackRate(playbackRate: CutDraft["playbackRate"]) {
    updateDraft((current) => ({ ...current, playbackRate }));
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
  }

  function setConfidenceReviewThreshold(percent: number) {
    if (!Number.isFinite(percent)) return;
    const threshold = Math.max(0, Math.min(100, percent)) / 100;
    updateDraft((current) => ({ ...current, confidenceReviewThreshold: threshold }));
  }

  function seekTo(
    time: number,
    resumePlayback?: boolean,
    selectedCutId?: string,
  ) {
    const clamped = Math.max(analysisStart, Math.min(analysisEnd, time));
    setPlaybackTime(clamped);
    const video = videoRef.current;
    selectedSeekRef.current = video && selectedCutId
      ? { cutId: selectedCutId, target: clamped, arrived: false }
      : null;
    if (!video) return;
    const shouldResume = resumePlayback ?? !video.paused;
    if (shouldResume) {
      resumeAfterSeekRef.current = true;
      const waitingForSeek = requestPlayingSeek(video, clamped);
      if (!waitingForSeek && selectedSeekRef.current) {
        selectedSeekRef.current.arrived = true;
      }
    } else {
      const alreadyAtTarget = Math.abs(video.currentTime - clamped) < 0.01 && !video.seeking;
      video.currentTime = clamped;
      if (alreadyAtTarget && selectedSeekRef.current) {
        selectedSeekRef.current.arrived = true;
      }
    }
  }

  function trackPlayback(time: number) {
    setPlaybackTime(time);
    const selectedSeek = selectedSeekRef.current;
    if (selectedSeek) {
      const atTarget = Math.abs(time - selectedSeek.target) <= 0.25;
      if (atTarget) selectedSeek.arrived = true;
      if (!selectedSeek.arrived || atTarget) {
        setSelectedId(selectedSeek.cutId);
        return;
      }
      selectedSeekRef.current = null;
    }
    if (previewEndRef.current !== null) return;
    if (selectedSuppressionId) return;
    if (focusLocked) return;
    const reachedCut = playbackFocusCut(sortedCuts, time);
    if (reachedCut) {
      setSelectedId((current) => current === reachedCut.id ? current : reachedCut.id);
    }
  }

  function selectCut(cut: EditableCut) {
    setSelectedSuppressionId("");
    setSelectedId(cut.id);
    seekTo(cut.keepStart, undefined, cut.id);
    setEditorMessage(null);
  }

  function selectSuppression(suggestion: SuppressionSuggestion) {
    const video = videoRef.current;
    video?.pause();
    const overlapping = sortedCuts.find(
      (cut) => cut.coreStart < suggestion.end && suggestion.start < cut.coreEnd,
    );
    if (overlapping) setSelectedId(overlapping.id);
    setSelectedSuppressionId(suggestion.id);
    seekTo(Math.max(analysisStart, suggestion.start - 2), false);
    setEditorMessage(
      `${preciseTime(suggestion.start)}–${preciseTime(suggestion.end)} · review without changing its current decision.`,
    );
  }

  function setSuppressionPolicy(policy: SuppressionPolicyId) {
    updateDraft((current) => ({ ...current, selectedSuppressionPolicy: policy }));
    if (policy === "none") setSelectedSuppressionId("");
    setEditorMessage(
      policy === "none"
        ? "Suppression suggestions are dormant; saved review choices are preserved."
        : `${SUPPRESSION_POLICY_LABELS[policy]} suppression selected. Untouched suggestions default to Suppress.`,
    );
  }

  function setSuppressionDecision(
    suggestion: SuppressionSuggestion,
    decision: "keep" | "suppress",
  ) {
    updateDraft((current) => ({
      ...current,
      suppressionDecisionOverrides: {
        ...current.suppressionDecisionOverrides,
        [suggestion.logicalId]: decision,
      },
    }));
    setEditorMessage(
      decision === "suppress"
        ? "Suggestion applied to the derived export. Manual ranges still win."
        : "Suggestion kept in the export and retained for review.",
    );
  }

  function navigateSuppression(offset: -1 | 1) {
    const next = nextSuppressionSuggestion(
      suppressionSuggestions,
      selectedSuppression?.id ?? "",
      offset,
    );
    if (next) selectSuppression(next);
  }

  function reviewNextSuppression() {
    const timelineAnchor = selectedSuppression
      ? Math.max(playbackTime, selectedSuppression.start)
      : playbackTime;
    const next = nextSuppressionAfterTime(suppressionSuggestions, timelineAnchor);
    if (next) selectSuppression(next);
  }

  async function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    previewEndRef.current = null;
    previewStopRef.current = null;
    if (video.paused) {
      if (cutPreviewEnabled) {
        const target = nextFinalCutTime(finalIntervals, video.currentTime)
          ?? finalIntervals[0]?.start;
        if (target === undefined) return;
        if (Math.abs(target - video.currentTime) > 0.01) {
          seekTo(target, true);
          return;
        }
      }
      await video.play();
    } else {
      video.pause();
    }
  }

  function previewSelected() {
    if (!selected || !videoRef.current) return;
    previewStopRef.current = null;
    previewEndRef.current = selected.keepEnd;
    setSelectedId(selected.id);
    seekTo(selected.keepStart, true, selected.id);
  }

  function setBoundary(id: string, side: BoundarySide, value: number) {
    updateCut(id, (cut) => {
      if (side === "start") {
        return {
          ...cut,
          keepStart: roundTime(
            Math.max(analysisStart, Math.min(cut.coreStart, value)),
          ),
        };
      }
      return {
        ...cut,
        keepEnd: roundTime(
          Math.max(cut.coreEnd, Math.min(analysisEnd, value)),
        ),
      };
    });
  }

  function beginBoundaryDrag(
    side: BoundarySide,
    event: ReactPointerEvent<HTMLButtonElement>,
  ) {
    if (!selected || !detailRailRef.current) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const bounds = detailRailRef.current.getBoundingClientRect();
    boundaryDragRef.current = {
      pointerId: event.pointerId,
      side,
      left: bounds.left,
      width: bounds.width,
      windowStart: focus.start,
      windowEnd: focus.end,
    };
  }

  function moveBoundary(event: ReactPointerEvent<HTMLButtonElement>) {
    const drag = boundaryDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !selected) return;
    event.preventDefault();
    const ratio = Math.max(0, Math.min(1, (event.clientX - drag.left) / drag.width));
    const time = drag.windowStart + ratio * (drag.windowEnd - drag.windowStart);
    setBoundary(selected.id, drag.side, time);
  }

  function endBoundaryDrag(event: ReactPointerEvent<HTMLButtonElement>) {
    if (boundaryDragRef.current?.pointerId === event.pointerId) {
      boundaryDragRef.current = null;
    }
  }

  function nudgeBoundary(side: BoundarySide, delta: number) {
    if (!selected) return;
    setBoundary(
      selected.id,
      side,
      (side === "start" ? selected.keepStart : selected.keepEnd) + delta,
    );
  }

  function resetSelectedPadding() {
    if (!selected) return;
    updateCut(selected.id, (cut) => ({
      ...cut,
      keepStart: Math.max(
        analysisStart,
        cut.coreStart - draft.beforePaddingSeconds,
      ),
      keepEnd: Math.min(
        analysisEnd,
        cut.coreEnd + draft.afterPaddingSeconds,
      ),
    }));
  }

  function setGlobalPadding(side: "before" | "after", paddingSeconds: number) {
    updateDraft((current) => {
      const before = side === "before" ? paddingSeconds : current.beforePaddingSeconds;
      const after = side === "after" ? paddingSeconds : current.afterPaddingSeconds;
      return applyPaddingToCachedCuts(
        current,
        before,
        after,
        initialAnalysis.duration,
        analysisStart,
        analysisEnd,
      );
    });
    setEditorMessage(
      `Applied ${paddingSeconds.toFixed(1)} seconds ${side} every model-predicted cut.`,
    );
  }

  function setJoinGapSeconds(seconds: number) {
    if (!Number.isFinite(seconds)) return;
    updateDraft((current) => ({
      ...current,
      joinGapSeconds: Math.max(0, Math.min(10, seconds)),
    }));
    setEditorMessage(`Joining final export gaps shorter than ${seconds.toFixed(1)} seconds.`);
  }

  function toggleSelected() {
    if (!selected) return;
    updateCut(selected.id, (cut) => ({ ...cut, included: !cut.included }));
  }

  function toggleCutPreview(enabled: boolean) {
    updateDraft((current) => ({ ...current, cutPreviewEnabled: enabled }));
    previewEndRef.current = null;
    previewStopRef.current = null;
    if (!enabled || !videoRef.current) return;
    const target = nextFinalCutTime(finalIntervals, videoRef.current.currentTime)
      ?? finalIntervals[0]?.start;
    if (target !== undefined && Math.abs(target - videoRef.current.currentTime) > 0.01) {
      seekTo(target);
    }
  }

  function markManualBoundary() {
    if (manualStart === null) {
      updateDraft((current) => ({
        ...current,
        pendingManualStart: playbackTime,
        pendingIgnoreStart: null,
      }));
      setEditorMessage(`Missed cut starts at ${preciseTime(playbackTime)}.`);
      return;
    }
    const start = Math.min(manualStart, playbackTime);
    const end = Math.max(manualStart, playbackTime);
    if (end - start < 0.1) {
      setEditorMessage("A manual cut must be at least 0.1 seconds long.");
      return;
    }
    const id = nextId("M", draft.cuts.map((cut) => cut.id));
    updateDraft((current) => ({
      ...current,
      pendingManualStart: null,
      cuts: [
        ...current.cuts,
        {
          id,
          coreStart: roundTime(start),
          coreEnd: roundTime(end),
          keepStart: roundTime(start),
          keepEnd: roundTime(end),
          confidence: 1,
          included: true,
          origin: "manual",
        },
      ],
    }));
    setSelectedId(id);
    setEditorMessage(`Added ${id} from ${preciseTime(start)} to ${preciseTime(end)}.`);
  }

  function deleteManualCut(id: string) {
    const deletedIndex = sortedCuts.findIndex((cut) => cut.id === id);
    const deleted = sortedCuts[deletedIndex];
    if (!deleted || deleted.origin !== "manual") return;
    const nextCuts = sortedCuts.filter((cut) => cut.id !== id);
    const fallback = nextCuts[Math.max(0, deletedIndex - 1)] ?? nextCuts[0] ?? null;
    updateDraft((current) => ({
      ...current,
      cuts: current.cuts.filter((cut) => cut.id !== id),
    }));
    if (selected?.id === id) setSelectedId(fallback?.id ?? "");
    setEditorMessage(`Removed added cut ${id}.`);
  }

  function markIgnoredBoundary() {
    if (ignoreStart === null) {
      updateDraft((current) => ({
        ...current,
        pendingIgnoreStart: playbackTime,
        pendingManualStart: null,
      }));
      setEditorMessage(`Ignored section starts at ${preciseTime(playbackTime)}.`);
      return;
    }
    const start = Math.min(ignoreStart, playbackTime);
    const end = Math.max(ignoreStart, playbackTime);
    if (end - start < 0.1) {
      setEditorMessage("An ignored section must be at least 0.1 seconds long.");
      return;
    }
    const id = nextId("I", draft.ignoredIntervals.map((interval) => interval.id));
    updateDraft((current) => ({
      ...current,
      pendingIgnoreStart: null,
      ignoredIntervals: [
        ...current.ignoredIntervals,
        { id, start: roundTime(start), end: roundTime(end), reason: ignoreReason },
      ],
    }));
    setEditorMessage(`Ignored ${preciseTime(start)} to ${preciseTime(end)}.`);
  }

  function removeIgnoredInterval(id: string) {
    updateDraft((current) => ({
      ...current,
      ignoredIntervals: current.ignoredIntervals.filter((interval) => interval.id !== id),
    }));
  }

  function navigateCut(offset: number) {
    if (selectedIndex < 0 || sortedCuts.length === 0) return;
    const index = Math.max(0, Math.min(sortedCuts.length - 1, selectedIndex + offset));
    selectCut(sortedCuts[index]);
  }

  function setCutReviewed(id: string, reviewed: boolean) {
    setDraft((current) => ({
      ...current,
      updatedAt: new Date().toISOString(),
      reviewedCutIds: reviewed
        ? [...new Set([...current.reviewedCutIds, id])]
        : current.reviewedCutIds.filter((cutId) => cutId !== id),
    }));
  }

  function toggleSelectedReviewed() {
    if (!selectedReviewCandidate) return;
    const reviewed = !selectedIsReviewed;
    setCutReviewed(selectedReviewCandidate.id, reviewed);
    setEditorMessage(
      reviewed
        ? `${selectedReviewCandidate.id} marked reviewed.`
        : `${selectedReviewCandidate.id} returned to the review queue.`,
    );
  }

  function reviewNextLowConfidenceCut() {
    if (unreviewedLowConfidenceCuts.length === 0) return;
    const currentIndex = lowConfidenceCuts.findIndex((cut) => cut.id === selected?.id);
    const markCurrentReviewed = Boolean(
      selectedReviewCandidate && !reviewedCutIds.has(selectedReviewCandidate.id),
    );
    if (markCurrentReviewed && selectedReviewCandidate) {
      setCutReviewed(selectedReviewCandidate.id, true);
    }
    const remaining = unreviewedLowConfidenceCuts.filter(
      (cut) => cut.id !== selectedReviewCandidate?.id,
    );
    if (remaining.length === 0) {
      setEditorMessage(
        markCurrentReviewed && selectedReviewCandidate
          ? `${selectedReviewCandidate.id} marked reviewed. Review queue complete.`
          : "Review queue complete.",
      );
      return;
    }
    const remainingIds = new Set(remaining.map((cut) => cut.id));
    let next: EditableCut | undefined;
    if (currentIndex >= 0) {
      for (let offset = 1; offset <= lowConfidenceCuts.length; offset += 1) {
        const candidate = lowConfidenceCuts[
          (currentIndex + offset) % lowConfidenceCuts.length
        ];
        if (remainingIds.has(candidate.id)) {
          next = candidate;
          break;
        }
      }
    } else {
      next = remaining.find((cut) => cut.keepStart >= playbackTime) ?? remaining[0];
    }
    if (!next) return;
    selectCut(next);
    const reviewedPrefix = markCurrentReviewed && selectedReviewCandidate
      ? `${selectedReviewCandidate.id} marked reviewed. `
      : "";
    setEditorMessage(
      reviewedPrefix + (isModelDisagreement(next)
        ? `${next.id} was detected by only one model. Validate it and remove it if it is not a rally.`
        : `${next.id} has ${Math.round(next.confidence * 100)}% model confidence. Review it and remove it if needed.`),
    );
  }

  function timelineSeekTime(
    clientX: number,
    drag: Pick<TimelineDrag, "left" | "width" | "windowStart" | "windowEnd">,
  ) {
    const ratio = Math.max(0, Math.min(1, (clientX - drag.left) / drag.width));
    setSelectedSuppressionId("");
    seekTo(drag.windowStart + ratio * (drag.windowEnd - drag.windowStart));
  }

  function beginTimelineSeek(
    event: ReactPointerEvent<HTMLDivElement>,
    windowStart: number,
    windowEnd: number,
  ) {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const target = event.target instanceof Element ? event.target : null;
    const tapCutId = target
      ?.closest<HTMLButtonElement>("[data-overview-cut-id]")
      ?.dataset.overviewCutId ?? null;
    timelineDragRef.current = {
      pointerId: event.pointerId,
      left: bounds.left,
      width: bounds.width,
      windowStart,
      windowEnd,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
      seekOnTap: event.target === event.currentTarget,
      tapCutId,
    };
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Seeking still works in browsers without pointer capture support.
    }
  }

  function moveTimelineSeek(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = timelineDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (!drag.moved) {
      const horizontalDistance = Math.abs(event.clientX - drag.startX);
      const verticalDistance = Math.abs(event.clientY - drag.startY);
      if (horizontalDistance < 4 || verticalDistance > horizontalDistance) return;
      drag.moved = true;
    }
    event.preventDefault();
    timelineSeekTime(event.clientX, drag);
  }

  function endTimelineSeek(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = timelineDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (drag.moved) {
      timelineSeekTime(event.clientX, drag);
      suppressTimelineClickUntilRef.current = Date.now() + 800;
    } else if (drag.tapCutId) {
      suppressTimelineClickUntilRef.current = Date.now() + 800;
      const tappedCut = draft.cuts.find((cut) => cut.id === drag.tapCutId);
      if (tappedCut) selectCut(tappedCut);
    } else if (drag.seekOnTap) {
      timelineSeekTime(event.clientX, drag);
    }
    timelineDragRef.current = null;
  }

  function cancelTimelineSeek(event: ReactPointerEvent<HTMLDivElement>) {
    if (timelineDragRef.current?.pointerId === event.pointerId) {
      timelineDragRef.current = null;
    }
  }

  function resetDraft() {
    if (!window.confirm("Discard every on-device correction for this analysis?")) return;
    try {
      window.localStorage.removeItem(cutDraftStorageKey(seed.analysisId));
    } catch {
      // The in-memory reset still succeeds if storage is unavailable.
    }
    setPreparedExport(null);
    setExportState("idle");
    setExportProgress(null);
    setExportError(null);
    setDraft({ ...initialDraft, updatedAt: new Date().toISOString() });
    setSelectedId(initialDraft.cuts[0]?.id ?? "");
    setSelectedSuppressionId("");
    setEditorMessage("Reset to the inferred model ranges.");
  }

  function downloadEditList() {
    const payload = {
      schemaVersion: 2,
      source: {
        analysisId: initialAnalysis.id,
        recordingId: initialAnalysis.recordingId,
        filename: initialAnalysis.sourceFilename,
        durationSeconds: initialAnalysis.duration,
        gameStartSeconds: analysisStart,
        gameEndSeconds: analysisEnd,
      },
      generatedAt: new Date().toISOString(),
      cuts: draft.cuts,
      reviewedCutIds: draft.reviewedCutIds,
      ignoredIntervals: draft.ignoredIntervals,
      suppression: {
        selectedPolicy: draft.selectedSuppressionPolicy,
        artifact: initialAnalysis.suppression
          ? {
              modelId: initialAnalysis.suppression.modelId,
              artifactSha256: initialAnalysis.suppression.artifactSha256,
              weightsSha256: initialAnalysis.suppression.weightsSha256,
              decoderVersion: initialAnalysis.suppression.decoderVersion,
              policyContractVersion: initialAnalysis.suppression.policyContractVersion,
            }
          : null,
        suggestions: initialAnalysis.suppression?.suggestions ?? [],
        decisionOverrides: draft.suppressionDecisionOverrides,
        userTouchedCutIds: draft.userTouchedCutIds,
        decisions: (initialAnalysis.suppression?.suggestions ?? []).map((suggestion) => ({
          suggestionId: suggestion.id,
          logicalId: suggestion.logicalId,
          state: suppressionSuggestionState(suggestion, draft),
        })),
      },
      finalIntervals,
      finalIntervalProvenance: materialized.provenance,
    };
    const url = URL.createObjectURL(
      new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = downloadFilename(initialAnalysis.sourceFilename);
    link.click();
    URL.revokeObjectURL(url);
  }

  async function shareModelFeedback() {
    try {
      const bundle = createModelFeedbackBundle(
        initialAnalysis,
        draft,
        finalIntervals,
      );
      const blob = modelFeedbackBlob(bundle);
      const filename = modelFeedbackFilename(initialAnalysis.sourceFilename);
      const file = new File([blob], filename, { type: blob.type });
      if (navigator.share && navigator.canShare?.({ files: [file] })) {
        try {
          await navigator.share({
            files: [file],
            title: `VolleyCut model feedback · ${initialAnalysis.sourceFilename}`,
          });
          setEditorMessage(
            initialAnalysis.features
              ? "Shared model feedback with features, inference, and corrections."
              : "Shared inference and corrections; this older analysis has no retained feature matrix.",
          );
          return;
        } catch (cause) {
          if (cause instanceof DOMException && cause.name === "AbortError") return;
          // Fall through to a normal browser download when native sharing fails.
        }
      }
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      setEditorMessage(
        initialAnalysis.features
          ? "Downloaded model feedback with features, inference, and corrections."
          : "Downloaded inference and corrections; this older analysis has no retained feature matrix.",
      );
    } catch (cause) {
      setEditorMessage(
        `Could not create model feedback: ${cause instanceof Error ? cause.message : String(cause)}`,
      );
    }
  }

  async function exportVideo() {
    if (!sourceFile || exportState === "exporting" || finalIntervals.length === 0) return;
    const requestedMode = exportMode;
    let directStreamFailure: string | null = null;
    setExportError(null);
    setExportState("exporting");
    setExportProgress({
      completedSeconds: 0,
      totalSeconds: keptSeconds,
      elapsedSeconds: 0,
      detail: "Preparing original local video",
    });
    try {
      const { exportRawQualityReel } = await import("@/lib/on-device/export");
      const runExport = (mode: VideoExportMode) =>
        exportRawQualityReel(
          sourceFile,
          finalIntervals,
          setExportProgress,
          initialAnalysis.duration,
          setExportWakeLock,
          mode,
        );
      let prepared: PreparedVideoExport | null;
      try {
        prepared = await runExport(requestedMode);
      } catch (cause) {
        if (
          !chromeOnIos ||
          requestedMode !== "stream-download" ||
          !opfsSupported ||
          (cause instanceof DOMException && cause.name === "AbortError")
        ) {
          throw cause;
        }
        directStreamFailure = cause instanceof Error ? cause.message : String(cause);
        setExportMode("opfs");
        setStreamFallbackReason(directStreamFailure);
        setExportProgress({
          completedSeconds: 0,
          totalSeconds: keptSeconds,
          elapsedSeconds: 0,
          detail: "Direct download failed; retrying in private device storage",
        });
        prepared = await runExport("opfs");
      }
      setPreparedExport(prepared);
      setExportState("done");
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") {
        setExportState("idle");
        setExportProgress(null);
        return;
      }
      const message = cause instanceof Error ? cause.message : String(cause);
      setExportError(
        directStreamFailure
          ? `Direct download failed (${directStreamFailure}). The private-storage fallback also failed: ${message}`
          : message,
      );
      setExportState("error");
    }
  }

  async function deliverExport() {
    if (!preparedExport) return;
    setExportError(null);
    try {
      await deliverPreparedVideoExport(preparedExport);
      setPreparedExport(null);
      setExportState("done");
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setExportError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  return (
    <main className={styles.page}>
      {header}

      <section className={styles.sourcePicker} aria-label="Local inference source">
        <div className={styles.sourceMeta}>
          <span>LOCAL VIDEO</span>
          <strong>{initialAnalysis.sourceFilename}</strong>
        </div>
        <div className={styles.sourceMeta}>
          <span>INFERENCE</span>
          <strong title={initialAnalysis.modelId}>
            {modelDisplayName(initialAnalysis.modelId)} · {initialAnalysis.rallies.length} ranges
          </strong>
        </div>
        <span className={styles.storageState} data-ready={storageReady || undefined}>
          <i /> {storageMessage}
        </span>
        {!sourceFile && (
          <label className={styles.attachSource}>
            Reconnect source for playback &amp; export
            <input
              type="file"
              accept="video/*,.mkv,.webm,.mov,.mp4,.m4v"
              onChange={(event) => onAttachSource(event.currentTarget.files?.[0] ?? null)}
            />
          </label>
        )}
      </section>

      {sourceError && <p className={styles.sourceError}>{sourceError}</p>}

      <section className={styles.editorShell}>
        <aside className={styles.summaryCard} aria-label="Final edit settings">
          <div className={styles.summaryStats}>
            <span>FINAL EDIT LIST</span>
            <strong>{formatTime(keptSeconds)}</strong>
            <div>
              <p>{keptCount} kept · {removedCount} removed</p>
              {fullyIgnoredCount > 0 && <p>{fullyIgnoredCount} enabled rallies fully ignored</p>}
              <p>{draft.ignoredIntervals.length} ignored source sections</p>
              <p>Duration includes padding and joined short gaps; ignored time is excluded</p>
            </div>
          </div>
          <div className={styles.suppressionControls}>
            <div>
              <span>FALSE-POSITIVE SUPPRESSION</span>
              <strong>
                {SUPPRESSION_POLICY_LABELS[draft.selectedSuppressionPolicy]}
              </strong>
            </div>
            {initialAnalysis.suppression ? (
              <>
                <label
                  className={styles.suppressionSelect}
                  htmlFor="suppression-policy"
                >
                  <span>Suppression level</span>
                  <select
                    id="suppression-policy"
                    value={draft.selectedSuppressionPolicy}
                    onChange={(event) => setSuppressionPolicy(
                      event.currentTarget.value as SuppressionPolicyId,
                    )}
                    title={draft.selectedSuppressionPolicy === "none"
                      ? "Preserve the existing production output"
                      : SUPPRESSION_POLICY_DIAGNOSTIC_NAMES[draft.selectedSuppressionPolicy]}
                  >
                  {(initialAnalysis.suppression.identicalPolicyResults
                    ? ["none", "conservative"] as const
                    : ["none", "conservative", "balanced", "aggressive"] as const
                  ).map((policy) => (
                    <option key={policy} value={policy}>
                      {initialAnalysis.suppression?.identicalPolicyResults && policy === "conservative"
                        ? "Suppression suggestions"
                        : SUPPRESSION_POLICY_LABELS[policy]}
                    </option>
                  ))}
                  </select>
                </label>
                {draft.selectedSuppressionPolicy === "none" ? (
                  <small>Choose a suppression level to automatically remove its suggested false positives.</small>
                ) : suppressionSuggestions.length === 0 ? (
                  <small>No suppression suggestions for this game.</small>
                ) : (
                  <small>
                    Untouched suggestions are automatically suppressed. Choose Keep while reviewing to restore one.
                  </small>
                )}
              </>
            ) : (
              <>
                <small>
                  This saved analysis has no retained suppression layer. No suppression remains fully usable.
                </small>
                {sourceFile && (
                  <button type="button" onClick={onRequestSuppression}>
                    Add suppression from cached features
                  </button>
                )}
              </>
            )}
          </div>
          <div className={styles.paddingControls}>
            <div className={styles.paddingControl}>
              <label htmlFor="cut-padding-before">
                <span>Before</span>
                <output>{draft.beforePaddingSeconds.toFixed(1)}s</output>
              </label>
              <input
                id="cut-padding-before"
                aria-label="Padding before each inferred cut"
                type="range"
                min="0"
                max="10"
                step="0.5"
                value={draft.beforePaddingSeconds}
                onChange={(event) => setGlobalPadding("before", Number(event.currentTarget.value))}
              />
              <div><span>0s</span><span>10s</span></div>
            </div>
            <div className={styles.paddingControl}>
              <label htmlFor="cut-padding-after">
                <span>After</span>
                <output>{draft.afterPaddingSeconds.toFixed(1)}s</output>
              </label>
              <input
                id="cut-padding-after"
                aria-label="Padding after each inferred cut"
                type="range"
                min="0"
                max="10"
                step="0.5"
                value={draft.afterPaddingSeconds}
                onChange={(event) => setGlobalPadding("after", Number(event.currentTarget.value))}
              />
              <div><span>0s</span><span>10s</span></div>
            </div>
            <div className={styles.paddingControl}>
              <label htmlFor="cut-join-gap">
                <span>Join gaps under</span>
                <output>{draft.joinGapSeconds.toFixed(1)}s</output>
              </label>
              <input
                id="cut-join-gap"
                aria-label="Join final export gaps shorter than"
                type="range"
                min="0"
                max="10"
                step="0.5"
                value={draft.joinGapSeconds}
                onChange={(event) => setJoinGapSeconds(Number(event.currentTarget.value))}
              />
              <div><span>Off</span><span>10s</span></div>
            </div>
            <small>Padding applies to inferred cuts. Light gray gaps are retained when they are shorter than the join setting.</small>
          </div>
          <label className={styles.cutPreviewToggle}>
            <input
              type="checkbox"
              checked={cutPreviewEnabled}
              onChange={(event) => toggleCutPreview(event.currentTarget.checked)}
            />
            <span>
              <strong>Play final cut only</strong>
              <small>Skip removed rallies, ignored sections, and unselected gaps at or above the join setting.</small>
            </span>
          </label>
          {chromeOnIos && (
            <div className={`${styles.cutPreviewToggle} ${styles.streamExportStatus}`}>
              <span>
                <strong>
                  {exportMode === "stream-download"
                    ? "Direct download enabled"
                    : "Private-storage fallback enabled"}
                </strong>
                <small>
                  {streamFallbackReason
                    ? `The Service Worker download failed or was unavailable (${streamFallbackReason}). This export will use OPFS, then offer Share or save.`
                    : "Chrome on iOS streams directly to Downloads by default. If that fails, VolleyCut automatically retries once using OPFS."}
                </small>
              </span>
            </div>
          )}
          <div className={styles.summaryActions}>
            <button
              type="button"
              className={styles.exportButton}
              disabled={
                exportState === "exporting" || finalIntervals.length === 0 || !localExportSupported
              }
              title={
                localExportSupported
                  ? undefined
                  : "MP4 export requires video/audio WebCodecs encoders and writable local storage."
              }
              onClick={() => void (preparedExport ? deliverExport() : exportVideo())}
            >
              {exportState === "exporting"
                ? "Encoding MP4…"
                : preparedExport
                  ? "Share or save MP4"
                  : exportMode === "stream-download"
                    ? "Stream MP4 to Downloads"
                  : directDiskSupported
                    ? "Save MP4 video"
                    : "Create MP4 video"}
            </button>
            <button type="button" className={styles.quietButton} onClick={downloadEditList}>
              Download JSON edit list
            </button>
            <button
              type="button"
              className={styles.quietButton}
              onClick={() => void shareModelFeedback()}
            >
              Share / download model feedback
            </button>
            <button type="button" className={styles.quietButton} onClick={resetDraft}>
              Reset inferred edits
            </button>
          </div>
          <div className={styles.exportDetails} aria-live="polite">
            <p>
              Exports the final edit at the original dimensions using a very-high-quality AVC/AAC encode.
              Video data stays on this device. {chromeOnIos
                ? "Chrome on iOS streams directly by default and retries with private browser storage only if the stream fails."
                : "Desktop and Android use the standard native file or private browser storage path."}
            </p>
            <p>
              Model feedback JSON contains source-aligned features, probability traces, untouched
              inference ranges, corrections, and final ranges—never video bytes. Disabled model
              ranges are labeled false positives; included manual ranges are labeled false negatives.
            </p>
            {!initialAnalysis.features && (
              <strong>
                This saved analysis predates feature capture. Its feedback file will still contain
                inference and corrections, but not the feature matrix.
              </strong>
            )}
            {!localExportSupported && (
              <strong>
                MP4 export requires video/audio WebCodecs encoders and writable local storage.
              </strong>
            )}
            {exportProgress && exportState === "exporting" && (
              <div className={styles.exportProgress}>
                <span>{exportProgress.detail}</span>
                <strong>{exportPercent.toFixed(0)}%</strong>
                <div><i style={{ width: `${exportPercent}%` }} /></div>
                <dl className={styles.exportTiming}>
                  <div>
                    <dt>Elapsed</dt>
                    <dd>{preciseTime(exportProgress.elapsedSeconds)}</dd>
                  </div>
                  <div>
                    <dt>Estimated remaining</dt>
                    <dd>
                      {exportEta === null
                        ? "Estimating…"
                        : exportEta <= 1
                          ? "Finalizing…"
                          : `About ${preciseTime(exportEta)}`}
                    </dd>
                  </div>
                </dl>
                <small>
                  {exportRate && exportRate > 0
                    ? `${exportRate.toFixed(2)}× real-time encoding`
                    : "Measuring encoding speed and ETA…"}
                  {exportWakeLock === "active" ? " · screen awake" : ""}
                </small>
              </div>
            )}
            {preparedExport && (
              <strong>
                Encoding is complete in private device storage. Tap Share or save MP4 to open the
                iOS share sheet or download the file.
              </strong>
            )}
            {exportState === "done" && !preparedExport && (
              <strong>
                {exportMode === "stream-download" ? "MP4 stream completed" : "MP4 export completed"}
                {exportProgress ? ` in ${preciseTime(exportProgress.elapsedSeconds)}` : ""}.
              </strong>
            )}
            {exportError && <strong className={styles.exportError}>{exportError}</strong>}
          </div>
        </aside>

        <div className={styles.playerColumn}>
          <div
            className={styles.videoStage}
            style={{ aspectRatio: `${initialAnalysis.width} / ${initialAnalysis.height}` }}
          >
            {initialAnalysis.videoUrl ? (
              <video
                ref={videoRef}
                src={initialAnalysis.videoUrl}
                preload="metadata"
                playsInline
                controls
                onLoadedMetadata={(event) => {
                  event.currentTarget.currentTime = analysisStart;
                  setPlaybackTime(analysisStart);
                }}
                onTimeUpdate={(event) => {
                  const time = event.currentTarget.currentTime;
                  if (time < analysisStart - 0.01) {
                    seekTo(analysisStart, !event.currentTarget.paused);
                    return;
                  }
                  if (time >= analysisEnd - 0.01) {
                    resumeAfterSeekRef.current = false;
                    selectedSeekRef.current = null;
                    previewStopRef.current = null;
                    event.currentTarget.pause();
                    if (Math.abs(time - analysisEnd) > 0.001) {
                      event.currentTarget.currentTime = analysisEnd;
                    }
                    previewEndRef.current = null;
                    trackPlayback(analysisEnd);
                    return;
                  }
                  if (previewStopRef.current !== null) {
                    resumeAfterSeekRef.current = false;
                    event.currentTarget.pause();
                    setPlaybackTime(previewStopRef.current);
                    return;
                  }
                  const selectedSeek = selectedSeekRef.current;
                  const previewEnd = previewEndRef.current;
                  const previewSeekArrived = !selectedSeek ||
                    selectedSeek.arrived ||
                    Math.abs(time - selectedSeek.target) <= 0.25;
                  if (
                    previewEnd !== null &&
                    previewSeekArrived &&
                    time >= previewEnd - 0.01
                  ) {
                    resumeAfterSeekRef.current = false;
                    selectedSeekRef.current = null;
                    previewEndRef.current = null;
                    const correctToBoundary = Math.abs(time - previewEnd) > 0.01;
                    previewStopRef.current = correctToBoundary ? previewEnd : null;
                    event.currentTarget.pause();
                    if (correctToBoundary) {
                      event.currentTarget.currentTime = previewEnd;
                    }
                    setPlaybackTime(previewEnd);
                    return;
                  }
                  trackPlayback(time);
                  if (previewEndRef.current !== null) return;
                  if (cutPreviewEnabled) {
                    const target = nextFinalCutTime(finalIntervals, time);
                    if (target === null) {
                      resumeAfterSeekRef.current = false;
                      event.currentTarget.pause();
                      return;
                    }
                    if (Math.abs(target - time) > 0.01) {
                      resumeAfterSeekRef.current = true;
                      requestPlayingSeek(event.currentTarget, target);
                      trackPlayback(target);
                      return;
                    }
                  }
                }}
                onPlay={() => {
                  resumeAfterSeekRef.current = true;
                  setIsPlaying(true);
                }}
                onPause={(event) => {
                  if (!event.currentTarget.seeking) resumeAfterSeekRef.current = false;
                  setIsPlaying(false);
                }}
                onEnded={() => {
                  resumeAfterSeekRef.current = false;
                  setIsPlaying(false);
                }}
                onSeeked={(event) => {
                  if (previewStopRef.current !== null) {
                    const previewStop = previewStopRef.current;
                    previewStopRef.current = null;
                    resumeAfterSeekRef.current = false;
                    event.currentTarget.pause();
                    setPlaybackTime(previewStop);
                    return;
                  }
                  const selectedSeek = selectedSeekRef.current;
                  if (
                    selectedSeek &&
                    Math.abs(event.currentTarget.currentTime - selectedSeek.target) <= 0.25
                  ) {
                    selectedSeek.arrived = true;
                  }
                  if (resumeAfterSeekRef.current) {
                    void event.currentTarget.play().catch(() => undefined);
                  }
                }}
              />
            ) : (
              <div className={styles.noVideo}>Video unavailable</div>
            )}
            <div className={styles.timecode}>
              {preciseTime(playbackTime)} <span>/ game ends {preciseTime(analysisEnd)}</span>
            </div>
          </div>

          <div className={styles.transport}>
            <button type="button" onClick={() => seekTo(playbackTime - 1)}>−1s</button>
            <button type="button" onClick={() => seekTo(playbackTime - 0.1)}>−0.1s</button>
            <button
              type="button"
              className={styles.playButton}
              onClick={togglePlayback}
              disabled={!initialAnalysis.videoUrl}
            >
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button type="button" onClick={() => seekTo(playbackTime + 0.1)}>+0.1s</button>
            <button type="button" onClick={() => seekTo(playbackTime + 1)}>+1s</button>
          </div>
          <label className={styles.playbackRateControl}>
            <span>Playback speed</span>
            <select
              aria-label="Video playback speed"
              value={draft.playbackRate}
              onChange={(event) => setPlaybackRate(
                Number(event.currentTarget.value) as CutDraft["playbackRate"],
              )}
            >
              {PLAYBACK_RATES.map((rate) => (
                <option key={rate} value={rate}>{rate}x</option>
              ))}
            </select>
          </label>

          <section className={styles.overviewSection}>
            <div className={styles.sectionHeading}>
              <div>
                <span>GAME WINDOW</span>
                <strong>Tap or slide to seek · select a range to refine</strong>
              </div>
              <small>
                {preciseTime(analysisStart)}–{preciseTime(analysisEnd)} · {draft.cuts.length} ranges · light gray = joined gap
              </small>
            </div>
            <div className={styles.confidenceReview}>
              <label htmlFor="confidence-review-threshold">
                <span>Review disagreements and confidence below</span>
                <span className={styles.confidenceInput}>
                  <input
                    id="confidence-review-threshold"
                    aria-label="Highlight model ranges below confidence percent"
                    type="number"
                    inputMode="numeric"
                    min="0"
                    max="100"
                    step="1"
                    value={Math.round(draft.confidenceReviewThreshold * 100)}
                    onChange={(event) => setConfidenceReviewThreshold(
                      event.currentTarget.valueAsNumber,
                    )}
                  />
                  <span>%</span>
                </span>
              </label>
              <div className={styles.reviewStatus}>
                <p>
                  {disagreementCount} disagreements · {reviewedLowConfidenceCount}/{lowConfidenceCuts.length} reviewed · {unreviewedLowConfidenceCuts.length} remaining
                </p>
                {draft.selectedSuppressionPolicy !== "none" && suppressionSuggestions.length > 0 && (
                  <p data-suppression="true">
                    {suppressionSuggestions.length} suppression suggestions · {appliedSuppressionCount} applied · {Math.max(0, noSuppressionSeconds - keptSeconds).toFixed(1)}s removed
                  </p>
                )}
              </div>
              <div className={styles.reviewActions}>
                <button
                  type="button"
                  onClick={reviewNextLowConfidenceCut}
                  disabled={unreviewedLowConfidenceCuts.length === 0}
                >
                  {unreviewedLowConfidenceCuts.length === 0
                    ? "Review complete"
                    : selectedReviewCandidate && !selectedIsReviewed
                      ? "Mark reviewed & next"
                      : "Review next"}
                </button>
                {draft.selectedSuppressionPolicy !== "none" && suppressionSuggestions.length > 0 && (
                  <button
                    type="button"
                    className={styles.nextSuppression}
                    onClick={reviewNextSuppression}
                  >
                    Next suppression
                  </button>
                )}
              </div>
            </div>
            <div className={styles.timelineLegend} aria-label="Timeline legend">
              <span><i data-kind="ordinary" /> Kept rally</span>
              <span><i data-kind="suppressed" /> Suppressed</span>
              <span><i data-kind="suggestion-kept" /> Suggestion kept</span>
              <span><i data-kind="ignored" /> Ignored source</span>
              <span><i data-kind="joined" /> Joined gap</span>
            </div>
            <div
              className={styles.overviewRail}
              onPointerDown={(event) =>
                beginTimelineSeek(event, analysisStart, analysisEnd)
              }
              onPointerMove={moveTimelineSeek}
              onPointerUp={endTimelineSeek}
              onPointerCancel={cancelTimelineSeek}
              onLostPointerCapture={cancelTimelineSeek}
              onClickCapture={(event) => {
                if (Date.now() > suppressTimelineClickUntilRef.current) return;
                event.preventDefault();
                event.stopPropagation();
                suppressTimelineClickUntilRef.current = 0;
              }}
              role="group"
              aria-label="Marked game overview"
            >
              {overviewIgnoredIntervals.map((interval) => (
                <span
                  key={interval.id}
                  className={styles.overviewIgnored}
                  style={{
                    left: `${timelinePercent(interval.start - analysisStart, analysisDuration)}%`,
                    width: `${timelinePercent(interval.end - interval.start, analysisDuration)}%`,
                  }}
                />
              ))}
              {overviewCuts.map((cut) => (
                <button
                  type="button"
                  key={cut.id}
                  className={styles.overviewCut}
                  data-overview-cut-id={cut.id}
                  data-selected={cut.id === selected?.id || undefined}
                  data-included={cut.included || undefined}
                  data-ignored={cut.included && !effectiveKeptIds.has(cut.id) || undefined}
                  data-low-confidence={
                    cut.origin === "cached-label" &&
                    (isModelDisagreement(cut) ||
                      cut.confidence < draft.confidenceReviewThreshold) || undefined
                  }
                  data-disagreement={isModelDisagreement(cut) || undefined}
                  data-reviewed={reviewedCutIds.has(cut.id) || undefined}
                  data-origin={cut.origin}
                  style={{
                    left: `${timelinePercent(cut.keepStart - analysisStart, analysisDuration)}%`,
                    width: `${timelinePercent(cut.keepEnd - cut.keepStart, analysisDuration)}%`,
                  }}
                  onClick={() => selectCut(cut)}
                  aria-label={`${!cut.included ? "Removed" : effectiveKeptIds.has(cut.id) ? "Keep" : "Ignored"} ${cut.id}, ${preciseTime(cut.keepStart)} to ${preciseTime(cut.keepEnd)}, ${modelAgreementLabel(cut)}, ${Math.round(cut.confidence * 100)}% review confidence${reviewedCutIds.has(cut.id) ? ", reviewed" : ""}`}
                >
                  <span
                    className={styles.overviewPadding}
                    style={{
                      left: 0,
                      width: `${timelinePercent(cut.coreStart - cut.keepStart, cut.keepEnd - cut.keepStart)}%`,
                    }}
                  />
                  <span
                    className={styles.overviewCore}
                    style={{
                      left: `${timelinePercent(cut.coreStart - cut.keepStart, cut.keepEnd - cut.keepStart)}%`,
                      width: `${timelinePercent(cut.coreEnd - cut.coreStart, cut.keepEnd - cut.keepStart)}%`,
                    }}
                  />
                  <span
                    className={styles.overviewPadding}
                    style={{
                      left: `${timelinePercent(cut.coreEnd - cut.keepStart, cut.keepEnd - cut.keepStart)}%`,
                      width: `${timelinePercent(cut.keepEnd - cut.coreEnd, cut.keepEnd - cut.keepStart)}%`,
                    }}
                  />
                </button>
              ))}
              {suppressionSuggestions.map((suggestion) => {
                const state = suppressionSuggestionState(suggestion, draft);
                return (
                  <button
                    type="button"
                    key={suggestion.id}
                    className={styles.overviewSuppression}
                    data-state={state}
                    data-selected={suggestion.id === selectedSuppression?.id || undefined}
                    style={{
                      left: `${timelinePercent(suggestion.start - analysisStart, analysisDuration)}%`,
                      width: `${timelinePercent(suggestion.end - suggestion.start, analysisDuration)}%`,
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                      selectSuppression(suggestion);
                    }}
                    aria-label={`${state === "suppressed" ? "Suppressed" : state === "edited-kept" ? "Edited rally—kept" : "Suggestion kept"}, ${preciseTime(suggestion.start)} to ${preciseTime(suggestion.end)}, ${Math.round(suggestion.score * 100)}% score`}
                    title={`${state === "suppressed" ? "Suppressed" : state === "edited-kept" ? "Edited rally—kept" : "Suggestion kept"} · ${preciseTime(suggestion.start)}–${preciseTime(suggestion.end)}`}
                  />
                );
              })}
              {finalIntervals.flatMap((interval) =>
                (interval.joinedGaps ?? [])
                  .filter((gap) => activeMarkStart === null || gap.end >= activeMarkStart)
                  .map((gap) => (
                    <span
                      key={`joined-gap-${gap.start}-${gap.end}-${interval.cutIds.join("-")}`}
                      className={styles.overviewJoinedGap}
                      style={{
                        left: `${timelinePercent(gap.start - analysisStart, analysisDuration)}%`,
                        width: `${timelinePercent(gap.end - gap.start, analysisDuration)}%`,
                      }}
                      title={`Retained short gap · ${preciseTime(gap.start)} to ${preciseTime(gap.end)}`}
                    />
                  )),
              )}
              <span
                className={styles.playhead}
                style={{
                  left: `${timelinePercent(playbackTime - analysisStart, analysisDuration)}%`,
                }}
              />
            </div>
            <div className={styles.overviewTimes}>
              <span>{formatTime(analysisStart)}</span>
              <span>{formatTime((analysisStart + analysisEnd) / 2)}</span>
              <span>{formatTime(analysisEnd)}</span>
            </div>
          </section>
        </div>

      </section>

      <section className={styles.focusEditor} aria-label="Focused range editor">
        <div className={styles.focusHeader}>
          <div>
            <span>FOCUSED RANGE</span>
            <strong>
              {selectedSuppression
                ? `Suppression suggestion · ${preciseTime(selectedSuppression.start)}–${preciseTime(selectedSuppression.end)} · ${Math.round(selectedSuppression.score * 100)}% score`
                : selected
                  ? `${selected.id} · ${selected.origin === "manual" ? "Manual" : `${modelAgreementLabel(selected)} · ${Math.round(selected.confidence * 100)}% review confidence${selectedIsReviewed ? " · reviewed" : ""}`}`
                  : "No range selected"}
            </strong>
          </div>
          <div className={styles.focusHeaderControls}>
            <label className={styles.focusLock}>
              <input
                type="checkbox"
                checked={focusLocked}
                onChange={(event) => setFocusLocked(event.currentTarget.checked)}
              />
              <span>Lock focus</span>
            </label>
            {selectedSuppression ? (
              <div className={styles.rangeNavigation}>
                <button type="button" onClick={() => navigateSuppression(-1)}>
                  Previous
                </button>
                <span>{selectedSuppressionIndex + 1} / {suppressionSuggestions.length}</span>
                <button type="button" onClick={() => navigateSuppression(1)}>
                  Next
                </button>
              </div>
            ) : selected && (
              <div className={styles.rangeNavigation}>
                <button type="button" onClick={() => navigateCut(-1)} disabled={selectedIndex <= 0}>
                  Previous
                </button>
                <span>{selectedIndex + 1} / {sortedCuts.length}</span>
                <button
                  type="button"
                  onClick={() => navigateCut(1)}
                  disabled={selectedIndex >= sortedCuts.length - 1}
                >
                  Next
                </button>
              </div>
            )}
          </div>
        </div>

        {selected ? (
          <>
            <div className={styles.detailTimes}>
              <span>{preciseTime(focus.start)}</span>
              <span>{preciseTime((focus.start + focus.end) / 2)}</span>
              <span>{preciseTime(focus.end)}</span>
            </div>
            <div
              ref={detailRailRef}
              className={styles.detailRail}
              onPointerDown={(event) => {
                if ((event.target as HTMLElement).closest(`.${styles.boundaryHandle}`)) return;
                beginTimelineSeek(event, focus.start, focus.end);
              }}
              onPointerMove={moveTimelineSeek}
              onPointerUp={endTimelineSeek}
              onPointerCancel={cancelTimelineSeek}
              onLostPointerCapture={cancelTimelineSeek}
            >
              <span
                className={styles.keptRange}
                data-included={selected.included || undefined}
                data-ignored={selected.included && !effectiveKeptIds.has(selected.id) || undefined}
                data-low-confidence={
                  selected.origin === "cached-label" &&
                  (isModelDisagreement(selected) ||
                    selected.confidence < draft.confidenceReviewThreshold) || undefined
                }
                data-disagreement={isModelDisagreement(selected) || undefined}
                style={{
                  left: `${timelinePercent(selected.keepStart - focus.start, focus.end - focus.start)}%`,
                  width: `${timelinePercent(selected.keepEnd - selected.keepStart, focus.end - focus.start)}%`,
                }}
              />
              <span
                className={styles.coreRange}
                data-included={selected.included || undefined}
                data-ignored={selected.included && !effectiveKeptIds.has(selected.id) || undefined}
                data-low-confidence={
                  selected.origin === "cached-label" &&
                  (isModelDisagreement(selected) ||
                    selected.confidence < draft.confidenceReviewThreshold) || undefined
                }
                data-disagreement={isModelDisagreement(selected) || undefined}
                style={{
                  left: `${timelinePercent(selected.coreStart - focus.start, focus.end - focus.start)}%`,
                  width: `${timelinePercent(selected.coreEnd - selected.coreStart, focus.end - focus.start)}%`,
                }}
              >
                <small>INFERRED CORE</small>
              </span>
              {selectedSuppression && (
                <span
                  className={styles.detailSuppression}
                  data-state={suppressionSuggestionState(selectedSuppression, draft)}
                  style={{
                    left: `${timelinePercent(selectedSuppression.start - focus.start, focus.end - focus.start)}%`,
                    width: `${timelinePercent(selectedSuppression.end - selectedSuppression.start, focus.end - focus.start)}%`,
                  }}
                  aria-label={`Selected suppression suggestion, ${preciseTime(selectedSuppression.start)} to ${preciseTime(selectedSuppression.end)}`}
                >
                  <small>
                    {suppressionSuggestionState(selectedSuppression, draft) === "suppressed"
                      ? "SUPPRESSED"
                      : suppressionSuggestionState(selectedSuppression, draft) === "edited-kept"
                        ? "EDITED RALLY—KEPT"
                        : "SUGGESTION KEPT"}
                  </small>
                </span>
              )}
              {!selectedSuppression && <button
                type="button"
                className={`${styles.boundaryHandle} ${styles.startHandle}`}
                style={{
                  left: `${timelinePercent(selected.keepStart - focus.start, focus.end - focus.start)}%`,
                }}
                aria-label={`Adjust kept start at ${preciseTime(selected.keepStart)}`}
                onPointerDown={(event) => beginBoundaryDrag("start", event)}
                onPointerMove={moveBoundary}
                onPointerUp={endBoundaryDrag}
                onPointerCancel={endBoundaryDrag}
                onLostPointerCapture={endBoundaryDrag}
              >
                <i />
              </button>}
              {!selectedSuppression && <button
                type="button"
                className={`${styles.boundaryHandle} ${styles.endHandle}`}
                style={{
                  left: `${timelinePercent(selected.keepEnd - focus.start, focus.end - focus.start)}%`,
                }}
                aria-label={`Adjust kept end at ${preciseTime(selected.keepEnd)}`}
                onPointerDown={(event) => beginBoundaryDrag("end", event)}
                onPointerMove={moveBoundary}
                onPointerUp={endBoundaryDrag}
                onPointerCancel={endBoundaryDrag}
                onLostPointerCapture={endBoundaryDrag}
              >
                <i />
              </button>}
              <span
                className={styles.detailPlayhead}
                style={{
                  left: `${timelinePercent(playbackTime - focus.start, focus.end - focus.start)}%`,
                }}
              />
            </div>

            {!selectedSuppression && <div className={styles.boundaryControls}>
              <fieldset>
                <legend>Kept start</legend>
                <output>{preciseTime(selected.keepStart)}</output>
                <div>
                  <button type="button" onClick={() => nudgeBoundary("start", -1)}>−1s</button>
                  <button type="button" onClick={() => nudgeBoundary("start", -0.1)}>−0.1s</button>
                  <button type="button" onClick={() => nudgeBoundary("start", 0.1)}>+0.1s</button>
                  <button type="button" onClick={() => nudgeBoundary("start", 1)}>+1s</button>
                </div>
              </fieldset>
              <fieldset>
                <legend>Kept end</legend>
                <output>{preciseTime(selected.keepEnd)}</output>
                <div>
                  <button type="button" onClick={() => nudgeBoundary("end", -1)}>−1s</button>
                  <button type="button" onClick={() => nudgeBoundary("end", -0.1)}>−0.1s</button>
                  <button type="button" onClick={() => nudgeBoundary("end", 0.1)}>+0.1s</button>
                  <button type="button" onClick={() => nudgeBoundary("end", 1)}>+1s</button>
                </div>
              </fieldset>
            </div>}

            <div className={styles.focusActions}>
              {selectedSuppression ? (
                <>
                  <p className={styles.suppressionStateText}>
                    {suppressionSuggestionState(selectedSuppression, draft) === "suppressed"
                      ? "This suggestion is currently removed from the derived export."
                      : suppressionSuggestionState(selectedSuppression, draft) === "edited-kept"
                        ? "Kept because an overlapping inferred rally was already edited. Choose Suppress to override that protection."
                        : "You explicitly kept this suggestion in the export."}
                  </p>
                  <button
                    type="button"
                    className={styles.suppressAction}
                    data-active={suppressionSuggestionState(selectedSuppression, draft) === "suppressed" || undefined}
                    onClick={() => setSuppressionDecision(selectedSuppression, "suppress")}
                  >
                    Suppress
                  </button>
                  <button
                    type="button"
                    data-active={suppressionSuggestionState(selectedSuppression, draft) !== "suppressed" || undefined}
                    onClick={() => setSuppressionDecision(selectedSuppression, "keep")}
                  >
                    Keep
                  </button>
                </>
              ) : (
                <>
                  <button
                    type="button"
                    className={styles.primaryAction}
                    data-included={selected.included || undefined}
                    onClick={toggleSelected}
                  >
                    {selected.included ? "✓ Keep this range" : "+ Restore this range"}
                  </button>
                  <button type="button" onClick={previewSelected}>Preview cut</button>
                  <button type="button" onClick={resetSelectedPadding}>Reset padding</button>
                  {selectedReviewCandidate && (
                    <button
                  type="button"
                  className={styles.reviewAction}
                  data-reviewed={selectedIsReviewed || undefined}
                  onClick={toggleSelectedReviewed}
                >
                  {selectedIsReviewed ? "✓ Reviewed" : "Mark reviewed"}
                    </button>
                  )}
                  {selected.origin === "manual" && (
                    <button
                  type="button"
                  className={styles.dangerButton}
                  onClick={() => deleteManualCut(selected.id)}
                >
                  Delete manual cut
                    </button>
                  )}
                </>
              )}
            </div>
          </>
        ) : (
          <p className={styles.emptyMessage}>Add a missed cut at the current playhead to begin.</p>
        )}
      </section>

      <section className={styles.markingTools}>
        <div className={styles.markingCard}>
          <div>
            <span>ADD A MISSED CUT</span>
            <strong>{manualStart === null ? "Find the first frame" : `Started ${preciseTime(manualStart)}`}</strong>
            <p>Seek the video, mark the start, then seek and mark the end.</p>
          </div>
          <button type="button" onClick={markManualBoundary}>
            {manualStart === null ? `Mark start · ${preciseTime(playbackTime)}` : `Mark end · ${preciseTime(playbackTime)}`}
          </button>
          {manualStart !== null && (
            <button
              type="button"
              className={styles.quietButton}
              onClick={() => {
                updateDraft((current) => ({ ...current, pendingManualStart: null }));
                setEditorMessage(null);
              }}
            >
              Cancel
            </button>
          )}
        </div>

        <div className={styles.markingCard}>
          <div>
            <span>IGNORE SOURCE SECTION</span>
            <strong>{ignoreStart === null ? "Exclude unusable footage" : `Started ${preciseTime(ignoreStart)}`}</strong>
            <p>Ignored time is removed from the final edit without becoming a negative label.</p>
          </div>
          <select
            aria-label="Ignored section reason"
            value={ignoreReason}
            onChange={(event) => updateDraft((current) => ({
              ...current,
              ignoreReason: event.target.value,
            }))}
          >
            <option value="non-game-content">Non-game content</option>
            <option value="camera-gap">Camera gap</option>
            <option value="partial-rally">Partial rally</option>
            <option value="boundary-ambiguous">Boundary ambiguous</option>
          </select>
          <button type="button" onClick={markIgnoredBoundary}>
            {ignoreStart === null ? `Mark start · ${preciseTime(playbackTime)}` : `Mark end · ${preciseTime(playbackTime)}`}
          </button>
          {ignoreStart !== null && (
            <button
              type="button"
              className={styles.quietButton}
              onClick={() => {
                updateDraft((current) => ({ ...current, pendingIgnoreStart: null }));
                setEditorMessage(null);
              }}
            >
              Cancel
            </button>
          )}
        </div>
      </section>

      {editorMessage && <p className={styles.editorMessage}>{editorMessage}</p>}

      {manualCuts.length > 0 && (
        <section className={styles.addedCutList}>
          <div className={styles.sectionHeading}>
            <div>
              <span>ADDED MISSED CUTS</span>
              <strong>Manual ranges saved on this device</strong>
            </div>
          </div>
          <div>
            {manualCuts.map((cut) => (
              <article key={cut.id}>
                <button type="button" onClick={() => selectCut(cut)}>
                  {preciseTime(cut.keepStart)}–{preciseTime(cut.keepEnd)}
                </button>
                <span>{cut.id} · {(cut.keepEnd - cut.keepStart).toFixed(1)}s</span>
                <button type="button" onClick={() => deleteManualCut(cut.id)}>
                  Remove
                </button>
              </article>
            ))}
          </div>
        </section>
      )}

      {draft.ignoredIntervals.length > 0 && (
        <section className={styles.ignoredList}>
          <div className={styles.sectionHeading}>
            <div><span>IGNORED SOURCE</span><strong>Excluded from the derived edit list</strong></div>
          </div>
          <div>
            {[...draft.ignoredIntervals]
              .sort((left, right) => left.start - right.start)
              .map((interval) => (
                <article key={interval.id}>
                  <button type="button" onClick={() => seekTo(interval.start)}>
                    {preciseTime(interval.start)}–{preciseTime(interval.end)}
                  </button>
                  <span>{interval.reason.replaceAll("-", " ")}</span>
                  <button type="button" onClick={() => removeIgnoredInterval(interval.id)}>
                    Remove
                  </button>
                </article>
              ))}
          </div>
        </section>
      )}

      <section className={styles.cutList}>
        <div className={styles.sectionHeading}>
          <div><span>ALL CUTS</span><strong>Model predictions and manual additions</strong></div>
          <small>{keptCount} kept · {fullyIgnoredCount} fully ignored</small>
        </div>
        <div className={styles.cutCards}>
          {sortedCuts.map((cut, index) => (
            <article
              key={cut.id}
              data-selected={cut.id === selected?.id || undefined}
              data-included={cut.included || undefined}
              data-ignored={cut.included && !effectiveKeptIds.has(cut.id) || undefined}
              data-low-confidence={
                cut.origin === "cached-label" &&
                (isModelDisagreement(cut) ||
                  cut.confidence < draft.confidenceReviewThreshold) || undefined
              }
              data-disagreement={isModelDisagreement(cut) || undefined}
              data-reviewed={reviewedCutIds.has(cut.id) || undefined}
              data-suppression={suppressionSuggestions.some(
                (suggestion) => suggestion.start < cut.coreEnd && suggestion.end > cut.coreStart,
              ) || undefined}
            >
              <button type="button" className={styles.cutSelect} onClick={() => selectCut(cut)}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{cut.id}</strong>
                <small>{preciseTime(cut.keepStart)}–{preciseTime(cut.keepEnd)}</small>
                <em>{cut.origin === "manual" ? "MANUAL" : suppressionSuggestions.some(
                  (suggestion) => suggestion.start < cut.coreEnd && suggestion.end > cut.coreStart,
                ) ? "SUPPRESS" : reviewedCutIds.has(cut.id) ? "REVIEWED" : isModelDisagreement(cut) ? "CHECK" : `${Math.round(cut.confidence * 100)}%`}</em>
              </button>
              <button
                type="button"
                className={styles.cutToggle}
                onClick={() => updateCut(cut.id, (current) => ({
                  ...current,
                  included: !current.included,
                }))}
              >
                {!cut.included ? "Removed" : effectiveKeptIds.has(cut.id) ? "Keep" : "Ignored"}
              </button>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
