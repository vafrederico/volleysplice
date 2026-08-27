import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";

import { GuidedTour } from "@/components/GuidedTour";
import { ScoreOverlay } from "@/components/ScoreOverlay";
import { ScoreTrackingPanel } from "@/components/ScoreTrackingPanel";
import { SiteFooter } from "@/components/SiteFooter";
import {
  activeSuppressionSuggestions,
  applyPaddingToCachedCuts,
  buildFinalCutIntervals,
  createCutDraft,
  cutDraftStorageKey,
  cutDraftStorageKeys,
  DEFAULT_SUPPRESSION_SCOPE,
  materializeFinalCutIntervals,
  MIN_CUT_SECONDS,
  nextFinalCutTime,
  parseCutDraft,
  playbackFocusCut,
  PLAYBACK_RATES,
  SUPPRESSION_SCOPE_IDS,
  suppressionSuggestionScope,
  totalFinalCutSeconds,
  suppressionSuggestionState,
  setCutCoreEnd,
  setCutCoreStart,
  splitCutAt,
  type CutDraft,
  type CutDraftSeed,
  type EditableCut,
  type SuppressionScope,
} from "@/lib/cut-draft";
import { formatTime, timelinePercent } from "@/lib/edit-list";
import {
  createInferenceProgressSteps,
  finishInferenceStep,
  type InferenceProgressStep,
  type InferenceProgressStepId,
  updateSpecialistInferenceStep,
} from "@/lib/inference-progress";
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
import { openLocalMedia } from "@/lib/on-device/media";
import { requestPlayingSeek } from "@/lib/on-device/player";
import { prepareScoreOverlay } from "@/lib/score-overlay";
import {
  addServeMarker,
  addSideSwitchMarker,
  isScoreTimestampIgnored,
  scoreBoundaryTimestamp,
  scoreTrackingOutsideExcludedRallies,
  scoreTrackingOutsideIgnoredIntervals,
  type ScoreTracking,
} from "@/lib/score-tracking";
import {
  nextSuppressionAfterTime,
  nextSuppressionSuggestion,
  type SuppressionPolicyId,
  type SuppressionSuggestion,
} from "@/lib/on-device/suppression-policy";
import { prepareServiceWorkerStreamDownload } from "@/lib/on-device/stream-download";
import type { WakeLockState } from "@/lib/on-device/wake-lock";
import type {
  OnDeviceSideSwitchOutput,
  OnDeviceServingSideOutput,
} from "@/lib/on-device/types";
import type { ProductAnalysis } from "@/lib/product-analysis";

import styles from "./CutEditor.module.css";

type CutEditorProps = {
  header: ReactNode;
  initialAnalysis: ProductAnalysis;
  initialScoreTrackingEnabled: boolean;
  sideSwitchEnabled: boolean;
  importedInitialDraft?: CutDraft;
  sourceFile: File | null;
  sourceError: string | null;
  onAttachSource: (file: File | null) => void;
  onRequestSuppression: () => void;
  onServingSideAnalysis: (output: OnDeviceServingSideOutput) => void;
  onSideSwitchAnalysis: (output: OnDeviceSideSwitchOutput) => void;
};

type ExportState = "idle" | "exporting" | "done" | "error";
type BoundarySide = "start" | "end";
type BoundaryEdge = "output" | "core";

type BoundaryDrag = {
  pointerId: number;
  edge: BoundaryEdge;
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
  tapMarker:
    | { kind: "serve"; markerId: string; timestamp: number }
    | { kind: "switch"; timestamp: number }
    | null;
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


function isModelDisagreement(cut: EditableCut): boolean {
  return cut.agreement === "all-labels-v2-only" ||
    cut.agreement === "previous-production-only";
}

function modelAgreementLabel(cut: EditableCut): string {
  if (cut.agreement === "both-models") return "Found automatically";
  if (cut.agreement === "all-labels-v2-only" || cut.agreement === "previous-production-only")
    return "Needs a quick check";
  return "Suggested clip";
}

function cleanupLabel(policy: SuppressionPolicyId): string {
  switch (policy) {
    case "none": return "Off";
    case "conservative": return "Light";
    case "balanced": return "Standard";
    case "aggressive": return "Strong";
  }
}

export function CutEditor({
  header,
  initialAnalysis,
  initialScoreTrackingEnabled,
  sideSwitchEnabled,
  importedInitialDraft,
  sourceFile,
  sourceError,
  onAttachSource,
  onRequestSuppression,
  onServingSideAnalysis,
  onSideSwitchAnalysis,
}: CutEditorProps) {
  const analysisStart = initialAnalysis.analysisWindow.start;
  const analysisEnd = initialAnalysis.analysisWindow.end;
  const videoRef = useRef<HTMLVideoElement>(null);
  const detailRailRef = useRef<HTMLDivElement>(null);
  const boundaryDragRef = useRef<BoundaryDrag | null>(null);
  const timelineDragRef = useRef<TimelineDrag | null>(null);
  const scoreInferenceStartedRef = useRef(false);
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
      scoreTrackingEnabled: initialScoreTrackingEnabled,
      rallies: initialAnalysis.rallies,
      ignoredIntervals: initialAnalysis.ignoredIntervals,
      suppressionContractVersion: initialAnalysis.suppression?.policyContractVersion,
    }),
    [initialAnalysis, initialScoreTrackingEnabled, analysisStart, analysisEnd],
  );
  const initialDraft = useMemo(() => createCutDraft(seed), [seed]);
  const [draft, setDraft] = useState<CutDraft>(initialDraft);
  const [storageReady, setStorageReady] = useState(false);
  const [storageMessage, setStorageMessage] = useState("Opening your saved edits…");
  const [selectedId, setSelectedId] = useState(initialDraft.cuts[0]?.id ?? "");
  const [selectedSuppressionId, setSelectedSuppressionId] = useState("");
  const [selectedServeMarkerId, setSelectedServeMarkerId] = useState("");
  const [scoreInferenceStatus, setScoreInferenceStatus] = useState<
    "idle" | "running" | "done" | "error"
  >("idle");
  const [scoreInferenceMessage, setScoreInferenceMessage] = useState<string | null>(null);
  const [scoreInferenceSteps, setScoreInferenceSteps] = useState<InferenceProgressStep[]>([]);
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
  const materialized = useMemo(
    () => materializeFinalCutIntervals(draft, initialAnalysis.suppression),
    [draft, initialAnalysis.suppression],
  );
  const finalIntervals = materialized.intervals;
  const effectiveKeptIds = useMemo(
    () => new Set(finalIntervals.flatMap((interval) => interval.cutIds)),
    [finalIntervals],
  );
  const excludedRallyIds = useMemo(
    () => new Set(
      draft.cuts
        .filter((cut) => !effectiveKeptIds.has(cut.id))
        .map((cut) => cut.id),
    ),
    [draft.cuts, effectiveKeptIds],
  );
  const activeScoreTracking = useMemo(
    () => scoreTrackingOutsideExcludedRallies(
      scoreTrackingOutsideIgnoredIntervals(
        draft.scoreTracking,
        draft.ignoredIntervals,
      ),
      excludedRallyIds,
    ),
    [draft.ignoredIntervals, draft.scoreTracking, excludedRallyIds],
  );
  const activeScoreRallyRanges = useMemo(
    () => draft.cuts.filter((cut) => effectiveKeptIds.has(cut.id)),
    [draft.cuts, effectiveKeptIds],
  );
  const scoreBoundaryTime = useMemo(
    () => scoreBoundaryTimestamp(
      playbackTime,
      activeScoreRallyRanges,
      activeScoreTracking,
      finalIntervals,
    ),
    [activeScoreRallyRanges, activeScoreTracking, finalIntervals, playbackTime],
  );
  const preparedScoreOverlay = useMemo(
    () => prepareScoreOverlay({
      scoreTracking: draft.scoreTracking,
      renderPointTimeline: draft.renderScoreTimeline,
      excludedRallyIds: [...excludedRallyIds],
      ignoredIntervals: draft.ignoredIntervals,
      rallyRanges: activeScoreRallyRanges,
      mergedRanges: finalIntervals,
    }),
    [
      activeScoreRallyRanges,
      draft.ignoredIntervals,
      draft.renderScoreTimeline,
      draft.scoreTracking,
      excludedRallyIds,
      finalIntervals,
    ],
  );

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
      const imported = importedInitialDraft
        ? parseCutDraft(JSON.stringify(importedInitialDraft), seed)
        : null;
      const next = restored ?? imported ?? initialDraft;
      setDraft(next);
      setSelectedId((current) =>
        next.cuts.some((cut) => cut.id === current) ? current : next.cuts[0]?.id ?? "",
      );
      setStorageMessage(
        restored
          ? "Your saved edits are ready"
          : imported
            ? "Your saved project is ready"
            : "Changes save automatically",
      );
      setStorageReady(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [importedInitialDraft, initialDraft, seed]);

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
    const resumableMissingStage = Boolean(
      sourceFile &&
      ((!initialAnalysis.servingSide && initialAnalysis.productionServeOutputs) ||
        (sideSwitchEnabled && !initialAnalysis.sideSwitch && initialAnalysis.productionComponents &&
          initialAnalysis.productionStateOutputs)),
    );
    if (
      !storageReady ||
      !draft.scoreTracking.enabled ||
      scoreInferenceStatus === "running" ||
      (scoreInferenceStartedRef.current && !resumableMissingStage)
    ) return;
    scoreInferenceStartedRef.current = true;
    if (initialAnalysis.servingSide) {
      applyServingSideOutput(initialAnalysis.servingSide);
    }
    if (initialAnalysis.sideSwitch) {
      applySideSwitchOutput(initialAnalysis.sideSwitch);
    }
    const servingReady = Boolean(initialAnalysis.servingSide);
    const switchingReady = !sideSwitchEnabled || Boolean(initialAnalysis.sideSwitch);
    if (servingReady && switchingReady) {
      setScoreInferenceStatus("done");
      const review = initialAnalysis.servingSide!.candidates.filter(
        (candidate) => candidate.verdict === "review",
      ).length;
      setScoreInferenceMessage(
        `${review} serve ${review === 1 ? "marker needs" : "markers need"} review · ${initialAnalysis.sideSwitch?.candidates.length ?? 0} side switches found.`,
      );
      return;
    }
    const canRunServing = !servingReady && Boolean(initialAnalysis.productionServeOutputs);
    const canRunSwitching = sideSwitchEnabled && !switchingReady && Boolean(
      initialAnalysis.productionComponents && initialAnalysis.productionStateOutputs,
    );
    if (sourceFile && (canRunServing || canRunSwitching)) {
      void runScoreInference();
      return;
    }
    setScoreInferenceStatus("idle");
    setScoreInferenceMessage(
      sourceFile
        ? "This older project cannot find score markers automatically. You can still add them yourself."
        : "Choose the original video once so VolleyCut can find and save the score markers.",
    );
  }, [draft.scoreTracking.enabled, draft.scoreTracking.serveMarkers, initialAnalysis.productionComponents, initialAnalysis.productionServeOutputs, initialAnalysis.productionStateOutputs, initialAnalysis.servingSide, initialAnalysis.sideSwitch, scoreInferenceStatus, sideSwitchEnabled, sourceFile, storageReady]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = draft.playbackRate;
  }, [draft.playbackRate]);

  useEffect(() => {
    const activeServeMarkers = activeScoreTracking.serveMarkers;
    if (!draft.scoreTracking.enabled || activeServeMarkers.length === 0) {
      setSelectedServeMarkerId("");
      return;
    }
    setSelectedServeMarkerId((current) =>
      activeServeMarkers.some((marker) => marker.id === current)
        ? current
        : [...activeServeMarkers].sort(
            (left, right) => left.timestamp - right.timestamp || left.id.localeCompare(right.id),
          )[0].id,
    );
  }, [
    activeScoreTracking,
    draft.scoreTracking.enabled,
  ]);

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
  const keptSeconds = totalFinalCutSeconds(finalIntervals);
  const noSuppressionSeconds = useMemo(
    () => totalFinalCutSeconds(buildFinalCutIntervals(
      { ...draft, selectedSuppressionPolicy: "none" },
      initialAnalysis.suppression,
    )),
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
  const selectedSuppressionScope = selectedSuppression
    ? suppressionSuggestionScope(selectedSuppression, draft)
    : DEFAULT_SUPPRESSION_SCOPE;
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
  const reviewedCutIds = useMemo(() => new Set(draft.reviewedCutIds), [draft.reviewedCutIds]);
  const unreviewedLowConfidenceCuts = lowConfidenceCuts.filter(
    (cut) => !reviewedCutIds.has(cut.id),
  );
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
  const browserExportSupported = encodingSupported && exportStorageReady;
  const localExportSupported = Boolean(sourceFile) && browserExportSupported;
  const exportUnavailableMessage = !sourceFile
    ? "Choose the original video above before creating your final video."
    : !browserExportSupported
      ? "This browser cannot save the finished video. Try the latest version of Chrome or Edge."
      : null;

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

  function updateScoreTracking(scoreTracking: ScoreTracking) {
    updateDraft((current) => ({ ...current, scoreTracking }));
  }

  function applyServingSideOutput(output: OnDeviceServingSideOutput) {
    updateDraft((current) => {
      const existingByRally = new Map(
        current.scoreTracking.serveMarkers
          .filter((marker) => marker.rallyId)
          .map((marker) => [marker.rallyId!, marker]),
      );
      let scoreTracking: ScoreTracking = {
        ...current.scoreTracking,
        serveMarkers: current.scoreTracking.serveMarkers.filter(
          (marker) => marker.origin === "manual",
        ),
      };
      for (const candidate of output.candidates) {
        const existing = existingByRally.get(candidate.id);
        const wasCorrected = Boolean(
          existing?.modelSide && existing.side !== existing.modelSide,
        );
        if (candidate.verdict === "not-serve" && !wasCorrected) continue;
        const modelSide = candidate.verdict === "review" || candidate.verdict === "not-serve"
          ? "review"
          : candidate.side;
        scoreTracking = addServeMarker(
          scoreTracking,
          candidate.anchor,
          wasCorrected ? existing!.side : modelSide,
          {
            id: `serve-${candidate.id}`,
            origin: "model",
            modelSide,
            rallyId: candidate.id,
          },
        );
        if (existing?.ignorePreviousPoint) {
          scoreTracking = {
            ...scoreTracking,
            serveMarkers: scoreTracking.serveMarkers.map((marker) =>
              marker.rallyId === candidate.id
                ? { ...marker, ignorePreviousPoint: true }
                : marker,
            ),
          };
        }
      }
      return { ...current, scoreTracking };
    });
  }

  function applySideSwitchOutput(output: OnDeviceSideSwitchOutput) {
    updateDraft((current) => {
      let scoreTracking: ScoreTracking = {
        ...current.scoreTracking,
        sideSwitchMarkers: current.scoreTracking.sideSwitchMarkers.filter(
          (marker) => marker.origin === "manual",
        ),
      };
      for (const candidate of output.candidates) {
        scoreTracking = addSideSwitchMarker(
          scoreTracking,
          candidate.timestamp,
          {
            id: `switch-${candidate.id}`,
            origin: "model",
            modelConfidence: candidate.probability,
            modelEventId: candidate.id,
            rallyIds: candidate.sourceRangeIds,
          },
        );
      }
      return { ...current, scoreTracking };
    });
  }

  async function runScoreInference() {
    if (!sourceFile || scoreInferenceStatus === "running") return;
    scoreInferenceStartedRef.current = true;
    setScoreInferenceStatus("running");
    setScoreInferenceMessage("Getting ready to find score markers…");
    const missingSteps: InferenceProgressStepId[] = [
      ...(!initialAnalysis.servingSide && initialAnalysis.productionServeOutputs
        ? ["serving-side" as const]
        : []),
      ...(sideSwitchEnabled && !initialAnalysis.sideSwitch && initialAnalysis.productionComponents &&
        initialAnalysis.productionStateOutputs
        ? ["side-switch" as const]
        : []),
    ];
    setScoreInferenceSteps(createInferenceProgressSteps(missingSteps));
    let media: Awaited<ReturnType<typeof openLocalMedia>> | null = null;
    try {
      media = await openLocalMedia(sourceFile);
      let servingSide = initialAnalysis.servingSide;
      let sideSwitch = initialAnalysis.sideSwitch;
      const needsServing = Boolean(
        !servingSide && initialAnalysis.productionServeOutputs,
      );
      const needsSwitch = Boolean(
        sideSwitchEnabled &&
          !sideSwitch &&
          initialAnalysis.productionComponents &&
          initialAnalysis.productionStateOutputs,
      );
      if (needsServing || needsSwitch) {
        const { inferScoreSpecialists } = await import(
          "@/lib/on-device/score-specialists"
        );
        const inferred = await inferScoreSpecialists(
          media,
          initialAnalysis.roi,
          {
            servingSide: needsServing
              ? {
                  analysis: initialAnalysis,
                  onProgress: (progress) => {
                    setScoreInferenceMessage(progress.detail);
                    setScoreInferenceSteps((current) =>
                      updateSpecialistInferenceStep(
                        current,
                        "serving-side",
                        progress,
                        performance.now(),
                      ),
                    );
                  },
                }
              : undefined,
            sideSwitch: needsSwitch
              ? {
                  analysis: {
                    intervals: initialAnalysis.rallies,
                    times: initialAnalysis.inferenceTimes,
                    deadStateProbabilities: initialAnalysis.probabilities.deadState,
                    productionComponents: initialAnalysis.productionComponents!,
                    productionStateOutputs: initialAnalysis.productionStateOutputs!,
                  },
                  onProgress: (progress) => {
                    setScoreInferenceMessage(progress.detail);
                    setScoreInferenceSteps((current) =>
                      updateSpecialistInferenceStep(
                        current,
                        "side-switch",
                        progress,
                        performance.now(),
                      ),
                    );
                  },
                }
              : undefined,
          },
        );
        if (inferred.servingSide) {
          servingSide = inferred.servingSide;
          onServingSideAnalysis(inferred.servingSide);
          applyServingSideOutput(inferred.servingSide);
          setScoreInferenceSteps((current) =>
            finishInferenceStep(
              current,
              "serving-side",
              `Serve markers ready · ${inferred.servingSide!.candidates.length} rallies checked`,
              performance.now(),
            ),
          );
        }
        if (inferred.sideSwitch) {
          sideSwitch = inferred.sideSwitch;
          onSideSwitchAnalysis(inferred.sideSwitch);
          applySideSwitchOutput(inferred.sideSwitch);
          setScoreInferenceSteps((current) =>
            finishInferenceStep(
              current,
              "side-switch",
              `Side switches ready · ${inferred.sideSwitch!.candidates.length} found`,
              performance.now(),
            ),
          );
        }
      }
      if (!servingSide && !sideSwitch) {
        throw new Error("This saved project cannot rebuild its score markers.");
      }
      setScoreInferenceStatus("done");
      const visible = servingSide?.candidates.filter(
        (candidate) => candidate.verdict !== "not-serve",
      ) ?? [];
      const review = visible.filter((candidate) => candidate.verdict === "review").length;
      setScoreInferenceMessage(
        `${review} serve ${review === 1 ? "marker needs" : "markers need"} review · ${sideSwitch?.candidates.length ?? 0} side switches found.`,
      );
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      for (const failedStep of missingSteps) {
        setScoreInferenceSteps((current) =>
          finishInferenceStep(
            current,
            failedStep,
            message,
            performance.now(),
            "error",
          ),
        );
      }
      setScoreInferenceStatus("error");
      setScoreInferenceMessage(
        `VolleyCut could not finish finding score markers: ${message}. You can still check or add them yourself.`,
      );
    } finally {
      media?.input.dispose();
    }
  }

  function toggleScoreTracking(enabled: boolean) {
    updateDraft((current) => ({
      ...current,
      scoreTracking: { ...current.scoreTracking, enabled },
      renderScoreOverlay: enabled ? current.renderScoreOverlay : false,
      renderScoreTimeline: enabled ? current.renderScoreTimeline : false,
    }));
    if (enabled) scoreInferenceStartedRef.current = false;
  }

  function toggleScoreOverlay(renderScoreOverlay: boolean) {
    if (renderScoreOverlay) scoreInferenceStartedRef.current = false;
    updateDraft((current) => ({
      ...current,
      scoreTracking: renderScoreOverlay
        ? { ...current.scoreTracking, enabled: true }
        : current.scoreTracking,
      renderScoreOverlay,
      renderScoreTimeline: renderScoreOverlay
        ? current.renderScoreTimeline
        : false,
    }));
  }

  function toggleScoreTimeline(renderScoreTimeline: boolean) {
    updateDraft((current) => ({
      ...current,
      renderScoreTimeline: current.renderScoreOverlay && renderScoreTimeline,
    }));
  }

  function selectServeMarker(markerId: string, timestamp: number) {
    setSelectedServeMarkerId(markerId);
    seekTo(timestamp, false);
    setEditorMessage(`Selected serve marker at ${preciseTime(timestamp)}.`);
  }

  function selectSideSwitchMarker(markerId: string, timestamp: number) {
    setSelectedServeMarkerId(markerId);
    seekTo(timestamp, false);
    setEditorMessage(`Selected side switch at ${preciseTime(timestamp)}.`);
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
      `${preciseTime(suggestion.start)}–${preciseTime(suggestion.end)} · review this section. Nothing changes until you choose what to do.`,
    );
  }

  function setSuppressionPolicy(policy: SuppressionPolicyId) {
    updateDraft((current) => ({ ...current, selectedSuppressionPolicy: policy }));
    if (policy === "none") setSelectedSuppressionId("");
    setEditorMessage(
      policy === "none"
        ? "Automatic cleanup is off. Your previous choices are still saved."
        : `${cleanupLabel(policy)} cleanup selected. Suggested non-play moments will be left out unless you keep them.`,
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

  function setSuppressionScope(
    suggestion: SuppressionSuggestion,
    scope: SuppressionScope,
  ) {
    updateDraft((current) => ({
      ...current,
      suppressionScopeOverrides: scope === DEFAULT_SUPPRESSION_SCOPE
        ? Object.fromEntries(
            Object.entries(current.suppressionScopeOverrides).filter(
              ([id]) => id !== suggestion.logicalId,
            ),
          )
        : {
            ...current.suppressionScopeOverrides,
            [suggestion.logicalId]: scope,
          },
    }));
    setEditorMessage(
      scope === "whole-rally"
        ? "The whole rally will be left out, including the extra time around it."
        : "Only the highlighted part will be left out; the rest of the rally stays.",
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

  function setKeepBoundary(id: string, side: BoundarySide, value: number) {
    updateCut(id, (cut) => {
      if (cut.origin === "manual") {
        if (side === "start") {
          const start = roundTime(
            Math.max(analysisStart, Math.min(cut.keepEnd - MIN_CUT_SECONDS, value)),
          );
          return { ...cut, coreStart: start, keepStart: start };
        }
        const end = roundTime(
          Math.max(cut.keepStart + MIN_CUT_SECONDS, Math.min(analysisEnd, value)),
        );
        return { ...cut, coreEnd: end, keepEnd: end };
      }
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

  function setCoreBoundary(id: string, side: BoundarySide, value: number) {
    updateDraft((current) => side === "start"
      ? setCutCoreStart(current, id, value, analysisStart)
      : setCutCoreEnd(current, id, value, analysisEnd));
  }

  function beginBoundaryDrag(
    edge: BoundaryEdge,
    side: BoundarySide,
    event: ReactPointerEvent<HTMLButtonElement>,
  ) {
    if (!selected || !detailRailRef.current) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const bounds = detailRailRef.current.getBoundingClientRect();
    boundaryDragRef.current = {
      pointerId: event.pointerId,
      edge,
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
    if (drag.edge === "core") {
      setCoreBoundary(selected.id, drag.side, time);
    } else {
      setKeepBoundary(selected.id, drag.side, time);
    }
  }

  function endBoundaryDrag(event: ReactPointerEvent<HTMLButtonElement>) {
    if (boundaryDragRef.current?.pointerId === event.pointerId) {
      boundaryDragRef.current = null;
    }
  }

  function nudgeKeepBoundary(side: BoundarySide, delta: number) {
    if (!selected) return;
    setKeepBoundary(
      selected.id,
      side,
      (side === "start" ? selected.keepStart : selected.keepEnd) + delta,
    );
  }

  function splitSelectedAtPlayhead() {
    if (!selected) return;
    const result = splitCutAt(
      draft,
      selected.id,
      playbackTime,
      analysisStart,
      analysisEnd,
    );
    if (!result) {
      setEditorMessage("Move the playhead inside the rally before splitting.");
      return;
    }
    updateDraft(() => result.draft);
    setSelectedSuppressionId("");
    setSelectedId(result.newCut.id);
    setEditorMessage(
      `Split ${selected.id} at ${preciseTime(playbackTime)}. Both parts can now be trimmed independently.`,
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
      `Added ${paddingSeconds.toFixed(1)} seconds ${side} every suggested clip.`,
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
        ? "Clip checked."
        : "Clip added back to the review list.",
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
          ? "Clip checked. Your review list is complete."
          : "Your review list is complete.",
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
      ? "Clip checked. "
      : "";
    setEditorMessage(
      reviewedPrefix + "Check that this clip contains a rally, then keep it or leave it out.",
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
    const markerTarget = target?.closest<HTMLButtonElement>(
      "[data-timeline-marker]",
    );
    const markerTimestamp = Number(markerTarget?.dataset.markerTimestamp);
    const tapMarker = Number.isFinite(markerTimestamp)
      ? markerTarget?.dataset.timelineMarker === "serve" &&
        markerTarget.dataset.markerId
        ? {
            kind: "serve" as const,
            markerId: markerTarget.dataset.markerId,
            timestamp: markerTimestamp,
          }
        : markerTarget?.dataset.timelineMarker === "switch"
          ? { kind: "switch" as const, timestamp: markerTimestamp }
          : null
      : null;
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
      tapMarker,
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
    } else if (drag.tapMarker) {
      suppressTimelineClickUntilRef.current = Date.now() + 800;
      if (drag.tapMarker.kind === "serve") {
        selectServeMarker(
          drag.tapMarker.markerId,
          drag.tapMarker.timestamp,
        );
      } else {
        seekTo(drag.tapMarker.timestamp, false);
      }
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
      for (const key of cutDraftStorageKeys(seed.analysisId)) {
        window.localStorage.removeItem(key);
      }
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
    setSelectedServeMarkerId("");
    scoreInferenceStartedRef.current = false;
    setScoreInferenceStatus("idle");
    setScoreInferenceMessage(null);
    setEditorMessage("Your review has been reset to VolleyCut's original suggestions.");
  }

  async function saveProjectFile() {
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
            title: `VolleyCut project · ${initialAnalysis.sourceFilename}`,
          });
          setEditorMessage("Shared your VolleyCut project file.");
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
      setEditorMessage("Saved your VolleyCut project file.");
    } catch (cause) {
      setEditorMessage(
        `Could not save the project: ${cause instanceof Error ? cause.message : String(cause)}`,
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
          {
            scoreOverlay: draft.scoreTracking.enabled && draft.renderScoreOverlay
              ? {
                  scoreTracking: draft.scoreTracking,
                  renderPointTimeline: draft.renderScoreTimeline,
                  excludedRallyIds: [...excludedRallyIds],
                  ignoredIntervals: draft.ignoredIntervals,
                  rallyRanges: activeScoreRallyRanges,
                  mergedRanges: finalIntervals,
                }
              : undefined,
          },
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

  function renderOverviewRow(windowStart: number, windowEnd: number, rowIndex: number) {
    const windowDuration = Math.max(0.001, windowEnd - windowStart);
    const clip = (start: number, end: number) => ({
      start: Math.max(windowStart, start),
      end: Math.min(windowEnd, end),
    });
    const position = (start: number, end: number) => ({
      left: `${timelinePercent(start - windowStart, windowDuration)}%`,
      width: `${timelinePercent(end - start, windowDuration)}%`,
    });
    const playheadInRow = playbackTime >= windowStart &&
      (rowIndex === 1 ? playbackTime <= windowEnd : playbackTime < windowEnd);

    return <div className={styles.overviewRow} key={`${windowStart}-${windowEnd}`}>
      <div
        className={styles.overviewRail}
        onPointerDown={(event) => beginTimelineSeek(event, windowStart, windowEnd)}
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
        aria-label={`Marked game overview, ${rowIndex === 0 ? "first" : "second"} half`}
      >
        {overviewIgnoredIntervals.map((interval) => {
          const segment = clip(interval.start, interval.end);
          return segment.end > segment.start ? <span
            key={interval.id}
            className={styles.overviewIgnored}
            style={position(segment.start, segment.end)}
          /> : null;
        })}
        {overviewCuts.map((cut) => {
          const segment = clip(cut.keepStart, cut.keepEnd);
          if (segment.end <= segment.start) return null;
          const before = clip(cut.keepStart, cut.coreStart);
          const core = clip(cut.coreStart, cut.coreEnd);
          const after = clip(cut.coreEnd, cut.keepEnd);
          const innerPosition = (start: number, end: number) => ({
            left: `${timelinePercent(start - segment.start, segment.end - segment.start)}%`,
            width: `${timelinePercent(end - start, segment.end - segment.start)}%`,
          });
          return <button
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
            style={position(segment.start, segment.end)}
            onClick={() => selectCut(cut)}
            aria-label={`${!cut.included ? "Left out" : effectiveKeptIds.has(cut.id) ? "Included" : "Left out"} clip, ${preciseTime(cut.keepStart)} to ${preciseTime(cut.keepEnd)}, ${modelAgreementLabel(cut)}${reviewedCutIds.has(cut.id) ? ", checked" : ""}`}
          >
            {before.end > before.start && <span
              className={styles.overviewPadding}
              style={innerPosition(before.start, before.end)}
            />}
            {core.end > core.start && <span
              className={styles.overviewCore}
              style={innerPosition(core.start, core.end)}
            />}
            {after.end > after.start && <span
              className={styles.overviewPadding}
              style={innerPosition(after.start, after.end)}
            />}
          </button>;
        })}
        {suppressionSuggestions.map((suggestion) => {
          const segment = clip(suggestion.start, suggestion.end);
          if (segment.end <= segment.start) return null;
          const state = suppressionSuggestionState(suggestion, draft);
          return <button
            type="button"
            key={suggestion.id}
            className={styles.overviewSuppression}
            data-state={state}
            data-selected={suggestion.id === selectedSuppression?.id || undefined}
            style={position(segment.start, segment.end)}
            onClick={(event) => {
              event.stopPropagation();
              selectSuppression(suggestion);
            }}
            aria-label={`${state === "suppressed" ? "Suppressed" : state === "edited-kept" ? "Edited rally—kept" : "Suggestion kept"}, ${preciseTime(suggestion.start)} to ${preciseTime(suggestion.end)}, ${Math.round(suggestion.score * 100)}% score`}
            title={`${state === "suppressed" ? "Suppressed" : state === "edited-kept" ? "Edited rally—kept" : "Suggestion kept"} · ${preciseTime(suggestion.start)}–${preciseTime(suggestion.end)}`}
          />;
        })}
        {finalIntervals.flatMap((interval) =>
          (interval.joinedGaps ?? [])
            .filter((gap) => activeMarkStart === null || gap.end >= activeMarkStart)
            .map((gap) => {
              const segment = clip(gap.start, gap.end);
              return segment.end > segment.start ? <span
                key={`joined-gap-${gap.start}-${gap.end}-${interval.cutIds.join("-")}`}
                className={styles.overviewJoinedGap}
                style={position(segment.start, segment.end)}
                title={`Retained short gap · ${preciseTime(gap.start)} to ${preciseTime(gap.end)}`}
              /> : null;
            }),
        )}
        {draft.scoreTracking.enabled && draft.scoreTracking.serveMarkers
          .filter((marker) => marker.timestamp >= windowStart &&
            (rowIndex === 1 ? marker.timestamp <= windowEnd : marker.timestamp < windowEnd) &&
            !isScoreTimestampIgnored(marker.timestamp, draft.ignoredIntervals) &&
            (!marker.rallyId || !excludedRallyIds.has(marker.rallyId)))
          .map((marker) => (
            <button
              type="button"
              key={marker.id}
              className={styles.serveTimelineMarker}
              data-timeline-marker="serve"
              data-marker-id={marker.id}
              data-marker-timestamp={marker.timestamp}
              data-side={marker.side}
              data-selected={marker.id === selectedServeMarkerId || undefined}
              style={{ left: `${timelinePercent(marker.timestamp - windowStart, windowDuration)}%` }}
              aria-label={`Serve marker at ${preciseTime(marker.timestamp)}, ${marker.side === "review" ? "review needed" : `${marker.side} side`}`}
              title={`Serve · ${preciseTime(marker.timestamp)} · ${marker.side}`}
              onClick={(event) => {
                event.stopPropagation();
                selectServeMarker(marker.id, marker.timestamp);
              }}
            >
              <span aria-hidden="true">🏐</span>
              <i />
            </button>
          ))}
        {draft.scoreTracking.enabled && activeScoreTracking.sideSwitchMarkers
          .filter((marker) => marker.timestamp >= windowStart &&
            (rowIndex === 1 ? marker.timestamp <= windowEnd : marker.timestamp < windowEnd))
          .map((marker) => (
            <button
              type="button"
              key={marker.id}
              className={styles.switchTimelineMarker}
              data-timeline-marker="switch"
              data-marker-timestamp={marker.timestamp}
              style={{ left: `${timelinePercent(marker.timestamp - windowStart, windowDuration)}%` }}
              aria-label={`Team side switch at ${preciseTime(marker.timestamp)}, applies to the next serve marker`}
              title={`Team side switch · ${preciseTime(marker.timestamp)} · applies to next serve`}
              onClick={(event) => {
                event.stopPropagation();
                selectSideSwitchMarker(marker.id, marker.timestamp);
              }}
            >
              <span aria-hidden="true">⇄</span>
              <i />
            </button>
          ))}
        {playheadInRow && <span
          className={styles.playhead}
          style={{ left: `${timelinePercent(playbackTime - windowStart, windowDuration)}%` }}
        />}
      </div>
      <div className={styles.overviewTimes}>
        <span>{formatTime(windowStart)}</span>
        <span>{formatTime((windowStart + windowEnd) / 2)}</span>
        <span>{formatTime(windowEnd)}</span>
      </div>
    </div>;
  }

  return (
    <main
      className={styles.page}
      data-design="classic"
    >
      {header}

      <nav className={styles.workflow} aria-label="Editing progress">
        <span data-done="true"><b>1</b> Video</span>
        <i aria-hidden="true" />
        <span data-current="true"><b>2</b> Review</span>
        <i aria-hidden="true" />
        <span><b>3</b> Export</span>
      </nav>

      <section
        className={styles.sourcePicker}
        data-tour="editor-source"
        aria-label="Current video"
      >
        <div className={styles.sourceMeta}>
          <span>NOW REVIEWING</span>
          <strong>{initialAnalysis.sourceFilename}</strong>
        </div>
        <div className={styles.sourceMeta}>
          <span>RALLIES FOUND</span>
          <strong>{initialAnalysis.rallies.length} suggested clips</strong>
        </div>
        <span className={styles.storageState} data-ready={storageReady || undefined}>
          <i /> {storageMessage}
        </span>
        {!sourceFile && (
          <label className={styles.attachSource}>
            Choose original video to continue
            <input
              type="file"
              accept="video/*,.mkv,.webm,.mov,.mp4,.m4v"
              onChange={(event) => onAttachSource(event.currentTarget.files?.[0] ?? null)}
            />
          </label>
        )}
      </section>

      {sourceError && <p className={styles.sourceError}>{sourceError}</p>}

      <label
        className={styles.scoreTrackingToggle}
        data-enabled={draft.scoreTracking.enabled || undefined}
        data-tour="editor-score"
      >
        <input
          type="checkbox"
          checked={draft.scoreTracking.enabled}
          onChange={(event) => toggleScoreTracking(event.currentTarget.checked)}
        />
        <span>
          <strong>Add a scoreboard <em>Optional</em></strong>
          <small>
            VolleyCut builds the score from serve markers. You’ll check which
            court side is serving and add any missing serves or team side switches.
          </small>
        </span>
      </label>

      <section className={styles.editorShell}>
        <aside
          className={styles.summaryCard}
          data-tour="editor-settings"
          aria-label="Final edit settings"
        >
          <div className={styles.summaryStats}>
            <span>YOUR FINAL VIDEO</span>
            <strong>{formatTime(keptSeconds)}</strong>
            <div>
              <p>{keptCount} clips included</p>
              <p>{removedCount + fullyIgnoredCount} clips left out</p>
              <p>{unreviewedLowConfidenceCuts.length === 0 ? "Everything has been checked" : `${unreviewedLowConfidenceCuts.length} suggested clips to check`}</p>
            </div>
          </div>
          <div className={styles.finalVideoOptions}>
            <div className={styles.finalVideoOptionsHeading}>
              <span>FINAL VIDEO OPTIONS</span>
            </div>
            <div className={styles.previewToggles}>
            <label
              className={styles.cutPreviewToggle}
              data-tour="editor-play-final-cut"
            >
              <input
                type="checkbox"
                checked={cutPreviewEnabled}
                onChange={(event) => toggleCutPreview(event.currentTarget.checked)}
              />
              <span>
                <strong>Play only the final video</strong>
                <small>The player skips every part that will not be saved.</small>
              </span>
            </label>
            <label
              className={`${styles.cutPreviewToggle} ${styles.scoreOverlayToggle}`}
              data-enabled={draft.scoreTracking.enabled && draft.renderScoreOverlay || undefined}
              data-tour="editor-score-overlay"
            >
              <input
                type="checkbox"
                checked={draft.scoreTracking.enabled && draft.renderScoreOverlay}
                onChange={(event) => toggleScoreOverlay(event.currentTarget.checked)}
              />
              <span>
                <strong>Add scores to the final video</strong>
                <small>
                  {draft.scoreTracking.enabled
                    ? "Uses your checked serve and side-switch markers in the saved video."
                    : "Turns on score tracking so you can check the markers before saving."}
                </small>
              </span>
            </label>
            {draft.scoreTracking.enabled && draft.renderScoreOverlay && (
              <label
                className={`${styles.cutPreviewToggle} ${styles.scoreTimelineToggle}`}
                data-enabled={draft.renderScoreTimeline || undefined}
              >
                <input
                  type="checkbox"
                  checked={draft.renderScoreTimeline}
                  onChange={(event) =>
                    toggleScoreTimeline(event.currentTarget.checked)
                  }
                />
                <span>
                  <strong>Show the point history</strong>
                  <small>
                    Show the two team rails beside the score when a new point starts.
                  </small>
                </span>
              </label>
            )}
            </div>
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
                      ? `The direct download was unavailable (${streamFallbackReason}). VolleyCut will use private on-device storage instead.`
                      : "VolleyCut will save directly to Downloads when possible and automatically try another private on-device method if needed."}
                  </small>
                </span>
              </div>
            )}
          </div>
          <div className={styles.summaryActions}>
            <button
              type="button"
              className={styles.exportButton}
              data-tour="editor-export-video"
              disabled={
                exportState === "exporting" || finalIntervals.length === 0 || !localExportSupported
              }
              title={exportUnavailableMessage ?? undefined}
              onClick={() => void (preparedExport ? deliverExport() : exportVideo())}
            >
              {exportState === "exporting"
                ? "Creating your video…"
                : preparedExport
                  ? "Share or save video"
                  : exportMode === "stream-download"
                    ? "Save final video"
                  : directDiskSupported
                    ? "Save final video"
                    : "Create final video"}
            </button>
            <details className={styles.projectTools}>
              <summary>Project options</summary>
              <p className={styles.projectSaveHelp}>
                Save your edits and score markers as a project file. The video is
                not included; open the project later and choose the original video again.
              </p>
              <button
                type="button"
                className={styles.quietButton}
                onClick={() => void saveProjectFile()}
              >
                Save project
              </button>
              <button type="button" className={styles.quietButton} onClick={resetDraft}>
                Start this review over
              </button>
            </details>
            <details className={styles.advancedSettings}>
              <summary>
                <span>Fine-tune cleanup and timing</span>
                <small>Optional</small>
              </summary>
              <div className={styles.advancedSettingsBody}>
                <div className={styles.suppressionControls} data-tour="editor-suppression">
                  <div>
                    <span>AUTOMATIC CLEANUP</span>
                    <strong>{cleanupLabel(draft.selectedSuppressionPolicy)}</strong>
                  </div>
                  {initialAnalysis.suppression ? (
                    <>
                      <label
                        className={styles.suppressionSelect}
                        htmlFor="suppression-policy"
                      >
                        <span>Cleanup strength</span>
                        <select
                          id="suppression-policy"
                          value={draft.selectedSuppressionPolicy}
                          onChange={(event) => setSuppressionPolicy(
                            event.currentTarget.value as SuppressionPolicyId,
                          )}
                        >
                          {(["none", "conservative", "balanced", "aggressive"] as const).map((policy) => (
                            <option key={policy} value={policy}>
                              {cleanupLabel(policy)}
                            </option>
                          ))}
                        </select>
                      </label>
                      {draft.selectedSuppressionPolicy === "none" ? (
                        <small>Choose a level to remove moments that probably are not live play.</small>
                      ) : suppressionSuggestions.length === 0 ? (
                        <small>No extra cleanup is suggested for this game.</small>
                      ) : (
                        <small>
                          VolleyCut will leave out {suppressionSuggestions.length} likely non-play moments. You can keep any of them while reviewing.
                        </small>
                      )}
                    </>
                  ) : (
                    <>
                      <small>
                        Automatic cleanup is not available for this older project.
                      </small>
                      {sourceFile && (
                        <button type="button" onClick={onRequestSuppression}>
                          Add automatic cleanup
                        </button>
                      )}
                    </>
                  )}
                </div>
                <div className={styles.paddingControls}>
                  <div className={styles.paddingPair} data-tour="editor-padding">
                    <div className={styles.paddingControl}>
                      <label htmlFor="cut-padding-before">
                        <span>Before</span>
                        <output>{draft.beforePaddingSeconds.toFixed(1)}s</output>
                      </label>
                      <input
                        id="cut-padding-before"
                        aria-label="Extra time before each clip"
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
                        aria-label="Extra time after each clip"
                        type="range"
                        min="0"
                        max="10"
                        step="0.5"
                        value={draft.afterPaddingSeconds}
                        onChange={(event) => setGlobalPadding("after", Number(event.currentTarget.value))}
                      />
                      <div><span>0s</span><span>10s</span></div>
                    </div>
                  </div>
                  <div className={styles.paddingControl} data-tour="editor-join-gaps">
                    <label htmlFor="cut-join-gap">
                      <span>Join gaps under</span>
                      <output>{draft.joinGapSeconds.toFixed(1)}s</output>
                    </label>
                    <input
                      id="cut-join-gap"
                      aria-label="Keep short breaks under this length"
                      type="range"
                      min="0"
                      max="10"
                      step="0.5"
                      value={draft.joinGapSeconds}
                      onChange={(event) => setJoinGapSeconds(Number(event.currentTarget.value))}
                    />
                    <div><span>Off</span><span>10s</span></div>
                  </div>
                  <small>
                    Add breathing room around each rally, or keep very short breaks between nearby rallies.
                  </small>
                </div>
              </div>
            </details>
          </div>
          <div className={styles.exportDetails} aria-live="polite">
            <p>
              Saves a high-quality video at its original size. Your video stays on this device.
            </p>
            {exportUnavailableMessage && (
              <strong className={sourceFile ? styles.exportError : styles.exportNotice}>
                {exportUnavailableMessage}
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
                    ? `${exportRate.toFixed(2)}× video speed`
                    : "Estimating the time remaining…"}
                  {exportWakeLock === "active" ? " · screen awake" : ""}
                </small>
              </div>
            )}
            {preparedExport && (
              <strong>
                Your video is ready. Tap Share or save video to choose where it goes.
              </strong>
            )}
            {exportState === "done" && !preparedExport && (
              <strong>
                Your video has been saved
                {exportProgress ? ` in ${preciseTime(exportProgress.elapsedSeconds)}` : ""}.
              </strong>
            )}
            {exportError && <strong className={styles.exportError}>{exportError}</strong>}
          </div>
        </aside>

        <div className={styles.playerColumn}>
          <div className={styles.playerWorkspace}>
            {draft.scoreTracking.enabled && (
              <ScoreTrackingPanel
                tracking={draft.scoreTracking}
                ignoredIntervals={draft.ignoredIntervals}
                excludedRallyIds={excludedRallyIds}
                playbackTime={playbackTime}
                scoreBoundaryTime={scoreBoundaryTime}
                selectedServeMarkerId={selectedServeMarkerId}
                inferenceStatus={scoreInferenceStatus}
                inferenceMessage={scoreInferenceMessage}
                inferenceSteps={scoreInferenceSteps}
                canRunInference={Boolean(
                  sourceFile &&
                  ((!initialAnalysis.servingSide && initialAnalysis.productionServeOutputs) ||
                    (sideSwitchEnabled && !initialAnalysis.sideSwitch && initialAnalysis.productionComponents &&
                      initialAnalysis.productionStateOutputs))
                )}
                onChange={updateScoreTracking}
                onSelectServeMarker={selectServeMarker}
                onSelectSideSwitchMarker={selectSideSwitchMarker}
                onRunInference={() => {
                  scoreInferenceStartedRef.current = false;
                  void runScoreInference();
                }}
              />
            )}

            <div className={styles.videoColumn}>
              <div
                className={styles.videoStage}
                data-tour="editor-video"
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
            {draft.scoreTracking.enabled && draft.renderScoreOverlay && (
              <ScoreOverlay
                className={styles.videoScoreOverlay}
                prepared={preparedScoreOverlay}
                timestamp={playbackTime}
                videoWidth={initialAnalysis.width}
                videoHeight={initialAnalysis.height}
              />
            )}
            <div className={styles.timecode}>
              {preciseTime(playbackTime)} <span>/ game ends {preciseTime(analysisEnd)}</span>
            </div>
              </div>

              <div className={styles.transport} data-tour="editor-transport">
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
            </div>

          </div>

          <section className={styles.overviewSection} data-tour="editor-overview">
            <div className={styles.sectionHeading}>
              <div>
                <span>GAME TIMELINE</span>
                <strong>Select a clip to check or adjust it</strong>
              </div>
              <small>
                {draft.cuts.length} clips · tap anywhere to move through the video
              </small>
            </div>
            <div className={styles.confidenceReview}>
              <div className={styles.reviewStatus}>
                <p>
                  {unreviewedLowConfidenceCuts.length === 0
                    ? "All suggested clips have been checked"
                    : `${unreviewedLowConfidenceCuts.length} suggested ${unreviewedLowConfidenceCuts.length === 1 ? "clip" : "clips"} to check`}
                </p>
                {draft.selectedSuppressionPolicy !== "none" && suppressionSuggestions.length > 0 && (
                  <p data-suppression="true">
                    Automatic cleanup left out {Math.max(0, noSuppressionSeconds - keptSeconds).toFixed(1)} seconds
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
                      ? "Looks good · next"
                      : "Check next clip"}
                </button>
                {draft.selectedSuppressionPolicy !== "none" && suppressionSuggestions.length > 0 && (
                  <button
                    type="button"
                    className={styles.nextSuppression}
                    onClick={reviewNextSuppression}
                  >
                    Check next cleanup
                  </button>
                )}
              </div>
            </div>
            <div className={styles.timelineLegend} aria-label="Timeline legend">
              <span><i data-kind="ordinary" /> Included</span>
              <span><i data-kind="suppressed" /> Left out</span>
              <span><i data-kind="suggestion-kept" /> Kept after review</span>
              <span><i data-kind="ignored" /> Not part of game</span>
              <span><i data-kind="joined" /> Short break kept</span>
              {draft.scoreTracking.enabled && (
                <>
                  <span><i data-kind="serve-marker">🏐</i> Serve</span>
                  <span><i data-kind="side-switch">⇄</i> Side switch</span>
                </>
              )}
            </div>
            <div className={styles.overviewRows} data-tour="editor-score-markers">
              {renderOverviewRow(
                analysisStart,
                (analysisStart + analysisEnd) / 2,
                0,
              )}
              {renderOverviewRow(
                (analysisStart + analysisEnd) / 2,
                analysisEnd,
                1,
              )}
            </div>
          </section>
        </div>

      </section>

      <section
        className={styles.focusEditor}
        data-tour="editor-focus"
        aria-label="Focused range editor"
      >
        <div className={styles.focusHeader}>
          <div>
            <span>SELECTED CLIP</span>
            <strong>
              {selectedSuppression
                ? `Cleanup suggestion · ${preciseTime(selectedSuppression.start)}–${preciseTime(selectedSuppression.end)}`
                : selected
                  ? `Clip ${selectedIndex + 1} · ${selected.origin === "manual" ? "Added by you" : modelAgreementLabel(selected)}${selectedIsReviewed ? " · checked" : ""}`
                  : "No clip selected"}
            </strong>
          </div>
          <div className={styles.focusHeaderControls}>
            <label className={styles.focusLock}>
              <input
                type="checkbox"
                checked={focusLocked}
                onChange={(event) => setFocusLocked(event.currentTarget.checked)}
              />
              <span>Keep selected</span>
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
                <small>{selected.origin === "manual" ? "ADDED RALLY" : "RALLY"}</small>
              </span>
              {selectedSuppression && (
                <span
                  className={styles.detailSuppression}
                  data-state={suppressionSuggestionState(selectedSuppression, draft)}
                  style={{
                    left: `${timelinePercent(selectedSuppression.start - focus.start, focus.end - focus.start)}%`,
                    width: `${timelinePercent(selectedSuppression.end - selectedSuppression.start, focus.end - focus.start)}%`,
                  }}
                  aria-label={`Selected cleanup suggestion, ${preciseTime(selectedSuppression.start)} to ${preciseTime(selectedSuppression.end)}`}
                >
                  <small>
                    {suppressionSuggestionState(selectedSuppression, draft) === "suppressed"
                      ? selectedSuppressionScope === "whole-rally"
                        ? "WHOLE RALLY SUPPRESSED"
                        : "VETO REGION SUPPRESSED"
                      : suppressionSuggestionState(selectedSuppression, draft) === "edited-kept"
                        ? "EDITED RALLY—KEPT"
                        : "SUGGESTION KEPT"}
                  </small>
                </span>
              )}
              {!selectedSuppression && selected.origin === "cached-label" && <button
                type="button"
                className={`${styles.boundaryHandle} ${styles.startHandle}`}
                style={{
                  left: `${timelinePercent(selected.keepStart - focus.start, focus.end - focus.start)}%`,
                }}
                aria-label={`Adjust output start at ${preciseTime(selected.keepStart)}`}
                onPointerDown={(event) => beginBoundaryDrag("output", "start", event)}
                onPointerMove={moveBoundary}
                onPointerUp={endBoundaryDrag}
                onPointerCancel={endBoundaryDrag}
                onLostPointerCapture={endBoundaryDrag}
              >
                <i />
              </button>}
              {!selectedSuppression && selected.origin === "cached-label" && <button
                type="button"
                className={`${styles.boundaryHandle} ${styles.endHandle}`}
                style={{
                  left: `${timelinePercent(selected.keepEnd - focus.start, focus.end - focus.start)}%`,
                }}
                aria-label={`Adjust output end at ${preciseTime(selected.keepEnd)}`}
                onPointerDown={(event) => beginBoundaryDrag("output", "end", event)}
                onPointerMove={moveBoundary}
                onPointerUp={endBoundaryDrag}
                onPointerCancel={endBoundaryDrag}
                onLostPointerCapture={endBoundaryDrag}
              >
                <i />
              </button>}
              {!selectedSuppression && <button
                type="button"
                className={`${styles.boundaryHandle} ${styles.coreHandle} ${styles.startHandle}`}
                style={{
                  left: `${timelinePercent(selected.coreStart - focus.start, focus.end - focus.start)}%`,
                }}
                aria-label={`Adjust rally start at ${preciseTime(selected.coreStart)}`}
                onPointerDown={(event) => beginBoundaryDrag("core", "start", event)}
                onPointerMove={moveBoundary}
                onPointerUp={endBoundaryDrag}
                onPointerCancel={endBoundaryDrag}
                onLostPointerCapture={endBoundaryDrag}
              >
                <i />
              </button>}
              {!selectedSuppression && <button
                type="button"
                className={`${styles.boundaryHandle} ${styles.coreHandle} ${styles.endHandle}`}
                style={{
                  left: `${timelinePercent(selected.coreEnd - focus.start, focus.end - focus.start)}%`,
                }}
                aria-label={`Adjust rally end at ${preciseTime(selected.coreEnd)}`}
                onPointerDown={(event) => beginBoundaryDrag("core", "end", event)}
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

            {!selectedSuppression && <>
              <div
                className={styles.rallyRangeEditor}
                data-tour="editor-change-duration"
              >
                <div className={styles.rallyRangeSummary}>
                  <span>RALLY LENGTH</span>
                  <output>
                    {preciseTime(selected.coreStart)} – {preciseTime(selected.coreEnd)}
                  </output>
                </div>
                <p>
                  Drag the orange handles to change where this rally starts and ends.
                </p>
                <div className={styles.rallyEdgeActions}>
                  <button
                    type="button"
                    disabled={playbackTime > selected.coreEnd - MIN_CUT_SECONDS}
                    onClick={() => setCoreBoundary(selected.id, "start", playbackTime)}
                  >
                    Set rally start here
                  </button>
                  <button
                    type="button"
                    className={styles.splitAction}
                    data-tour="editor-split-rally"
                    disabled={
                      playbackTime < selected.coreStart + MIN_CUT_SECONDS ||
                      playbackTime > selected.coreEnd - MIN_CUT_SECONDS
                    }
                    onClick={splitSelectedAtPlayhead}
                  >
                    Split here
                  </button>
                  <button
                    type="button"
                    disabled={playbackTime < selected.coreStart + MIN_CUT_SECONDS}
                    onClick={() => setCoreBoundary(selected.id, "end", playbackTime)}
                  >
                    Set rally end here
                  </button>
                </div>
              </div>

              {selected.origin === "cached-label" && <div className={styles.outputEdgeEditor}>
                <p>EXTRA TIME AROUND THIS RALLY</p>
                <div className={styles.boundaryControls}>
                  <fieldset>
                    <legend>Output start</legend>
                    <output>{preciseTime(selected.keepStart)}</output>
                    <div>
                      <button type="button" onClick={() => nudgeKeepBoundary("start", -1)}>−1s</button>
                      <button type="button" onClick={() => nudgeKeepBoundary("start", -0.1)}>−0.1s</button>
                      <button type="button" onClick={() => nudgeKeepBoundary("start", 0.1)}>+0.1s</button>
                      <button type="button" onClick={() => nudgeKeepBoundary("start", 1)}>+1s</button>
                    </div>
                  </fieldset>
                  <fieldset>
                    <legend>Output end</legend>
                    <output>{preciseTime(selected.keepEnd)}</output>
                    <div>
                      <button type="button" onClick={() => nudgeKeepBoundary("end", -1)}>−1s</button>
                      <button type="button" onClick={() => nudgeKeepBoundary("end", -0.1)}>−0.1s</button>
                      <button type="button" onClick={() => nudgeKeepBoundary("end", 0.1)}>+0.1s</button>
                      <button type="button" onClick={() => nudgeKeepBoundary("end", 1)}>+1s</button>
                    </div>
                  </fieldset>
                </div>
              </div>}
            </>}

            <div className={styles.focusActions}>
              {selectedSuppression ? (
                <>
                  <p className={styles.suppressionStateText}>
                    {suppressionSuggestionState(selectedSuppression, draft) === "suppressed"
                      ? selectedSuppressionScope === "whole-rally"
                        ? "The whole rally will be left out."
                        : "The highlighted part will be left out; the rest stays."
                      : suppressionSuggestionState(selectedSuppression, draft) === "edited-kept"
                        ? "This stays because you already edited an overlapping rally."
                        : "You chose to keep this part in the final video."}
                  </p>
                  <fieldset className={styles.suppressionScopeControl}>
                    <legend>What should be left out?</legend>
                    <div
                      className={styles.suppressionScopeOptions}
                      role="group"
                      aria-label="What should be left out"
                    >
                      {SUPPRESSION_SCOPE_IDS.map((scope) => (
                        <button
                          type="button"
                          key={scope}
                          data-active={selectedSuppressionScope === scope || undefined}
                          aria-pressed={selectedSuppressionScope === scope}
                          title={scope === "whole-rally"
                            ? "Leave out the entire rally"
                            : "Leave out only the highlighted part"}
                          onClick={() => setSuppressionScope(selectedSuppression, scope)}
                        >
                          {scope === "whole-rally" ? "Whole rally" : "Only red section"}
                        </button>
                      ))}
                    </div>
                    <small>
                      {selectedSuppressionScope === "whole-rally"
                        ? "Leaves out the entire rally and the extra time around it."
                        : "Leaves out only the highlighted part; the rest of the rally stays."}
                    </small>
                  </fieldset>
                  <button
                    type="button"
                    className={styles.suppressAction}
                    data-active={suppressionSuggestionState(selectedSuppression, draft) === "suppressed" || undefined}
                    onClick={() => setSuppressionDecision(selectedSuppression, "suppress")}
                  >
                    Leave out
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
                    {selected.included ? "✓ Include this clip" : "+ Put this clip back"}
                  </button>
                  <button type="button" onClick={previewSelected}>Preview clip</button>
                  <button type="button" onClick={resetSelectedPadding}>Reset extra time</button>
                  {selectedReviewCandidate && (
                    <button
                  type="button"
                  className={styles.reviewAction}
                  data-reviewed={selectedIsReviewed || undefined}
                  onClick={toggleSelectedReviewed}
                >
                  {selectedIsReviewed ? "✓ Checked" : "Looks good"}
                    </button>
                  )}
                  {selected.origin === "manual" && (
                    <button
                  type="button"
                  className={styles.dangerButton}
                  onClick={() => deleteManualCut(selected.id)}
                >
                  Delete added clip
                    </button>
                  )}
                </>
              )}
            </div>
          </>
        ) : (
          <p className={styles.emptyMessage}>Choose a clip from the timeline, or add a missed rally below.</p>
        )}
      </section>

      <section className={styles.markingTools} data-tour="editor-marking">
        <div className={styles.markingCard}>
          <div>
            <span>ADD A MISSED RALLY</span>
            <strong>{manualStart === null ? "Find the first frame" : `Started ${preciseTime(manualStart)}`}</strong>
            <p>Move to the start of the rally and mark it, then do the same for the end.</p>
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
            <span>LEAVE OUT A SECTION</span>
            <strong>{ignoreStart === null ? "Remove unusable footage" : `Started ${preciseTime(ignoreStart)}`}</strong>
            <p>Use this for camera gaps, breaks, or anything that is not part of the game.</p>
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
            <option value="boundary-ambiguous">Unclear start or end</option>
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
              <span>RALLIES YOU ADDED</span>
              <strong>Saved automatically on this device</strong>
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
            <div><span>SECTIONS LEFT OUT</span><strong>These will not appear in the final video</strong></div>
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

      <section className={styles.cutList} data-tour="editor-cuts">
        <div className={styles.sectionHeading}>
          <div><span>ALL CLIPS</span><strong>Every rally in your final video</strong></div>
          <small>{keptCount} included · {removedCount + fullyIgnoredCount} left out</small>
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
                <em>{cut.origin === "manual" ? "ADDED" : suppressionSuggestions.some(
                  (suggestion) => suggestion.start < cut.coreEnd && suggestion.end > cut.coreStart,
                ) ? "CLEANUP" : reviewedCutIds.has(cut.id) ? "CHECKED" : isModelDisagreement(cut) ? "CHECK" : "READY"}</em>
              </button>
              <button
                type="button"
                className={styles.cutToggle}
                onClick={() => updateCut(cut.id, (current) => ({
                  ...current,
                  included: !current.included,
                }))}
              >
                {!cut.included ? "Left out" : effectiveKeptIds.has(cut.id) ? "Included" : "Left out"}
              </button>
            </article>
          ))}
        </div>
      </section>

      <GuidedTour
        stage="editor"
        scoreTrackingEnabled={draft.scoreTracking.enabled}
      />
      <SiteFooter />
    </main>
  );
}
