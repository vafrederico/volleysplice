import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";

import {
  materializeFinalCutIntervals,
  rallySuppressionDecisionKey,
  PLAYBACK_RATES,
  type CutDraft,
  type EditableCut,
} from "@/lib/cut-draft";
import { GOOGLE_PLAY_URL } from "@/lib/android-app";
import { timelinePercent } from "@/lib/edit-list";
import {
  alignServeMarkersToRallyStarts,
  deriveScoreAt,
  removeServeMarker,
  removeSideSwitchMarker,
  scoreTrackingOutsideExcludedRallies,
  scoreTrackingOutsideIgnoredIntervals,
  type ScoreTracking,
} from "@/lib/score-tracking";
import {
  buildYouTubeChapters,
  defaultYouTubeChapterOptions,
  youtubeChaptersFilename,
  youtubeChaptersText,
} from "@/lib/youtube-chapters";
import { ScoreOverlay } from "@/components/ScoreOverlay";
import { RallyDeskGuidedTour } from "@/components/GuidedTour";
import {
  createModelFeedbackBundle,
  modelFeedbackBlob,
  modelFeedbackFilename,
} from "@/lib/model-feedback";
import { prepareScoreOverlay, scorePointTimelineSnapshot } from "@/lib/score-overlay";
import { runtimeAssetUrl } from "@/lib/runtime-assets";
import { scrollElementIntoContainer } from "@/lib/scroll-container";
import type { ReadyDesignReview } from "../useDesignReview";

import { ResizableWorkspace, useWorkspaceLayout } from "./ResizableWorkspace";
import { AppMenu } from "./AppMenu";
import { cleanupDecisionsForRally, cleanupReviewDecisions, normalizeCleanupDecisions } from "./cleanup-decisions";
import { editHistoryKey, moveHistory, readEditHistory, recordEdit, writeEditHistory } from "./edit-history";
import { nextReviewItem, shortcutAction, type ReviewItem } from "./keyboard-shortcuts";

import "./taste-designs.css";

export type TasteDesignMeta = {
  slug: string;
  title: string;
  description: string;
};

export const TASTE_DESIGNS: TasteDesignMeta[] = [
  {
    slug: "rally-desk",
    title: "Rally Desk",
    description: "A keyboard-minded review console for coaches who want the next decision close at hand.",
  },
  {
    slug: "court-canvas",
    title: "Court Canvas",
    description: "A spacious, visual workspace that keeps the match and its story in the foreground.",
  },
  {
    slug: "review-lane",
    title: "Review Lane",
    description: "A calm, guided flow that presents one clear task at a time without hiding advanced tools.",
  },
  {
    slug: "film-room",
    title: "Film Room",
    description: "A cinematic dark workspace built around playback, rapid clip checks, and score context.",
  },
  {
    slug: "match-ledger",
    title: "Match Ledger",
    description: "A precise project record that makes every retained clip, marker, and output accountable.",
  },
];

type TasteDesignRouteProps = { slug: string };
type Stage = "source" | "analysis" | "review" | "deliver";
type ExportKind = "video" | "chapters" | "feedback";
type Side = "near" | "far";
type MarkerSide = Side | "review";

type Clip = {
  id: string;
  label: string;
  start: number;
  end: number;
  confidence: number;
  included: boolean;
  reviewed: boolean;
  origin: "model" | "manual";
  agreement?: EditableCut["agreement"];
};

type DesignServeMarker = {
  id: string;
  timestamp: number;
  side: MarkerSide;
  ignorePreviousPoint: boolean;
  origin: "model" | "manual";
  rallyId?: string;
};

type DesignSideSwitchMarker = {
  id: string;
  timestamp: number;
  origin: "model" | "manual";
};

type SuppressionDecision = "pending" | "kept" | "excluded";

type ExcludedRange = {
  id: string;
  start: number;
  end: number;
  reason: "camera-break" | "warmup" | "other";
};

const ANALYSIS_PHASES = [
  { at: 8, label: "Open local video" },
  { at: 20, label: "Read picture and sound" },
  { at: 64, label: "Find rally candidates" },
  { at: 84, label: "Check serve side" },
  { at: 96, label: "Prepare the review" },
] as const;

function formatTime(seconds: number): string {
  const safe = Math.max(0, Math.round(seconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = safe % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function formatPreciseTime(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = (safe % 60).toFixed(1).padStart(4, "0");
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${remainder}`
    : `${minutes}:${remainder}`;
}

function clipsFromReview(review: ReadyDesignReview): Clip[] {
  const reviewed = new Set(review.draft.reviewedCutIds);
  return review.draft.cuts.map((cut) => ({
    id: cut.id,
    label:
      cut.origin === "manual"
        ? "Missed rally added by you"
        : cut.agreement === "both-models"
          ? "Found by both rally models"
          : cut.agreement
            ? "Single-model rally suggestion"
            : "Detected rally",
    start: cut.coreStart,
    end: cut.coreEnd,
    confidence: cut.confidence,
    included: cut.included,
    reviewed: reviewed.has(cut.id),
    origin: cut.origin === "manual" ? "manual" : "model",
    agreement: cut.agreement,
  }));
}

function excludedFromReview(review: ReadyDesignReview): ExcludedRange[] {
  return review.draft.ignoredIntervals.map((interval) => ({
    id: interval.id,
    start: interval.start,
    end: interval.end,
    reason: interval.reason.includes("camera")
      ? "camera-break"
      : interval.reason.includes("warm")
        ? "warmup"
        : "other",
  }));
}

function suppressionFromReview(
  review: ReadyDesignReview,
): Record<string, SuppressionDecision> {
  return cleanupReviewDecisions(review.draft, review.cleanupSuggestions);
}

function clipRequiresConfidenceReview(clip: Clip, threshold: number): boolean {
  return clip.origin === "model" && (
    clip.agreement === "all-labels-v2-only" ||
    clip.agreement === "previous-production-only" ||
    clip.confidence < threshold
  );
}

function rangeHasReviewableFootage(
  start: number,
  end: number,
  ignoredIntervals: readonly { start: number; end: number }[],
): boolean {
  if (end <= start) return false;
  let cursor = start;
  for (const interval of [...ignoredIntervals].sort(
    (left, right) => left.start - right.start || left.end - right.end,
  )) {
    if (interval.end <= cursor) continue;
    if (interval.start > cursor) return true;
    cursor = Math.max(cursor, interval.end);
    if (cursor >= end) return false;
  }
  return cursor < end;
}

type CleanupStrength = "off" | "light" | "standard" | "strong";

function cleanupStrengthForPolicy(
  policy: CutDraft["selectedSuppressionPolicy"],
): CleanupStrength {
  switch (policy) {
    case "none":
      return "off";
    case "conservative":
      return "light";
    case "balanced":
      return "standard";
    case "aggressive":
      return "strong";
  }
}

function suppressionPolicyForCleanup(
  cleanup: CleanupStrength,
): CutDraft["selectedSuppressionPolicy"] {
  switch (cleanup) {
    case "off":
      return "none";
    case "light":
      return "conservative";
    case "standard":
      return "balanced";
    case "strong":
      return "aggressive";
  }
}

function usePrototype(
  review: ReadyDesignReview,
  initialStage: Stage = "review",
  initialDark = false,
  followPlayhead = false,
) {
  const [baseDraft, setBaseDraft] = useState(review.draft);
  const initialClips = clipsFromReview(review);
  const firstClip =
    initialClips.find((clip) =>
      clip.included &&
      !clip.reviewed &&
      clipRequiresConfidenceReview(clip, review.draft.confidenceReviewThreshold),
    ) ??
    initialClips[0];
  const [stage, setStage] = useState<Stage>(initialStage);
  const [dark, setDark] = useState(initialDark);
  const [helpOpen, setHelpOpen] = useState(false);
  const [fileName, setFileName] = useState(review.sourceName);
  const [expectedSourceName, setExpectedSourceName] = useState(review.sourceName);
  const [needsSource, setNeedsSource] = useState(review.sourceNeedsReconnect);
  const [videoUrl, setVideoUrl] = useState<string | null>(review.videoUrl);
  const [sourceStatus, setSourceStatus] = useState("Saved analysis and review edits loaded from the live app.");
  const [gameStart, setGameStart] = useState(review.gameStart);
  const [gameEnd, setGameEnd] = useState(review.gameEnd);
  const [windowError, setWindowError] = useState<string | null>(null);
  const [cropCourt, setCropCourt] = useState(review.cropCourt);
  const [detectSwitches, setDetectSwitches] = useState(review.sideSwitchEnabled);
  const [analysisProgress, setAnalysisProgress] = useState(100);
  const [clips, setClips] = useState<Clip[]>(initialClips);
  const cleanupSuggestions = useMemo(() => review.cleanupSuggestions.flatMap((suggestion) => {
    const related = clips.filter((clip) => clip.start < suggestion.end && clip.end > suggestion.start);
    return related.map((clip) => ({ ...suggestion, cutId: clip.id }));
  }), [clips, review.cleanupSuggestions]);
  const [manualStart, setManualStart] = useState<number | null>(review.draft.pendingManualStart);
  const [excludedStart, setExcludedStart] = useState<number | null>(review.draft.pendingIgnoreStart);
  const [excludedReason, setExcludedReason] = useState<ExcludedRange["reason"]>("camera-break");
  const [excludedRanges, setExcludedRanges] = useState<ExcludedRange[]>(() => excludedFromReview(review));
  const [suppressionDecisions, setSuppressionDecisions] = useState<Record<string, SuppressionDecision>>(() => suppressionFromReview(review));
  const [selectedId, setSelectedId] = useState(firstClip?.id ?? "");
  const [openedClipId, setOpenedClipId] = useState<string | null>(null);
  const initialPlayhead = firstClip?.start ?? review.gameStart;
  const [playhead, setPlayheadValue] = useState(initialPlayhead);
  const [seekRequest, setSeekRequest] = useState({ revision: 0, target: initialPlayhead });
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState<(typeof PLAYBACK_RATES)[number]>(review.draft.playbackRate);
  const [finalPreview, setFinalPreview] = useState(review.draft.cutPreviewEnabled);
  const [beforePadding, setBeforePadding] = useState(review.draft.beforePaddingSeconds);
  const [afterPadding, setAfterPadding] = useState(review.draft.afterPaddingSeconds);
  const [joinGap, setJoinGap] = useState(review.draft.joinGapSeconds);
  const [cleanup, setCleanup] = useState<CleanupStrength>(
    cleanupStrengthForPolicy(review.draft.selectedSuppressionPolicy),
  );
  const [reviewMessage, setReviewMessage] = useState(`${initialClips.filter((clip) => clip.included && !clip.reviewed && clipRequiresConfidenceReview(clip, review.draft.confidenceReviewThreshold)).length} clips remain to check in the current review.`);
  const [scoreEnabled, setScoreEnabled] = useState(review.draft.scoreTracking.enabled);
  const [scoreOverlay, setScoreOverlay] = useState(review.draft.renderScoreOverlay);
  const [teamOne, setTeamOne] = useState(review.draft.scoreTracking.team1Name);
  const [teamTwo, setTeamTwo] = useState(review.draft.scoreTracking.team2Name);
  const [servingSide, setServingSide] = useState<Side>("near");
  const [savedScoreMarkers, setScoreMarkers] = useState<DesignServeMarker[]>(review.draft.scoreTracking.serveMarkers);
  const scoreMarkers = useMemo(
    () => alignServeMarkersToRallyStarts(savedScoreMarkers, clips),
    [savedScoreMarkers, clips],
  );
  const [removedModelMarkerIds, setRemovedModelMarkerIds] = useState(review.draft.scoreTracking.removedModelMarkerIds);
  const [sideSwitchMarkers, setSideSwitchMarkers] = useState<DesignSideSwitchMarker[]>(review.draft.scoreTracking.sideSwitchMarkers);
  const [selectedScoreMarkerId, setSelectedScoreMarkerId] = useState(
    review.draft.scoreTracking.serveMarkers.find((marker) => marker.side === "review")?.id ??
      review.draft.scoreTracking.serveMarkers[0]?.id ??
      "",
  );
  const [selectedSideSwitchId, setSelectedSideSwitchId] = useState("");
  const [scoreMessage, setScoreMessage] = useState(`${review.draft.scoreTracking.serveMarkers.filter((marker) => marker.side === "review").length} serve markers need a court-side check.`);
  const [exportKind, setExportKind] = useState<ExportKind>("video");
  const [localExportProgress, setExportProgress] = useState(0);
  const [localExportSpeed, setExportSpeed] = useState<number | null>(null);
  const [localExportEtaSeconds, setExportEtaSeconds] = useState<number | null>(null);
  const [localExportStatus, setExportStatus] = useState("Choose an output to preview with the current review data.");
  const [includeScore, setIncludeScore] = useState(true);
  const [chapterScore, setChapterScore] = useState(true);
  const [chapterServer, setChapterServer] = useState(true);
  const [projectName, setProjectName] = useState(review.projectName);
  const [storageMessage, setStorageMessage] = useState("Live review data loaded");
  const videoElementRef = useRef<HTMLVideoElement>(null);
  const videoExportJob = review.exportJob;
  const exportProgress =
    exportKind === "video" ? (videoExportJob?.progress ?? 0) : localExportProgress;
  const exportSpeed =
    exportKind === "video" ? (videoExportJob?.speed ?? null) : localExportSpeed;
  const exportEtaSeconds =
    exportKind === "video"
      ? (videoExportJob?.etaSeconds ?? null)
      : localExportEtaSeconds;
  const exportStatus =
    exportKind === "video"
      ? (videoExportJob?.detail ??
        "Choose an output to preview with the current review data.")
      : localExportStatus;
  const exportBusy =
    exportKind === "video" &&
    (videoExportJob?.status === "queued" ||
      videoExportJob?.status === "exporting");
  const showExportProgress =
    exportKind === "video"
      ? videoExportJob !== null
      : localExportProgress > 0;

  function setPlayhead(value: number) {
    setOpenedClipId(null);
    const target = Math.max(gameStart, Math.min(gameEnd, value));
    setPlayheadValue(target);
    setSeekRequest((current) => ({ revision: current.revision + 1, target }));
    const video = videoElementRef.current;
    if (video && video.readyState > 0 && Math.abs(video.currentTime - target) > 0.01) {
      video.currentTime = Math.min(Number.isFinite(video.duration) ? video.duration : gameEnd, target);
    }
  }

  const workingDraft = useMemo<CutDraft>(() => {
    const previousCuts = new Map(baseDraft.cuts.map((cut) => [cut.id, cut]));
    const previousIgnored = new Map(
      baseDraft.ignoredIntervals.map((interval) => [interval.id, interval]),
    );
    const suppressionDecisionOverrides = {
      ...normalizeCleanupDecisions(baseDraft, cleanupSuggestions).suppressionDecisionOverrides,
    };
    for (const [cutId, decision] of Object.entries(suppressionDecisions)) {
      const key = rallySuppressionDecisionKey(cutId);
      if (!decision || decision === "pending") delete suppressionDecisionOverrides[key];
      else suppressionDecisionOverrides[key] = decision === "kept" ? "keep" : "suppress";
    }
    return {
      ...baseDraft,
      beforePaddingSeconds: beforePadding,
      afterPaddingSeconds: afterPadding,
      joinGapSeconds: joinGap,
      pendingManualStart: manualStart,
      pendingIgnoreStart: excludedStart,
      cutPreviewEnabled: finalPreview,
      playbackRate,
      renderScoreOverlay: scoreOverlay,
      selectedSuppressionPolicy: suppressionPolicyForCleanup(cleanup),
      suppressionDecisionOverrides,
      reviewedCutIds: clips.filter((clip) => clip.reviewed && clip.origin === "model").map((clip) => clip.id),
      cuts: clips.map((clip) => {
        const previous = previousCuts.get(clip.id);
        const startChanged = previous
          ? Math.abs(previous.coreStart - clip.start) > 0.000_5
          : true;
        const endChanged = previous
          ? Math.abs(previous.coreEnd - clip.end) > 0.000_5
          : true;
        return {
          id: clip.id,
          coreStart: clip.start,
          coreEnd: clip.end,
          keepStart:
            !startChanged && beforePadding === baseDraft.beforePaddingSeconds
              ? previous!.keepStart
              : Math.max(gameStart, clip.start - beforePadding),
          keepEnd:
            !endChanged && afterPadding === baseDraft.afterPaddingSeconds
              ? previous!.keepEnd
              : Math.min(gameEnd, clip.end + afterPadding),
          confidence: clip.confidence,
          included: clip.included,
          origin: clip.origin === "manual" ? "manual" : "cached-label",
          ...(clip.agreement ? { agreement: clip.agreement } : {}),
        };
      }),
      ignoredIntervals: excludedRanges.map((interval) => ({
        id: interval.id,
        start: interval.start,
        end: interval.end,
        reason:
          previousIgnored.get(interval.id)?.reason ??
          (interval.reason === "camera-break"
            ? "camera-break"
            : interval.reason === "warmup"
              ? "warmup"
              : "non-game-content"),
      })),
      userTouchedCutIds: [
        ...new Set([
          ...baseDraft.userTouchedCutIds.filter((id) => clips.some((clip) => clip.id === id && clip.origin === "model")),
          ...clips.flatMap((clip) => {
            const previous = previousCuts.get(clip.id);
            return previous?.origin === "cached-label" &&
              (previous.included !== clip.included ||
                Math.abs(previous.coreStart - clip.start) > 0.000_5 ||
                Math.abs(previous.coreEnd - clip.end) > 0.000_5)
              ? [clip.id]
              : [];
          }),
        ]),
      ],
      scoreTracking: {
        ...baseDraft.scoreTracking,
        enabled: scoreEnabled,
        team1Name: teamOne.trim() || "Team 1",
        team2Name: teamTwo.trim() || "Team 2",
        serveMarkers: scoreMarkers,
        sideSwitchMarkers,
        removedModelMarkerIds,
      },
    };
  }, [
    afterPadding,
    beforePadding,
    cleanup,
    clips,
    excludedRanges,
    excludedStart,
    finalPreview,
    gameEnd,
    gameStart,
    joinGap,
    manualStart,
    playbackRate,
    cleanupSuggestions,
    baseDraft,
    scoreEnabled,
    scoreMarkers,
    scoreOverlay,
    sideSwitchMarkers,
    removedModelMarkerIds,
    suppressionDecisions,
    teamOne,
    teamTwo,
  ]);

  const [initialHistory] = useState(() => {
    let raw: string | null = null;
    try { raw = window.localStorage.getItem(editHistoryKey(review.projectId)); } catch { /* Keep history in memory. */ }
    return readEditHistory(raw, workingDraft, review.draftSeed,
      (draft) => normalizeCleanupDecisions(draft, cleanupSuggestions));
  });
  const historyRef = useRef(initialHistory);
  const [historyStorageFailed, setHistoryStorageFailed] = useState(false);

  function persistHistory() {
    let saved = false;
    try { saved = writeEditHistory(window.localStorage, editHistoryKey(review.projectId), historyRef.current); } catch { /* Storage may be disabled. */ }
    setHistoryStorageFailed(!saved);
  }

  useEffect(() => {
    const next = recordEdit(historyRef.current, workingDraft);
    if (next !== historyRef.current) {
      historyRef.current = next;
      persistHistory();
    }
  }, [workingDraft]);

  function applyReviewDraft(draft: CutDraft) {
    setBaseDraft(draft);
    setClips(clipsFromReview({ ...review, draft }));
    setExcludedRanges(excludedFromReview({ ...review, draft }));
    setManualStart(draft.pendingManualStart);
    setExcludedStart(draft.pendingIgnoreStart);
    setBeforePadding(draft.beforePaddingSeconds);
    setAfterPadding(draft.afterPaddingSeconds);
    setJoinGap(draft.joinGapSeconds);
    setCleanup(cleanupStrengthForPolicy(draft.selectedSuppressionPolicy));
    setScoreEnabled(draft.scoreTracking.enabled);
    setScoreOverlay(draft.renderScoreOverlay);
    setTeamOne(draft.scoreTracking.team1Name);
    setTeamTwo(draft.scoreTracking.team2Name);
    setScoreMarkers(draft.scoreTracking.serveMarkers);
    setSideSwitchMarkers(draft.scoreTracking.sideSwitchMarkers);
    setRemovedModelMarkerIds(draft.scoreTracking.removedModelMarkerIds);
    setSuppressionDecisions(cleanupReviewDecisions(draft, review.cleanupSuggestions));
  }

  function restoreHistory(direction: "undo" | "redo") {
    const current = recordEdit(historyRef.current, workingDraft);
    const next = moveHistory(current, direction);
    if (next === current) {
      setReviewMessage(`Nothing to ${direction}.`);
      return;
    }
    historyRef.current = next;
    const draft = next.present;
    applyReviewDraft(draft);
    setReviewMessage(direction === "undo" ? "Last edit undone." : "Edit restored.");
    persistHistory();
  }

  function resetProjectChanges() {
    applyReviewDraft(review.modelDraft);
    setPlaying(false);
    setFinalPreview(review.modelDraft.cutPreviewEnabled);
    setPlaybackRate(review.modelDraft.playbackRate);
    setGameStart(review.gameStart);
    setGameEnd(review.gameEnd);
    setServingSide("near");
    setExcludedReason("camera-break");
    setSelectedScoreMarkerId("");
    setSelectedSideSwitchId("");
    setPlayhead(review.gameStart);
    setStage("review");
    setReviewMessage("Project restored to the saved model results. Ctrl+Z undoes the reset.");
  }

  useEffect(() => {
    review.saveDraft(workingDraft);
    setStorageMessage("Saved to the live review on this device");
  }, [review, workingDraft]);

  useEffect(() => {
    setNeedsSource(review.sourceNeedsReconnect);
    setVideoUrl(review.videoUrl);
  }, [review.sourceNeedsReconnect, review.videoUrl]);

  useEffect(() => {
    if (analysisProgress <= 0 || analysisProgress >= 100) return;
    const timer = window.setInterval(() => {
      setAnalysisProgress((current) => Math.min(100, current + 7));
    }, 520);
    return () => window.clearInterval(timer);
  }, [analysisProgress]);

  useEffect(() => {
    if (!playing || videoUrl) return;
    const timer = window.setInterval(() => {
      setPlayheadValue((current) => {
        if (!finalPreview) return current >= gameEnd ? gameStart : current + 1;
        const ranges = clips
          .filter((clip) => clip.included)
          .map((clip) => ({
            start: Math.max(gameStart, clip.start - beforePadding),
            end: Math.min(gameEnd, clip.end + afterPadding),
          }))
          .sort((left, right) => left.start - right.start);
        if (ranges.length === 0) return gameStart;
        const next = current + 1;
        const excluded = excludedRanges.find((range) => next >= range.start && next < range.end);
        if (excluded) return excluded.end;
        const active = ranges.find((range) => current >= range.start && current < range.end);
        if (active && next <= active.end) return next;
        return ranges.find((range) => range.start > current)?.start ?? ranges[0].start;
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [afterPadding, beforePadding, clips, excludedRanges, finalPreview, gameEnd, gameStart, playing, videoUrl]);

  useEffect(() => {
    setExportProgress(0);
    setExportSpeed(null);
    setExportEtaSeconds(null);
  }, [exportKind, includeScore, workingDraft]);

  const included = clips.filter((clip) => clip.included);
  const finalMaterialization = useMemo(
    () => materializeFinalCutIntervals(workingDraft, review.suppression),
    [review.suppression, workingDraft],
  );
  const finalIntervals = finalMaterialization.intervals;
  const effectiveKeptIds = useMemo(
    () => new Set(finalIntervals.flatMap((interval) => interval.cutIds)),
    [finalIntervals],
  );
  const reviewClipIds = useMemo(
    () => new Set(clips
      .filter((clip) =>
        clip.included &&
        !clip.reviewed &&
        effectiveKeptIds.has(clip.id) &&
        rangeHasReviewableFootage(
          clip.start,
          clip.end,
          workingDraft.ignoredIntervals,
        ) &&
        clipRequiresConfidenceReview(clip, workingDraft.confidenceReviewThreshold),
      )
      .map((clip) => clip.id)),
    [clips, effectiveKeptIds, workingDraft.confidenceReviewThreshold, workingDraft.ignoredIntervals],
  );
  const remaining = reviewClipIds.size;
  const suppressionReviewIds = useMemo(() => {
    const suggestionsByCut = new Map<string, typeof cleanupSuggestions>();
    for (const suggestion of cleanupSuggestions) {
      if (!suggestion.cutId) continue;
      suggestionsByCut.set(suggestion.cutId, [
        ...(suggestionsByCut.get(suggestion.cutId) ?? []),
        suggestion,
      ]);
    }
    return new Set(
      Object.entries(suppressionDecisions)
        .filter(([, decision]) => decision === "pending")
        .filter(([id]) => {
          const suggestions = suggestionsByCut.get(id) ?? [];
          if (suggestions.length > 0) {
            return suggestions.some((suggestion) =>
              rangeHasReviewableFootage(
                suggestion.start,
                suggestion.end,
                workingDraft.ignoredIntervals,
              ),
            );
          }
          const clip = clips.find((candidate) => candidate.id === id);
          return clip
            ? rangeHasReviewableFootage(
                clip.start,
                clip.end,
                workingDraft.ignoredIntervals,
              )
            : false;
        })
        .map(([id]) => id),
    );
  }, [clips, cleanupSuggestions, suppressionDecisions, workingDraft.ignoredIntervals]);
  const suppressionPending = suppressionReviewIds.size;
  const clipReviewTaskIds = useMemo(
    () => new Set([
      ...reviewClipIds,
      ...suppressionReviewIds,
    ]),
    [reviewClipIds, suppressionReviewIds],
  );
  const excludedRallyIds = useMemo(
    () => new Set(workingDraft.cuts
      .filter((cut) => !effectiveKeptIds.has(cut.id))
      .map((cut) => cut.id)),
    [effectiveKeptIds, workingDraft.cuts],
  );
  const activeScoreTracking = useMemo<ScoreTracking>(
    () => scoreTrackingOutsideExcludedRallies(
      scoreTrackingOutsideIgnoredIntervals(
        workingDraft.scoreTracking,
        workingDraft.ignoredIntervals,
      ),
      excludedRallyIds,
    ),
    [excludedRallyIds, workingDraft.ignoredIntervals, workingDraft.scoreTracking],
  );
  const scoreOverlayOptions = useMemo(
    () => ({
      scoreTracking: workingDraft.scoreTracking,
      renderPointTimeline: true,
      excludedRallyIds: [...excludedRallyIds],
      ignoredIntervals: workingDraft.ignoredIntervals,
      rallyRanges: workingDraft.cuts.filter((cut) => effectiveKeptIds.has(cut.id)),
      mergedRanges: finalIntervals,
    }),
    [excludedRallyIds, effectiveKeptIds, finalIntervals, workingDraft],
  );
  const preparedScoreOverlay = useMemo(
    () => prepareScoreOverlay(scoreOverlayOptions),
    [scoreOverlayOptions],
  );
  const activeScoreMarkers = activeScoreTracking.serveMarkers;
  const activeSideSwitchMarkers = activeScoreTracking.sideSwitchMarkers;
  const openedCut = workingDraft.cuts.find((cut) => cut.id === openedClipId);
  const withinOpenedCut = openedCut != null && playhead >= openedCut.keepStart && playhead < openedCut.keepEnd;
  useEffect(() => {
    if (!withinOpenedCut) setOpenedClipId(null);
  }, [withinOpenedCut]);
  const currentPlayingCut = useMemo(
    () => {
      // Opening a rally seeks to its padding, which can overlap another rally's core.
      // Keep edits directed at the opened rally until playback leaves it or the user seeks.
      if (withinOpenedCut) return openedCut;
      const ordered = [...workingDraft.cuts].sort(
        (left, right) => left.keepStart - right.keepStart || left.keepEnd - right.keepEnd,
      );
      const containing = ordered
        .filter((cut) => playhead >= cut.keepStart && playhead < cut.keepEnd)
        .sort((left, right) => {
        const leftCore = playhead >= left.coreStart && playhead < left.coreEnd;
        const rightCore = playhead >= right.coreStart && playhead < right.coreEnd;
        if (leftCore !== rightCore) return leftCore ? -1 : 1;
        return Math.abs(left.coreStart - playhead) - Math.abs(right.coreStart - playhead);
        })[0];
      if (containing) return containing;
      return [...ordered]
        .filter((cut) => cut.keepEnd <= playhead)
        .sort((left, right) => right.keepEnd - left.keepEnd)[0] ?? ordered[0] ?? null;
    },
    [playhead, workingDraft.cuts, withinOpenedCut, openedCut],
  );
  const currentServeMarker = useMemo(
    () => {
      const rallyMarker = currentPlayingCut
        ? activeScoreMarkers.find((marker) => marker.rallyId === currentPlayingCut.id)
        : null;
      if (rallyMarker) return rallyMarker;
      return [...activeScoreMarkers]
        .filter((marker) => marker.timestamp <= playhead)
        .sort((left, right) => right.timestamp - left.timestamp)[0] ?? null;
    },
    [activeScoreMarkers, currentPlayingCut, playhead],
  );
  const currentSideSwitchMarker = useMemo(
    () => activeSideSwitchMarkers.find(
      (marker) => Math.abs(marker.timestamp - playhead) < 0.05,
    ) ?? null,
    [activeSideSwitchMarkers, playhead],
  );
  const selected = clips.find((clip) =>
    clip.id === (followPlayhead ? currentPlayingCut?.id : selectedId),
  ) ?? (followPlayhead ? undefined : clips[0]);
  const serveReviewMarkers = activeScoreMarkers.filter(
    (marker) => marker.side === "review",
  );
  const serveReview = serveReviewMarkers.length;
  const selectedScoreMarker = followPlayhead
    ? currentSideSwitchMarker ? null : currentServeMarker
    : scoreMarkers.find((marker) => marker.id === selectedScoreMarkerId) ?? null;
  const selectedSideSwitch = followPlayhead
    ? currentSideSwitchMarker
    : sideSwitchMarkers.find((marker) => marker.id === selectedSideSwitchId) ?? null;
  const selectedServeReview = selectedScoreMarker
    ? serveReviewMarkers.find((marker) => marker.id === selectedScoreMarker.id) ?? null
    : null;
  const nextServeReview = selectedServeReview ?? [...serveReviewMarkers]
    .sort((left, right) => left.timestamp - right.timestamp)
    .find((marker) => marker.timestamp >= playhead) ??
    [...serveReviewMarkers]
      .sort((left, right) => left.timestamp - right.timestamp)
      .at(0) ?? null;
  const derivedScore = useMemo(
    () => deriveScoreAt(activeScoreTracking, playhead),
    [activeScoreTracking, playhead],
  );
  const keptSeconds = finalIntervals.reduce(
    (total, interval) => total + Math.max(0, interval.end - interval.start),
    0,
  );
  const chapterOptions = useMemo(
    () => ({
      ...defaultYouTubeChapterOptions(scoreEnabled, activeSideSwitchMarkers.length > 0),
      includeScore: scoreEnabled && chapterScore,
      includeServingTeam: scoreEnabled && chapterServer,
      includeSideSwitches: scoreEnabled && activeSideSwitchMarkers.length > 0,
    }),
    [activeSideSwitchMarkers.length, chapterScore, chapterServer, scoreEnabled],
  );
  const chapters = useMemo(
    () => buildYouTubeChapters({
      intervals: finalIntervals,
      cuts: workingDraft.cuts.filter((cut) => effectiveKeptIds.has(cut.id)),
      scoreTracking: scoreEnabled ? activeScoreTracking : null,
      options: chapterOptions,
    }),
    [activeScoreTracking, chapterOptions, effectiveKeptIds, finalIntervals, scoreEnabled, workingDraft.cuts],
  );
  const chapterText = useMemo(() => youtubeChaptersText(chapters), [chapters]);

  async function chooseVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file) return;
    if (!/\.(mp4|mov|m4v|webm|mkv)$/i.test(file.name)) {
      setSourceStatus("That file does not look like a supported video. Choose MP4, MOV, WebM, or MKV.");
      return;
    }
    setSourceStatus("Checking that video against the saved project…");
    const result = await review.attachSource(file);
    setSourceStatus(result.message);
    if (!result.ok || !result.videoUrl) return;
    setFileName(file.name);
    setNeedsSource(false);
    setExpectedSourceName(file.name);
    setVideoUrl(result.videoUrl);
    setStorageMessage("Source linked for playback; review edits remain saved");
  }

  function importProject(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".json")) {
      setSourceStatus("Choose a saved VolleySplice project JSON file.");
      return;
    }
    const importedExpectedSource = review.sourceName;
    setProjectName(file.name.replace(/\.json$/i, ""));
    setFileName("Original video needs reconnection");
    setExpectedSourceName(importedExpectedSource);
    setNeedsSource(true);
    setSourceStatus(`Saved corrections opened. Reconnect ${importedExpectedSource} for playback or MP4 export.`);
    setStorageMessage("Imported review is ready");
    setStage("review");
  }

  function beginAnalysis() {
    if (gameEnd - gameStart < 10) {
      setWindowError("Choose at least 10 seconds of game footage before analysis.");
      return;
    }
    setAnalysisProgress(8);
    setStage("analysis");
    setStorageMessage("Analysis is running in this browser");
  }

  function updateGameStart(requested: number) {
    if (!Number.isFinite(requested) || requested < 0 || requested > gameEnd - 10) {
      setWindowError("Game start must be at least 10 seconds before game end.");
      return;
    }
    setGameStart(requested);
    setWindowError(null);
  }

  function updateGameEnd(requested: number) {
    if (!Number.isFinite(requested) || requested < gameStart + 10 || requested > review.duration) {
      setWindowError("Game end must be at least 10 seconds after game start and within the source duration.");
      return;
    }
    setGameEnd(requested);
    setWindowError(null);
  }

  function resumeProject() {
    setFileName(review.sourceName);
    setNeedsSource(review.sourceNeedsReconnect);
    setExpectedSourceName(review.sourceName);
    setProjectName(review.projectName);
    setSourceStatus("Saved analysis and corrections restored from the live review.");
    setStorageMessage("Live review resumed");
    setStage("review");
  }

  function updateSelected(patch: Partial<Clip>) {
    if (!selected) return;
    setClips((current) => current.map((clip) => clip.id === selected.id ? { ...clip, ...patch } : clip));
    setStorageMessage("Saving to the live review…");
  }

  function setCurrentRallyIncluded(included: boolean) {
    if (!selected) return;
    updateSelected({ included, reviewed: true });
    setSuppressionDecisions((current) => cleanupDecisionsForRally(
      current, selected.id, included ? "kept" : "excluded",
    ));
    setReviewMessage(`${selected.id} ${included ? "kept in" : "removed from"} the final cut.`);
  }

  function selectClip(id: string) {
    const clip = clips.find((candidate) => candidate.id === id);
    if (!clip) return;
    const draftCut = workingDraft.cuts.find((candidate) => candidate.id === id);
    const target = draftCut?.keepStart ?? Math.max(gameStart, clip.start - beforePadding);
    setSelectedId(id);
    setPlayhead(target);
    setOpenedClipId(id);
    setReviewMessage(`${clip.id} opened at its ${formatTime(target)} padded start.`);
  }

  function reviewNext() {
    if (selected && reviewClipIds.has(selected.id)) updateSelected({ reviewed: true });
    const next = clips.find((clip) => reviewClipIds.has(clip.id) && clip.id !== selected?.id);
    if (next) {
      selectClip(next.id);
      setReviewMessage(`${selected?.id ?? "Clip"} checked. ${next.id} is next.`);
    } else {
      setReviewMessage("Every included clip has been checked. Export is ready when you are.");
    }
  }

  function openClipReview() {
    const candidates = clips.filter((clip) => reviewClipIds.has(clip.id));
    const next = candidates.find((clip) => clip.start >= playhead) ?? candidates[0];
    if (next) {
      selectClip(next.id);
      setReviewMessage(`${next.id} needs a confidence review.`);
    }
  }

  function openServeReview() {
    if (nextServeReview) selectScoreMarker(nextServeReview.id);
  }

  function splitSelectedAtPlayhead() {
    if (!selected || playhead <= selected.start + 0.2 || playhead >= selected.end - 0.2) {
      setReviewMessage("Move the playhead inside the selected clip before splitting it.");
      return;
    }
    const used = new Set(clips.map((clip) => clip.id));
    let leftId = `${selected.id}A`;
    let rightId = `${selected.id}B`;
    while (used.has(leftId) || used.has(rightId)) {
      leftId += "A";
      rightId += "B";
    }
    const left: Clip = { ...selected, id: leftId, end: playhead, label: `${selected.label}, first part`, reviewed: true };
    const right: Clip = { ...selected, id: rightId, start: playhead, label: `${selected.label}, second part`, reviewed: true };
    setClips((current) => current.flatMap((clip) => clip.id === selected.id ? [left, right] : [clip]));
    const suppression = suppressionDecisions[selected.id];
    if (suppression) {
      setSuppressionDecisions((current) => {
        const { [selected.id]: _removed, ...rest } = current;
        return { ...rest, [leftId]: suppression, [rightId]: suppression };
      });
    }
    setSelectedId(rightId);
    setReviewMessage(`${selected.id} split at ${formatTime(playhead)}. Both new clips remain in the final-cut review.`);
  }

  function markManualBoundary() {
    if (manualStart === null) {
      setExcludedStart(null);
      setManualStart(playhead);
      setReviewMessage(`Missed rally starts at ${formatTime(playhead)}. Move the playhead, then mark its end.`);
      return;
    }
    const start = Math.min(manualStart, playhead);
    const end = Math.max(manualStart, playhead);
    if (end - start < 0.3) {
      setReviewMessage("Move the playhead at least 0.3 seconds before finishing the missed rally.");
      return;
    }
    const id = `M${String(clips.filter((clip) => clip.origin === "manual").length + 1).padStart(2, "0")}`;
    const next: Clip = {
      id,
      label: "Missed rally added by you",
      start,
      end,
      confidence: 1,
      included: true,
      reviewed: true,
      origin: "manual",
    };
    setClips((current) => [...current, next]);
    setManualStart(null);
    setSelectedId(id);
    setReviewMessage(`${id} added from ${formatTime(start)} to ${formatTime(end)}. Adjust either edge if needed.`);
  }

  function cancelManualBoundary() {
    setManualStart(null);
    setReviewMessage("Missed-rally marking canceled.");
  }

  function markExcludedBoundary(reason: ExcludedRange["reason"] = excludedReason) {
    if (excludedStart === null) {
      setManualStart(null);
      setExcludedStart(playhead);
      setReviewMessage(`Excluded section starts at ${formatTime(playhead)}. Move the playhead, then mark its end.`);
      return;
    }
    const start = Math.min(excludedStart, playhead);
    const end = Math.max(excludedStart, playhead);
    if (end - start < 0.3) {
      setReviewMessage("Move the playhead at least 0.3 seconds before finishing the excluded section.");
      return;
    }
    const id = `I${String(excludedRanges.length + 1).padStart(2, "0")}`;
    setExcludedRanges((current) => [...current, { id, start, end, reason }]);
    setExcludedStart(null);
    setReviewMessage(`${id} excludes ${formatTime(start)} to ${formatTime(end)} from review, score, and export.`);
  }

  function cancelExcludedBoundary() {
    setExcludedStart(null);
    setReviewMessage("Excluded-section marking canceled.");
  }

  function removeExcludedRange(id: string) {
    setExcludedRanges((current) => current.filter((range) => range.id !== id));
    setReviewMessage(`${id} restored to the project timeline.`);
  }

  function decideSuppression(keep: boolean) {
    if (!selected || !suppressionDecisions[selected.id]) return;
    setSuppressionDecisions((current) => cleanupDecisionsForRally(
      current, selected.id, keep ? "kept" : "excluded",
    ));
    updateSelected({ included: keep, reviewed: true });
    setReviewMessage(
      keep
        ? `${selected.id} kept after reviewing the automatic cleanup suggestion.`
        : `${selected.id} confirmed as excluded footage. You can restore it later.`,
    );
  }

  function reviewNextSuppression() {
    const pendingIds = [...suppressionReviewIds];
    if (pendingIds.length === 0) {
      setReviewMessage("Every automatic cleanup suggestion has been reviewed.");
      return;
    }
    const currentIndex = selected ? pendingIds.indexOf(selected.id) : -1;
    const nextId = pendingIds[(currentIndex + 1) % pendingIds.length];
    setFinalPreview(false);
    selectClip(nextId);
    setReviewMessage(`${nextId} was excluded by automatic cleanup. Watch it, then keep it or confirm the exclusion.`);
  }

  function applySuppressionToSimilar() {
    const pendingIds = new Set(
      Object.entries(suppressionDecisions)
        .filter(([, decision]) => decision === "pending")
        .map(([id]) => id),
    );
    if (pendingIds.size === 0) return;
    setSuppressionDecisions((current) => Object.fromEntries(
      Object.entries(current).map(([id, decision]) => [
        id,
        decision === "pending" ? "excluded" : decision,
      ]),
    ));
    setClips((current) => current.map((clip) =>
      pendingIds.has(clip.id) ? { ...clip, included: false, reviewed: true } : clip,
    ));
    setReviewMessage(`${pendingIds.size} similar cleanup ${pendingIds.size === 1 ? "suggestion was" : "suggestions were"} confirmed as excluded.`);
  }

  function chooseServingSide(side: Side) {
    setServingSide(side);
    const target = selectedScoreMarker ?? nextServeReview;
    if (!target) {
      setScoreMessage(`${side === "near" ? "Near" : "Far"} is selected for the next manual serve marker.`);
      return;
    }
    setScoreMarkers((current) => current.map((marker) =>
      marker.id === target.id ? { ...marker, side } : marker,
    ));
    setSelectedScoreMarkerId(target.id);
    setScoreMessage(
      `${formatTime(target.timestamp)} corrected to the ${side} court. The derived score and chapters updated.`,
    );
  }

  function updateScoreMarkerSide(id: string, side: MarkerSide) {
    const marker = scoreMarkers.find((candidate) => candidate.id === id);
    if (!marker) return;
    setScoreMarkers((current) => current.map((candidate) =>
      candidate.id === id ? { ...candidate, side } : candidate,
    ));
    setSelectedScoreMarkerId(id);
    setSelectedSideSwitchId("");
    setScoreMessage(
      side === "review"
        ? `${formatTime(marker.timestamp)} now needs a court-side review.`
        : `${formatTime(marker.timestamp)} set to the ${side} court. Scores and chapters were recalculated.`,
    );
  }

  function setScoreMarkerReplay(id: string, ignored: boolean) {
    const marker = scoreMarkers.find((candidate) => candidate.id === id);
    if (!marker) return;
    setScoreMarkers((current) => current.map((candidate) =>
      candidate.id === id ? { ...candidate, ignorePreviousPoint: ignored } : candidate,
    ));
    setScoreMessage(
      ignored
        ? `The rally before ${formatTime(marker.timestamp)} is marked as a replay.`
        : `The rally before ${formatTime(marker.timestamp)} counts again.`,
    );
  }

  function selectScoreMarker(id: string) {
    const marker = scoreMarkers.find((candidate) => candidate.id === id);
    if (!marker) return;
    setSelectedScoreMarkerId(id);
    setSelectedSideSwitchId("");
    setPlayhead(marker.timestamp);
    setScoreMessage(`Serve ${marker.id} selected at ${formatPreciseTime(marker.timestamp)}.`);
  }

  function selectSideSwitch(id: string) {
    const marker = sideSwitchMarkers.find((candidate) => candidate.id === id);
    if (!marker) return;
    setSelectedSideSwitchId(id);
    setSelectedScoreMarkerId("");
    setPlayhead(marker.timestamp);
    setScoreMessage(`Side switch ${marker.id} selected at ${formatPreciseTime(marker.timestamp)}.`);
  }

  function addServeAtPlayhead() {
    let sequence = 1;
    while (scoreMarkers.some((marker) => marker.id === `manual-serve-${sequence}`)) sequence += 1;
    const id = `manual-serve-${sequence}`;
    setScoreMarkers((current) => [
      ...current,
      { id, timestamp: playhead, side: servingSide, ignorePreviousPoint: false, origin: "manual" },
    ]);
    setSelectedScoreMarkerId(id);
    setSelectedSideSwitchId("");
    setScoreMessage(`Serve marker added at ${formatTime(playhead)} on the ${servingSide} court. The previous rally was recalculated.`);
  }

  function addSideSwitchAtPlayhead() {
    let sequence = 1;
    while (sideSwitchMarkers.some((marker) => marker.id === `manual-switch-${sequence}`)) sequence += 1;
    const id = `manual-switch-${sequence}`;
    setSideSwitchMarkers((current) => [...current, { id, timestamp: playhead, origin: "manual" }]);
    setSelectedSideSwitchId(id);
    setSelectedScoreMarkerId("");
    setScoreMessage(`Team side switch added at ${formatTime(playhead)}. Later serve markers were remapped to teams.`);
  }

  function removeLastScoreMarker() {
    const manual = [...scoreMarkers].reverse().find((marker) => marker.origin === "manual");
    if (!manual) {
      setScoreMessage("No manual serve marker is available to remove.");
      return;
    }
    setScoreMarkers((current) => current.filter((marker) => marker.id !== manual.id));
    setScoreMessage(`Manual serve at ${formatTime(manual.timestamp)} removed. The score was derived again.`);
  }

  function toggleLastPointReplay() {
    const ordered = [...scoreMarkers].sort(
      (left, right) => left.timestamp - right.timestamp || left.id.localeCompare(right.id),
    );
    const target = selectedScoreMarker ?? [...ordered].reverse().find((marker) => marker.side !== "review");
    if (!target) return;
    setScoreMarkers((current) => current.map((marker) =>
      marker.id === target.id
        ? { ...marker, ignorePreviousPoint: !marker.ignorePreviousPoint }
        : marker,
    ));
    setScoreMessage(
      target.ignorePreviousPoint
        ? "The previous rally counts again in the derived score."
        : "The previous rally is marked as a replay and no point is awarded at that serve.",
    );
  }

  function removeSelectedEvent() {
    if (selectedScoreMarker) {
      const next = removeServeMarker(workingDraft.scoreTracking, selectedScoreMarker.id);
      setScoreMarkers(next.serveMarkers);
      setRemovedModelMarkerIds(next.removedModelMarkerIds);
      setSelectedScoreMarkerId("");
      setScoreMessage(`Serve at ${formatTime(selectedScoreMarker.timestamp)} removed.`);
      return;
    }
    if (selectedSideSwitch) {
      const next = removeSideSwitchMarker(workingDraft.scoreTracking, selectedSideSwitch.id);
      setSideSwitchMarkers(next.sideSwitchMarkers);
      setRemovedModelMarkerIds(next.removedModelMarkerIds);
      setSelectedSideSwitchId("");
      setScoreMessage(`Side switch at ${formatTime(selectedSideSwitch.timestamp)} removed.`);
    }
  }

  function downloadChapters() {
    if (chapters.length === 0) {
      setExportStatus("No retained rally is available for a chapter export.");
      return;
    }
    const blob = new Blob([`${chapterText}\n`], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = youtubeChaptersFilename(fileName || review.sourceName);
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    setExportProgress(100);
    setExportStatus(`${chapters.length} YouTube ${chapters.length === 1 ? "chapter" : "chapters"} downloaded.`);
  }

  async function copyChapters() {
    if (chapters.length === 0 || !chapterText.trim()) {
      setExportStatus("No retained rally is available to copy as chapters.");
      return;
    }
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(chapterText);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = chapterText;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.append(textarea);
        textarea.select();
        const copied = document.execCommand("copy");
        textarea.remove();
        if (!copied) throw new Error("Clipboard access is unavailable");
      }
      setExportProgress(100);
      setExportStatus(`${chapters.length} YouTube ${chapters.length === 1 ? "chapter" : "chapters"} copied to the clipboard.`);
    } catch (cause) {
      setExportStatus(
        `Could not copy the chapters: ${cause instanceof Error ? cause.message : String(cause)}`,
      );
    }
  }

  function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  async function prepareExport() {
    if (exportKind === "video" && needsSource) {
      setExportStatus(`Reconnect the matching ${expectedSourceName} source before preparing an MP4.`);
      return;
    }
    if (exportKind === "chapters") {
      downloadChapters();
      return;
    }
    if (exportKind === "feedback") {
      if (!review.productAnalysis) {
        setExportStatus("This project cannot be saved in the current session.");
        return;
      }
      const bundle = createModelFeedbackBundle(
        review.productAnalysis,
        workingDraft,
        finalIntervals,
      );
      downloadBlob(
        modelFeedbackBlob(bundle),
        modelFeedbackFilename(review.sourceName),
      );
      setExportProgress(100);
      setExportStatus("The current VolleySplice project was downloaded.");
      return;
    }
    if (!review.sourceFile || finalIntervals.length === 0) return;
    if (exportBusy) return;
    review.queueVideoExport({
      intervals: finalIntervals,
      scoreOverlay: scoreEnabled && includeScore
        ? scoreOverlayOptions
        : undefined,
    });
  }

  return {
    stage, setStage, dark, setDark, helpOpen, setHelpOpen,
    fileName, expectedSourceName, needsSource, sourceStatus, gameStart, gameEnd,
    videoUrl,
    windowError, updateGameStart, updateGameEnd,
    cropCourt, setCropCourt, detectSwitches, setDetectSwitches,
    analysisProgress, setAnalysisProgress, clips, selected,
    selectedId: followPlayhead ? currentPlayingCut?.id ?? "" : selectedId,
    manualStart,
    excludedStart, excludedReason, setExcludedReason, excludedRanges,
    suppressionDecisions, suppressionPending, suppressionReviewIds,
    playhead, setPlayhead, trackPlayhead: setPlayheadValue, videoElementRef,
    seekRequest: seekRequest.revision, seekTarget: seekRequest.target,
    playing, setPlaying, finalPreview, setFinalPreview,
    playbackRate, setPlaybackRate,
    beforePadding, setBeforePadding,
    afterPadding, setAfterPadding, joinGap, setJoinGap, cleanup, setCleanup,
    restoreHistory, historyStorageFailed, resetProjectChanges,
    reviewMessage, scoreEnabled, setScoreEnabled, scoreOverlay, setScoreOverlay,
    teamOne, setTeamOne, teamTwo, setTeamTwo, scoreOne: derivedScore.team1Score,
    scoreTwo: derivedScore.team2Score, servingTeam: derivedScore.servingTeamId,
    servingSide, scoreMarkers, activeScoreMarkers, sideSwitchMarkers,
    activeSideSwitchMarkers, serveReview, nextServeReview, selectedScoreMarker,
    selectedScoreMarkerId: followPlayhead ? selectedScoreMarker?.id ?? "" : selectedScoreMarkerId,
    selectedSideSwitch,
    selectedSideSwitchId: followPlayhead ? selectedSideSwitch?.id ?? "" : selectedSideSwitchId,
    activeScoreTracking,
    scoreMessage, exportKind, setExportKind, exportProgress,
    setExportProgress, exportSpeed, exportEtaSeconds, exportStatus,
    exportBusy, showExportProgress, includeScore, setIncludeScore, chapterScore,
    setChapterScore, chapterServer, setChapterServer, chapters, chapterText,
    finalIntervals, workingDraft, effectiveKeptIds, reviewClipIds,
    clipReviewTaskIds, preparedScoreOverlay,
    currentPlayingClipId: currentPlayingCut?.id ?? null,
    currentServeMarkerId: currentServeMarker?.id ?? null,
    cleanupSuggestions, projectName, storageMessage,
    projects: review.projects, selectedProjectId: review.projectId,
    workActivity: review.workActivity,
    selectProject: review.selectProject,
    duration: review.duration, width: review.width, height: review.height,
    included, remaining, keptSeconds, chooseVideo, importProject, beginAnalysis,
    resumeProject, updateSelected, setCurrentRallyIncluded, selectClip, reviewNext, openClipReview,
    openServeReview, splitSelectedAtPlayhead,
    markManualBoundary, cancelManualBoundary, markExcludedBoundary,
    cancelExcludedBoundary, removeExcludedRange, decideSuppression,
    reviewNextSuppression, applySuppressionToSimilar,
    chooseServingSide, updateScoreMarkerSide, setScoreMarkerReplay,
    selectScoreMarker, selectSideSwitch,
    addServeAtPlayhead, addSideSwitchAtPlayhead,
    removeLastScoreMarker, toggleLastPointReplay, removeSelectedEvent,
    downloadChapters, copyChapters, prepareExport,
  };
}

type Prototype = ReturnType<typeof usePrototype>;

function Brand({ compact = false, logo = false }: { compact?: boolean; logo?: boolean }) {
  return (
    <a className="td-brand" href="/" aria-label="VolleySplice home">
      {logo ? (
        <>
          <img src={runtimeAssetUrl("volleysplice-icon-transparent.png")} alt="" />
          <strong className="td-wordmark">volley<span>splice</span></strong>
        </>
      ) : (
        <>
          <span aria-hidden="true">V/S</span>
          {!compact && <strong>VolleySplice</strong>}
        </>
      )}
    </a>
  );
}

function UtilityLinks({ state, quiet = false }: { state: Prototype; quiet?: boolean }) {
  return (
    <nav className={`td-utility ${quiet ? "is-quiet" : ""}`} aria-label="Product links and downloads">
      <a href="https://www.volleysplice.com/privacy.html">Privacy</a>
      <a href="https://www.volleysplice.com/terms.html">Terms</a>
      <a href={GOOGLE_PLAY_URL}>Get it on Google Play</a>
      <button type="button" onClick={() => state.setDark(!state.dark)} aria-pressed={state.dark}>
        {state.dark ? "Light theme" : "Dark theme"}
      </button>
    </nav>
  );
}

const STAGES: { id: Stage; label: string; hint: string }[] = [
  { id: "source", label: "Set up", hint: "Choose the game" },
  { id: "analysis", label: "Analyze", hint: "Find the action" },
  { id: "review", label: "Review", hint: "Correct the cut" },
  { id: "deliver", label: "Export", hint: "Save and share" },
];

function StageButtons({
  state,
  className = "",
  editorOnly = false,
}: {
  state: Prototype;
  className?: string;
  editorOnly?: boolean;
}) {
  const stages = editorOnly
    ? STAGES.filter((item) => item.id === "review" || item.id === "deliver")
    : STAGES;
  return (
    <nav className={`td-stages ${className}`} data-editor-only={editorOnly || undefined} aria-label="VolleySplice workflow">
      {stages.map((item) => (
        <button
          key={item.id}
          type="button"
          data-active={state.stage === item.id || undefined}
          onClick={() => state.setStage(item.id)}
          aria-current={state.stage === item.id ? "step" : undefined}
        >
          <strong>{item.label}</strong>
          <small>{item.hint}</small>
        </button>
      ))}
    </nav>
  );
}

function ProjectChip({ state }: { state: Prototype }) {
  return (
    <button className="td-project-chip" type="button" onClick={state.resumeProject}>
      <span>{state.projectName}</span>
      <small>{state.storageMessage}. Compatible edits reopen in the live editor.</small>
    </button>
  );
}

function RallyDeskProjectSelector({
  state,
  className,
  onDeleteProject,
}: {
  state: Prototype;
  className: string;
  onDeleteProject?: () => void;
}) {
  const projectStatusLabel = (
    project: ReadyDesignReview["projects"][number],
  ) => {
    if (project.status === "ready" && project.exportJob) {
      switch (project.exportJob.status) {
        case "queued": return "Export queued";
        case "exporting": return `Exporting ${project.exportJob.progress}%`;
        case "saved": return "Export saved";
        case "error": return "Export stopped";
      }
    }
    switch (project.status) {
      case "ready": return "Ready";
      case "analyzing": return "Analyzing";
      case "queued": return "Queued";
      case "waiting": return "Needs video";
      case "error": return "Stopped";
    }
  };
  return (
    <div className={className} data-tour="rd-project-selector">
      <label>
        <span>Current review</span>
        <select
          aria-label="Current review project"
          value={state.selectedProjectId}
          onChange={(event) => {
            if (event.currentTarget.value === "__new__") {
              state.selectProject(null);
              return;
            }
            state.selectProject(event.currentTarget.value);
          }}
        >
          <option value="__new__">＋ Start a new project…</option>
          {state.projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.name} · {projectStatusLabel(project)}
            </option>
          ))}
        </select>
        <small>{state.fileName}</small>
      </label>
      {state.workActivity.length > 0 && (
        <details className="rd-analysis-activity">
          <summary
            aria-label={`${state.workActivity.length} active project ${state.workActivity.length === 1 ? "job" : "jobs"}`}
          >
            <i aria-hidden="true" />
          </summary>
          <div className="rd-analysis-popover" role="status">
            <header>
              <strong>Project activity</strong>
              <span>{state.workActivity.length} active</span>
            </header>
            {state.workActivity.map((activity) => (
              <article key={`${activity.kind}-${activity.projectId}`}>
                <div>
                  <strong>{activity.name}</strong>
                  <small>{activity.detail}</small>
                </div>
                <em>
                  {activity.progress === null
                    ? "Queued"
                    : `${activity.progress}%`}
                </em>
                <span aria-hidden="true">
                  <i
                    style={{
                      width: `${activity.progress ?? 0}%`,
                    }}
                  />
                </span>
              </article>
            ))}
          </div>
        </details>
      )}
      {onDeleteProject && (
        <button className="rd-delete-project" type="button" onClick={onDeleteProject}>
          Delete
        </button>
      )}
    </div>
  );
}

function ReconnectNotice({ state }: { state: Prototype }) {
  if (!state.needsSource) return null;
  return (
    <aside className="td-reconnect" role="status">
      <div>
        <strong>Reconnect the original video</strong>
        <span>
          Choose {state.expectedSourceName}. The live app verifies the saved fingerprint,
          size, and duration, so a different video remains disconnected. Other project exports stay available.
        </span>
        <small>{state.sourceStatus}</small>
      </div>
      <label className="td-secondary-button">
        Choose matching source
        <input type="file" accept="video/*,.mkv,.webm,.mov,.mp4,.m4v" onChange={state.chooseVideo} />
      </label>
    </aside>
  );
}

function SourceSetup({ state, intro }: { state: Prototype; intro?: ReactNode }) {
  return (
    <section className="td-source-panel" aria-labelledby="td-source-title">
      <header className="td-section-intro">
        <p className="td-kicker">{state.fileName ? "Local project setup" : "Bump. Set. Splice."}</p>
        <h1 id="td-source-title">Choose the part worth watching.</h1>
        <p>{intro ?? "Open a game video, trim unused setup time, and let VolleySplice analyze only what matters."}</p>
      </header>

      <div className="td-source-grid">
        <div className="td-file-zone">
          <p className="td-field-label">Game video</p>
          <strong>{state.fileName || "No video chosen"}</strong>
          <span>{state.sourceStatus}</span>
          <div className="td-inline-actions">
            <label className="td-primary-button">
              Choose video
              <input type="file" accept="video/*,.mkv,.webm,.mov,.mp4,.m4v" onChange={state.chooseVideo} />
            </label>
            <label className="td-secondary-button">
              Import project
              <input type="file" accept="application/json,.json" onChange={state.importProject} />
            </label>
          </div>
          <p className="td-privacy-line">The source stays on this device. Saved projects never contain video bytes.</p>
        </div>

        <div className="td-source-options">
          <fieldset>
            <legend>Game window</legend>
            <p>Leave the full range when the recording contains only the match.</p>
            <div className="td-two-fields">
              <label>
                <span>Game starts</span>
                <input aria-invalid={Boolean(state.windowError)} aria-describedby="td-game-window-error" type="number" min="0" max={state.gameEnd - 10} value={state.gameStart} onChange={(event) => state.updateGameStart(event.currentTarget.valueAsNumber)} />
                <small>{formatTime(state.gameStart)}</small>
              </label>
              <label>
                <span>Game ends</span>
                <input aria-invalid={Boolean(state.windowError)} aria-describedby="td-game-window-error" type="number" min={state.gameStart + 10} max={state.duration} value={state.gameEnd} onChange={(event) => state.updateGameEnd(event.currentTarget.valueAsNumber)} />
                <small>{formatTime(state.gameEnd)}</small>
              </label>
            </div>
            <p id="td-game-window-error" className="td-field-error" role={state.windowError ? "alert" : undefined}>
              {state.windowError ?? "The selected game window must be at least 10 seconds."}
            </p>
            <div className="td-window-rail" aria-label={`${formatTime(state.gameStart)} to ${formatTime(state.gameEnd)} selected`}>
              <i style={{ left: `${Math.min(100, (state.gameStart / state.duration) * 100)}%`, right: `${Math.max(0, 100 - (state.gameEnd / state.duration) * 100)}%` }} />
            </div>
          </fieldset>
          <details>
            <summary>Camera and match options</summary>
            <label className="td-check-row">
              <input type="checkbox" checked={state.cropCourt} onChange={(event) => state.setCropCourt(event.currentTarget.checked)} />
              <span><strong>Limit analysis to this court</strong><small>Useful when another court or a large crowd area is visible.</small></span>
            </label>
            <label className="td-check-row">
              <input type="checkbox" checked={state.detectSwitches} onChange={(event) => state.setDetectSwitches(event.currentTarget.checked)} />
              <span><strong>Teams change court sides</strong><small>Look for side switches so score tracking remains accurate.</small></span>
            </label>
          </details>
        </div>
      </div>

      <footer className="td-action-footer">
        <button className="td-primary-button" type="button" onClick={state.beginAnalysis}>Find the rallies</button>
        <span>{formatTime(state.gameEnd - state.gameStart)} selected for on-device analysis</span>
      </footer>
    </section>
  );
}

function AnalysisStatus({ state, compact = false }: { state: Prototype; compact?: boolean }) {
  const currentPhase = [...ANALYSIS_PHASES].reverse().find((phase) => state.analysisProgress >= phase.at);
  return (
    <section className={`td-analysis-panel ${compact ? "is-compact" : ""}`} aria-labelledby="td-analysis-title">
      <header className="td-section-intro">
        <p className="td-kicker">On-device analysis</p>
        <h1 id="td-analysis-title">{state.analysisProgress >= 100 ? "The review is ready." : "Finding the action."}</h1>
        <p>{state.analysisProgress >= 100 ? "Rallies, serve markers, and possible side switches are saved with this local project." : "Keep this tab open. You can return to another completed project while this one continues."}</p>
      </header>
      <div className="td-analysis-progress" aria-live="polite">
        <div className="td-progress-number">
          <strong>{state.analysisProgress}%</strong>
          <span>{currentPhase?.label ?? "Waiting to start"}</span>
        </div>
        <div className="td-progress-track" role="progressbar" aria-label="Analysis progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={state.analysisProgress}>
          <i style={{ width: `${state.analysisProgress}%` }} />
        </div>
        <ol className="td-phase-list">
          {ANALYSIS_PHASES.map((phase) => (
            <li key={phase.label} data-done={state.analysisProgress >= phase.at || undefined} data-current={currentPhase?.label === phase.label && state.analysisProgress < 100 || undefined}>
              <span>{state.analysisProgress >= phase.at ? "Done" : "Waiting"}</span>
              <strong>{phase.label}</strong>
            </li>
          ))}
        </ol>
      </div>
      <aside className="td-queue-card">
        <div><span>Current project</span><strong>{state.projectName}</strong></div>
        <div><span>Source</span><strong>{state.fileName}</strong></div>
        <div><span>Privacy</span><strong>No upload</strong></div>
      </aside>
      <footer className="td-action-footer">
        {state.analysisProgress >= 100 ? (
          <button className="td-primary-button" type="button" onClick={() => state.setStage("review")}>Open review</button>
        ) : (
          <button className="td-secondary-button" type="button" onClick={() => { state.setAnalysisProgress(0); state.setStage("source"); }}>Stop run</button>
        )}
        <span>{state.analysisProgress >= 100 ? "Analysis and draft are stored locally." : "The screen-awake request is active while analysis runs."}</span>
      </footer>
    </section>
  );
}

function VideoStage({
  state,
  minimal = false,
  desk = false,
  resizeControl,
}: {
  state: Prototype;
  minimal?: boolean;
  desk?: boolean;
  resizeControl?: ReactNode;
}) {
  const videoRef = state.videoElementRef;

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !state.videoUrl) return;
    if (Math.abs(video.currentTime - state.seekTarget) > 0.01) {
      video.currentTime = Math.min(video.duration || state.gameEnd, state.seekTarget);
    }
  }, [state.gameEnd, state.seekRequest, state.videoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !state.videoUrl) return;
    video.playbackRate = state.playbackRate;
  }, [state.playbackRate, state.videoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !state.videoUrl) return;
    if (state.playing) {
      void video.play().catch(() => state.setPlaying(false));
    } else {
      video.pause();
    }
  }, [state.playing, state.setPlaying, state.videoUrl]);

  function updateFromVideo(video: HTMLVideoElement) {
    let next = video.currentTime;
    if (state.finalPreview) {
      const ranges = state.finalIntervals;
      if (!ranges.some((range) => range.start <= next && next < range.end)) {
        next = ranges.find((range) => range.start > next)?.start ?? ranges[0]?.start ?? state.gameStart;
      }
      if (Math.abs(video.currentTime - next) > 0.04) video.currentTime = next;
    }
    state.trackPlayhead(Math.max(state.gameStart, Math.min(state.gameEnd, next)));
  }

  return (
    <section className={`td-video ${minimal ? "is-minimal" : ""} ${desk ? "is-rally-desk" : ""}`} data-tour={desk ? "rd-video" : undefined} aria-label="Current volleyball video review">
      <div className="td-video-picture" data-video={state.videoUrl ? true : undefined}>
        {state.videoUrl ? (
          <video
            ref={videoRef}
            src={state.videoUrl}
            playsInline
            preload="metadata"
            aria-label={`${state.fileName} review video`}
            onLoadedMetadata={(event) => {
              event.currentTarget.currentTime = Math.min(
                event.currentTarget.duration,
                state.playhead,
              );
              event.currentTarget.playbackRate = state.playbackRate;
            }}
            onTimeUpdate={(event) => updateFromVideo(event.currentTarget)}
            onPlay={() => state.setPlaying(true)}
            onPause={() => state.setPlaying(false)}
            onEnded={() => state.setPlaying(false)}
            onClick={() => state.setPlaying(!state.playing)}
          />
        ) : (
          <>
            <div className="td-court" aria-hidden="true"><i /><i /><i /></div>
            <p>{state.fileName}</p>
          </>
        )}
        {state.scoreEnabled && state.scoreOverlay && (
          <ScoreOverlay
            prepared={state.preparedScoreOverlay}
            timestamp={state.playhead}
            videoWidth={state.width}
            videoHeight={state.height}
            className="td-production-score-overlay"
          />
        )}
        <output>{formatTime(state.playhead)}</output>
      </div>
      <div className={desk ? "rd-player-controls" : undefined}>
        <div className="td-transport">
          <div className="td-playback-buttons">
            <button type="button" onClick={() => state.setPlayhead(Math.max(state.gameStart, state.playhead - 1))}>Back 1s</button>
            <button className="td-play" type="button" onClick={() => state.setPlaying(!state.playing)} aria-pressed={state.playing}>{state.playing ? "Pause" : "Play"}</button>
            <button type="button" onClick={() => state.setPlayhead(Math.min(state.gameEnd, state.playhead + 1))}>Ahead 1s</button>
            <label className="td-playhead-control">
              <span>Playhead</span>
              <input type="range" min={state.gameStart} max={state.gameEnd} step="any" value={state.playhead} onChange={(event) => state.setPlayhead(Number(event.currentTarget.value))} />
            </label>
          </div>
          <div className="td-playback-settings">
            {desk && (
              <>
                <label className="td-speed-control">
                  <span>Speed</span>
                  <select
                    aria-label="Playback speed"
                    value={state.playbackRate}
                    onChange={(event) => state.setPlaybackRate(Number(event.currentTarget.value) as (typeof PLAYBACK_RATES)[number])}
                  >
                    {PLAYBACK_RATES.map((rate) => <option key={rate} value={rate}>{rate}×</option>)}
                  </select>
                </label>
                <div className="td-transport-toggles">
                  <label>
                    <input
                      type="checkbox"
                      checked={state.scoreEnabled && state.scoreOverlay}
                      disabled={!state.scoreEnabled}
                      onChange={(event) => state.setScoreOverlay(event.currentTarget.checked)}
                    />
                    <span>Add score to video</span>
                  </label>
                  <label>
                    <input type="checkbox" checked={state.finalPreview} onChange={(event) => state.setFinalPreview(event.currentTarget.checked)} />
                    <span>Play final cut</span>
                  </label>
                </div>
              </>
            )}
          </div>
        </div>
        {resizeControl}
      </div>
      {!desk && (
        <label className="td-final-preview">
          <input type="checkbox" checked={state.finalPreview} onChange={(event) => state.setFinalPreview(event.currentTarget.checked)} />
          <span><strong>Play final cut only</strong><small>{state.finalPreview ? "Removed and excluded footage is skipped." : "Playing the continuous source window."}</small></span>
        </label>
      )}
    </section>
  );
}

function Timeline({ state, labeled = true }: { state: Prototype; labeled?: boolean }) {
  return (
    <section className="td-timeline" aria-labelledby={labeled ? "td-timeline-title" : undefined}>
      {labeled && (
        <header>
          <div><p className="td-kicker">Game timeline</p><h2 id="td-timeline-title">Choose a clip to check.</h2></div>
          <span>{state.remaining} need review</span>
        </header>
      )}
      <div className="td-timeline-rail" role="list" aria-label="Detected clips">
        {state.clips.map((clip) => (
          <div
            key={clip.id}
            role="listitem"
            style={{ flexGrow: Math.max(1, clip.end - clip.start) }}
          >
            <button
              type="button"
              data-selected={clip.id === state.selectedId || undefined}
              data-included={clip.included || undefined}
              data-review={state.reviewClipIds.has(clip.id) || undefined}
              data-manual={clip.origin === "manual" || undefined}
              data-suppression={state.suppressionReviewIds.has(clip.id) || undefined}
              onClick={() => state.selectClip(clip.id)}
              aria-label={`${clip.id}, ${clip.label}, ${clip.included ? "included" : "left out"}`}
            >
              <span>{clip.id}</span>
              <small>{clip.reviewed ? "Checked" : state.clipReviewTaskIds.has(clip.id) ? "Check" : clip.included ? "Included" : "Out"}</small>
            </button>
          </div>
        ))}
      </div>
      <div className="td-timeline-key" aria-label="Timeline legend">
        <span data-kind="included">Included</span>
        <span data-kind="review">Needs review</span>
        <span data-kind="out">Left out</span>
        <span data-kind="manual">Added manually</span>
        <span data-kind="suppression">Cleanup review</span>
      </div>
    </section>
  );
}

function rallyTimelineStatus(state: Prototype, id: string) {
  if (!state.effectiveKeptIds.has(id)) {
    return state.suppressionReviewIds.has(id) ? "pending-suppression" : "removed";
  }
  return state.reviewClipIds.has(id) ? "review" : "kept";
}

const RALLY_TIMELINE_LABELS = {
  kept: "Kept rally",
  review: "Needs review",
  "pending-suppression": "Suppressed · awaiting review",
  removed: "Removed rally",
};

function RallyFocusTimeline({ state, clip }: { state: Prototype; clip: Clip }) {
  const cut = state.workingDraft.cuts.find((candidate) => candidate.id === clip.id);
  if (!cut) return null;
  const duration = Math.max(0, state.gameEnd - state.gameStart);
  const minimumSpan = Math.min(24, duration || 24);
  const center = (cut.keepStart + cut.keepEnd) / 2;
  const span = Math.min(duration || minimumSpan, Math.max(minimumSpan, cut.keepEnd - cut.keepStart + 10));
  let start = Math.max(state.gameStart, center - span / 2);
  const end = Math.min(state.gameEnd, start + span);
  start = Math.max(state.gameStart, end - span);
  const windowDuration = Math.max(0.001, end - start);
  const position = (rangeStart: number, rangeEnd: number) => ({
    left: `${timelinePercent(rangeStart - start, windowDuration)}%`,
    width: `${timelinePercent(rangeEnd - rangeStart, windowDuration)}%`,
  });
  const seek = (event: ReactPointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left) / Math.max(1, bounds.width)));
    state.setPlayhead(start + ratio * windowDuration);
  };

  return (
    <section className="rd-rally-focus" data-status={rallyTimelineStatus(state, clip.id)} aria-label={`Current rally timeline · ${RALLY_TIMELINE_LABELS[rallyTimelineStatus(state, clip.id)]}`}>
      <div className="rd-rally-focus-times"><span>{formatPreciseTime(start)}</span><span>{formatPreciseTime((start + end) / 2)}</span><span>{formatPreciseTime(end)}</span></div>
      <div
        className="rd-rally-focus-rail"
        onPointerDown={(event) => { seek(event); try { event.currentTarget.setPointerCapture(event.pointerId); } catch { /* Seeking still works when capture is unavailable. */ } }}
        onPointerMove={(event) => { if (event.currentTarget.hasPointerCapture(event.pointerId)) seek(event); }}
        onPointerUp={(event) => { if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
      >
        <span className="rd-rally-focus-padding" data-included={state.effectiveKeptIds.has(clip.id) || undefined} style={position(cut.keepStart, cut.keepEnd)} />
        <span className="rd-rally-focus-core" data-included={state.effectiveKeptIds.has(clip.id) || undefined} style={position(cut.coreStart, cut.coreEnd)}><small>{clip.origin === "manual" ? "ADDED RALLY" : "RALLY"}</small></span>
        {state.playhead >= start && state.playhead <= end && <span className="rd-rally-focus-playhead" style={{ left: `${timelinePercent(state.playhead - start, windowDuration)}%` }} />}
      </div>
      <div className="rd-rally-focus-key"><span>Padding</span><span>Rally core</span><em>Click or drag to seek</em></div>
    </section>
  );
}

function ClipInspector({ state, condensed = false, current = false }: { state: Prototype; condensed?: boolean; current?: boolean }) {
  const clip = state.selected;
  if (!clip) return <section className="td-empty-state"><h2>{current ? "No rally at the playhead" : "No clips yet"}</h2><p>{current ? "Move the playhead into a rally to show its controls. Missed-rally and excluded-footage tools remain available below." : "Mark both edges of a missed rally at the playhead."}</p>{!current && <button type="button" onClick={state.markManualBoundary}>Mark rally start</button>}</section>;
  const suppression = state.suppressionDecisions[clip.id];
  const finalStart = Math.max(state.gameStart, clip.start - state.beforePadding);
  const finalEnd = Math.min(state.gameEnd, clip.end + state.afterPadding);
  const effectiveIncluded = state.effectiveKeptIds.has(clip.id);
  return (
    <section className={`td-clip-inspector ${condensed ? "is-condensed" : ""}`} data-tour={current ? "rd-current-rally" : undefined} aria-labelledby="td-clip-title">
      <header>
        <div>
          <p className="td-kicker">{current ? "Current rally" : "Selected clip"}</p>
          <h2 id="td-clip-title">{clip.id} <span>{clip.label}</span></h2>
        </div>
        <span className="td-confidence">{clip.origin === "manual" ? "Added by you" : `${Math.round(clip.confidence * 100)}% confidence`}</span>
      </header>
      {current && <RallyFocusTimeline state={state} clip={clip} />}
      <div className="td-range-editor">
        <div>
          <span>Rally core starts</span>
          <div><button type="button" aria-label="Move rally core start 0.5 seconds earlier" onClick={() => state.updateSelected({ start: Math.max(state.gameStart, clip.start - 0.5) })}>-0.5</button><output>{formatPreciseTime(clip.start)}</output><button type="button" aria-label="Move rally core start 0.5 seconds later" onClick={() => state.updateSelected({ start: Math.min(clip.end - 0.1, clip.start + 0.5) })}>+0.5</button></div>
        </div>
        <div>
          <span>Rally core ends</span>
          <div><button type="button" aria-label="Move rally core end 0.5 seconds earlier" onClick={() => state.updateSelected({ end: Math.max(clip.start + 0.1, clip.end - 0.5) })}>-0.5</button><output>{formatPreciseTime(clip.end)}</output><button type="button" aria-label="Move rally core end 0.5 seconds later" onClick={() => state.updateSelected({ end: Math.min(state.gameEnd, clip.end + 0.5) })}>+0.5</button></div>
        </div>
      </div>
      <div className="td-final-range">
        <span>Final padded edges</span>
        <strong>{formatPreciseTime(finalStart)} to {formatPreciseTime(finalEnd)}</strong>
        <small>Core plus {state.beforePadding.toFixed(1)}s before and {state.afterPadding.toFixed(1)}s after. Excluded ranges are still removed.</small>
      </div>
      <button className="td-split-button" type="button" onClick={state.splitSelectedAtPlayhead}>Split {current ? "current rally" : "selected clip"} at {formatPreciseTime(state.playhead)}</button>
      {suppression && (
        <aside className="td-suppression-review" data-state={suppression}>
          <div><strong>Automatic cleanup suggestion</strong><span>{suppression === "pending" ? "VolleySplice suggested leaving this footage out. Watch it before deciding." : suppression === "kept" ? "Kept after your review." : "Confirmed as left out. You can still keep it."}</span></div>
        </aside>
      )}
      {current ? (
        <div className="rd-rally-decision" role="group" aria-label="Keep or remove current rally">
          <button type="button" aria-pressed={effectiveIncluded} data-active={effectiveIncluded || undefined} onClick={() => state.setCurrentRallyIncluded(true)}>Keep rally</button>
          <button
            type="button"
            aria-pressed={!effectiveIncluded}
            data-remove
            data-active={!effectiveIncluded || undefined}
            data-pending-suppression={suppression === "pending" && !effectiveIncluded || undefined}
            onClick={() => state.setCurrentRallyIncluded(false)}
          >Remove rally</button>
        </div>
      ) : (
        <div className="td-clip-actions">
          <button className="td-primary-button" type="button" onClick={state.reviewNext}>{clip.reviewed ? "Next clip" : "Looks good, next"}</button>
          <button className="td-secondary-button" type="button" onClick={() => state.updateSelected({ included: !clip.included })}>{clip.included ? "Leave clip out" : "Put clip back"}</button>
        </div>
      )}
      {!current && <section className="td-range-drafts" aria-labelledby={`td-range-tools-${clip.id}`}>
        <header><strong id={`td-range-tools-${clip.id}`}>Add or exclude a range</strong><span>Uses the current playhead for both boundaries.</span></header>
        <div className="td-draft-grid">
          <div>
            <strong>Missed rally</strong>
            <span>{state.manualStart === null ? "Move to its first frame." : `Start marked at ${formatPreciseTime(state.manualStart)}. Move to its last frame.`}</span>
            <div><button type="button" onClick={state.markManualBoundary}>{state.manualStart === null ? "Set rally start" : "Set rally end"}</button>{state.manualStart !== null && <button type="button" onClick={state.cancelManualBoundary}>Cancel</button>}</div>
          </div>
          <div>
            <strong>Excluded footage</strong>
            <span>{state.excludedStart === null ? "For camera gaps, warmups, or footage outside the game." : `Start marked at ${formatPreciseTime(state.excludedStart)}. Move to its end.`}</span>
            <label><span>Reason</span><select value={state.excludedReason} onChange={(event) => state.setExcludedReason(event.currentTarget.value as ExcludedRange["reason"])}><option value="camera-break">Camera break</option><option value="warmup">Warmup</option><option value="other">Other non-game footage</option></select></label>
            <div><button type="button" onClick={() => state.markExcludedBoundary()}>{state.excludedStart === null ? "Set excluded start" : "Set excluded end"}</button>{state.excludedStart !== null && <button type="button" onClick={state.cancelExcludedBoundary}>Cancel</button>}</div>
          </div>
        </div>
        <div className="td-excluded-list" aria-label="Excluded footage ranges">
          {state.excludedRanges.map((range) => (
            <div key={range.id}><span><strong>{range.id}</strong> {formatPreciseTime(range.start)} to {formatPreciseTime(range.end)} ({range.reason.replace("-", " ")})</span><button type="button" onClick={() => state.removeExcludedRange(range.id)}>Remove</button></div>
          ))}
          {state.excludedRanges.length === 0 && <p>No excluded sections.</p>}
        </div>
      </section>}
      {!current && <p className="td-status-message" aria-live="polite">{state.reviewMessage}</p>}
    </section>
  );
}

function RallyDeskRangeTools({ state }: { state: Prototype }) {
  return (
    <>
      <section className="rd-sidebar-tool" data-tour="rd-range-tools" aria-labelledby="rd-missed-rally-title">
        <header><div><p className="td-kicker">Rally tools</p><h2 id="rd-missed-rally-title">Missed rally</h2></div></header>
        <p>{state.manualStart === null ? "Move to the first frame of the missed rally." : `Start marked at ${formatPreciseTime(state.manualStart)}. Move to its last frame.`}</p>
        <div className="rd-sidebar-tool-actions">
          <button type="button" onClick={state.markManualBoundary}>{state.manualStart === null ? "Set rally start" : "Set rally end"}</button>
          {state.manualStart !== null && <button type="button" onClick={state.cancelManualBoundary}>Cancel</button>}
        </div>
      </section>
      <section className="rd-sidebar-tool" data-tour="rd-excluded-footage" aria-labelledby="rd-excluded-footage-title">
        <header><div><p className="td-kicker">Cut tools</p><h2 id="rd-excluded-footage-title">Excluded footage</h2></div></header>
        <p>{state.excludedStart === null ? "Move to the first frame of footage that should never appear in the export." : `Start marked at ${formatPreciseTime(state.excludedStart)}. Move to its end.`}</p>
        <div className="rd-sidebar-tool-actions">
          <button type="button" onClick={() => state.markExcludedBoundary("other")}>{state.excludedStart === null ? "Set excluded start" : "Set excluded end"}</button>
          {state.excludedStart !== null && <button type="button" onClick={state.cancelExcludedBoundary}>Cancel</button>}
        </div>
        <div className="rd-sidebar-excluded-list" aria-label="Excluded footage ranges">
          {state.excludedRanges.map((range) => (
            <div key={range.id}><span><strong>{range.id}</strong> {formatPreciseTime(range.start)}–{formatPreciseTime(range.end)}</span><button type="button" onClick={() => state.removeExcludedRange(range.id)}>Remove</button></div>
          ))}
          {state.excludedRanges.length === 0 && <p>No excluded sections.</p>}
        </div>
        <p className="rd-sidebar-tool-status" aria-live="polite">{state.reviewMessage}{state.historyStorageFailed && " Undo history is available this session, but browser storage could not save it."}</p>
      </section>
    </>
  );
}

function FineTune({ state, modal = false }: { state: Prototype; modal?: boolean }) {
  return (
    <details className="td-fine-tune" data-modal={modal || undefined} open={modal || undefined}>
      <summary>Fine-tune final cut</summary>
      <div className="td-tune-grid">
        <label><span>Extra before <output>{state.beforePadding.toFixed(1)}s</output></span><input type="range" min="0" max="10" step="0.5" value={state.beforePadding} onChange={(event) => state.setBeforePadding(Number(event.currentTarget.value))} /></label>
        <label><span>Extra after <output>{state.afterPadding.toFixed(1)}s</output></span><input type="range" min="0" max="10" step="0.5" value={state.afterPadding} onChange={(event) => state.setAfterPadding(Number(event.currentTarget.value))} /></label>
        <label><span>Join gaps under <output>{state.joinGap.toFixed(1)}s</output></span><input type="range" min="0" max="10" step="0.5" value={state.joinGap} onChange={(event) => state.setJoinGap(Number(event.currentTarget.value))} /></label>
        <label><span>Automatic cleanup</span><select value={state.cleanup} onChange={(event) => state.setCleanup(event.currentTarget.value as CleanupStrength)}><option value="off">Off</option><option value="light">Light</option><option value="standard">Standard</option><option value="strong">Strong</option></select></label>
      </div>
    </details>
  );
}

function ScorePanel({ state, compact = false }: { state: Prototype; compact?: boolean }) {
  const targetServe = state.selectedScoreMarker ?? state.nextServeReview;
  return (
    <section className={`td-score-panel ${compact ? "is-compact" : ""}`} aria-labelledby="td-score-title">
      <header>
        <div><p className="td-kicker">Optional scoreboard</p><h2 id="td-score-title">Track the match.</h2></div>
        <label className="td-switch"><input type="checkbox" checked={state.scoreEnabled} onChange={(event) => state.setScoreEnabled(event.currentTarget.checked)} /><span>{state.scoreEnabled ? "On" : "Off"}</span></label>
      </header>
      {state.scoreEnabled && (
        <>
          <div className="td-score-teams">
            <div>
              <label><span>Team 1, starts near</span><input value={state.teamOne} onChange={(event) => state.setTeamOne(event.currentTarget.value)} /></label>
              <div className="td-score-value"><output aria-label={`${state.teamOne} derived score`}>{state.scoreOne}</output></div>
            </div>
            <div>
              <label><span>Team 2, starts far</span><input value={state.teamTwo} onChange={(event) => state.setTeamTwo(event.currentTarget.value)} /></label>
              <div className="td-score-value"><output aria-label={`${state.teamTwo} derived score`}>{state.scoreTwo}</output></div>
            </div>
          </div>
          <p className="td-score-rule">Scores are derived from the ordered serve markers. Every resolved serve after the first awards the previous rally to the team now serving.</p>
          <fieldset className="td-side-choice">
            <legend>{targetServe ? `Serve at ${formatPreciseTime(targetServe.timestamp)}` : "Side for the next manual serve"}</legend>
            <button type="button" aria-pressed={targetServe ? targetServe.side === "near" : state.servingSide === "near"} onClick={() => state.chooseServingSide("near")}>Near court</button>
            <button type="button" aria-pressed={targetServe ? targetServe.side === "far" : state.servingSide === "far"} onClick={() => state.chooseServingSide("far")}>Far court</button>
          </fieldset>
          <div className="td-marker-actions">
            <button type="button" onClick={state.addServeAtPlayhead}>Add serve at {formatPreciseTime(state.playhead)}</button>
            <button type="button" onClick={state.addSideSwitchAtPlayhead}>Add team side switch</button>
          </div>
          <div className="td-marker-summary"><span>{state.activeScoreMarkers.length} active serve markers ({state.scoreMarkers.length} saved)</span><span>{state.activeSideSwitchMarkers.length} active side switches ({state.sideSwitchMarkers.length} saved)</span><span>Serving now: {state.servingTeam === "team-1" ? state.teamOne : state.servingTeam === "team-2" ? state.teamTwo : "Needs review"}</span><button type="button" onClick={state.removeLastScoreMarker}>Remove manual serve</button></div>
          {state.serveReview > 0 && (
            <div className="td-score-review" role="status">
              <span>{state.serveReview} serve {state.serveReview === 1 ? "marker needs" : "markers need"} a side check. Choose Near or Far above to resolve the next one.</span>
            </div>
          )}
          <button className="td-replay-button" type="button" onClick={state.toggleLastPointReplay}>Toggle replay for previous rally</button>
          <p className="td-status-message" aria-live="polite">{state.scoreMessage}</p>
          <label className="td-check-row"><input type="checkbox" checked={state.scoreOverlay} onChange={(event) => state.setScoreOverlay(event.currentTarget.checked)} /><span><strong>Preview score on video</strong><small>The same scoreboard can be burned into the final MP4.</small></span></label>
        </>
      )}
    </section>
  );
}

function ReviewSummary({
  state,
  reviewActions = false,
  onOpenSettings,
}: {
  state: Prototype;
  reviewActions?: boolean;
  onOpenSettings?: () => void;
}) {
  const clipReviewCount = state.remaining;
  return (
    <aside className="td-review-summary" data-review-actions={reviewActions || undefined} data-settings={Boolean(onOpenSettings) || undefined} data-tour={reviewActions ? "rd-review-summary" : undefined}>
      {reviewActions && (
        <nav className="td-review-shortcuts" aria-label="Items needing review">
          <button type="button" disabled={state.serveReview === 0} onClick={state.openServeReview}><strong>{state.serveReview}</strong><span>Review serves</span></button>
          <button type="button" disabled={clipReviewCount === 0} onClick={state.openClipReview}><strong>{clipReviewCount}</strong><span>Review clips</span></button>
          <button type="button" disabled={state.suppressionPending === 0} onClick={state.reviewNextSuppression}><strong>{state.suppressionPending}</strong><span>Review cleaned up</span></button>
        </nav>
      )}
      <div><strong>{state.included.length}</strong><span>clips included</span></div>
      <div><strong>{formatTime(state.keptSeconds)}</strong><span>planned video</span></div>
      <div><strong>{clipReviewCount + state.suppressionPending + state.serveReview}</strong><span>review tasks</span></div>
      {onOpenSettings && (
        <button className="td-review-settings" type="button" onClick={onOpenSettings} aria-label="Final cut settings" title="Final cut settings">⚙</button>
      )}
      <button type="button" data-tour={reviewActions ? "rd-export" : undefined} onClick={() => state.setStage("deliver")}>Export</button>
    </aside>
  );
}

const EXPORTS: { id: ExportKind; label: string; description: string }[] = [
  { id: "video", label: "Final MP4", description: "Original dimensions, encoded from the local source." },
  { id: "chapters", label: "YouTube chapters", description: "Timestamps aligned to the shortened final video." },
  { id: "feedback", label: "Save the project", description: "Download this review so it can be reopened later, without video bytes." },
];

function ExportWorkspace({ state, compact = false }: { state: Prototype; compact?: boolean }) {
  const actionLabel = state.exportKind === "video"
    ? "Save MP4"
    : state.exportKind === "chapters" ? "Download text file" : "Save project";
  return (
    <section className={`td-export-panel ${compact ? "is-compact" : ""}`} aria-labelledby="td-export-title">
      <header className="td-section-intro">
        <p className="td-kicker">Local export</p>
        <h1 id="td-export-title">Make the final cut useful.</h1>
        <p>Choose one output. Every option is prepared in this browser from the reviewed project.</p>
      </header>
      <div className="td-export-grid">
        <fieldset className="td-export-kinds">
          <legend>Output</legend>
          {EXPORTS.map((item) => (
            <label key={item.id} data-selected={state.exportKind === item.id || undefined}>
              <input type="radio" name="taste-export-kind" value={item.id} checked={state.exportKind === item.id} onChange={() => { state.setExportKind(item.id); state.setExportProgress(0); }} />
              <span><strong>{item.label}</strong><small>{item.description}</small></span>
            </label>
          ))}
        </fieldset>
        <div className="td-export-options">
          <ReconnectNotice state={state} />
          <div className="td-export-recap">
            <span>Final plan</span>
            <strong>{state.included.length} clips, about {formatTime(state.keptSeconds)}</strong>
            <small>Extra time: {state.beforePadding}s before and {state.afterPadding}s after. Gaps under {state.joinGap}s stay in.</small>
          </div>
          <section className="td-readiness" aria-labelledby="td-readiness-title">
            <header><h2 id="td-readiness-title">Ready check</h2><span>Export remains available</span></header>
            <ul>
              <li data-ready={!state.needsSource || undefined}><strong>Source video</strong><span>{state.needsSource ? "Reconnect for playback and MP4" : "Connected locally"}</span></li>
              <li data-ready={state.remaining === 0 || undefined}><strong>Suggested clips</strong><span>{state.remaining === 0 ? "All included clips checked" : `${state.remaining} still need review`}</span></li>
              <li data-ready={!state.scoreEnabled || state.serveReview === 0 || undefined}><strong>Score markers</strong><span>{!state.scoreEnabled ? "Score tracking is off" : state.serveReview === 0 ? "All serve sides checked" : `${state.serveReview} serve sides need review`}</span></li>
              <li data-ready={state.suppressionPending === 0 || undefined}><strong>Automatic cleanup</strong><span>{state.suppressionPending === 0 ? "Every suggestion reviewed" : `${state.suppressionPending} excluded suggestion needs review`}</span></li>
              <li data-ready><strong>Final ranges</strong><span>{state.included.length} clips, {state.excludedRanges.length} excluded {state.excludedRanges.length === 1 ? "section" : "sections"}</span></li>
            </ul>
            {(state.remaining > 0 || state.serveReview > 0 || state.suppressionPending > 0 || state.needsSource) && <p>You can still prepare any available output. MP4 alone requires the matching source. Unresolved items remain in the saved review.</p>}
          </section>
          {state.exportKind === "video" && (
            <>
              <label className="td-check-row"><input type="checkbox" checked={state.includeScore} disabled={!state.scoreEnabled} onChange={(event) => state.setIncludeScore(event.currentTarget.checked)} /><span><strong>Render score on final video</strong><small>Available after the serve and side markers are checked.</small></span></label>
              <p className="td-privacy-line">The browser uses the original local video. Nothing is uploaded.</p>
            </>
          )}
          {state.exportKind === "chapters" && (
            <div className="td-chapter-options">
              <label className="td-check-row"><input type="checkbox" checked={state.chapterScore} onChange={(event) => state.setChapterScore(event.currentTarget.checked)} /><span><strong>Include score</strong><small>Uses the score at each visible serve.</small></span></label>
              <label className="td-check-row"><input type="checkbox" checked={state.chapterServer} onChange={(event) => state.setChapterServer(event.currentTarget.checked)} /><span><strong>Include serving team</strong><small>Uses the team names from score tracking.</small></span></label>
              <pre>{state.chapterText || "No retained rally is available for chapters."}</pre>
            </div>
          )}
          {state.exportKind === "feedback" && <p className="td-info-box">Includes the review decisions, rally ranges, excluded footage, score-marker edits, and analysis needed to reopen this project. It does not include the source video.</p>}
          <div className="td-export-actions">
            <button className="td-primary-button" type="button" onClick={() => void state.prepareExport()} disabled={state.exportBusy || (state.exportKind === "video" && state.needsSource)}>{state.exportKind === "video" && state.needsSource ? "Reconnect source for MP4" : actionLabel}</button>
            {state.exportKind === "chapters" && (
              <button className="td-secondary-button" type="button" onClick={() => void state.copyChapters()} disabled={!state.chapterText}>Copy to clipboard</button>
            )}
          </div>
          {state.showExportProgress && (
            <div className="td-export-progress" aria-live="polite">
              <div role="progressbar" aria-label="Export progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={state.exportProgress}><i style={{ width: `${state.exportProgress}%` }} /></div>
              <span>{state.exportProgress}%</span>
              {state.exportKind === "video" && (
                <div className="td-export-metrics">
                  <span><b>Speed</b><strong>{state.exportSpeed === null ? "Calculating…" : `${state.exportSpeed < 10 ? state.exportSpeed.toFixed(2) : state.exportSpeed.toFixed(1)}× realtime`}</strong></span>
                  <span><b>ETA</b><strong>{state.exportEtaSeconds !== null ? formatTime(Math.ceil(state.exportEtaSeconds)) : state.exportProgress >= 100 ? "Complete" : state.exportSpeed === null ? "Calculating…" : "Finalizing"}</strong></span>
                </div>
              )}
            </div>
          )}
          <p className="td-status-message">{state.exportStatus}</p>
        </div>
      </div>
    </section>
  );
}

function HelpDrawer({ state }: { state: Prototype }) {
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!state.helpOpen) return;
    returnFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    closeRef.current?.focus();
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        state.setHelpOpen(false);
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = [...dialogRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex]:not([tabindex="-1"])',
      )].filter((element) => element.getClientRects().length > 0);
      if (focusable.length === 0) {
        event.preventDefault();
        dialogRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable.at(-1)!;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      returnFocusRef.current?.focus();
    };
  }, [state.helpOpen, state.setHelpOpen]);

  if (!state.helpOpen) return null;
  return (
    <div className="td-help-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) state.setHelpOpen(false); }}>
      <aside ref={dialogRef} className="td-help" role="dialog" aria-modal="true" aria-labelledby="td-help-title" aria-describedby="td-help-description" tabIndex={-1}>
        <header><div><p className="td-kicker">Help</p><h2 id="td-help-title">What would you like to do?</h2></div><button ref={closeRef} type="button" onClick={() => state.setHelpOpen(false)}>Close</button></header>
        <p id="td-help-description" className="td-help-description">Choose a topic. Escape closes this panel and returns focus to the control that opened it.</p>
        <button type="button" onClick={() => { state.setStage("source"); state.setHelpOpen(false); }}><strong>Set up a new game</strong><span>Choose the game window and camera options.</span></button>
        <button type="button" onClick={() => { state.setStage("review"); state.setHelpOpen(false); }}><strong>Correct a suggested clip</strong><span>Adjust either edge, leave it out, or add a missed rally.</span></button>
        <button type="button" onClick={() => { state.setStage("deliver"); state.setHelpOpen(false); }}><strong>Choose an export</strong><span>Prepare a video, create chapters, or save the current project.</span></button>
        <div className="td-help-links"><a href="https://www.volleysplice.com/privacy.html">Read privacy policy</a><a href="https://www.volleysplice.com/terms.html">Read terms</a><a href={GOOGLE_PLAY_URL}>Get it on Google Play</a></div>
      </aside>
    </div>
  );
}

function StandardStage({ state }: { state: Prototype }) {
  if (state.stage === "source") return <SourceSetup state={state} />;
  if (state.stage === "analysis") return <AnalysisStatus state={state} />;
  if (state.stage === "deliver") return <ExportWorkspace state={state} />;
  return null;
}

function rallyAgreementLabel(clip: Clip): string {
  if (clip.origin === "manual") return "Added manually";
  if (clip.agreement === "both-models") return "Both models";
  if (clip.agreement === "all-labels-v2-only") return "All-label model";
  if (clip.agreement === "previous-production-only") return "Previous model";
  return "Detected";
}

function RallyDeskClipRegister({ state }: { state: Prototype }) {
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!state.currentPlayingClipId) return;
    const list = listRef.current;
    const currentClip = list?.querySelector<HTMLElement>("[data-playing]");
    if (list && currentClip) scrollElementIntoContainer(list, currentClip);
  }, [state.currentPlayingClipId]);

  return (
    <section className="rd-register" data-tour="rd-register" aria-label="Clip register">
      <header>
        <div><p className="td-kicker">Match ledger</p><h2>Clip register</h2></div>
        <span>{state.included.length}/{state.clips.length} kept</span>
      </header>
      <div className="rd-register-list" ref={listRef}>
        {state.clips.map((clip) => {
          const effective = state.effectiveKeptIds.has(clip.id);
          const needsSuppressionReview = state.suppressionReviewIds.has(clip.id);
          const needsReview = state.reviewClipIds.has(clip.id) || needsSuppressionReview;
          return (
            <button
              key={clip.id}
              type="button"
              data-playing={state.currentPlayingClipId === clip.id || undefined}
              data-out={!effective || undefined}
              data-review={needsReview || undefined}
              onClick={() => state.selectClip(clip.id)}
            >
              <span className="rd-register-id">{clip.id}</span>
              <span className="rd-register-range">{formatPreciseTime(clip.start)}–{formatPreciseTime(clip.end)}</span>
              <strong>{effective ? needsReview ? "Included · check" : "Included" : clip.included ? "Removed by cleanup" : "Left out"}</strong>
              <small>{rallyAgreementLabel(clip)}{clip.origin === "model" ? ` · ${Math.round(clip.confidence * 100)}%` : ""} · {(clip.end - clip.start).toFixed(1)}s</small>
              {needsSuppressionReview && <em>Cleanup decision</em>}
            </button>
          );
        })}
      </div>
      <footer>
        <button type="button" onClick={state.markManualBoundary}>
          {state.manualStart === null ? "+ Mark missed rally start" : `Finish rally from ${formatPreciseTime(state.manualStart)}`}
        </button>
      </footer>
    </section>
  );
}

function RallyDeskEventRail({ state }: { state: Prototype }) {
  const listRef = useRef<HTMLDivElement>(null);
  const events = [
    ...state.activeScoreMarkers.map((marker) => ({ kind: "serve" as const, marker })),
    ...state.activeSideSwitchMarkers.map((marker) => ({ kind: "switch" as const, marker })),
  ].sort((left, right) =>
    left.marker.timestamp - right.marker.timestamp || left.marker.id.localeCompare(right.marker.id),
  );

  useEffect(() => {
    if (!state.currentServeMarkerId) return;
    const list = listRef.current;
    const currentMarker = list?.querySelector<HTMLElement>("[data-playing]");
    if (list && currentMarker) scrollElementIntoContainer(list, currentMarker);
  }, [state.currentServeMarkerId]);

  return (
    <section className="rd-events" data-tour="rd-events" aria-label="Serve and side-switch register">
      <header>
        <div><p className="td-kicker">Match events</p><h2>Serves & sides</h2></div>
        <label><input type="checkbox" checked={state.scoreEnabled} onChange={(event) => state.setScoreEnabled(event.currentTarget.checked)} /><span>{state.scoreEnabled ? "On" : "Off"}</span></label>
      </header>
      <div className="rd-live-score" aria-label="Score at playhead">
        <span><input aria-label="Team 1 name" value={state.teamOne} onChange={(event) => state.setTeamOne(event.currentTarget.value)} /><strong>{state.scoreOne}</strong></span>
        <i>{formatPreciseTime(state.playhead)}</i>
        <span><input aria-label="Team 2 name" value={state.teamTwo} onChange={(event) => state.setTeamTwo(event.currentTarget.value)} /><strong>{state.scoreTwo}</strong></span>
      </div>
      <RallyDeskPointTimeline state={state} />
      {state.scoreEnabled ? (
        <>
          <div className="rd-event-list" ref={listRef}>
            {events.map((event) => {
              if (event.kind === "switch") {
                const selected = event.marker.id === state.selectedSideSwitchId;
                return (
                  <button key={event.marker.id} type="button" data-kind="switch" data-playing={selected || undefined} onClick={() => state.selectSideSwitch(event.marker.id)}>
                    <b aria-hidden="true">⇄</b><span><strong>Side switch</strong><small>{formatPreciseTime(event.marker.timestamp)} · {event.marker.origin}</small></span>
                  </button>
                );
              }
              const score = deriveScoreAt(state.activeScoreTracking, event.marker.timestamp);
              return (
                <button key={event.marker.id} type="button" data-kind="serve" data-playing={event.marker.id === state.currentServeMarkerId || undefined} data-review={event.marker.side === "review" || undefined} onClick={() => state.selectScoreMarker(event.marker.id)}>
                  <b className="rd-serve-icon" aria-hidden="true">🏐</b>
                  <span><strong>{event.marker.side === "review" ? "Serve needs review" : `${event.marker.side === "near" ? "Near" : "Far"} court serve`}</strong><small>{formatPreciseTime(event.marker.timestamp)} · {score.team1Score}–{score.team2Score}{event.marker.ignorePreviousPoint ? " · replay" : ""}</small></span>
                </button>
              );
            })}
            {events.length === 0 && <p>No match events are saved yet.</p>}
          </div>
          {state.selectedScoreMarker && (
            <section className="rd-event-editor" aria-label="Current serve settings">
              <header><strong>{state.selectedScoreMarker.id}</strong><span>{formatPreciseTime(state.selectedScoreMarker.timestamp)}</span></header>
              <fieldset>
                <legend>Serving court</legend>
                {(["near", "far"] as const).map((side) => (
                  <button key={side} type="button" aria-pressed={state.selectedScoreMarker?.side === side} onClick={() => state.updateScoreMarkerSide(state.selectedScoreMarker!.id, side)}>{side === "near" ? "Near" : "Far"}</button>
                ))}
              </fieldset>
              <label><input type="checkbox" checked={state.selectedScoreMarker.ignorePreviousPoint} onChange={(event) => state.setScoreMarkerReplay(state.selectedScoreMarker!.id, event.currentTarget.checked)} /><span>Previous rally was a replay</span></label>
              <button className="rd-remove-event" type="button" onClick={state.removeSelectedEvent}>Remove serve</button>
            </section>
          )}
          {state.selectedSideSwitch && (
            <section className="rd-event-editor"><header><strong>Side switch</strong><span>{formatPreciseTime(state.selectedSideSwitch.timestamp)}</span></header><p>Team-to-court mapping changes from the next serve.</p><button className="rd-remove-event" type="button" onClick={state.removeSelectedEvent}>Remove side switch</button></section>
          )}
          <div className="rd-event-add"><button type="button" onClick={state.addServeAtPlayhead}>+ Serve at playhead</button><button type="button" onClick={state.addSideSwitchAtPlayhead}>+ Side switch</button></div>
        </>
      ) : <p className="rd-events-off">Enable score tracking to edit serves, derived scores, and chapter labels.</p>}
    </section>
  );
}

function RallyDeskPointTimeline({ state }: { state: Prototype }) {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const timeline = scorePointTimelineSnapshot(state.preparedScoreOverlay, state.playhead);
  const latestPointId = timeline.points[timeline.points.length - 1]?.serveMarkerId ?? "";

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    scroller.scrollLeft = scroller.scrollWidth;
  }, [latestPointId, timeline.points.length]);

  return (
    <section className="rd-score-timeline" aria-label="Point timeline through the current playhead">
      <header><strong>Point timeline</strong><span>Latest rally on the right</span></header>
      <div className="rd-score-timeline-scroll" ref={scrollerRef}>
        {timeline.points.length === 0 ? (
          <p>Points appear here as rallies finish.</p>
        ) : (
          <div
            className="rd-score-point-strip"
            style={{ gridTemplateColumns: `repeat(${timeline.points.length}, 30px)` }}
          >
            {timeline.points.map((point, index) => (
              <button
                key={`${point.serveMarkerId}-${index}`}
                type="button"
                data-team={point.winnerTeamId}
                style={{ gridColumn: index + 1, gridRow: point.winnerTeamId === "team-1" ? 1 : 2 }}
                onClick={() => state.selectScoreMarker(point.serveMarkerId)}
                aria-label={`${point.winnerTeamId === "team-1" ? state.teamOne : state.teamTwo} point ${point.teamPointNumber}`}
                title={`Rally point ${point.teamPointNumber} · select to seek`}
              >{point.teamPointNumber}</button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

type RallyDeskTimelineTarget =
  | { kind: "clip"; id: string }
  | { kind: "serve"; id: string }
  | { kind: "switch"; id: string };

type RallyDeskTimelineDrag = {
  pointerId: number;
  left: number;
  width: number;
  start: number;
  end: number;
  startX: number;
  startY: number;
  moved: boolean;
  tapTarget: RallyDeskTimelineTarget | null;
};

function RallyDeskTimeline({ state }: { state: Prototype }) {
  const midpoint = state.gameStart + (state.gameEnd - state.gameStart) / 2;
  const dragRef = useRef<RallyDeskTimelineDrag | null>(null);
  const suppressClickUntilRef = useRef(0);

  function seekFromPointer(clientX: number, drag: RallyDeskTimelineDrag) {
    const ratio = Math.max(0, Math.min(1, (clientX - drag.left) / Math.max(1, drag.width)));
    state.setPlayhead(drag.start + ratio * (drag.end - drag.start));
  }

  function beginSeek(event: ReactPointerEvent<HTMLDivElement>, start: number, end: number) {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const target = event.target instanceof Element ? event.target : null;
    const action = target?.closest<HTMLButtonElement>(
      "[data-timeline-cut-id], [data-timeline-event]",
    );
    const id = action?.dataset.timelineCutId ?? action?.dataset.timelineEventId;
    const kind = action?.dataset.timelineCutId
      ? "clip"
      : action?.dataset.timelineEvent === "serve"
        ? "serve"
        : action?.dataset.timelineEvent === "switch"
          ? "switch"
          : null;
    dragRef.current = {
      pointerId: event.pointerId,
      left: rect.left,
      width: rect.width,
      start,
      end,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
      tapTarget: id && kind ? { kind, id } : null,
    };
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Seeking still works while the pointer remains over the rail.
    }
  }

  function moveSeek(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (!drag.moved) {
      const horizontalDistance = Math.abs(event.clientX - drag.startX);
      const verticalDistance = Math.abs(event.clientY - drag.startY);
      if (horizontalDistance < 4 || verticalDistance > horizontalDistance) return;
      drag.moved = true;
    }
    event.preventDefault();
    seekFromPointer(event.clientX, drag);
  }

  function endSeek(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (drag.moved) {
      seekFromPointer(event.clientX, drag);
      suppressClickUntilRef.current = Date.now() + 800;
    } else if (drag.tapTarget) {
      suppressClickUntilRef.current = Date.now() + 800;
      if (drag.tapTarget.kind === "clip") state.selectClip(drag.tapTarget.id);
      else if (drag.tapTarget.kind === "serve") state.selectScoreMarker(drag.tapTarget.id);
      else state.selectSideSwitch(drag.tapTarget.id);
    } else {
      seekFromPointer(event.clientX, drag);
    }
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function cancelSeek(event: ReactPointerEvent<HTMLDivElement>) {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  }

  function renderRow(start: number, end: number, rowIndex: number) {
    const duration = Math.max(0.001, end - start);
    const segment = (segmentStart: number, segmentEnd: number) => ({ start: Math.max(start, segmentStart), end: Math.min(end, segmentEnd) });
    const position = (segmentStart: number, segmentEnd: number) => ({ left: `${timelinePercent(segmentStart - start, duration)}%`, width: `${timelinePercent(segmentEnd - segmentStart, duration)}%` });
    const inRow = (timestamp: number) => timestamp >= start && (rowIndex === 1 ? timestamp <= end : timestamp < end);
    const playheadInRow = inRow(state.playhead);
    return (
      <div className="rd-timeline-row" key={`${start}-${end}`}>
        <div
          className="rd-timeline-rail"
          role="group"
          aria-label={`${rowIndex === 0 ? "First" : "Second"} half of game timeline`}
          onPointerDown={(event) => beginSeek(event, start, end)}
          onPointerMove={moveSeek}
          onPointerUp={endSeek}
          onPointerCancel={cancelSeek}
          onLostPointerCapture={cancelSeek}
          onClickCapture={(event) => {
            if (Date.now() > suppressClickUntilRef.current) return;
            event.preventDefault();
            event.stopPropagation();
            suppressClickUntilRef.current = 0;
          }}
        >
          {state.workingDraft.ignoredIntervals.map((interval) => {
            const clipped = segment(interval.start, interval.end);
            return clipped.end > clipped.start ? <span key={interval.id} className="rd-timeline-ignored" style={position(clipped.start, clipped.end)} title={`Excluded · ${formatPreciseTime(interval.start)}–${formatPreciseTime(interval.end)}`} /> : null;
          })}
          {state.workingDraft.cuts.map((cut) => {
            const clipped = segment(cut.keepStart, cut.keepEnd);
            if (clipped.end <= clipped.start) return null;
            const inner = (innerStart: number, innerEnd: number) => ({ left: `${timelinePercent(innerStart - clipped.start, clipped.end - clipped.start)}%`, width: `${timelinePercent(innerEnd - innerStart, clipped.end - clipped.start)}%` });
            const core = segment(Math.max(clipped.start, cut.coreStart), Math.min(clipped.end, cut.coreEnd));
            return (
              <button
                key={cut.id}
                type="button"
                className="rd-timeline-cut"
                data-timeline-cut-id={cut.id}
                data-playing={cut.id === state.currentPlayingClipId || undefined}
                data-kept={state.effectiveKeptIds.has(cut.id) || undefined}
                data-status={rallyTimelineStatus(state, cut.id)}
                data-review={state.reviewClipIds.has(cut.id) || undefined}
                data-origin={cut.origin}
                style={position(clipped.start, clipped.end)}
                onClick={(event) => {
                  event.stopPropagation();
                  state.selectClip(cut.id);
                }}
                title={`${cut.id} · ${RALLY_TIMELINE_LABELS[rallyTimelineStatus(state, cut.id)]} · ${formatPreciseTime(cut.keepStart)}–${formatPreciseTime(cut.keepEnd)}`}
              >
                <span className="rd-timeline-padding" />
                {core.end > core.start && <i className="rd-timeline-core" style={inner(core.start, core.end)} />}
              </button>
            );
          })}
          {state.cleanupSuggestions.map((suggestion) => {
            const cut = state.workingDraft.cuts.find((cut) => cut.id === suggestion.cutId);
            const clipped = segment(Math.max(suggestion.start, cut?.coreStart ?? suggestion.start), Math.min(suggestion.end, cut?.coreEnd ?? suggestion.end));
            const decision = (suggestion.cutId ? state.workingDraft.suppressionDecisionOverrides[rallySuppressionDecisionKey(suggestion.cutId)] : undefined)
              ?? state.workingDraft.suppressionDecisionOverrides[suggestion.logicalId]
              ?? (suggestion.cutId && state.suppressionReviewIds.has(suggestion.cutId)
                && !state.effectiveKeptIds.has(suggestion.cutId) ? "pending" : "keep");
            return clipped.end > clipped.start ? <span key={`${suggestion.id}-${suggestion.cutId}`} className="rd-timeline-suppression" data-state={decision} style={position(clipped.start, clipped.end)} title={`Cleanup ${decision} · ${Math.round(suggestion.score * 100)}%`} /> : null;
          })}
          {state.finalIntervals.flatMap((interval) => (interval.joinedGaps ?? []).map((gap) => {
            const clipped = segment(gap.start, gap.end);
            return clipped.end > clipped.start ? <span key={`${gap.start}-${gap.end}`} className="rd-timeline-gap" style={position(clipped.start, clipped.end)} title="Short gap retained in final cut" /> : null;
          }))}
          {state.scoreEnabled && state.activeScoreMarkers.filter((marker) => inRow(marker.timestamp)).map((marker) => (
            <button
              key={marker.id}
              type="button"
              className="rd-timeline-serve"
              data-timeline-event="serve"
              data-timeline-event-id={marker.id}
              data-side={marker.side}
              data-playing={marker.id === state.currentServeMarkerId || undefined}
              style={{ left: `${timelinePercent(marker.timestamp - start, duration)}%` }}
              onClick={(event) => {
                event.stopPropagation();
                state.selectScoreMarker(marker.id);
              }}
              aria-label={`Serve at ${formatPreciseTime(marker.timestamp)}, ${marker.side}`}
            ><span className="rd-serve-icon" aria-hidden="true">🏐</span></button>
          ))}
          {state.scoreEnabled && state.activeSideSwitchMarkers.filter((marker) => inRow(marker.timestamp)).map((marker) => (
            <button
              key={marker.id}
              type="button"
              className="rd-timeline-switch"
              data-timeline-event="switch"
              data-timeline-event-id={marker.id}
              data-playing={marker.id === state.selectedSideSwitchId || undefined}
              style={{ left: `${timelinePercent(marker.timestamp - start, duration)}%` }}
              onClick={(event) => {
                event.stopPropagation();
                state.selectSideSwitch(marker.id);
              }}
              aria-label={`Side switch at ${formatPreciseTime(marker.timestamp)}`}
            >⇄</button>
          ))}
          {playheadInRow && <span className="rd-timeline-playhead" style={{ left: `${timelinePercent(state.playhead - start, duration)}%` }} />}
        </div>
        <div className="rd-timeline-times"><span>{formatTime(start)}</span><span>{formatTime((start + end) / 2)}</span><span>{formatTime(end)}</span></div>
      </div>
    );
  }

  return (
    <section className="rd-timeline" data-tour="rd-timeline" aria-labelledby="rd-timeline-title">
      <header><div><p className="td-kicker">Game timeline</p><h2 id="rd-timeline-title">Source time, cuts, and match events</h2></div><span>Drag anywhere to seek</span></header>
      {renderRow(state.gameStart, midpoint, 0)}
      {renderRow(midpoint, state.gameEnd, 1)}
      <div className="rd-timeline-key"><span data-kind="core">Rally core</span><span data-kind="padding">Padding</span><span data-kind="review">Needs review</span><span data-kind="suppression">Pending suppression</span><span data-kind="removed">Removed</span><span data-kind="serve">Serve</span><span data-kind="switch">Side switch</span><span data-kind="ignored">Excluded</span><span data-kind="gap">Joined gap</span></div>
    </section>
  );
}

function RallyDeskSettingsModal({
  state,
  onClose,
}: {
  state: Prototype;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div className="rd-settings-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section className="rd-settings-dialog" role="dialog" aria-modal="true" aria-labelledby="rd-settings-title">
        <header>
          <div><p className="td-kicker">Final cut settings</p><h2 id="rd-settings-title">Fine-tune final cut</h2></div>
          <button type="button" onClick={onClose} aria-label="Close final cut settings">×</button>
        </header>
        <p>Adjust the padding around every rally, short joined gaps, and automatic cleanup strength.</p>
        <FineTune state={state} modal />
        <footer><button className="td-primary-button" type="button" onClick={onClose}>Done</button></footer>
      </section>
    </div>
  );
}

function useRallyDeskShortcuts(state: Prototype, settingsOpen: boolean) {
  const reviewCursor = useRef<ReviewItem | null>(null);
  useEffect(() => { reviewCursor.current = null; }, [state.selectedProjectId]);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (event.defaultPrevented || settingsOpen || state.helpOpen ||
        document.querySelector('dialog[open], [role="dialog"][aria-modal="true"]') ||
        target?.closest('select, [role="combobox"], .rd-menu-panel, [role="dialog"]')) return;
      const action = shortcutAction(event);
      if (!action) return;
      if (action !== "cancelRange" && target?.closest('input, textarea, [contenteditable]:not([contenteditable="false"]), [role="textbox"], [role="slider"]')) return;
      if (state.stage !== "review" && !["new", "projects", "export"].includes(action)) return;
      if (action === "cancelRange" && state.manualStart === null && state.excludedStart === null) return;
      event.preventDefault();
      switch (action) {
        case "undo": state.restoreHistory("undo"); break;
        case "redo": state.restoreHistory("redo"); break;
        case "playPause": state.setPlaying((playing) => !playing); break;
        case "new": state.selectProject(null); break;
        case "export": state.setStage("deliver"); break;
        case "projects": {
          const select = [...document.querySelectorAll<HTMLSelectElement>('[data-tour="rd-project-selector"] select')]
            .find((element) => element.getClientRects().length > 0);
          if (select) {
            select.scrollIntoView({ block: "nearest" });
            select.focus();
            // Native picker support varies; focus still enables normal arrow navigation.
            try { select.showPicker?.(); } catch { /* Leave the select focused. */ }
          }
          break;
        }
        case "near":
        case "far":
          if (state.scoreEnabled && state.selectedScoreMarker)
            state.updateScoreMarkerSide(state.selectedScoreMarker.id, action);
          break;
        case "keep": state.setCurrentRallyIncluded(true); break;
        case "remove": state.setCurrentRallyIncluded(false); break;
        case "split": state.splitSelectedAtPlayhead(); break;
        case "serve": if (state.scoreEnabled) state.addServeAtPlayhead(); break;
        case "exclude": state.markExcludedBoundary("other"); break;
        case "missed": state.markManualBoundary(); break;
        case "cancelRange":
          if (state.manualStart !== null) state.cancelManualBoundary();
          if (state.excludedStart !== null) state.cancelExcludedBoundary();
          break;
        case "switch": if (state.scoreEnabled) state.addSideSwitchAtPlayhead(); break;
        case "removeEvent": if (state.scoreEnabled) state.removeSelectedEvent(); break;
        case "back":
        case "forward":
          state.setPlayhead(Math.max(state.gameStart, Math.min(state.gameEnd,
            state.playhead + (action === "back" ? -5 : 5))));
          break;
        case "faster":
        case "slower": {
          const index = PLAYBACK_RATES.indexOf(state.playbackRate);
          state.setPlaybackRate(PLAYBACK_RATES[Math.max(0, Math.min(PLAYBACK_RATES.length - 1,
            index + (action === "faster" ? 1 : -1)))]);
          break;
        }
        case "review": {
          const items: ReviewItem[] = state.clips.flatMap((clip) => {
            const kind = state.suppressionReviewIds.has(clip.id) ? "cleanup"
              : state.reviewClipIds.has(clip.id) ? "clip" : null;
            return kind ? [{ kind, id: clip.id, time: clip.start }] : [];
          });
          if (state.scoreEnabled) items.push(...state.activeScoreMarkers
            .filter((marker) => marker.side === "review")
            .map((marker): ReviewItem => ({ kind: "serve", id: marker.id, time: marker.timestamp })));
          const next = nextReviewItem(items, reviewCursor.current);
          reviewCursor.current = next;
          if (next) {
            state.setFinalPreview(false);
            state.setPlaying(false);
            if (next.kind === "serve") state.selectScoreMarker(next.id);
            else state.selectClip(next.id);
          }
          break;
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });
}

export function RallyDesk({
  review,
  onDeleteProject,
}: {
  review: ReadyDesignReview;
  onDeleteProject?: () => void;
}) {
  const state = usePrototype(review, "review", false, true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  useRallyDeskShortcuts(state, settingsOpen);
  const layoutPreferences = useWorkspaceLayout();
  return (
    <main className="taste-root taste-rally-desk" data-theme={state.dark ? "dark" : "light"}>
      <header className="rd-topbar" data-review={state.stage === "review" || undefined}>
        <Brand logo />
        {state.stage === "review" && <RallyDeskProjectSelector state={state} className="rd-topbar-project" onDeleteProject={onDeleteProject} />}
        <StageButtons state={state} editorOnly />
        {state.stage === "review" && <ReviewSummary state={state} reviewActions onOpenSettings={() => setSettingsOpen(true)} />}
        <AppMenu projectName={state.projectName} onResetProject={state.resetProjectChanges} dark={state.dark} onToggleTheme={() => state.setDark(!state.dark)} onResetLayout={() => layoutPreferences.onLayoutChange({})} />
      </header>
      {state.stage === "review" ? (
        <ResizableWorkspace
          {...layoutPreferences}
          ledger={<><RallyDeskEventRail state={state} /><RallyDeskClipRegister state={state} /></>}
          inspector={<><ClipInspector state={state} condensed current /><RallyDeskRangeTools state={state} /></>}
          mobileRegister={<RallyDeskClipRegister state={state} />}
        >
          {(videoResize) => <>
            <header className="rd-review-head"><div><p className="td-kicker">Current review workspace</p><RallyDeskProjectSelector state={state} className="rd-review-project" onDeleteProject={onDeleteProject} /></div><ReviewSummary state={state} reviewActions onOpenSettings={() => setSettingsOpen(true)} /></header>
            <ReconnectNotice state={state} />
            <div className="rd-mobile-events"><RallyDeskEventRail state={state} /></div>
            <div className="rd-media-stack"><VideoStage state={state} desk resizeControl={videoResize} /><RallyDeskTimeline state={state} /></div>
          </>}
        </ResizableWorkspace>
      ) : (
        <div className="rd-shell"><div className="rd-main"><RallyDeskProjectSelector state={state} className="rd-review-project" onDeleteProject={onDeleteProject} /><StandardStage state={state} /></div></div>
      )}
      <HelpDrawer state={state} />
      {settingsOpen && <RallyDeskSettingsModal state={state} onClose={() => setSettingsOpen(false)} />}
      {state.stage === "review" && <RallyDeskGuidedTour />}
    </main>
  );
}

function CourtCanvas({ review }: { review: ReadyDesignReview }) {
  const state = usePrototype(review);
  return (
    <main className="taste-root taste-court-canvas" data-theme={state.dark ? "dark" : "light"}>
      <header className="cc-header">
        <div className="cc-header-row"><Brand /><ProjectChip state={state} /><UtilityLinks state={state} /></div>
        <StageButtons state={state} />
      </header>
      <div className="cc-page">
        {state.stage === "review" ? (
          <>
            <section className="cc-opening">
              <div><p className="td-kicker">Current review</p><h1>See the match.<br /><em>Shape the cut.</em></h1><p>Keep attention on the rally while corrections, score context, and export readiness stay within reach.</p></div>
              <ReviewSummary state={state} />
            </section>
            <ReconnectNotice state={state} />
            <Timeline state={state} />
            <section className="cc-review-grid">
              <VideoStage state={state} />
              <ClipInspector state={state} />
            </section>
            <section className="cc-lower-grid"><ScorePanel state={state} /><div className="cc-settings"><h2>Final cut shape</h2><FineTune state={state} /><button className="td-primary-button" type="button" onClick={() => state.setStage("deliver")}>Choose export</button></div></section>
          </>
        ) : <StandardStage state={state} />}
      </div>
      <footer className="cc-footer"><span>Your video stays on this device.</span><UtilityLinks state={state} /></footer>
      <HelpDrawer state={state} />
    </main>
  );
}

function ReviewLane({ review }: { review: ReadyDesignReview }) {
  const state = usePrototype(review);
  return (
    <main className="taste-root taste-review-lane" data-theme={state.dark ? "dark" : "light"}>
      <header className="rl-header"><Brand /><ProjectChip state={state} /><UtilityLinks state={state} /></header>
      <div className="rl-layout">
        <aside className="rl-steps">
          <p className="td-kicker">Your path</p>
          <StageButtons state={state} />
          <div className="rl-reassurance"><strong>Safe to close</strong><span>{state.storageMessage}. The source video itself is never copied into project storage.</span></div>
        </aside>
        <div className="rl-main">
          {state.stage === "review" ? (
            <section className="rl-task-card">
              <header className="rl-task-header"><div><p className="td-kicker">Current task</p><h1>Check {state.selected?.id}.</h1><p>Watch the suggested range, correct it if needed, then move to the next uncertain clip.</p></div><span>{state.remaining} remaining</span></header>
              <ReconnectNotice state={state} />
              <VideoStage state={state} minimal />
              <Timeline state={state} labeled={false} />
              <ClipInspector state={state} condensed />
              <details className="rl-secondary">
                <summary>Score and serving side</summary>
                <ScorePanel state={state} compact />
              </details>
              <details className="rl-secondary">
                <summary>Padding and automatic cleanup</summary>
                <FineTune state={state} />
              </details>
              <footer><span>{state.storageMessage}</span><button className="td-primary-button" type="button" onClick={() => state.setStage("deliver")}>Finish review</button></footer>
            </section>
          ) : <StandardStage state={state} />}
        </div>
      </div>
      <HelpDrawer state={state} />
    </main>
  );
}

function FilmRoom({ review }: { review: ReadyDesignReview }) {
  const state = usePrototype(review, "review", true);
  const [tray, setTray] = useState<"clip" | "score">("clip");
  return (
    <main className="taste-root taste-film-room" data-theme={state.dark ? "dark" : "light"}>
      <header className="fr-header"><Brand /><StageButtons state={state} /><div className="fr-actions"><ProjectChip state={state} /><a href="https://www.volleysplice.com/privacy.html">Privacy</a><a href="https://www.volleysplice.com/terms.html">Terms</a><a href={GOOGLE_PLAY_URL}>Google Play</a><button type="button" onClick={() => state.setDark(!state.dark)}>{state.dark ? "Light theme" : "Dark theme"}</button></div></header>
      {state.stage === "review" ? (
        <div className="fr-room">
          <section className="fr-screen">
            <div className="fr-title"><div><p className="td-kicker">Current film room</p><h1>{state.projectName}</h1></div><ReviewSummary state={state} /></div>
            <ReconnectNotice state={state} />
            <VideoStage state={state} />
            <Timeline state={state} />
            <FineTune state={state} />
          </section>
          <aside className="fr-tray">
            <nav aria-label="Review tray"><button type="button" data-active={tray === "clip" || undefined} onClick={() => setTray("clip")}>Clip</button><button type="button" data-active={tray === "score" || undefined} onClick={() => setTray("score")}>Score</button></nav>
            {tray === "clip" ? <ClipInspector state={state} condensed /> : <ScorePanel state={state} compact />}
            <button className="td-primary-button fr-export" type="button" onClick={() => state.setStage("deliver")}>Open export room</button>
          </aside>
        </div>
      ) : (
        <div className="fr-stage-wrap"><StandardStage state={state} /></div>
      )}
      <HelpDrawer state={state} />
    </main>
  );
}

function MatchLedger({ review }: { review: ReadyDesignReview }) {
  const state = usePrototype(review);
  return (
    <main className="taste-root taste-match-ledger" data-theme={state.dark ? "dark" : "light"}>
      <header className="ml-header"><Brand /><StageButtons state={state} /><UtilityLinks state={state} /></header>
      <div className="ml-document">
        <header className="ml-document-head">
          <div><p className="td-kicker">Local match project</p><h1>{state.projectName}</h1><span>{state.fileName}</span></div>
          <ProjectChip state={state} />
        </header>
        {state.stage === "review" ? (
          <>
            <ReviewSummary state={state} />
            <ReconnectNotice state={state} />
            <Timeline state={state} />
            <div className="ml-review-grid">
              <aside className="ml-clip-ledger" aria-label="Clip register">
                <header><h2>Clip register</h2><span>{state.clips.length} detected or added</span></header>
                {state.clips.map((clip) => (
                  <button key={clip.id} type="button" data-selected={state.selectedId === clip.id || undefined} onClick={() => state.selectClip(clip.id)}>
                    <strong>{clip.id}</strong><span>{formatTime(clip.start)} to {formatTime(clip.end)}</span><small>{clip.included ? clip.reviewed ? "Included, checked" : "Included, check" : "Left out"}</small>
                  </button>
                ))}
                <button className="ml-add" type="button" onClick={state.markManualBoundary}>{state.manualStart === null ? "Set missed-rally start" : "Set missed-rally end"}</button>
              </aside>
              <div className="ml-review-body"><VideoStage state={state} minimal /><ClipInspector state={state} /><FineTune state={state} /></div>
              <ScorePanel state={state} compact />
            </div>
            <footer className="ml-review-footer"><span>{state.storageMessage}</span><button className="td-primary-button" type="button" onClick={() => state.setStage("deliver")}>Prepare outputs</button></footer>
          </>
        ) : <StandardStage state={state} />}
      </div>
      <footer className="ml-footer"><span>VolleySplice keeps projects and analysis in this browser.</span><UtilityLinks state={state} /></footer>
      <HelpDrawer state={state} />
    </main>
  );
}

function UnknownTasteDesign({ slug }: { slug: string }) {
  return (
    <main className="taste-root taste-unknown" data-theme="light">
      <Brand />
      <section><p className="td-kicker">Design not found</p><h1>No Taste concept uses “{slug}”.</h1><p>Choose one of the five product-workspace prototypes.</p><nav>{TASTE_DESIGNS.map((design) => <a key={design.slug} href={`/${design.slug}`}>{design.title}</a>)}</nav></section>
    </main>
  );
}

export function TasteDesignRoute({
  slug,
  review,
}: TasteDesignRouteProps & { review: ReadyDesignReview }) {
  switch (slug) {
    case "rally-desk": return <RallyDesk review={review} />;
    case "court-canvas": return <CourtCanvas review={review} />;
    case "review-lane": return <ReviewLane review={review} />;
    case "film-room": return <FilmRoom review={review} />;
    case "match-ledger": return <MatchLedger review={review} />;
    default: return <UnknownTasteDesign slug={slug} />;
  }
}
