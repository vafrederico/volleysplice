"use client";

import Link from "next/link";
import { useMemo, useRef, useState, useSyncExternalStore, useTransition } from "react";
import { useRouter } from "next/navigation";

import { Brand } from "@/components/brand";
import { RallyTimeline, type TimelineTrack } from "@/components/rally-timeline";
import type {
  AnalysisKind,
  AnalysisOption,
  ReviewAnalysis,
  ReviewVideoOption,
  TrainingCorpus,
  TrainingCorpusView,
} from "@/lib/analysis-types";
import {
  buildEditList,
  DEFAULT_JOIN_GAP_SECONDS,
  formatTime,
  type Rally,
} from "@/lib/edit-list";
import {
  DEFAULT_VISIBLE_MODEL_IDS,
  PREFERRED_ENVIRONMENT_EXPERIMENT_MODEL_ID,
} from "@/lib/experiment-models";
import { parseLabelDocument, type IgnoredInterval, type LabelDocument } from "@/lib/annotations";
import {
  buildLiveTimeComparisonSegments,
  buildPaddingSegments,
  calculateDurationDeltaPercent,
  calculateF1,
  calculateLiveTimeMetrics,
  excludeIgnoredTime,
  markModelPaddingOrigins,
  padAndMergeRallies,
  totalRallySeconds,
} from "@/lib/timeline-comparison";

const MODEL_VISIBILITY_STORAGE_KEY = "volleycut:model-timeline-visibility:v2";
const MODEL_VISIBILITY_EVENT = "volleycut:model-timeline-visibility";
const ACTIVITY_PADDING_STORAGE_KEY = "volleycut:activity-padding:v1";
const ACTIVITY_PADDING_EVENT = "volleycut:activity-padding";
const DEFAULT_VISIBLE_MODEL_KEYS = new Set(
  [...DEFAULT_VISIBLE_MODEL_IDS].map((id) => `without-beach:${id}`),
);
const DEFAULT_ACTIVITY_PADDING = {
  before: 3,
  after: 2,
  joinGap: DEFAULT_JOIN_GAP_SECONDS,
} as const;
type ActivityPadding = Readonly<{ before: number; after: number; joinGap: number }>;
type ModelFilterBasis = "coreHuman" | "paddedHuman";
type ModelFilterMetric = "precision" | "recall" | "f1";
type ModelFilterOperator = "greater" | "less";
let cachedVisibilityValue: string | null | undefined;
let cachedVisibleModelKeys = DEFAULT_VISIBLE_MODEL_KEYS;
let fallbackVisibilityValue: string | null = null;
let cachedPaddingValue: string | null | undefined;
let cachedActivityPadding: ActivityPadding = DEFAULT_ACTIVITY_PADDING;
let fallbackPaddingValue: string | null = null;

function visibleModelSnapshot(): Set<string> {
  if (typeof window === "undefined") return DEFAULT_VISIBLE_MODEL_KEYS;
  let value: string | null;
  try {
    value = window.localStorage.getItem(MODEL_VISIBILITY_STORAGE_KEY);
  } catch {
    value = fallbackVisibilityValue;
  }
  if (value === cachedVisibilityValue) return cachedVisibleModelKeys;
  cachedVisibilityValue = value;
  try {
    const saved = JSON.parse(value ?? "null") as { visible?: unknown } | null;
    cachedVisibleModelKeys = saved && Array.isArray(saved.visible)
      ? new Set(saved.visible.filter((item): item is string => typeof item === "string"))
      : DEFAULT_VISIBLE_MODEL_KEYS;
  } catch {
    cachedVisibleModelKeys = DEFAULT_VISIBLE_MODEL_KEYS;
  }
  return cachedVisibleModelKeys;
}

function subscribeToModelVisibility(onChange: () => void): () => void {
  const handleStorage = (event: StorageEvent) => {
    if (event.key === MODEL_VISIBILITY_STORAGE_KEY) {
      cachedVisibilityValue = undefined;
      onChange();
    }
  };
  const handleLocalChange = () => onChange();
  window.addEventListener("storage", handleStorage);
  window.addEventListener(MODEL_VISIBILITY_EVENT, handleLocalChange);
  return () => {
    window.removeEventListener("storage", handleStorage);
    window.removeEventListener(MODEL_VISIBILITY_EVENT, handleLocalChange);
  };
}

function saveModelVisibility(visible: Set<string>): void {
  const value = JSON.stringify({ visible: [...visible].sort() });
  fallbackVisibilityValue = value;
  try {
    window.localStorage.setItem(MODEL_VISIBILITY_STORAGE_KEY, value);
  } catch {
    // The in-memory fallback keeps the toggle working for this tab.
  }
  cachedVisibilityValue = undefined;
  window.dispatchEvent(new Event(MODEL_VISIBILITY_EVENT));
}

function activityPaddingSnapshot(): ActivityPadding {
  if (typeof window === "undefined") return DEFAULT_ACTIVITY_PADDING;
  let value: string | null;
  try {
    value = window.localStorage.getItem(ACTIVITY_PADDING_STORAGE_KEY);
  } catch {
    value = fallbackPaddingValue;
  }
  if (value === cachedPaddingValue) return cachedActivityPadding;
  cachedPaddingValue = value;
  try {
    const saved = JSON.parse(value ?? "null") as {
      before?: unknown;
      after?: unknown;
      joinGap?: unknown;
    } | null;
    cachedActivityPadding = saved &&
      typeof saved.before === "number" && Number.isFinite(saved.before) &&
      typeof saved.after === "number" && Number.isFinite(saved.after)
      ? {
          before: Math.max(0, Math.min(8, saved.before)),
          after: Math.max(0, Math.min(8, saved.after)),
          joinGap: typeof saved.joinGap === "number" && Number.isFinite(saved.joinGap)
            ? Math.max(0, Math.min(10, saved.joinGap))
            : DEFAULT_JOIN_GAP_SECONDS,
        }
      : DEFAULT_ACTIVITY_PADDING;
  } catch {
    cachedActivityPadding = DEFAULT_ACTIVITY_PADDING;
  }
  return cachedActivityPadding;
}

function subscribeToActivityPadding(onChange: () => void): () => void {
  const handleStorage = (event: StorageEvent) => {
    if (event.key === ACTIVITY_PADDING_STORAGE_KEY) {
      cachedPaddingValue = undefined;
      onChange();
    }
  };
  const handleLocalChange = () => onChange();
  window.addEventListener("storage", handleStorage);
  window.addEventListener(ACTIVITY_PADDING_EVENT, handleLocalChange);
  return () => {
    window.removeEventListener("storage", handleStorage);
    window.removeEventListener(ACTIVITY_PADDING_EVENT, handleLocalChange);
  };
}

function saveActivityPadding(padding: ActivityPadding): void {
  const value = JSON.stringify(padding);
  fallbackPaddingValue = value;
  try {
    window.localStorage.setItem(ACTIVITY_PADDING_STORAGE_KEY, value);
  } catch {
    // The in-memory fallback keeps the sliders working for this tab.
  }
  cachedPaddingValue = undefined;
  window.dispatchEvent(new Event(ACTIVITY_PADDING_EVENT));
}

const demoRallies: Rally[] = [
  { id: "R01", start: 24, end: 37, confidence: 0.78, included: true },
  { id: "R02", start: 52, end: 71, confidence: 0.74, included: true },
  { id: "R03", start: 91, end: 104, confidence: 0.57, included: true },
];

const demoAnalysis: ReviewAnalysis = {
  id: "demo",
  recordingId: "demo",
  title: "Example analysis",
  variantLabel: "Demo",
  variantDescription: null,
  kind: "unknown",
  method: "demo",
  modelVersion: null,
  addedAt: null,
  trainingCorpus: "reference",
  trainingCorpusLabel: "Reference",
  datasetRole: "not-applicable",
  datasetRoleLabel: "Not applicable",
  duration: 150,
  width: 16,
  height: 9,
  sourceFilename: "Add a video to begin",
  videoUrl: null,
  courtPreviewUrl: null,
  courtConfidence: 0,
  courtSource: "demo",
  courtLines: [],
  cameraStability: 1,
  warnings: ["This is sample data. Run an analyzer to review a real recording."],
  rallies: demoRallies,
  ignoredIntervals: [],
};

type ReviewEditorProps = {
  initialAnalysis: ReviewAnalysis | null;
  analysisOptions: AnalysisOption[];
  videoOptions: ReviewVideoOption[];
  comparisonAnalyses: ReviewAnalysis[];
  initialTime: number;
  corpusView: TrainingCorpusView;
};

function tone(
  kind: AnalysisKind,
  trainingCorpus: TrainingCorpus,
): "model" | "model-no-beach" | "heuristic" | "sol" | "gold" {
  if (kind === "model" && trainingCorpus === "without-beach") {
    return "model-no-beach";
  }
  if (kind === "model" || kind === "sol" || kind === "gold") return kind;
  return "heuristic";
}

function visibleInCorpus(
  analysis: AnalysisOption,
  corpus: TrainingCorpusView,
): boolean {
  return (
    analysis.trainingCorpus === "reference" ||
    corpus === "both" ||
    analysis.trainingCorpus === corpus
  );
}

function preferredAnalysis(
  video: ReviewVideoOption,
  corpus: TrainingCorpusView,
): AnalysisOption | undefined {
  const analyses = video.analyses.filter((analysis) =>
    visibleInCorpus(analysis, corpus),
  );
  return (
    analyses.find(
      (analysis) =>
        analysis.id === `${PREFERRED_ENVIRONMENT_EXPERIMENT_MODEL_ID}--${video.id}`,
    ) ??
    analyses.find((analysis) => analysis.kind === "gold") ??
    analyses.find((analysis) => analysis.kind === "sol") ??
    analyses.find(
      (analysis) =>
        corpus === "without-beach" &&
        analysis.id === `model-nb-audiovisual-v2-final--${video.id}`,
    ) ??
    analyses.find(
      (analysis) =>
        analysis.id === `model-full-percentile-v1--${video.id}`,
    ) ??
    analyses.find(
      (analysis) =>
        analysis.kind === "model" && analysis.modelVersion === "full-percentile-v1",
    ) ??
    analyses.find((analysis) => analysis.kind === "model") ??
    analyses.find(
      (analysis) => analysis.kind === "heuristic" && analysis.id.endsWith("-v2"),
    ) ??
    analyses.find((analysis) => analysis.kind === "heuristic") ??
    analyses[0]
  );
}

function analysisDescription(kind: AnalysisKind): string {
  if (kind === "model") {
    return "The trained temporal classifier produced these live-play intervals. Compare them against the verified reference and other analyzers below.";
  }
  if (kind === "heuristic") {
    return "Court motion and supporting audio produced these no-model activity suggestions. Compare them against the trained model and verified reference.";
  }
  if (kind === "sol") {
    return "Blind audiovisual frame and audio review produced these Sol candidates before continuous human verification.";
  }
  if (kind === "gold") {
    return "These are the continuously reviewed serve-contact-to-dead-ball reference intervals used for training and evaluation.";
  }
  return "Review the selected rally suggestions against the shared timeline.";
}

function analysisSlug(analysis: Pick<ReviewAnalysis, "id" | "recordingId">): string {
  const recordingSuffix = `--${analysis.recordingId}`;
  if (analysis.id.endsWith(recordingSuffix)) {
    return analysis.id.slice(0, -recordingSuffix.length);
  }
  return analysis.id;
}

function modelVisibilityKey(
  analysis: Pick<ReviewAnalysis, "id" | "recordingId" | "trainingCorpus">,
): string {
  return `${analysis.trainingCorpus}:${analysisSlug(analysis)}`;
}

function metricPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function signedPercent(value: number | null): string | undefined {
  if (value === null) return undefined;
  const normalized = Math.abs(value) < 0.05 ? 0 : value;
  return `${normalized > 0 ? "+" : ""}${normalized.toFixed(1)}% vs padded human`;
}

function timelineTrackLabel(analysis: ReviewAnalysis): string {
  return analysis.kind === "model"
    ? analysis.variantLabel.replace(/^Trained model\s*[·\-–—]\s*/i, "")
    : analysis.variantLabel;
}

export function ReviewEditor({
  initialAnalysis,
  analysisOptions,
  videoOptions,
  comparisonAnalyses,
  initialTime,
  corpusView,
}: ReviewEditorProps) {
  const analysis = initialAnalysis ?? demoAnalysis;
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [rallies, setRallies] = useState(analysis.rallies);
  const [selectedId, setSelectedId] = useState(analysis.rallies[0]?.id ?? "");
  const activityPadding = useSyncExternalStore(
    subscribeToActivityPadding,
    activityPaddingSnapshot,
    () => DEFAULT_ACTIVITY_PADDING,
  );
  const preRoll = activityPadding.before;
  const postRoll = activityPadding.after;
  const joinGap = activityPadding.joinGap;
  const [playbackTime, setPlaybackTime] = useState(initialTime);
  const [isPlaying, setIsPlaying] = useState(false);
  const [modelFilterBasis, setModelFilterBasis] = useState<ModelFilterBasis>("coreHuman");
  const [modelFilterMetric, setModelFilterMetric] = useState<ModelFilterMetric>("f1");
  const [modelFilterOperator, setModelFilterOperator] = useState<ModelFilterOperator>("greater");
  const [modelFilterThreshold, setModelFilterThreshold] = useState("");
  const visibleModelKeys = useSyncExternalStore(
    subscribeToModelVisibility,
    visibleModelSnapshot,
    () => DEFAULT_VISIBLE_MODEL_KEYS,
  );
  const videoRef = useRef<HTMLVideoElement>(null);
  const intervals = useMemo(
    () => buildEditList(rallies, preRoll, postRoll, analysis.duration, joinGap),
    [rallies, preRoll, postRoll, analysis.duration, joinGap],
  );
  const selected = rallies.find((rally) => rally.id === selectedId) ?? rallies[0] ?? null;
  const keptSeconds = intervals.reduce(
    (total, interval) => total + interval.keptEnd - interval.keptStart,
    0,
  );
  const keptCount = rallies.filter((rally) => rally.included).length;
  const currentVideo =
    videoOptions.find((video) => video.id === analysis.recordingId) ?? null;
  const analysisIndex = analysisOptions.findIndex((option) => option.id === analysis.id);
  const modelAnalyses = useMemo(
    () => comparisonAnalyses.filter((candidate) => candidate.kind === "model"),
    [comparisonAnalyses],
  );
  const humanAnalysis = useMemo(
    () => comparisonAnalyses.find((candidate) => candidate.kind === "gold") ?? null,
    [comparisonAnalyses],
  );
  const [ignoredOverride, setIgnoredOverride] = useState<{
    recordingId: string;
    ranges: IgnoredInterval[];
  } | null>(null);
  const ignoredRanges = useMemo(
    () => {
      if (ignoredOverride && ignoredOverride.recordingId === humanAnalysis?.recordingId) {
        return ignoredOverride.ranges;
      }
      return humanAnalysis?.ignoredIntervals ?? [];
    },
    [humanAnalysis, ignoredOverride],
  );
  const [ignoreStart, setIgnoreStart] = useState<number | null>(null);
  const [ignoreReason, setIgnoreReason] = useState("non-game-content");
  const [ignoreSaving, setIgnoreSaving] = useState(false);
  const [ignoreMessage, setIgnoreMessage] = useState<string | null>(null);
  const [ignoreError, setIgnoreError] = useState<string | null>(null);
  const visibleComparisonAnalyses = useMemo(
    () =>
      comparisonAnalyses.filter(
        (candidate) =>
          candidate.kind !== "model" ||
          visibleModelKeys.has(modelVisibilityKey(candidate)),
      ),
    [comparisonAnalyses, visibleModelKeys],
  );
  const humanRallies = useMemo(
    () => humanAnalysis?.rallies ?? [],
    [humanAnalysis],
  );
  const paddedModelRallies = useMemo(
    () => new Map(
      modelAnalyses.map((candidate) => [
        candidate.id,
        padAndMergeRallies(
          candidate.id === analysis.id ? rallies : candidate.rallies,
          preRoll,
          postRoll,
          candidate.duration,
          joinGap,
        ),
      ]),
    ),
    [analysis.id, joinGap, modelAnalyses, postRoll, preRoll, rallies],
  );
  const paddedHumanRallies = useMemo(
    () => padAndMergeRallies(
      humanRallies,
      preRoll,
      postRoll,
      analysis.duration,
      joinGap,
    ),
    [analysis.duration, humanRallies, joinGap, postRoll, preRoll],
  );
  const paddedHumanSeconds = useMemo(
    () => totalRallySeconds(excludeIgnoredTime(paddedHumanRallies, ignoredRanges)),
    [ignoredRanges, paddedHumanRallies],
  );
  const modelTimelineStats = useMemo(
    () => new Map(
      modelAnalyses.map((candidate) => {
        const modelRallies = paddedModelRallies.get(candidate.id) ?? [];
        const evaluatedModelRallies = excludeIgnoredTime(modelRallies, ignoredRanges);
        return [
          candidate.id,
          {
            videoSeconds: totalRallySeconds(evaluatedModelRallies),
            coreHuman: calculateLiveTimeMetrics(modelRallies, humanRallies, ignoredRanges),
            paddedHuman: calculateLiveTimeMetrics(modelRallies, paddedHumanRallies, ignoredRanges),
          },
        ] as const;
      }),
    ),
    [humanRallies, ignoredRanges, modelAnalyses, paddedHumanRallies, paddedModelRallies],
  );
  const modelToggleAnalyses = useMemo(() => {
    if (modelFilterThreshold.trim() === "") return modelAnalyses;
    const threshold = Number(modelFilterThreshold);
    if (!Number.isFinite(threshold)) return modelAnalyses;
    return modelAnalyses.filter((candidate) => {
      const stats = modelTimelineStats.get(candidate.id);
      if (!stats) return false;
      const value = stats[modelFilterBasis][modelFilterMetric] * 100;
      return modelFilterOperator === "greater"
        ? value > threshold
        : value < threshold;
    });
  }, [
    modelAnalyses,
    modelFilterBasis,
    modelFilterMetric,
    modelFilterOperator,
    modelFilterThreshold,
    modelTimelineStats,
  ]);
  const tracks = useMemo<TimelineTrack[]>(
    () => {
      const comparisonTracks: TimelineTrack[] = visibleComparisonAnalyses.map((candidate) => {
        const modelCoreRallies = candidate.kind === "model"
          ? candidate.id === analysis.id ? rallies : candidate.rallies
          : [];
        const candidateCoreRallies = candidate.id === analysis.id ? rallies : candidate.rallies;
        const paddedExportRallies = candidate.kind === "model"
          ? paddedModelRallies.get(candidate.id) ?? []
          : padAndMergeRallies(
              candidateCoreRallies,
              preRoll,
              postRoll,
              candidate.duration,
              joinGap,
            );
        const trackRallies = candidate.kind === "model"
          ? paddedExportRallies
          : candidateCoreRallies;
        const comparisonSegments = candidate.kind === "model" && humanRallies.length > 0
          ? markModelPaddingOrigins(
              buildLiveTimeComparisonSegments(trackRallies, humanRallies, ignoredRanges),
              modelCoreRallies,
              preRoll,
              postRoll,
              candidate.duration,
            )
          : null;
        const exportedRallies = excludeIgnoredTime(paddedExportRallies, ignoredRanges);
        const joinedGapRallies = excludeIgnoredTime(
          paddedExportRallies.flatMap((rally, rallyIndex) =>
            rally.joinedGaps.map((gap, gapIndex) => ({
              id: `${candidate.id}-joined-gap-${rallyIndex + 1}-${gapIndex + 1}`,
              start: gap.start,
              end: gap.end,
              confidence: 1,
              included: true,
            }))),
          ignoredRanges,
        );
        const missingHumanSegments = candidate.id === humanAnalysis?.id || humanRallies.length === 0
          ? []
          : (comparisonSegments ?? buildLiveTimeComparisonSegments(
              exportedRallies,
              humanRallies,
              ignoredRanges,
            )).filter((segment) => segment.kind === "missed");
        const slug = analysisSlug(candidate);
        const stats = modelTimelineStats.get(candidate.id);
        const coreMetrics = stats
          ? `P ${metricPercent(stats.coreHuman.precision)} · R ${metricPercent(stats.coreHuman.recall)} · F1 ${metricPercent(stats.coreHuman.f1)}`
          : "P — · R — · F1 —";
        const paddedMetrics = stats
          ? `P ${metricPercent(stats.paddedHuman.precision)} · R ${metricPercent(stats.paddedHuman.recall)} · F1 ${metricPercent(stats.paddedHuman.f1)}`
          : "P — · R — · F1 —";
        const exportDelta = stats
          ? signedPercent(calculateDurationDeltaPercent(stats.videoSeconds, paddedHumanSeconds))
          : undefined;
        const hybridF1 = stats
          ? metricPercent(calculateF1(stats.paddedHuman.precision, stats.coreHuman.recall))
          : undefined;
        return {
          id: candidate.id,
          label: timelineTrackLabel(candidate),
          title: [
            candidate.variantLabel,
            candidate.variantDescription,
            candidate.kind === "model" ? `Slug: ${slug}` : null,
            candidate.addedAt ? `Added: ${new Date(candidate.addedAt).toLocaleString()}` : null,
          ].filter(Boolean).join("\n") || undefined,
          detail: candidate.kind === "model"
            ? slug
            : `${candidate.trainingCorpusLabel} · ${candidate.datasetRoleLabel} · ${trackRallies.length} rallies`,
          summary: candidate.kind === "model"
            ? {
                exportTime: stats ? formatTime(stats.videoSeconds) : "—",
                exportDelta,
                coreMetrics,
                paddedMetrics,
                hybridF1,
              }
            : candidate.kind === "gold"
              ? { exportTime: formatTime(paddedHumanSeconds) }
            : undefined,
          active: candidate.id === analysis.id,
          exportIntervals: exportedRallies.map((rally, index) => ({
            id: `${candidate.id}-export-${index + 1}`,
            start: rally.start,
            end: rally.end,
            title: `${candidate.variantLabel} · exported with ${preRoll}s before and ${postRoll}s after · ${formatTime(rally.start)}–${formatTime(rally.end)}`,
          })),
          joinedGapIntervals: joinedGapRallies.map((gap, index) => ({
            id: `${candidate.id}-joined-gap-visible-${index + 1}`,
            start: gap.start,
            end: gap.end,
            title: `${candidate.variantLabel} · retained because the cut gap is under ${joinGap}s · ${formatTime(gap.start)}–${formatTime(gap.end)}`,
          })),
          missingHumanIntervals: missingHumanSegments.map((segment, index) => ({
            id: `${candidate.id}-missing-human-${index + 1}`,
            start: segment.start,
            end: segment.end,
            title: `${candidate.variantLabel} · missed unpadded human rally time · ${formatTime(segment.start)}–${formatTime(segment.end)}`,
          })),
          intervals: comparisonSegments
            ? comparisonSegments.map((segment) => ({
              id: segment.id,
              selectionId: segment.predictionId ?? null,
              start: segment.start,
              end: segment.end,
              tone: `model-${segment.kind}` as const,
              paddingOrigin: segment.paddingOrigin,
              title: `${candidate.variantLabel} · ${
                segment.kind === "match"
                  ? "matches human live time"
                  : segment.kind === "added"
                    ? "predicted outside human live time"
                    : "human live time missed by model"
              }${segment.paddingOrigin
                ? ` · ${segment.paddingOrigin === "both" ? "before + after padding" : `${segment.paddingOrigin} padding`}`
                : " · model core"} · ${formatTime(segment.start)}–${formatTime(segment.end)}`,
            }))
            : [
                ...(
                  candidate.kind === "gold" || candidate.kind === "sol"
                    ? buildPaddingSegments(
                        trackRallies,
                        preRoll,
                        postRoll,
                        candidate.duration,
                      ).map((segment) => ({
                        id: `${candidate.kind}-${segment.id}`,
                        selectionId: null,
                        start: segment.start,
                        end: segment.end,
                        tone: candidate.kind === "gold"
                          ? "gold-padding" as const
                          : "sol-padding" as const,
                        title: `${candidate.variantLabel} · activity padding · ${formatTime(segment.start)}–${formatTime(segment.end)}`,
                      }))
                    : []
                ),
                ...trackRallies.map((rally) => ({
                  id: rally.id,
                  start: rally.start,
                  end: rally.end,
                  confidence: rally.confidence,
                  tone: tone(candidate.kind, candidate.trainingCorpus),
                  title: `${candidate.variantLabel} · ${rally.id} · ${formatTime(rally.start)}–${formatTime(rally.end)}`,
                })),
              ],
        };
      });
      if (ignoredRanges.length > 0) {
        const ignoredTrack: TimelineTrack = {
          id: "ignored-evaluation-ranges",
          label: "Ignored evaluation",
          detail: `${ignoredRanges.length} excluded ${ignoredRanges.length === 1 ? "range" : "ranges"}`,
          intervals: ignoredRanges.map((range, index) => ({
            id: `ignored-${index + 1}`,
            selectionId: null,
            start: range.start,
            end: range.end,
            tone: "ignored",
            title: `Ignored · ${range.reason} · ${formatTime(range.start)}–${formatTime(range.end)}`,
          })),
        };
        const goldIndex = comparisonTracks.findIndex((track) => track.id === humanAnalysis?.id);
        comparisonTracks.splice(goldIndex >= 0 ? goldIndex + 1 : 0, 0, ignoredTrack);
      }
      return comparisonTracks;
    },
    [
      analysis.id,
      humanAnalysis?.id,
      humanRallies,
      ignoredRanges,
      joinGap,
      modelTimelineStats,
      paddedModelRallies,
      paddedHumanSeconds,
      postRoll,
      preRoll,
      rallies,
      visibleComparisonAnalyses,
    ],
  );

  async function mutateIgnoredRanges(
    mutate: (document: LabelDocument) => IgnoredInterval[],
    successMessage: string,
  ) {
    if (!humanAnalysis) return;
    setIgnoreSaving(true);
    setIgnoreError(null);
    setIgnoreMessage(null);
    try {
      const taskUrl = `/api/labeling/tasks/${encodeURIComponent(humanAnalysis.recordingId)}`;
      const response = await fetch(taskUrl, { cache: "no-store" });
      if (!response.ok) throw new Error("Could not load the current label document");
      const document = parseLabelDocument(await response.json());
      const nextRanges = mutate(document).sort(
        (left, right) => left.start - right.start || left.end - right.end,
      );
      const draft: LabelDocument = {
        ...document,
        annotation: {
          ...document.annotation,
          status: "in-progress",
          reviewedAt: null,
        },
        ignoredIntervals: nextRanges,
      };
      const saveResponse = await fetch(`${taskUrl}/draft`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });
      const result = await saveResponse.json() as { error?: string; savedAt?: string };
      if (!saveResponse.ok || !result.savedAt) {
        throw new Error(result.error ?? "The ignored ranges could not be saved");
      }
      setIgnoredOverride({
        recordingId: humanAnalysis.recordingId,
        ranges: nextRanges,
      });
      setIgnoreStart(null);
      setIgnoreMessage(`${successMessage} Saved ${new Date(result.savedAt).toLocaleTimeString()}.`);
    } catch (error) {
      setIgnoreError(error instanceof Error ? error.message : "The ignored ranges could not be saved");
    } finally {
      setIgnoreSaving(false);
    }
  }

  function finishIgnoredRange() {
    if (ignoreStart === null) return;
    const start = Math.min(ignoreStart, playbackTime);
    const end = Math.max(ignoreStart, playbackTime);
    if (end - start < 0.1) {
      setIgnoreError("An ignored range must be at least 0.1 seconds long.");
      return;
    }
    void mutateIgnoredRanges((document) => {
      const overlaps = (row: { start: number; end: number }) =>
        row.start < end && row.end > start;
      if (document.rallies.some(overlaps)) {
        throw new Error("Ignored ranges cannot overlap a human-labeled rally.");
      }
      if (document.hardNegatives.some(overlaps)) {
        throw new Error("Ignored ranges cannot overlap a hard-negative label.");
      }
      if (document.ignoredIntervals.some(overlaps)) {
        throw new Error("This range overlaps an existing ignored range.");
      }
      return [
        ...document.ignoredIntervals,
        { start, end, reason: ignoreReason },
      ];
    }, `Ignored ${formatTime(start)}–${formatTime(end)} for evaluation.`);
  }

  function removeIgnoredRange(range: IgnoredInterval) {
    void mutateIgnoredRanges(
      (document) => document.ignoredIntervals.filter(
        (candidate) => candidate.start !== range.start || candidate.end !== range.end,
      ),
      `Restored ${formatTime(range.start)}–${formatTime(range.end)} to evaluation.`,
    );
  }

  function toggleModelVisibility(candidate: ReviewAnalysis) {
    const key = modelVisibilityKey(candidate);
    const next = new Set(visibleModelKeys);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    saveModelVisibility(next);
  }

  function navigate(
    videoId: string,
    analysisId: string,
    time = 0,
    corpus = corpusView,
  ) {
    const query = new URLSearchParams({
      video: videoId,
      analysis: analysisId,
      corpus,
    });
    if (time > 0) query.set("time", time.toFixed(3));
    startTransition(() => router.push(`/?${query.toString()}`));
  }

  function selectVideo(id: string) {
    const video = videoOptions.find((candidate) => candidate.id === id);
    const next = video ? preferredAnalysis(video, corpusView) : undefined;
    if (video && next) navigate(video.id, next.id);
  }

  function selectCorpus(nextCorpus: TrainingCorpusView) {
    if (!currentVideo || nextCorpus === corpusView) return;
    const eligible = currentVideo.analyses.filter((candidate) =>
      visibleInCorpus(candidate, nextCorpus),
    );
    const current = eligible.find((candidate) => candidate.id === analysis.id);
    const sameVariant = eligible.find(
      (candidate) =>
        candidate.modelVersion === analysis.modelVersion &&
        candidate.variantLabel === analysis.variantLabel,
    );
    const next = current ?? sameVariant ?? preferredAnalysis(currentVideo, nextCorpus);
    if (next) navigate(currentVideo.id, next.id, playbackTime, nextCorpus);
  }

  function selectAnalysis(id: string, time = playbackTime) {
    if (!currentVideo || !id || id === analysis.id) return;
    navigate(currentVideo.id, id, time);
  }

  function seekTo(time: number, intervalId?: string) {
    const clamped = Math.max(0, Math.min(analysis.duration, time));
    setPlaybackTime(clamped);
    if (intervalId) setSelectedId(intervalId);
    if (videoRef.current) videoRef.current.currentTime = clamped;
  }

  function toggleRally(id: string) {
    setRallies((current) =>
      current.map((rally) =>
        rally.id === id ? { ...rally, included: !rally.included } : rally,
      ),
    );
  }

  async function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) await video.play();
    else video.pause();
  }

  const confidenceLabel =
    analysis.kind === "gold"
      ? "VERIFICATION"
      : analysis.kind === "model"
        ? "MODEL SCORE"
        : analysis.kind === "sol"
          ? "SOL CONFIDENCE"
          : "HEURISTIC CONFIDENCE";

  return (
    <main>
      <header className="topbar">
        <Brand className="brand" label="LAB" priority />
        <div className="top-actions">
          <Link href="/on-device">Try local browser cut →</Link>
          <Link href="/edit">Open cut editor →</Link>
          <Link href="/label">Open labeling station →</Link>
          <div className="project-state">
            <i /> {initialAnalysis ? `${videoOptions.length} videos ready` : "Demo mode"}
          </div>
        </div>
      </header>

      {videoOptions.length > 0 && (
        <nav className="analysis-picker" aria-label="Video and analysis selection">
          <div>
            <span className="picker-kicker">COMPARISON DATASET</span>
            <strong>Choose video and analyzer</strong>
          </div>
          <div className="picker-fields">
            <label>
              <span>Training corpus</span>
              <select
                aria-label="Training corpus"
                value={corpusView}
                disabled={isPending}
                onChange={(event) =>
                  selectCorpus(event.target.value as TrainingCorpusView)
                }
              >
                <option value="original">Original training</option>
                <option value="without-beach">Without beach</option>
                <option value="both">Both corpora</option>
              </select>
            </label>
            <label>
              <span>Video</span>
              <select
                aria-label="Video"
                value={analysis.recordingId}
                disabled={isPending}
                onChange={(event) => selectVideo(event.target.value)}
              >
                {videoOptions.map((video) => (
                  <option key={video.id} value={video.id}>
                    {video.environment} · {video.title}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Analysis source</span>
              <select
                aria-label="Analysis source"
                title={analysis.variantDescription ?? undefined}
                value={analysis.id}
                disabled={isPending}
                onChange={(event) => selectAnalysis(event.target.value)}
              >
                {analysisOptions.map((option) => (
                  <option
                    key={option.id}
                    value={option.id}
                    title={option.variantDescription ?? undefined}
                  >
                    {option.variantLabel} · {option.trainingCorpusLabel} · {option.datasetRoleLabel} · {option.rallyCount}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="analysis-position" aria-live="polite">
            <strong>{Math.max(0, analysisIndex) + 1}</strong>
            <span>/ {analysisOptions.length}</span>
            <small>{isPending ? "LOADING…" : analysis.recordingId}</small>
          </div>
        </nav>
      )}

      <section className="hero">
        <div>
          <div className="analysis-badges">
            <span
              data-kind={analysis.kind}
              title={analysis.variantDescription ?? undefined}
            >
              {analysis.variantLabel}
            </span>
            <span data-role={analysis.datasetRole}>{analysis.datasetRoleLabel}</span>
            <span data-corpus={analysis.trainingCorpus}>
              {analysis.trainingCorpusLabel}
            </span>
          </div>
          <h1>Find the rallies.<br /><em>Compare the evidence.</em></h1>
          <p className="intro">{analysisDescription(analysis.kind)}</p>
          <p className="source-name">SOURCE / {analysis.sourceFilename}</p>
        </div>
        <div className="scorecard">
          <span>RALLIES KEPT</span>
          <strong>{keptCount}<small> / {rallies.length}</small></strong>
          <div className="progress">
            <b style={{ width: `${rallies.length ? (keptCount / rallies.length) * 100 : 0}%` }} />
          </div>
          <p>{formatTime(keptSeconds)} kept from {formatTime(analysis.duration)}</p>
        </div>
      </section>

      {analysis.warnings.length > 0 && (
        <section className="warnings" aria-label="Analysis warnings">
          {analysis.warnings.map((warning) => <p key={warning}>⚑ {warning}</p>)}
        </section>
      )}

      <section className="workspace">
        <aside>
          {selected ? (
            <>
              <div className="aside-heading"><span>SELECTED RALLY</span><strong>{selected.id}</strong></div>
              <div className={`confidence ${selected.confidence < 0.7 ? "warn" : ""}`}>
                <span>{confidenceLabel}</span>
                <strong>{analysis.kind === "gold" ? "VERIFIED" : `${Math.round(selected.confidence * 100)}%`}</strong>
              </div>
              <dl>
                <div><dt>Suggested start</dt><dd>{formatTime(selected.start)}</dd></div>
                <div><dt>Suggested end</dt><dd>{formatTime(selected.end)}</dd></div>
                <div><dt>Core duration</dt><dd>{(selected.end - selected.start).toFixed(1)}s</dd></div>
                <div><dt>Dataset role</dt><dd>{analysis.datasetRoleLabel}</dd></div>
                <div><dt>Training corpus</dt><dd>{analysis.trainingCorpusLabel}</dd></div>
                <div><dt>Method</dt><dd>{analysis.method}</dd></div>
              </dl>
              <div className="selected-rally-actions">
                <button className="include" onClick={() => toggleRally(selected.id)}>
                  {selected.included ? "✓ Included in export" : "+ Restore to export"}
                </button>
                {analysis.courtPreviewUrl && (
                  <a className="diagnostic-link" href={analysis.courtPreviewUrl} target="_blank">
                    Open court diagnostic ↗
                  </a>
                )}
              </div>
            </>
          ) : (
            <div className="empty-state"><span>NO RALLIES FOUND</span><p>This source produced no rally candidates for the selected video.</p></div>
          )}
        </aside>
        <div className="viewer">
          <div
            className={`court video-stage ${analysis.videoUrl ? "has-video" : ""}`}
            style={{ aspectRatio: `${analysis.width} / ${analysis.height}` }}
          >
            {analysis.videoUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={analysis.videoUrl}
                  preload="metadata"
                  controls
                  onLoadedMetadata={(event) => {
                    event.currentTarget.currentTime = initialTime;
                    setPlaybackTime(initialTime);
                  }}
                  onTimeUpdate={(event) => setPlaybackTime(event.currentTarget.currentTime)}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                />
                <svg className="court-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                  {analysis.courtLines.map((line, index) => (
                    <line key={index} x1={line.x1 * 100} y1={line.y1 * 100} x2={line.x2 * 100} y2={line.y2 * 100} />
                  ))}
                </svg>
              </>
            ) : (
              <div className="court-lines"><span /><span /><span /></div>
            )}
            <button
              className={`play ${isPlaying ? "playing" : ""}`}
              aria-label={isPlaying ? "Pause preview" : "Play preview"}
              onClick={togglePlayback}
              disabled={!analysis.videoUrl}
            >
              {isPlaying ? "Ⅱ" : "▶"}
            </button>
            <div className="timecode">{formatTime(playbackTime)} <span>/ {formatTime(analysis.duration)}</span></div>
            <div className="court-status">
              COURT / {analysis.courtSource} / {analysis.courtSource === "manual-roi"
                ? "PROVIDED"
                : `${Math.round(analysis.courtConfidence * 100)}%`}
            </div>
          </div>

          {humanAnalysis && (
            <div className="ignore-range-editor" aria-label="Ignored evaluation ranges">
              <div className="ignore-range-heading">
                <div>
                  <strong>Ignored evaluation ranges</strong>
                  <span>Excluded from model scoring and export-duration comparisons</span>
                </div>
                <span>{ignoredRanges.length} {ignoredRanges.length === 1 ? "range" : "ranges"}</span>
              </div>
              <div className="ignore-range-actions">
                <label>
                  <span>Reason</span>
                  <select
                    value={ignoreReason}
                    onChange={(event) => setIgnoreReason(event.target.value)}
                    disabled={ignoreSaving}
                  >
                    <option value="partial-rally">Partial rally</option>
                    <option value="camera-gap">Camera gap</option>
                    <option value="boundary-ambiguous">Boundary ambiguous</option>
                    <option value="non-game-content">Non-game content</option>
                  </select>
                </label>
                {ignoreStart === null ? (
                  <button
                    type="button"
                    onClick={() => {
                      setIgnoreStart(playbackTime);
                      setIgnoreError(null);
                      setIgnoreMessage(null);
                    }}
                    disabled={ignoreSaving}
                  >
                    Mark start at {formatTime(playbackTime)}
                  </button>
                ) : (
                  <>
                    <span className="ignore-range-pending">
                      Start {formatTime(ignoreStart)} · current {formatTime(playbackTime)}
                    </span>
                    <button type="button" onClick={finishIgnoredRange} disabled={ignoreSaving}>
                      {ignoreSaving ? "Saving…" : "Mark end & save"}
                    </button>
                    <button
                      type="button"
                      className="quiet"
                      onClick={() => setIgnoreStart(null)}
                      disabled={ignoreSaving}
                    >
                      Cancel
                    </button>
                  </>
                )}
                <Link href="/label">
                  Open full label editor ↗
                </Link>
              </div>
              {ignoredRanges.length > 0 && (
                <div className="ignored-range-list">
                  {ignoredRanges.map((range) => (
                    <div key={`${range.start}-${range.end}-${range.reason}`}>
                      <button
                        type="button"
                        className="ignored-range-time"
                        onClick={() => seekTo(range.start)}
                      >
                        {formatTime(range.start)}–{formatTime(range.end)}
                      </button>
                      <span>{range.reason.replaceAll("-", " ")}</span>
                      <button
                        type="button"
                        className="ignored-range-delete"
                        aria-label={`Delete ignored range ${formatTime(range.start)} to ${formatTime(range.end)}`}
                        onClick={() => removeIgnoredRange(range)}
                        disabled={ignoreSaving}
                      >
                        Delete
                      </button>
                    </div>
                  ))}
                </div>
              )}
              {ignoreError && <p className="ignore-range-error">{ignoreError}</p>}
              {ignoreMessage && <p className="ignore-range-message">{ignoreMessage}</p>}
            </div>
          )}

          <div className="comparison-heading">
            <div>
              <strong>All analysis tracks</strong>
              <span>Same video clock · vertical line is the current playhead</span>
            </div>
            <span>
              {visibleComparisonAnalyses.length} / {comparisonAnalyses.length} sources shown
              {ignoredRanges.length > 0 ? ` · ${ignoredRanges.length} ignored` : ""}
            </span>
          </div>
          {modelAnalyses.length > 0 && (
            <div className="model-timeline-controls" aria-label="Model timeline visibility">
              <div className="model-timeline-legend" aria-label="Model comparison legend">
                <span data-tone="match">Human match</span>
                <span data-tone="added">Added rally</span>
                <span data-tone="missed">Missed rally</span>
                <span data-tone="gold-padding">Human padding</span>
                <span data-tone="sol-padding">Sol padding</span>
                <span data-tone="ignored">Ignored evaluation</span>
                <span data-tone="before-padding">Before padding</span>
                <span data-tone="after-padding">After padding</span>
                <span data-tone="export">Final padded export</span>
                <span data-tone="joined-gap">Joined short gap</span>
                <span data-tone="export-missed">Missed human core</span>
              </div>
              <div className="model-toggle-filter" aria-label="Filter model toggles">
                <span>Show toggles where</span>
                <label>
                  <span className="sr-only">Human label comparison</span>
                  <select
                    aria-label="Human label comparison"
                    value={modelFilterBasis}
                    onChange={(event) => setModelFilterBasis(event.target.value as ModelFilterBasis)}
                  >
                    <option value="coreHuman">Core human</option>
                    <option value="paddedHuman">Padded human</option>
                  </select>
                </label>
                <label>
                  <span className="sr-only">Metric</span>
                  <select
                    aria-label="Metric"
                    value={modelFilterMetric}
                    onChange={(event) => setModelFilterMetric(event.target.value as ModelFilterMetric)}
                  >
                    <option value="precision">Precision</option>
                    <option value="recall">Recall</option>
                    <option value="f1">F1</option>
                  </select>
                </label>
                <label>
                  <span className="sr-only">Comparison operator</span>
                  <select
                    className="model-filter-operator"
                    aria-label="Comparison operator"
                    value={modelFilterOperator}
                    onChange={(event) => setModelFilterOperator(event.target.value as ModelFilterOperator)}
                  >
                    <option value="greater">&gt;</option>
                    <option value="less">&lt;</option>
                  </select>
                </label>
                <label className="model-filter-threshold">
                  <span className="sr-only">Minimum percentage</span>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    inputMode="decimal"
                    placeholder="x"
                    aria-label="Minimum percentage"
                    value={modelFilterThreshold}
                    onChange={(event) => setModelFilterThreshold(event.target.value)}
                  />
                  <span>%</span>
                </label>
                <small>{modelToggleAnalyses.length} / {modelAnalyses.length} models</small>
                {modelFilterThreshold !== "" && (
                  <button type="button" onClick={() => setModelFilterThreshold("")}>Clear</button>
                )}
              </div>
              <div className="model-toggles">
                {modelToggleAnalyses.map((candidate) => {
                  const slug = analysisSlug(candidate);
                  const checked = visibleModelKeys.has(modelVisibilityKey(candidate));
                  return (
                    <label
                      key={candidate.id}
                      title={[
                        candidate.variantLabel,
                        slug,
                        candidate.variantDescription,
                      ].filter(Boolean).join("\n")}
                    >
                      <input
                        type="checkbox"
                        aria-label={`${checked ? "Hide" : "Show"} model ${slug}`}
                        checked={checked}
                        onChange={() => toggleModelVisibility(candidate)}
                      />
                      <span className="model-toggle-copy">
                        <strong>{timelineTrackLabel(candidate)}</strong>
                        <small>{slug}</small>
                      </span>
                    </label>
                  );
                })}
                {modelToggleAnalyses.length === 0 && (
                  <p className="model-toggle-empty">No models exceed this threshold.</p>
                )}
              </div>
            </div>
          )}
          <RallyTimeline
            duration={analysis.duration}
            currentTime={playbackTime}
            tracks={tracks}
            selectedTrackId={analysis.id}
            selectedIntervalId={selectedId}
            onTrackSelect={(trackId) => selectAnalysis(trackId)}
            onSeek={(time, trackId, intervalId) => {
              if (trackId !== analysis.id) {
                if (currentVideo) navigate(currentVideo.id, trackId, time);
                return;
              }
              seekTo(time, intervalId);
            }}
            ariaLabel="Comparison of verified, heuristic, Sol, and trained-model rallies"
          />
        </div>

      </section>

      <section className="controls">
        <div><p className="eyebrow">EDIT DECISION LIST</p><h2>Give every point<br />room to breathe.</h2></div>
        <label>Before activity <output>{preRoll}s</output><input type="range" min="0" max="8" value={preRoll} onChange={(event) => saveActivityPadding({ ...activityPadding, before: Number(event.target.value) })} /></label>
        <label>After activity <output>{postRoll}s</output><input type="range" min="0" max="8" value={postRoll} onChange={(event) => saveActivityPadding({ ...activityPadding, after: Number(event.target.value) })} /></label>
        <label>Join gaps under <output>{joinGap}s</output><input type="range" min="0" max="10" step="0.5" value={joinGap} onChange={(event) => saveActivityPadding({ ...activityPadding, joinGap: Number(event.target.value) })} /></label>
        <div className="export"><span>ESTIMATED EXPORT</span><strong>{formatTime(keptSeconds)}</strong><button disabled>Export coming next</button></div>
      </section>
    </main>
  );
}
