import { readFile } from "node:fs/promises";
import path from "node:path";
import type { Metadata } from "next";

import {
  getSideSwitchReviewReportPath,
  loadFullVideoSideSwitchMarkerState,
  loadSideSwitchReviewState,
} from "@/lib/server/side-switch-review";
import { loadSideSwitchReviewProposalBundle } from "@/lib/server/side-switch-review-proposals";

import { SideSwitchReviewClient } from "./side-switch-review-client";
import type {
  AppearanceReport,
  FullVideoSideSwitchMarker,
  SideSwitchRecording,
} from "./types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const metadata: Metadata = {
  title: "Side-switch model review · VolleySplice",
  description:
    "Review V5, production-state V5, and V6 side-switch proposals against volleyball footage.",
  robots: { index: false, follow: false },
};

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function optionalFiniteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function boundedRange(
  value: unknown,
  startKey: string,
  endKey: string,
  duration: number,
): { start: number; end: number } | null {
  if (!isRecord(value)) return null;
  const start = optionalFiniteNumber(value[startKey]);
  const end = optionalFiniteNumber(value[endKey]);
  if (start === null || end === null || end <= start) return null;
  const boundedStart = Math.max(0, Math.min(duration, start));
  const boundedEnd = Math.max(0, Math.min(duration, end));
  return boundedEnd > boundedStart
    ? { start: boundedStart, end: boundedEnd }
    : null;
}

function recordingFromPayload(
  reportFile: AppearanceReport["labels"]["files"][number],
  payload: unknown,
  fallbackDuration: number,
): SideSwitchRecording {
  const root = isRecord(payload) ? payload : {};
  const labelRecording = isRecord(root.recording) ? root.recording : null;
  const source = isRecord(root.source) ? root.source : null;
  const sourceFile = source && isRecord(source.file) ? source.file : null;
  const sourceMedia = source && isRecord(source.media) ? source.media : null;
  const runtimeVariant = optionalString(source?.runtimeVariant);
  const annotation = isRecord(root.annotation) ? root.annotation : null;
  const durationSeconds = finiteNumber(
    labelRecording?.durationSeconds ??
      sourceMedia?.duration ??
      reportFile.durationSeconds,
    fallbackDuration,
  );
  const videoFilename =
    optionalString(labelRecording?.videoFilename) ??
    optionalString(sourceFile?.name) ??
    reportFile.videoFilename ??
    `${reportFile.recordingId}.mp4`;
  const continuousVideoReviewed =
    annotation?.continuousVideoReviewed === true ||
    reportFile.candidateSource?.continuousVideoReviewed === true;

  const sourceSideSwitches = Array.isArray(root.sideSwitches)
    ? root.sideSwitches.flatMap((value) => {
        if (!isRecord(value)) return [];
        const time = optionalFiniteNumber(value.time);
        if (time === null || time < 0 || time > durationSeconds) return [];
        const notes = optionalString(value.notes);
        return [{ time, ...(notes ? { notes } : {}) }];
      })
    : [];

  const gameWindowValue = source?.gameWindow;
  const gameWindowRange = boundedRange(
    gameWindowValue,
    "start",
    "end",
    durationSeconds,
  );
  const gameWindow = gameWindowRange
    ? { start: gameWindowRange.start, end: gameWindowRange.end }
    : null;

  const initialInference = isRecord(root.initialInference)
    ? root.initialInference
    : null;
  const productionModelRanges = Array.isArray(initialInference?.ranges)
    ? initialInference.ranges.flatMap((value, index) => {
        const range = boundedRange(value, "start", "end", durationSeconds);
        if (!range || !isRecord(value)) return [];
        return [
          {
            id: optionalString(value.id) ?? `model-${index + 1}`,
            ...range,
            confidence: optionalFiniteNumber(value.confidence),
            ...(optionalString(value.agreement)
              ? { agreement: optionalString(value.agreement) }
              : {}),
          },
        ];
      })
    : [];

  const corrections = isRecord(root.corrections) ? root.corrections : null;
  const productionEditorRanges = Array.isArray(corrections?.correctedRanges)
    ? corrections.correctedRanges.flatMap((value, index) => {
        if (!isRecord(value)) return [];
        const core = boundedRange(
          value,
          "coreStart",
          "coreEnd",
          durationSeconds,
        );
        const keep = boundedRange(
          value,
          "keepStart",
          "keepEnd",
          durationSeconds,
        );
        if (!core || !keep || typeof value.included !== "boolean") return [];
        return [
          {
            id: optionalString(value.id) ?? `editor-${index + 1}`,
            coreStart: core.start,
            coreEnd: core.end,
            keepStart: keep.start,
            keepEnd: keep.end,
            confidence: optionalFiniteNumber(value.confidence),
            included: value.included,
            ...(optionalString(value.origin)
              ? { origin: optionalString(value.origin) }
              : {}),
            ...(optionalString(value.agreement)
              ? { agreement: optionalString(value.agreement) }
              : {}),
          },
        ];
      })
    : [];

  const productionIgnoredIntervals = Array.isArray(
    corrections?.ignoredIntervals,
  )
    ? corrections.ignoredIntervals.flatMap((value, index) => {
        const range = boundedRange(value, "start", "end", durationSeconds);
        if (!range || !isRecord(value)) return [];
        return [
          {
            id: optionalString(value.id) ?? `ignored-${index + 1}`,
            ...range,
            ...(optionalString(value.reason)
              ? { reason: optionalString(value.reason) }
              : {}),
          },
        ];
      })
    : [];

  const productionFinalIntervals = Array.isArray(root.finalExportIntervals)
    ? root.finalExportIntervals.flatMap((value) => {
        const range = boundedRange(value, "start", "end", durationSeconds);
        if (!range || !isRecord(value)) return [];
        return [
          {
            ...range,
            cutIds: Array.isArray(value.cutIds)
              ? value.cutIds.filter(
                  (cutId): cutId is string => typeof cutId === "string",
                )
              : [],
          },
        ];
      })
    : [];

  return {
    recordingId: reportFile.recordingId,
    environment: reportFile.environment,
    durationSeconds,
    videoFilename,
    sourceType: reportFile.sourceType,
    targetStatus: reportFile.targetStatus,
    feedbackProducer: runtimeVariant
      ? runtimeVariant.startsWith("native-android")
        ? "android"
        : "production-web"
      : null,
    continuousVideoReviewed,
    gameWindow,
    sourceSideSwitches,
    productionModelRanges,
    productionEditorRanges,
    productionIgnoredIntervals,
    productionFinalIntervals,
  };
}

async function loadRecordings(
  report: AppearanceReport,
): Promise<SideSwitchRecording[]> {
  const eventsByRecording = new Map<string, number>();
  for (const event of report.events) {
    const current = eventsByRecording.get(event.recordingId) ?? 0;
    eventsByRecording.set(
      event.recordingId,
      Math.max(current, event.gapEnd, event.transitionTime),
    );
  }

  return Promise.all(
    report.labels.files.map(async (file) => {
      const fallbackDuration = finiteNumber(
        file.durationSeconds,
        (eventsByRecording.get(file.recordingId) ?? 0) + 5,
      );
      const sourcePath = file.path
        ? path.resolve(file.path)
        : path.join(
            path.resolve(report.labels.directory),
            `${file.recordingId}.labels.json`,
          );
      try {
        return recordingFromPayload(
          file,
          JSON.parse(await readFile(sourcePath, "utf8")) as unknown,
          fallbackDuration,
        );
      } catch (error) {
        return {
          ...recordingFromPayload(file, null, fallbackDuration),
          timelineLoadError:
            error instanceof Error ? error.message : String(error),
        };
      }
    }),
  );
}

function clientReport(report: AppearanceReport): AppearanceReport {
  return {
    ...report,
    labels: {
      ...report.labels,
      directory: path.basename(report.labels.directory),
      files: report.labels.files.map((file) => ({
        ...file,
        path: file.path ? path.basename(file.path) : null,
      })),
    },
  };
}

export default async function SideSwitchReviewPage() {
  const sourcePath = getSideSwitchReviewReportPath();
  try {
    const report = JSON.parse(
      await readFile(sourcePath, "utf8"),
    ) as AppearanceReport;
    const [recordings, proposalBundle] = await Promise.all([
      loadRecordings(report),
      loadSideSwitchReviewProposalBundle(
        new Set(report.events.map((event) => event.eventId)),
      ),
    ]);
    let initialDecisions: Record<string, "switch" | "no-switch" | "unclear"> =
      {};
    let initialSavedAt: string | null = null;
    let decisionLoadError: string | undefined;
    try {
      const state = await loadSideSwitchReviewState();
      if (
        state.reportKind === report.kind &&
        state.reportCreatedAt === report.createdAt
      ) {
        initialDecisions = state.decisions;
        initialSavedAt = state.savedAt;
      }
    } catch (error) {
      decisionLoadError =
        error instanceof Error ? error.message : String(error);
    }
    let initialMarkers: FullVideoSideSwitchMarker[] = [];
    let initialReviewedRecordingIds: string[] = [];
    let initialMarkersSavedAt: string | null = null;
    let markerLoadError: string | undefined;
    try {
      const state = await loadFullVideoSideSwitchMarkerState();
      if (
        state.reportKind === report.kind &&
        state.reportCreatedAt === report.createdAt
      ) {
        initialMarkers = state.markers;
        initialReviewedRecordingIds = state.reviewedRecordingIds;
        initialMarkersSavedAt = state.savedAt;
      }
    } catch (error) {
      markerLoadError = error instanceof Error ? error.message : String(error);
    }
    return (
      <SideSwitchReviewClient
        report={clientReport(report)}
        recordings={recordings}
        proposalBundle={proposalBundle}
        reportPath={path.basename(sourcePath)}
        initialDecisions={initialDecisions}
        initialSavedAt={initialSavedAt}
        decisionLoadError={decisionLoadError}
        initialMarkers={initialMarkers}
        initialReviewedRecordingIds={initialReviewedRecordingIds}
        initialMarkersSavedAt={initialMarkersSavedAt}
        markerLoadError={markerLoadError}
      />
    );
  } catch (error) {
    return (
      <SideSwitchReviewClient
        report={null}
        recordings={[]}
        proposalBundle={{ layers: [], byEventId: {}, errors: [] }}
        reportPath={sourcePath}
        loadError={error instanceof Error ? error.message : String(error)}
        initialDecisions={{}}
        initialSavedAt={null}
        initialMarkers={[]}
        initialReviewedRecordingIds={[]}
        initialMarkersSavedAt={null}
      />
    );
  }
}
