import { readFile } from "node:fs/promises";
import path from "node:path";
import type { Metadata } from "next";

import {
  getServingSideReviewReportPath,
  loadServingSideReviewState,
} from "@/lib/server/serving-side-review";

import { ServingSideReviewClient } from "./serving-side-review-client";
import type { ServingRecording, ServingReport } from "./types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const metadata: Metadata = {
  title: "Serving-side review · VolleySplice",
  description:
    "Review interpretable serving-side evidence across the NAS video corpus.",
  robots: { index: false, follow: false },
};

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function loadRecordings(
  report: ServingReport,
): Promise<ServingRecording[]> {
  const rallyEnds = new Map<string, number>();
  for (const rally of report.rallies) {
    rallyEnds.set(
      rally.recordingId,
      Math.max(rallyEnds.get(rally.recordingId) ?? 0, rally.end, rally.start),
    );
  }
  return Promise.all(
    report.labels.files.map(async (file) => {
      const fallbackDuration = (rallyEnds.get(file.recordingId) ?? 0) + 5;
      if (file.durationSeconds && file.videoFilename) {
        return {
          recordingId: file.recordingId,
          environment: file.environment,
          durationSeconds: file.durationSeconds,
          videoFilename: file.videoFilename,
          sourceType: file.sourceType,
          targetStatus: file.targetStatus,
        };
      }
      const labelPath = file.path
        ? path.resolve(report.labels.directory, path.basename(file.path))
        : "";
      try {
        const payload = JSON.parse(
          await readFile(labelPath, "utf8"),
        ) as unknown;
        const recording =
          isRecord(payload) && isRecord(payload.recording)
            ? payload.recording
            : null;
        return {
          recordingId: file.recordingId,
          environment: file.environment,
          durationSeconds: finiteNumber(
            recording?.durationSeconds,
            fallbackDuration,
          ),
          videoFilename:
            typeof recording?.videoFilename === "string"
              ? recording.videoFilename
              : (file.videoFilename ?? `${file.recordingId}.mp4`),
          sourceType: file.sourceType,
          targetStatus: file.targetStatus,
        };
      } catch {
        return {
          recordingId: file.recordingId,
          environment: file.environment,
          durationSeconds: fallbackDuration,
          videoFilename: file.videoFilename ?? `${file.recordingId}.mp4`,
          sourceType: file.sourceType,
          targetStatus: file.targetStatus,
        };
      }
    }),
  );
}

function clientReport(report: ServingReport): ServingReport {
  return {
    ...report,
    labels: {
      ...report.labels,
      directory: path.basename(report.labels.directory),
      manifest: report.labels.manifest
        ? path.basename(report.labels.manifest)
        : null,
      files: report.labels.files.map((file) => ({
        ...file,
        path: file.path ? path.basename(file.path) : null,
        videoPath: file.videoPath ? path.basename(file.videoPath) : undefined,
      })),
    },
  };
}

export default async function ServingSideReviewPage() {
  const sourcePath = getServingSideReviewReportPath();
  try {
    const report = JSON.parse(
      await readFile(sourcePath, "utf8"),
    ) as ServingReport;
    const recordings = await loadRecordings(report);
    let initialDecisions: Record<string, "near" | "far" | "unclear"> = {};
    let initialSavedAt: string | null = null;
    let decisionLoadError: string | undefined;
    try {
      const state = await loadServingSideReviewState();
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
    return (
      <ServingSideReviewClient
        report={clientReport(report)}
        recordings={recordings}
        reportPath={path.basename(sourcePath)}
        initialDecisions={initialDecisions}
        initialSavedAt={initialSavedAt}
        decisionLoadError={decisionLoadError}
      />
    );
  } catch (error) {
    return (
      <ServingSideReviewClient
        report={null}
        recordings={[]}
        reportPath={sourcePath}
        loadError={error instanceof Error ? error.message : String(error)}
        initialDecisions={{}}
        initialSavedAt={null}
      />
    );
  }
}
