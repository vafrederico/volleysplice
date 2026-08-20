import { readFile } from "node:fs/promises";
import path from "node:path";
import type { Metadata } from "next";

import {
  getSideSwitchReviewReportPath,
  loadSideSwitchReviewState,
} from "@/lib/server/side-switch-review";
import { loadSideSwitchReviewProposalBundle } from "@/lib/server/side-switch-review-proposals";

import { SideSwitchReviewClient } from "./side-switch-review-client";
import type { AppearanceReport, SideSwitchRecording } from "./types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const metadata: Metadata = {
  title: "Side-switch model review · VolleyCut",
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
      const fallbackDuration =
        (eventsByRecording.get(file.recordingId) ?? 0) + 5;
      const labelPath = path.join(
        path.resolve(report.labels.directory),
        `${file.recordingId}.labels.json`,
      );
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
            recording?.durationSeconds ?? file.durationSeconds,
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
    return (
      <SideSwitchReviewClient
        report={clientReport(report)}
        recordings={recordings}
        proposalBundle={proposalBundle}
        reportPath={path.basename(sourcePath)}
        initialDecisions={initialDecisions}
        initialSavedAt={initialSavedAt}
        decisionLoadError={decisionLoadError}
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
      />
    );
  }
}
