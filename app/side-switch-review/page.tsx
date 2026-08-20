import { readFile } from "node:fs/promises";
import path from "node:path";
import type { Metadata } from "next";

import { SideSwitchReviewClient } from "./side-switch-review-client";
import type {
  AppearanceReport,
  SideSwitchRecording,
} from "./types";

const DEFAULT_LABELING_WORKSPACE =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09";
const DEFAULT_REPORT = path.join(
  DEFAULT_LABELING_WORKSPACE,
  "reports",
  "side-switch",
  "appearance-diagnostic-all-v1.json",
);

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const metadata: Metadata = {
  title: "Side-switch appearance review · VolleyCut",
  description:
    "Review label-only appearance evidence for detecting volleyball side switches.",
  robots: { index: false, follow: false },
};

function reportPath(): string {
  return path.resolve(
    /* turbopackIgnore: true */
    process.env.VOLLEYCUT_SIDE_SWITCH_REPORT ?? DEFAULT_REPORT,
  );
}

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
      const fallbackDuration = (eventsByRecording.get(file.recordingId) ?? 0) + 5;
      const labelPath = path.join(
        path.resolve(report.labels.directory),
        `${file.recordingId}.labels.json`,
      );
      try {
        const payload = JSON.parse(await readFile(labelPath, "utf8")) as unknown;
        const recording = isRecord(payload) && isRecord(payload.recording)
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
              : `${file.recordingId}.mp4`,
        };
      } catch {
        return {
          recordingId: file.recordingId,
          environment: file.environment,
          durationSeconds: fallbackDuration,
          videoFilename: `${file.recordingId}.mp4`,
        };
      }
    }),
  );
}

export default async function SideSwitchReviewPage() {
  const sourcePath = reportPath();
  try {
    const report = JSON.parse(await readFile(sourcePath, "utf8")) as AppearanceReport;
    const recordings = await loadRecordings(report);
    return (
      <SideSwitchReviewClient
        report={report}
        recordings={recordings}
        reportPath={sourcePath}
      />
    );
  } catch (error) {
    return (
      <SideSwitchReviewClient
        report={null}
        recordings={[]}
        reportPath={sourcePath}
        loadError={error instanceof Error ? error.message : String(error)}
      />
    );
  }
}
