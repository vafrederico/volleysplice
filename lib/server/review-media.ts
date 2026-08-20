import { readFile, stat } from "node:fs/promises";
import path from "node:path";

import { getServingSideReviewReportPath } from "@/lib/server/serving-side-review";
import { getSideSwitchReviewReportPath } from "@/lib/server/side-switch-review";

export type ReviewMediaKind = "side-switch" | "serving-side";

export class ReviewMediaNotFoundError extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function reportPath(kind: ReviewMediaKind): string {
  return kind === "side-switch"
    ? getSideSwitchReviewReportPath()
    : getServingSideReviewReportPath();
}

export async function resolveReviewMedia(
  kind: ReviewMediaKind,
  recordingId: string,
): Promise<{ path: string; filename: string }> {
  if (!/^[A-Za-z0-9:_-]+$/.test(recordingId)) {
    throw new ReviewMediaNotFoundError("recording id is invalid");
  }
  let payload: unknown;
  try {
    payload = JSON.parse(await readFile(reportPath(kind), "utf8")) as unknown;
  } catch {
    throw new ReviewMediaNotFoundError("review report is unavailable");
  }
  if (
    !isRecord(payload) ||
    !isRecord(payload.labels) ||
    !Array.isArray(payload.labels.files)
  ) {
    throw new ReviewMediaNotFoundError("review report has no media index");
  }
  const file = payload.labels.files.find(
    (candidate) => isRecord(candidate) && candidate.recordingId === recordingId,
  );
  if (!isRecord(file))
    throw new ReviewMediaNotFoundError("recording is not in the review report");

  let mediaPath: string | null =
    typeof file.videoPath === "string" ? path.resolve(file.videoPath) : null;
  let filename =
    typeof file.videoFilename === "string"
      ? file.videoFilename
      : `${recordingId}.mp4`;
  if (
    !mediaPath &&
    typeof payload.labels.directory === "string" &&
    typeof file.path === "string"
  ) {
    try {
      const labelPayload = JSON.parse(
        await readFile(
          path.resolve(payload.labels.directory, path.basename(file.path)),
          "utf8",
        ),
      ) as unknown;
      if (isRecord(labelPayload) && isRecord(labelPayload.recording)) {
        const recording = labelPayload.recording;
        if (typeof recording.video === "string") {
          const candidate = path.resolve(
            payload.labels.directory,
            recording.video,
          );
          mediaPath = candidate;
        }
        if (typeof recording.videoFilename === "string")
          filename = recording.videoFilename;
      }
    } catch {
      mediaPath = null;
    }
  }
  if (!mediaPath)
    throw new ReviewMediaNotFoundError("recording has no video path");
  try {
    const metadata = await stat(mediaPath);
    if (!metadata.isFile() || metadata.size <= 0) throw new Error("not a file");
  } catch {
    throw new ReviewMediaNotFoundError("recording video is unavailable");
  }
  return { path: mediaPath, filename: path.basename(filename) };
}
