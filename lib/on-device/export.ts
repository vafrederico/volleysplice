import {
  AppendOnlyStreamTarget,
  AudioSampleSink,
  AudioSampleSource,
  Mp4OutputFormat,
  Output,
  Quality,
  StreamTarget,
  VideoSampleSink,
  VideoSampleSource,
  canEncodeAudio,
  canEncodeVideo,
  type StreamTargetChunk,
} from "mediabunny";

import { openLocalMedia } from "./media";
import { PRODUCTION_MODEL_ID } from "../production-model";
import {
  clipSampleToInterval,
  normalizeExportIntervals,
  timelinesHaveMatchingDuration,
  type ExportInterval,
} from "./export-math";
import { holdScreenWakeLock, type WakeLockState } from "./wake-lock";
import { startServiceWorkerStreamDownload } from "./stream-download";

export type { ExportInterval } from "./export-math";
export type ExportProgress = {
  completedSeconds: number;
  totalSeconds: number;
  elapsedSeconds: number;
  detail: string;
};

type SaveFileHandle = {
  createWritable(): Promise<WritableStream<unknown>>;
};

type SavePicker = (options: {
  suggestedName: string;
  types: Array<{ description: string; accept: Record<string, string[]> }>;
}) => Promise<SaveFileHandle>;

type ExportDestination = {
  kind: "direct" | "opfs" | "stream";
  createWritable(): Promise<WritableStream<unknown>>;
  finish(): Promise<PreparedVideoExport | null>;
  discard(): Promise<void>;
};

export type PreparedVideoExport = {
  file: File;
  fileName: string;
};

export type PreparedVideoDelivery = "shared" | "downloaded";
export type VideoExportMode = "compatible" | "opfs" | "stream-download";

const OPFS_EXPORT_NAME = "volleycut-latest-export.mp4";

function safeBaseName(filename: string): string {
  return (
    filename
      .replace(/\.[^.]+$/, "")
      .replace(/[^a-zA-Z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "volleysplice"
  );
}

function picker(): SavePicker | null {
  const candidate = (window as unknown as { showSaveFilePicker?: SavePicker }).showSaveFilePicker;
  return candidate?.bind(window) ?? null;
}

export function supportsOpfsExport(): boolean {
  return typeof navigator !== "undefined" && typeof navigator.storage?.getDirectory === "function";
}

async function chooseExportDestination(
  fileName: string,
  mode: VideoExportMode,
): Promise<ExportDestination> {
  if (mode === "stream-download") {
    // Start the browser download synchronously inside the initiating tap. The encoder receives
    // one pull credit at a time from the Service Worker, keeping buffering bounded.
    const download = startServiceWorkerStreamDownload(fileName);
    return {
      kind: "stream",
      createWritable: async () => download.writable,
      finish: async () => null,
      discard: () => download.cancel("The video export stopped before completion."),
    };
  }

  const savePicker = mode === "compatible" ? picker() : null;
  if (savePicker) {
    // Keep this call before any await so Chromium retains the initiating click's activation.
    const fileHandle = await savePicker({
      suggestedName: fileName,
      types: [{ description: "MP4 video", accept: { "video/mp4": [".mp4"] } }],
    });
    return {
      kind: "direct",
      createWritable: () => fileHandle.createWritable(),
      finish: async () => null,
      discard: async () => undefined,
    };
  }

  if (!supportsOpfsExport()) {
    throw new Error(
      "Video export needs either direct file access or origin-private file storage in this browser.",
    );
  }

  const root = await navigator.storage.getDirectory();
  const fileHandle = await root.getFileHandle(OPFS_EXPORT_NAME, { create: true });
  if (typeof fileHandle.createWritable !== "function") {
    await root.removeEntry(OPFS_EXPORT_NAME).catch(() => undefined);
    throw new Error("This browser can open origin-private storage but cannot write the exported video.");
  }

  return {
    kind: "opfs",
    createWritable: () => fileHandle.createWritable({ keepExistingData: false }),
    finish: async () => {
      const storedFile = await fileHandle.getFile();
      return {
        // A File made from another Blob remains file-backed; this gives the share sheet a useful name
        // without copying the completed video into a JavaScript ArrayBuffer.
        file: new File([storedFile], fileName, {
          type: "video/mp4",
          lastModified: storedFile.lastModified,
        }),
        fileName,
      };
    },
    discard: () => root.removeEntry(OPFS_EXPORT_NAME).catch(() => undefined),
  };
}

function downloadPreparedFile(prepared: PreparedVideoExport): void {
  const url = URL.createObjectURL(prepared.file);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = prepared.fileName;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export async function deliverPreparedVideoExport(
  prepared: PreparedVideoExport,
): Promise<PreparedVideoDelivery> {
  const shareData: ShareData = {
    files: [prepared.file],
    title: prepared.fileName,
  };
  if (
    typeof navigator.share === "function" &&
    (typeof navigator.canShare !== "function" || navigator.canShare(shareData))
  ) {
    // Do not await anything before this call: Web Share consumes the current tap's activation.
    await navigator.share(shareData);
    return "shared";
  }

  downloadPreparedFile(prepared);
  return "downloaded";
}

export async function exportRawQualityReel(
  file: File,
  requestedIntervals: readonly ExportInterval[],
  onProgress?: (progress: ExportProgress) => void,
  expectedTimelineDuration?: number,
  onWakeLockState?: (state: WakeLockState) => void,
  mode: VideoExportMode = "compatible",
): Promise<PreparedVideoExport | null> {
  const outputName = `${safeBaseName(file.name)}-volleysplice.mp4`;
  // Destination selection stays at the top of the user-initiated call for the desktop picker.
  const destination = await chooseExportDestination(outputName, mode);
  let media: Awaited<ReturnType<typeof openLocalMedia>>;
  try {
    media = await openLocalMedia(file);
  } catch (error) {
    await destination.discard();
    throw error;
  }
  if (
    expectedTimelineDuration !== undefined &&
    !timelinesHaveMatchingDuration(media.info.duration, expectedTimelineDuration)
  ) {
    media.input.dispose();
    await destination.discard();
    throw new Error(
      `The export master is ${media.info.duration.toFixed(2)} s long, but the analyzed file is ${expectedTimelineDuration.toFixed(2)} s. Choose the matching raw master with the same timeline.`,
    );
  }
  const intervals = normalizeExportIntervals(requestedIntervals, media.info.duration);
  if (!intervals.length) {
    media.input.dispose();
    await destination.discard();
    throw new Error("Select at least one non-empty interval before exporting.");
  }
  const totalSeconds = intervals.reduce((total, interval) => total + interval.end - interval.start, 0);
  const videoQuality = new Quality("very-high");
  const audioQuality = new Quality("high");
  let videoSupported: boolean;
  let audioSupported: boolean;
  try {
    [videoSupported, audioSupported] = await Promise.all([
      canEncodeVideo("avc", {
        width: media.info.width,
        height: media.info.height,
        quality: videoQuality,
        hardwareAcceleration: "prefer-hardware",
      }),
      media.audioTrack
        ? canEncodeAudio("aac", {
            numberOfChannels: media.info.channels ?? 2,
            sampleRate: media.info.sampleRate ?? 48_000,
            quality: audioQuality,
          })
        : Promise.resolve(true),
    ]);
  } catch (error) {
    media.input.dispose();
    await destination.discard();
    throw error;
  }
  if (!videoSupported || !audioSupported) {
    media.input.dispose();
    await destination.discard();
    throw new Error(
      `This browser cannot encode the source at ${media.info.width}×${media.info.height} as AVC/AAC.`,
    );
  }

  let writable: WritableStream<unknown>;
  try {
    writable = await destination.createWritable();
  } catch (error) {
    media.input.dispose();
    await destination.discard();
    throw error;
  }
  const output = new Output({
    // A regular MP4 patches its mdat header after encoding. Fragmented MP4 is the disk-free
    // mode that Mediabunny guarantees can be written monotonically to a download stream.
    format: new Mp4OutputFormat({
      fastStart: destination.kind === "stream" ? "fragmented" : false,
    }),
    target:
      destination.kind === "stream"
        ? new AppendOnlyStreamTarget(writable as WritableStream<Uint8Array>)
        : new StreamTarget(writable as WritableStream<StreamTargetChunk>, {
            chunked: true,
            chunkSize: 1024 * 1024,
          }),
  });
  const videoSource = new VideoSampleSource({
    codec: "avc",
    quality: videoQuality,
    keyFrameInterval: 2,
    latencyMode: "quality",
    hardwareAcceleration: "prefer-hardware",
  });
  output.addVideoTrack(videoSource, { rotation: media.info.rotation });
  const audioSource = media.audioTrack
    ? new AudioSampleSource({ codec: "aac", quality: audioQuality })
    : null;
  if (audioSource) output.addAudioTrack(audioSource);

  const releaseWakeLock = await holdScreenWakeLock(onWakeLockState ?? (() => undefined));
  let outputStarted = false;
  let encodingStartedAt = 0;
  let lastProgressAt = -Infinity;
  const reportProgress = (completedSeconds: number, detail: string, force = false) => {
    const now = performance.now();
    if (!force && now - lastProgressAt < 250) return;
    lastProgressAt = now;
    onProgress?.({
      completedSeconds,
      totalSeconds,
      elapsedSeconds: encodingStartedAt > 0 ? (now - encodingStartedAt) / 1000 : 0,
      detail,
    });
  };
  try {
    await output.start();
    outputStarted = true;
    encodingStartedAt = performance.now();
    reportProgress(0, `Encoding original ${media.info.width}×${media.info.height} frames`, true);
    const videoPump = async () => {
      const sink = new VideoSampleSink(media.videoTrack, {
        hardwareAcceleration: "prefer-hardware",
      });
      let outputOffset = 0;
      try {
        for (const interval of intervals) {
          let first = true;
          const clipDuration = interval.end - interval.start;
          for await (const sample of sink.samples(interval.start, interval.end)) {
            try {
              // The sink intentionally returns the frame covering the requested start,
              // which may begin just before it. Clip that frame's timing so it cannot
              // overlap the following frame in the concatenated output.
              const timing = clipSampleToInterval(sample.timestamp, sample.duration, interval);
              if (!timing) continue;
              sample.setTimestamp(outputOffset + timing.timestamp);
              sample.setDuration(timing.duration);
              await videoSource.add(sample, first ? { keyFrame: true } : undefined);
              first = false;
              reportProgress(
                Math.min(totalSeconds, outputOffset + timing.timestamp + timing.duration),
                `Encoding original ${media.info.width}×${media.info.height} frames`,
              );
            } finally {
              sample.close();
            }
          }
          outputOffset += clipDuration;
        }
      } finally {
        videoSource.close();
      }
    };

    const audioPump = async () => {
      if (!audioSource || !media.audioTrack) return;
      const sink = new AudioSampleSink(media.audioTrack);
      let outputOffset = 0;
      try {
        for (const interval of intervals) {
          for await (const sample of sink.samples(interval.start, interval.end)) {
            let outgoing = sample;
            try {
              const firstFrame = Math.max(
                0,
                Math.round((interval.start - sample.timestamp) * sample.sampleRate),
              );
              const endFrame = Math.min(
                sample.numberOfFrames,
                Math.round((interval.end - sample.timestamp) * sample.sampleRate),
              );
              if (endFrame <= firstFrame) continue;
              if (firstFrame > 0 || endFrame < sample.numberOfFrames) {
                outgoing = sample.trim(firstFrame, endFrame);
                sample.close();
              }
              outgoing.setTimestamp(outputOffset + Math.max(0, outgoing.timestamp - interval.start));
              await audioSource.add(outgoing);
            } finally {
              outgoing.close();
            }
          }
          outputOffset += interval.end - interval.start;
        }
      } finally {
        audioSource.close();
      }
    };

    await Promise.all([videoPump(), audioPump()]);
    reportProgress(
      totalSeconds,
      destination.kind === "opfs"
        ? "Finalizing MP4 in private device storage"
        : destination.kind === "stream"
          ? "Sending final MP4 bytes to browser download"
        : "Finalizing MP4 on disk",
      true,
    );
    await output.finalize();
    const prepared = await destination.finish();
    reportProgress(
      totalSeconds,
      prepared
        ? "MP4 ready to share or save"
        : destination.kind === "stream"
          ? "MP4 stream handed to browser download"
          : "MP4 written from the original file",
      true,
    );
    return prepared;
  } catch (error) {
    if (outputStarted && output.state !== "finalized" && output.state !== "canceled") {
      await output.cancel().catch(() => undefined);
    }
    await destination.discard();
    throw error;
  } finally {
    media.input.dispose();
    await releaseWakeLock();
  }
}

export function downloadEditDecisionList(
  file: File,
  intervals: readonly ExportInterval[],
  duration: number,
): void {
  const payload = {
    schemaVersion: 1,
    source: { name: file.name, size: file.size, lastModified: file.lastModified, duration },
    modelId: PRODUCTION_MODEL_ID,
    intervals: normalizeExportIntervals(intervals, duration),
  };
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${safeBaseName(file.name)}-volleysplice-edl.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}
