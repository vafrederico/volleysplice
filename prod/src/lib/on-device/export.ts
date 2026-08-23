import {
  AppendOnlyStreamTarget,
  AudioSampleSink,
  AudioSampleSource,
  Mp4OutputFormat,
  Output,
  Quality,
  StreamTarget,
  VideoSample,
  VideoSampleSink,
  VideoSampleSource,
  canEncodeAudio,
  canEncodeVideo,
  type StreamTargetChunk,
} from "mediabunny";

import { openLocalMedia } from "./media";
import {
  clipSampleToInterval,
  normalizeExportIntervals,
  timelinesHaveMatchingDuration,
  type ExportInterval,
} from "./export-math";
import { holdScreenWakeLock, type WakeLockState } from "./wake-lock";
import {
  supportsOpfsExport,
  type PreparedVideoExport,
} from "./export-delivery";
import { startServiceWorkerStreamDownload } from "./stream-download";
import {
  formatOverlayTeamLabel,
  prepareScoreOverlay,
  scoreOverlayLayout,
  scoreOverlaySnapshot,
  SCORE_OVERLAY_COLORS,
  type PreparedScoreOverlay,
  type ScoreOverlayOptions,
  type ScoreOverlaySnapshot,
} from "../score-overlay";

export type { ExportInterval } from "./export-math";
export type { PreparedVideoExport } from "./export-delivery";
export type { ScoreOverlayOptions } from "../score-overlay";

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

const OPFS_EXPORT_NAME = "volleycut-latest-export.mp4";
export type VideoExportMode = "compatible" | "opfs" | "stream-download";

export type VideoExportOptions = {
  scoreOverlay?: ScoreOverlayOptions;
};

type ExportCanvas = HTMLCanvasElement | OffscreenCanvas;
type ExportCanvasContext =
  | CanvasRenderingContext2D
  | OffscreenCanvasRenderingContext2D;

function createExportCanvas(width: number, height: number): {
  canvas: ExportCanvas;
  context: ExportCanvasContext;
} {
  const canvas: ExportCanvas = typeof OffscreenCanvas === "function"
    ? new OffscreenCanvas(width, height)
    : Object.assign(document.createElement("canvas"), { width, height });
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("This browser cannot draw the score overlay.");
  return { canvas, context };
}

function scoreOverlayPath(
  context: ExportCanvasContext,
  width: number,
  height: number,
  borderWidth: number,
  radius: number,
): void {
  const inset = borderWidth / 2;
  const right = width - inset;
  const bottom = height - inset;
  context.beginPath();
  context.moveTo(inset, inset);
  context.lineTo(right, inset);
  context.lineTo(right, bottom - radius);
  context.quadraticCurveTo(right, bottom, right - radius, bottom);
  context.lineTo(inset, bottom);
  context.closePath();
}

function drawScoreOverlay(
  context: ExportCanvasContext,
  videoWidth: number,
  videoHeight: number,
  snapshot: ScoreOverlaySnapshot,
): void {
  const shortestEdge = Math.max(1, Math.min(videoWidth, videoHeight));
  const provisionalHeight = Math.round(
    Math.min(76, Math.max(36, shortestEdge * 0.064)),
  );
  context.font = `700 ${Math.round(provisionalHeight * 0.39)}px sans-serif`;
  const layout = scoreOverlayLayout(context, videoWidth, videoHeight, snapshot);
  const firstScoreX = layout.team1Width;
  const team2X = firstScoreX + layout.scoreWidth;
  const secondScoreX = team2X + layout.team2Width;

  context.save();
  scoreOverlayPath(
    context,
    layout.width,
    layout.height,
    layout.borderWidth,
    layout.radius,
  );
  context.clip();
  context.fillStyle = SCORE_OVERLAY_COLORS.team1;
  context.fillRect(0, 0, layout.team1Width, layout.height);
  context.fillStyle = SCORE_OVERLAY_COLORS.scoreBackground;
  context.fillRect(firstScoreX, 0, layout.scoreWidth, layout.height);
  context.fillStyle = SCORE_OVERLAY_COLORS.team2;
  context.fillRect(team2X, 0, layout.team2Width, layout.height);
  context.fillStyle = SCORE_OVERLAY_COLORS.scoreBackground;
  context.fillRect(secondScoreX, 0, layout.scoreWidth, layout.height);

  context.font = `700 ${layout.fontSize}px sans-serif`;
  context.textBaseline = "middle";
  context.fillStyle = SCORE_OVERLAY_COLORS.teamText;
  context.textAlign = "center";
  context.fillText(
    formatOverlayTeamLabel(
      snapshot.team1Name,
      snapshot.servingTeamId === "team-1",
    ),
    layout.team1Width / 2,
    layout.height / 2,
    layout.team1Width - layout.horizontalPadding * 2,
  );
  context.fillText(
    formatOverlayTeamLabel(
      snapshot.team2Name,
      snapshot.servingTeamId === "team-2",
    ),
    team2X + layout.team2Width / 2,
    layout.height / 2,
    layout.team2Width - layout.horizontalPadding * 2,
  );
  context.fillStyle = SCORE_OVERLAY_COLORS.scoreText;
  context.textAlign = "center";
  context.fillText(
    snapshot.team1ScoreLabel,
    firstScoreX + layout.scoreWidth / 2,
    layout.height / 2,
  );
  context.fillText(
    snapshot.team2ScoreLabel,
    secondScoreX + layout.scoreWidth / 2,
    layout.height / 2,
  );

  context.strokeStyle = SCORE_OVERLAY_COLORS.border;
  context.lineWidth = layout.borderWidth;
  for (const x of [firstScoreX, team2X, secondScoreX]) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, layout.height);
    context.stroke();
  }
  scoreOverlayPath(
    context,
    layout.width,
    layout.height,
    layout.borderWidth,
    layout.radius,
  );
  context.stroke();
  context.restore();
}

function safeBaseName(filename: string): string {
  return (
    filename
      .replace(/\.[^.]+$/, "")
      .replace(/[^a-zA-Z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "volleycut"
  );
}

function picker(): SavePicker | null {
  const candidate = (window as unknown as { showSaveFilePicker?: SavePicker }).showSaveFilePicker;
  return candidate?.bind(window) ?? null;
}

async function chooseExportDestination(
  fileName: string,
  mode: VideoExportMode,
): Promise<ExportDestination> {
  if (mode === "stream-download") {
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

export async function exportRawQualityReel(
  file: File,
  requestedIntervals: readonly ExportInterval[],
  onProgress?: (progress: ExportProgress) => void,
  expectedTimelineDuration?: number,
  onWakeLockState?: (state: WakeLockState) => void,
  mode: VideoExportMode = "compatible",
  options: VideoExportOptions = {},
): Promise<PreparedVideoExport | null> {
  const outputName = `${safeBaseName(file.name)}-volleycut.mp4`;
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
      `The source is ${media.info.duration.toFixed(2)} s long, but the analyzed timeline is ${expectedTimelineDuration.toFixed(2)} s.`,
    );
  }
  const intervals = normalizeExportIntervals(requestedIntervals, media.info.duration);
  if (!intervals.length) {
    media.input.dispose();
    await destination.discard();
    throw new Error("Select at least one non-empty interval before exporting.");
  }

  const totalSeconds = intervals.reduce((total, interval) => total + interval.end - interval.start, 0);
  const preparedScoreOverlay: PreparedScoreOverlay | null = options.scoreOverlay
    ? prepareScoreOverlay(options.scoreOverlay)
    : null;
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
  // Overlay frames are rendered in display orientation, so rotation is baked in.
  output.addVideoTrack(videoSource, {
    rotation: preparedScoreOverlay ? 0 : media.info.rotation,
  });
  const audioSource = media.audioTrack
    ? new AudioSampleSource({ codec: "aac", quality: audioQuality })
    : null;
  if (audioSource) output.addAudioTrack(audioSource);

  const releaseWakeLock = await holdScreenWakeLock(onWakeLockState ?? (() => undefined));
  let outputStarted = false;
  let encodingStartedAt = 0;
  let lastProgressAt = -Infinity;
  let progressTimer: number | null = null;
  let currentCompletedSeconds = 0;
  let currentDetail = "Preparing original local video";
  const emitProgress = () => {
    const now = performance.now();
    onProgress?.({
      completedSeconds: currentCompletedSeconds,
      totalSeconds,
      elapsedSeconds: encodingStartedAt > 0 ? (now - encodingStartedAt) / 1000 : 0,
      detail: currentDetail,
    });
  };
  const reportProgress = (completedSeconds: number, detail: string, force = false) => {
    const now = performance.now();
    currentCompletedSeconds = completedSeconds;
    currentDetail = detail;
    if (!force && now - lastProgressAt < 250) return;
    lastProgressAt = now;
    emitProgress();
  };

  try {
    await output.start();
    outputStarted = true;
    encodingStartedAt = performance.now();
    reportProgress(0, `Encoding original ${media.info.width}×${media.info.height} frames`, true);
    progressTimer = window.setInterval(emitProgress, 500);

    const videoPump = async () => {
      const sink = new VideoSampleSink(media.videoTrack, {
        hardwareAcceleration: "prefer-hardware",
      });
      const overlaySurface = preparedScoreOverlay
        ? createExportCanvas(media.info.width, media.info.height)
        : null;
      let outputOffset = 0;
      try {
        for (const interval of intervals) {
          let first = true;
          const clipDuration = interval.end - interval.start;
          for await (const sample of sink.samples(interval.start, interval.end)) {
            try {
              const timing = clipSampleToInterval(sample.timestamp, sample.duration, interval);
              if (!timing) continue;
              const outputTimestamp = outputOffset + timing.timestamp;
              if (overlaySurface && preparedScoreOverlay) {
                overlaySurface.context.clearRect(
                  0,
                  0,
                  media.info.width,
                  media.info.height,
                );
                sample.draw(
                  overlaySurface.context,
                  0,
                  0,
                  media.info.width,
                  media.info.height,
                );
                drawScoreOverlay(
                  overlaySurface.context,
                  media.info.width,
                  media.info.height,
                  scoreOverlaySnapshot(preparedScoreOverlay, sample.timestamp),
                );
                const overlaidSample = new VideoSample(overlaySurface.canvas, {
                  timestamp: outputTimestamp,
                  duration: timing.duration,
                });
                try {
                  await videoSource.add(
                    overlaidSample,
                    first ? { keyFrame: true } : undefined,
                  );
                } finally {
                  overlaidSample.close();
                }
              } else {
                sample.setTimestamp(outputTimestamp);
                sample.setDuration(timing.duration);
                await videoSource.add(sample, first ? { keyFrame: true } : undefined);
              }
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
          : "MP4 saved from the original local video",
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
    if (progressTimer !== null) window.clearInterval(progressTimer);
    media.input.dispose();
    await releaseWakeLock();
  }
}
