import {
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
import {
  clipSampleToInterval,
  normalizeExportIntervals,
  timelinesHaveMatchingDuration,
  type ExportInterval,
} from "./export-math";
import { holdScreenWakeLock, type WakeLockState } from "./wake-lock";

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

function safeBaseName(filename: string): string {
  return (
    filename
      .replace(/\.[^.]+$/, "")
      .replace(/[^a-zA-Z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "volleycut"
  );
}

function picker(): SavePicker {
  const candidate = (window as unknown as { showSaveFilePicker?: SavePicker }).showSaveFilePicker;
  if (!candidate) {
    throw new Error(
      "Direct-to-disk export needs the File System Access API. Use desktop Chrome or Edge.",
    );
  }
  return candidate.bind(window);
}

export async function exportRawQualityReel(
  file: File,
  requestedIntervals: readonly ExportInterval[],
  onProgress?: (progress: ExportProgress) => void,
  expectedTimelineDuration?: number,
  onWakeLockState?: (state: WakeLockState) => void,
): Promise<void> {
  // Open the picker while the button click still has browser user activation.
  const fileHandle = await picker()({
    suggestedName: `${safeBaseName(file.name)}-volleycut.mp4`,
    types: [{ description: "MP4 video", accept: { "video/mp4": [".mp4"] } }],
  });
  const media = await openLocalMedia(file);
  if (
    expectedTimelineDuration !== undefined &&
    !timelinesHaveMatchingDuration(media.info.duration, expectedTimelineDuration)
  ) {
    media.input.dispose();
    throw new Error(
      `The source is ${media.info.duration.toFixed(2)} s long, but the analyzed timeline is ${expectedTimelineDuration.toFixed(2)} s.`,
    );
  }
  const intervals = normalizeExportIntervals(requestedIntervals, media.info.duration);
  if (!intervals.length) {
    media.input.dispose();
    throw new Error("Select at least one non-empty interval before exporting.");
  }

  const totalSeconds = intervals.reduce((total, interval) => total + interval.end - interval.start, 0);
  const videoQuality = new Quality("very-high");
  const audioQuality = new Quality("high");
  const [videoSupported, audioSupported] = await Promise.all([
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
  if (!videoSupported || !audioSupported) {
    media.input.dispose();
    throw new Error(
      `This browser cannot encode the source at ${media.info.width}×${media.info.height} as AVC/AAC.`,
    );
  }

  const writable = await fileHandle.createWritable();
  const output = new Output({
    format: new Mp4OutputFormat({ fastStart: false }),
    target: new StreamTarget(writable as WritableStream<StreamTargetChunk>, {
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
      let outputOffset = 0;
      try {
        for (const interval of intervals) {
          let first = true;
          const clipDuration = interval.end - interval.start;
          for await (const sample of sink.samples(interval.start, interval.end)) {
            try {
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
    reportProgress(totalSeconds, "Finalizing MP4 on disk", true);
    await output.finalize();
    reportProgress(totalSeconds, "MP4 saved from the original local video", true);
  } catch (error) {
    if (outputStarted && output.state !== "finalized" && output.state !== "canceled") {
      await output.cancel().catch(() => undefined);
    }
    throw error;
  } finally {
    if (progressTimer !== null) window.clearInterval(progressTimer);
    media.input.dispose();
    await releaseWakeLock();
  }
}
