import { VideoSampleSink } from "mediabunny";

import type { OpenedMedia } from "./media.ts";
import {
  SERVING_SIDE_HEIGHT,
  SERVING_SIDE_WIDTH,
} from "./serving-side-model.ts";
import { samplesAtTimestampsFromSequentialPass } from "./sequential-samples.ts";
import type { NormalizedRoi } from "./types.ts";
import { loadOpenCv } from "./visual-features.ts";

const SIDE_SWITCH_WIDTH = 256;
const SIDE_SWITCH_HEIGHT = 144;

function timestampKey(timestamp: number): string {
  return timestamp.toFixed(9);
}

function conversionCanvas(width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context)
    throw new Error("Could not create the specialist sampling canvas.");
  return { canvas, context };
}

export type SpecialistFramePlan = {
  servingSideTimes?: readonly number[];
  sideSwitchTimes?: readonly number[];
};

export type SampledSpecialistFrames = {
  servingSide: Map<number, Uint8Array>;
  sideSwitch: Map<string, Uint8Array>;
};

/** Decodes the union of both specialist schedules in one continuous pass. */
export async function sampleSpecialistFramesSequentially(
  media: OpenedMedia,
  roi: NormalizedRoi,
  plan: SpecialistFramePlan,
  onProgress?: (completed: number, total: number) => void,
): Promise<SampledSpecialistFrames> {
  const servingTimestampByKey = new Map(
    (plan.servingSideTimes ?? []).map((timestamp) => [
      timestampKey(timestamp),
      timestamp,
    ]),
  );
  const sideSwitchKeys = new Set(
    (plan.sideSwitchTimes ?? []).map(timestampKey),
  );
  const requestedByKey = new Map<string, number>();
  for (const timestamp of [
    ...(plan.servingSideTimes ?? []),
    ...(plan.sideSwitchTimes ?? []),
  ]) {
    requestedByKey.set(timestampKey(timestamp), timestamp);
  }
  const requested = [...requestedByKey.values()].sort(
    (left, right) => left - right,
  );
  if (requested.length === 0) {
    return { servingSide: new Map(), sideSwitch: new Map() };
  }

  const cv = await loadOpenCv();
  const left = Math.round(roi.x * media.info.width);
  const top = Math.round(roi.y * media.info.height);
  const right = Math.round((roi.x + roi.width) * media.info.width);
  const bottom = Math.round((roi.y + roi.height) * media.info.height);
  const crop = {
    left,
    top,
    width: Math.max(1, right - left),
    height: Math.max(1, bottom - top),
  };
  const serving = conversionCanvas(SERVING_SIDE_WIDTH, SERVING_SIDE_HEIGHT);
  const sideSwitch = conversionCanvas(SIDE_SWITCH_WIDTH, SIDE_SWITCH_HEIGHT);
  const frames: SampledSpecialistFrames = {
    servingSide: new Map(),
    sideSwitch: new Map(),
  };
  const sink = new VideoSampleSink(media.videoTrack, {
    hardwareAcceleration: "prefer-hardware",
  });
  const iterator = samplesAtTimestampsFromSequentialPass(
    sink,
    requested,
    () => undefined,
  );
  let completed = 0;
  try {
    while (completed < requested.length) {
      const next = await iterator.next();
      if (next.done) break;
      const sample = next.value;
      if (!sample) {
        throw new Error(
          `No specialist video frame was available at ${requested[completed]} seconds.`,
        );
      }
      const key = timestampKey(requested[completed]!);
      try {
        const servingTimestamp = servingTimestampByKey.get(key);
        if (servingTimestamp !== undefined) {
          serving.context.clearRect(
            0,
            0,
            serving.canvas.width,
            serving.canvas.height,
          );
          sample.drawWithFit(serving.context, { fit: "fill", crop });
          const rgba = cv.imread(serving.canvas);
          const gray = new cv.Mat();
          try {
            cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
            frames.servingSide.set(
              servingTimestamp,
              Uint8Array.from(gray.data),
            );
          } finally {
            rgba.delete();
            gray.delete();
          }
        }
        if (sideSwitchKeys.has(key)) {
          sideSwitch.context.clearRect(
            0,
            0,
            sideSwitch.canvas.width,
            sideSwitch.canvas.height,
          );
          sample.drawWithFit(sideSwitch.context, { fit: "fill", crop });
          const rgba = cv.imread(sideSwitch.canvas);
          const bgr = new cv.Mat();
          try {
            cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
            frames.sideSwitch.set(key, Uint8Array.from(bgr.data));
          } finally {
            rgba.delete();
            bgr.delete();
          }
        }
      } finally {
        sample.close();
      }
      completed += 1;
      if (completed % 8 === 0 || completed === requested.length) {
        onProgress?.(completed, requested.length);
      }
    }
  } finally {
    await iterator.return(undefined);
  }
  if (completed !== requested.length) {
    throw new Error("Shared specialist frame sampling ended early.");
  }
  return frames;
}
