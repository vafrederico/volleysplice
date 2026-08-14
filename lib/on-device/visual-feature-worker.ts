import { VideoSample } from "mediabunny";

import { ANALYSIS_HEIGHT, ANALYSIS_WIDTH } from "./feature-schema";
import type {
  VisualFeatureWorkerRequest,
  VisualFeatureWorkerResponse,
  WorkerCrop,
} from "./visual-feature-worker-protocol";
import { extractVisualFeaturesFromImageData } from "./visual-features";

type CvRuntime = typeof import("@techstark/opencv-js");
type CvThenable = {
  then?: (ready: (runtime: CvRuntime) => void) => unknown;
} & Record<string, unknown>;
type WorkerGlobalWithCv = typeof globalThis & { cv?: CvThenable };
type FeatureWorkerScope = {
  location: Location;
  postMessage(message: VisualFeatureWorkerResponse, transfer?: Transferable[]): void;
  close(): void;
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<VisualFeatureWorkerRequest>) => void,
  ): void;
};
type MipmappedVideoSample = VideoSample & {
  _drawWithFitAndMipmapping: (
    canvas: OffscreenCanvas,
    context: OffscreenCanvasRenderingContext2D,
    options: {
      fit: "fill";
      rotation: 0 | 90 | 180 | 270;
      crop: WorkerCrop;
      targetIsFresh: boolean;
      fillBlack: boolean;
    },
  ) => void;
};

const workerScope = self as unknown as FeatureWorkerScope;
let runtimePromise: Promise<CvRuntime> | null = null;
let previousGray: import("@techstark/opencv-js").Mat | null = null;
let canvasIsFresh = true;
let detailedProfiling = true;
const canvas = new OffscreenCanvas(ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
const firefox = navigator.userAgent.includes("Firefox");
const context = (() => {
  const candidate = canvas.getContext("2d", { alpha: firefox });
  if (!candidate) throw new Error("The extraction worker could not create a 2D canvas context.");
  return candidate;
})();

function withoutThen(candidate: CvThenable): CvRuntime {
  return new Proxy(candidate, {
    get(target, property, receiver) {
      return property === "then" ? undefined : Reflect.get(target, property, receiver);
    },
  }) as unknown as CvRuntime;
}

async function loadWorkerOpenCv(): Promise<CvRuntime> {
  runtimePromise ??= (async () => {
    const assetUrl = new URL("/on-device/opencv-worker.js", workerScope.location.origin).href;
    await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ assetUrl);
    const candidate = (globalThis as WorkerGlobalWithCv).cv;
    if (!candidate) throw new Error("OpenCV did not publish its worker runtime.");
    if (typeof candidate.then !== "function") return candidate as unknown as CvRuntime;
    return new Promise<CvRuntime>((resolve) => {
      candidate.then?.(() => resolve(withoutThen(candidate)));
    });
  })();
  return runtimePromise;
}

function post(response: VisualFeatureWorkerResponse, transfer: Transferable[] = []) {
  workerScope.postMessage(response, transfer);
}

async function processFrame(
  request: Extract<VisualFeatureWorkerRequest, { type: "frame" }>,
): Promise<void> {
  const startedAt = detailedProfiling ? performance.now() : 0;
  let sample: VideoSample | null = null;
  try {
    const cv = await loadWorkerOpenCv();
    sample = new VideoSample(request.frame, {
      timestamp: request.timestamp,
      duration: request.duration,
      rotation: request.rotation,
    });
    const canvasDrawStartedAt = detailedProfiling ? performance.now() : 0;
    (sample as MipmappedVideoSample)._drawWithFitAndMipmapping(canvas, context, {
      fit: "fill",
      rotation: request.rotation,
      crop: request.crop,
      targetIsFresh: canvasIsFresh,
      fillBlack: firefox,
    });
    canvasIsFresh = false;
    const canvasDrawMs = detailedProfiling ? performance.now() - canvasDrawStartedAt : 0;
    const readbackStartedAt = detailedProfiling ? performance.now() : 0;
    const imageData = context.getImageData(0, 0, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
    const readbackMs = detailedProfiling ? performance.now() - readbackStartedAt : 0;
    const result = extractVisualFeaturesFromImageData(
      cv,
      imageData,
      previousGray,
      readbackMs,
      detailedProfiling,
    );
    previousGray?.delete();
    previousGray = result.gray;
    const values = new Float32Array(result.values);
    post(
      {
        type: "result",
        id: request.id,
        timestamp: request.timestamp,
        values: values.buffer,
        timing: result.timing,
        canvasDrawMs,
        workerElapsedMs: detailedProfiling ? performance.now() - startedAt : 0,
      },
      [values.buffer],
    );
  } finally {
    if (sample) sample.close();
    else request.frame.close();
  }
}

let queue = Promise.resolve();
workerScope.addEventListener("message", (event: MessageEvent<VisualFeatureWorkerRequest>) => {
  const request = event.data;
  if (request.type === "dispose") {
    queue = queue.finally(() => {
      previousGray?.delete();
      previousGray = null;
      workerScope.close();
    });
    return;
  }
  queue = queue.then(async () => {
    try {
      if (request.type === "initialize") {
        detailedProfiling = request.detailedProfiling;
        const startedAt = detailedProfiling ? performance.now() : 0;
        await loadWorkerOpenCv();
        post({
          type: "ready",
          openCvLoadMs: detailedProfiling ? performance.now() - startedAt : 0,
        });
      } else {
        await processFrame(request);
      }
    } catch (error) {
      post({
        type: "error",
        id: request.type === "frame" ? request.id : null,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  });
});
