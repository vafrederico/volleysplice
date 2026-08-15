import type {
  VisualFeatureWorkerRequest,
  VisualFeatureWorkerResponse,
  WorkerCrop,
} from "./visual-feature-worker-protocol";
import { runtimeAssetUrl } from "../runtime-assets";
import type { FeatureReductionKernel } from "./types";

export type WorkerFeatureResult = Extract<
  VisualFeatureWorkerResponse,
  { type: "result" }
>;

type PendingFrame = {
  resolve: (result: WorkerFeatureResult) => void;
  reject: (error: Error) => void;
};

export class VisualFeatureWorkerClient {
  readonly openCvLoadMs: number;
  readonly reductionKernelLoadMs: number;
  private readonly worker: Worker;
  private readonly pending = new Map<number, PendingFrame>();
  private nextId = 0;
  private stopped = false;

  private constructor(worker: Worker, openCvLoadMs: number, reductionKernelLoadMs: number) {
    this.worker = worker;
    this.openCvLoadMs = openCvLoadMs;
    this.reductionKernelLoadMs = reductionKernelLoadMs;
    worker.addEventListener("message", this.handleMessage);
    worker.addEventListener("error", this.handleWorkerError);
    worker.addEventListener("messageerror", this.handleWorkerError);
  }

  static async create(
    detailedProfiling: boolean,
    reductionKernel: FeatureReductionKernel,
  ): Promise<VisualFeatureWorkerClient> {
    if (!("Worker" in globalThis) || !("OffscreenCanvas" in globalThis)) {
      throw new Error("Dedicated extraction workers are unavailable.");
    }
    const worker = new Worker(new URL("./visual-feature-worker.ts", import.meta.url), {
      type: "module",
      name: "volleycut-visual-features",
    });
    try {
      const ready = await new Promise<
        Extract<VisualFeatureWorkerResponse, { type: "ready" }>
      >((resolve, reject) => {
        const timeout = window.setTimeout(
          () => reject(new Error("The extraction worker did not initialize in time.")),
          30_000,
        );
        const handleMessage = (event: MessageEvent<VisualFeatureWorkerResponse>) => {
          if (event.data.type !== "ready" && event.data.type !== "error") return;
          window.clearTimeout(timeout);
          worker.removeEventListener("message", handleMessage);
          worker.removeEventListener("error", handleError);
          if (event.data.type === "ready") resolve(event.data);
          else reject(new Error(event.data.message));
        };
        const handleError = () => {
          window.clearTimeout(timeout);
          worker.removeEventListener("message", handleMessage);
          reject(new Error("The extraction worker failed during initialization."));
        };
        worker.addEventListener("message", handleMessage);
        worker.addEventListener("error", handleError, { once: true });
        const request: VisualFeatureWorkerRequest = {
          type: "initialize",
          detailedProfiling,
          reductionKernel,
          openCvUrl: runtimeAssetUrl("opencv-worker.js"),
          reductionWasmUrl: runtimeAssetUrl("feature-reductions.wasm"),
        };
        worker.postMessage(request);
      });
      return new VisualFeatureWorkerClient(
        worker,
        ready.openCvLoadMs,
        ready.reductionKernelLoadMs,
      );
    } catch (error) {
      worker.terminate();
      throw error;
    }
  }

  extract(
    frame: VideoFrame,
    metadata: {
      timestamp: number;
      duration: number;
      rotation: 0 | 90 | 180 | 270;
      crop: WorkerCrop;
    },
  ): Promise<WorkerFeatureResult> {
    if (this.stopped) {
      frame.close();
      return Promise.reject(new Error("The extraction worker has stopped."));
    }
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const request: VisualFeatureWorkerRequest = { type: "frame", id, frame, ...metadata };
      try {
        this.worker.postMessage(request, [frame]);
      } catch (error) {
        this.pending.delete(id);
        frame.close();
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  dispose(): void {
    if (this.stopped) return;
    this.stopped = true;
    const error = new Error("The extraction worker was disposed.");
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
    this.worker.removeEventListener("message", this.handleMessage);
    this.worker.removeEventListener("error", this.handleWorkerError);
    this.worker.removeEventListener("messageerror", this.handleWorkerError);
    this.worker.terminate();
  }

  private handleMessage = (event: MessageEvent<VisualFeatureWorkerResponse>) => {
    const response = event.data;
    if (response.type === "ready") return;
    if (response.type === "error") {
      const error = new Error(response.message);
      if (response.id !== null) {
        const request = this.pending.get(response.id);
        this.pending.delete(response.id);
        request?.reject(error);
      } else {
        this.rejectAll(error);
      }
      return;
    }
    const request = this.pending.get(response.id);
    this.pending.delete(response.id);
    request?.resolve(response);
  };

  private handleWorkerError = () => {
    this.rejectAll(new Error("The extraction worker stopped unexpectedly."));
  };

  private rejectAll(error: Error) {
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
  }
}
