import type { VisualFeatureResult } from "./visual-features";
import type { FeatureReductionKernel } from "./types";

export type WorkerCrop = {
  left: number;
  top: number;
  width: number;
  height: number;
};

export type VisualFeatureWorkerRequest =
  | {
      type: "initialize";
      detailedProfiling: boolean;
      reductionKernel: FeatureReductionKernel;
      openCvUrl: string;
      reductionWasmUrl: string;
    }
  | {
      type: "frame";
      id: number;
      frame: VideoFrame;
      timestamp: number;
      duration: number;
      rotation: 0 | 90 | 180 | 270;
      crop: WorkerCrop;
    }
  | { type: "dispose" };

export type VisualFeatureWorkerResponse =
  | {
      type: "ready";
      openCvLoadMs: number;
      reductionKernelLoadMs: number;
    }
  | {
      type: "result";
      id: number;
      timestamp: number;
      values: ArrayBuffer;
      timing: VisualFeatureResult["timing"];
      canvasDrawMs: number;
      workerElapsedMs: number;
    }
  | {
      type: "error";
      id: number | null;
      message: string;
    };
