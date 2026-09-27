import type { NeuralBundle, NeuralRallySelection } from "./rally-model";
import type { NeuralPipelineConfig } from "./neural-contract";
import type { NormalizedRoi } from "./types";
import type { EmbeddingSequence } from "./neural-cache";

export type NeuralWorkerRequest =
  | { id: number; type: "initialize"; baseUrl: string; ortBaseUrl: string; openCvUrl: string;
      bundle: NeuralBundle; config: NeuralPipelineConfig; selection: NeuralRallySelection; needEncoder: boolean }
  | { id: number; type: "frame"; frame: VideoFrame; timestamp: number; duration: number;
      rotation: 0 | 90 | 180 | 270; width: number; height: number; roi: NormalizedRoi; offset: number }
  | { id: number; type: "temporal"; times: Float64Array; rankedAv: Float32Array; embedding: EmbeddingSequence; duration: number };
export type NeuralWorkerResponse =
  | { id: number; type: "ready"; provider: "webgpu" | "wasm" }
  | { id: number; type: "frame"; tokens: Float32Array; quality: Float32Array }
  | { id: number; type: "progress"; completed: number; total: number }
  | { id: number; type: "temporal"; probabilities: Float32Array; rallies: { start: number; end: number; confidence: number }[] }
  | { id: number; type: "error"; message: string };
