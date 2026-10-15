import type { EmbeddingSequence } from "./neural-cache";

// Image buffers live only for the active analysis; project persistence stays small.
const contexts = new WeakMap<object, EmbeddingSequence>();
export function rememberServingEmbeddings(analysis: object, embedding: EmbeddingSequence): void {
  contexts.set(analysis, embedding);
}
export function takeServingEmbeddings(analysis: object): EmbeddingSequence | undefined {
  const embedding = contexts.get(analysis);
  contexts.delete(analysis);
  return embedding;
}
