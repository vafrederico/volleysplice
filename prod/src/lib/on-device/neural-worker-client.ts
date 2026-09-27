import type { NeuralWorkerRequest, NeuralWorkerResponse } from "./neural-worker-protocol";

type WithoutId<T> = T extends unknown ? Omit<T, "id"> : never;
export class NeuralWorkerClient {
  private readonly worker: Worker;
  private nextId = 0;
  private stopped = false;
  private readonly pending = new Map<number, { resolve: (value: NeuralWorkerResponse) => void; reject: (error: Error) => void }>();
  constructor(private readonly signal?: AbortSignal,
    private readonly onProgress?: (completed: number, total: number) => void) {
    signal?.throwIfAborted();
    this.worker = new Worker(new URL("./neural-worker.ts", import.meta.url), { type: "module", name: "volleysplice-rally" });
    this.worker.addEventListener("message", (event: MessageEvent<NeuralWorkerResponse>) => {
      const value = event.data;
      if (value.type === "progress") { this.onProgress?.(value.completed, value.total); return; }
      const pending = this.pending.get(value.id);
      this.pending.delete(value.id);
      if (value.type === "error") pending?.reject(new Error(value.message)); else pending?.resolve(value);
    });
    this.worker.addEventListener("error", () => this.dispose(new Error("Neural analysis stopped. Try the Production ensemble option or retry.")));
    this.worker.addEventListener("messageerror", () => this.dispose(new Error("The neural worker returned an unreadable result.")));
    signal?.addEventListener("abort", this.abort, { once: true });
  }
  private abort = () => this.dispose(new DOMException("Analysis cancelled", "AbortError"));
  request<T extends NeuralWorkerResponse["type"]>(request: WithoutId<NeuralWorkerRequest>, expected: T,
    transfer: Transferable[] = []): Promise<Extract<NeuralWorkerResponse, { type: T }>> {
    this.signal?.throwIfAborted();
    if (this.stopped) return Promise.reject(new Error("The neural worker has stopped."));
    const id = this.nextId++;
    return new Promise<NeuralWorkerResponse>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try { this.worker.postMessage({ ...request, id }, transfer); }
      catch (error) { this.pending.delete(id); reject(error); }
    }).then(value => {
      if (value.type !== expected) throw new Error("The neural worker returned an unexpected result.");
      return value as Extract<NeuralWorkerResponse, { type: T }>;
    });
  }
  dispose(error = new Error("The neural worker was closed.")): void {
    if (this.stopped) return;
    this.stopped = true;
    this.signal?.removeEventListener("abort", this.abort);
    this.worker.terminate();
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
}
