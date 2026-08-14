import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const rootWorkerPath = new URL(
  "../../public/volleycut-export-sw.js",
  import.meta.url,
);
const prodWorkerPath = new URL(
  "../../prod/public/volleycut-export-sw.js",
  import.meta.url,
);

type WorkerHandlers = {
  message: (event: { data: unknown; ports: MessagePort[] }) => void;
  fetch: (event: {
    request: Request;
    respondWith(response: Promise<Response>): void;
  }) => void;
};

async function loadWorker(): Promise<WorkerHandlers> {
  const source = await readFile(rootWorkerPath, "utf8");
  const listeners = new Map<string, (event: never) => void>();
  const workerSelf = {
    addEventListener(type: string, listener: (event: never) => void) {
      listeners.set(type, listener);
    },
    skipWaiting: async () => undefined,
    clients: { claim: async () => undefined },
  };
  vm.runInNewContext(source, {
    self: workerSelf,
    ArrayBuffer,
    Error,
    Map,
    Promise,
    ReadableStream,
    Response,
    URL,
    Uint8Array,
    clearTimeout,
    decodeURIComponent,
    encodeURIComponent,
    setTimeout,
  });
  return {
    message: listeners.get("message") as WorkerHandlers["message"],
    fetch: listeners.get("fetch") as WorkerHandlers["fetch"],
  };
}

test("local and production apps ship the same export Service Worker", async () => {
  const [rootSource, prodSource] = await Promise.all([
    readFile(rootWorkerPath, "utf8"),
    readFile(prodWorkerPath, "utf8"),
  ]);
  assert.equal(prodSource, rootSource);
});

test("Service Worker streams transferred chunks with download headers", async () => {
  const handlers = await loadWorker();
  const channel = new MessageChannel();
  const protocol = "volleycut-export-v3";
  const chunks = [new Uint8Array([0, 1, 2]), new Uint8Array([3, 4])];
  let nextChunk = 0;

  channel.port1.onmessage = (event) => {
    if (event.data?.protocol !== protocol) return;
    if (event.data.type === "consumer-attached") {
      channel.port1.postMessage({
        protocol,
        type: "start",
        attempt: event.data.attempt,
      });
      return;
    }
    if (event.data.type !== "pull") return;
    const chunk = chunks[nextChunk++];
    if (!chunk) {
      channel.port1.postMessage({ protocol, type: "close" });
      return;
    }
    const buffer = chunk.slice().buffer;
    channel.port1.postMessage(
      { protocol, type: "chunk", attempt: event.data.attempt, buffer },
      [buffer],
    );
  };
  channel.port1.start();

  handlers.message({
    data: {
      protocol,
      type: "prepare",
      token: "test-token",
      fileName: "match final.mp4",
    },
    ports: [channel.port2],
  });

  let responsePromise: Promise<Response> | null = null;
  handlers.fetch({
    request: new Request(
      "https://example.test/__volleycut_export_download__/test-token",
    ),
    respondWith(response) {
      responsePromise = response;
    },
  });
  assert.ok(responsePromise);
  const response = await responsePromise;
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-type"), "video/mp4");
  assert.match(
    response.headers.get("content-disposition") ?? "",
    /match final\.mp4/,
  );
  assert.equal(response.headers.get("cache-control"), "no-store, no-transform");

  const bytes = new Uint8Array(await response.arrayBuffer());
  assert.deepEqual([...bytes], [0, 1, 2, 3, 4]);
  assert.equal(
    nextChunk,
    3,
    "the worker must pull each chunk and then request EOF",
  );
  channel.port1.close();
});

test("a download fetch may race ahead of stream preparation", async () => {
  const handlers = await loadWorker();
  const channel = new MessageChannel();
  const protocol = "volleycut-export-v3";
  channel.port1.onmessage = (event) => {
    if (event.data?.protocol !== protocol) return;
    if (event.data.type === "consumer-attached") {
      channel.port1.postMessage({
        protocol,
        type: "start",
        attempt: event.data.attempt,
      });
    } else if (event.data.type === "pull") {
      channel.port1.postMessage({ protocol, type: "close" });
    }
  };
  channel.port1.start();
  let responsePromise: Promise<Response> | null = null;
  handlers.fetch({
    request: new Request(
      "https://example.test/__volleycut_export_download__/racing-token",
    ),
    respondWith(response) {
      responsePromise = response;
    },
  });
  assert.ok(responsePromise);
  handlers.message({
    data: {
      protocol,
      type: "prepare",
      token: "racing-token",
      fileName: "racing.mp4",
    },
    ports: [channel.port2],
  });
  const response = await responsePromise;
  assert.equal(response.status, 200);
  assert.equal((await response.arrayBuffer()).byteLength, 0);
  channel.port1.close();
});

test("a probe consumer can detach before bytes and reattach to the same export", async () => {
  const handlers = await loadWorker();
  const channel = new MessageChannel();
  const protocol = "volleycut-export-v3";
  const attempts: number[] = [];
  let firstAttachedResolve: (() => void) | null = null;
  const firstAttached = new Promise<void>((resolve) => {
    firstAttachedResolve = resolve;
  });

  channel.port1.onmessage = (event) => {
    if (event.data?.protocol !== protocol) return;
    if (event.data.type === "consumer-attached") {
      attempts.push(event.data.attempt);
      if (event.data.attempt === 1) {
        firstAttachedResolve?.();
      } else {
        channel.port1.postMessage({
          protocol,
          type: "start",
          attempt: event.data.attempt,
        });
      }
    } else if (event.data.type === "pull") {
      channel.port1.postMessage({ protocol, type: "close" });
    }
  };
  channel.port1.start();
  handlers.message({
    data: {
      protocol,
      type: "prepare",
      token: "reattach-token",
      fileName: "reattach.mp4",
    },
    ports: [channel.port2],
  });

  const fetchDownload = () => {
    let responsePromise: Promise<Response> | null = null;
    handlers.fetch({
      request: new Request(
        "https://example.test/__volleycut_export_download__/reattach-token",
      ),
      respondWith(response) {
        responsePromise = response;
      },
    });
    assert.ok(responsePromise);
    return responsePromise;
  };

  const firstResponse = await fetchDownload();
  const firstReader = firstResponse.body!.getReader();
  const firstRead = firstReader.read();
  await firstAttached;
  await firstReader.cancel("browser converting navigation to download");
  await firstRead;

  const secondResponse = await fetchDownload();
  assert.equal((await secondResponse.arrayBuffer()).byteLength, 0);
  assert.deepEqual(attempts, [1, 2]);
  channel.port1.close();
});

test("a native download refetch after the MP4 header receives a replayed prefix", async () => {
  const handlers = await loadWorker();
  const channel = new MessageChannel();
  const protocol = "volleycut-export-v3";
  const header = new Uint8Array(28).fill(7);
  const remainder = new Uint8Array([8, 9, 10]);
  const attempts: number[] = [];
  let activeAttempt = 0;
  let firstChunkSent = false;
  let remainderSent = false;
  let closeSent = false;

  channel.port1.onmessage = (event) => {
    if (event.data?.protocol !== protocol) return;
    if (event.data.type === "consumer-attached") {
      activeAttempt = event.data.attempt;
      attempts.push(activeAttempt);
      channel.port1.postMessage({
        protocol,
        type: "start",
        attempt: activeAttempt,
      });
      if (activeAttempt === 2) {
        // Simulate a producer chunk that was granted to the iframe consumer immediately before
        // WebKit replaced it with the native download request.
        remainderSent = true;
        const buffer = remainder.slice().buffer;
        channel.port1.postMessage(
          { protocol, type: "chunk", attempt: 1, buffer },
          [buffer],
        );
      }
    } else if (event.data.type === "pull" && activeAttempt === 1) {
      if (firstChunkSent) return;
      firstChunkSent = true;
      const buffer = header.slice().buffer;
      channel.port1.postMessage(
        { protocol, type: "chunk", attempt: activeAttempt, buffer },
        [buffer],
      );
    } else if (event.data.type === "pull" && activeAttempt === 2) {
      if (remainderSent && !closeSent) {
        closeSent = true;
        channel.port1.postMessage({ protocol, type: "close" });
      }
    }
  };
  channel.port1.start();
  handlers.message({
    data: {
      protocol,
      type: "prepare",
      token: "native-refetch-token",
      fileName: "native-refetch.mp4",
    },
    ports: [channel.port2],
  });

  const fetchDownload = () => {
    let responsePromise: Promise<Response> | null = null;
    handlers.fetch({
      request: new Request(
        "https://example.test/__volleycut_export_download__/native-refetch-token",
      ),
      respondWith(response) {
        responsePromise = response;
      },
    });
    assert.ok(responsePromise);
    return responsePromise;
  };

  const probeResponse = await fetchDownload();
  const probeReader = probeResponse.body!.getReader();
  const probeChunk = await probeReader.read();
  assert.deepEqual([...probeChunk.value!], [...header]);

  const nativeResponse = await fetchDownload();
  await assert.rejects(probeReader.read(), /replaced the download consumer/);
  const nativeBytes = new Uint8Array(await nativeResponse.arrayBuffer());
  assert.deepEqual([...nativeBytes], [...header, ...remainder]);
  assert.deepEqual(attempts, [1, 2]);
  channel.port1.close();
});

test("two exports can run sequentially through one active worker", async () => {
  const handlers = await loadWorker();
  const protocol = "volleycut-export-v3";

  for (const [index, expectedByte] of [71, 72].entries()) {
    const token = `repeat-${index}`;
    const channel = new MessageChannel();
    let sent = false;
    channel.port1.onmessage = (event) => {
      if (event.data?.protocol !== protocol) return;
      if (event.data.type === "consumer-attached") {
        channel.port1.postMessage({
          protocol,
          type: "start",
          attempt: event.data.attempt,
        });
      } else if (event.data.type === "pull" && !sent) {
        sent = true;
        const buffer = new Uint8Array([expectedByte]).buffer;
        channel.port1.postMessage(
          { protocol, type: "chunk", attempt: event.data.attempt, buffer },
          [buffer],
        );
      } else if (event.data.type === "pull") {
        channel.port1.postMessage({ protocol, type: "close" });
      }
    };
    channel.port1.start();
    handlers.message({
      data: {
        protocol,
        type: "prepare",
        token,
        fileName: `repeat-${index}.mp4`,
      },
      ports: [channel.port2],
    });

    let responsePromise: Promise<Response> | null = null;
    handlers.fetch({
      request: new Request(
        `https://example.test/__volleycut_export_download__/${token}`,
      ),
      respondWith(response) {
        responsePromise = response;
      },
    });
    assert.ok(responsePromise);
    const bytes = new Uint8Array(await (await responsePromise).arrayBuffer());
    assert.deepEqual([...bytes], [expectedByte]);
    channel.port1.close();
  }
});
