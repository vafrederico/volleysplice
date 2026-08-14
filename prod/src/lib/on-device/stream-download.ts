const PROTOCOL = "volleycut-export-v2";
const CONSUMER_SETTLE_MS = 500;
const PREPARE_TIMEOUT_MS = 15_000;
const PULL_TIMEOUT_MS = 60_000;
const IFRAME_CLEANUP_MS = 60_000;

type ReadyRegistration = {
  registration: ServiceWorkerRegistration;
  worker: ServiceWorker;
};

export type StreamDownloadReadiness = {
  ready: boolean;
  reason: string | null;
};

export type StreamDownload = {
  writable: WritableStream<Uint8Array>;
  cancel(reason?: unknown): Promise<void>;
};

let readyRegistration: ReadyRegistration | null = null;
let registrationPromise: Promise<StreamDownloadReadiness> | null = null;

export function supportsServiceWorkerStreamDownload(): boolean {
  return (
    typeof window !== "undefined" &&
    window.isSecureContext &&
    "serviceWorker" in navigator &&
    typeof MessageChannel !== "undefined" &&
    typeof WritableStream !== "undefined"
  );
}

function waitForActivation(
  registration: ServiceWorkerRegistration,
): Promise<ServiceWorker> {
  const candidate = registration.installing ?? registration.waiting;
  if (!candidate) {
    if (registration.active) return Promise.resolve(registration.active);
    return Promise.reject(
      new Error("The export service worker did not install."),
    );
  }

  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      candidate.removeEventListener("statechange", onStateChange);
      reject(
        new Error("The export service worker did not become active in time."),
      );
    }, 15_000);
    const onStateChange = () => {
      if (candidate.state === "activated") {
        window.clearTimeout(timeout);
        candidate.removeEventListener("statechange", onStateChange);
        resolve(candidate);
      } else if (candidate.state === "redundant") {
        window.clearTimeout(timeout);
        candidate.removeEventListener("statechange", onStateChange);
        reject(
          new Error("The export service worker installation was replaced."),
        );
      }
    };
    candidate.addEventListener("statechange", onStateChange);
    onStateChange();
  });
}

export function prepareServiceWorkerStreamDownload(
  scriptUrl: string,
): Promise<StreamDownloadReadiness> {
  if (registrationPromise) return registrationPromise;
  registrationPromise = (async () => {
    if (!supportsServiceWorkerStreamDownload()) {
      return {
        ready: false,
        reason: window.isSecureContext
          ? "This browser does not expose the required Service Worker APIs."
          : "Direct streaming requires HTTPS.",
      };
    }

    try {
      const absoluteScriptUrl = new URL(scriptUrl, window.location.href);
      const registration = await navigator.serviceWorker.register(
        absoluteScriptUrl,
        {
          scope: new URL("./", absoluteScriptUrl).pathname,
        },
      );
      await registration.update().catch(() => undefined);
      const worker = await waitForActivation(registration);
      readyRegistration = { registration, worker };
      return { ready: true, reason: null };
    } catch (cause) {
      readyRegistration = null;
      registrationPromise = null;
      return {
        ready: false,
        reason: cause instanceof Error ? cause.message : String(cause),
      };
    }
  })();
  return registrationPromise;
}

function randomToken(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

function exactTransferBuffer(chunk: Uint8Array): ArrayBuffer {
  if (
    chunk.buffer instanceof ArrayBuffer &&
    chunk.byteOffset === 0 &&
    chunk.byteLength === chunk.buffer.byteLength
  ) {
    return chunk.buffer;
  }
  return chunk.slice().buffer;
}

export function startServiceWorkerStreamDownload(
  fileName: string,
): StreamDownload {
  const ready = readyRegistration;
  if (!ready?.worker || ready.worker.state !== "activated") {
    throw new Error(
      "Experimental direct download is not ready. Use compatible export and retry.",
    );
  }

  const token = randomToken();
  const channel = new MessageChannel();
  let pullCredits = 0;
  let failed: Error | null = null;
  let downloadFrame: HTMLIFrameElement | null = null;
  let consumerSettleTimer: number | null = null;
  let prepareTimer: number | null = null;
  let pendingPull: {
    resolve: () => void;
    reject: (error: Error) => void;
    timeout: number;
  } | null = null;

  const fail = (error: Error) => {
    failed = error;
    if (consumerSettleTimer !== null) window.clearTimeout(consumerSettleTimer);
    if (prepareTimer !== null) window.clearTimeout(prepareTimer);
    if (pendingPull) {
      window.clearTimeout(pendingPull.timeout);
      pendingPull.reject(error);
      pendingPull = null;
    }
  };

  channel.port1.onmessage = (event: MessageEvent) => {
    const message = event.data as {
      protocol?: string;
      type?: string;
      reason?: string;
      attempt?: number;
    };
    if (message?.protocol !== PROTOCOL) return;
    if (message.type === "prepared") {
      if (prepareTimer !== null) {
        window.clearTimeout(prepareTimer);
        prepareTimer = null;
      }
      if (downloadFrame) return;
      const downloadUrl = new URL(
        `__volleycut_export_download__/${token}`,
        ready.registration.scope,
      );
      downloadFrame = document.createElement("iframe");
      downloadFrame.hidden = true;
      downloadFrame.setAttribute("aria-hidden", "true");
      downloadFrame.src = downloadUrl.href;
      document.body.append(downloadFrame);
    } else if (
      message.type === "consumer-attached" &&
      Number.isInteger(message.attempt)
    ) {
      if (consumerSettleTimer !== null)
        window.clearTimeout(consumerSettleTimer);
      const attempt = message.attempt;
      consumerSettleTimer = window.setTimeout(() => {
        consumerSettleTimer = null;
        channel.port1.postMessage({
          protocol: PROTOCOL,
          type: "start",
          attempt,
        });
      }, CONSUMER_SETTLE_MS);
    } else if (message.type === "consumer-detached") {
      if (consumerSettleTimer !== null) {
        window.clearTimeout(consumerSettleTimer);
        consumerSettleTimer = null;
      }
    } else if (message.type === "pull") {
      if (pendingPull) {
        const waiter = pendingPull;
        pendingPull = null;
        window.clearTimeout(waiter.timeout);
        waiter.resolve();
      } else {
        pullCredits += 1;
      }
    } else if (message.type === "cancel") {
      fail(
        new Error(
          message.reason || "The browser canceled the streamed download.",
        ),
      );
    }
  };
  channel.port1.start();
  prepareTimer = window.setTimeout(() => {
    fail(
      new Error(
        "The browser did not prepare the streamed download. Retry compatible export.",
      ),
    );
  }, PREPARE_TIMEOUT_MS);

  const waitForPull = () => {
    if (failed) return Promise.reject(failed);
    if (pullCredits > 0) {
      pullCredits -= 1;
      return Promise.resolve();
    }
    return new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        pendingPull = null;
        const error = new Error(
          "The browser did not begin the streamed download. Retry with compatible export.",
        );
        failed = error;
        reject(error);
      }, PULL_TIMEOUT_MS);
      pendingPull = { resolve, reject, timeout };
    });
  };

  const writable = new WritableStream<Uint8Array>({
    async write(chunk) {
      await waitForPull();
      if (failed) throw failed;
      const buffer = exactTransferBuffer(chunk);
      channel.port1.postMessage({ protocol: PROTOCOL, type: "chunk", buffer }, [
        buffer,
      ]);
    },
    close() {
      if (consumerSettleTimer !== null)
        window.clearTimeout(consumerSettleTimer);
      if (prepareTimer !== null) window.clearTimeout(prepareTimer);
      channel.port1.postMessage({ protocol: PROTOCOL, type: "close" });
      channel.port1.close();
      const frame = downloadFrame;
      if (frame) window.setTimeout(() => frame.remove(), IFRAME_CLEANUP_MS);
    },
    abort(reason) {
      if (consumerSettleTimer !== null)
        window.clearTimeout(consumerSettleTimer);
      if (prepareTimer !== null) window.clearTimeout(prepareTimer);
      channel.port1.postMessage({
        protocol: PROTOCOL,
        type: "error",
        reason:
          reason instanceof Error
            ? reason.message
            : String(reason ?? "Export canceled"),
      });
      channel.port1.close();
      downloadFrame?.remove();
      downloadFrame = null;
    },
  });

  ready.worker.postMessage(
    { protocol: PROTOCOL, type: "prepare", token, fileName },
    [channel.port2],
  );

  return {
    writable,
    async cancel(reason) {
      await writable.abort(reason).catch(() => undefined);
    },
  };
}
