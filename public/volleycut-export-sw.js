const PROTOCOL = "volleycut-export-v2";
const DOWNLOAD_MARKER = "/__volleycut_export_download__/";
const SESSION_TIMEOUT_MS = 120_000;
const REATTACH_TIMEOUT_MS = 10_000;
const sessions = new Map();
const waitingFetches = new Map();

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) =>
  event.waitUntil(self.clients.claim()),
);

function contentDisposition(fileName) {
  const ascii = String(fileName || "volleycut-export.mp4")
    .replace(/[^\x20-\x7e]/g, "_")
    .replace(/["\\]/g, "_");
  const encoded = encodeURIComponent(
    String(fileName || "volleycut-export.mp4"),
  ).replace(
    /[!'()*]/g,
    (character) => `%${character.charCodeAt(0).toString(16).toUpperCase()}`,
  );
  return `attachment; filename="${ascii}"; filename*=UTF-8''${encoded}`;
}

function errorResponse(message, status = 500) {
  return new Response(message, {
    status,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

function downloadResponse(session) {
  if (session.claimed)
    return errorResponse("This export download was already claimed.", 410);
  session.claimed = true;
  clearTimeout(session.expiry);
  clearTimeout(session.reattachExpiry);
  session.attempt += 1;
  session.consumerStarted = false;
  session.consumerAnnounced = false;
  const attempt = session.attempt;

  const stream = new ReadableStream({
    pull(controller) {
      if (session.failed) throw session.failed;
      if (session.pendingPull) return session.pendingPull.promise;
      let resolvePull;
      let rejectPull;
      const promise = new Promise((resolve, reject) => {
        resolvePull = resolve;
        rejectPull = reject;
      });
      session.pendingPull = {
        promise,
        resolve: resolvePull,
        reject: rejectPull,
      };
      session.controller = controller;
      if (!session.consumerAnnounced) {
        session.consumerAnnounced = true;
        session.port.postMessage({
          protocol: PROTOCOL,
          type: "consumer-attached",
          attempt,
        });
      } else if (session.consumerStarted) {
        session.port.postMessage({ protocol: PROTOCOL, type: "pull" });
      }
      return promise;
    },
    cancel(reason) {
      if (!session.consumerStarted) {
        session.pendingPull?.resolve();
        session.pendingPull = null;
        session.controller = null;
        session.claimed = false;
        session.consumerAnnounced = false;
        session.port.postMessage({
          protocol: PROTOCOL,
          type: "consumer-detached",
          attempt,
        });
        session.reattachExpiry = setTimeout(() => {
          session.port.postMessage({
            protocol: PROTOCOL,
            type: "cancel",
            reason: "The browser did not reattach the native download.",
          });
          session.port.close();
          sessions.delete(session.token);
        }, REATTACH_TIMEOUT_MS);
        return;
      }
      session.port.postMessage({
        protocol: PROTOCOL,
        type: "cancel",
        reason:
          reason instanceof Error
            ? reason.message
            : String(reason || "Download canceled"),
      });
      session.port.close();
      sessions.delete(session.token);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Disposition": contentDisposition(session.fileName),
      "Cache-Control": "no-store, no-transform",
      "X-Content-Type-Options": "nosniff",
      "Content-Security-Policy": "default-src 'none'",
    },
  });
}

function acceptSession(message, port) {
  const token = typeof message.token === "string" ? message.token : "";
  if (!token || !port || sessions.has(token)) return;

  const session = {
    token,
    fileName:
      typeof message.fileName === "string"
        ? message.fileName
        : "volleycut-export.mp4",
    port,
    claimed: false,
    controller: null,
    pendingPull: null,
    failed: null,
    expiry: null,
    reattachExpiry: null,
    attempt: 0,
    consumerStarted: false,
    consumerAnnounced: false,
  };
  session.expiry = setTimeout(() => {
    sessions.delete(token);
    const waiter = waitingFetches.get(token);
    if (waiter) {
      waitingFetches.delete(token);
      waiter.resolve(
        errorResponse("The export stream was not started in time.", 408),
      );
    }
    port.close();
  }, SESSION_TIMEOUT_MS);

  port.onmessage = (event) => {
    const incoming = event.data;
    if (incoming?.protocol !== PROTOCOL) return;
    const pending = session.pendingPull;
    if (incoming.type === "start") {
      if (
        !pending ||
        !session.claimed ||
        incoming.attempt !== session.attempt ||
        session.consumerStarted
      )
        return;
      session.consumerStarted = true;
      port.postMessage({ protocol: PROTOCOL, type: "pull" });
    } else if (incoming.type === "chunk") {
      if (!pending || !(incoming.buffer instanceof ArrayBuffer)) return;
      session.pendingPull = null;
      try {
        session.controller.enqueue(new Uint8Array(incoming.buffer));
        pending.resolve();
      } catch (cause) {
        pending.reject(cause);
      }
    } else if (incoming.type === "close") {
      clearTimeout(session.reattachExpiry);
      session.pendingPull = null;
      session.controller?.close();
      pending?.resolve();
      sessions.delete(token);
      port.close();
    } else if (incoming.type === "error") {
      clearTimeout(session.reattachExpiry);
      const error = new Error(
        incoming.reason || "The video encoder stopped the export.",
      );
      session.failed = error;
      session.pendingPull = null;
      session.controller?.error(error);
      pending?.reject(error);
      sessions.delete(token);
      port.close();
    }
  };
  port.start();
  sessions.set(token, session);
  port.postMessage({ protocol: PROTOCOL, type: "prepared" });

  const waiter = waitingFetches.get(token);
  if (waiter) {
    waitingFetches.delete(token);
    waiter.resolve(downloadResponse(session));
  }
}

self.addEventListener("message", (event) => {
  const message = event.data;
  if (message?.protocol !== PROTOCOL || message.type !== "prepare") return;
  acceptSession(message, event.ports[0]);
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  const markerIndex = url.pathname.lastIndexOf(DOWNLOAD_MARKER);
  if (event.request.method !== "GET" || markerIndex < 0) return;

  const token = decodeURIComponent(
    url.pathname.slice(markerIndex + DOWNLOAD_MARKER.length),
  );
  const session = sessions.get(token);
  if (session) {
    event.respondWith(Promise.resolve(downloadResponse(session)));
    return;
  }

  event.respondWith(
    new Promise((resolve) => {
      const timeout = setTimeout(() => {
        waitingFetches.delete(token);
        resolve(errorResponse("No matching export stream was prepared.", 404));
      }, 5_000);
      waitingFetches.set(token, {
        resolve(response) {
          clearTimeout(timeout);
          resolve(response);
        },
      });
    }),
  );
});
