const PROTOCOL = "volleycut-export-v3";
const DOWNLOAD_MARKER = "/__volleycut_export_download__/";
const SESSION_TIMEOUT_MS = 120_000;
const REATTACH_TIMEOUT_MS = 10_000;
const MAX_REPLAY_BYTES = 8 * 1024 * 1024;
const MAX_CONSUMER_ATTEMPTS = 3;
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

function failSession(session, reason) {
  clearTimeout(session.reattachExpiry);
  const error = reason instanceof Error ? reason : new Error(String(reason));
  session.failed = error;
  session.pendingPull?.reject(error);
  session.pendingPull = null;
  try {
    session.controller?.error(error);
  } catch {
    // The browser may already have closed the replaced response.
  }
  session.port.postMessage({
    protocol: PROTOCOL,
    type: "cancel",
    reason: error.message,
  });
  session.port.close();
  sessions.delete(session.token);
}

function detachConsumer(session, attempt) {
  if (attempt !== session.attempt) return;
  session.pendingPull?.resolve();
  session.pendingPull = null;
  session.controller = null;
  session.claimed = false;
  session.consumerStarted = false;
  session.consumerAnnounced = false;
  session.port.postMessage({
    protocol: PROTOCOL,
    type: "consumer-detached",
    attempt,
  });
  clearTimeout(session.reattachExpiry);
  session.reattachExpiry = setTimeout(() => {
    failSession(session, "The browser did not reattach the native download.");
  }, REATTACH_TIMEOUT_MS);
}

function fulfillPendingPull(session, attempt) {
  const pending = session.pendingPull;
  if (
    !pending ||
    attempt !== session.attempt ||
    !session.claimed ||
    !session.consumerStarted
  )
    return;

  if (session.replayable && session.replayIndex < session.replayChunks.length) {
    const chunk = session.replayChunks[session.replayIndex++];
    session.pendingPull = null;
    try {
      session.controller.enqueue(chunk.slice());
      pending.resolve();
    } catch (cause) {
      pending.reject(cause);
    }
    return;
  }

  session.port.postMessage({ protocol: PROTOCOL, type: "pull", attempt });
}

function downloadResponse(session) {
  if (session.claimed) {
    if (!session.replayable || session.attempt >= MAX_CONSUMER_ATTEMPTS) {
      return errorResponse("This export download cannot be replayed.", 409);
    }

    // WebKit may first consume the iframe navigation and then issue another request after the
    // user accepts its native download prompt. Replace that probe and replay the bounded prefix.
    const replacedAttempt = session.attempt;
    const replacedPull = session.pendingPull;
    session.pendingPull = null;
    try {
      session.controller?.error(
        new Error("The browser replaced the download consumer."),
      );
    } catch {
      // The native download may close the iframe response before issuing its replacement fetch.
    }
    replacedPull?.resolve();
    session.controller = null;
    session.claimed = false;
    session.consumerStarted = false;
    session.consumerAnnounced = false;
    session.port.postMessage({
      protocol: PROTOCOL,
      type: "consumer-detached",
      attempt: replacedAttempt,
    });
  }
  session.claimed = true;
  clearTimeout(session.expiry);
  clearTimeout(session.reattachExpiry);
  session.attempt += 1;
  session.consumerStarted = false;
  session.consumerAnnounced = false;
  session.replayIndex = 0;
  const attempt = session.attempt;

  const stream = new ReadableStream({
    pull(controller) {
      if (attempt !== session.attempt) {
        return;
      }
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
        fulfillPendingPull(session, attempt);
      }
      return promise;
    },
    cancel(reason) {
      if (attempt !== session.attempt) return;
      if (session.replayable && session.attempt < MAX_CONSUMER_ATTEMPTS) {
        detachConsumer(session, attempt);
        return;
      }
      failSession(
        session,
        reason instanceof Error
          ? reason
          : new Error(String(reason || "Download canceled")),
      );
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "video/mp4",
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
    replayChunks: [],
    replayBytes: 0,
    replayable: true,
    replayIndex: 0,
    closed: false,
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
    if (incoming?.protocol !== PROTOCOL || session.closed) return;
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
      fulfillPendingPull(session, session.attempt);
    } else if (incoming.type === "chunk") {
      if (
        !(incoming.buffer instanceof ArrayBuffer) ||
        !Number.isInteger(incoming.attempt) ||
        incoming.attempt < 1 ||
        incoming.attempt > session.attempt
      )
        return;
      const chunk = new Uint8Array(incoming.buffer);
      const canDeliverNow =
        pending &&
        session.claimed &&
        session.consumerStarted &&
        session.replayIndex === session.replayChunks.length;

      if (
        session.replayable &&
        session.replayBytes + chunk.byteLength <= MAX_REPLAY_BYTES
      ) {
        // The chunk may have been granted to a consumer just before WebKit replaced it. Retain
        // it first, then let the current consumer replay bytes in strict file order.
        session.replayChunks.push(chunk);
        session.replayBytes += chunk.byteLength;
        fulfillPendingPull(session, session.attempt);
      } else if (canDeliverNow) {
        session.replayable = false;
        session.replayChunks = [];
        session.replayBytes = 0;
        session.replayIndex = 0;
        session.pendingPull = null;
        try {
          session.controller.enqueue(chunk);
          pending.resolve();
        } catch (cause) {
          pending.reject(cause);
        }
      } else {
        failSession(
          session,
          "The browser replaced the download after its replay buffer was exhausted.",
        );
      }
    } else if (incoming.type === "close") {
      session.closed = true;
      clearTimeout(session.reattachExpiry);
      session.pendingPull = null;
      if (!session.claimed || !session.controller) {
        failSession(
          session,
          "The browser detached before the streamed response finished.",
        );
        return;
      }
      try {
        session.controller.close();
        pending?.resolve();
      } catch (cause) {
        pending?.reject(cause);
        failSession(session, cause);
        return;
      }
      port.postMessage({ protocol: PROTOCOL, type: "closed" });
      sessions.delete(token);
      port.close();
    } else if (incoming.type === "error") {
      session.closed = true;
      clearTimeout(session.reattachExpiry);
      const error = new Error(
        incoming.reason || "The video encoder stopped the export.",
      );
      session.failed = error;
      session.pendingPull = null;
      try {
        session.controller?.error(error);
      } catch {
        // The browser may have already closed its response while the encoder was aborting.
      }
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
