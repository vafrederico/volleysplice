/** Stop this consumer's wait without cancelling or evicting shared work.
 * Keep both settlement handlers attached so an abandoned load can resolve or
 * reject later without resuming its consumer or becoming unhandled.
 */
export function awaitWithAbort<T>(pending: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return pending;
  return new Promise<T>((resolve, reject) => {
    const cleanup = () => signal.removeEventListener("abort", abort);
    const abort = () => {
      cleanup();
      reject(signal.reason ?? new DOMException("Analysis cancelled", "AbortError"));
    };
    pending.then(
      (value) => { cleanup(); resolve(value); },
      (cause) => { cleanup(); reject(cause); },
    );
    if (signal.aborted) abort();
    else signal.addEventListener("abort", abort, { once: true });
  });
}
