export type WakeLockState =
  | "idle"
  | "requesting"
  | "active"
  | "paused"
  | "released"
  | "unavailable"
  | "denied";

export async function holdScreenWakeLock(
  onState: (state: WakeLockState) => void,
): Promise<() => Promise<void>> {
  if (!("wakeLock" in navigator)) {
    onState("unavailable");
    return async () => undefined;
  }
  let stopped = false;
  let acquired = false;
  let sentinel: WakeLockSentinel | null = null;
  const acquire = async () => {
    if (stopped || document.visibilityState !== "visible" || sentinel) return;
    onState("requesting");
    try {
      sentinel = await navigator.wakeLock.request("screen");
      if (stopped) {
        await sentinel.release();
        sentinel = null;
        return;
      }
      acquired = true;
      onState("active");
      sentinel.addEventListener(
        "release",
        () => {
          sentinel = null;
          if (stopped) return;
          if (document.visibilityState === "visible") void acquire();
          else onState("paused");
        },
        { once: true },
      );
    } catch {
      if (!stopped) onState(document.visibilityState === "visible" ? "denied" : "paused");
    }
  };
  const handleVisibility = () => {
    if (document.visibilityState === "visible" && !sentinel) void acquire();
    else if (document.visibilityState !== "visible") onState("paused");
  };
  document.addEventListener("visibilitychange", handleVisibility);
  await acquire();
  return async () => {
    stopped = true;
    document.removeEventListener("visibilitychange", handleVisibility);
    await sentinel?.release().catch(() => undefined);
    sentinel = null;
    if (acquired) onState("released");
  };
}
