"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { formatTime } from "@/lib/edit-list";
import { openUrlMedia, type OpenedMedia } from "@/lib/on-device/media";
import { analyzeOpenedMedia } from "@/lib/on-device/pipeline";
import type {
  AnalysisProgress,
  NormalizedRoi,
  OnDeviceMediaInfo,
} from "@/lib/on-device/types";

import styles from "./on-device-batch.module.css";

const BATCH_URL = "/api/on-device-batch";

type BatchPhase = "loading" | "ready" | "running" | "complete" | "error" | "blocked";

type BatchVideo = {
  id: string;
  filename: string;
  originalFilename: string;
  environment: string;
  duration: number;
  size: number;
  roi: NormalizedRoi;
  mediaUrl: string;
  completed: boolean;
};

type BatchCatalog = {
  schemaVersion: 1;
  modelId: string;
  featurePath: "training-proxy";
  videos: BatchVideo[];
};

type VisibleProgress = AnalysisProgress | {
  stage: "opening" | "saving";
  completed: number;
  total: number;
  detail: string;
};

type Failure = { recordingId: string | null; message: string };

function compactBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value.toFixed(unit < 2 ? 0 : 1)} ${units[unit]}`;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : "The browser batch failed unexpectedly.";
}

function tokenFromFragment(): string | null {
  const token = new URLSearchParams(window.location.hash.slice(1)).get("token")?.trim();
  return token || null;
}

function authorization(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isRoi(value: unknown): value is NormalizedRoi {
  if (!isRecord(value)) return false;
  const fields = [value.x, value.y, value.width, value.height];
  return (
    fields.every((field) => typeof field === "number" && Number.isFinite(field)) &&
    (value.x as number) >= 0 &&
    (value.y as number) >= 0 &&
    (value.width as number) > 0 &&
    (value.height as number) > 0 &&
    (value.x as number) + (value.width as number) <= 1.000001 &&
    (value.y as number) + (value.height as number) <= 1.000001
  );
}

function parseCatalog(value: unknown): BatchCatalog {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    typeof value.modelId !== "string" ||
    value.featurePath !== "training-proxy" ||
    !Array.isArray(value.videos)
  ) {
    throw new Error("The batch catalog has an unexpected format.");
  }
  for (const candidate of value.videos) {
    if (
      !isRecord(candidate) ||
      typeof candidate.id !== "string" ||
      typeof candidate.filename !== "string" ||
      typeof candidate.originalFilename !== "string" ||
      typeof candidate.environment !== "string" ||
      typeof candidate.duration !== "number" ||
      !Number.isFinite(candidate.duration) ||
      candidate.duration <= 0 ||
      typeof candidate.size !== "number" ||
      !Number.isFinite(candidate.size) ||
      candidate.size <= 0 ||
      !isRoi(candidate.roi) ||
      typeof candidate.mediaUrl !== "string" ||
      typeof candidate.completed !== "boolean"
    ) {
      throw new Error("The batch catalog contains an invalid video entry.");
    }
  }
  return value as BatchCatalog;
}

async function responseError(response: Response): Promise<string> {
  try {
    const value: unknown = await response.json();
    if (isRecord(value) && typeof value.error === "string" && value.error.trim()) {
      return value.error.trim();
    }
  } catch {
    // Fall through to the status-only error.
  }
  return `Request failed with status ${response.status}.`;
}

async function fetchCatalog(token: string): Promise<BatchCatalog> {
  const response = await fetch(BATCH_URL, {
    headers: authorization(token),
    cache: "no-store",
  });
  if (!response.ok) throw new Error(await responseError(response));
  return parseCatalog(await response.json());
}

function sameOriginMediaUrl(value: string): URL {
  const url = new URL(value, window.location.origin);
  if (url.origin !== window.location.origin) {
    throw new Error("The batch catalog returned a cross-origin media URL.");
  }
  url.hash = "";
  return url;
}

function progressPercent(progress: VisibleProgress | null): number {
  if (!progress || !Number.isFinite(progress.total) || progress.total <= 0) return 0;
  return Math.max(0, Math.min(100, progress.completed / progress.total * 100));
}

function markCompleted(catalog: BatchCatalog, recordingId: string): BatchCatalog {
  return {
    ...catalog,
    videos: catalog.videos.map((video) =>
      video.id === recordingId ? { ...video, completed: true } : video
    ),
  };
}

export function OnDeviceBatchClient() {
  const [catalog, setCatalog] = useState<BatchCatalog | null>(null);
  const [phase, setPhase] = useState<BatchPhase>("loading");
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [progress, setProgress] = useState<VisibleProgress | null>(null);
  const [currentMediaInfo, setCurrentMediaInfo] = useState<OnDeviceMediaInfo | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [canResume, setCanResume] = useState(false);
  const tokenRef = useRef<string | null>(null);
  const runningRef = useRef(false);
  const autoStartedRef = useRef(false);

  const runBatch = useCallback(async (initialCatalog: BatchCatalog) => {
    const token = tokenRef.current;
    if (!token || runningRef.current) return;
    runningRef.current = true;
    setFailure(null);
    setPhase("running");
    let workingCatalog = initialCatalog;
    try {
      const remaining = workingCatalog.videos.filter((video) => !video.completed);
      if (remaining.length === 0) {
        setCurrentId(null);
        setProgress(null);
        setPhase("complete");
        return;
      }

      for (const video of remaining) {
        setCurrentId(video.id);
        setCurrentMediaInfo(null);
        setProgress({
          stage: "opening",
          completed: 0,
          total: 1,
          detail: "Opening the ranged proxy with Mediabunny",
        });
        let media: OpenedMedia | null = null;
        try {
          media = await openUrlMedia(sameOriginMediaUrl(video.mediaUrl), {
            headers: authorization(token),
            cache: "no-store",
            credentials: "same-origin",
          });
          setCurrentMediaInfo(media.info);
          const analysis = await analyzeOpenedMedia(
            media,
            video.roi,
            "training-proxy",
            (next) => setProgress(next),
          );
          if (analysis.modelId !== workingCatalog.modelId) {
            throw new Error(
              `The browser loaded ${analysis.modelId}, but the batch expects ${workingCatalog.modelId}.`,
            );
          }
          setProgress({
            stage: "saving",
            completed: 0,
            total: 1,
            detail: `Saving ${analysis.intervals.length} exact prediction ranges`,
          });
          const response = await fetch(BATCH_URL, {
            method: "POST",
            headers: {
              ...authorization(token),
              "Content-Type": "application/json",
            },
            cache: "no-store",
            body: JSON.stringify({
              recordingId: video.id,
              modelId: analysis.modelId,
              featurePath: analysis.featurePath,
              media: media.info satisfies OnDeviceMediaInfo,
              intervals: analysis.intervals,
              provenance: {
                secureContext: window.isSecureContext,
                userAgent: navigator.userAgent,
                completedAt: new Date().toISOString(),
              },
            }),
          });
          if (!response.ok && response.status !== 409) {
            throw new Error(await responseError(response));
          }
          if (response.status === 409) {
            const refreshed = await fetchCatalog(token);
            const completedAfterConflict = refreshed.videos.some(
              (candidate) => candidate.id === video.id && candidate.completed,
            );
            if (!completedAfterConflict) {
              throw new Error(
                "The result path already exists but does not contain a complete analysis. " +
                "Resolve that server-side conflict before resuming.",
              );
            }
            workingCatalog = refreshed;
          } else {
            workingCatalog = markCompleted(workingCatalog, video.id);
          }
          setCatalog(workingCatalog);
        } catch (error) {
          setFailure({ recordingId: video.id, message: message(error) });
          setPhase("error");
          return;
        } finally {
          media?.input.dispose();
        }
      }
      setCurrentId(null);
      setCurrentMediaInfo(null);
      setProgress(null);
      setPhase("complete");
    } finally {
      runningRef.current = false;
    }
  }, []);

  const refreshAndResume = useCallback(async () => {
    const token = tokenRef.current;
    if (!token || runningRef.current) return;
    setFailure(null);
    setPhase("loading");
    try {
      const fresh = await fetchCatalog(token);
      setCatalog(fresh);
      await runBatch(fresh);
    } catch (error) {
      setFailure({ recordingId: null, message: message(error) });
      setPhase("error");
    }
  }, [runBatch]);

  useEffect(() => {
    let active = true;
    void (async () => {
      await Promise.resolve();
      if (!active) return;
      const token = tokenFromFragment();
      if (!token) {
        setFailure({
          recordingId: null,
          message: "Missing batch token. Add it only as #token=… to this page URL.",
        });
        setPhase("blocked");
        return;
      }
      if (!window.isSecureContext) {
        setFailure({ recordingId: null, message: "This batch runner requires HTTPS." });
        setPhase("blocked");
        return;
      }
      if (!("VideoDecoder" in window) || !("AudioDecoder" in window)) {
        setFailure({
          recordingId: null,
          message: "This browser does not expose the WebCodecs decoders required by the model.",
        });
        setPhase("blocked");
        return;
      }
      tokenRef.current = token;
      setCanResume(true);
      try {
        const loaded = await fetchCatalog(token);
        if (!active) return;
        setCatalog(loaded);
        setPhase("ready");
      } catch (error) {
        if (!active) return;
        setFailure({ recordingId: null, message: message(error) });
        setPhase("error");
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!catalog || phase !== "ready" || autoStartedRef.current) return;
    autoStartedRef.current = true;
    void runBatch(catalog);
  }, [catalog, phase, runBatch]);

  const completed = catalog?.videos.filter((video) => video.completed).length ?? 0;
  const total = catalog?.videos.length ?? 0;
  const current = catalog?.videos.find((video) => video.id === currentId) ?? null;
  const overallPercent = total > 0 ? completed / total * 100 : 0;
  const stagePercent = progressPercent(progress);
  const actionLabel = phase === "complete" ? "Refresh completed catalog" : "Retry and resume";
  const statusLabel = phase === "running"
    ? "RUNNING IN THIS BROWSER"
    : phase === "complete"
      ? "ALL PREDICTIONS SAVED"
      : phase.toUpperCase();
  const rows = useMemo(() => catalog?.videos ?? [], [catalog]);

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Link className={styles.brand} href="/">VOLLEYCUT <span>BATCH</span></Link>
        <div className={styles.security}>TOKEN / FRAGMENT ONLY · HTTPS</div>
      </header>

      <section className={styles.hero}>
        <p className={styles.eyebrow}>BROWSER-NATIVE DATASET INFERENCE</p>
        <h1>Extract every range.<br /><em>Upload no video.</em></h1>
        <p>
          Each prepared proxy is range-read, decoded, measured, and inferred locally in this
          browser. Only the returned intervals, media metadata, and concise provenance are saved.
          Videos are processed one at a time and completed artifacts are skipped on resume.
        </p>
      </section>

      {failure && (
        <section className={styles.error} role="alert">
          <strong>{failure.recordingId ? `FAILED / ${failure.recordingId}` : "BATCH UNAVAILABLE"}</strong>
          <p>{failure.message}</p>
          {phase === "error" && canResume && (
            <button type="button" onClick={() => void refreshAndResume()}>
              {actionLabel}
            </button>
          )}
        </section>
      )}

      <section className={styles.summary} aria-live="polite">
        <div>
          <span>STATUS</span>
          <strong data-phase={phase}>{statusLabel}</strong>
        </div>
        <div>
          <span>MODEL</span>
          <strong>{catalog?.modelId ?? "Waiting for authorized catalog"}</strong>
        </div>
        <div className={styles.count}>
          <span>VIDEOS SAVED</span>
          <strong>{completed}<small> / {total || "–"}</small></strong>
        </div>
        <div className={styles.overallTrack} aria-label={`${Math.round(overallPercent)}% complete`}>
          <i style={{ width: `${overallPercent}%` }} />
        </div>
      </section>

      {current && progress && (
        <section className={styles.current}>
          <div>
            <span>CURRENT VIDEO · {current.environment}</span>
            <strong>{current.originalFilename}</strong>
            <p>{progress.detail}</p>
          </div>
          <output>{Math.round(stagePercent)}%</output>
          <div className={styles.stageTrack}>
            <i style={{ width: `${stagePercent}%` }} />
          </div>
          <small>
            {progress.stage.toUpperCase()} · {current.id}
            {currentMediaInfo
              ? ` · ${currentMediaInfo.mimeType} · ${currentMediaInfo.videoCodec}/${currentMediaInfo.audioCodec} · ${currentMediaInfo.width}×${currentMediaInfo.height} · ${currentMediaInfo.sampleRate ?? "?"} Hz · ${currentMediaInfo.channels ?? "?"} ch · rotation ${currentMediaInfo.rotation} · ${currentMediaInfo.videoCodecString || "no codec string"}`
              : ""}
          </small>
        </section>
      )}

      {rows.length > 0 && (
        <section className={styles.queue}>
          <header>
            <div>
              <p className={styles.eyebrow}>RESUMABLE WORK QUEUE</p>
              <h2>Nine comparison videos</h2>
            </div>
            {phase === "complete" && (
              <button type="button" onClick={() => void refreshAndResume()}>
                Refresh status
              </button>
            )}
          </header>
          <div className={styles.rows}>
            {rows.map((video, index) => {
              const failed = failure?.recordingId === video.id;
              const active = currentId === video.id && phase === "running";
              return (
                <article
                  key={video.id}
                  data-status={video.completed ? "complete" : failed ? "failed" : active ? "active" : "pending"}
                >
                  <span className={styles.index}>{String(index + 1).padStart(2, "0")}</span>
                  <div>
                    <strong>{video.id}</strong>
                    <span title={video.originalFilename}>{video.originalFilename}</span>
                  </div>
                  <span>{video.environment}</span>
                  <span>{formatTime(video.duration)}</span>
                  <span>{compactBytes(video.size)}</span>
                  <b>{video.completed ? "SAVED" : failed ? "FAILED" : active ? "RUNNING" : "PENDING"}</b>
                </article>
              );
            })}
          </div>
        </section>
      )}

      <footer className={styles.footer}>
        <p>
          The bearer token remains in the URL fragment and is attached only to same-origin API
          requests. It is never included in prediction artifacts.
        </p>
        <div><Link href="/">Comparison UI</Link><Link href="/on-device">Single-video runner</Link></div>
      </footer>
    </main>
  );
}
