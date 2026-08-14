import { useEffect, useRef, useState } from "react";

import { CutEditor } from "@/components/CutEditor";
import { openLocalMedia, type OpenedMedia } from "@/lib/on-device/media";
import {
  analyzeOpenedMedia,
  DEFAULT_FEATURE_REDUCTION_KERNEL,
  DEFAULT_VIDEO_DECODE_STRATEGY,
  VIDEO_DECODER_HARDWARE_ACCELERATION,
} from "@/lib/on-device/pipeline";
import { DEFAULT_ON_DEVICE_RUNTIME_VARIANT } from "@/lib/on-device/runtime-variants";
import type {
  AnalysisProgress,
  NormalizedRoi,
  OnDeviceAnalysis,
  OnDeviceMediaInfo,
} from "@/lib/on-device/types";
import { holdScreenWakeLock, type WakeLockState } from "@/lib/on-device/wake-lock";
import type { ProductAnalysis } from "@/lib/product-analysis";
import { runtimeAssetUrl } from "@/lib/runtime-assets";

import styles from "./App.module.css";

type WorkState = "empty" | "opening" | "ready" | "analyzing" | "error";

const COURT_CENTERED_ROI: NormalizedRoi = {
  x: 0.03,
  y: 0.12,
  width: 0.94,
  height: 0.86,
};
const FULL_FRAME_ROI: NormalizedRoi = { x: 0, y: 0, width: 1, height: 1 };

function compactBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let index = 0;
  while (value >= 1000 && index < units.length - 1) {
    value /= 1000;
    index += 1;
  }
  return `${value.toFixed(index < 2 ? 0 : 1)} ${units[index]}`;
}

function formatDuration(seconds: number): string {
  const safe = Math.max(0, Math.round(seconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = safe % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function progressPercent(progress: AnalysisProgress | null): number {
  if (!progress) return 0;
  const fraction = progress.total > 0 ? progress.completed / progress.total : 0;
  switch (progress.stage) {
    case "opening": return 2;
    case "video": return Math.min(82, Math.max(3, fraction * 82));
    case "audio": return 82 + Math.min(10, Math.max(0, fraction * 10));
    case "normalizing": return 94;
    case "inference": return 98;
    case "complete": return 100;
  }
}

function clampRoi(roi: NormalizedRoi): NormalizedRoi {
  const x = Math.max(0, Math.min(0.99, roi.x));
  const y = Math.max(0, Math.min(0.99, roi.y));
  return {
    x,
    y,
    width: Math.max(0.01, Math.min(1 - x, roi.width)),
    height: Math.max(0.01, Math.min(1 - y, roi.height)),
  };
}

function localId(file: File, info: OnDeviceMediaInfo): string {
  const source = `${file.name}\u0000${file.size}\u0000${file.lastModified}\u0000${info.duration}`;
  let hash = 2166136261;
  for (const character of source) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return `local-${(hash >>> 0).toString(36)}`;
}

export function App() {
  const [workState, setWorkState] = useState<WorkState>("empty");
  const [file, setFile] = useState<File | null>(null);
  const [info, setInfo] = useState<OnDeviceMediaInfo | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [roi, setRoi] = useState<NormalizedRoi>(COURT_CENTERED_ROI);
  const [progress, setProgress] = useState<AnalysisProgress | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [wakeLockState, setWakeLockState] = useState<WakeLockState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<ProductAnalysis | null>(null);
  const openedMedia = useRef<OpenedMedia | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const elapsedTimer = useRef<number | null>(null);
  const resumePreviewAfterSeek = useRef(false);

  const webCodecsReady =
    "VideoDecoder" in window && "AudioDecoder" in window && "VideoFrame" in window;
  const secureContext = window.isSecureContext;
  const busy = workState === "opening" || workState === "analyzing";

  useEffect(() => {
    return () => {
      openedMedia.current?.input.dispose();
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      if (elapsedTimer.current !== null) window.clearInterval(elapsedTimer.current);
    };
  }, []);

  useEffect(() => {
    if (analysis) window.scrollTo({ top: 0, behavior: "instant" });
  }, [analysis]);

  function replacePreviewUrl(next: string | null) {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = next;
    setPreviewUrl(next);
  }

  async function chooseFile(selected: File | null) {
    if (!selected) return;
    if (!webCodecsReady) {
      setError(
        secureContext
          ? "This browser does not expose the WebCodecs APIs needed for local analysis. Use Safari 26 or a current Chrome or Edge release."
          : "Local analysis requires HTTPS. Deploy this static app over HTTPS, or use localhost during development.",
      );
      setWorkState("error");
      return;
    }

    openedMedia.current?.input.dispose();
    openedMedia.current = null;
    setFile(selected);
    setInfo(null);
    setAnalysis(null);
    setRoi(COURT_CENTERED_ROI);
    setError(null);
    resumePreviewAfterSeek.current = false;
    setProgress({
      stage: "opening",
      completed: 0,
      total: 1,
      detail: "Reading container metadata locally",
    });
    setWorkState("opening");
    replacePreviewUrl(URL.createObjectURL(selected));

    try {
      const opened = await openLocalMedia(selected);
      openedMedia.current = opened;
      setInfo(opened.info);
      setProgress(null);
      setWorkState("ready");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setProgress(null);
      setWorkState("error");
    }
  }

  async function runAnalysis() {
    if (!openedMedia.current || !file || !info || !previewUrl) return;
    setError(null);
    setWorkState("analyzing");
    setElapsedSeconds(0);
    const startedAt = performance.now();
    elapsedTimer.current = window.setInterval(() => {
      setElapsedSeconds((performance.now() - startedAt) / 1000);
    }, 500);
    const releaseWakeLock = await holdScreenWakeLock(setWakeLockState);
    try {
      const featurePath: OnDeviceAnalysis["featurePath"] = "local-source";
      const result = await analyzeOpenedMedia(
        openedMedia.current,
        roi,
        featurePath,
        DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
        setProgress,
        { name: file.name, size: file.size, lastModified: file.lastModified },
        {
          detailedProfiling: false,
          decodeStrategy: DEFAULT_VIDEO_DECODE_STRATEGY,
          decoderAcceleration: VIDEO_DECODER_HARDWARE_ACCELERATION,
          reductionKernel: DEFAULT_FEATURE_REDUCTION_KERNEL,
        },
      );
      const recordingId = localId(file, info);
      setAnalysis({
        id: `${recordingId}-${result.modelId}-${DEFAULT_ON_DEVICE_RUNTIME_VARIANT}`,
        recordingId,
        kind: "model",
        modelId: result.modelId,
        duration: info.duration,
        width: info.width,
        height: info.height,
        sourceFilename: file.name,
        videoUrl: previewUrl,
        rallies: result.intervals,
        ignoredIntervals: [],
      });
      openedMedia.current.input.dispose();
      openedMedia.current = null;
      setWorkState("ready");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setWorkState("error");
    } finally {
      if (elapsedTimer.current !== null) window.clearInterval(elapsedTimer.current);
      elapsedTimer.current = null;
      setElapsedSeconds((performance.now() - startedAt) / 1000);
      await releaseWakeLock();
    }
  }

  function startOver() {
    openedMedia.current?.input.dispose();
    openedMedia.current = null;
    replacePreviewUrl(null);
    setFile(null);
    setInfo(null);
    setAnalysis(null);
    setProgress(null);
    setError(null);
    setElapsedSeconds(0);
    setWakeLockState("idle");
    setWorkState("empty");
  }

  if (analysis) {
    return (
      <CutEditor
        key={analysis.id}
        initialAnalysis={analysis}
        sourceFile={file!}
        onStartOver={startOver}
      />
    );
  }

  const percent = progressPercent(progress);
  const featurePerformance = progress?.performance ?? null;
  const featureRate = featurePerformance && featurePerformance.videoElapsedMs > 0
    ? featurePerformance.generatedVideoSeconds / (featurePerformance.videoElapsedMs / 1000)
    : null;
  const analysisEtaSeconds = progress?.stage === "complete"
    ? 0
    : progress?.stage === "video" && featureRate && featureRate > 0
      ? Math.max(0, progress.total - progress.completed) / featureRate
      : elapsedSeconds >= 2 && percent > 2
        ? elapsedSeconds * (100 - percent) / percent
        : null;

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <span className={styles.brand}>
          <img src={runtimeAssetUrl("volleycut-logo.png")} alt="VolleyCut" />
          <span>LOCAL CUT</span>
        </span>
        <div className={styles.capabilities}>
          <span data-ok={secureContext || undefined}>HTTPS</span>
          <span data-ok={webCodecsReady || undefined}>WebCodecs</span>
          <span data-ok>Private by design</span>
        </div>
      </header>

      <section className={styles.hero}>
        <p>ONE PRIVATE WORKFLOW</p>
        <h1>Load. Detect. <em>Refine.</em></h1>
        <p className={styles.lede}>
          Choose a volleyball video, generate audiovisual features and rally predictions
          on this device, then refine every cut in the editor. The selected video is
          never uploaded.
        </p>
        <div className={styles.pipeline} aria-label="Local processing pipeline">
          <span><b>01</b> Local video</span><i>→</i>
          <span><b>02</b> Browser features</span><i>→</i>
          <span><b>03</b> Local inference</span><i>→</i>
          <span><b>04</b> Cut editor</span>
        </div>
      </section>

      {!secureContext && (
        <p className={styles.notice}>
          This page is not in a secure context. The interface is available, but local
          media analysis needs HTTPS or localhost.
        </p>
      )}
      {error && <p className={styles.error}>{error}</p>}

      <section className={styles.importCard}>
        <div>
          <p>STEP 01 · SOURCE</p>
          <h2>{file?.name ?? "Choose a volleyball video"}</h2>
          <p>
            {file
              ? `${compactBytes(file.size)} · read directly from this browser tab`
              : "MP4, WebM, MOV, MKV, and other browser-decodable containers are supported."}
          </p>
        </div>
        <label className={styles.fileButton} data-disabled={busy || undefined}>
          {file ? "Choose another" : "Choose video"}
          <input
            type="file"
            accept="video/*,.mkv,.webm,.mov,.mp4,.m4v"
            disabled={busy}
            onChange={(event) => void chooseFile(event.currentTarget.files?.[0] ?? null)}
          />
        </label>
      </section>

      {info && previewUrl && (
        <section className={styles.workspace}>
          <div className={styles.viewer}>
            <div
              className={styles.videoStage}
              style={{ aspectRatio: `${info.width} / ${info.height}` }}
            >
              <video
                src={previewUrl}
                controls
                preload="metadata"
                playsInline
                onPlay={() => {
                  resumePreviewAfterSeek.current = true;
                }}
                onPause={(event) => {
                  if (!event.currentTarget.seeking) resumePreviewAfterSeek.current = false;
                }}
                onEnded={() => {
                  resumePreviewAfterSeek.current = false;
                }}
                onSeeked={(event) => {
                  if (resumePreviewAfterSeek.current) {
                    void event.currentTarget.play().catch(() => undefined);
                  }
                }}
              />
              <div
                className={styles.roiBox}
                style={{
                  left: `${roi.x * 100}%`,
                  top: `${roi.y * 100}%`,
                  width: `${roi.width * 100}%`,
                  height: `${roi.height * 100}%`,
                }}
              >
                <span>FEATURE CROP</span>
              </div>
            </div>
          </div>

          <aside className={styles.inspector}>
            <p>STEP 02 · FEATURES</p>
            <h2>Confirm the camera crop</h2>
            <dl>
              <div><dt>Duration</dt><dd>{formatDuration(info.duration)}</dd></div>
              <div><dt>Frame</dt><dd>{info.width} × {info.height}</dd></div>
              <div><dt>Video</dt><dd>{info.videoCodecString ?? info.videoCodec}</dd></div>
              <div><dt>Audio</dt><dd>{info.hasAudio ? info.audioCodec ?? "Available" : "No track"}</dd></div>
            </dl>
            <p className={styles.cropHelp}>
              Keep the court and players inside the box. Exclude static borders, stands,
              or neighboring courts when practical.
            </p>
            <div className={styles.presetButtons}>
              <button type="button" onClick={() => setRoi(COURT_CENTERED_ROI)}>
                Court centered
              </button>
              <button type="button" onClick={() => setRoi(FULL_FRAME_ROI)}>
                Full frame
              </button>
            </div>
            <div className={styles.roiGrid}>
              {(["x", "y", "width", "height"] as const).map((field) => (
                <label key={field}>
                  <span>{field} <output>{Math.round(roi[field] * 100)}%</output></span>
                  <input
                    aria-label={`Feature crop ${field}`}
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={roi[field]}
                    onChange={(event) => setRoi(clampRoi({
                      ...roi,
                      [field]: Number(event.currentTarget.value),
                    }))}
                    disabled={busy}
                  />
                </label>
              ))}
            </div>
            <button
              className={styles.analyzeButton}
              type="button"
              onClick={() => void runAnalysis()}
              disabled={busy}
            >
              {workState === "analyzing" ? "Analyzing on this device…" : "Generate cuts locally"}
            </button>
            <small className={styles.runtimeNote}>
              Bundled model · OpenCV worker · feature WASM · FFmpeg audio-resampler WASM
            </small>
          </aside>
        </section>
      )}

      {progress && (
        <section className={styles.progressCard} aria-live="polite">
          <div>
            <p>LOCAL PROCESSING · {progress.stage.toUpperCase()}</p>
            <strong>{progress.detail}</strong>
          </div>
          <output>{Math.round(percent)}%</output>
          <div className={styles.progressTrack}><i style={{ width: `${percent}%` }} /></div>
          <div className={styles.progressTiming}>
            <div>
              <span>Elapsed</span>
              <strong>{formatDuration(elapsedSeconds)}</strong>
            </div>
            <div>
              <span>Estimated remaining</span>
              <strong>
                {analysisEtaSeconds === null
                  ? "Estimating…"
                  : analysisEtaSeconds <= 1
                    ? "Finishing…"
                    : `About ${formatDuration(analysisEtaSeconds)}`}
              </strong>
            </div>
          </div>
          <p>
            {progress.stage === "video" && featureRate
              ? `${featureRate.toFixed(2)}× real-time feature generation`
              : "Feature extraction, audio analysis, and inference run locally."}
            {progress.featureCache?.resumedRows
              ? ` · resumed ${progress.featureCache.resumedRows.toLocaleString()} saved frames`
              : ""}
            {wakeLockState === "active" ? " · screen wake lock active" : ""}
          </p>
        </section>
      )}

      <footer className={styles.footer}>
        <span>All media, features, predictions, and edit drafts stay in this browser.</span>
        <span>Chrome, Edge, or Safari 26 · HTTPS required outside localhost</span>
      </footer>
    </main>
  );
}
