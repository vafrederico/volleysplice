"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import { buildEditList, formatTime } from "@/lib/edit-list";
import { FEATURE_CACHE_CHUNK_ROWS } from "@/lib/on-device/feature-cache";
import { ANALYSIS_FPS } from "@/lib/on-device/feature-schema";
import {
  downloadEditDecisionList,
  exportRawQualityReel,
  type ExportProgress,
} from "@/lib/on-device/export";
import { openLocalMedia, type OpenedMedia } from "@/lib/on-device/media";
import {
  analyzeOpenedMedia,
  VIDEO_DECODER_HARDWARE_ACCELERATION,
} from "@/lib/on-device/pipeline";
import { clampRoi, fullFrameRoi, inferRoiProfile } from "@/lib/on-device/roi";
import type {
  AnalysisProgress,
  FeatureExtractionPerformance,
  NormalizedRoi,
  OnDeviceAnalysis,
  OnDeviceMediaInfo,
  RoiProfile,
} from "@/lib/on-device/types";

import styles from "./on-device.module.css";

type WorkState = "empty" | "opening" | "ready" | "analyzing" | "exporting" | "done" | "error";

type BrowserCompatibility = {
  checked: boolean;
  secureContext: boolean;
  decode: boolean;
  encode: boolean;
  directDisk: boolean;
  screenWakeLock: boolean;
  mediaCapabilities: boolean;
  webGpu: "checking" | "available" | "unavailable";
  gpuName: string | null;
  gpuSource: "WebGPU" | "WebGL" | null;
  logicalProcessors: number | null;
  deviceMemoryGb: number | null;
  jsHeapUsedBytes: number | null;
  jsHeapLimitBytes: number | null;
};

type WakeLockState =
  | "idle"
  | "requesting"
  | "active"
  | "paused"
  | "released"
  | "unavailable"
  | "denied";

type SelectedMediaDiagnostics = {
  checking: boolean;
  codec: string | null;
  webCodecsSupported: boolean | null;
  smooth: boolean | null;
  powerEfficient: boolean | null;
  averageFrameRate: number | null;
  averageBitrate: number | null;
};

type GpuAdapterInfoLike = {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
};

type GpuAdapterLike = {
  info?: GpuAdapterInfoLike;
  requestAdapterInfo?: () => Promise<GpuAdapterInfoLike>;
};

type NavigatorWithDiagnostics = Navigator & {
  deviceMemory?: number;
  gpu?: {
    requestAdapter(options?: { powerPreference?: "low-power" | "high-performance" }): Promise<GpuAdapterLike | null>;
  };
};

type PerformanceWithMemory = Performance & {
  memory?: {
    usedJSHeapSize: number;
    jsHeapSizeLimit: number;
  };
};

const EMPTY_MEDIA_DIAGNOSTICS: SelectedMediaDiagnostics = {
  checking: false,
  codec: null,
  webCodecsSupported: null,
  smooth: null,
  powerEfficient: null,
  averageFrameRate: null,
  averageBitrate: null,
};

function gpuName(info: GpuAdapterInfoLike | undefined): string | null {
  if (!info) return null;
  const description = info.description?.trim();
  if (description) return description;
  const parts = [info.vendor, info.architecture, info.device]
    .map((part) => part?.trim())
    .filter((part): part is string => Boolean(part));
  return parts.length > 0 ? [...new Set(parts)].join(" · ") : null;
}

function webGlRenderer(): string | null {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
  if (!context) return null;
  const extension = context.getExtension("WEBGL_debug_renderer_info");
  if (!extension) return null;
  const renderer = context.getParameter(extension.UNMASKED_RENDERER_WEBGL);
  return typeof renderer === "string" && renderer.trim() ? renderer.trim() : null;
}

function heapSnapshot(): Pick<
  BrowserCompatibility,
  "jsHeapUsedBytes" | "jsHeapLimitBytes"
> {
  const memory = (performance as PerformanceWithMemory).memory;
  return {
    jsHeapUsedBytes: memory?.usedJSHeapSize ?? null,
    jsHeapLimitBytes: memory?.jsHeapSizeLimit ?? null,
  };
}

function videoContentType(mimeType: string, codec: string): string {
  const container = mimeType.split(";", 1)[0]?.trim() || "video/mp4";
  return `${container}; codecs="${codec.replaceAll('"', "")}"`;
}

async function inspectSelectedMedia(
  media: OpenedMedia,
  fileSize: number,
): Promise<SelectedMediaDiagnostics> {
  const [decoderConfig, packetStats] = await Promise.all([
    media.videoTrack.getDecoderConfig().catch(() => null),
    media.videoTrack.computePacketStats(120).catch(() => null),
  ]);
  const codec = decoderConfig?.codec ?? media.info.videoCodecString;
  let webCodecsSupported: boolean | null = null;
  if (codec && "VideoDecoder" in window) {
    try {
      const support = await VideoDecoder.isConfigSupported({
        ...decoderConfig,
        codec,
        codedWidth: decoderConfig?.codedWidth ?? media.info.width,
        codedHeight: decoderConfig?.codedHeight ?? media.info.height,
        hardwareAcceleration: VIDEO_DECODER_HARDWARE_ACCELERATION,
      });
      webCodecsSupported = support.supported ?? false;
    } catch {
      webCodecsSupported = false;
    }
  }

  const averageFrameRate = packetStats?.averagePacketRate ?? null;
  const averageBitrate =
    packetStats?.averageBitrate ??
    (media.info.duration > 0 ? (fileSize * 8) / media.info.duration : null);
  let smooth: boolean | null = null;
  let powerEfficient: boolean | null = null;
  if (codec && "mediaCapabilities" in navigator) {
    try {
      const capability = await navigator.mediaCapabilities.decodingInfo({
        type: "file",
        video: {
          contentType: videoContentType(media.info.mimeType, codec),
          width: media.info.width,
          height: media.info.height,
          bitrate: Math.max(1, Math.round(averageBitrate ?? 5_000_000)),
          framerate: Math.max(1, averageFrameRate ?? 30),
        },
      });
      smooth = capability.supported ? capability.smooth : false;
      powerEfficient = capability.supported ? capability.powerEfficient : false;
    } catch {
      // Browsers may expose MediaCapabilities but reject a particular container/codec pair.
    }
  }

  return {
    checking: false,
    codec,
    webCodecsSupported,
    smooth,
    powerEfficient,
    averageFrameRate,
    averageBitrate,
  };
}

async function holdScreenWakeLock(
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
      sentinel.addEventListener("release", () => {
        sentinel = null;
        if (stopped) return;
        if (document.visibilityState === "visible") void acquire();
        else onState("paused");
      }, { once: true });
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

export type OnDeviceUiFixture = {
  schemaVersion: 1;
  provenance: {
    kind: "browser-on-device-prediction";
    capturedAt: string;
    modelId: string;
    note: string;
  };
  source: {
    name: string;
    size: number;
    lastModified: number;
    info: OnDeviceMediaInfo;
  };
  profile: RoiProfile;
  roi: NormalizedRoi;
  featurePath: OnDeviceAnalysis["featurePath"];
  analysis: {
    modelId: string;
    featurePath: OnDeviceAnalysis["featurePath"];
    intervals: OnDeviceAnalysis["intervals"];
  };
  exportDefaults: {
    preRoll: number;
    postRoll: number;
  };
};

function compactBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value.toFixed(unit < 2 ? 0 : 1)} ${units[unit]}`;
}

function compactBitrate(bitsPerSecond: number): string {
  if (bitsPerSecond >= 1_000_000) return `${(bitsPerSecond / 1_000_000).toFixed(1)} Mbps`;
  if (bitsPerSecond >= 1_000) return `${Math.round(bitsPerSecond / 1_000)} Kbps`;
  return `${Math.round(bitsPerSecond)} bps`;
}

function percent(progress: AnalysisProgress | null): number {
  if (!progress || progress.total <= 0) return 0;
  return Math.min(100, Math.max(0, (progress.completed / progress.total) * 100));
}

function preciseTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00.00";
  const minutes = Math.floor(Math.max(0, seconds) / 60);
  const remainder = Math.max(0, seconds) - minutes * 60;
  return `${minutes}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function timingDuration(milliseconds: number): string {
  if (milliseconds < 1000) return `${milliseconds.toFixed(milliseconds < 10 ? 1 : 0)} ms`;
  return preciseTime(milliseconds / 1000);
}

function timingSummary(
  milliseconds: number,
  profile: FeatureExtractionPerformance,
  includePerFrame = true,
): string {
  const parts = [timingDuration(milliseconds)];
  if (includePerFrame && profile.sampledFrames > 0) {
    parts.push(`${(milliseconds / profile.sampledFrames).toFixed(1)} ms/frame`);
  }
  if (profile.videoElapsedMs > 0) {
    parts.push(`${((milliseconds / profile.videoElapsedMs) * 100).toFixed(0)}%`);
  }
  return parts.join(" · ");
}

function RoiControls({
  roi,
  profile,
  onChange,
  onReset,
  onFullFrame,
  disabled,
}: {
  roi: NormalizedRoi;
  profile: RoiProfile;
  onChange: (roi: NormalizedRoi) => void;
  onReset: () => void;
  onFullFrame: () => void;
  disabled: boolean;
}) {
  const controls: Array<{ key: keyof NormalizedRoi; label: string; max: number }> = [
    { key: "x", label: "Left", max: 1 - roi.width },
    { key: "y", label: "Top", max: 1 - roi.height },
    { key: "width", label: "Width", max: 1 - roi.x },
    { key: "height", label: "Height", max: 1 - roi.y },
  ];
  return (
    <details className={styles.roiPanel}>
      <summary>
        <span>Camera crop</span>
        <strong>{profile.label}</strong>
      </summary>
      <p>
        This is a coarse feature crop, not a drawn court or court-line label. The model&apos;s
        training recordings used one rectangle per stationary camera.
      </p>
      <div className={styles.roiButtons}>
        <button type="button" onClick={onReset} disabled={disabled}>Reset camera profile</button>
        <button type="button" onClick={onFullFrame} disabled={disabled}>Use full frame</button>
      </div>
      <div className={styles.roiGrid}>
        {controls.map(({ key, label, max }) => (
          <label key={key}>
            <span>{label} <output>{Math.round(roi[key] * 100)}%</output></span>
            <input
              type="range"
              min={0}
              max={Math.max(0.01, max)}
              step={0.005}
              value={roi[key]}
              disabled={disabled}
              onChange={(event) =>
                onChange(clampRoi({ ...roi, [key]: Number(event.target.value) }))
              }
            />
          </label>
        ))}
      </div>
    </details>
  );
}

export function OnDeviceClient({ fixture = null }: { fixture?: OnDeviceUiFixture | null }) {
  const fixtureAnalysis = useMemo<OnDeviceAnalysis | null>(
    () =>
      fixture
        ? {
            ...fixture.analysis,
            intervals: fixture.analysis.intervals.map((interval) => ({ ...interval })),
            times: new Float64Array(0),
            rallyProbabilities: new Float32Array(0),
            serveProbabilities: new Float32Array(0),
            deadStateProbabilities: new Float32Array(0),
          }
        : null,
    [fixture],
  );
  const [workState, setWorkState] = useState<WorkState>(fixture ? "done" : "empty");
  const [file, setFile] = useState<File | null>(null);
  const [fixtureActive, setFixtureActive] = useState(fixture !== null);
  const [hasOpenedMedia, setHasOpenedMedia] = useState(false);
  const [exportFile, setExportFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [info, setInfo] = useState<OnDeviceMediaInfo | null>(fixture?.source.info ?? null);
  const [profile, setProfile] = useState<RoiProfile>(() => fixture?.profile ?? inferRoiProfile(""));
  const [roi, setRoi] = useState<NormalizedRoi>(() => fixture?.roi ?? profile.roi);
  const [featurePath, setFeaturePath] = useState<OnDeviceAnalysis["featurePath"]>(
    fixture?.featurePath ?? "raw-virtual-proxy",
  );
  const [analysis, setAnalysis] = useState<OnDeviceAnalysis | null>(fixtureAnalysis);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress | null>(null);
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewWarning, setPreviewWarning] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(
    fixtureAnalysis?.intervals[0]?.id ?? null,
  );
  const [preRoll, setPreRoll] = useState(fixture?.exportDefaults.preRoll ?? 3);
  const [postRoll, setPostRoll] = useState(fixture?.exportDefaults.postRoll ?? 2);
  const [compatibility, setCompatibility] = useState<BrowserCompatibility>({
    checked: false,
    secureContext: false,
    decode: false,
    encode: false,
    directDisk: false,
    screenWakeLock: false,
    mediaCapabilities: false,
    webGpu: "checking",
    gpuName: null,
    gpuSource: null,
    logicalProcessors: null,
    deviceMemoryGb: null,
    jsHeapUsedBytes: null,
    jsHeapLimitBytes: null,
  });
  const [mediaDiagnostics, setMediaDiagnostics] = useState<SelectedMediaDiagnostics>(
    EMPTY_MEDIA_DIAGNOSTICS,
  );
  const [analysisElapsedSeconds, setAnalysisElapsedSeconds] = useState<number | null>(null);
  const [wakeLockState, setWakeLockState] = useState<WakeLockState>("idle");
  const [featureCacheState, setFeatureCacheState] = useState<
    AnalysisProgress["featureCache"] | null
  >(null);
  const openedMedia = useRef<OpenedMedia | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const busy = workState === "opening" || workState === "analyzing" || workState === "exporting";
  const editList = useMemo(
    () =>
      analysis && info
        ? buildEditList(analysis.intervals, preRoll, postRoll, info.duration)
        : [],
    [analysis, info, preRoll, postRoll],
  );
  const keptSeconds = editList.reduce(
    (total, interval) => total + interval.keptEnd - interval.keptStart,
    0,
  );
  const uiFixtureMode = fixture !== null && fixtureActive;
  const displayedFileSize = uiFixtureMode ? (fixture?.source.size ?? 0) : (file?.size ?? 0);
  const sourceName = file?.name ?? (uiFixtureMode ? fixture?.source.name : null);
  const featurePerformance = analysisProgress?.performance ?? null;
  const processingRate =
    featurePerformance && featurePerformance.videoElapsedMs > 0
      ? featurePerformance.generatedVideoSeconds / (featurePerformance.videoElapsedMs / 1000)
      : null;
  const extractionOtherMs = featurePerformance
    ? Math.max(
        0,
        featurePerformance.extractionMs -
          featurePerformance.canvasReadbackMs -
          featurePerformance.imageOperationsMs -
          featurePerformance.phaseCorrelationMs -
          featurePerformance.opticalFlowMs -
          featurePerformance.javascriptMs -
          (featurePerformance.workerActive ? featurePerformance.canvasDrawMs : 0),
      )
    : 0;
  const featureBlockingMs = featurePerformance?.workerActive
    ? featurePerformance.workerBlockingMs
    : featurePerformance?.extractionMs ?? 0;
  const videoPipelineOtherMs = featurePerformance
    ? Math.max(
        0,
        featurePerformance.videoElapsedMs -
          featurePerformance.openCvLoadMs -
          featurePerformance.decoderCanvasMs -
          featureBlockingMs -
          featurePerformance.cacheIoMs,
      )
    : 0;

  useEffect(() => {
    let active = true;
    queueMicrotask(async () => {
      if (!active) return;
      const diagnosticNavigator = navigator as NavigatorWithDiagnostics;
      const fallbackRenderer = webGlRenderer();
      setCompatibility({
        checked: true,
        secureContext: window.isSecureContext,
        decode: "VideoDecoder" in window && "AudioDecoder" in window,
        encode: "VideoEncoder" in window && "AudioEncoder" in window,
        directDisk: "showSaveFilePicker" in window,
        screenWakeLock: "wakeLock" in navigator,
        mediaCapabilities: "mediaCapabilities" in navigator,
        webGpu: diagnosticNavigator.gpu ? "checking" : "unavailable",
        gpuName: fallbackRenderer,
        gpuSource: fallbackRenderer ? "WebGL" : null,
        logicalProcessors: navigator.hardwareConcurrency || null,
        deviceMemoryGb: diagnosticNavigator.deviceMemory ?? null,
        ...heapSnapshot(),
      });
      if (!diagnosticNavigator.gpu) return;
      try {
        const adapter = await diagnosticNavigator.gpu.requestAdapter();
        if (!active) return;
        const info = adapter
          ? adapter.info ?? (await adapter.requestAdapterInfo?.().catch(() => undefined))
          : undefined;
        setCompatibility((current) => ({
          ...current,
          webGpu: adapter ? "available" : "unavailable",
          gpuName: gpuName(info) ?? current.gpuName,
          gpuSource: gpuName(info) ? "WebGPU" : current.gpuSource,
        }));
      } catch {
        if (active) {
          setCompatibility((current) => ({ ...current, webGpu: "unavailable" }));
        }
      }
    });
    return () => {
      active = false;
      openedMedia.current?.input.dispose();
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function chooseFile(selected: File | null) {
    if (!selected) return;
    if (!compatibility.decode) {
      setError(
        compatibility.secureContext
          ? "This browser does not expose the WebCodecs decoders required for local analysis. Use current desktop Chrome or Edge."
          : "Local analysis requires a secure browser context. Open this app over HTTPS (localhost is also trusted for development).",
      );
      return;
    }
    openedMedia.current?.input.dispose();
    openedMedia.current = null;
    setHasOpenedMedia(false);
    setFixtureActive(false);
    setFile(selected);
    setExportFile(selected);
    setInfo(null);
    setAnalysis(null);
    setAnalysisElapsedSeconds(null);
    setFeatureCacheState(null);
    setWakeLockState("idle");
    setSelectedId(null);
    setError(null);
    setPreviewWarning(false);
    setMediaDiagnostics({ ...EMPTY_MEDIA_DIAGNOSTICS, checking: true });
    setAnalysisProgress({ stage: "opening", completed: 0, total: 1, detail: "Reading container metadata locally" });
    setWorkState("opening");
    const nextProfile = inferRoiProfile(selected.name);
    setProfile(nextProfile);
    setRoi(nextProfile.roi);
    setPreviewUrl(URL.createObjectURL(selected));
    try {
      const opened = await openLocalMedia(selected);
      openedMedia.current = opened;
      setHasOpenedMedia(true);
      setInfo(opened.info);
      setFeaturePath(
        opened.info.width <= 1280 && opened.info.height <= 720
          ? "training-proxy"
          : "raw-virtual-proxy",
      );
      setWorkState("ready");
      setAnalysisProgress(null);
      void inspectSelectedMedia(opened, selected.size)
        .then(setMediaDiagnostics)
        .catch(() => setMediaDiagnostics(EMPTY_MEDIA_DIAGNOSTICS));
    } catch (cause) {
      setHasOpenedMedia(false);
      setMediaDiagnostics(EMPTY_MEDIA_DIAGNOSTICS);
      setError(cause instanceof Error ? cause.message : String(cause));
      setWorkState("error");
    }
  }

  async function runAnalysis() {
    if (!openedMedia.current || !info) return;
    setError(null);
    setAnalysis(null);
    setAnalysisProgress(null);
    setSelectedId(null);
    setWorkState("analyzing");
    const startedAt = performance.now();
    setAnalysisElapsedSeconds(0);
    const releaseWakeLock = await holdScreenWakeLock(setWakeLockState);
    try {
      const result = await analyzeOpenedMedia(
        openedMedia.current,
        roi,
        featurePath,
        (progress) => {
          setAnalysisProgress(progress);
          if (progress.featureCache) setFeatureCacheState(progress.featureCache);
          setAnalysisElapsedSeconds((performance.now() - startedAt) / 1000);
          setCompatibility((current) => ({ ...current, ...heapSnapshot() }));
        },
        file
          ? { name: file.name, size: file.size, lastModified: file.lastModified }
          : undefined,
      );
      setAnalysisElapsedSeconds((performance.now() - startedAt) / 1000);
      setAnalysis(result);
      setSelectedId(result.intervals[0]?.id ?? null);
      setWorkState("done");
    } catch (cause) {
      setAnalysisElapsedSeconds((performance.now() - startedAt) / 1000);
      setError(cause instanceof Error ? cause.message : String(cause));
      setWorkState("error");
    } finally {
      await releaseWakeLock();
    }
  }

  function updateRoi(next: NormalizedRoi) {
    setRoi(next);
    setProfile({ id: "custom", label: "Custom camera crop", roi: next, source: "camera-default" });
  }

  function updateInterval(
    id: string,
    patch: Partial<Pick<OnDeviceAnalysis["intervals"][number], "start" | "end" | "included">>,
  ) {
    setAnalysis((current) =>
      current
        ? {
            ...current,
            intervals: current.intervals.map((interval) =>
              interval.id === id ? { ...interval, ...patch } : interval,
            ),
          }
        : current,
    );
  }

  function seek(seconds: number) {
    if (!videoRef.current) return;
    videoRef.current.currentTime = seconds;
    void videoRef.current.play().catch(() => undefined);
  }

  async function exportReel() {
    if (!exportFile || !analysis) return;
    setError(null);
    setWorkState("exporting");
    setExportProgress({ completedSeconds: 0, totalSeconds: keptSeconds, detail: "Preparing original media" });
    try {
      await exportRawQualityReel(
        exportFile,
        editList.map(({ keptStart, keptEnd }) => ({ start: keptStart, end: keptEnd })),
        setExportProgress,
        info?.duration,
      );
      setWorkState("done");
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") {
        setWorkState("done");
        return;
      }
      setError(cause instanceof Error ? cause.message : String(cause));
      setWorkState("error");
    }
  }

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Brand className={styles.brand} label="LOCAL" priority />
        <div className={styles.statusRow}>
          <span data-ok={compatibility.decode}>WebCodecs</span>
          <span data-ok={compatibility.webGpu === "available"}>WebGPU</span>
          <span data-ok={compatibility.encode}>Encode</span>
          <span data-ok={compatibility.directDisk}>Direct-to-disk</span>
          <span data-ok>Nothing uploaded</span>
        </div>
      </header>

      {uiFixtureMode && (
        <div className={styles.fixtureNotice} role="status">
          <strong>UI fixture mode.</strong> These {analysis?.intervals.length ?? 0} predictions were
          cached from a real browser on-device run. No video is opened or decoded on this route.
          Choose a local file below to switch into the live pipeline.
        </div>
      )}

      <section className={styles.hero}>
        <p className={styles.eyebrow}>DESKTOP PROOF OF CONCEPT · MODEL-9C92B8E9333F</p>
        <h1>Your match stays<br /><em>on this machine.</em></h1>
        <p className={styles.lede}>
          The hosted app supplies code and a 100 KB model bundle. Your browser opens the local
          file, derives audiovisual features, runs all three CPU heads, and exports from the
          original media. No video bytes are sent to this server.
        </p>
        <div className={styles.architecture} aria-label="Local processing architecture">
          <span><b>01</b> Original file</span><i>→</i>
          <span><b>02</b> 192×108 · 4 Hz features</span><i>→</i>
          <span><b>03</b> CPU inference</span><i>→</i>
          <span><b>04</b> Original-size MP4</span>
        </div>
      </section>

      {compatibility.checked && (!compatibility.secureContext || !compatibility.decode) && (
        <div className={styles.compatibilityNotice} role="status">
          <strong>Browser media processing is unavailable on this origin.</strong>{" "}
          {!compatibility.secureContext
            ? "WebCodecs is restricted on plain HTTP. Deploy this web app over HTTPS to analyze local files; the files still remain on-device."
            : "Use a current desktop Chrome or Edge build with WebCodecs enabled."}
        </div>
      )}

      <section className={styles.importCard}>
        <div>
          <p className={styles.step}>STEP 1 / LOCAL SOURCE</p>
          <h2>{sourceName ?? "Choose a match recording"}</h2>
          <p>
            {uiFixtureMode
              ? "Cached browser prediction loaded for fast UI work. Opening another file starts the real local pipeline."
              : "MP4, MOV, WebM, and Matroska are demuxed lazily. A 4K file is not copied into app storage or uploaded."}
          </p>
        </div>
        <label
          className={styles.fileButton}
          data-disabled={busy || (compatibility.checked && !compatibility.decode)}
        >
          {sourceName ? "Choose another file" : "Open local video"}
          <input
            type="file"
            accept="video/*,.mkv,.webm,.mov,.mp4"
            disabled={busy || (compatibility.checked && !compatibility.decode)}
            onChange={(event) => void chooseFile(event.target.files?.[0] ?? null)}
          />
        </label>
      </section>

      {error && <div className={styles.error} role="alert"><strong>Local operation stopped.</strong> {error}</div>}

      {(file || uiFixtureMode) && (
        <section className={styles.workspace}>
          <div className={styles.viewerColumn}>
            <div
              className={styles.videoStage}
              style={{ aspectRatio: info ? `${info.width} / ${info.height}` : "16 / 9" }}
            >
              {previewUrl ? (
                <video
                  ref={videoRef}
                  src={previewUrl}
                  controls
                  preload="metadata"
                  onError={() => setPreviewWarning(true)}
                />
              ) : uiFixtureMode ? (
                <div className={styles.fixtureStage}>
                  <span>ON-DEVICE PREDICTION CACHE</span>
                  <strong>{info?.width}×{info?.height}</strong>
                  <small>Video intentionally omitted</small>
                </div>
              ) : null}
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
              {analysis?.intervals.map((interval) =>
                selectedId === interval.id ? (
                  <div className={styles.nowPlaying} key={interval.id}>{interval.id}</div>
                ) : null,
              )}
            </div>
            {previewWarning && (
              <p className={styles.inlineWarning}>
                The HTML player cannot preview this container, but Mediabunny/WebCodecs may still analyze it.
              </p>
            )}

            {analysis && info && (
              <div className={styles.timelineBlock}>
                <div className={styles.timelineLegend}>
                  <span><i data-kind="prediction" /> Prediction</span>
                  <span><i data-kind="export" /> Padded export · overlaps merged</span>
                </div>
                <div className={styles.timeline} aria-label="Candidate rally and padded export timeline">
                  <div className={styles.exportRanges} aria-label="Merged padded export sections">
                    {editList.map((interval, index) => (
                      <button
                        type="button"
                        key={`${interval.keptStart}-${interval.keptEnd}`}
                        aria-label={`Seek to padded export section ${index + 1}`}
                        title={`Export ${index + 1}: ${preciseTime(interval.keptStart)}–${preciseTime(interval.keptEnd)} · ${interval.rallyIds.join(", ")}`}
                        style={{
                          left: `${(interval.keptStart / info.duration) * 100}%`,
                          width: `${Math.max(0.2, ((interval.keptEnd - interval.keptStart) / info.duration) * 100)}%`,
                        }}
                        onClick={() => seek(interval.keptStart)}
                      />
                    ))}
                  </div>
                  <div className={styles.predictionRanges} aria-label="Model predictions">
                    {analysis.intervals.map((interval) => (
                      <button
                        type="button"
                        key={interval.id}
                        aria-label={`Seek to prediction ${interval.id}`}
                        title={`${interval.id} · ${preciseTime(interval.start)}–${preciseTime(interval.end)}`}
                        data-selected={selectedId === interval.id}
                        data-included={interval.included}
                        style={{
                          left: `${(interval.start / info.duration) * 100}%`,
                          width: `${Math.max(0.2, ((interval.end - interval.start) / info.duration) * 100)}%`,
                        }}
                        onClick={() => {
                          setSelectedId(interval.id);
                          seek(interval.start);
                        }}
                      />
                    ))}
                  </div>
                </div>
                {uiFixtureMode && (
                  <div className={styles.fixtureRolls} aria-label="Cached timeline padding controls">
                    <label>
                      Pre-roll <output>{preRoll}s</output>
                      <input
                        type="range"
                        min={0}
                        max={8}
                        step={0.5}
                        value={preRoll}
                        onChange={(event) => setPreRoll(Number(event.target.value))}
                      />
                    </label>
                    <label>
                      Post-roll <output>{postRoll}s</output>
                      <input
                        type="range"
                        min={0}
                        max={8}
                        step={0.5}
                        value={postRoll}
                        onChange={(event) => setPostRoll(Number(event.target.value))}
                      />
                    </label>
                  </div>
                )}
                <p className={styles.timelineSummary}>
                  {editList.length} export {editList.length === 1 ? "section" : "sections"} after
                  applying {preRoll}s pre-roll and {postRoll}s post-roll. Touching or overlapping
                  padding is exported once.
                </p>
              </div>
            )}
          </div>

          <aside className={styles.inspector}>
            <p className={styles.step}>MEDIA PROBE</p>
            {info ? (
              <dl>
                <div><dt>Display</dt><dd>{info.width}×{info.height}</dd></div>
                <div><dt>Duration</dt><dd>{formatTime(info.duration)}</dd></div>
                <div><dt>Video</dt><dd>{info.videoCodec}</dd></div>
                <div><dt>Audio</dt><dd>{info.hasAudio ? `${info.audioCodec} · ${info.sampleRate} Hz` : "None"}</dd></div>
                <div><dt>File size</dt><dd>{compactBytes(displayedFileSize)}</dd></div>
                <div><dt>Decode</dt><dd>{info.canDecodeVideo && info.canDecodeAudio ? "Ready" : "Unsupported"}</dd></div>
              </dl>
            ) : (
              <p className={styles.muted}>Reading metadata…</p>
            )}

            {info && (
              <section className={styles.diagnostics} aria-labelledby="device-path-title">
                <header>
                  <div>
                    <p className={styles.step}>PERFORMANCE DIAGNOSTICS</p>
                    <h3 id="device-path-title">This video on this device</h3>
                  </div>
                  <span
                    className={styles.diagnosticState}
                    data-state={
                      mediaDiagnostics.checking
                        ? "checking"
                        : mediaDiagnostics.webCodecsSupported === false
                          ? "limited"
                          : "ready"
                    }
                  >
                    {mediaDiagnostics.checking
                      ? "CHECKING"
                      : mediaDiagnostics.webCodecsSupported === false
                        ? "LIMITED"
                        : "READY"}
                  </span>
                </header>
                <dl className={styles.diagnosticList}>
                  <div>
                    <dt>WebCodecs API</dt>
                    <dd data-tone={compatibility.decode ? "good" : "bad"}>
                      {compatibility.decode ? "Available" : "Unavailable"}
                    </dd>
                  </div>
                  <div>
                    <dt>Selected codec</dt>
                    <dd>
                      {mediaDiagnostics.checking
                        ? "Checking…"
                        : mediaDiagnostics.codec ?? info.videoCodecString ?? info.videoCodec}
                      {mediaDiagnostics.webCodecsSupported !== null &&
                        ` · ${mediaDiagnostics.webCodecsSupported ? "supported" : "unsupported"}`}
                    </dd>
                  </div>
                  <div>
                    <dt>Source load</dt>
                    <dd>
                      {mediaDiagnostics.averageFrameRate
                        ? `${mediaDiagnostics.averageFrameRate.toFixed(1)} fps`
                        : "Frame rate unknown"}
                      {mediaDiagnostics.averageBitrate
                        ? ` · ${compactBitrate(mediaDiagnostics.averageBitrate)}`
                        : ""}
                    </dd>
                  </div>
                  <div>
                    <dt>Decode preference</dt>
                    <dd>Prefer hardware</dd>
                  </div>
                  <div>
                    <dt>Power-efficient decode</dt>
                    <dd>
                      {mediaDiagnostics.powerEfficient === null
                        ? compatibility.mediaCapabilities
                          ? "Not reported"
                          : "API unavailable"
                        : mediaDiagnostics.powerEfficient
                          ? "Reported"
                          : "Not reported"}
                    </dd>
                  </div>
                  <div>
                    <dt>Smooth decode</dt>
                    <dd>
                      {mediaDiagnostics.smooth === null
                        ? "Not reported"
                        : mediaDiagnostics.smooth
                          ? "Reported"
                          : "Not reported"}
                    </dd>
                  </div>
                  <div>
                    <dt>WebGPU</dt>
                    <dd data-tone={compatibility.webGpu === "available" ? "good" : undefined}>
                      {compatibility.webGpu === "checking"
                        ? "Checking…"
                        : compatibility.webGpu === "available"
                          ? "Available · not used yet"
                          : "Unavailable"}
                    </dd>
                  </div>
                  <div>
                    <dt>GPU adapter</dt>
                    <dd title={compatibility.gpuName ?? undefined}>
                      {compatibility.gpuName
                        ? `${compatibility.gpuName}${compatibility.gpuSource ? ` · ${compatibility.gpuSource}` : ""}`
                        : "Identity hidden"}
                    </dd>
                  </div>
                  <div>
                    <dt>CPU capacity</dt>
                    <dd>
                      {compatibility.logicalProcessors
                        ? `${compatibility.logicalProcessors} logical processors`
                        : "Not reported"}
                    </dd>
                  </div>
                  <div>
                    <dt>Device memory</dt>
                    <dd>
                      {compatibility.deviceMemoryGb
                        ? `About ${compatibility.deviceMemoryGb} GB`
                        : "Not exposed"}
                    </dd>
                  </div>
                  <div>
                    <dt>CPU / GPU usage</dt>
                    <dd>Not exposed by browsers</dd>
                  </div>
                  <div>
                    <dt>Screen wake lock</dt>
                    <dd data-tone={wakeLockState === "active" ? "good" : undefined}>
                      {!compatibility.screenWakeLock || wakeLockState === "unavailable"
                        ? "Unavailable"
                        : wakeLockState === "requesting"
                          ? "Requesting…"
                          : wakeLockState === "active"
                            ? "Active during analysis"
                            : wakeLockState === "paused"
                              ? "Paused while tab is hidden"
                              : wakeLockState === "released"
                                ? "Released after analysis"
                                : wakeLockState === "denied"
                                  ? "Request denied"
                                  : "Ready for analysis"}
                    </dd>
                  </div>
                  <div>
                    <dt>Feature resume</dt>
                    <dd data-tone={featureCacheState?.enabled ? "good" : undefined}>
                      {featureCacheState
                        ? featureCacheState.enabled
                          ? featureCacheState.resumedRows > 0
                            ? `Resumed ${featureCacheState.resumedRows.toLocaleString()} · ${featureCacheState.savedRows.toLocaleString()} saved`
                            : `${featureCacheState.savedRows.toLocaleString()} frames saved`
                          : "IndexedDB unavailable"
                        : `IndexedDB · every ${FEATURE_CACHE_CHUNK_ROWS / ANALYSIS_FPS}s of video`}
                    </dd>
                  </div>
                  {compatibility.jsHeapUsedBytes !== null && (
                    <div>
                      <dt>JavaScript heap</dt>
                      <dd>
                        {compactBytes(compatibility.jsHeapUsedBytes)}
                        {compatibility.jsHeapLimitBytes
                          ? ` / ${compactBytes(compatibility.jsHeapLimitBytes)}`
                          : ""}
                      </dd>
                    </div>
                  )}
                  {analysisElapsedSeconds !== null && (
                    <>
                      <div>
                        <dt>Analysis elapsed</dt>
                        <dd>{preciseTime(analysisElapsedSeconds)}</dd>
                      </div>
                      <div>
                        <dt>Feature speed</dt>
                        <dd>
                          {featurePerformance?.generatedFrames === 0 &&
                          featureCacheState?.resumedRows
                            ? featureCacheState.complete
                              ? "Cache hit · no new frames timed"
                              : "Waiting for first new frame · cache excluded"
                            : processingRate === null
                            ? "Starting…"
                            : `${processingRate.toFixed(2)}× real time · ${(
                                processingRate * ANALYSIS_FPS
                              ).toFixed(1)} frames/s · cache excluded`}
                        </dd>
                      </div>
                    </>
                  )}
                  {featurePerformance && (
                    <>
                      <div className={styles.diagnosticSectionRow}>
                        <dt>Fresh frames profiled</dt>
                        <dd>
                          {featurePerformance.generatedFrames.toLocaleString()} generated
                          {featurePerformance.sampledFrames !== featurePerformance.generatedFrames
                            ? ` · ${featurePerformance.sampledFrames.toLocaleString()} incl. warm-up`
                            : ""}
                        </dd>
                      </div>
                      <div>
                        <dt>Video feature wall time</dt>
                        <dd>{timingDuration(featurePerformance.videoElapsedMs)}</dd>
                      </div>
                      <div>
                        <dt>Extraction execution</dt>
                        <dd data-tone={featurePerformance.workerActive ? "good" : undefined}>
                          {featurePerformance.sampledFrames === 0
                            ? "Not used · complete cache"
                            : featurePerformance.workerActive
                            ? "Dedicated worker · two-frame queue"
                            : "Main thread fallback"}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          {featurePerformance.workerActive
                            ? "Decode blocking"
                            : "Decode + canvas blocking"}
                        </dt>
                        <dd>
                          {timingSummary(
                            featurePerformance.decoderCanvasMs,
                            featurePerformance,
                          )}
                        </dd>
                      </div>
                      {!featurePerformance.workerActive &&
                        featurePerformance.canvasDrawFrames > 0 && (
                        <>
                          <div className={styles.diagnosticSubstage}>
                            <dt>↳ Decoder / sample wait</dt>
                            <dd>
                              {timingSummary(
                                featurePerformance.decoderWaitMs,
                                featurePerformance,
                              )}
                            </dd>
                          </div>
                          <div className={styles.diagnosticSubstage}>
                            <dt>↳ Canvas draw</dt>
                            <dd>
                              {timingSummary(
                                featurePerformance.canvasDrawMs,
                                featurePerformance,
                              )}
                            </dd>
                          </div>
                        </>
                      )}
                      {!featurePerformance.workerActive &&
                        featurePerformance.decoderOverlapMs >= 0.5 && (
                        <div className={styles.diagnosticSubstage}>
                          <dt>↳ Decode request overlapped</dt>
                          <dd>
                            {timingSummary(
                              featurePerformance.decoderOverlapMs,
                              featurePerformance,
                            )} · hidden by other work
                          </dd>
                        </div>
                      )}
                      <div>
                        <dt>
                          {featurePerformance.workerActive
                            ? "Worker feature compute"
                            : "Feature extraction"}
                        </dt>
                        <dd>
                          {timingSummary(featurePerformance.extractionMs, featurePerformance)}
                        </dd>
                      </div>
                      {featurePerformance.workerActive && (
                        <>
                          <div className={styles.diagnosticSubstage}>
                            <dt>↳ Main thread waiting for worker</dt>
                            <dd>
                              {timingSummary(
                                featurePerformance.workerBlockingMs,
                                featurePerformance,
                              )}
                            </dd>
                          </div>
                          <div className={styles.diagnosticSubstage}>
                            <dt>↳ Worker compute overlapped</dt>
                            <dd>
                              {timingSummary(
                                featurePerformance.workerOverlapMs,
                                featurePerformance,
                              )} · hidden by decoding
                            </dd>
                          </div>
                          <div className={styles.diagnosticSubstage}>
                            <dt>↳ Canvas draw</dt>
                            <dd>
                              {timingSummary(
                                featurePerformance.canvasDrawMs,
                                featurePerformance,
                              )}
                            </dd>
                          </div>
                        </>
                      )}
                      <div className={styles.diagnosticSubstage}>
                        <dt>↳ Canvas readback</dt>
                        <dd>
                          {timingSummary(
                            featurePerformance.canvasReadbackMs,
                            featurePerformance,
                          )}
                        </dd>
                      </div>
                      <div className={styles.diagnosticSubstage}>
                        <dt>↳ Image filters</dt>
                        <dd>
                          {timingSummary(
                            featurePerformance.imageOperationsMs,
                            featurePerformance,
                          )}
                        </dd>
                      </div>
                      <div className={styles.diagnosticSubstage}>
                        <dt>↳ Phase correlation</dt>
                        <dd>
                          {timingSummary(
                            featurePerformance.phaseCorrelationMs,
                            featurePerformance,
                          )}
                        </dd>
                      </div>
                      <div className={styles.diagnosticSubstage}>
                        <dt>↳ Optical flow</dt>
                        <dd>
                          {timingSummary(
                            featurePerformance.opticalFlowMs,
                            featurePerformance,
                          )}
                        </dd>
                      </div>
                      <div className={styles.diagnosticSubstage}>
                        <dt>↳ JavaScript stats</dt>
                        <dd>
                          {timingSummary(
                            featurePerformance.javascriptMs,
                            featurePerformance,
                          )}
                        </dd>
                      </div>
                      {extractionOtherMs >= 0.5 && (
                        <div className={styles.diagnosticSubstage}>
                          <dt>↳ Allocation + cleanup</dt>
                          <dd>{timingSummary(extractionOtherMs, featurePerformance)}</dd>
                        </div>
                      )}
                      <div>
                        <dt>OpenCV startup</dt>
                        <dd>
                          {timingSummary(
                            featurePerformance.openCvLoadMs,
                            featurePerformance,
                            false,
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>IndexedDB I/O</dt>
                        <dd>
                          {timingSummary(
                            featurePerformance.cacheIoMs,
                            featurePerformance,
                            false,
                          )} · excluded from frames
                        </dd>
                      </div>
                      {videoPipelineOtherMs >= 0.5 && (
                        <div>
                          <dt>Pipeline overhead</dt>
                          <dd>
                            {timingSummary(
                              videoPipelineOtherMs,
                              featurePerformance,
                              false,
                            )}
                          </dd>
                        </div>
                      )}
                    </>
                  )}
                </dl>
                <p className={styles.pipelineNote}>
                  Video decode requests hardware acceleration, but browsers do not confirm which
                  decoder they selected. When supported, visual features run in OpenCV WASM on a
                  dedicated worker with a two-frame decode queue; model inference still runs on the
                  CPU. WebGPU availability does not accelerate this version.
                  Feature checkpoints stay in this browser&apos;s IndexedDB and are keyed to the exact
                  file and crop. After a refresh, choose the same file again to resume. Stage timing
                  and feature speed count only frames generated in the current run; restored frames
                  are excluded.
                </p>
              </section>
            )}

            <RoiControls
              roi={roi}
              profile={profile}
              disabled={busy}
              onChange={updateRoi}
              onReset={() => {
                const inferred = inferRoiProfile(sourceName ?? "");
                setProfile(inferred);
                setRoi(inferred.roi);
              }}
              onFullFrame={() => {
                const full = fullFrameRoi();
                setProfile(full);
                setRoi(full.roi);
              }}
            />

            <label className={styles.sourceMode}>
              <span>This selected file is</span>
              <select
                value={featurePath}
                disabled={busy}
                onChange={(event) => setFeaturePath(event.target.value as OnDeviceAnalysis["featurePath"])}
              >
                <option value="raw-virtual-proxy">Original / high-quality source</option>
                <option value="training-proxy">FFmpeg training-style proxy</option>
              </select>
            </label>
            <p className={styles.modeNote}>
              {featurePath === "raw-virtual-proxy"
                ? "Decoded directly from the original into transient 192×108 frames. This is feasible, but it is a new feature distribution versus training."
                : "Use this only when the selected file really is the 960×540 H.264/AAC proxy. It is the parity target."}
            </p>

            <button
              className={styles.analyzeButton}
              type="button"
              disabled={!hasOpenedMedia || !info || busy || !info.canDecodeVideo || !compatibility.decode}
              onClick={() => void runAnalysis()}
            >
              {workState === "analyzing"
                ? "Analyzing locally…"
                : uiFixtureMode && !hasOpenedMedia
                  ? "Open a file for live analysis"
                  : analysis
                    ? "Run again"
                    : "Analyze on this device"}
            </button>
          </aside>
        </section>
      )}

      {analysisProgress && (workState === "analyzing" || workState === "opening") && (
        <section className={styles.progressCard} aria-live="polite">
          <div>
            <p className={styles.step}>{analysisProgress.stage.toUpperCase()}</p>
            <strong>{analysisProgress.detail}</strong>
          </div>
          <output>{Math.round(percent(analysisProgress))}%</output>
          <div className={styles.progressTrack}><i style={{ width: `${percent(analysisProgress)}%` }} /></div>
          <p>Keep this tab open. Whole-recording percentile ranks and future context require reaching the end before inference.</p>
        </section>
      )}

      {analysis && info && (
        <section className={styles.results}>
          <header>
            <div>
              <p className={styles.step}>STEP 3 / REVIEW</p>
              <h2>{analysis.intervals.length} candidate rallies</h2>
            </div>
            <div className={styles.resultStats}>
              <span><b>{analysis.intervals.filter((item) => item.included).length}</b> kept</span>
              <span><b>{formatTime(keptSeconds)}</b> output</span>
              <span><b>CPU</b> inference</span>
            </div>
          </header>
          <div className={styles.rallyList}>
            {analysis.intervals.map((interval) => (
              <article key={interval.id} data-selected={selectedId === interval.id}>
                <button
                  type="button"
                  className={styles.rallySeek}
                  onClick={() => {
                    setSelectedId(interval.id);
                    seek(interval.start);
                  }}
                >
                  <strong>{interval.id}</strong>
                  <span>{preciseTime(interval.start)} → {preciseTime(interval.end)}</span>
                </button>
                <label>
                  <span>Start</span>
                  <input
                    type="number"
                    min={0}
                    max={interval.end}
                    step={0.05}
                    value={interval.start.toFixed(2)}
                    onChange={(event) => updateInterval(interval.id, { start: Number(event.target.value) })}
                  />
                </label>
                <label>
                  <span>End</span>
                  <input
                    type="number"
                    min={interval.start}
                    max={info.duration}
                    step={0.05}
                    value={interval.end.toFixed(2)}
                    onChange={(event) => updateInterval(interval.id, { end: Number(event.target.value) })}
                  />
                </label>
                <span className={styles.confidence}>{Math.round(interval.confidence * 100)}%</span>
                <label className={styles.keepToggle}>
                  <input
                    type="checkbox"
                    checked={interval.included}
                    onChange={(event) => updateInterval(interval.id, { included: event.target.checked })}
                  />
                  <span>{interval.included ? "Keep" : "Skip"}</span>
                </label>
              </article>
            ))}
          </div>
        </section>
      )}

      {analysis && info && file && !uiFixtureMode && (
        <section className={styles.exportCard}>
          <div className={styles.exportIntro}>
            <p className={styles.step}>STEP 4 / ORIGINAL-QUALITY EXPORT</p>
            <h2>Return to the master file.</h2>
            <p>
              Selected intervals are decoded again from <strong>{exportFile?.name}</strong>, joined,
              and encoded once at its original display dimensions. The 192×108 analysis frames are
              never used for output.
            </p>
            <p className={styles.exportCaveat}>
              Exact cut boundaries require transcoding. “Original-quality” here means original
              resolution plus a very-high-quality single encode—not byte-identical smart copy.
            </p>
          </div>
          <div className={styles.exportControls}>
            <div className={styles.rolls}>
              <label>Pre-roll <output>{preRoll}s</output><input type="range" min={0} max={8} step={0.5} value={preRoll} onChange={(event) => setPreRoll(Number(event.target.value))} /></label>
              <label>Post-roll <output>{postRoll}s</output><input type="range" min={0} max={8} step={0.5} value={postRoll} onChange={(event) => setPostRoll(Number(event.target.value))} /></label>
            </div>
            <label className={styles.masterPicker}>
              Use a different raw master
              <input
                type="file"
                accept="video/*,.mkv,.webm,.mov,.mp4"
                disabled={busy}
                onChange={(event) => setExportFile(event.target.files?.[0] ?? file)}
              />
            </label>
            {exportFile !== file && (
              <p className={styles.modeNote}>The alternate master must have the same timeline as the analyzed proxy.</p>
            )}
            <div className={styles.exportActions}>
              <button
                type="button"
                disabled={
                  busy ||
                  !editList.length ||
                  !compatibility.encode ||
                  !compatibility.directDisk
                }
                title={
                  compatibility.encode && compatibility.directDisk
                    ? undefined
                    : "Original-size export requires WebCodecs encoders and direct-to-disk access."
                }
                onClick={() => void exportReel()}
              >
                {workState === "exporting" ? "Encoding…" : "Save original-size MP4"}
              </button>
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={() =>
                  downloadEditDecisionList(
                    file,
                    editList.map(({ keptStart, keptEnd }) => ({ start: keptStart, end: keptEnd })),
                    info.duration,
                  )
                }
              >
                Download JSON EDL
              </button>
            </div>
            {exportProgress && workState === "exporting" && (
              <div className={styles.exportProgress}>
                <span>{exportProgress.detail}</span>
                <div><i style={{ width: `${exportProgress.totalSeconds ? (exportProgress.completedSeconds / exportProgress.totalSeconds) * 100 : 0}%` }} /></div>
              </div>
            )}
          </div>
        </section>
      )}

      <footer className={styles.footer}>
        <p><strong>POC boundary:</strong> Chrome/Edge desktop first. Browser decoding and encoding still depend on the machine&apos;s codec support.</p>
        <div className={styles.footerLinks}>
          <Link href={uiFixtureMode ? "/on-device" : "/on-device-ui"}>
            {uiFixtureMode ? "Open live pipeline" : "Open cached UI fixture"} →
          </Link>
          <Link href="/">Back to dataset review →</Link>
        </div>
      </footer>
    </main>
  );
}
