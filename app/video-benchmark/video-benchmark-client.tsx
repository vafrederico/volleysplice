"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import { ANALYSIS_FPS } from "@/lib/on-device/feature-schema";
import {
  type OpenedMedia,
  openLocalMedia,
  openUrlMedia,
} from "@/lib/on-device/media";
import {
  DEFAULT_FEATURE_REDUCTION_KERNEL,
  DEFAULT_VIDEO_DECODE_STRATEGY,
  extractBrowserFeatures,
  VIDEO_DECODER_HARDWARE_ACCELERATION,
} from "@/lib/on-device/pipeline";
import { fullFrameRoi, inferRoiProfile } from "@/lib/on-device/roi";
import { DEFAULT_ON_DEVICE_RUNTIME_VARIANT } from "@/lib/on-device/runtime-variants";
import type {
  AnalysisProgress,
  FeatureExtractionPerformance,
  FeatureReductionKernel,
  NormalizedRoi,
  OnDeviceMediaInfo,
  VideoDecoderAcceleration,
  VideoDecodeStrategy,
} from "@/lib/on-device/types";

import shared from "../audio-benchmark/audio-benchmark.module.css";
import styles from "./video-benchmark.module.css";

type NasSource = {
  name: string;
  size: number;
  modifiedAt: string;
  mediaUrl: string;
};

type SelectedSource =
  | { kind: "nas"; name: string; size: number; mediaUrl: string }
  | { kind: "local"; name: string; size: number; file: File };

type MediaDiagnostics = {
  codec: string | null;
  averageFrameRate: number | null;
  averageBitrate: number | null;
  smooth: boolean | null;
  powerEfficient: boolean | null;
};

type BrowserSnapshot = {
  secureContext: boolean;
  logicalProcessors: number | null;
  deviceMemoryGb: number | null;
  gpu: string | null;
  userAgent: string;
};

type BenchmarkRun = {
  id: string;
  createdAt: string;
  sourceName: string;
  sourceSize: number;
  durationSeconds: number;
  roiLabel: string;
  roi: NormalizedRoi;
  media: OnDeviceMediaInfo;
  diagnostics: MediaDiagnostics;
  browser: BrowserSnapshot;
  performance: FeatureExtractionPerformance;
  wallMs: number;
  longTaskCount: number;
  longTaskMs: number;
  longestTaskMs: number;
  heapBeforeBytes: number | null;
  heapPeakBytes: number | null;
  heapAfterBytes: number | null;
  checksum: number;
};

type PerformanceWithMemory = Performance & {
  memory?: { usedJSHeapSize: number };
};

type NavigatorWithMemory = Navigator & { deviceMemory?: number };

const EMPTY_DIAGNOSTICS: MediaDiagnostics = {
  codec: null,
  averageFrameRate: null,
  averageBitrate: null,
  smooth: null,
  powerEfficient: null,
};

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

function seconds(value: number): string {
  if (value < 60) return `${value.toFixed(value < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}m ${Math.round(value % 60)}s`;
}

function milliseconds(value: number): string {
  return value >= 1000
    ? `${(value / 1000).toFixed(2)}s`
    : `${value.toFixed(0)}ms`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "The video benchmark failed unexpectedly.";
}

function heapUsed(): number | null {
  return (performance as PerformanceWithMemory).memory?.usedJSHeapSize ?? null;
}

function checksum(values: Float32Array): number {
  let total = 0;
  const stride = Math.max(1, Math.floor(values.length / 4096));
  for (let index = 0; index < values.length; index += stride)
    total += values[index];
  return total;
}

function webGlRenderer(): string | null {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
  if (!context) return null;
  const extension = context.getExtension("WEBGL_debug_renderer_info");
  const renderer = extension
    ? context.getParameter(extension.UNMASKED_RENDERER_WEBGL)
    : context.getParameter(context.RENDERER);
  return typeof renderer === "string" && renderer.trim()
    ? renderer.trim()
    : null;
}

function browserSnapshot(): BrowserSnapshot {
  const browser = navigator as NavigatorWithMemory;
  return {
    secureContext: window.isSecureContext,
    logicalProcessors: navigator.hardwareConcurrency || null,
    deviceMemoryGb: browser.deviceMemory ?? null,
    gpu: webGlRenderer(),
    userAgent: navigator.userAgent,
  };
}

function videoContentType(mimeType: string, codec: string): string {
  const container = mimeType.split(";", 1)[0]?.trim() || "video/mp4";
  return `${container}; codecs="${codec.replaceAll('"', "")}"`;
}

async function inspectMedia(
  media: OpenedMedia,
  fileSize: number,
): Promise<MediaDiagnostics> {
  const [decoderConfig, packetStats] = await Promise.all([
    media.videoTrack.getDecoderConfig().catch(() => null),
    media.videoTrack.computePacketStats(120).catch(() => null),
  ]);
  const codec = decoderConfig?.codec ?? media.info.videoCodecString;
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
      // MediaCapabilities may reject otherwise decodable container/codec pairs.
    }
  }
  return { codec, averageFrameRate, averageBitrate, smooth, powerEfficient };
}

function diagnosis(run: BenchmarkRun): string {
  const timing = run.performance;
  const sourceRatio =
    timing.decodedSourceFrames && timing.sampledFrames
      ? timing.decodedSourceFrames / timing.sampledFrames
      : null;
  if (sourceRatio !== null && sourceRatio > 4) {
    return `Sequential decoding reads ${sourceRatio.toFixed(1)} source frames per analyzed frame. Compare sparse seeking to test whether avoiding discarded work offsets the added seek overhead.`;
  }
  if (timing.decoderCanvasMs > timing.workerBlockingMs * 1.5) {
    return "Decoder and frame delivery dominate the critical path. Compare sparse versus sequential decode and the hardware-acceleration request.";
  }
  if (timing.workerBlockingMs > timing.decoderCanvasMs) {
    return "Visual feature work is making the decoder wait. The reduction kernel and worker pipeline are the highest-leverage experiments.";
  }
  if (timing.workerOverlapMs > timing.workerBlockingMs) {
    return "Most feature work overlaps video delivery. The pipeline is behaving as intended; duration scaling and decoder throughput are the next checks.";
  }
  return "No single measured phase dominates. Compare repeated full-file runs to distinguish startup cost, decode variance, and sustained throughput.";
}

function requestedDuration(value: string, mediaDuration: number): number {
  return value === "full"
    ? mediaDuration
    : Math.min(mediaDuration, Number(value));
}

function boolLabel(value: boolean | null): string {
  return value === null ? "Unknown" : value ? "Yes" : "No";
}

export function VideoBenchmarkClient() {
  const [nasSources, setNasSources] = useState<NasSource[]>([]);
  const [nasRoot, setNasRoot] = useState<string | null>(null);
  const [selected, setSelected] = useState<SelectedSource | null>(null);
  const [mediaInfo, setMediaInfo] = useState<OnDeviceMediaInfo | null>(null);
  const [diagnostics, setDiagnostics] =
    useState<MediaDiagnostics>(EMPTY_DIAGNOSTICS);
  const [duration, setDuration] = useState("full");
  const [decodeStrategy, setDecodeStrategy] = useState<VideoDecodeStrategy>(
    DEFAULT_VIDEO_DECODE_STRATEGY,
  );
  const [decoderAcceleration, setDecoderAcceleration] =
    useState<VideoDecoderAcceleration>(VIDEO_DECODER_HARDWARE_ACCELERATION);
  const [reductionKernel, setReductionKernel] =
    useState<FeatureReductionKernel>(DEFAULT_FEATURE_REDUCTION_KERNEL);
  const [useFullFrame, setUseFullFrame] = useState(false);
  const [status, setStatus] = useState<
    "idle" | "opening" | "ready" | "running" | "error"
  >("idle");
  const [progress, setProgress] = useState<AnalysisProgress | null>(null);
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [failure, setFailure] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [browser, setBrowser] = useState<BrowserSnapshot | null>(null);
  const runStartedAt = useRef<number | null>(null);

  useEffect(() => {
    setBrowser(browserSnapshot());
    fetch("/api/audio-benchmark/sources", { cache: "no-store" })
      .then(async (response) =>
        response.ok
          ? (response.json() as Promise<{ root: string; sources: NasSource[] }>)
          : null,
      )
      .then((catalog) => {
        if (!catalog) return;
        setNasRoot(catalog.root);
        setNasSources(catalog.sources);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (status !== "running") return;
    const update = () => {
      if (runStartedAt.current !== null)
        setElapsedMs(performance.now() - runStartedAt.current);
    };
    update();
    const timer = window.setInterval(update, 100);
    return () => window.clearInterval(timer);
  }, [status]);

  const openSelected = useCallback(
    (source: SelectedSource): Promise<OpenedMedia> => {
      if (source.kind === "local") return openLocalMedia(source.file);
      return openUrlMedia(new URL(source.mediaUrl, window.location.origin));
    },
    [],
  );

  const inspect = useCallback(
    async (source: SelectedSource) => {
      setSelected(source);
      setMediaInfo(null);
      setDiagnostics(EMPTY_DIAGNOSTICS);
      setFailure(null);
      setStatus("opening");
      try {
        const media = await openSelected(source);
        try {
          setMediaInfo(media.info);
          setDiagnostics(await inspectMedia(media, source.size));
          setStatus("ready");
        } finally {
          media.input.dispose();
        }
      } catch (error) {
        setFailure(errorMessage(error));
        setStatus("error");
      }
    },
    [openSelected],
  );

  const runBenchmark = useCallback(async () => {
    if (!selected || !mediaInfo || !browser || status === "running") return;
    const durationSeconds = requestedDuration(duration, mediaInfo.duration);
    const roiProfile = useFullFrame
      ? fullFrameRoi()
      : inferRoiProfile(selected.name);
    setFailure(null);
    setProgress(null);
    setStatus("running");
    setElapsedMs(0);
    runStartedAt.current = performance.now();

    const longTasks: PerformanceEntry[] = [];
    let observer: PerformanceObserver | null = null;
    if (typeof PerformanceObserver !== "undefined") {
      try {
        observer = new PerformanceObserver((list) =>
          longTasks.push(...list.getEntries()),
        );
        observer.observe({ type: "longtask", buffered: false });
      } catch {
        observer = null;
      }
    }
    const heapBefore = heapUsed();
    let heapPeak = heapBefore;
    const heapTimer = window.setInterval(() => {
      const current = heapUsed();
      if (current !== null) heapPeak = Math.max(heapPeak ?? current, current);
    }, 100);
    let media: OpenedMedia | null = null;
    try {
      media = await openSelected(selected);
      const startedAt = performance.now();
      const sequence = await extractBrowserFeatures(
        { ...media, audioTrack: null },
        roiProfile.roi,
        DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
        setProgress,
        undefined,
        {
          detailedProfiling: true,
          decodeStrategy,
          decoderAcceleration,
          reductionKernel,
          durationLimitSeconds: durationSeconds,
        },
      );
      const wallMs = performance.now() - startedAt;
      await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      const heapAfter = heapUsed();
      if (heapAfter !== null)
        heapPeak = Math.max(heapPeak ?? heapAfter, heapAfter);
      const longTaskMs = longTasks.reduce(
        (total, entry) => total + entry.duration,
        0,
      );
      setRuns((current) =>
        [
          {
            id: crypto.randomUUID(),
            createdAt: new Date().toISOString(),
            sourceName: selected.name,
            sourceSize: selected.size,
            durationSeconds,
            roiLabel: roiProfile.label,
            roi: roiProfile.roi,
            media: mediaInfo,
            diagnostics,
            browser,
            performance: sequence.performance,
            wallMs,
            longTaskCount: longTasks.length,
            longTaskMs,
            longestTaskMs: longTasks.reduce(
              (largest, entry) => Math.max(largest, entry.duration),
              0,
            ),
            heapBeforeBytes: heapBefore,
            heapPeakBytes: heapPeak,
            heapAfterBytes: heapAfter,
            checksum: checksum(sequence.values),
          },
          ...current,
        ].slice(0, 12),
      );
      setStatus("ready");
      setProgress(null);
    } catch (error) {
      setFailure(errorMessage(error));
      setStatus("error");
    } finally {
      window.clearInterval(heapTimer);
      observer?.disconnect();
      media?.input.dispose();
      if (runStartedAt.current !== null)
        setElapsedMs(performance.now() - runStartedAt.current);
      runStartedAt.current = null;
    }
  }, [
    browser,
    decodeStrategy,
    decoderAcceleration,
    diagnostics,
    duration,
    mediaInfo,
    openSelected,
    reductionKernel,
    selected,
    status,
    useFullFrame,
  ]);

  const latest = runs[0] ?? null;
  const benchmarkDuration = mediaInfo
    ? requestedDuration(duration, mediaInfo.duration)
    : 0;
  const progressPercent =
    progress && progress.total > 0
      ? Math.min(100, Math.max(0, (progress.completed / progress.total) * 100))
      : 0;
  const livePerformance = progress?.performance;
  const liveSpeed =
    livePerformance && livePerformance.videoElapsedMs > 0
      ? livePerformance.generatedVideoSeconds /
        (livePerformance.videoElapsedMs / 1000)
      : null;

  const phaseRows = useMemo(() => {
    if (!latest) return [];
    const timing = latest.performance;
    const startup = timing.openCvLoadMs + timing.reductionKernelLoadMs;
    const measured =
      startup +
      timing.decoderCanvasMs +
      timing.workerBlockingMs +
      timing.cacheIoMs;
    return [
      { label: "Runtime startup", value: startup, tone: "startup" },
      {
        label: "Decoder + frame delivery",
        value: timing.decoderCanvasMs,
        tone: "decode",
      },
      {
        label: "Worker blocking",
        value: timing.workerBlockingMs,
        tone: "worker",
      },
      { label: "Cache I/O", value: timing.cacheIoMs, tone: "cache" },
      {
        label: "Unattributed / orchestration",
        value: Math.max(0, timing.videoElapsedMs - measured),
        tone: "other",
      },
    ];
  }, [latest]);

  function exportRuns() {
    const blob = new Blob(
      [JSON.stringify({ schemaVersion: 1, runs }, null, 2)],
      {
        type: "application/json",
      },
    );
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `volleycut-video-benchmark-${new Date().toISOString().replaceAll(":", "-")}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className={shared.shell}>
      <header className={shared.topbar}>
        <Brand className={shared.brand} label="Video lab" />
        <div className={shared.headerMeta}>
          <span data-ok={browser?.secureContext ?? false}>secure context</span>
          <Link href="/audio-benchmark">Audio lab ↗</Link>
          <Link href="/on-device">On-device app ↗</Link>
        </div>
      </header>

      <section className={shared.hero}>
        <p>
          WebCodecs pipeline profiler · {ANALYSIS_FPS} analysis frames/sec · 160
          × 90
        </p>
        <h1>
          Decode. Measure. <em>Explain.</em>
        </h1>
        <div className={shared.heroAside}>
          <strong>
            {latest
              ? `${(latest.durationSeconds / (latest.performance.videoElapsedMs / 1000)).toFixed(1)}×`
              : "—"}
          </strong>
          <span>latest real-time speed</span>
        </div>
      </section>

      <section className={shared.workbench}>
        <div className={shared.sourcePanel}>
          <div className={shared.sectionHeading}>
            <span>01 / SOURCE</span>
            <strong>Load the complete problem recording</strong>
          </div>
          <label className={shared.filePicker}>
            <input
              type="file"
              accept="video/mp4,video/quicktime,video/*"
              disabled={status === "running"}
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                if (file)
                  void inspect({
                    kind: "local",
                    name: file.name,
                    size: file.size,
                    file,
                  });
              }}
            />
            Choose local video
          </label>
          {nasSources.length > 0 && (
            <label className={shared.nasPicker}>
              <span>Problem recordings on NAS</span>
              <select
                value={selected?.kind === "nas" ? selected.name : ""}
                disabled={status === "running"}
                onChange={(event) => {
                  const source = nasSources.find(
                    (item) => item.name === event.target.value,
                  );
                  if (source) void inspect({ kind: "nas", ...source });
                }}
              >
                <option value="">
                  Select one of {nasSources.length} recordings…
                </option>
                {nasSources.map((source) => (
                  <option key={source.name} value={source.name}>
                    {source.name} · {compactBytes(source.size)}
                  </option>
                ))}
              </select>
              {nasRoot && <small>{nasRoot}</small>}
            </label>
          )}
          <div
            className={shared.sourceReadout}
            data-active={Boolean(mediaInfo)}
          >
            <span>
              {status === "opening"
                ? "Probing track and packet statistics…"
                : (selected?.name ?? "No source selected")}
            </span>
            <strong>{selected ? compactBytes(selected.size) : "—"}</strong>
          </div>
        </div>

        <div className={shared.controlPanel}>
          <div className={shared.sectionHeading}>
            <span>02 / RUN</span>
            <strong>Profile the production visual extractor</strong>
          </div>
          <div className={`${shared.controls} ${styles.videoControls}`}>
            <label>
              <span>Duration</span>
              <select
                value={duration}
                disabled={!mediaInfo || status === "running"}
                onChange={(event) => setDuration(event.target.value)}
              >
                <option value="30">30 seconds</option>
                <option value="60">1 minute</option>
                <option value="300">5 minutes</option>
                <option value="full">Whole file</option>
              </select>
            </label>
            <label>
              <span>Decode</span>
              <select
                value={decodeStrategy}
                disabled={!mediaInfo || status === "running"}
                onChange={(event) =>
                  setDecodeStrategy(event.target.value as VideoDecodeStrategy)
                }
              >
                <option value="sequential">Sequential</option>
                <option value="sparse">Sparse seeks</option>
              </select>
            </label>
            <label>
              <span>Decoder request</span>
              <select
                value={decoderAcceleration}
                disabled={!mediaInfo || status === "running"}
                onChange={(event) =>
                  setDecoderAcceleration(
                    event.target.value as VideoDecoderAcceleration,
                  )
                }
              >
                <option value="prefer-hardware">Prefer hardware</option>
                <option value="no-preference">No preference</option>
              </select>
            </label>
            <label>
              <span>Reduction</span>
              <select
                value={reductionKernel}
                disabled={!mediaInfo || status === "running"}
                onChange={(event) =>
                  setReductionKernel(
                    event.target.value as FeatureReductionKernel,
                  )
                }
              >
                <option value="wasm">WASM</option>
                <option value="javascript">JavaScript</option>
              </select>
            </label>
          </div>
          <label className={styles.roiToggle}>
            <input
              type="checkbox"
              checked={useFullFrame}
              disabled={!mediaInfo || status === "running"}
              onChange={(event) => setUseFullFrame(event.target.checked)}
            />
            <span>Use full frame instead of inferred court ROI</span>
          </label>
          <button
            type="button"
            className={shared.runButton}
            disabled={!mediaInfo || !browser || status === "running"}
            onClick={() => void runBenchmark()}
          >
            {status === "running"
              ? `Running whole pipeline · ${(elapsedMs / 1000).toFixed(1)}s`
              : `Run ${duration === "full" ? "whole-file" : seconds(benchmarkDuration)} video benchmark`}
          </button>
        </div>
      </section>

      {failure && <p className={shared.error}>{failure}</p>}

      {mediaInfo && (
        <section className={`${shared.mediaStrip} ${styles.mediaStrip}`}>
          <div>
            <span>Duration</span>
            <strong>{seconds(mediaInfo.duration)}</strong>
          </div>
          <div>
            <span>Picture</span>
            <strong>
              {mediaInfo.width} × {mediaInfo.height}
            </strong>
          </div>
          <div>
            <span>Codec</span>
            <strong>{diagnostics.codec ?? mediaInfo.videoCodec}</strong>
          </div>
          <div>
            <span>Source fps</span>
            <strong>{diagnostics.averageFrameRate?.toFixed(2) ?? "—"}</strong>
          </div>
          <div>
            <span>Bitrate</span>
            <strong>
              {diagnostics.averageBitrate
                ? `${(diagnostics.averageBitrate / 1_000_000).toFixed(1)} Mbps`
                : "—"}
            </strong>
          </div>
          <div>
            <span>Benchmark</span>
            <strong>{seconds(benchmarkDuration)}</strong>
          </div>
          <div>
            <span>Smooth hint</span>
            <strong>{boolLabel(diagnostics.smooth)}</strong>
          </div>
          <div>
            <span>Power-efficient hint</span>
            <strong>{boolLabel(diagnostics.powerEfficient)}</strong>
          </div>
        </section>
      )}

      {status === "running" && (
        <section className={shared.liveTrace}>
          <div>
            <span>LIVE / VIDEO</span>
            <strong>{progress?.detail ?? "Initializing decoder"}</strong>
          </div>
          <div className={shared.progressTrack}>
            <i style={{ width: `${progressPercent}%` }} />
          </div>
          <b>
            {liveSpeed === null
              ? "measuring"
              : `${liveSpeed.toFixed(2)}× RT · ${livePerformance?.generatedFrames.toLocaleString()} frames`}
          </b>
        </section>
      )}

      {latest && (
        <>
          <section className={shared.results}>
            <header>
              <div className={shared.sectionHeading}>
                <span>03 / LATEST RESULT</span>
                <strong>{latest.sourceName}</strong>
              </div>
              <p>{diagnosis(latest)}</p>
            </header>
            <div className={shared.resultHero}>
              <div>
                <strong>
                  {milliseconds(latest.performance.videoElapsedMs)}
                </strong>
                <span>video wall time</span>
              </div>
              <div>
                <strong>
                  {(
                    latest.durationSeconds /
                    (latest.performance.videoElapsedMs / 1000)
                  ).toFixed(2)}
                  ×
                </strong>
                <span>real-time speed</span>
              </div>
              <div>
                <strong>
                  {(
                    latest.performance.generatedFrames /
                    (latest.performance.videoElapsedMs / 1000)
                  ).toFixed(2)}
                </strong>
                <span>analysis frames/sec</span>
              </div>
              <div>
                <strong>{latest.longTaskCount}</strong>
                <span>main-thread long tasks</span>
              </div>
            </div>
            <div className={shared.breakdown}>
              {phaseRows.map((phase) => {
                const share =
                  latest.performance.videoElapsedMs > 0
                    ? (phase.value / latest.performance.videoElapsedMs) * 100
                    : 0;
                return (
                  <div key={phase.label}>
                    <span>{phase.label}</span>
                    <div>
                      <i
                        data-tone={phase.tone}
                        style={{
                          width: `${Math.max(0.5, Math.min(100, share))}%`,
                        }}
                      />
                    </div>
                    <b>{milliseconds(phase.value)}</b>
                    <small>{share.toFixed(1)}%</small>
                  </div>
                );
              })}
            </div>
            <div className={shared.counters}>
              <span>
                <b>{latest.performance.generatedFrames.toLocaleString()}</b>{" "}
                generated frames
              </span>
              <span>
                <b>{latest.performance.sampledFrames.toLocaleString()}</b>{" "}
                sampled frames
              </span>
              <span>
                <b>
                  {latest.performance.decodedSourceFrames?.toLocaleString() ??
                    "Unavailable in sparse mode"}
                </b>{" "}
                source frames decoded
              </span>
              <span>
                <b>
                  {latest.performance.decodedSourceFrames === null
                    ? "Unavailable"
                    : Math.max(
                        0,
                        latest.performance.decodedSourceFrames -
                          latest.performance.sampledFrames,
                      ).toLocaleString()}
                </b>{" "}
                source frames discarded
              </span>
              <span>
                <b>{latest.performance.canvasDrawFrames.toLocaleString()}</b>{" "}
                canvas draws
              </span>
              <span>
                <b>{milliseconds(latest.longTaskMs)}</b> long-task time · max{" "}
                {milliseconds(latest.longestTaskMs)}
              </span>
            </div>
          </section>

          <section className={styles.telemetry}>
            <header>
              <div className={shared.sectionHeading}>
                <span>04 / DEEP TELEMETRY</span>
                <strong>Browser equivalent of Android diagnostics</strong>
              </div>
            </header>
            <div className={styles.telemetryGrid}>
              <article>
                <h2>Decoder & delivery</h2>
                <dl>
                  <div>
                    <dt>Requested acceleration</dt>
                    <dd>{latest.performance.decoderAcceleration}</dd>
                  </div>
                  <div>
                    <dt>Decode strategy</dt>
                    <dd>{latest.performance.decodeStrategy}</dd>
                  </div>
                  <div>
                    <dt>Decoder + canvas critical</dt>
                    <dd>{milliseconds(latest.performance.decoderCanvasMs)}</dd>
                  </div>
                  <div>
                    <dt>Decoder wait</dt>
                    <dd>{milliseconds(latest.performance.decoderWaitMs)}</dd>
                  </div>
                  <div>
                    <dt>Decoder overlap</dt>
                    <dd>{milliseconds(latest.performance.decoderOverlapMs)}</dd>
                  </div>
                  <div>
                    <dt>Canvas draw</dt>
                    <dd>{milliseconds(latest.performance.canvasDrawMs)}</dd>
                  </div>
                  <div>
                    <dt>Source decode fps</dt>
                    <dd>
                      {latest.performance.decodedSourceFrames === null
                        ? "Unavailable"
                        : (
                            latest.performance.decodedSourceFrames /
                            (latest.performance.videoElapsedMs / 1000)
                          ).toFixed(1)}
                    </dd>
                  </div>
                  <div>
                    <dt>Source frames / sample</dt>
                    <dd>
                      {latest.performance.decodedSourceFrames === null ||
                      latest.performance.sampledFrames === 0
                        ? "Unavailable"
                        : (
                            latest.performance.decodedSourceFrames /
                            latest.performance.sampledFrames
                          ).toFixed(2)}
                    </dd>
                  </div>
                </dl>
              </article>
              <article>
                <h2>Feature worker</h2>
                <dl>
                  <div>
                    <dt>Worker active</dt>
                    <dd>{latest.performance.workerActive ? "Yes" : "No"}</dd>
                  </div>
                  <div>
                    <dt>Total worker work</dt>
                    <dd>{milliseconds(latest.performance.extractionMs)}</dd>
                  </div>
                  <div>
                    <dt>Worker blocking</dt>
                    <dd>{milliseconds(latest.performance.workerBlockingMs)}</dd>
                  </div>
                  <div>
                    <dt>Worker overlap</dt>
                    <dd>{milliseconds(latest.performance.workerOverlapMs)}</dd>
                  </div>
                  <div>
                    <dt>Canvas readback</dt>
                    <dd>{milliseconds(latest.performance.canvasReadbackMs)}</dd>
                  </div>
                  <div>
                    <dt>Image operations</dt>
                    <dd>
                      {milliseconds(latest.performance.imageOperationsMs)}
                    </dd>
                  </div>
                  <div>
                    <dt>Phase correlation</dt>
                    <dd>
                      {milliseconds(latest.performance.phaseCorrelationMs)}
                    </dd>
                  </div>
                  <div>
                    <dt>Optical flow</dt>
                    <dd>{milliseconds(latest.performance.opticalFlowMs)}</dd>
                  </div>
                </dl>
              </article>
              <article>
                <h2>Runtime & memory</h2>
                <dl>
                  <div>
                    <dt>OpenCV load</dt>
                    <dd>{milliseconds(latest.performance.openCvLoadMs)}</dd>
                  </div>
                  <div>
                    <dt>Lab wrapper wall</dt>
                    <dd>{milliseconds(latest.wallMs)}</dd>
                  </div>
                  <div>
                    <dt>Reduction kernel / load</dt>
                    <dd>
                      {latest.performance.reductionKernel} /{" "}
                      {milliseconds(latest.performance.reductionKernelLoadMs)}
                    </dd>
                  </div>
                  <div>
                    <dt>JavaScript feature time</dt>
                    <dd>{milliseconds(latest.performance.javascriptMs)}</dd>
                  </div>
                  <div>
                    <dt>WASM reduction time</dt>
                    <dd>{milliseconds(latest.performance.wasmReductionMs)}</dd>
                  </div>
                  <div>
                    <dt>JS heap start</dt>
                    <dd>
                      {latest.heapBeforeBytes === null
                        ? "Unavailable"
                        : compactBytes(latest.heapBeforeBytes)}
                    </dd>
                  </div>
                  <div>
                    <dt>JS heap peak</dt>
                    <dd>
                      {latest.heapPeakBytes === null
                        ? "Unavailable"
                        : compactBytes(latest.heapPeakBytes)}
                    </dd>
                  </div>
                  <div>
                    <dt>JS heap end</dt>
                    <dd>
                      {latest.heapAfterBytes === null
                        ? "Unavailable"
                        : compactBytes(latest.heapAfterBytes)}
                    </dd>
                  </div>
                </dl>
              </article>
              <article>
                <h2>Device & source</h2>
                <dl>
                  <div>
                    <dt>Logical processors</dt>
                    <dd>{latest.browser.logicalProcessors ?? "Unavailable"}</dd>
                  </div>
                  <div>
                    <dt>Device memory hint</dt>
                    <dd>
                      {latest.browser.deviceMemoryGb === null
                        ? "Unavailable"
                        : `${latest.browser.deviceMemoryGb} GB`}
                    </dd>
                  </div>
                  <div>
                    <dt>GPU renderer</dt>
                    <dd>{latest.browser.gpu ?? "Unavailable"}</dd>
                  </div>
                  <div>
                    <dt>MediaCapabilities smooth</dt>
                    <dd>{boolLabel(latest.diagnostics.smooth)}</dd>
                  </div>
                  <div>
                    <dt>Power-efficient hint</dt>
                    <dd>{boolLabel(latest.diagnostics.powerEfficient)}</dd>
                  </div>
                  <div>
                    <dt>ROI</dt>
                    <dd>
                      {latest.roiLabel} · x {latest.roi.x.toFixed(3)}, y{" "}
                      {latest.roi.y.toFixed(3)}, w {latest.roi.width.toFixed(3)}
                      , h {latest.roi.height.toFixed(3)}
                    </dd>
                  </div>
                  <div>
                    <dt>Cache I/O</dt>
                    <dd>
                      {milliseconds(latest.performance.cacheIoMs)} (disabled for
                      lab)
                    </dd>
                  </div>
                </dl>
              </article>
            </div>
          </section>

          <section className={styles.androidGap}>
            <div>
              <span>ANDROID-ONLY SIGNALS</span>
              <strong>Unavailable from browser APIs</strong>
            </div>
            <p>
              Actual decoder name and hardware/software selection · codec
              operating rate and priority · decode-only input/output suppression
              · analysis-thread CPU time · Java/native/PSS memory · GC runs and
              time · device thermal state · decoded timestamp error.
            </p>
          </section>
        </>
      )}

      {runs.length > 0 && (
        <section className={shared.history}>
          <header>
            <div className={shared.sectionHeading}>
              <span>05 / RUN HISTORY</span>
              <strong>Compare duration and decoder experiments</strong>
            </div>
            <button type="button" onClick={exportRuns}>
              Export JSON
            </button>
          </header>
          <div className={shared.tableWrap}>
            <table>
              <thead>
                <tr>
                  <th>Source / duration</th>
                  <th>Configuration</th>
                  <th>Total</th>
                  <th>Speed</th>
                  <th>Analysis fps</th>
                  <th>Source decoded</th>
                  <th>Worker block</th>
                  <th>Long tasks</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id}>
                    <td>
                      <strong>{run.sourceName}</strong>
                      <small>
                        {seconds(run.durationSeconds)} · {run.roiLabel}
                      </small>
                    </td>
                    <td>
                      {run.performance.decodeStrategy} ·{" "}
                      {run.performance.decoderAcceleration} ·{" "}
                      {run.performance.reductionKernel}
                    </td>
                    <td>{milliseconds(run.performance.videoElapsedMs)}</td>
                    <td>
                      {(
                        run.durationSeconds /
                        (run.performance.videoElapsedMs / 1000)
                      ).toFixed(2)}
                      ×
                    </td>
                    <td>
                      {(
                        run.performance.generatedFrames /
                        (run.performance.videoElapsedMs / 1000)
                      ).toFixed(2)}
                    </td>
                    <td>
                      {run.performance.decodedSourceFrames?.toLocaleString() ??
                        "—"}
                    </td>
                    <td>{milliseconds(run.performance.workerBlockingMs)}</td>
                    <td>
                      {run.longTaskCount} / {milliseconds(run.longTaskMs)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <footer className={shared.footer}>
        <span>
          Runs use the production visual feature extractor, with audio and cache
          disabled so video cost is isolated.
        </span>
        <span>
          Whole file is the default · NAS streaming is development-only ·
          results stay in this tab unless exported.
        </span>
      </footer>
    </main>
  );
}
