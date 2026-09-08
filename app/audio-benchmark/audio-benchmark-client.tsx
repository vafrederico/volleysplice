"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import {
  type AudioFeatureExtractionMetrics,
  extractAudioFeatures,
} from "@/lib/on-device/audio-features";
import { ANALYSIS_FPS } from "@/lib/on-device/feature-schema";
import {
  type OpenedMedia,
  openLocalMediaForAudio,
  openUrlMediaForAudio,
} from "@/lib/on-device/media";
import type { OnDeviceRuntimeVariant } from "@/lib/on-device/runtime-variants";
import type {
  AnalysisProgress,
  OnDeviceMediaInfo,
} from "@/lib/on-device/types";

import styles from "./audio-benchmark.module.css";

type NasSource = {
  name: string;
  size: number;
  modifiedAt: string;
  mediaUrl: string;
};

type SelectedSource =
  | { kind: "nas"; name: string; size: number; mediaUrl: string }
  | { kind: "local"; name: string; size: number; file: File };

type BenchmarkRun = {
  id: string;
  createdAt: string;
  sourceName: string;
  sourceSize: number;
  windowStart: number;
  checksum: number;
  longTaskCount: number;
  longTaskMs: number;
  heapDeltaBytes: number | null;
  metrics: AudioFeatureExtractionMetrics;
};

type PerformanceWithMemory = Performance & {
  memory?: { usedJSHeapSize: number };
};

const PHASES: Array<{
  key: keyof AudioFeatureExtractionMetrics;
  label: string;
  tone: string;
}> = [
  { key: "decodeWaitMs", label: "Media decode + reads", tone: "decode" },
  { key: "sampleCopyAndDspMs", label: "Copy, resample + FFT", tone: "dsp" },
  { key: "featureTransformMs", label: "Feature transforms", tone: "transform" },
  { key: "poolingMs", label: "4 fps pooling", tone: "pool" },
  { key: "resamplerInitMs", label: "Runtime initialization", tone: "init" },
  { key: "finalizeMs", label: "Final buffered frame", tone: "finalize" },
];

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
    : "The audio benchmark failed unexpectedly.";
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

function diagnosis(run: BenchmarkRun): string {
  const { metrics } = run;
  const largest = [...PHASES].sort(
    (left, right) => Number(metrics[right.key]) - Number(metrics[left.key]),
  )[0];
  if (run.longTaskMs > metrics.totalMs * 0.35) {
    return "Main-thread work is causing visible stalls. Moving DSP off-thread is the next high-leverage experiment.";
  }
  if (largest.key === "decodeWaitMs") {
    return "Media decode and source reads dominate. Compare a local file with the NAS source to separate codec cost from network I/O.";
  }
  if (largest.key === "sampleCopyAndDspMs") {
    return "Resampling and FFT work dominate. The extractor's allocation and worker strategy are the likely levers.";
  }
  if (largest.key === "featureTransformMs") {
    return "Percentile and rolling feature transforms dominate after decode; profile the long-window normalization pass.";
  }
  return "No single phase dominates. Compare repeated runs and the alternate resampler before changing the pipeline.";
}

function makeTimes(start: number, end: number): Float64Array {
  const count = Math.max(1, Math.ceil((end - start) * ANALYSIS_FPS));
  return Float64Array.from(
    { length: count },
    (_, index) => start + index / ANALYSIS_FPS,
  );
}

export function AudioBenchmarkClient() {
  const [nasSources, setNasSources] = useState<NasSource[]>([]);
  const [nasRoot, setNasRoot] = useState<string | null>(null);
  const [selected, setSelected] = useState<SelectedSource | null>(null);
  const [mediaInfo, setMediaInfo] = useState<OnDeviceMediaInfo | null>(null);
  const [runtimeVariant, setRuntimeVariant] =
    useState<OnDeviceRuntimeVariant>("linear-v1");
  const [windowStart, setWindowStart] = useState(0);
  const [windowLength, setWindowLength] = useState("120");
  const [status, setStatus] = useState<
    "idle" | "opening" | "ready" | "running" | "error"
  >("idle");
  const [progress, setProgress] = useState<AnalysisProgress | null>(null);
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [failure, setFailure] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [secureContext, setSecureContext] = useState(false);
  const runStartedAt = useRef<number | null>(null);

  useEffect(() => {
    setSecureContext(window.isSecureContext);
    fetch("/api/audio-benchmark/sources", { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) return null;
        return response.json() as Promise<{
          root: string;
          sources: NasSource[];
        }>;
      })
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
        if (source.kind === "local") return openLocalMediaForAudio(source.file);
        return openUrlMediaForAudio(
          new URL(source.mediaUrl, window.location.origin),
        );
    },
    [],
  );

  const inspect = useCallback(
    async (source: SelectedSource) => {
      setSelected(source);
      setMediaInfo(null);
      setFailure(null);
      setStatus("opening");
      try {
        const media = await openSelected(source);
        try {
          if (!media.info.hasAudio || !media.info.canDecodeAudio) {
            throw new Error(
              "This source does not contain browser-decodable audio.",
            );
          }
          setMediaInfo(media.info);
          setWindowStart(0);
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
    if (!selected || !mediaInfo || status === "running") return;
    const start = Math.max(
      0,
      Math.min(windowStart, Math.max(0, mediaInfo.duration - 0.25)),
    );
    const requestedLength =
      windowLength === "full"
        ? mediaInfo.duration - start
        : Number(windowLength);
    const end = Math.min(mediaInfo.duration, start + requestedLength);
    if (end <= start) {
      setFailure("Choose an analysis window with a positive duration.");
      return;
    }

    setFailure(null);
    setProgress(null);
    setStatus("running");
    setElapsedMs(0);
    runStartedAt.current = performance.now();
    const heapBefore = heapUsed();
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

    let media: OpenedMedia | null = null;
    try {
      media = await openSelected(selected);
      let metrics: AudioFeatureExtractionMetrics | null = null;
      const values = await extractAudioFeatures(
        media.audioTrack,
        makeTimes(start, end),
        { start, end },
        runtimeVariant,
        setProgress,
        {
          onMetrics: (value) => {
            metrics = value;
          },
        },
      );
      await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      if (!metrics)
        throw new Error("The extractor did not return timing metrics.");
      const heapAfter = heapUsed();
      const run: BenchmarkRun = {
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        sourceName: selected.name,
        sourceSize: selected.size,
        windowStart: start,
        checksum: checksum(values),
        longTaskCount: longTasks.length,
        longTaskMs: longTasks.reduce(
          (total, entry) => total + entry.duration,
          0,
        ),
        heapDeltaBytes:
          heapBefore === null || heapAfter === null
            ? null
            : heapAfter - heapBefore,
        metrics,
      };
      setRuns((current) => [run, ...current].slice(0, 12));
      setStatus("ready");
      setProgress(null);
    } catch (error) {
      setFailure(errorMessage(error));
      setStatus("error");
    } finally {
      observer?.disconnect();
      media?.input.dispose();
      if (runStartedAt.current !== null) {
        setElapsedMs(performance.now() - runStartedAt.current);
      }
      runStartedAt.current = null;
    }
  }, [
    mediaInfo,
    openSelected,
    runtimeVariant,
    selected,
    status,
    windowLength,
    windowStart,
  ]);

  const latest = runs[0] ?? null;
  const windowEnd = useMemo(() => {
    if (!mediaInfo) return 0;
    const length =
      windowLength === "full" ? mediaInfo.duration : Number(windowLength);
    return Math.min(mediaInfo.duration, Math.max(0, windowStart) + length);
  }, [mediaInfo, windowLength, windowStart]);
  const progressPercent =
    progress && progress.total > 0
      ? Math.min(100, Math.max(0, (progress.completed / progress.total) * 100))
      : 0;

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
    anchor.download = `volleysplice-audio-benchmark-${new Date().toISOString().replaceAll(":", "-")}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Brand className={styles.brand} label="Audio lab" />
        <div className={styles.headerMeta}>
          <span data-ok={secureContext}>secure context</span>
          <Link href="/video-benchmark">Video lab ↗</Link>
          <Link href="/on-device">On-device app ↗</Link>
        </div>
      </header>

      <section className={styles.hero}>
        <p>Browser-native pipeline profiler · 16 kHz · 20 FFT frames/sec</p>
        <h1>
          Where did the <em>seconds</em> go?
        </h1>
        <div className={styles.heroAside}>
          <strong>
            {latest ? `${latest.metrics.realtimeFactor.toFixed(1)}×` : "—"}
          </strong>
          <span>latest real-time speed</span>
        </div>
      </section>

      <section className={styles.workbench}>
        <div className={styles.sourcePanel}>
          <div className={styles.sectionHeading}>
            <span>01 / SOURCE</span>
            <strong>Reproduce the slow clip</strong>
          </div>
          <label className={styles.filePicker}>
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
            <label className={styles.nasPicker}>
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
            className={styles.sourceReadout}
            data-active={Boolean(mediaInfo)}
          >
            <span>
              {status === "opening"
                ? "Probing media…"
                : (selected?.name ?? "No source selected")}
            </span>
            <strong>{selected ? compactBytes(selected.size) : "—"}</strong>
          </div>
        </div>

        <div className={styles.controlPanel}>
          <div className={styles.sectionHeading}>
            <span>02 / RUN</span>
            <strong>Isolate a representative window</strong>
          </div>
          <div className={styles.controls}>
            <label>
              <span>Start (seconds)</span>
              <input
                type="number"
                min="0"
                max={Math.max(0, (mediaInfo?.duration ?? 0) - 0.25)}
                step="1"
                value={windowStart}
                disabled={!mediaInfo || status === "running"}
                onChange={(event) => setWindowStart(Number(event.target.value))}
              />
            </label>
            <label>
              <span>Window</span>
              <select
                value={windowLength}
                disabled={!mediaInfo || status === "running"}
                onChange={(event) => setWindowLength(event.target.value)}
              >
                <option value="30">30 seconds</option>
                <option value="120">2 minutes</option>
                <option value="300">5 minutes</option>
                <option value="full">Full remainder</option>
              </select>
            </label>
            <label>
              <span>Resampler</span>
              <select
                value={runtimeVariant}
                disabled={!mediaInfo || status === "running"}
                onChange={(event) =>
                  setRuntimeVariant(
                    event.target.value as OnDeviceRuntimeVariant,
                  )
                }
              >
                <option value="linear-v1">Linear JS</option>
                <option value="libswresample-wasm-v1">
                  libswresample WASM
                </option>
              </select>
            </label>
          </div>
          <button
            type="button"
            className={styles.runButton}
            disabled={!mediaInfo || status === "running"}
            onClick={() => void runBenchmark()}
          >
            {status === "running"
              ? `Running · ${(elapsedMs / 1000).toFixed(1)}s`
              : "Run audio benchmark"}
          </button>
        </div>
      </section>

      {failure && <p className={styles.error}>{failure}</p>}

      {mediaInfo && (
        <section className={styles.mediaStrip}>
          <div>
            <span>Duration</span>
            <strong>{seconds(mediaInfo.duration)}</strong>
          </div>
          <div>
            <span>Audio</span>
            <strong>{mediaInfo.audioCodec?.toUpperCase() ?? "—"}</strong>
          </div>
          <div>
            <span>Input rate</span>
            <strong>{mediaInfo.sampleRate?.toLocaleString() ?? "—"} Hz</strong>
          </div>
          <div>
            <span>Channels</span>
            <strong>{mediaInfo.channels ?? "—"}</strong>
          </div>
          <div>
            <span>Benchmark</span>
            <strong>{seconds(windowEnd - Math.max(0, windowStart))}</strong>
          </div>
          <div>
            <span>Output rows</span>
            <strong>
              {Math.ceil(
                (windowEnd - Math.max(0, windowStart)) * ANALYSIS_FPS,
              ).toLocaleString()}
            </strong>
          </div>
        </section>
      )}

      {status === "running" && (
        <section className={styles.liveTrace}>
          <div>
            <span>LIVE / AUDIO</span>
            <strong>{progress?.detail ?? "Initializing decoder"}</strong>
          </div>
          <div className={styles.progressTrack}>
            <i style={{ width: `${progressPercent}%` }} />
          </div>
          <b>
            {progress
              ? `${seconds(progress.completed)} / ${seconds(progress.total)}`
              : "waiting"}
          </b>
        </section>
      )}

      {latest && (
        <section className={styles.results}>
          <header>
            <div className={styles.sectionHeading}>
              <span>03 / LATEST RESULT</span>
              <strong>{latest.sourceName}</strong>
            </div>
            <p>{diagnosis(latest)}</p>
          </header>
          <div className={styles.resultHero}>
            <div>
              <strong>{milliseconds(latest.metrics.totalMs)}</strong>
              <span>wall time</span>
            </div>
            <div>
              <strong>{latest.metrics.realtimeFactor.toFixed(1)}×</strong>
              <span>real-time speed</span>
            </div>
            <div>
              <strong>{latest.longTaskCount}</strong>
              <span>main-thread long tasks</span>
            </div>
            <div>
              <strong>{milliseconds(latest.longTaskMs)}</strong>
              <span>long-task time</span>
            </div>
          </div>
          <div className={styles.breakdown}>
            {PHASES.map((phase) => {
              const value = Number(latest.metrics[phase.key]);
              const share =
                latest.metrics.totalMs > 0
                  ? (value / latest.metrics.totalMs) * 100
                  : 0;
              return (
                <div key={phase.key}>
                  <span>{phase.label}</span>
                  <div>
                    <i
                      data-tone={phase.tone}
                      style={{ width: `${Math.max(0.5, share)}%` }}
                    />
                  </div>
                  <b>{milliseconds(value)}</b>
                  <small>{share.toFixed(1)}%</small>
                </div>
              );
            })}
          </div>
          <div className={styles.counters}>
            <span>
              <b>{latest.metrics.decodedChunks.toLocaleString()}</b> decoded
              chunks
            </span>
            <span>
              <b>{latest.metrics.decodedSourceFrames.toLocaleString()}</b>{" "}
              source frames
            </span>
            <span>
              <b>{latest.metrics.featureFrames.toLocaleString()}</b> FFT frames
            </span>
            <span>
              <b>{latest.metrics.outputRows.toLocaleString()}</b> pooled rows
            </span>
            <span>
              <b>
                {latest.heapDeltaBytes === null
                  ? "—"
                  : compactBytes(Math.abs(latest.heapDeltaBytes))}
              </b>{" "}
              heap{" "}
              {latest.heapDeltaBytes !== null && latest.heapDeltaBytes < 0
                ? "released"
                : "growth"}
            </span>
          </div>
        </section>
      )}

      {runs.length > 0 && (
        <section className={styles.history}>
          <header>
            <div className={styles.sectionHeading}>
              <span>04 / RUN HISTORY</span>
              <strong>Compare warm, cold and runtime runs</strong>
            </div>
            <button type="button" onClick={exportRuns}>
              Export JSON
            </button>
          </header>
          <div className={styles.tableWrap}>
            <table>
              <thead>
                <tr>
                  <th>Source / window</th>
                  <th>Runtime</th>
                  <th>Total</th>
                  <th>Speed</th>
                  <th>Decode</th>
                  <th>DSP + FFT</th>
                  <th>Transforms</th>
                  <th>Long tasks</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id}>
                    <td>
                      <strong>{run.sourceName}</strong>
                      <small>
                        {seconds(run.windowStart)} →{" "}
                        {seconds(run.windowStart + run.metrics.windowSeconds)}
                      </small>
                    </td>
                    <td>
                      {run.metrics.runtimeVariant === "linear-v1"
                        ? "Linear JS"
                        : "libswresample"}
                    </td>
                    <td>{milliseconds(run.metrics.totalMs)}</td>
                    <td>{run.metrics.realtimeFactor.toFixed(1)}×</td>
                    <td>{milliseconds(run.metrics.decodeWaitMs)}</td>
                    <td>{milliseconds(run.metrics.sampleCopyAndDspMs)}</td>
                    <td>{milliseconds(run.metrics.featureTransformMs)}</td>
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

      <footer className={styles.footer}>
        <span>
          Measurements use the same extractor and feature schema as the
          on-device web app.
        </span>
        <span>
          NAS streaming is development-only · results stay in this tab unless
          exported.
        </span>
      </footer>
    </main>
  );
}
