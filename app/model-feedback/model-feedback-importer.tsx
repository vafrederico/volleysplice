"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import {
  RallyTimeline,
  type TimelineInterval,
  type TimelineTrack,
} from "@/components/rally-timeline";
import {
  ModelFeedbackValidationError,
  type ParsedModelFeedback,
  parseModelFeedbackText,
} from "@/lib/model-feedback";

import styles from "./model-feedback.module.css";

type ImportSummary = {
  id: string;
  importedAt: string;
  producer: "production-web" | "android";
  projectId: string;
  sourceName: string;
  duration: number;
  featureRows: number | null;
  correctedRanges: number;
  sourceLinked: boolean;
  mediaUrl: string | null;
};

type SourceMode = "local" | "server";
type LinkStatus = { tone: "good" | "warn"; message: string } | null;

const SAMPLE_BYTES = 1024 * 1024;
const CHART_WIDTH = 1000;
const CHART_HEIGHT = 190;

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

function clock(seconds: number): string {
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = safe % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`
    : `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function hex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes), (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
}

async function sampledFileFingerprint(file: File): Promise<string | null> {
  if (typeof crypto === "undefined" || !crypto.subtle) return null;
  const size = new ArrayBuffer(8);
  new DataView(size).setBigUint64(0, BigInt(file.size), true);
  const pieces =
    file.size <= SAMPLE_BYTES * 2
      ? [size, await file.arrayBuffer()]
      : [
          size,
          await file.slice(0, SAMPLE_BYTES).arrayBuffer(),
          await file.slice(file.size - SAMPLE_BYTES).arrayBuffer(),
        ];
  const length = pieces.reduce((total, piece) => total + piece.byteLength, 0);
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const piece of pieces) {
    bytes.set(new Uint8Array(piece), offset);
    offset += piece.byteLength;
  }
  return `sampled-sha256-v1:${hex(await crypto.subtle.digest("SHA-256", bytes))}`;
}

function sampleIndices(length: number, maximum = 900): number[] {
  if (length <= maximum) return Array.from({ length }, (_, index) => index);
  return Array.from({ length: maximum }, (_, index) =>
    Math.min(length - 1, Math.round((index / (maximum - 1)) * (length - 1))),
  );
}

function chartPath(
  times: Float64Array,
  values: Float32Array,
  duration: number,
  minimum: number,
  maximum: number,
): string {
  if (times.length === 0 || values.length !== times.length) return "";
  const span = Math.max(1e-9, maximum - minimum);
  return sampleIndices(times.length)
    .map((index, point) => {
      const x = Math.max(
        0,
        Math.min(CHART_WIDTH, (times[index] / duration) * CHART_WIDTH),
      );
      const y =
        CHART_HEIGHT -
        Math.max(0, Math.min(1, (values[index] - minimum) / span)) *
          CHART_HEIGHT;
      return `${point === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function featureColumn(
  bundle: ParsedModelFeedback,
  column: number,
): Float32Array {
  if (!bundle.features) return new Float32Array(0);
  const values = new Float32Array(bundle.features.rows);
  for (let row = 0; row < bundle.features.rows; row += 1) {
    values[row] =
      bundle.features.values[row * bundle.features.columns + column];
  }
  return values;
}

export function ModelFeedbackImporter({
  initialImportId,
}: {
  initialImportId: string | null;
}) {
  const [bundle, setBundle] = useState<ParsedModelFeedback | null>(null);
  const [bundleText, setBundleText] = useState<string | null>(null);
  const [bundleName, setBundleName] = useState<string | null>(null);
  const [imports, setImports] = useState<ImportSummary[]>([]);
  const [sourceMode, setSourceMode] = useState<SourceMode>("local");
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [sourceLocal, setSourceLocal] = useState(false);
  const [serverPath, setServerPath] = useState("");
  const [linkStatus, setLinkStatus] = useState<LinkStatus>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [selectedImportId, setSelectedImportId] = useState<string | null>(
    initialImportId,
  );
  const [selectedFeature, setSelectedFeature] = useState(0);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [selectedEvidence, setSelectedEvidence] = useState<{
    trackId: string;
    intervalId: string;
  } | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const objectUrl = useRef<string | null>(null);
  const initialized = useRef(false);

  function releaseLocalSource() {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = null;
    setSourceLocal(false);
  }

  async function refreshImports() {
    try {
      const response = await fetch("/api/model-feedback", {
        cache: "no-store",
      });
      if (!response.ok) return;
      const payload = (await response.json()) as { imports?: ImportSummary[] };
      setImports(Array.isArray(payload.imports) ? payload.imports : []);
    } catch {
      // Local file inspection remains useful when the persistent store is unavailable.
    }
  }

  async function acceptBundle(
    text: string,
    name: string,
    linkedSourceUrl: string | null = null,
  ) {
    try {
      const parsed = parseModelFeedbackText(text);
      releaseLocalSource();
      setBundle(parsed);
      setBundleText(text);
      setBundleName(name);
      setSelectedFeature(0);
      setPlaybackTime(parsed.source.gameWindow.start);
      setSelectedEvidence(null);
      setSelectedImportId(null);
      setServerPath("");
      setSourceUrl(linkedSourceUrl);
      setSourceMode(linkedSourceUrl ? "server" : "local");
      setLinkStatus(
        linkedSourceUrl
          ? {
              tone: "good",
              message: "Loaded the permanently linked server source.",
            }
          : null,
      );
      setError(null);
    } catch (cause) {
      setError(
        cause instanceof ModelFeedbackValidationError
          ? cause.message
          : "The selected model-feedback file could not be parsed.",
      );
    }
  }

  async function loadStored(id: string) {
    if (!id) return;
    setLoadingId(id);
    setError(null);
    try {
      const response = await fetch(
        `/api/model-feedback/${encodeURIComponent(id)}`,
        {
          cache: "no-store",
        },
      );
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as {
          error?: string;
        } | null;
        throw new Error(
          payload?.error ?? "The stored import could not be loaded.",
        );
      }
      const text = await response.text();
      await acceptBundle(
        text,
        `${id}.model-feedback.json`,
        response.headers.get("X-VolleyCut-Source-Url"),
      );
      const encodedSourcePath = response.headers.get("X-VolleyCut-Source-Path");
      if (encodedSourcePath) {
        try {
          setServerPath(decodeURIComponent(encodedSourcePath));
        } catch {
          setServerPath("");
        }
      } else {
        setServerPath("");
        setLinkStatus({
          tone: "warn",
          message: "Loaded saved feedback JSON. No source file is linked.",
        });
      }
      setSelectedImportId(id);
      window.history.replaceState(
        null,
        "",
        `/model-feedback?import=${encodeURIComponent(id)}`,
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoadingId(null);
    }
  }

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    void refreshImports();
    if (initialImportId) void loadStored(initialImportId);
  });

  useEffect(() => {
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  async function chooseLocalSource(file: File | null) {
    if (!file || !bundle) return;
    releaseLocalSource();
    setLinkStatus(null);
    setError(null);
    if (file.size !== bundle.source.file.sizeBytes) {
      setError(
        `That file is ${file.size} bytes; the bundle expects ${bundle.source.file.sizeBytes} bytes.`,
      );
      return;
    }
    try {
      const actual = bundle.source.file.sampledFingerprint
        ? await sampledFileFingerprint(file)
        : null;
      if (actual && actual !== bundle.source.file.sampledFingerprint) {
        setError(
          "The selected local file fingerprint does not match the feedback bundle.",
        );
        return;
      }
      const url = URL.createObjectURL(file);
      objectUrl.current = url;
      setSourceUrl(url);
      setSourceLocal(true);
      setLinkStatus({
        tone:
          actual || !bundle.source.file.sampledFingerprint ? "good" : "warn",
        message: actual
          ? "Local source size and fingerprint match. This link lasts only for this browser session."
          : bundle.source.file.sampledFingerprint
            ? "Source size matches. This browser context cannot verify SHA-256; the link remains session-only."
            : "Source size matches. This bundle has no fingerprint; the link remains session-only.",
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function saveFeedback() {
    if (!bundleText || !bundle) return;
    setSaving(true);
    setError(null);
    setLinkStatus(null);
    try {
      const form = new FormData();
      form.set(
        "bundle",
        new File([bundleText], bundleName ?? "model-feedback.json", {
          type: "application/json",
        }),
      );
      const requestedSourcePath =
        sourceMode === "server" ? serverPath.trim() : "";
      if (requestedSourcePath) form.set("sourcePath", requestedSourcePath);
      const response = await fetch("/api/model-feedback", {
        method: "POST",
        body: form,
      });
      const payload = (await response.json().catch(() => null)) as {
        error?: string;
        import?: ImportSummary;
      } | null;
      if (!response.ok || !payload?.import) {
        throw new Error(
          payload?.error ?? "The permanent import could not be saved.",
        );
      }
      if (payload.import.mediaUrl) {
        releaseLocalSource();
        setSourceUrl(payload.import.mediaUrl);
        setSourceLocal(false);
        setSourceMode("server");
        setLinkStatus({
          tone: "good",
          message:
            "Feedback JSON and verified server source link saved to NAS.",
        });
      } else {
        if (!sourceLocal) setSourceUrl(null);
        setLinkStatus({
          tone: "good",
          message: "Feedback JSON saved to NAS without a linked source.",
        });
      }
      setSelectedImportId(payload.import.id);
      window.history.replaceState(
        null,
        "",
        `/model-feedback?import=${encodeURIComponent(payload.import.id)}`,
      );
      await refreshImports();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setSaving(false);
    }
  }

  function seek(seconds: number) {
    setPlaybackTime(seconds);
    if (videoRef.current) videoRef.current.currentTime = seconds;
  }

  const featureValues = useMemo(
    () =>
      bundle ? featureColumn(bundle, selectedFeature) : new Float32Array(0),
    [bundle, selectedFeature],
  );
  const featureBounds = useMemo(() => {
    if (featureValues.length === 0) return { minimum: 0, maximum: 1 };
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    for (const value of featureValues) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    return { minimum, maximum: maximum === minimum ? minimum + 1 : maximum };
  }, [featureValues]);

  const labels = bundle?.corrections.labels;
  const producerLabel =
    bundle?.producer === "android" ? "Android app" : "Production web app";
  const evidenceTracks = useMemo<TimelineTrack[]>(() => {
    if (!bundle) return [];
    return [
      {
        id: "initial-inference",
        label: "Initial inference",
        detail: `${bundle.initialInference.ranges.length} model ranges`,
        intervals: bundle.initialInference.ranges.map((range) => ({
          id: range.id,
          start: range.start,
          end: range.end,
          confidence: range.confidence,
          tone: "model" as const,
          title: `${range.id} · initial model · ${clock(range.start)}–${clock(range.end)} · ${Math.round(range.confidence * 100)}%`,
        })),
      },
      {
        id: "corrected-labels",
        label: "Corrected labels",
        detail: `${bundle.corrections.correctedRanges.length} reviewed ranges`,
        intervals: bundle.corrections.correctedRanges.flatMap(
          (range): TimelineInterval[] => {
            const title = `${range.id} · ${range.included ? "included" : "excluded"} correction · ${clock(range.coreStart)}–${clock(range.coreEnd)}`;
            if (!range.included) {
              return [
                {
                  id: `${range.id}-excluded`,
                  selectionId: range.id,
                  start: range.coreStart,
                  end: range.coreEnd,
                  confidence: range.confidence,
                  tone: "negative" as const,
                  title,
                },
              ];
            }
            return [
              {
                id: `${range.id}-before`,
                selectionId: range.id,
                start: range.keepStart,
                end: range.coreStart,
                confidence: range.confidence,
                tone: "gold" as const,
                paddingOrigin: "before" as const,
                title: `${title} · before padding`,
              },
              {
                id: `${range.id}-core`,
                selectionId: range.id,
                start: range.coreStart,
                end: range.coreEnd,
                confidence: range.confidence,
                tone: "gold" as const,
                title,
              },
              {
                id: `${range.id}-after`,
                selectionId: range.id,
                start: range.coreEnd,
                end: range.keepEnd,
                confidence: range.confidence,
                tone: "gold" as const,
                paddingOrigin: "after" as const,
                title: `${title} · after padding`,
              },
            ].filter((interval) => interval.end > interval.start);
          },
        ),
      },
      {
        id: "ignored-time",
        label: "Ignored time",
        detail: `${bundle.corrections.ignoredIntervals.length} excluded spans`,
        intervals: bundle.corrections.ignoredIntervals.map((range) => ({
          id: range.id,
          start: range.start,
          end: range.end,
          tone: "ignored" as const,
          title: `${range.id} · ${range.reason} · ${clock(range.start)}–${clock(range.end)}`,
        })),
      },
      {
        id: "final-export",
        label: "Final export",
        detail: `${bundle.finalExportIntervals.length} output intervals`,
        intervals: bundle.finalExportIntervals.map((range, index) => ({
          id: `export-${index + 1}`,
          start: range.start,
          end: range.end,
          tone: "model-match" as const,
          title: `Export ${index + 1} · ${clock(range.start)}–${clock(range.end)} · ${range.cutIds.join(", ")}`,
        })),
      },
    ];
  }, [bundle]);

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Brand className={styles.brand} label="LAB" priority />
        <nav>
          <Link href="/">Dataset review</Link>
          <Link href="/edit">Cut editor</Link>
          <Link href="/label">Labeling station</Link>
        </nav>
      </header>

      <section
        className={styles.libraryBar}
        aria-label="Saved feedback library"
      >
        <div>
          <p className={styles.kicker}>SAVED FEEDBACK</p>
          <strong>NAS feedback library</strong>
        </div>
        <label className={styles.savedPicker}>
          <span>Select a saved feedback bundle</span>
          <select
            value={loadingId ?? selectedImportId ?? ""}
            disabled={loadingId !== null || imports.length === 0}
            onChange={(event) => void loadStored(event.target.value)}
          >
            <option value="">
              {imports.length
                ? "Choose a saved import"
                : "No saved imports yet"}
            </option>
            {imports.map((item) => (
              <option key={item.id} value={item.id}>
                {item.sourceName} ·{" "}
                {item.producer === "android" ? "Android" : "Web"} ·{" "}
                {item.sourceLinked ? "linked source" : "JSON only"} ·{" "}
                {new Date(item.importedAt).toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        <span className={styles.libraryCount}>
          {imports.length} saved {imports.length === 1 ? "bundle" : "bundles"}
        </span>
      </section>

      <section className={styles.hero}>
        <p>MODEL ITERATION INTAKE</p>
        <h1>
          Bring field feedback
          <br />
          <em>back into the lab.</em>
        </h1>
        <span>
          Import the shared JSON emitted by the production web or Android
          editor. Feature matrices, probability traces, corrections, ignored
          time, and final export decisions stay aligned to the original source
          clock.
        </span>
      </section>

      <section className={styles.intake}>
        <div>
          <p className={styles.kicker}>1 / FEEDBACK BUNDLE</p>
          <h2>{bundleName ?? "Open a model-feedback JSON file"}</h2>
          <p>
            Schema v1 files from both product runtimes use the same validated
            import path.
          </p>
        </div>
        <label className={styles.fileButton}>
          {bundle ? "Choose another JSON" : "Choose feedback JSON"}
          <input
            type="file"
            accept="application/json,.json"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file)
                void file.text().then((text) => acceptBundle(text, file.name));
              if (file)
                window.history.replaceState(null, "", "/model-feedback");
              event.currentTarget.value = "";
            }}
          />
        </label>
      </section>

      {error && (
        <div className={styles.error} role="alert">
          <strong>Import stopped.</strong> {error}
        </div>
      )}

      {bundle && (
        <>
          <section className={styles.sourceCard}>
            <div className={styles.sourceHeading}>
              <div>
                <p className={styles.kicker}>2 / SAVE &amp; SOURCE</p>
                <h2>{bundle.source.file.name}</h2>
                <span>
                  {compactBytes(bundle.source.file.sizeBytes)} ·{" "}
                  {clock(bundle.source.media.duration)} · source video is not
                  embedded
                </span>
              </div>
              <div className={styles.modeTabs}>
                <button
                  type="button"
                  data-active={sourceMode === "local"}
                  onClick={() => setSourceMode("local")}
                >
                  Local only
                </button>
                <button
                  type="button"
                  data-active={sourceMode === "server"}
                  onClick={() => setSourceMode("server")}
                >
                  Server path
                </button>
              </div>
            </div>
            {sourceMode === "local" ? (
              <div className={styles.linkForm}>
                <div>
                  <strong>Choose the source on this device</strong>
                  <span>
                    The browser keeps an object URL only until this tab closes
                    or reloads.
                  </span>
                </div>
                <label className={styles.fileButton}>
                  {sourceLocal ? "Replace local source" : "Link local source"}
                  <input
                    type="file"
                    accept="video/*,.mkv,.webm,.mov,.mp4"
                    onChange={(event) =>
                      void chooseLocalSource(event.target.files?.[0] ?? null)
                    }
                  />
                </label>
              </div>
            ) : (
              <div className={styles.linkForm}>
                <label className={styles.pathField}>
                  <span>Optional absolute path visible to the dev server</span>
                  <input
                    value={serverPath}
                    placeholder="/mnt/freenas/volleycut/raw/match.mp4"
                    onChange={(event) => setServerPath(event.target.value)}
                  />
                </label>
              </div>
            )}
            <div className={styles.saveRow}>
              <div>
                <strong>Save feedback JSON to NAS</strong>
                <span>
                  {sourceMode === "server" && serverPath.trim()
                    ? "The source path will be verified and linked to this saved bundle."
                    : "No permanent source link is required; you can attach one later by saving again."}
                </span>
              </div>
              <button
                type="button"
                className={styles.saveButton}
                disabled={saving}
                onClick={() => void saveFeedback()}
              >
                {saving
                  ? "Saving…"
                  : sourceMode === "server" && serverPath.trim()
                    ? "Verify source & save to NAS"
                    : "Save feedback JSON to NAS"}
              </button>
            </div>
            {linkStatus && (
              <p className={styles.linkStatus} data-tone={linkStatus.tone}>
                {linkStatus.message}
              </p>
            )}
          </section>

          <section className={styles.summary}>
            <article>
              <span>PRODUCER</span>
              <strong>{producerLabel}</strong>
              <small>{bundle.source.runtimeVariant}</small>
            </article>
            <article>
              <span>BASE FEATURES</span>
              <strong>
                {bundle.features
                  ? bundle.features.rows.toLocaleString()
                  : "Unavailable"}
              </strong>
              <small>
                {bundle.features
                  ? `${bundle.features.columns} columns · ${bundle.features.analysisFps} fps`
                  : "See bundle warning"}
              </small>
            </article>
            <article>
              <span>CORRECTED RANGES</span>
              <strong>{bundle.corrections.correctedRanges.length}</strong>
              <small>
                {labels?.falsePositives.length ?? 0} FP ·{" "}
                {labels?.falseNegatives.length ?? 0} FN
              </small>
            </article>
            <article>
              <span>FINAL EXPORTS</span>
              <strong>{bundle.finalExportIntervals.length}</strong>
              <small>
                {bundle.corrections.beforePaddingSeconds}s before ·{" "}
                {bundle.corrections.afterPaddingSeconds}s after
              </small>
            </article>
          </section>

          {sourceUrl && (
            <section className={styles.videoCard}>
              <video
                ref={videoRef}
                src={sourceUrl}
                controls
                preload="metadata"
                playsInline
                onLoadedMetadata={(event) => {
                  event.currentTarget.currentTime = Math.min(
                    playbackTime,
                    event.currentTarget.duration || playbackTime,
                  );
                }}
                onTimeUpdate={(event) =>
                  setPlaybackTime(event.currentTarget.currentTime)
                }
              />
              <div>
                <p className={styles.kicker}>SOURCE PREVIEW</p>
                <strong>
                  {sourceLocal
                    ? "Browser-local source"
                    : "Permanent server link"}
                </strong>
                <span>
                  Timeline rows below seek this source without changing bundle
                  timestamps.
                </span>
              </div>
            </section>
          )}

          <section className={styles.evidence}>
            <header>
              <div>
                <p className={styles.kicker}>3 / RANGE EVIDENCE</p>
                <h2>Predictions, corrections, and exports</h2>
              </div>
              <span>
                {clock(bundle.source.gameWindow.start)}–
                {clock(bundle.source.gameWindow.end)} analyzed
              </span>
            </header>
            <div className={styles.timelineLegend}>
              <span data-tone="initial">Initial model</span>
              <span data-tone="corrected">Included correction</span>
              <span data-tone="excluded">Excluded correction</span>
              <span data-tone="ignored">Ignored</span>
              <span data-tone="export">Final export</span>
              <span data-tone="before-padding">Before padding</span>
              <span data-tone="after-padding">After padding</span>
              <span data-tone="playhead">Current playhead</span>
            </div>
            <div className={styles.editorTimeline}>
              <RallyTimeline
                duration={bundle.source.media.duration}
                currentTime={playbackTime}
                tracks={evidenceTracks}
                selectedTrackId={selectedEvidence?.trackId}
                selectedIntervalId={selectedEvidence?.intervalId}
                ariaLabel="Production-editor style timeline for imported range evidence"
                onSeek={(time, trackId, intervalId) => {
                  seek(time);
                  if (intervalId) setSelectedEvidence({ trackId, intervalId });
                }}
              />
            </div>
          </section>

          <section className={styles.traces}>
            <header>
              <div>
                <p className={styles.kicker}>4 / MODEL TRACES</p>
                <h2>Source-aligned probability heads</h2>
              </div>
              <span>
                {bundle.initialInference.timestamps.length.toLocaleString()}{" "}
                samples · {bundle.initialInference.probabilityModelId}
              </span>
            </header>
            {bundle.initialInference.timestamps.length > 0 ? (
              <div className={styles.chart}>
                <svg
                  viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
                  preserveAspectRatio="none"
                  role="img"
                  aria-label="Rally, serve, and dead-state probability traces"
                >
                  <path
                    data-trace="rally"
                    d={chartPath(
                      bundle.initialInference.timestamps,
                      bundle.initialInference.rallyProbabilities,
                      bundle.source.media.duration,
                      0,
                      1,
                    )}
                  />
                  <path
                    data-trace="serve"
                    d={chartPath(
                      bundle.initialInference.timestamps,
                      bundle.initialInference.serveProbabilities,
                      bundle.source.media.duration,
                      0,
                      1,
                    )}
                  />
                  <path
                    data-trace="dead"
                    d={chartPath(
                      bundle.initialInference.timestamps,
                      bundle.initialInference.deadStateProbabilities,
                      bundle.source.media.duration,
                      0,
                      1,
                    )}
                  />
                </svg>
                <div className={styles.traceLegend}>
                  <span data-trace="rally">Rally</span>
                  <span data-trace="serve">Serve</span>
                  <span data-trace="dead">Dead state</span>
                </div>
              </div>
            ) : (
              <p className={styles.empty}>
                This bundle does not contain probability samples.
              </p>
            )}
          </section>

          <section className={styles.traces}>
            <header>
              <div>
                <p className={styles.kicker}>5 / BASE FEATURES</p>
                <h2>Decoded audiovisual feature matrix</h2>
              </div>
              {bundle.features && (
                <label className={styles.featurePicker}>
                  <span>Feature column</span>
                  <select
                    value={selectedFeature}
                    onChange={(event) =>
                      setSelectedFeature(Number(event.target.value))
                    }
                  >
                    {bundle.features.names.map((name, index) => (
                      <option key={name} value={index}>
                        {index + 1}. {name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </header>
            {bundle.features ? (
              <div className={styles.chart}>
                <svg
                  viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
                  preserveAspectRatio="none"
                  role="img"
                  aria-label={`Feature trace for ${bundle.features.names[selectedFeature]}`}
                >
                  <path
                    data-trace="feature"
                    d={chartPath(
                      bundle.features.timestamps,
                      featureValues,
                      bundle.source.media.duration,
                      featureBounds.minimum,
                      featureBounds.maximum,
                    )}
                  />
                </svg>
                <div className={styles.featureScale}>
                  <span>{featureBounds.maximum.toFixed(3)}</span>
                  <strong>{bundle.features.names[selectedFeature]}</strong>
                  <span>{featureBounds.minimum.toFixed(3)}</span>
                </div>
              </div>
            ) : (
              <p className={styles.empty}>
                Features were not retained for this analysis. Probability traces
                and correction labels are still available.
              </p>
            )}
          </section>

          <section className={styles.labels}>
            <header>
              <div>
                <p className={styles.kicker}>6 / TRAINING LABELS</p>
                <h2>Explicit correction outcomes</h2>
              </div>
            </header>
            <div className={styles.labelGrid}>
              {labels &&
                (
                  [
                    ["False positives", labels.falsePositives, "fp"],
                    ["False negatives", labels.falseNegatives, "fn"],
                    [
                      "Confirmed model ranges",
                      labels.confirmedModelRanges,
                      "confirmed",
                    ],
                    [
                      "Discarded manual ranges",
                      labels.discardedManualRanges,
                      "discarded",
                    ],
                  ] as const
                ).map(([label, ranges, tone]) => (
                  <article key={label} data-tone={tone}>
                    <header>
                      <strong>{label}</strong>
                      <span>{ranges.length}</span>
                    </header>
                    {ranges.length ? (
                      ranges.map((range) => (
                        <button
                          key={range.id}
                          type="button"
                          onClick={() => seek(range.start)}
                        >
                          <strong>{range.id}</strong>
                          <span>
                            {clock(range.start)}–{clock(range.end)}
                          </span>
                        </button>
                      ))
                    ) : (
                      <p>None</p>
                    )}
                  </article>
                ))}
            </div>
          </section>

          <section className={styles.metadata}>
            <details>
              <summary>Import metadata and feature schema</summary>
              <dl>
                <div>
                  <dt>Project</dt>
                  <dd>{bundle.source.projectId}</dd>
                </div>
                <div>
                  <dt>Analysis</dt>
                  <dd>{bundle.source.analysisId}</dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd>{bundle.initialInference.modelId}</dd>
                </div>
                <div>
                  <dt>Generated</dt>
                  <dd>{new Date(bundle.generatedAt).toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Media</dt>
                  <dd>
                    {bundle.source.media.width}×{bundle.source.media.height} ·{" "}
                    {bundle.source.media.videoCodec}
                  </dd>
                </div>
                <div>
                  <dt>ROI</dt>
                  <dd>
                    {Object.values(bundle.source.featureRoi)
                      .map((value) => value.toFixed(3))
                      .join(" · ")}
                  </dd>
                </div>
                <div className={styles.wide}>
                  <dt>Feature names</dt>
                  <dd>{bundle.features?.names.join(", ") ?? "Unavailable"}</dd>
                </div>
                <div className={styles.wide}>
                  <dt>Fingerprint</dt>
                  <dd>
                    {bundle.source.file.sampledFingerprint ?? "Unavailable"}
                  </dd>
                </div>
              </dl>
              {bundle.warnings.map((warning) => (
                <p key={warning} className={styles.warning}>
                  ⚑ {warning}
                </p>
              ))}
            </details>
          </section>
        </>
      )}
    </main>
  );
}
