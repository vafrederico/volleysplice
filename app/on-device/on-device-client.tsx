"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import { buildEditList, formatTime } from "@/lib/edit-list";
import {
  downloadEditDecisionList,
  exportRawQualityReel,
  type ExportProgress,
} from "@/lib/on-device/export";
import { openLocalMedia, type OpenedMedia } from "@/lib/on-device/media";
import { analyzeOpenedMedia } from "@/lib/on-device/pipeline";
import { clampRoi, fullFrameRoi, inferRoiProfile } from "@/lib/on-device/roi";
import type {
  AnalysisProgress,
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
};

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
  });
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

  useEffect(() => {
    let active = true;
    queueMicrotask(() => {
      if (!active) return;
      setCompatibility({
        checked: true,
        secureContext: window.isSecureContext,
        decode: "VideoDecoder" in window && "AudioDecoder" in window,
        encode: "VideoEncoder" in window && "AudioEncoder" in window,
        directDisk: "showSaveFilePicker" in window,
      });
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
    setSelectedId(null);
    setError(null);
    setPreviewWarning(false);
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
    } catch (cause) {
      setHasOpenedMedia(false);
      setError(cause instanceof Error ? cause.message : String(cause));
      setWorkState("error");
    }
  }

  async function runAnalysis() {
    if (!openedMedia.current || !info) return;
    setError(null);
    setAnalysis(null);
    setSelectedId(null);
    setWorkState("analyzing");
    try {
      const result = await analyzeOpenedMedia(
        openedMedia.current,
        roi,
        featurePath,
        setAnalysisProgress,
      );
      setAnalysis(result);
      setSelectedId(result.intervals[0]?.id ?? null);
      setWorkState("done");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setWorkState("error");
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
          <span data-ok={compatibility.decode}>Decode</span>
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
