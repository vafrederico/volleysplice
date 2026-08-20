"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import { formatTime } from "@/lib/edit-list";

import base from "../side-switch-review/side-switch-review.module.css";
import styles from "./serving-side-results.module.css";
import type {
  ServingSideResult,
  ServingSideResultRecording,
  ServingSideResultsData,
} from "./types";

type OutcomeFilter =
  | "all"
  | "wrong"
  | "correct"
  | "near-as-far"
  | "far-as-near";

type Props = {
  data: ServingSideResultsData | null;
  evaluationPath: string;
  initialRecordingId: string | null;
  initialOutcome: OutcomeFilter;
  loadError?: string;
};

const OUTCOME_FILTERS: Array<{ value: OutcomeFilter; label: string }> = [
  { value: "wrong", label: "Mistakes" },
  { value: "all", label: "All" },
  { value: "correct", label: "Correct" },
  { value: "near-as-far", label: "Near recall misses" },
  { value: "far-as-near", label: "Far recall misses" },
];

const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 2];
const FEATURE_LABELS: Array<[string, string]> = [
  ["pixelMotionMargin", "Whole-frame motion margin"],
  ["paletteChangeMargin", "Whole-frame palette margin"],
  ["baselinePixelMotionMargin", "Baseline motion margin"],
  ["baselinePaletteChangeMargin", "Baseline palette margin"],
  ["hogAreaChangeMargin", "HOG area-change margin"],
  ["hogCountChangeMargin", "HOG count-change margin"],
  ["baselineHogAreaChangeMargin", "Baseline HOG margin"],
];

function percentage(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function decimal(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

function videoUrl(recordingId: string): string {
  return `/api/review-media/serving-side/${encodeURIComponent(recordingId)}`;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    target.tagName === "INPUT" ||
    target.tagName === "SELECT" ||
    target.tagName === "TEXTAREA"
  );
}

function matchesOutcome(
  result: ServingSideResult,
  outcome: OutcomeFilter,
): boolean {
  if (outcome === "wrong") return !result.correct;
  if (outcome === "correct") return result.correct;
  if (outcome === "near-as-far")
    return result.human === "near" && result.prediction === "far";
  if (outcome === "far-as-near")
    return result.human === "far" && result.prediction === "near";
  return true;
}

function confidence(result: ServingSideResult): number {
  return result.prediction === "near"
    ? result.nearProbability
    : 1 - result.nearProbability;
}

function recordingMetrics(rows: ServingSideResult[]) {
  const near = rows.filter((row) => row.human === "near");
  const far = rows.filter((row) => row.human === "far");
  return {
    rows: rows.length,
    nearRecallMisses: near.filter((row) => row.prediction === "far").length,
    farRecallMisses: far.filter((row) => row.prediction === "near").length,
    nearRecall: near.length
      ? near.filter((row) => row.prediction === "near").length / near.length
      : 0,
    farRecall: far.length
      ? far.filter((row) => row.prediction === "far").length / far.length
      : 0,
  };
}

function unavailable(evaluationPath: string, loadError?: string) {
  return (
    <main className={base.shell}>
      <header className={base.topbar}>
        <Brand className={base.brand} label="Serving-side results" priority />
        <nav>
          <Link href="/">Rally model review ↗</Link>
          <Link href="/serving-side-review">Serving-side labels ↗</Link>
        </nav>
      </header>
      <section className={base.unavailable}>
        <p className={base.eyebrow}>SERVING-SIDE MODEL · RESULT REVIEW</p>
        <h1>
          Results not <em>available.</em>
        </h1>
        <p>
          Train the serving-side specialist first, or set{" "}
          <code>VOLLEYCUT_SERVING_SIDE_EVALUATION</code>.
        </p>
        <code className={base.path}>{evaluationPath}</code>
        {loadError && <small>{loadError}</small>}
      </section>
    </main>
  );
}

function ResultTimeline({
  rows,
  duration,
  selectedId,
  currentTime,
  onSelect,
  onSeek,
}: {
  rows: ServingSideResult[];
  duration: number;
  selectedId: string;
  currentTime: number;
  onSelect: (id: string) => void;
  onSeek: (time: number) => void;
}) {
  return (
    <section className={styles.timeline} aria-label="Recording result timeline">
      <header>
        <div>
          <strong>Full-video result map</strong>
          <span>
            Red marks are mistakes; green marks are correct predictions.
          </span>
        </div>
        <span>{formatTime(duration)}</span>
      </header>
      <div className={styles.timelineLabels}>
        <span>0:00</span>
        <span>{formatTime(duration / 2)}</span>
        <span>{formatTime(duration)}</span>
      </div>
      <div
        className={styles.timelineRail}
        onClick={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect();
          onSeek(
            Math.max(
              0,
              Math.min(
                duration,
                ((event.clientX - bounds.left) / bounds.width) * duration,
              ),
            ),
          );
        }}
        role="presentation"
      >
        {rows.map((row) => (
          <button
            type="button"
            className={styles.timelineEvent}
            data-correct={row.correct ? "true" : "false"}
            data-selected={row.rallyId === selectedId ? "true" : "false"}
            style={{ left: `${Math.min(100, (row.start / duration) * 100)}%` }}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(row.rallyId);
            }}
            title={`${formatTime(row.start)} · human ${row.human} · model ${row.prediction}`}
            aria-label={`Select ${row.rallyId}`}
            key={row.rallyId}
          />
        ))}
        <span
          className={styles.playhead}
          style={{ left: `${Math.min(100, (currentTime / duration) * 100)}%` }}
        />
      </div>
      <div className={styles.timelineLegend}>
        <span data-tone="wrong">mistake</span>
        <span data-tone="correct">correct</span>
        <span data-tone="selected">selected</span>
      </div>
    </section>
  );
}

function LoadedResults({
  data,
  evaluationPath,
  initialRecordingId,
  initialOutcome,
}: Omit<Props, "data" | "loadError"> & { data: ServingSideResultsData }) {
  const defaultRecording =
    data.recordings.find(
      (recording) => recording.recordingId === initialRecordingId,
    ) ??
    data.recordings[0] ??
    null;
  const [environment, setEnvironment] = useState("all");
  const [recordingId, setRecordingId] = useState(
    defaultRecording?.recordingId ?? "",
  );
  const [outcome, setOutcome] = useState<OutcomeFilter>(initialOutcome);
  const [selectedId, setSelectedId] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const videoRef = useRef<HTMLVideoElement>(null);

  const environments = useMemo(
    () => [...new Set(data.recordings.map((row) => row.environment))].sort(),
    [data.recordings],
  );
  const visibleRecordings = useMemo(
    () =>
      data.recordings.filter(
        (recording) =>
          environment === "all" || recording.environment === environment,
      ),
    [data.recordings, environment],
  );
  const recording =
    data.recordings.find((row) => row.recordingId === recordingId) ??
    visibleRecordings[0] ??
    null;
  const recordingRows = useMemo(
    () =>
      data.results
        .filter((row) => row.recordingId === recording?.recordingId)
        .sort((left, right) => left.start - right.start),
    [data.results, recording?.recordingId],
  );
  const filteredRows = useMemo(
    () => recordingRows.filter((row) => matchesOutcome(row, outcome)),
    [outcome, recordingRows],
  );
  const selected =
    filteredRows.find((row) => row.rallyId === selectedId) ??
    filteredRows[0] ??
    null;
  const selectedIndex = selected
    ? filteredRows.findIndex((row) => row.rallyId === selected.rallyId)
    : -1;
  const localMetrics = useMemo(
    () => recordingMetrics(recordingRows),
    [recordingRows],
  );
  const duration = Math.max(recording?.durationSeconds ?? 1, 1);

  const selectResult = useCallback((rallyId: string) => {
    setSelectedId(rallyId);
  }, []);

  const move = useCallback(
    (offset: number) => {
      if (!filteredRows.length) return;
      const nextIndex =
        selectedIndex < 0
          ? 0
          : (selectedIndex + offset + filteredRows.length) %
            filteredRows.length;
      const next = filteredRows[nextIndex];
      if (next) setSelectedId(next.rallyId);
    },
    [filteredRows, selectedIndex],
  );

  const seek = useCallback(
    (time: number) => {
      const target = Math.max(0, Math.min(duration, time));
      setCurrentTime(target);
      if (videoRef.current) videoRef.current.currentTime = target;
    },
    [duration],
  );

  useEffect(() => {
    if (recording && recording.recordingId !== recordingId) {
      setRecordingId(recording.recordingId);
    }
  }, [recording, recordingId]);

  useEffect(() => {
    if (filteredRows.some((row) => row.rallyId === selectedId)) return;
    setSelectedId(filteredRows[0]?.rallyId ?? "");
  }, [filteredRows, selectedId]);

  useEffect(() => {
    if (!selected) return;
    const target = Math.max(0, selected.start - 2);
    setCurrentTime(target);
    if (videoRef.current && videoRef.current.readyState >= 1) {
      videoRef.current.currentTime = target;
    }
  }, [selected]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    if (!recording) return;
    const parameters = new URLSearchParams();
    parameters.set("video", recording.recordingId);
    parameters.set("outcome", outcome);
    window.history.replaceState(
      null,
      "",
      `/serving-side-results?${parameters}`,
    );
  }, [outcome, recording]);

  useEffect(() => {
    function handleShortcut(event: KeyboardEvent) {
      if (event.defaultPrevented || isEditableTarget(event.target)) return;
      if (event.key.toLowerCase() === "j") {
        event.preventDefault();
        move(1);
      } else if (event.key.toLowerCase() === "p") {
        event.preventDefault();
        move(-1);
      }
    }
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [move]);

  return (
    <main className={base.shell}>
      <header className={base.topbar}>
        <Brand className={base.brand} label="Serving-side results" priority />
        <nav>
          <Link href="/">Rally model review ↗</Link>
          <Link href="/serving-side-review">Human labels ↗</Link>
          <Link href="/side-switch-review">Side-switch review ↗</Link>
        </nav>
      </header>

      <header className={base.hero}>
        <div>
          <p className={base.eyebrow}>FROZEN HELD-OUT EVALUATION · PER VIDEO</p>
          <h1>
            See every <em>model mistake.</em>
          </h1>
          <p className={base.intro}>
            Compare the completed human near/far label with the specialist’s
            frozen prediction at each serve. Start on mistakes, then inspect the
            source video around the serve anchor.
          </p>
          <p className={base.sourceLine}>
            {data.recordings.length} videos · {data.results.length} held-out
            serves · model {data.modelFingerprint.slice(0, 12)}…
          </p>
        </div>
        <div className={base.heroMetric}>
          <span>All held-out recordings</span>
          <strong>{percentage(data.metrics.balancedAccuracy)}</strong>
          <small>balanced accuracy</small>
          <b>
            near recall {percentage(data.metrics.nearRecall)} · far recall{" "}
            {percentage(data.metrics.farRecall)}
          </b>
        </div>
      </header>

      <section
        className={base.summaryStrip}
        aria-label="Selected video metrics"
      >
        <div>
          <span>Video serves</span>
          <strong>{localMetrics.rows}</strong>
        </div>
        <div>
          <span>Near recall misses</span>
          <strong data-tone="warning">{localMetrics.nearRecallMisses}</strong>
        </div>
        <div>
          <span>Far recall misses</span>
          <strong data-tone="warning">{localMetrics.farRecallMisses}</strong>
        </div>
        <div>
          <span>Near recall</span>
          <strong data-tone="switch">
            {percentage(localMetrics.nearRecall)}
          </strong>
        </div>
        <div>
          <span>Far recall</span>
          <strong>{percentage(localMetrics.farRecall)}</strong>
        </div>
        <div>
          <span>Queue shown</span>
          <strong>{filteredRows.length}</strong>
        </div>
      </section>

      <section className={base.filterPanel} aria-label="Result filters">
        <div className={base.filterHeading}>
          <span className={base.panelKicker}>01 / VIDEO</span>
          <strong>Choose a result set</strong>
          <small>Recall miss = human side predicted as the opposite side</small>
        </div>
        <label>
          <span>Environment</span>
          <select
            value={environment}
            onChange={(event) => {
              const nextEnvironment = event.target.value;
              setEnvironment(nextEnvironment);
              const nextRecording = data.recordings.find(
                (candidate) =>
                  nextEnvironment === "all" ||
                  candidate.environment === nextEnvironment,
              );
              if (nextRecording) setRecordingId(nextRecording.recordingId);
            }}
          >
            <option value="all">All environments</option>
            {environments.map((item) => (
              <option value={item} key={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Video</span>
          <select
            value={recording?.recordingId ?? ""}
            onChange={(event) => setRecordingId(event.target.value)}
          >
            {visibleRecordings.map((item) => (
              <option value={item.recordingId} key={item.recordingId}>
                {item.split} · {item.recordingId} · {item.errors} wrong
              </option>
            ))}
          </select>
        </label>
        <div className={base.filterChoices}>
          <span>Outcome</span>
          <div>
            {OUTCOME_FILTERS.map((item) => (
              <button
                type="button"
                data-active={outcome === item.value ? "true" : "false"}
                onClick={() => setOutcome(item.value)}
                key={item.value}
              >
                {item.label} ·{" "}
                {
                  recordingRows.filter((row) => matchesOutcome(row, item.value))
                    .length
                }
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className={base.workbench}>
        <aside className={base.queue}>
          <header>
            <div>
              <span className={base.panelKicker}>RESULT QUEUE</span>
              <strong>{filteredRows.length} serves</strong>
            </div>
            <small>{outcome.replaceAll("-", " ")}</small>
          </header>
          <div className={base.queueList}>
            {filteredRows.map((row, index) => (
              <button
                type="button"
                className={base.queueItem}
                data-active={
                  row.rallyId === selected?.rallyId ? "true" : "false"
                }
                data-kind={row.correct ? row.human : "candidate"}
                onClick={() => selectResult(row.rallyId)}
                key={row.rallyId}
              >
                <span className={base.queueIndex}>
                  {String(index + 1).padStart(3, "0")}
                </span>
                <span className={base.queueMain}>
                  <strong>{formatTime(row.start)}</strong>
                  <small>
                    human {row.human} → model {row.prediction}
                  </small>
                </span>
                <span className={base.queueScore}>
                  {percentage(confidence(row), 0)}
                  <i data-decision={row.correct ? row.human : "unclear"} />
                </span>
              </button>
            ))}
            {!filteredRows.length && (
              <p className={base.emptyQueue}>
                No results match this outcome filter.
              </p>
            )}
          </div>
        </aside>

        <section className={base.inspector} aria-label="Selected model result">
          {selected && recording ? (
            <>
              <header className={base.inspectorHeader}>
                <div>
                  <p className={base.eyebrow}>
                    {selected.environment} · {selected.split} ·{" "}
                    {selected.correct ? "correct" : "mistake"}
                  </p>
                  <h2>{selected.rallyId}</h2>
                </div>
                <div className={base.navigationButtons}>
                  <button type="button" onClick={() => move(-1)}>
                    ← Previous <kbd>P</kbd>
                  </button>
                  <span>
                    {selectedIndex + 1} / {filteredRows.length}
                  </span>
                  <button type="button" onClick={() => move(1)}>
                    Next <kbd>J</kbd> →
                  </button>
                </div>
              </header>

              <section
                className={styles.verdict}
                data-correct={selected.correct ? "true" : "false"}
              >
                <div>
                  <span>Human label</span>
                  <strong>{selected.human}</strong>
                  <small>completed review decision</small>
                </div>
                <div className={styles.arrow} aria-hidden="true">
                  →
                </div>
                <div>
                  <span>Model prediction</span>
                  <strong>{selected.prediction}</strong>
                  <small>{percentage(confidence(selected))} confidence</small>
                </div>
                <div className={styles.outcome}>
                  <span>Outcome</span>
                  <strong>{selected.correct ? "Correct" : "Wrong"}</strong>
                  <small>
                    near probability {percentage(selected.nearProbability)}
                  </small>
                </div>
              </section>

              <div className={base.videoStage}>
                <video
                  key={recording.recordingId}
                  ref={videoRef}
                  controls
                  playsInline
                  preload="metadata"
                  src={videoUrl(recording.recordingId)}
                  onLoadedMetadata={(event) => {
                    const target = Math.max(0, selected.start - 2);
                    event.currentTarget.currentTime = target;
                    event.currentTarget.playbackRate = playbackRate;
                    setCurrentTime(target);
                  }}
                  onTimeUpdate={(event) =>
                    setCurrentTime(event.currentTarget.currentTime)
                  }
                  aria-label={`Video results for ${recording.recordingId}`}
                >
                  Your browser does not support video playback.
                </video>
                <span className={base.videoTime}>
                  {formatTime(currentTime)}
                </span>
                <span className={base.videoLabel}>
                  {recording.videoFilename}
                </span>
                <span
                  className={styles.videoVerdict}
                  data-correct={selected.correct ? "true" : "false"}
                >
                  HUMAN {selected.human} · MODEL {selected.prediction}
                </span>
              </div>

              <div className={base.reviewToolbar}>
                <label className={base.speedControl}>
                  <span>Playback speed</span>
                  <select
                    value={playbackRate}
                    onChange={(event) =>
                      setPlaybackRate(Number(event.target.value))
                    }
                  >
                    {PLAYBACK_RATES.map((rate) => (
                      <option value={rate} key={rate}>
                        {rate}×
                      </option>
                    ))}
                  </select>
                </label>
                <div className={styles.seekButtons}>
                  <button
                    type="button"
                    onClick={() => seek(selected.start - 3)}
                  >
                    −3s
                  </button>
                  <button type="button" onClick={() => seek(selected.start)}>
                    Serve anchor {formatTime(selected.start)}
                  </button>
                  <button
                    type="button"
                    onClick={() => seek(selected.start + 3)}
                  >
                    +3s
                  </button>
                </div>
                <div className={base.shortcutLegend}>
                  <span>
                    <kbd>P</kbd> previous
                  </span>
                  <span>
                    <kbd>J</kbd> next
                  </span>
                </div>
              </div>

              <ResultTimeline
                rows={recordingRows}
                duration={duration}
                selectedId={selected.rallyId}
                currentTime={currentTime}
                onSelect={(id) => {
                  if (!filteredRows.some((row) => row.rallyId === id)) {
                    setOutcome("all");
                  }
                  selectResult(id);
                }}
                onSeek={seek}
              />

              <section className={styles.probabilityPanel}>
                <header>
                  <div>
                    <span className={base.panelKicker}>MODEL SCORE</span>
                    <strong>Near-side probability</strong>
                  </div>
                  <strong>{percentage(selected.nearProbability, 2)}</strong>
                </header>
                <div className={styles.probabilityLabels}>
                  <span>Far</span>
                  <span>threshold {percentage(data.threshold, 2)}</span>
                  <span>Near</span>
                </div>
                <div className={styles.probabilityRail}>
                  <span
                    className={styles.threshold}
                    style={{ left: `${data.threshold * 100}%` }}
                  />
                  <span
                    className={styles.probabilityMarker}
                    data-correct={selected.correct ? "true" : "false"}
                    style={{ left: `${selected.nearProbability * 100}%` }}
                  />
                </div>
              </section>

              <section className={styles.detailGrid}>
                <article className={styles.contextCard}>
                  <span className={base.panelKicker}>HUMAN CONTEXT</span>
                  <strong>{selected.notes ?? "No rally note"}</strong>
                  <small>
                    {selected.sourceGroup} ·{" "}
                    {selected.sourceType ?? "unknown source"}
                  </small>
                  <div className={styles.tags}>
                    {selected.tags.map((tag) => (
                      <span key={tag}>{tag}</span>
                    ))}
                  </div>
                </article>
                <article className={styles.featureCard}>
                  <span className={base.panelKicker}>
                    EXISTING FEATURE EVIDENCE
                  </span>
                  <div>
                    {FEATURE_LABELS.map(([name, label]) => (
                      <dl key={name}>
                        <dt>{label}</dt>
                        <dd>{decimal(selected.features[name])}</dd>
                      </dl>
                    ))}
                  </div>
                  <small>Positive signed margins generally favor near.</small>
                </article>
              </section>
            </>
          ) : (
            <div className={base.emptyInspector}>
              <span>No result selected</span>
              <p>Choose another outcome filter or video.</p>
            </div>
          )}
        </section>
      </section>

      <footer className={base.footer}>
        <span>Frozen evaluation</span>
        <code>{evaluationPath}</code>
        <span>
          Human labels and report features are SHA-256-bound to this evaluation.
        </span>
      </footer>
    </main>
  );
}

export function ServingSideResultsClient({
  data,
  evaluationPath,
  initialRecordingId,
  initialOutcome,
  loadError,
}: Props) {
  if (!data) return unavailable(evaluationPath, loadError);
  return (
    <LoadedResults
      data={data}
      evaluationPath={evaluationPath}
      initialRecordingId={initialRecordingId}
      initialOutcome={initialOutcome}
    />
  );
}
