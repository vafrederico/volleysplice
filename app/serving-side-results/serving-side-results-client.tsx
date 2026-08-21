"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import { formatTime } from "@/lib/edit-list";

import base from "../side-switch-review/side-switch-review.module.css";
import styles from "./serving-side-results.module.css";
import type {
  ServingSideCorrectionState,
  ServingSideHumanLabel,
  ServingSideResult,
  ServingSideResultRecording,
  ServingSideResultsData,
} from "./types";

type OutcomeFilter =
  | "all"
  | "wrong"
  | "correct"
  | "uncertain"
  | "near-as-far"
  | "far-as-near"
  | "rally-recovered"
  | "serve-as-not-serve"
  | "not-serve-as-serve"
  | "not-serve";

type Props = {
  data: ServingSideResultsData | null;
  evaluationPath: string;
  initialRecordingId: string | null;
  initialOutcome: OutcomeFilter;
  initialRallyId: string | null;
  loadError?: string;
};

const OUTCOME_FILTERS: Array<{ value: OutcomeFilter; label: string }> = [
  { value: "wrong", label: "Mistakes" },
  { value: "uncertain", label: "Uncertain · review" },
  { value: "all", label: "All" },
  { value: "correct", label: "Correct" },
  { value: "near-as-far", label: "Near recall misses" },
  { value: "far-as-near", label: "Far recall misses" },
  { value: "rally-recovered", label: "Rally-recovered serves" },
  { value: "serve-as-not-serve", label: "Serve gate misses" },
  { value: "not-serve-as-serve", label: "False serves" },
  { value: "not-serve", label: "Not a serve" },
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

function evaluationRole(split: string): string {
  return split === "test" ? "protected held-out" : "development · in-sample";
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
  if (outcome === "uncertain")
    return (
      result.reviewRecommendation === "review" || result.serveReviewRecommended
    );
  if (outcome === "near-as-far")
    return result.human === "near" && result.finalPrediction !== "near";
  if (outcome === "far-as-near")
    return result.human === "far" && result.finalPrediction !== "far";
  if (outcome === "rally-recovered")
    return result.serveDecisionSource === "production-rally-recovery";
  if (outcome === "serve-as-not-serve")
    return (
      result.human !== "not-serve" && result.servePrediction === "not-serve"
    );
  if (outcome === "not-serve-as-serve")
    return result.human === "not-serve" && result.servePrediction === "serve";
  if (outcome === "not-serve") return result.human === "not-serve";
  return true;
}

function confidence(result: ServingSideResult): number {
  return result.prediction === "near"
    ? result.nearProbability
    : 1 - result.nearProbability;
}

function needsReview(result: ServingSideResult): boolean {
  return (
    result.reviewRecommendation === "review" || result.serveReviewRecommended
  );
}

function recordingMetrics(rows: ServingSideResult[]) {
  const near = rows.filter((row) => row.human === "near");
  const far = rows.filter((row) => row.human === "far");
  return {
    rows: near.length + far.length,
    notServes: rows.filter((row) => row.human === "not-serve").length,
    uncertain: rows.filter(needsReview).length,
    recoveredServes: rows.filter(
      (row) => row.serveDecisionSource === "production-rally-recovery",
    ).length,
    serveGateMisses: rows.filter(
      (row) => row.human !== "not-serve" && row.servePrediction === "not-serve",
    ).length,
    falseServes: rows.filter(
      (row) => row.human === "not-serve" && row.servePrediction === "serve",
    ).length,
    nearRecallMisses: near.filter((row) => row.finalPrediction !== "near")
      .length,
    farRecallMisses: far.filter((row) => row.finalPrediction !== "far").length,
    nearRecall: near.length
      ? near.filter((row) => row.finalPrediction === "near").length /
        near.length
      : 0,
    farRecall: far.length
      ? far.filter((row) => row.finalPrediction === "far").length / far.length
      : 0,
  };
}

function aggregateMetrics(rows: ServingSideResult[]) {
  const near = rows.filter((row) => row.human === "near");
  const far = rows.filter((row) => row.human === "far");
  const nearRecall = near.length
    ? near.filter((row) => row.finalPrediction === "near").length / near.length
    : 0;
  const farRecall = far.length
    ? far.filter((row) => row.finalPrediction === "far").length / far.length
    : 0;
  return {
    balancedAccuracy: (nearRecall + farRecall) / 2,
    nearRecall,
    farRecall,
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
          <Link href="/serving-side-flight-review">Flight error review ↗</Link>
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
            Red marks are mistakes; green marks are correct; amber marks are
            review-band predictions or correctly rejected non-serves.
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
            data-outcome={
              needsReview(row)
                ? "uncertain"
                : row.correct
                  ? row.human === "not-serve"
                    ? "not-serve"
                    : "correct"
                  : "wrong"
            }
            data-selected={row.rallyId === selectedId ? "true" : "false"}
            style={{ left: `${Math.min(100, (row.start / duration) * 100)}%` }}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(row.rallyId);
            }}
            title={`${formatTime(row.start)} · human ${row.human} · model ${row.finalPrediction}`}
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
        <span data-tone="not-serve">not a serve</span>
        <span data-tone="uncertain">uncertain</span>
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
  initialRallyId,
}: Omit<Props, "data" | "loadError"> & { data: ServingSideResultsData }) {
  const [correctionState, setCorrectionState] =
    useState<ServingSideCorrectionState>(data.correctionState);
  const [correctionStatus, setCorrectionStatus] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");
  const [correctionError, setCorrectionError] = useState("");
  const results = useMemo(
    () =>
      data.results.map((row) => {
        const human =
          correctionState.corrections[row.rallyId] ?? row.originalHuman;
        return {
          ...row,
          human,
          humanCorrected: human !== row.originalHuman,
          correct: human === row.finalPrediction,
        };
      }),
    [correctionState.corrections, data.results],
  );
  const recordings = useMemo(
    () =>
      data.recordings.map((recording) => {
        const rows = results.filter(
          (row) => row.recordingId === recording.recordingId,
        );
        const correct = rows.filter((row) => row.correct).length;
        return {
          ...recording,
          rows: rows.length,
          correct,
          errors: rows.length - correct,
        };
      }),
    [data.recordings, results],
  );
  const defaultRecording =
    recordings.find(
      (recording) => recording.recordingId === initialRecordingId,
    ) ??
    recordings[0] ??
    null;
  const [environment, setEnvironment] = useState("all");
  const [recordingId, setRecordingId] = useState(
    defaultRecording?.recordingId ?? "",
  );
  const initialRally = data.results.find(
    (row) => row.rallyId === initialRallyId,
  );
  const [outcome, setOutcome] = useState<OutcomeFilter>(
    initialRally && !matchesOutcome(initialRally, initialOutcome)
      ? "all"
      : initialOutcome,
  );
  const [selectedId, setSelectedId] = useState(initialRallyId ?? "");
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const videoRef = useRef<HTMLVideoElement>(null);

  const environments = useMemo(
    () => [...new Set(recordings.map((row) => row.environment))].sort(),
    [recordings],
  );
  const visibleRecordings = useMemo(
    () =>
      recordings.filter(
        (recording) =>
          environment === "all" || recording.environment === environment,
      ),
    [environment, recordings],
  );
  const recording =
    recordings.find((row) => row.recordingId === recordingId) ??
    visibleRecordings[0] ??
    null;
  const recordingRows = useMemo(
    () =>
      results
        .filter((row) => row.recordingId === recording?.recordingId)
        .sort((left, right) => left.start - right.start),
    [recording?.recordingId, results],
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
  const overallMetrics = useMemo(() => aggregateMetrics(results), [results]);
  const duration = Math.max(recording?.durationSeconds ?? 1, 1);
  const isAllVideoInference = data.kind.endsWith("all-video-inference");

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

  const saveHumanCorrection = useCallback(
    async (row: ServingSideResult, human: ServingSideHumanLabel | null) => {
      const corrections = { ...correctionState.corrections };
      if (human === null || human === row.originalHuman) {
        delete corrections[row.rallyId];
      } else {
        corrections[row.rallyId] = human;
      }
      setCorrectionStatus("saving");
      setCorrectionError("");
      try {
        const response = await fetch("/api/serving-side-results/corrections", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            schemaVersion: 1,
            reportKind: correctionState.reportKind,
            reportCreatedAt: correctionState.reportCreatedAt,
            baseDecisionSha256: correctionState.baseDecisionSha256,
            corrections,
          }),
        });
        const payload = (await response.json()) as
          | ServingSideCorrectionState
          | { error?: string };
        if (!response.ok || !("corrections" in payload)) {
          throw new Error(
            "error" in payload && payload.error
              ? payload.error
              : "Correction could not be saved",
          );
        }
        setCorrectionState(payload);
        setCorrectionStatus("saved");
        const updatedHuman =
          payload.corrections[row.rallyId] ?? row.originalHuman;
        const updatedRow = {
          ...row,
          human: updatedHuman,
          humanCorrected: updatedHuman !== row.originalHuman,
          correct: updatedHuman === row.finalPrediction,
        };
        if (!matchesOutcome(updatedRow, outcome)) {
          setOutcome(updatedHuman === "not-serve" ? "not-serve" : "all");
        }
        setSelectedId(row.rallyId);
      } catch (error) {
        setCorrectionStatus("error");
        setCorrectionError(
          error instanceof Error
            ? error.message
            : "Correction could not be saved",
        );
      }
    },
    [correctionState, outcome],
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
    if (selected) parameters.set("rally", selected.rallyId);
    window.history.replaceState(
      null,
      "",
      `/serving-side-results?${parameters}`,
    );
  }, [outcome, recording, selected]);

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
          <Link href="/serving-side-flight-review">Flight error review ↗</Link>
          <Link href="/side-switch-review">Side-switch review ↗</Link>
        </nav>
      </header>

      <header className={base.hero}>
        <div>
          <p className={base.eyebrow}>
            {isAllVideoInference
              ? "FROZEN MODEL INFERENCE · ALL AVAILABLE VIDEOS"
              : "FROZEN HELD-OUT EVALUATION · PER VIDEO"}
          </p>
          <h1>
            See every <em>model mistake.</em>
          </h1>
          <p className={base.intro}>
            Compare the completed human near/far label with the specialist’s
            frozen prediction at each candidate. Start on mistakes, then inspect
            the source video around the serve anchor. Development videos are
            marked in-sample; the test video remains protected held-out.
          </p>
          <p className={base.sourceLine}>
            {recordings.length} videos · {results.length} reviewed candidates ·
            side model {data.modelFingerprint.slice(0, 12)}… · serve gate{" "}
            {data.serveGateFingerprint.slice(0, 12)}…
          </p>
          {data.reviewPolicy && (
            <p className={base.sourceLine}>
              Frozen {percentage(data.reviewPolicy.precisionTarget, 0)} review
              policy · {percentage(data.reviewPolicy.developmentReviewFraction)}{" "}
              development review fraction
            </p>
          )}
        </div>
        <div className={base.heroMetric}>
          <span>
            {isAllVideoInference
              ? "Final side + serve gate"
              : "All held-out recordings"}
          </span>
          <strong>{percentage(overallMetrics.balancedAccuracy)}</strong>
          <small>balanced accuracy</small>
          <b>
            near recall {percentage(overallMetrics.nearRecall)} · far recall{" "}
            {percentage(overallMetrics.farRecall)}
          </b>
          {isAllVideoInference && (
            <b>
              serve P {percentage(data.serveGateMetrics.precision)} · serve R{" "}
              {percentage(data.serveGateMetrics.recall)}
            </b>
          )}
          {isAllVideoInference && data.serveGateMetrics.recoveredServes > 0 && (
            <b>
              {data.serveGateMetrics.recoveredServes} rally-recovered ·{" "}
              {data.serveGateMetrics.recoveredTrueServes} true serves
            </b>
          )}
        </div>
      </header>

      <section
        className={base.summaryStrip}
        aria-label="Selected video metrics"
      >
        <div>
          <span>Side labels</span>
          <strong>{localMetrics.rows}</strong>
        </div>
        <div>
          <span>Not serves</span>
          <strong data-tone="warning">{localMetrics.notServes}</strong>
        </div>
        <div>
          <span>Uncertain</span>
          <strong data-tone="warning">{localMetrics.uncertain}</strong>
        </div>
        <div>
          <span>Rally-recovered</span>
          <strong data-tone="warning">{localMetrics.recoveredServes}</strong>
        </div>
        <div>
          <span>Serve gate misses</span>
          <strong data-tone="warning">{localMetrics.serveGateMisses}</strong>
        </div>
        <div>
          <span>False serves</span>
          <strong data-tone="warning">{localMetrics.falseServes}</strong>
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
      </section>

      <section className={base.filterPanel} aria-label="Result filters">
        <div className={base.filterHeading}>
          <span className={base.panelKicker}>01 / VIDEO</span>
          <strong>Choose a result set</strong>
          <small>
            Mistake = final near, far, or not-serve prediction differs from the
            current human label
          </small>
        </div>
        <label>
          <span>Environment</span>
          <select
            value={environment}
            onChange={(event) => {
              const nextEnvironment = event.target.value;
              setEnvironment(nextEnvironment);
              const nextRecording = recordings.find(
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
                {evaluationRole(item.split)} · {item.recordingId} ·{" "}
                {item.errors} wrong
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
              <strong>{filteredRows.length} candidates</strong>
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
                    human {row.human} → model {row.finalPrediction}
                  </small>
                </span>
                <span className={base.queueScore}>
                  {row.servePrediction === "not-serve"
                    ? "NO SERVE"
                    : row.serveReviewRecommended
                      ? "RECOVERED"
                      : row.reviewRecommendation === "review"
                        ? "REVIEW"
                        : percentage(confidence(row), 0)}
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
                    {selected.environment} · {evaluationRole(selected.split)} ·{" "}
                    {selected.human === "not-serve"
                      ? "not a serve"
                      : selected.correct
                        ? "correct"
                        : "mistake"}
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
                data-not-serve={
                  selected.correct && selected.human === "not-serve"
                    ? "true"
                    : "false"
                }
              >
                <div>
                  <span>Human label</span>
                  <strong>{selected.human}</strong>
                  <small>
                    {selected.humanCorrected
                      ? `corrected · frozen ${selected.originalHuman}`
                      : "frozen review decision"}
                  </small>
                </div>
                <div>
                  <span>Is this a serve?</span>
                  <strong>{selected.servePrediction}</strong>
                  <small>
                    {selected.serveDecisionSource ===
                    "production-rally-recovery"
                      ? "recovered by both-model production rally"
                      : selected.serveDecisionSource === "serve-head"
                        ? "at least one production serve head passed"
                        : "no serve-head or both-model rally evidence"}
                  </small>
                </div>
                <div>
                  <span>Serving side</span>
                  <strong>
                    {selected.servePrediction === "serve"
                      ? selected.prediction
                      : "not applied"}
                  </strong>
                  <small>
                    {percentage(confidence(selected))} side confidence
                  </small>
                </div>
                <div className={styles.outcome}>
                  <span>Outcome · review policy</span>
                  <strong>
                    {needsReview(selected)
                      ? "Review"
                      : selected.correct
                        ? "Correct"
                        : "Wrong"}
                  </strong>
                  <small>
                    final prediction {selected.finalPrediction}
                    {selected.serveReviewRecommended
                      ? " · rally-evidence recovery"
                      : selected.reviewRecommendation === "review"
                        ? " · uncertain side score"
                        : " · automatic side score"}
                  </small>
                </div>
              </section>

              <section className={styles.correctionPanel}>
                <div>
                  <span>Correct the human label</span>
                  <strong>
                    {selected.humanCorrected
                      ? `Corrected from ${selected.originalHuman} to ${selected.human}`
                      : `Frozen label: ${selected.originalHuman}`}
                  </strong>
                  <small>
                    Saved as a NAS correction overlay; the frozen training label
                    and model artifact remain unchanged.
                  </small>
                </div>
                <div className={styles.correctionButtons}>
                  <button
                    type="button"
                    data-active={selected.human === "near" ? "true" : "false"}
                    disabled={correctionStatus === "saving"}
                    onClick={() => saveHumanCorrection(selected, "near")}
                  >
                    Human is near
                  </button>
                  <button
                    type="button"
                    data-active={selected.human === "far" ? "true" : "false"}
                    disabled={correctionStatus === "saving"}
                    onClick={() => saveHumanCorrection(selected, "far")}
                  >
                    Human is far
                  </button>
                  <button
                    type="button"
                    data-active={
                      selected.human === "not-serve" ? "true" : "false"
                    }
                    disabled={correctionStatus === "saving"}
                    onClick={() => saveHumanCorrection(selected, "not-serve")}
                  >
                    Not a serve
                  </button>
                  <button
                    type="button"
                    disabled={
                      correctionStatus === "saving" || !selected.humanCorrected
                    }
                    onClick={() => saveHumanCorrection(selected, null)}
                  >
                    Restore frozen
                  </button>
                </div>
                <output data-status={correctionStatus} aria-live="polite">
                  {correctionStatus === "saving"
                    ? "Saving to NAS…"
                    : correctionStatus === "saved"
                      ? `Saved · ${Object.keys(correctionState.corrections).length} corrections`
                      : correctionStatus === "error"
                        ? correctionError
                        : `${Object.keys(correctionState.corrections).length} saved corrections`}
                </output>
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
                  data-not-serve={
                    selected.correct && selected.human === "not-serve"
                      ? "true"
                      : "false"
                  }
                >
                  {`HUMAN ${selected.human} · MODEL ${selected.finalPrediction}`}
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
                    <span className={base.panelKicker}>
                      PRODUCTION SERVE HEADS
                    </span>
                    <strong>Is this a serve?</strong>
                  </div>
                  <strong>{selected.servePrediction}</strong>
                </header>
                <div className={styles.serveHeadScores}>
                  {(
                    [
                      ["All-labels V2", selected.serveEvidence.allLabelsV2],
                      [
                        "Previous production",
                        selected.serveEvidence.previousProduction,
                      ],
                    ] as const
                  ).map(([label, evidence]) => (
                    <article key={label}>
                      <span>{label}</span>
                      <strong>{percentage(evidence.peakProbability, 2)}</strong>
                      <small>
                        peak {formatTime(evidence.peakTime)} · threshold{" "}
                        {percentage(evidence.threshold)} ·{" "}
                        {evidence.crossesThreshold ? "passes" : "below"}
                      </small>
                    </article>
                  ))}
                </div>
                <small className={styles.gateNote}>
                  Maximum source-aligned score within ±1 second of the candidate
                  anchor. Either head passing marks this as a serve. If both
                  miss, an anchor contained in a both-model production rally
                  recovers the side and requires review.
                </small>
                {selected.serveEvidence.productionRally?.interval && (
                  <small className={styles.gateNote}>
                    Production rally{" "}
                    {formatTime(
                      selected.serveEvidence.productionRally.interval.start,
                    )}
                    –
                    {formatTime(
                      selected.serveEvidence.productionRally.interval.end,
                    )}{" "}
                    ·{" "}
                    {selected.serveEvidence.productionRally.interval.agreement}
                    {selected.serveEvidence.productionRally.recoversServe
                      ? " · anchor contained · serve recovered"
                      : " · no recovery"}
                  </small>
                )}
              </section>

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
                  <span>
                    {data.reviewPolicy
                      ? "review " +
                        percentage(data.reviewPolicy.farThreshold, 1) +
                        "–" +
                        percentage(data.reviewPolicy.nearThreshold, 1)
                      : "threshold " + percentage(data.threshold, 2)}
                  </span>
                  <span>Near</span>
                </div>
                <div className={styles.probabilityRail}>
                  {data.reviewPolicy && (
                    <span
                      className={styles.reviewBand}
                      style={{
                        left: data.reviewPolicy.farThreshold * 100 + "%",
                        width:
                          (data.reviewPolicy.nearThreshold -
                            data.reviewPolicy.farThreshold) *
                            100 +
                          "%",
                      }}
                    />
                  )}
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
                {data.reviewPolicy && (
                  <small className={styles.gateNote}>
                    {selected.reviewRecommendation === "review"
                      ? "Review recommended: this score is inside the development-frozen uncertainty band."
                      : "Automatic " +
                        selected.reviewRecommendation +
                        " decision: this score is outside the uncertainty band."}{" "}
                    Development coverage{" "}
                    {percentage(data.reviewPolicy.developmentCoverage)} at{" "}
                    {percentage(data.reviewPolicy.developmentSelectiveAccuracy)}{" "}
                    selective accuracy.
                  </small>
                )}
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
          Frozen labels and report features are SHA-256-bound; corrections stay
          separate and label-revision-bound.
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
  initialRallyId,
  loadError,
}: Props) {
  if (!data) return unavailable(evaluationPath, loadError);
  return (
    <LoadedResults
      data={data}
      evaluationPath={evaluationPath}
      initialRecordingId={initialRecordingId}
      initialOutcome={initialOutcome}
      initialRallyId={initialRallyId}
    />
  );
}
