"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import { formatTime } from "@/lib/edit-list";

import base from "../side-switch-review/side-switch-review.module.css";
import styles from "./serving-side-flight-review.module.css";
import type {
  BallFlightVisibility,
  ContactTiming,
  MotionDirectionAssessment,
  ServerVisibility,
  ServingSideFlightAnnotation,
  ServingSideFlightAnnotationState,
  ServingSideFlightReviewData,
  ServingSideFlightReviewResult,
} from "./types";

type OutcomeFilter = "mistakes" | "all" | "correct";
type ReviewFilter = "unreviewed" | "all" | "reviewed";
type SavedDraft = Omit<ServingSideFlightAnnotation, "reviewedAt">;
type Draft = {
  serverVisibility: ServerVisibility | null;
  contactTiming: ContactTiming | null;
  correctedServeAnchorSeconds: number | null;
  ballFlightVisibility: BallFlightVisibility | null;
  motionDirection: MotionDirectionAssessment | null;
  notes: string;
};

type Props = {
  data: ServingSideFlightReviewData | null;
  evaluationPath: string;
  initialRallyId: string | null;
  loadError?: string;
};

const EMPTY_DRAFT: Draft = {
  serverVisibility: null,
  contactTiming: null,
  correctedServeAnchorSeconds: null,
  ballFlightVisibility: null,
  motionDirection: null,
  notes: "",
};

const SERVER_VISIBILITY_CHOICES: Array<{
  value: ServerVisibility;
  label: string;
  detail: string;
}> = [
  {
    value: "visible",
    label: "Visible",
    detail: "Server and contact are in frame",
  },
  {
    value: "partial",
    label: "Partial",
    detail: "Only part of the server/contact is visible",
  },
  {
    value: "offscreen",
    label: "Offscreen",
    detail: "Serve happens outside the frame",
  },
  {
    value: "unclear",
    label: "Can't tell",
    detail: "Visibility cannot be judged",
  },
];

const CONTACT_TIMING_CHOICES: Array<{
  value: ContactTiming;
  label: string;
  detail: string;
}> = [
  {
    value: "on-anchor",
    label: "Anchor is right",
    detail: "Contact occurs at the marked start",
  },
  {
    value: "before-anchor",
    label: "Contact is earlier",
    detail: "Actual contact occurs before the anchor",
  },
  {
    value: "after-anchor",
    label: "Contact is later",
    detail: "Actual contact occurs after the anchor",
  },
  {
    value: "unclear",
    label: "Can't tell",
    detail: "Contact timing cannot be judged",
  },
];

const BALL_VISIBILITY_CHOICES: Array<{
  value: BallFlightVisibility;
  label: string;
  detail: string;
}> = [
  {
    value: "visible",
    label: "Visible",
    detail: "Ball flight can be followed after contact",
  },
  {
    value: "not-visible",
    label: "Not visible",
    detail: "Ball flight cannot be followed",
  },
  { value: "unclear", label: "Can't tell", detail: "Too ambiguous to label" },
];

const MOTION_DIRECTION_CHOICES: Array<{
  value: MotionDirectionAssessment;
  label: string;
  detail: string;
}> = [
  {
    value: "matches-human-side",
    label: "Agrees",
    detail: "Visible flight/motion agrees with the human serving side",
  },
  {
    value: "opposes-human-side",
    label: "Opposite",
    detail: "Visible flight/motion appears to travel the opposite way",
  },
  {
    value: "unclear",
    label: "No clear direction",
    detail: "There is no reliable directional motion to judge",
  },
];

function percentage(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
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

function modelConfidence(result: ServingSideFlightReviewResult): number {
  return result.prediction === "near"
    ? result.probabilityNear
    : 1 - result.probabilityNear;
}

function draftFor(annotation: ServingSideFlightAnnotation | undefined): Draft {
  if (!annotation) return { ...EMPTY_DRAFT };
  return {
    serverVisibility: annotation.serverVisibility,
    contactTiming: annotation.contactTiming,
    correctedServeAnchorSeconds: annotation.correctedServeAnchorSeconds,
    ballFlightVisibility: annotation.ballFlightVisibility,
    motionDirection: annotation.motionDirection,
    notes: annotation.notes,
  };
}

function savableDraft(draft: Draft): SavedDraft | null {
  if (
    !draft.serverVisibility ||
    !draft.contactTiming ||
    !draft.ballFlightVisibility ||
    !draft.motionDirection
  ) {
    return null;
  }
  return {
    serverVisibility: draft.serverVisibility,
    contactTiming: draft.contactTiming,
    correctedServeAnchorSeconds: draft.correctedServeAnchorSeconds,
    ballFlightVisibility: draft.ballFlightVisibility,
    motionDirection: draft.motionDirection,
    notes: draft.notes,
  };
}

function ChoiceGroup<T extends string>({
  legend,
  help,
  value,
  choices,
  onChange,
}: {
  legend: string;
  help: string;
  value: T | null;
  choices: Array<{ value: T; label: string; detail: string }>;
  onChange: (value: T) => void;
}) {
  return (
    <fieldset className={styles.choiceGroup}>
      <legend>{legend}</legend>
      <p>{help}</p>
      <div>
        {choices.map((choice) => (
          <button
            type="button"
            data-active={value === choice.value ? "true" : "false"}
            onClick={() => onChange(choice.value)}
            title={choice.detail}
            key={choice.value}
          >
            <strong>{choice.label}</strong>
            <small>{choice.detail}</small>
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function unavailable(evaluationPath: string, loadError?: string) {
  return (
    <main className={base.shell}>
      <header className={base.topbar}>
        <Brand className={base.brand} label="Flight error review" priority />
        <nav>
          <Link href="/serving-side-results">Serving-side results ↗</Link>
        </nav>
      </header>
      <section className={base.unavailable}>
        <p className={base.eyebrow}>SERVING-SIDE FLIGHT MODEL · ERROR REVIEW</p>
        <h1>
          Review data not <em>available.</em>
        </h1>
        <p>
          The frozen flight evaluation or its source report could not be loaded.
        </p>
        <code className={base.path}>{evaluationPath}</code>
        {loadError && <small>{loadError}</small>}
      </section>
    </main>
  );
}

function LoadedReview({
  data,
  evaluationPath,
  initialRallyId,
}: Omit<Props, "data" | "loadError"> & { data: ServingSideFlightReviewData }) {
  const [annotationState, setAnnotationState] =
    useState<ServingSideFlightAnnotationState>(data.annotationState);
  const initial = data.results.find((row) => row.rallyId === initialRallyId);
  const [outcome, setOutcome] = useState<OutcomeFilter>(
    initial?.correct ? "all" : "mistakes",
  );
  const [reviewFilter, setReviewFilter] = useState<ReviewFilter>(
    initialRallyId ? "all" : "unreviewed",
  );
  const [recordingId, setRecordingId] = useState("all");
  const [selectedId, setSelectedId] = useState(initialRallyId ?? "");
  const [draft, setDraft] = useState<Draft>(() =>
    draftFor(
      initial ? data.annotationState.annotations[initial.rallyId] : undefined,
    ),
  );
  const [saveStatus, setSaveStatus] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");
  const [saveError, setSaveError] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(0.75);
  const videoRef = useRef<HTMLVideoElement>(null);

  const rows = useMemo(
    () =>
      [...data.results].sort((left, right) => {
        const recordingOrder = left.recordingId.localeCompare(
          right.recordingId,
        );
        return recordingOrder || left.start - right.start;
      }),
    [data.results],
  );
  const filteredRows = useMemo(
    () =>
      rows.filter((row) => {
        if (outcome === "mistakes" && row.correct) return false;
        if (outcome === "correct" && !row.correct) return false;
        if (recordingId !== "all" && row.recordingId !== recordingId)
          return false;
        const reviewed = Boolean(annotationState.annotations[row.rallyId]);
        if (reviewFilter === "unreviewed" && reviewed) return false;
        if (reviewFilter === "reviewed" && !reviewed) return false;
        return true;
      }),
    [annotationState.annotations, outcome, recordingId, reviewFilter, rows],
  );
  const selected =
    filteredRows.find((row) => row.rallyId === selectedId) ??
    filteredRows[0] ??
    null;
  const selectedIndex = selected
    ? filteredRows.findIndex((row) => row.rallyId === selected.rallyId)
    : -1;
  const recording = selected
    ? (data.recordings.find(
        (item) => item.recordingId === selected.recordingId,
      ) ?? null)
    : null;
  const errorRows = rows.filter((row) => !row.correct);
  const reviewedErrors = errorRows.filter(
    (row) => annotationState.annotations[row.rallyId],
  ).length;
  const currentAnnotation = selected
    ? annotationState.annotations[selected.rallyId]
    : undefined;
  const completedDraft = savableDraft(draft);

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
      const duration = recording?.durationSeconds ?? 0;
      const target = Math.max(0, Math.min(duration, time));
      setCurrentTime(target);
      if (videoRef.current) videoRef.current.currentTime = target;
    },
    [recording?.durationSeconds],
  );

  const save = useCallback(
    async (annotation: SavedDraft | null) => {
      if (!selected) return;
      setSaveStatus("saving");
      setSaveError("");
      try {
        const response = await fetch(
          "/api/serving-side-flight-review/annotations",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              schemaVersion: 1,
              experimentSha256: data.experimentSha256,
              rallyId: selected.rallyId,
              annotation,
            }),
          },
        );
        const payload = (await response.json()) as
          | ServingSideFlightAnnotationState
          | { error?: string };
        if (!response.ok || !("annotations" in payload)) {
          throw new Error(
            "error" in payload && payload.error
              ? payload.error
              : "Review could not be saved",
          );
        }
        setAnnotationState(payload);
        setSaveStatus("saved");
      } catch (error) {
        setSaveStatus("error");
        setSaveError(
          error instanceof Error ? error.message : "Review could not be saved",
        );
      }
    },
    [data.experimentSha256, selected],
  );

  useEffect(() => {
    if (filteredRows.some((row) => row.rallyId === selectedId)) return;
    setSelectedId(filteredRows[0]?.rallyId ?? "");
  }, [filteredRows, selectedId]);

  useEffect(() => {
    if (!selected) return;
    setDraft(draftFor(annotationState.annotations[selected.rallyId]));
    setSaveStatus("idle");
    setSaveError("");
    const target = Math.max(0, selected.start - 2.5);
    setCurrentTime(target);
    if (videoRef.current && videoRef.current.readyState >= 1) {
      videoRef.current.currentTime = target;
    }
  }, [annotationState.annotations, selected]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    if (!selected) return;
    const parameters = new URLSearchParams({ rally: selected.rallyId });
    window.history.replaceState(
      null,
      "",
      `/serving-side-flight-review?${parameters}`,
    );
  }, [selected]);

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
        <Brand className={base.brand} label="Flight error review" priority />
        <nav>
          <Link href="/serving-side-results">All-video results ↗</Link>
          <Link href="/serving-side-review">Human labels ↗</Link>
        </nav>
      </header>

      <header className={base.hero}>
        <div>
          <p className={base.eyebrow}>
            DEVELOPMENT CROSS-VALIDATION · FAILURE-MODE ANNOTATION
          </p>
          <h1>
            Explain every <em>flight-model miss.</em>
          </h1>
          <p className={base.intro}>
            Review the exact out-of-source-group predictions used for the new
            model metrics. Label what is visible and whether the serve anchor is
            correct. Camera movement is estimated automatically and is not a
            human label.
          </p>
          <p className={base.sourceLine}>
            {data.configuration} · {data.featureFamily} · L2 {data.l2} ·{" "}
            {evaluationPath}
          </p>
        </div>
        <div className={base.heroMetric}>
          <span>Error reviews complete</span>
          <strong>
            {reviewedErrors}/{errorRows.length}
          </strong>
          <small>{errorRows.length - reviewedErrors} mistakes remaining</small>
          <b>
            precision near {percentage(data.metrics.nearPrecision)} · far{" "}
            {percentage(data.metrics.farPrecision)}
          </b>
          <b>
            recall near {percentage(data.metrics.nearRecall)} · far{" "}
            {percentage(data.metrics.farRecall)}
          </b>
        </div>
      </header>

      <section className={base.filterPanel} aria-label="Flight review filters">
        <div className={base.filterHeading}>
          <span className={base.panelKicker}>01 / QUEUE</span>
          <strong>Choose what to review</strong>
          <small>
            Defaults to unreviewed mistakes from the selected v3 candidate.
          </small>
        </div>
        <label>
          <span>Video</span>
          <select
            value={recordingId}
            onChange={(event) => setRecordingId(event.target.value)}
          >
            <option value="all">All 28 development videos</option>
            {data.recordings
              .filter((item) => item.errors > 0 || outcome !== "mistakes")
              .map((item) => (
                <option value={item.recordingId} key={item.recordingId}>
                  {item.recordingId} · {item.errors} mistakes
                </option>
              ))}
          </select>
        </label>
        <div className={base.filterChoices}>
          <span>Outcome</span>
          <div>
            {(["mistakes", "all", "correct"] as const).map((value) => (
              <button
                type="button"
                data-active={outcome === value ? "true" : "false"}
                onClick={() => setOutcome(value)}
                key={value}
              >
                {value} ·{" "}
                {
                  rows.filter((row) =>
                    value === "all"
                      ? true
                      : value === "mistakes"
                        ? !row.correct
                        : row.correct,
                  ).length
                }
              </button>
            ))}
          </div>
        </div>
        <div className={base.filterChoices}>
          <span>Review state</span>
          <div>
            {(["unreviewed", "all", "reviewed"] as const).map((value) => (
              <button
                type="button"
                data-active={reviewFilter === value ? "true" : "false"}
                onClick={() => setReviewFilter(value)}
                key={value}
              >
                {value}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.workbench}>
        <aside className={base.queue}>
          <header>
            <div>
              <span className={base.panelKicker}>REVIEW QUEUE</span>
              <strong>{filteredRows.length} examples</strong>
            </div>
            <small>{reviewFilter}</small>
          </header>
          <div className={base.queueList}>
            {filteredRows.map((row, index) => {
              const reviewed = Boolean(
                annotationState.annotations[row.rallyId],
              );
              return (
                <button
                  type="button"
                  className={base.queueItem}
                  data-active={
                    row.rallyId === selected?.rallyId ? "true" : "false"
                  }
                  data-kind={row.correct ? row.human : "candidate"}
                  onClick={() => setSelectedId(row.rallyId)}
                  key={row.rallyId}
                >
                  <span className={base.queueIndex}>
                    {String(index + 1).padStart(3, "0")}
                  </span>
                  <span className={base.queueMain}>
                    <strong>{row.recordingId}</strong>
                    <small>
                      {formatTime(row.start)} · human {row.human} → model{" "}
                      {row.prediction}
                    </small>
                  </span>
                  <span className={styles.reviewBadge} data-reviewed={reviewed}>
                    {reviewed
                      ? "REVIEWED"
                      : percentage(modelConfidence(row), 0)}
                  </span>
                </button>
              );
            })}
            {!filteredRows.length && (
              <p className={base.emptyQueue}>
                No examples match these filters. Choose reviewed or all to
                revisit completed annotations.
              </p>
            )}
          </div>
        </aside>

        <section
          className={base.inspector}
          aria-label="Flight error annotation"
        >
          {selected && recording ? (
            <>
              <header className={base.inspectorHeader}>
                <div>
                  <p className={base.eyebrow}>
                    {selected.environment} · {selected.sourceGroup} ·{" "}
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
                data-correct={selected.correct}
              >
                <div>
                  <span>Human side</span>
                  <strong>{selected.human}</strong>
                  <small>correction-clean label</small>
                </div>
                <div>
                  <span>New model</span>
                  <strong>{selected.prediction}</strong>
                  <small>
                    {percentage(modelConfidence(selected))} confidence
                  </small>
                </div>
                <div>
                  <span>Result</span>
                  <strong>{selected.correct ? "Correct" : "Wrong"}</strong>
                  <small>out-of-source-group prediction</small>
                </div>
                <Link
                  href={`/serving-side-results?rally=${encodeURIComponent(selected.rallyId)}&outcome=all`}
                >
                  Correct human label ↗
                </Link>
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
                    const target = Math.max(0, selected.start - 2.5);
                    event.currentTarget.currentTime = target;
                    event.currentTarget.playbackRate = playbackRate;
                    setCurrentTime(target);
                  }}
                  onTimeUpdate={(event) =>
                    setCurrentTime(event.currentTarget.currentTime)
                  }
                  aria-label={`Flight review video for ${recording.recordingId}`}
                >
                  Your browser does not support video playback.
                </video>
                <span className={base.videoTime}>
                  {formatTime(currentTime)}
                </span>
                <span className={base.videoLabel}>
                  {recording.videoFilename}
                </span>
                <span className={styles.anchorLabel}>
                  SERVE ANCHOR {formatTime(selected.start)}
                </span>
              </div>

              <div className={styles.videoToolbar}>
                <label>
                  <span>Playback speed</span>
                  <select
                    value={playbackRate}
                    onChange={(event) =>
                      setPlaybackRate(Number(event.target.value))
                    }
                  >
                    {[0.25, 0.5, 0.75, 1, 1.5].map((rate) => (
                      <option value={rate} key={rate}>
                        {rate}×
                      </option>
                    ))}
                  </select>
                </label>
                <div>
                  <button type="button" onClick={() => seek(currentTime - 1)}>
                    −1s
                  </button>
                  <button type="button" onClick={() => seek(selected.start)}>
                    Go to anchor
                  </button>
                  <button type="button" onClick={() => seek(currentTime + 1)}>
                    +1s
                  </button>
                </div>
              </div>

              <section className={styles.annotationPanel}>
                <header>
                  <div>
                    <span className={base.panelKicker}>02 / FAILURE MODE</span>
                    <h3>What makes this example difficult?</h3>
                    <p>
                      Choose the closest answer. “Can’t tell” is useful data and
                      is better than guessing.
                    </p>
                  </div>
                  {currentAnnotation && (
                    <small>
                      Last reviewed{" "}
                      {new Date(currentAnnotation.reviewedAt).toLocaleString()}
                    </small>
                  )}
                </header>

                <ChoiceGroup
                  legend="Server visibility"
                  help="Can you see the person serving at ball contact?"
                  value={draft.serverVisibility}
                  choices={SERVER_VISIBILITY_CHOICES}
                  onChange={(serverVisibility) =>
                    setDraft((current) => ({ ...current, serverVisibility }))
                  }
                />

                <ChoiceGroup
                  legend="Serve-contact timing"
                  help="Does the marked serve anchor line up with actual ball contact?"
                  value={draft.contactTiming}
                  choices={CONTACT_TIMING_CHOICES}
                  onChange={(contactTiming) =>
                    setDraft((current) => ({
                      ...current,
                      contactTiming,
                      correctedServeAnchorSeconds:
                        contactTiming === "on-anchor" ||
                        contactTiming === "unclear"
                          ? null
                          : current.correctedServeAnchorSeconds,
                    }))
                  }
                />

                <div className={styles.exactAnchor}>
                  <div>
                    <strong>Optional exact contact time</strong>
                    <small>
                      Pause on contact, then save the current video time. This
                      is more useful than only saying earlier or later.
                    </small>
                  </div>
                  <button
                    type="button"
                    disabled={Math.abs(currentTime - selected.start) < 0.04}
                    onClick={() =>
                      setDraft((current) => ({
                        ...current,
                        contactTiming:
                          currentTime < selected.start
                            ? "before-anchor"
                            : "after-anchor",
                        correctedServeAnchorSeconds: Number(
                          currentTime.toFixed(3),
                        ),
                      }))
                    }
                  >
                    Use {formatTime(currentTime)} as contact
                  </button>
                  <output>
                    {draft.correctedServeAnchorSeconds === null
                      ? "No corrected time"
                      : `${formatTime(draft.correctedServeAnchorSeconds)} · ${(
                          draft.correctedServeAnchorSeconds - selected.start
                        ).toFixed(2)}s from anchor`}
                  </output>
                </div>

                <ChoiceGroup
                  legend="Ball flight visibility"
                  help="After contact, can you visually follow the ball moving across the frame?"
                  value={draft.ballFlightVisibility}
                  choices={BALL_VISIBILITY_CHOICES}
                  onChange={(ballFlightVisibility) =>
                    setDraft((current) => ({
                      ...current,
                      ballFlightVisibility,
                    }))
                  }
                />

                <ChoiceGroup
                  legend="Flight / motion direction"
                  help={`Does the visible motion agree with a ${selected.human}-side serve? This is about what you see, not an internal model value.`}
                  value={draft.motionDirection}
                  choices={MOTION_DIRECTION_CHOICES}
                  onChange={(motionDirection) =>
                    setDraft((current) => ({ ...current, motionDirection }))
                  }
                />

                <label className={styles.notes}>
                  <span>Optional note</span>
                  <textarea
                    value={draft.notes}
                    maxLength={1000}
                    rows={3}
                    placeholder="Example: another player crosses the frame at contact"
                    onChange={(event) =>
                      setDraft((current) => ({
                        ...current,
                        notes: event.target.value,
                      }))
                    }
                  />
                  <small>{draft.notes.length}/1000</small>
                </label>

                <footer>
                  <div>
                    <output data-status={saveStatus} aria-live="polite">
                      {saveStatus === "saving"
                        ? "Saving to NAS…"
                        : saveStatus === "saved"
                          ? `${Object.keys(annotationState.annotations).length} reviews saved on NAS`
                          : saveStatus === "error"
                            ? saveError
                            : currentAnnotation
                              ? "This example already has a saved review"
                              : "Ready to save this review"}
                    </output>
                    <small>
                      Bound to experiment {data.experimentSha256.slice(0, 12)}…
                    </small>
                  </div>
                  {currentAnnotation && (
                    <button
                      type="button"
                      className={styles.secondaryButton}
                      disabled={saveStatus === "saving"}
                      onClick={() => save(null)}
                    >
                      Clear review
                    </button>
                  )}
                  <button
                    type="button"
                    className={styles.saveButton}
                    disabled={saveStatus === "saving" || !completedDraft}
                    onClick={() => {
                      if (completedDraft) save(completedDraft);
                    }}
                  >
                    {completedDraft
                      ? reviewFilter === "unreviewed"
                        ? "Save + next unreviewed"
                        : "Save review"
                      : "Choose all four labels"}
                  </button>
                </footer>
              </section>
            </>
          ) : (
            <div className={styles.completed}>
              <p className={base.eyebrow}>QUEUE COMPLETE</p>
              <h2>No unreviewed examples remain in this filter.</h2>
              <button type="button" onClick={() => setReviewFilter("reviewed")}>
                Revisit reviewed examples
              </button>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

export function ServingSideFlightReviewClient(props: Props) {
  if (!props.data) return unavailable(props.evaluationPath, props.loadError);
  return <LoadedReview {...props} data={props.data} />;
}
