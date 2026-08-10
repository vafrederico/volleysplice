"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  downloadLabels,
  formatPreciseTime,
  parseLabelDocument,
  roundTime,
  type HardNegative,
  type IgnoredInterval,
  type LabelDocument,
  type RallyLabel,
} from "@/lib/annotations";
import styles from "./labeling-editor.module.css";

type IntervalKind = "rally" | "ignored" | "negative";

type PreparedTaskSummary = {
  id: string;
  priority: number;
  environment: LabelDocument["recording"]["environment"];
  split: LabelDocument["recording"]["split"];
  durationSeconds: number;
  originalFilename: string;
  videoFilename: string;
};

function overlaps(start: number, end: number, rows: Array<{ start: number; end: number }>): boolean {
  return rows.some((row) => start < row.end && row.start < end);
}

function totalSeconds(rows: Array<{ start: number; end: number }>): number {
  return rows.reduce((total, row) => total + row.end - row.start, 0);
}

export function LabelingEditor() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const preparedRequestRef = useRef<AbortController | null>(null);
  const [labels, setLabels] = useState<LabelDocument | null>(null);
  const [preparedTasks, setPreparedTasks] = useState<PreparedTaskSummary[]>([]);
  const [selectedPreparedTask, setSelectedPreparedTask] = useState("");
  const [preparedTasksLoading, setPreparedTasksLoading] = useState(true);
  const [savingDraft, setSavingDraft] = useState(false);
  const [lastSavedAt, setLastSavedAt] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoFilename, setVideoFilename] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [rallyStart, setRallyStart] = useState<number | null>(null);
  const [ignoredStart, setIgnoredStart] = useState<number | null>(null);
  const [negativeStart, setNegativeStart] = useState<number | null>(null);
  const [negativeCategory, setNegativeCategory] = useState("foreground-crossing");
  const [message, setMessage] = useState(
    "Choose a prepared pilot task, or use the local fallback files.",
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => {
    const controller = new AbortController();
    async function loadPreparedTasks() {
      try {
        const response = await fetch("/api/labeling/tasks", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Prepared pilot tasks are unavailable");
        const payload = (await response.json()) as { tasks?: PreparedTaskSummary[] };
        if (!Array.isArray(payload.tasks)) throw new Error("Prepared task list is invalid");
        setPreparedTasks(payload.tasks);
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(loadError instanceof Error ? loadError.message : "Could not list pilot tasks");
        }
      } finally {
        if (!controller.signal.aborted) setPreparedTasksLoading(false);
      }
    }
    void loadPreparedTasks();
    return () => controller.abort();
  }, []);

  const allRows = useMemo(() => {
    if (!labels) return [];
    return [...labels.rallies, ...labels.ignoredIntervals, ...labels.hardNegatives];
  }, [labels]);

  const selectedPreparedSummary = useMemo(
    () => preparedTasks.find((task) => task.id === selectedPreparedTask) ?? null,
    [preparedTasks, selectedPreparedTask],
  );

  const completionIssues = useMemo(() => {
    if (!labels) return ["Load a label task"];
    const issues: string[] = [];
    if (!videoUrl) issues.push("Select the matching proxy video");
    if (videoFilename && videoFilename !== labels.recording.videoFilename) {
      issues.push(`Selected video must be ${labels.recording.videoFilename}`);
    }
    if (
      videoDuration !== null &&
      Math.abs(videoDuration - labels.recording.durationSeconds) > Math.max(0.1, 1 / 24)
    ) {
      issues.push("Selected video duration does not match the task");
    }
    if (!labels.annotation.annotator.trim()) issues.push("Enter the annotator name");
    if (!labels.annotation.continuousVideoReviewed) issues.push("Confirm the complete video was reviewed");
    if (labels.recording.game.playersPerTeam === null) issues.push("Set players per team");
    if (labels.rallies.length === 0) issues.push("Label at least one rally");
    if (rallyStart !== null || ignoredStart !== null || negativeStart !== null) {
      issues.push("Finish or cancel the open interval marker");
    }
    const orderedGroups = [labels.rallies, labels.ignoredIntervals, labels.hardNegatives];
    orderedGroups.forEach((rows) => {
      rows.forEach((row, index) => {
        if (row.start < 0 || row.end <= row.start || row.end > labels.recording.durationSeconds) {
          issues.push(`Fix invalid interval at ${formatPreciseTime(row.start)} (#${index + 1})`);
        }
        if (index > 0 && row.start < rows[index - 1].end) issues.push("Intervals overlap or are unordered");
      });
    });
    labels.rallies.forEach((row) => {
      if (overlaps(row.start, row.end, labels.ignoredIntervals)) issues.push("A rally overlaps ignored time");
      if (overlaps(row.start, row.end, labels.hardNegatives)) issues.push("A rally overlaps a hard negative");
    });
    labels.ignoredIntervals.forEach((row) => {
      if (overlaps(row.start, row.end, labels.hardNegatives)) issues.push("Ignored time overlaps a hard negative");
    });
    return [...new Set(issues)];
  }, [ignoredStart, labels, negativeStart, rallyStart, videoDuration, videoFilename, videoUrl]);

  function markChanged(document: LabelDocument): LabelDocument {
    return {
      ...document,
      annotation: {
        ...document.annotation,
        status: "in-progress",
        reviewedAt: null,
      },
    };
  }

  async function loadTask(file: File | undefined) {
    if (!file) return;
    preparedRequestRef.current?.abort();
    setSelectedPreparedTask("");
    setLastSavedAt(null);
    setError(null);
    try {
      const document = parseLabelDocument(JSON.parse(await file.text()));
      setLabels(document);
      setRallyStart(null);
      setIgnoredStart(null);
      setNegativeStart(null);
      setCurrentTime(0);
      setMessage(
        document.annotation.status === "not-started"
          ? "Task loaded. Select the matching proxy video."
          : `Resumed ${document.rallies.length} rally labels.`,
      );
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load label document");
    }
  }

  function loadVideo(file: File | undefined) {
    if (!file) return;
    setError(null);
    const nextUrl = URL.createObjectURL(file);
    setVideoUrl(nextUrl);
    setVideoFilename(file.name);
    setVideoDuration(null);
    setMessage(`Loaded local video ${file.name}. Nothing is uploaded.`);
  }

  async function loadPreparedTask(id: string) {
    setSelectedPreparedTask(id);
    if (!id) return;
    preparedRequestRef.current?.abort();
    const controller = new AbortController();
    preparedRequestRef.current = controller;
    setError(null);
    setPreparedTasksLoading(true);
    try {
      const response = await fetch(`/api/labeling/tasks/${encodeURIComponent(id)}`, {
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("Could not load the selected pilot task");
      const documentSource = response.headers.get("X-VolleyCut-Document-Source");
      const savedAt = response.headers.get("X-VolleyCut-Saved-At");
      const document = parseLabelDocument(await response.json());
      setLabels(document);
      setVideoUrl(`/api/labeling/tasks/${encodeURIComponent(id)}/video`);
      setVideoFilename(document.recording.videoFilename);
      setVideoDuration(null);
      setCurrentTime(0);
      setRallyStart(null);
      setIgnoredStart(null);
      setNegativeStart(null);
      setLastSavedAt(savedAt);
      setMessage(
        documentSource === "draft"
          ? `Resumed the NAS draft for ${document.recording.id} with ${document.rallies.length} rallies.`
          : `Loaded ${document.recording.id} and its matching NAS proxy. No local file selection needed.`,
      );
    } catch (loadError) {
      if (!controller.signal.aborted) {
        setError(loadError instanceof Error ? loadError.message : "Could not load pilot task");
      }
    } finally {
      if (!controller.signal.aborted) setPreparedTasksLoading(false);
    }
  }

  async function saveDraftDirectly() {
    if (!labels) return;
    const preparedTask = preparedTasks.find((task) => task.id === labels.recording.id);
    if (!preparedTask) {
      setError("Direct save is available only for a prepared pilot task.");
      return;
    }
    const draft: LabelDocument = {
      ...labels,
      annotation: {
        ...labels.annotation,
        status: "in-progress",
        reviewedAt: null,
      },
    };
    setError(null);
    setSavingDraft(true);
    try {
      const response = await fetch(
        `/api/labeling/tasks/${encodeURIComponent(preparedTask.id)}/draft`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(draft),
        },
      );
      const result = (await response.json()) as { error?: string; savedAt?: string };
      if (!response.ok || !result.savedAt) {
        throw new Error(result.error ?? "The draft could not be saved");
      }
      setLabels(draft);
      setLastSavedAt(result.savedAt);
      setMessage(
        `Draft saved directly to the NAS at ${new Date(result.savedAt).toLocaleTimeString()}.`,
      );
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "The draft could not be saved");
    } finally {
      setSavingDraft(false);
    }
  }

  function seek(seconds: number) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.min(video.duration || Infinity, Math.max(0, video.currentTime + seconds));
    setCurrentTime(video.currentTime);
  }

  function seekTo(seconds: number) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, Math.min(video.duration || Infinity, seconds));
    setCurrentTime(video.currentTime);
    video.focus();
  }

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play();
    else video.pause();
  }

  function beginRally() {
    if (!labels || !videoRef.current) return;
    setRallyStart(roundTime(videoRef.current.currentTime));
    setMessage("Rally start marked. Seek to the first instant live play has ended, then press E.");
  }

  function finishRally() {
    if (!labels || rallyStart === null || !videoRef.current) return;
    const end = roundTime(videoRef.current.currentTime);
    if (!addInterval(rallyStart, end, "rally")) return;
    setRallyStart(null);
  }

  function toggleIgnored() {
    if (!labels || !videoRef.current) return;
    const time = roundTime(videoRef.current.currentTime);
    if (ignoredStart === null) {
      setIgnoredStart(time);
      setMessage("Ignore start marked. Seek to the end of the ambiguous/censored span and press ].");
      return;
    }
    if (addInterval(ignoredStart, time, "ignored")) setIgnoredStart(null);
  }

  function toggleNegative() {
    if (!labels || !videoRef.current) return;
    const time = roundTime(videoRef.current.currentTime);
    if (negativeStart === null) {
      setNegativeStart(time);
      setMessage("Hard-negative start marked. Seek to its end and press H again.");
      return;
    }
    if (addInterval(negativeStart, time, "negative")) setNegativeStart(null);
  }

  function addInterval(start: number, end: number, kind: IntervalKind): boolean {
    if (!labels) return false;
    setError(null);
    if (end <= start) {
      setError("Interval end must be after its start.");
      return false;
    }
    if (overlaps(start, end, allRows)) {
      setError("That interval overlaps an existing rally, ignored span, or hard negative.");
      return false;
    }
    const sortRows = <T extends { start: number }>(rows: T[]) =>
      [...rows].sort((left, right) => left.start - right.start);
    if (kind === "rally") {
      const rally: RallyLabel = { start, end, tags: [] };
      setLabels(markChanged({ ...labels, rallies: sortRows([...labels.rallies, rally]) }));
      setMessage(`Added rally ${formatPreciseTime(start)}–${formatPreciseTime(end)}.`);
    } else if (kind === "ignored") {
      const ignored: IgnoredInterval = { start, end, reason: "partial-rally" };
      setLabels(
        markChanged({ ...labels, ignoredIntervals: sortRows([...labels.ignoredIntervals, ignored]) }),
      );
      setMessage("Added an ignored interval. These samples will not be fitted or scored.");
    } else {
      const negative: HardNegative = { start, end, category: negativeCategory };
      setLabels(
        markChanged({ ...labels, hardNegatives: sortRows([...labels.hardNegatives, negative]) }),
      );
      setMessage("Added an optional hard-negative example.");
    }
    return true;
  }

  function cancelMarker() {
    setRallyStart(null);
    setIgnoredStart(null);
    setNegativeStart(null);
    setMessage("Open marker cancelled.");
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, textarea, select")) return;
      const key = event.key.toLowerCase();
      if (key === " ") {
        event.preventDefault();
        togglePlayback();
      } else if (key === "s") beginRally();
      else if (key === "e") finishRally();
      else if (key === "[") toggleIgnored();
      else if (key === "]" && ignoredStart !== null) toggleIgnored();
      else if (key === "h") toggleNegative();
      else if (key === "escape") cancelMarker();
      else if (key === "j") seek(event.shiftKey ? -1 : -0.1);
      else if (key === "k") seek(event.shiftKey ? 1 : 0.1);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  function updateRally(index: number, patch: Partial<RallyLabel>) {
    if (!labels) return;
    const rallies = labels.rallies.map((row, rowIndex) =>
      rowIndex === index ? { ...row, ...patch } : row,
    );
    setLabels(markChanged({ ...labels, rallies }));
  }

  function updateIgnored(index: number, patch: Partial<IgnoredInterval>) {
    if (!labels) return;
    const ignoredIntervals = labels.ignoredIntervals.map((row, rowIndex) =>
      rowIndex === index ? { ...row, ...patch } : row,
    );
    setLabels(markChanged({ ...labels, ignoredIntervals }));
  }

  function updateNegative(index: number, patch: Partial<HardNegative>) {
    if (!labels) return;
    const hardNegatives = labels.hardNegatives.map((row, rowIndex) =>
      rowIndex === index ? { ...row, ...patch } : row,
    );
    setLabels(markChanged({ ...labels, hardNegatives }));
  }

  function removeRow(kind: IntervalKind, index: number) {
    if (!labels) return;
    if (kind === "rally") {
      setLabels(markChanged({ ...labels, rallies: labels.rallies.filter((_, row) => row !== index) }));
    } else if (kind === "ignored") {
      setLabels(
        markChanged({
          ...labels,
          ignoredIntervals: labels.ignoredIntervals.filter((_, row) => row !== index),
        }),
      );
    } else {
      setLabels(
        markChanged({
          ...labels,
          hardNegatives: labels.hardNegatives.filter((_, row) => row !== index),
        }),
      );
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <Link href="/" className={styles.brand}>VOLLEYCUT <span>LABEL</span></Link>
        <div className={styles.local}>Local workspace · no cloud upload</div>
      </header>

      <section className={styles.intro}>
        <div>
          <p className={styles.eyebrow}>GOLD-LABEL WORKSTATION</p>
          <h1>Mark the ball live.<br /><em>Teach the cut.</em></h1>
        </div>
        <div className={styles.policy}>
          <strong>Boundary contract</strong>
          <p><b>Start:</b> the instant the server contacts the ball.</p>
          <p><b>End:</b> the first instant live play has ended—not the celebration or walk back.</p>
          <p>Short aces and service faults still count as rallies. Mark uncertain/partial footage as ignored.</p>
        </div>
      </section>

      <section className={styles.loaders}>
        <label className={styles.preparedTask}>
          <span>Prepared pilot task · recommended</span>
          <select
            value={selectedPreparedTask}
            disabled={preparedTasksLoading || preparedTasks.length === 0}
            onChange={(event) => void loadPreparedTask(event.target.value)}
          >
            <option value="">
              {preparedTasksLoading ? "Loading pilot tasks…" : "Choose a task and video…"}
            </option>
            {preparedTasks.map((task) => (
              <option key={task.id} value={task.id}>
                {task.priority}. {task.environment} · {task.originalFilename} · {formatPreciseTime(task.durationSeconds)}
              </option>
            ))}
          </select>
          <small>Loads both files directly from the prepared NAS workspace.</small>
        </label>
        <label>
          <span>Local fallback · task or saved draft</span>
          <input type="file" accept="application/json,.json" onChange={(event) => void loadTask(event.target.files?.[0])} />
        </label>
        <label>
          <span>Local fallback · matching proxy</span>
          <input type="file" accept="video/mp4,video/*" onChange={(event) => loadVideo(event.target.files?.[0])} />
        </label>
        <div className={styles.loaded}>
          <span>Original source</span>
          <strong>{selectedPreparedSummary?.originalFilename ?? "Local task or draft"}</strong>
          <span>Annotation proxy</span>
          <strong>{labels?.recording.videoFilename ?? "Load a task first"}</strong>
        </div>
      </section>

      {error && <div className={styles.error}>{error}</div>}
      <div className={styles.message}>{message}</div>

      <section className={styles.workbench}>
        <div className={styles.videoColumn}>
          <div className={styles.videoWrap}>
            {videoUrl ? (
              <video
                key={videoUrl}
                ref={videoRef}
                src={videoUrl}
                controls
                preload="metadata"
                onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                onLoadedMetadata={(event) => {
                  setVideoDuration(event.currentTarget.duration);
                  setCurrentTime(event.currentTarget.currentTime);
                }}
              />
            ) : (
              <div className={styles.videoEmpty}>Select the proxy listed by the task.</div>
            )}
            <div className={styles.timecode}>{formatPreciseTime(currentTime)}</div>
          </div>

          <div className={styles.transport}>
            <button onClick={() => seek(-1)}>−1s</button>
            <button onClick={() => seek(-0.1)}>−0.1s <kbd>J</kbd></button>
            <button className={styles.playButton} onClick={togglePlayback}>Play / pause <kbd>Space</kbd></button>
            <button onClick={() => seek(0.1)}>+0.1s <kbd>K</kbd></button>
            <button onClick={() => seek(1)}>+1s</button>
          </div>

          <div className={styles.markers}>
            <button className={styles.start} onClick={beginRally} disabled={!labels || !videoUrl || rallyStart !== null}>
              Mark serve contact <kbd>S</kbd>
            </button>
            <button className={styles.end} onClick={finishRally} disabled={rallyStart === null}>
              Mark end of play <kbd>E</kbd>
            </button>
            <button onClick={toggleIgnored} disabled={!labels || !videoUrl}>
              {ignoredStart === null ? "Start ignored span" : "Finish ignored span"} <kbd>[ ]</kbd>
            </button>
            <button onClick={cancelMarker} disabled={rallyStart === null && ignoredStart === null && negativeStart === null}>
              Cancel <kbd>Esc</kbd>
            </button>
          </div>

          {labels && (
            <div className={styles.timeline} aria-label="Label timeline">
              {labels.rallies.map((row, index) => (
                <button
                  key={`rally-${index}`}
                  className={styles.rallyBar}
                  style={{ left: `${(row.start / labels.recording.durationSeconds) * 100}%`, width: `${((row.end - row.start) / labels.recording.durationSeconds) * 100}%` }}
                  onClick={() => seekTo(row.start)}
                  title={`Rally ${index + 1}`}
                />
              ))}
              {labels.ignoredIntervals.map((row, index) => (
                <button
                  key={`ignored-${index}`}
                  className={styles.ignoredBar}
                  style={{ left: `${(row.start / labels.recording.durationSeconds) * 100}%`, width: `${((row.end - row.start) / labels.recording.durationSeconds) * 100}%` }}
                  onClick={() => seekTo(row.start)}
                  title={`Ignored ${index + 1}`}
                />
              ))}
            </div>
          )}
        </div>

        <aside className={styles.metadata}>
          <p className={styles.eyebrow}>TASK METADATA</p>
          <strong className={styles.taskId}>{labels?.recording.id ?? "No task"}</strong>
          {labels && (
            <>
              <dl>
                <div><dt>Environment</dt><dd>{labels.recording.environment}</dd></div>
                <div><dt>Split</dt><dd>{labels.recording.split}</dd></div>
                <div><dt>Duration</dt><dd>{formatPreciseTime(labels.recording.durationSeconds)}</dd></div>
                <div><dt>Rallies</dt><dd>{labels.rallies.length}</dd></div>
                <div><dt>Live time</dt><dd>{formatPreciseTime(totalSeconds(labels.rallies))}</dd></div>
              </dl>
              <label>Annotator
                <input value={labels.annotation.annotator} onChange={(event) => setLabels(markChanged({ ...labels, annotation: { ...labels.annotation, annotator: event.target.value } }))} />
              </label>
              <label>Players per team
                <select value={labels.recording.game.playersPerTeam ?? ""} onChange={(event) => setLabels(markChanged({ ...labels, recording: { ...labels.recording, game: { ...labels.recording.game, playersPerTeam: event.target.value ? Number(event.target.value) : null } } }))}>
                  <option value="">Confirm…</option>
                  {[1, 2, 3, 4, 5, 6].map((count) => <option key={count} value={count}>{count}</option>)}
                </select>
              </label>
              <label>Target points
                <input type="number" min="1" max="100" placeholder="e.g. 21" value={labels.recording.game.targetPoints ?? ""} onChange={(event) => setLabels(markChanged({ ...labels, recording: { ...labels.recording, game: { ...labels.recording.game, targetPoints: event.target.value ? Number(event.target.value) : null } } }))} />
              </label>
              <label>Format / scoring notes
                <input value={labels.recording.game.format ?? ""} onChange={(event) => setLabels(markChanged({ ...labels, recording: { ...labels.recording, game: { ...labels.recording.game, format: event.target.value || null } } }))} />
              </label>
              <label>Annotation notes
                <textarea rows={3} value={labels.annotation.notes} onChange={(event) => setLabels(markChanged({ ...labels, annotation: { ...labels.annotation, notes: event.target.value } }))} />
              </label>
              <label className={styles.reviewed}>
                <input type="checkbox" checked={labels.annotation.continuousVideoReviewed} onChange={(event) => setLabels(markChanged({ ...labels, annotation: { ...labels.annotation, continuousVideoReviewed: event.target.checked } }))} />
                I reviewed the entire continuous video, including short service faults.
              </label>
            </>
          )}
        </aside>
      </section>

      {labels && (
        <section className={styles.tables}>
          <div className={styles.tableHeading}>
            <div><p className={styles.eyebrow}>REQUIRED</p><h2>Rally intervals</h2></div>
            <span>{labels.rallies.length} rallies · click a time to seek</span>
          </div>
          <div className={styles.rows}>
            {labels.rallies.map((row, index) => (
              <div className={styles.row} key={`rally-row-${index}`}>
                <strong>R{String(index + 1).padStart(3, "0")}</strong>
                <button onClick={() => seekTo(row.start)}>{formatPreciseTime(row.start)}</button>
                <span>→</span>
                <button onClick={() => seekTo(row.end)}>{formatPreciseTime(row.end)}</button>
                <input aria-label="Rally start seconds" type="number" step="0.001" value={row.start} onChange={(event) => updateRally(index, { start: Number(event.target.value) })} />
                <input aria-label="Rally end seconds" type="number" step="0.001" value={row.end} onChange={(event) => updateRally(index, { end: Number(event.target.value) })} />
                <select aria-label="Rally tag" value={row.tags[0] ?? ""} onChange={(event) => updateRally(index, { tags: event.target.value ? [event.target.value] : [] })}>
                  <option value="">Normal rally</option>
                  <option value="service-fault">Service fault</option>
                  <option value="ace">Ace / very short</option>
                  <option value="interrupted-replay">Interrupted / replayed</option>
                </select>
                <button className={styles.delete} onClick={() => removeRow("rally", index)}>Delete</button>
              </div>
            ))}
            {labels.rallies.length === 0 && <p className={styles.empty}>No rallies yet. Play to serve contact and press S.</p>}
          </div>

          <div className={styles.secondaryGrid}>
            <div>
              <div className={styles.tableHeading}><div><p className={styles.eyebrow}>WHEN NEEDED</p><h2>Ignored spans</h2></div></div>
              <p className={styles.help}>Use for a partial rally at a file edge, camera gap, or truly unresolvable boundary. Do not use for ordinary dead time.</p>
              {labels.ignoredIntervals.map((row, index) => (
                <div className={styles.smallRow} key={`ignored-row-${index}`}>
                  <button onClick={() => seekTo(row.start)}>{formatPreciseTime(row.start)}–{formatPreciseTime(row.end)}</button>
                  <select value={row.reason} onChange={(event) => updateIgnored(index, { reason: event.target.value })}>
                    <option value="partial-rally">Partial rally</option>
                    <option value="camera-gap">Camera gap</option>
                    <option value="boundary-ambiguous">Boundary ambiguous</option>
                    <option value="non-game-content">Non-game content</option>
                  </select>
                  <button className={styles.delete} onClick={() => removeRow("ignored", index)}>Delete</button>
                </div>
              ))}
            </div>

            <div>
              <div className={styles.tableHeading}><div><p className={styles.eyebrow}>OPTIONAL · 3–5 PER VIDEO</p><h2>Hard negatives</h2></div></div>
              <p className={styles.help}>Tag confusing dead-time examples; ordinary between-point time is already negative.</p>
              <div className={styles.negativeMarker}>
                <select value={negativeCategory} onChange={(event) => setNegativeCategory(event.target.value)}>
                  <option value="foreground-crossing">Foreground crossing</option>
                  <option value="adjacent-court">Adjacent-court play</option>
                  <option value="celebration">Celebration</option>
                  <option value="setup-between-points">Setup between points</option>
                  <option value="timeout">Timeout</option>
                  <option value="camera-motion">Camera motion</option>
                  <option value="warmup">Warmup</option>
                  <option value="other">Other</option>
                </select>
                <button onClick={toggleNegative}>{negativeStart === null ? "Start hard negative" : "Finish hard negative"} <kbd>H</kbd></button>
              </div>
              {labels.hardNegatives.map((row, index) => (
                <div className={styles.smallRow} key={`negative-row-${index}`}>
                  <button onClick={() => seekTo(row.start)}>{formatPreciseTime(row.start)}–{formatPreciseTime(row.end)}</button>
                  <select value={row.category} onChange={(event) => updateNegative(index, { category: event.target.value })}>
                    <option value="foreground-crossing">Foreground crossing</option>
                    <option value="adjacent-court">Adjacent court</option>
                    <option value="celebration">Celebration</option>
                    <option value="setup-between-points">Point setup</option>
                    <option value="timeout">Timeout</option>
                    <option value="camera-motion">Camera motion</option>
                    <option value="warmup">Warmup</option>
                    <option value="other">Other</option>
                  </select>
                  <button className={styles.delete} onClick={() => removeRow("negative", index)}>Delete</button>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.exportPanel}>
            <div>
              <p className={styles.eyebrow}>SAVE OFTEN</p>
              <h2>Save your progress directly.</h2>
              <p>
                Prepared-task drafts save directly to the NAS and resume from the selector.
                Downloads remain available as backups. Completed files are validated before training.
              </p>
              {lastSavedAt && <p>Last direct save: {new Date(lastSavedAt).toLocaleString()}</p>}
            </div>
            <div className={styles.issueList}>
              {completionIssues.length > 0 ? completionIssues.map((issue) => <span key={issue}>• {issue}</span>) : <strong>Ready to export complete labels.</strong>}
            </div>
            <div className={styles.exportButtons}>
              <button
                className={styles.directSave}
                disabled={savingDraft || !preparedTasks.some((task) => task.id === labels.recording.id)}
                onClick={() => void saveDraftDirectly()}
              >
                {savingDraft ? "Saving…" : "Save draft to NAS"}
              </button>
              <button onClick={() => downloadLabels(labels, false)}>Download backup JSON</button>
              <button className={styles.complete} disabled={completionIssues.length > 0} onClick={() => downloadLabels(labels, true)}>Export completed labels</button>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
