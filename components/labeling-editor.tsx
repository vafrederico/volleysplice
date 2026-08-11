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
  type SideSwitch,
} from "@/lib/annotations";
import styles from "./labeling-editor.module.css";

type IntervalKind = "rally" | "ignored" | "negative";
type LabelingBatch = "full" | "pilot";

type PreparedTaskSummary = {
  id: string;
  batch: LabelingBatch;
  priority: number;
  environment: LabelDocument["recording"]["environment"];
  split: LabelDocument["recording"]["split"];
  durationSeconds: number;
  originalFilename: string;
  videoFilename: string;
  documentSource: "draft" | "prelabel" | "task";
  savedAt: string | null;
  annotationStatus: LabelDocument["annotation"]["status"];
  rallyCount: number;
};

type BatchSummary = Record<
  LabelingBatch,
  { ready: number; total: number; saved: number; prelabeled: number }
>;

const emptyBatchSummary: BatchSummary = {
  full: { ready: 0, total: 0, saved: 0, prelabeled: 0 },
  pilot: { ready: 0, total: 0, saved: 0, prelabeled: 0 },
};

const editableRallyTags = new Set(["service-fault", "ace", "interrupted-replay"]);
const playbackResumeKey = "volleycut.labeling.playback.v1";

type PlaybackResume = {
  version: 1;
  taskId: string;
  time: number;
};

function readPlaybackResume(): PlaybackResume | null {
  try {
    const raw = window.localStorage.getItem(playbackResumeKey);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<PlaybackResume>;
    if (
      value.version !== 1 ||
      typeof value.taskId !== "string" ||
      !/^[A-Za-z0-9_-]+$/.test(value.taskId) ||
      typeof value.time !== "number" ||
      !Number.isFinite(value.time) ||
      value.time < 0
    ) {
      window.localStorage.removeItem(playbackResumeKey);
      return null;
    }
    return value as PlaybackResume;
  } catch {
    try {
      window.localStorage.removeItem(playbackResumeKey);
    } catch {
      // Storage can be unavailable in privacy-restricted browser contexts.
    }
    return null;
  }
}

function overlaps(start: number, end: number, rows: Array<{ start: number; end: number }>): boolean {
  return rows.some((row) => start < row.end && row.start < end);
}

function totalSeconds(rows: Array<{ start: number; end: number }>): number {
  return rows.reduce((total, row) => total + row.end - row.start, 0);
}

export function LabelingEditor() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const preparedRequestRef = useRef<AbortController | null>(null);
  const pendingResumeSecondsRef = useRef<number | null>(null);
  const resumeAttemptedRef = useRef(false);
  const lastPersistedPlaybackRef = useRef<{ taskId: string; time: number } | null>(null);
  const [labels, setLabels] = useState<LabelDocument | null>(null);
  const [preparedTasks, setPreparedTasks] = useState<PreparedTaskSummary[]>([]);
  const [batchSummary, setBatchSummary] = useState<BatchSummary>(emptyBatchSummary);
  const [selectedBatch, setSelectedBatch] = useState<LabelingBatch>("full");
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
    "Choose a prepared full-corpus task, or use the local fallback files.",
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => {
    const controller = new AbortController();
    let firstLoad = true;
    async function loadPreparedTasks() {
      try {
        const response = await fetch("/api/labeling/tasks", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Prepared labeling tasks are unavailable");
        const payload = (await response.json()) as {
          batches?: BatchSummary;
          tasks?: PreparedTaskSummary[];
        };
        if (!Array.isArray(payload.tasks) || !payload.batches) {
          throw new Error("Prepared task catalog is invalid");
        }
        setPreparedTasks(payload.tasks);
        setBatchSummary(payload.batches);
        if (!resumeAttemptedRef.current) {
          resumeAttemptedRef.current = true;
          const resume = readPlaybackResume();
          const resumedTask = resume
            ? payload.tasks.find((task) => task.id === resume.taskId)
            : undefined;
          if (resume && resumedTask) {
            setSelectedBatch(resumedTask.batch);
            void loadPreparedTask(resume.taskId, resume.time);
          }
        }
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(loadError instanceof Error ? loadError.message : "Could not list labeling tasks");
        }
      } finally {
        if (!controller.signal.aborted && firstLoad) {
          setPreparedTasksLoading(false);
          firstLoad = false;
        }
      }
    }
    void loadPreparedTasks();
    const refresh = window.setInterval(() => void loadPreparedTasks(), 15_000);
    return () => {
      window.clearInterval(refresh);
      controller.abort();
    };
  }, []);

  const allRows = useMemo(() => {
    if (!labels) return [];
    return [...labels.rallies, ...labels.ignoredIntervals, ...labels.hardNegatives];
  }, [labels]);

  const selectedRallyIndex = useMemo(
    () =>
      labels?.rallies.findIndex(
        (row) => row.start < currentTime && currentTime < row.end,
      ) ?? -1,
    [currentTime, labels],
  );

  const previousRallyIndex = useMemo(() => {
    if (!labels) return -1;
    let previous = -1;
    labels.rallies.forEach((row, index) => {
      if (row.end <= currentTime) previous = index;
    });
    return previous;
  }, [currentTime, labels]);

  const playheadInsideRally = selectedRallyIndex >= 0;

  const selectedPreparedSummary = useMemo(
    () => preparedTasks.find((task) => task.id === selectedPreparedTask) ?? null,
    [preparedTasks, selectedPreparedTask],
  );

  const tasksForSelectedBatch = useMemo(
    () => preparedTasks.filter((task) => task.batch === selectedBatch),
    [preparedTasks, selectedBatch],
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
    labels.sideSwitches.forEach((marker, index) => {
      if (
        !Number.isFinite(marker.time) ||
        marker.time < 0 ||
        marker.time > labels.recording.durationSeconds ||
        (index > 0 && marker.time <= labels.sideSwitches[index - 1].time)
      ) {
        issues.push("Side-switch points must be in range and strictly ordered");
      }
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
    pendingResumeSecondsRef.current = null;
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
    pendingResumeSecondsRef.current = null;
    setError(null);
    const nextUrl = URL.createObjectURL(file);
    setVideoUrl(nextUrl);
    setVideoFilename(file.name);
    setVideoDuration(null);
    setMessage(`Loaded local video ${file.name}. Nothing is uploaded.`);
  }

  async function loadPreparedTask(id: string, resumeSeconds: number | null = null) {
    setSelectedPreparedTask(id);
    if (!id) return;
    pendingResumeSecondsRef.current =
      resumeSeconds !== null && Number.isFinite(resumeSeconds) && resumeSeconds >= 0
        ? resumeSeconds
        : null;
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
      if (!response.ok) throw new Error("Could not load the selected prepared task");
      const batch = response.headers.get("X-VolleyCut-Batch");
      const documentSource = response.headers.get("X-VolleyCut-Document-Source");
      const savedAt = response.headers.get("X-VolleyCut-Saved-At");
      const document = parseLabelDocument(await response.json());
      setLabels(document);
      setVideoUrl(`/api/labeling/tasks/${encodeURIComponent(id)}/video`);
      setVideoFilename(document.recording.videoFilename);
      setVideoDuration(null);
      setCurrentTime(pendingResumeSecondsRef.current ?? 0);
      setRallyStart(null);
      setIgnoredStart(null);
      setNegativeStart(null);
      setLastSavedAt(savedAt);
      if (batch === "full" || batch === "pilot") setSelectedBatch(batch);
      setMessage(
        documentSource === "draft"
          ? `Resumed the NAS draft for ${document.recording.id} with ${document.rallies.length} rallies.`
          : documentSource === "prelabel"
            ? `Loaded ${document.rallies.length} unvalidated GPT-5.6 Sol rally candidates for ${document.recording.id}. Review every boundary before completing.`
          : `Loaded ${document.recording.id} and its matching NAS proxy. No local file selection needed.`,
      );
    } catch (loadError) {
      if (!controller.signal.aborted) {
        pendingResumeSecondsRef.current = null;
        setError(loadError instanceof Error ? loadError.message : "Could not load prepared task");
      }
    } finally {
      if (!controller.signal.aborted) setPreparedTasksLoading(false);
    }
  }

  function persistPlaybackPosition(video: HTMLVideoElement, force = false) {
    if (
      !selectedPreparedTask ||
      labels?.recording.id !== selectedPreparedTask ||
      !videoUrl?.startsWith("/api/labeling/tasks/") ||
      !Number.isFinite(video.currentTime) ||
      video.currentTime < 0
    ) {
      return;
    }
    const previous = lastPersistedPlaybackRef.current;
    if (
      !force &&
      previous?.taskId === selectedPreparedTask &&
      Math.abs(previous.time - video.currentTime) < 0.25
    ) {
      return;
    }
    const resume: PlaybackResume = {
      version: 1,
      taskId: selectedPreparedTask,
      time: roundTime(video.currentTime),
    };
    try {
      window.localStorage.setItem(playbackResumeKey, JSON.stringify(resume));
      lastPersistedPlaybackRef.current = resume;
    } catch {
      // Browsers can deny local storage; labeling and NAS draft saves still work.
    }
  }

  function handleLoadedMetadata(video: HTMLVideoElement) {
    setVideoDuration(video.duration);
    const resumeSeconds = pendingResumeSecondsRef.current;
    pendingResumeSecondsRef.current = null;
    if (resumeSeconds !== null) {
      video.currentTime = Math.max(0, Math.min(video.duration || Infinity, resumeSeconds));
      setMessage(
        `Resumed ${labels?.recording.id ?? "prepared task"} at ${formatPreciseTime(video.currentTime)}.`,
      );
    }
    setCurrentTime(video.currentTime);
    persistPlaybackPosition(video, true);
  }

  async function saveDraftDirectly() {
    if (!labels) return;
    const preparedTask = preparedTasks.find((task) => task.id === labels.recording.id);
    if (!preparedTask) {
      setError("Direct save is available only for a prepared NAS task.");
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
      const result = (await response.json()) as {
        batch?: LabelingBatch;
        error?: string;
        savedAt?: string;
      };
      if (!response.ok || !result.savedAt) {
        throw new Error(result.error ?? "The draft could not be saved");
      }
      setLabels(draft);
      setLastSavedAt(result.savedAt);
      setPreparedTasks((current) =>
        current.map((task) =>
          task.id === preparedTask.id
            ? {
                ...task,
                annotationStatus: "in-progress",
                documentSource: "draft",
                rallyCount: draft.rallies.length,
                savedAt: result.savedAt ?? null,
              }
            : task,
        ),
      );
      if (!preparedTask.savedAt) {
        setBatchSummary((current) => ({
          ...current,
          [preparedTask.batch]: {
            ...current[preparedTask.batch],
            saved: current[preparedTask.batch].saved + 1,
          },
        }));
      }
      setMessage(
        `${preparedTask.batch === "full" ? "Full-corpus" : "Pilot"} draft saved directly to the NAS at ${new Date(result.savedAt).toLocaleTimeString()}.`,
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
    persistPlaybackPosition(video, true);
  }

  function seekTo(seconds: number) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, Math.min(video.duration || Infinity, seconds));
    setCurrentTime(video.currentTime);
    persistPlaybackPosition(video, true);
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
    const time = roundTime(videoRef.current.currentTime);
    const existingIndex = labels.rallies.findIndex(
      (row) => row.start < time && time < row.end,
    );
    if (existingIndex >= 0) {
      updateRally(existingIndex, { start: time });
      setMessage(
        `Moved rally ${existingIndex + 1} start to ${formatPreciseTime(time)}.`,
      );
      return;
    }
    setRallyStart(time);
    setMessage("Rally start marked. Seek to the first instant live play has ended, then press E.");
  }

  function moveRallyEnd(index: number, end: number): boolean {
    if (!labels) return false;
    const rally = labels.rallies[index];
    if (!rally || end <= rally.start) {
      setError("Rally end must be after its serve contact.");
      return false;
    }
    if (end > labels.recording.durationSeconds) {
      setError("Rally end cannot exceed the video duration.");
      return false;
    }
    const otherRallies = labels.rallies.filter((_, rowIndex) => rowIndex !== index);
    if (
      overlaps(rally.start, end, otherRallies) ||
      overlaps(rally.start, end, labels.ignoredIntervals) ||
      overlaps(rally.start, end, labels.hardNegatives)
    ) {
      setError(
        "That end would overlap the next rally, an ignored span, or a hard negative.",
      );
      return false;
    }
    setError(null);
    updateRally(index, { end });
    setMessage(`Moved rally ${index + 1} end to ${formatPreciseTime(end)}.`);
    return true;
  }

  function splitRallyFromOpenStart(index: number, start: number, split: number): boolean {
    if (!labels) return false;
    const existing = labels.rallies[index];
    if (!existing || split <= start || split >= existing.end) {
      setError("The split must be after the new start and before the existing rally end.");
      return false;
    }
    const otherRallies = labels.rallies.filter((_, rowIndex) => rowIndex !== index);
    if (
      overlaps(start, existing.end, otherRallies) ||
      overlaps(start, existing.end, labels.ignoredIntervals) ||
      overlaps(start, existing.end, labels.hardNegatives)
    ) {
      setError(
        "The split rally would overlap another rally, an ignored span, or a hard negative.",
      );
      return false;
    }
    const first: RallyLabel = { ...existing, start, end: split };
    const remainder: RallyLabel = { ...existing, start: split, end: existing.end };
    const rallies = [...otherRallies, first, remainder].sort(
      (left, right) => left.start - right.start,
    );
    setError(null);
    setLabels(markChanged({ ...labels, rallies }));
    setRallyStart(null);
    setMessage(
      `Split rally ${index + 1} at ${formatPreciseTime(split)}: the first part now starts at ${formatPreciseTime(start)}, and the remainder ends at ${formatPreciseTime(existing.end)}.`,
    );
    return true;
  }

  function finishRally() {
    if (!labels || !videoRef.current) return;
    const end = roundTime(videoRef.current.currentTime);
    if (rallyStart === null) {
      if (selectedRallyIndex >= 0) {
        moveRallyEnd(selectedRallyIndex, end);
        return;
      }
      if (previousRallyIndex >= 0) moveRallyEnd(previousRallyIndex, end);
      return;
    }
    const containingRallyIndex = labels.rallies.findIndex(
      (row) => row.start < end && end < row.end,
    );
    if (containingRallyIndex >= 0) {
      splitRallyFromOpenStart(containingRallyIndex, rallyStart, end);
      return;
    }
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

  function addSideSwitch() {
    if (!labels || !videoRef.current) return;
    const time = roundTime(videoRef.current.currentTime);
    if (labels.sideSwitches.some((marker) => marker.time === time)) {
      setError("A side switch is already marked at this timestamp.");
      return;
    }
    if (time > labels.recording.durationSeconds) {
      setError("A side switch cannot be marked beyond the task duration.");
      return;
    }
    const sideSwitches = [...labels.sideSwitches, { time }].sort(
      (left, right) => left.time - right.time,
    );
    setError(null);
    setLabels(markChanged({ ...labels, sideSwitches }));
    setMessage(`Marked a side switch at ${formatPreciseTime(time)}.`);
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
      if (target?.closest("input, textarea, select, button, [contenteditable='true']")) return;
      const key = event.key.toLowerCase();
      if (key === " ") {
        event.preventDefault();
        togglePlayback();
      } else if (key === "s") beginRally();
      else if (key === "e") finishRally();
      else if (key === "[") toggleIgnored();
      else if (key === "]" && ignoredStart !== null) toggleIgnored();
      else if (key === "h") toggleNegative();
      else if (key === "x") addSideSwitch();
      else if (key === "escape") cancelMarker();
      else if ((key === "delete" || key === "backspace") && selectedRallyIndex >= 0) {
        event.preventDefault();
        removeSelectedRally();
      }
      else if (key === "j") seek(event.shiftKey ? -1 : -0.1);
      else if (key === "k") seek(event.shiftKey ? 1 : 0.1);
      else if (key === "arrowleft") {
        event.preventDefault();
        seek(-1);
      } else if (key === "arrowright") {
        event.preventDefault();
        seek(1);
      }
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

  function updateSideSwitch(index: number, patch: Partial<SideSwitch>) {
    if (!labels) return;
    const sideSwitches = labels.sideSwitches.map((marker, markerIndex) =>
      markerIndex === index ? { ...marker, ...patch } : marker,
    );
    setLabels(markChanged({ ...labels, sideSwitches }));
  }

  function removeSideSwitch(index: number) {
    if (!labels) return;
    setLabels(
      markChanged({
        ...labels,
        sideSwitches: labels.sideSwitches.filter((_, markerIndex) => markerIndex !== index),
      }),
    );
    setMessage(`Deleted side-switch marker ${index + 1}.`);
  }

  function removeSelectedRally() {
    if (!labels || selectedRallyIndex < 0) return;
    const rally = labels.rallies[selectedRallyIndex];
    setLabels(
      markChanged({
        ...labels,
        rallies: labels.rallies.filter((_, index) => index !== selectedRallyIndex),
      }),
    );
    setError(null);
    setMessage(
      `Deleted rally ${selectedRallyIndex + 1} (${formatPreciseTime(rally.start)}–${formatPreciseTime(rally.end)}).`,
    );
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
          <span>Prepared NAS task · recommended</span>
          <select
            aria-label="Labeling batch"
            value={selectedBatch}
            disabled={preparedTasksLoading}
            onChange={(event) => {
              setSelectedBatch(event.target.value as LabelingBatch);
              setSelectedPreparedTask("");
            }}
          >
            <option value="full">
              Full corpus · {batchSummary.full.ready}/{batchSummary.full.total} ready · {batchSummary.full.saved} saved
              {batchSummary.full.prelabeled > 0 ? ` · ${batchSummary.full.prelabeled} AI prelabels` : ""}
            </option>
            <option value="pilot">
              Pilot · {batchSummary.pilot.ready}/{batchSummary.pilot.total} ready · {batchSummary.pilot.saved} saved
            </option>
          </select>
          <select
            aria-label="Prepared labeling task"
            value={selectedPreparedTask}
            disabled={preparedTasksLoading || tasksForSelectedBatch.length === 0}
            onChange={(event) => void loadPreparedTask(event.target.value)}
          >
            <option value="">
              {preparedTasksLoading
                ? "Loading prepared tasks…"
                : tasksForSelectedBatch.length === 0
                  ? "Waiting for this batch to be prepared…"
                  : "Choose a task and video…"}
            </option>
            {tasksForSelectedBatch.map((task) => (
              <option key={task.id} value={task.id}>
                {task.priority}. {task.environment} · {task.originalFilename} · {formatPreciseTime(task.durationSeconds)} · {task.savedAt ? `${task.rallyCount} rallies saved` : task.documentSource === "prelabel" ? `${task.rallyCount} AI rallies to review` : "not started"}
              </option>
            ))}
          </select>
          <small>Loads both files from the NAS. The list refreshes as full proxies finish.</small>
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
          <span>Batch</span>
          <strong>{selectedPreparedSummary?.batch ?? "Local fallback"}</strong>
          <span>Starting point</span>
          <strong>
            {selectedPreparedSummary?.documentSource === "prelabel"
              ? `Unvalidated GPT-5.6 Sol prelabel · ${labels?.prelabel?.ambiguities.length ?? 0} ambiguities`
              : selectedPreparedSummary?.documentSource === "draft"
                ? labels?.prelabel
                  ? "Human-saved NAS draft · started from AI prelabel"
                  : "Human-saved NAS draft"
                : "Blank task"}
          </strong>
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
                onTimeUpdate={(event) => {
                  setCurrentTime(event.currentTarget.currentTime);
                  persistPlaybackPosition(event.currentTarget);
                }}
                onSeeked={(event) => persistPlaybackPosition(event.currentTarget, true)}
                onPlay={(event) => persistPlaybackPosition(event.currentTarget, true)}
                onPause={(event) => persistPlaybackPosition(event.currentTarget, true)}
                onLoadedMetadata={(event) => handleLoadedMetadata(event.currentTarget)}
              />
            ) : (
              <div className={styles.videoEmpty}>Select the proxy listed by the task.</div>
            )}
            <div className={styles.timecode}>{formatPreciseTime(currentTime)}</div>
          </div>

          <div className={styles.transport}>
            <button onClick={() => seek(-1)}>−1s <kbd>←</kbd></button>
            <button onClick={() => seek(-0.1)}>−0.1s <kbd>J</kbd></button>
            <button className={styles.playButton} onClick={togglePlayback}>Play / pause <kbd>Space</kbd></button>
            <button onClick={() => seek(0.1)}>+0.1s <kbd>K</kbd></button>
            <button onClick={() => seek(1)}>+1s <kbd>→</kbd></button>
          </div>

          <div className={styles.markers}>
            <button className={styles.start} onClick={beginRally} disabled={!labels || !videoUrl || rallyStart !== null}>
              {playheadInsideRally ? "Move rally start" : "Mark serve contact"} <kbd>S</kbd>
            </button>
            <button
              className={styles.end}
              onClick={finishRally}
              disabled={
                !labels ||
                !videoUrl ||
                (rallyStart === null && selectedRallyIndex < 0 && previousRallyIndex < 0)
              }
            >
              {rallyStart !== null
                ? selectedRallyIndex >= 0
                  ? "Split rally here"
                  : "Mark end of play"
                : selectedRallyIndex >= 0
                  ? "Move rally end"
                  : "Extend previous rally"} <kbd>E</kbd>
            </button>
            <button
              className={styles.deleteSelected}
              onClick={removeSelectedRally}
              disabled={selectedRallyIndex < 0}
            >
              Delete selected <kbd>Del</kbd>
            </button>
            <button onClick={toggleIgnored} disabled={!labels || !videoUrl}>
              {ignoredStart === null ? "Start ignored span" : "Finish ignored span"} <kbd>[ ]</kbd>
            </button>
            <button onClick={addSideSwitch} disabled={!labels || !videoUrl}>
              Mark side switch <kbd>X</kbd>
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
                  className={`${styles.rallyBar} ${selectedRallyIndex === index ? styles.selectedRallyBar : ""}`}
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
              {labels.sideSwitches.map((marker, index) => (
                <button
                  key={`side-switch-${index}`}
                  className={styles.sideSwitchPoint}
                  style={{ left: `${(marker.time / labels.recording.durationSeconds) * 100}%` }}
                  onClick={() => seekTo(marker.time)}
                  title={`Side switch ${index + 1}${marker.notes ? ` · ${marker.notes}` : ""}`}
                  aria-label={`Seek to side switch ${index + 1}`}
                />
              ))}
              <div
                className={styles.playhead}
                style={{
                  left: `${Math.min(
                    100,
                    Math.max(0, (currentTime / labels.recording.durationSeconds) * 100),
                  )}%`,
                }}
                aria-hidden="true"
              />
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
              <div
                className={`${styles.row} ${selectedRallyIndex === index ? styles.selectedRow : ""}`}
                key={`rally-row-${index}`}
                aria-current={selectedRallyIndex === index ? "true" : undefined}
              >
                <strong
                  title={[
                    ...row.tags.filter((tag) => tag.startsWith("ai-") || tag.includes("confidence:")),
                    ...(row.notes ? [row.notes] : []),
                  ].join(" · ")}
                >
                  R{String(index + 1).padStart(3, "0")}{row.tags.includes("ai-prelabel") ? " AI" : ""}
                </strong>
                <button onClick={() => seekTo(row.start)}>{formatPreciseTime(row.start)}</button>
                <span>→</span>
                <button onClick={() => seekTo(row.end)}>{formatPreciseTime(row.end)}</button>
                <input aria-label="Rally start seconds" type="number" step="0.001" value={row.start} onChange={(event) => updateRally(index, { start: Number(event.target.value) })} />
                <input aria-label="Rally end seconds" type="number" step="0.001" value={row.end} onChange={(event) => updateRally(index, { end: Number(event.target.value) })} />
                <select
                  className={selectedRallyIndex === index ? styles.selectedClassification : undefined}
                  aria-label="Rally tag"
                  value={row.tags.find((tag) => editableRallyTags.has(tag)) ?? ""}
                  onChange={(event) => updateRally(index, {
                    tags: [
                      ...row.tags.filter((tag) => !editableRallyTags.has(tag)),
                      ...(event.target.value ? [event.target.value] : []),
                    ],
                  })}
                >
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

          <div className={styles.pointSection}>
            <div className={styles.tableHeading}>
              <div><p className={styles.eyebrow}>OPTIONAL · WHEN PRESENT</p><h2>Side switches</h2></div>
              <span>{labels.sideSwitches.length} point markers · press X at the switch</span>
            </div>
            <p className={styles.help}>
              Mark the moment teams switch court sides when the recording format includes it. Add a note if the exact transition is obscured.
            </p>
            <div className={styles.pointRows}>
              {labels.sideSwitches.map((marker, index) => (
                <div className={styles.pointRow} key={`side-switch-row-${index}`}>
                  <strong>SW{String(index + 1).padStart(2, "0")}</strong>
                  <button onClick={() => seekTo(marker.time)}>{formatPreciseTime(marker.time)}</button>
                  <input
                    aria-label={`Side switch ${index + 1} seconds`}
                    type="number"
                    step="0.001"
                    value={marker.time}
                    onChange={(event) => updateSideSwitch(index, { time: Number(event.target.value) })}
                  />
                  <input
                    aria-label={`Side switch ${index + 1} notes`}
                    placeholder="Optional note"
                    value={marker.notes ?? ""}
                    onChange={(event) => updateSideSwitch(index, {
                      notes: event.target.value || undefined,
                    })}
                  />
                  <button className={styles.delete} onClick={() => removeSideSwitch(index)}>Delete</button>
                </div>
              ))}
              {labels.sideSwitches.length === 0 && (
                <p className={styles.empty}>No side switches marked for this video.</p>
              )}
            </div>
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
