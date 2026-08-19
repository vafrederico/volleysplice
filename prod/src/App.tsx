import { useEffect, useMemo, useRef, useState } from "react";

import { CutEditor } from "@/components/CutEditor";
import { GuidedTour } from "@/components/GuidedTour";
import { ProjectHeader } from "@/components/ProjectHeader";
import { isUnsupportedSafariBrowser } from "@/lib/on-device/browser-support";
import {
  MIN_ANALYSIS_WINDOW_SECONDS,
  normalizeAnalysisWindow,
  type AnalysisWindow,
} from "@/lib/on-device/analysis-window";
import { deleteFeatureCachesForSource } from "@/lib/on-device/feature-cache";
import { type OpenedMedia, openLocalMedia } from "@/lib/on-device/media";
import {
  analyzeOpenedMedia,
  DEFAULT_FEATURE_REDUCTION_KERNEL,
  DEFAULT_VIDEO_DECODE_STRATEGY,
  VIDEO_DECODER_HARDWARE_ACCELERATION,
} from "@/lib/on-device/pipeline";
import { augmentStoredAnalysisWithSuppression } from "@/lib/on-device/production-inference";
import { DEFAULT_ON_DEVICE_RUNTIME_VARIANT } from "@/lib/on-device/runtime-variants";
import type {
  AnalysisProgress,
  NormalizedRoi,
  OnDeviceAnalysis,
  OnDeviceMediaInfo,
} from "@/lib/on-device/types";
import {
  holdScreenWakeLock,
  type WakeLockState,
} from "@/lib/on-device/wake-lock";
import type { ProductAnalysis } from "@/lib/product-analysis";
import {
  deleteProject,
  listProjects,
  projectAnalysisId,
  projectId,
  projectSource,
  putProject,
  SELECTED_PROJECT_STORAGE_KEY,
  sourceCanReconnectFile,
  sourceFileFingerprint,
  type VolleyCutProject,
} from "@/lib/project-store";
import { cutDraftStorageKeys } from "@/lib/cut-draft";

import styles from "./App.module.css";

type WorkState = "empty" | "opening" | "ready" | "error";

const FULL_FRAME_ROI: NormalizedRoi = { x: 0, y: 0, width: 1, height: 1 };

function compactBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let index = 0;
  while (value >= 1000 && index < units.length - 1) {
    value /= 1000;
    index += 1;
  }
  return `${value.toFixed(index < 2 ? 0 : 1)} ${units[index]}`;
}

function formatDuration(seconds: number): string {
  const safe = Math.max(0, Math.round(seconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = safe % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function preciseTime(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = safe % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`
    : `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function progressPercent(progress: AnalysisProgress | null): number {
  if (!progress) return 0;
  const fraction = progress.total > 0 ? progress.completed / progress.total : 0;
  switch (progress.stage) {
    case "opening":
      return 2;
    case "video":
      return Math.min(82, Math.max(3, fraction * 82));
    case "audio":
      return 82 + Math.min(10, Math.max(0, fraction * 10));
    case "normalizing":
      return 94;
    case "inference":
      return 98;
    case "complete":
      return 100;
  }
}

function clampRoi(roi: NormalizedRoi): NormalizedRoi {
  const x = Math.max(0, Math.min(0.99, roi.x));
  const y = Math.max(0, Math.min(0.99, roi.y));
  return {
    x,
    y,
    width: Math.max(0.01, Math.min(1 - x, roi.width)),
    height: Math.max(0.01, Math.min(1 - y, roi.height)),
  };
}

function sortProjects(projects: VolleyCutProject[]): VolleyCutProject[] {
  return [...projects].sort((left, right) =>
    right.updatedAt.localeCompare(left.updatedAt),
  );
}

export function App() {
  const [projects, setProjects] = useState<VolleyCutProject[]>([]);
  const [projectsLoaded, setProjectsLoaded] = useState(false);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(
    null,
  );
  const [queueIds, setQueueIds] = useState<string[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeProgress, setActiveProgress] = useState<AnalysisProgress | null>(
    null,
  );
  const [activeElapsedSeconds, setActiveElapsedSeconds] = useState(0);
  const [wakeLockState, setWakeLockState] = useState<WakeLockState>("idle");
  const [filesRevision, setFilesRevision] = useState(0);

  const [workState, setWorkState] = useState<WorkState>("empty");
  const [file, setFile] = useState<File | null>(null);
  const [info, setInfo] = useState<OnDeviceMediaInfo | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [roi, setRoi] = useState<NormalizedRoi>(FULL_FRAME_ROI);
  const [analysisWindow, setAnalysisWindow] = useState<AnalysisWindow>({
    start: 0,
    end: 0,
  });
  const [candidateProgress, setCandidateProgress] =
    useState<AnalysisProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedVideoUrl, setSelectedVideoUrl] = useState<string | null>(null);

  const projectsRef = useRef<VolleyCutProject[]>([]);
  const filesRef = useRef(new Map<string, File>());
  const activeJobRef = useRef<string | null>(null);
  const activeMediaRef = useRef<{
    projectId: string;
    media: OpenedMedia;
  } | null>(null);
  const deletedProjectIdsRef = useRef(new Set<string>());
  const previewUrlRef = useRef<string | null>(null);
  const elapsedTimerRef = useRef<number | null>(null);
  const mountedRef = useRef(true);
  const resumePreviewAfterSeek = useRef(false);
  const candidateVideoRef = useRef<HTMLVideoElement>(null);
  const suppressionAugmentingRef = useRef(new Set<string>());

  const safariUnsupported = isUnsupportedSafariBrowser();
  const webCodecsReady =
    !safariUnsupported &&
    "VideoDecoder" in window &&
    "AudioDecoder" in window &&
    "VideoFrame" in window;
  const secureContext = window.isSecureContext;
  const busy = workState === "opening";
  const selectedProject =
    projects.find((project) => project.id === selectedProjectId) ?? null;
  const selectedSourceFile = selectedProjectId
    ? (filesRef.current.get(selectedProjectId) ?? null)
    : null;

  function replaceProjects(next: VolleyCutProject[]) {
    const sorted = sortProjects(next);
    projectsRef.current = sorted;
    setProjects(sorted);
  }

  function commitProject(project: VolleyCutProject) {
    const next = projectsRef.current.some(
      (candidate) => candidate.id === project.id,
    )
      ? projectsRef.current.map((candidate) =>
          candidate.id === project.id ? project : candidate,
        )
      : [...projectsRef.current, project];
    replaceProjects(next);
    void putProject(project).catch(() => {
      if (mountedRef.current) {
        setError(
          "Browser project storage is unavailable. This project will last for this tab only.",
        );
      }
    });
  }

  function replacePreviewUrl(next: string | null) {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = next;
    setPreviewUrl(next);
  }

  function resetCandidate() {
    replacePreviewUrl(null);
    setFile(null);
    setInfo(null);
    setRoi(FULL_FRAME_ROI);
    setAnalysisWindow({ start: 0, end: 0 });
    setCandidateProgress(null);
    setError(null);
    setWorkState("empty");
    resumePreviewAfterSeek.current = false;
  }

  function selectProject(projectIdToSelect: string | null) {
    setSelectedProjectId(projectIdToSelect);
    setError(null);
    if (projectIdToSelect === null) resetCandidate();
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: replaceProjects only writes through stable state setters and a ref.
  useEffect(() => {
    let active = true;
    void listProjects()
      .then((storedProjects) => {
        if (!active) return;
        replaceProjects(storedProjects);
        let lastSelected: string | null = null;
        try {
          lastSelected = window.localStorage.getItem(
            SELECTED_PROJECT_STORAGE_KEY,
          );
        } catch {
          // Project selection still works when localStorage is restricted.
        }
        const initial =
          storedProjects.find((project) => project.id === lastSelected) ??
          storedProjects[0] ??
          null;
        setSelectedProjectId(initial?.id ?? null);
      })
      .catch(() => {
        if (active) {
          setError(
            "Browser project storage is unavailable. New projects will last for this tab only.",
          );
        }
      })
      .finally(() => {
        if (active) setProjectsLoaded(true);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!projectsLoaded) return;
    try {
      if (selectedProjectId) {
        window.localStorage.setItem(
          SELECTED_PROJECT_STORAGE_KEY,
          selectedProjectId,
        );
      } else {
        window.localStorage.removeItem(SELECTED_PROJECT_STORAGE_KEY);
      }
    } catch {
      // Remembering the project selector is a convenience, not a workflow requirement.
    }
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [projectsLoaded, selectedProjectId]);

  useEffect(() => {
    const project = projectsRef.current.find(
      (candidate) => candidate.id === selectedProjectId,
    );
    if (
      !project?.analysis ||
      project.status !== "ready" ||
      project.analysis.suppression ||
      !project.analysis.featureNames ||
      !project.analysis.featureValues ||
      suppressionAugmentingRef.current.has(project.id)
    ) {
      return;
    }
    suppressionAugmentingRef.current.add(project.id);
    let active = true;
    void augmentStoredAnalysisWithSuppression(
      project.analysis,
      project.analysisWindow,
    )
      .then((analysis) => {
        if (!active || !analysis) return;
        commitProject({
          ...project,
          analysis,
          updatedAt: new Date().toISOString(),
        });
      })
      .catch((cause) => {
        if (active) {
          setError(
            `Suppression suggestions could not be added from cached features: ${cause instanceof Error ? cause.message : String(cause)}`,
          );
        }
      })
      .finally(() => {
        suppressionAugmentingRef.current.delete(project.id);
      });
    return () => {
      active = false;
    };
  }, [selectedProjectId, projects]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: filesRevision intentionally invalidates the URL created from the file map ref.
  useEffect(() => {
    const source = selectedProjectId
      ? filesRef.current.get(selectedProjectId)
      : null;
    const url = source ? URL.createObjectURL(source) : null;
    setSelectedVideoUrl(url);
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [filesRevision, selectedProjectId]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      activeMediaRef.current?.media.input.dispose();
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      if (elapsedTimerRef.current !== null)
        window.clearInterval(elapsedTimerRef.current);
    };
  }, []);

  async function chooseFile(selected: File | null) {
    if (!selected) return;
    if (safariUnsupported) {
      setError(
        "Safari is not supported on macOS or iOS. Open VolleyCut in Google Chrome instead.",
      );
      setWorkState("error");
      return;
    }
    if (!webCodecsReady) {
      setError(
        secureContext
          ? "This browser does not expose the WebCodecs APIs needed for local analysis. Use a current Chrome or Edge release."
          : "Local analysis requires HTTPS. Deploy this static app over HTTPS, or use localhost during development.",
      );
      setWorkState("error");
      return;
    }

    setFile(selected);
    setInfo(null);
    setRoi(FULL_FRAME_ROI);
    setAnalysisWindow({ start: 0, end: 0 });
    setError(null);
    resumePreviewAfterSeek.current = false;
    setCandidateProgress({
      stage: "opening",
      completed: 0,
      total: 1,
      detail: "Reading container metadata locally",
    });
    setWorkState("opening");
    replacePreviewUrl(URL.createObjectURL(selected));

    let opened: OpenedMedia | null = null;
    try {
      opened = await openLocalMedia(selected);
      setInfo(opened.info);
      setAnalysisWindow({ start: 0, end: opened.info.duration });
      setCandidateProgress(null);
      setWorkState("ready");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setCandidateProgress(null);
      setWorkState("error");
    } finally {
      opened?.input.dispose();
    }
  }

  function enqueueProject(projectIdToQueue: string) {
    setQueueIds((current) =>
      current.includes(projectIdToQueue)
        ? current
        : [...current, projectIdToQueue],
    );
  }

  function setGameBoundary(side: "start" | "end", requestedTime: number) {
    if (!info || !Number.isFinite(requestedTime)) return;
    setAnalysisWindow((current) => {
      const normalized = normalizeAnalysisWindow(current, info.duration);
      if (side === "start") {
        return {
          ...normalized,
          start: Math.max(
            0,
            Math.min(
              normalized.end - MIN_ANALYSIS_WINDOW_SECONDS,
              Math.round(requestedTime * 10) / 10,
            ),
          ),
        };
      }
      return {
        ...normalized,
        end: Math.min(
          info.duration,
          Math.max(
            normalized.start + MIN_ANALYSIS_WINDOW_SECONDS,
            Math.round(requestedTime * 10) / 10,
          ),
        ),
      };
    });
    setError(null);
  }

  function markGameBoundary(side: "start" | "end") {
    setGameBoundary(side, candidateVideoRef.current?.currentTime ?? 0);
  }

  function createAndQueueProject() {
    if (!file || !info) return;
    const normalizedWindow = normalizeAnalysisWindow(
      analysisWindow,
      info.duration,
    );
    if (
      normalizedWindow.end - normalizedWindow.start <
      MIN_ANALYSIS_WINDOW_SECONDS
    ) {
      setError(
        `Mark at least ${MIN_ANALYSIS_WINDOW_SECONDS} second of game footage before queueing inference.`,
      );
      return;
    }
    const source = projectSource(file);
    const id = projectId(source, info, normalizedWindow);
    const existing = projectsRef.current.find((project) => project.id === id);
    filesRef.current.set(id, file);
    setFilesRevision((current) => current + 1);

    if (existing?.status === "ready" && existing.analysis) {
      setSelectedProjectId(id);
      resetCandidate();
      return;
    }

    const now = new Date().toISOString();
    const project: VolleyCutProject = {
      schemaVersion: 1,
      id,
      source,
      info,
      analysisWindow: normalizedWindow,
      roi,
      status: "queued",
      analysis: null,
      error: null,
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    };
    commitProject(project);
    setSelectedProjectId(id);
    enqueueProject(id);
    resetCandidate();
  }

  async function runProject(projectIdToRun: string) {
    const project = projectsRef.current.find(
      (candidate) => candidate.id === projectIdToRun,
    );
    const sourceFile = filesRef.current.get(projectIdToRun);
    if (
      !project ||
      !sourceFile ||
      deletedProjectIdsRef.current.has(projectIdToRun)
    )
      return;

    const source = project.source.fingerprint
      ? project.source
      : {
          ...project.source,
          fingerprint: await sourceFileFingerprint(sourceFile).catch(
            () => undefined,
          ),
        };
    if (deletedProjectIdsRef.current.has(projectIdToRun)) return;
    const running: VolleyCutProject = {
      ...project,
      source,
      status: "analyzing",
      error: null,
      updatedAt: new Date().toISOString(),
    };
    commitProject(running);
    setActiveProgress({
      stage: "opening",
      completed: 0,
      total: 1,
      detail: "Opening the local source for queued inference",
    });
    setActiveElapsedSeconds(0);
    const startedAt = performance.now();
    elapsedTimerRef.current = window.setInterval(() => {
      if (mountedRef.current)
        setActiveElapsedSeconds((performance.now() - startedAt) / 1000);
    }, 500);
    const releaseWakeLock = await holdScreenWakeLock(setWakeLockState);
    let opened: OpenedMedia | null = null;
    try {
      opened = await openLocalMedia(sourceFile);
      activeMediaRef.current = { projectId: projectIdToRun, media: opened };
      const result = await analyzeOpenedMedia(
        opened,
        running.roi,
        "local-source" satisfies OnDeviceAnalysis["featurePath"],
        DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
        (nextProgress) => {
          if (
            !deletedProjectIdsRef.current.has(projectIdToRun) &&
            mountedRef.current
          ) {
            setActiveProgress(nextProgress);
          }
        },
        {
          name: sourceFile.name,
          size: sourceFile.size,
          lastModified: sourceFile.lastModified,
        },
        {
          detailedProfiling: false,
          decodeStrategy: DEFAULT_VIDEO_DECODE_STRATEGY,
          decoderAcceleration: VIDEO_DECODER_HARDWARE_ACCELERATION,
          reductionKernel: DEFAULT_FEATURE_REDUCTION_KERNEL,
          analysisWindow: running.analysisWindow,
        },
      );
      if (deletedProjectIdsRef.current.has(projectIdToRun)) return;
      const ready: VolleyCutProject = {
        ...running,
        status: "ready",
        analysis: result,
        error: null,
        updatedAt: new Date().toISOString(),
      };
      commitProject(ready);
    } catch (cause) {
      if (deletedProjectIdsRef.current.has(projectIdToRun)) return;
      const failed: VolleyCutProject = {
        ...running,
        status: "error",
        error: cause instanceof Error ? cause.message : String(cause),
        updatedAt: new Date().toISOString(),
      };
      commitProject(failed);
    } finally {
      opened?.input.dispose();
      if (activeMediaRef.current?.projectId === projectIdToRun)
        activeMediaRef.current = null;
      if (elapsedTimerRef.current !== null)
        window.clearInterval(elapsedTimerRef.current);
      elapsedTimerRef.current = null;
      if (mountedRef.current)
        setActiveElapsedSeconds((performance.now() - startedAt) / 1000);
      await releaseWakeLock();
      if (deletedProjectIdsRef.current.has(projectIdToRun)) {
        await deleteFeatureCachesForSource(running.source).catch(
          () => undefined,
        );
      }
    }
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: the active-job ref serializes work while runProject reads the latest project and file refs.
  useEffect(() => {
    const next = queueIds[0];
    if (!next || activeJobRef.current) return;
    activeJobRef.current = next;
    setActiveJobId(next);
    void runProject(next).finally(() => {
      activeJobRef.current = null;
      if (mountedRef.current) {
        setQueueIds((current) => current.filter((id) => id !== next));
        setActiveJobId(null);
        setActiveProgress(null);
        setWakeLockState("idle");
      }
    });
  }, [queueIds]);

  function queueAttachedProject(project: VolleyCutProject) {
    if (!filesRef.current.has(project.id)) return;
    const queued: VolleyCutProject = {
      ...project,
      status: "queued",
      error: null,
      updatedAt: new Date().toISOString(),
    };
    commitProject(queued);
    enqueueProject(project.id);
  }

  async function attachSource(
    project: VolleyCutProject,
    selected: File | null,
  ) {
    if (!selected) return;
    if (!(await sourceCanReconnectFile(project.source, selected))) {
      setError(
        `Choose the original ${project.source.name} file (${compactBytes(project.source.size)}). The selected file does not match this project.`,
      );
      return;
    }
    filesRef.current.set(project.id, selected);
    setFilesRevision((current) => current + 1);
    setError(null);
    if (
      project.status !== "ready" ||
      (!project.analysis?.suppression &&
        (!project.analysis?.featureNames || !project.analysis?.featureValues))
    ) {
      queueAttachedProject(project);
    }
  }

  async function removeSelectedProject() {
    if (!selectedProject) return;
    if (
      !window.confirm(
        `Delete ${selectedProject.source.name} (${selectedProject.id}) and its cached inference?`,
      )
    )
      return;

    deletedProjectIdsRef.current.add(selectedProject.id);
    if (activeMediaRef.current?.projectId === selectedProject.id) {
      activeMediaRef.current.media.input.dispose();
    }
    setQueueIds((current) => current.filter((id) => id !== selectedProject.id));
    filesRef.current.delete(selectedProject.id);
    setFilesRevision((current) => current + 1);
    replaceProjects(
      projectsRef.current.filter(
        (project) => project.id !== selectedProject.id,
      ),
    );
    setSelectedProjectId(null);
    setError(null);

    const analysisId = projectAnalysisId(selectedProject);
    if (analysisId) {
      try {
        for (const key of cutDraftStorageKeys(analysisId))
          window.localStorage.removeItem(key);
      } catch {
        // IndexedDB deletion remains useful if localStorage is restricted.
      }
    }
    await Promise.allSettled([
      deleteProject(selectedProject.id),
      deleteFeatureCachesForSource(selectedProject.source),
    ]);
  }

  const activePercent = progressPercent(activeProgress);
  const queueLabel = activeJobId
    ? `${projects.find((project) => project.id === activeJobId)?.source.name ?? "Project"} · ${Math.round(activePercent)}%${queueIds.length > 1 ? ` · ${queueIds.length - 1} queued` : ""}`
    : queueIds.length > 0
      ? `${queueIds.length} queued`
      : null;
  const projectHeader = (
    <ProjectHeader
      projects={projects}
      selectedProjectId={selectedProjectId}
      queueLabel={queueLabel}
      onSelectProject={selectProject}
      onDeleteProject={() => void removeSelectedProject()}
    />
  );

  const productAnalysis = useMemo<ProductAnalysis | null>(() => {
    if (!selectedProject?.analysis || selectedProject.status !== "ready")
      return null;
    return {
      id: projectAnalysisId(selectedProject)!,
      recordingId: selectedProject.id,
      kind: "model",
      modelId: selectedProject.analysis.modelId,
      duration: selectedProject.info.duration,
      analysisWindow: selectedProject.analysisWindow,
      width: selectedProject.info.width,
      height: selectedProject.info.height,
      sourceFilename: selectedProject.source.name,
      source: selectedProject.source,
      mediaInfo: selectedProject.info,
      roi: selectedProject.roi,
      runtimeVariant: DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
      videoUrl: selectedVideoUrl,
      rallies: selectedProject.analysis.intervals,
      ignoredIntervals: [
        ...(selectedProject.analysisWindow.start > 0
          ? [{
              start: 0,
              end: selectedProject.analysisWindow.start,
              reason: "outside-game-window",
            }]
          : []),
        ...(selectedProject.analysisWindow.end < selectedProject.info.duration
          ? [{
              start: selectedProject.analysisWindow.end,
              end: selectedProject.info.duration,
              reason: "outside-game-window",
            }]
          : []),
      ],
      features:
        selectedProject.analysis.featureNames && selectedProject.analysis.featureValues
          ? {
              times: selectedProject.analysis.times,
              values: selectedProject.analysis.featureValues,
              rows: selectedProject.analysis.times.length,
              columns: selectedProject.analysis.featureNames.length,
              names: selectedProject.analysis.featureNames,
            }
          : null,
      inferenceTimes: selectedProject.analysis.times,
      probabilities: {
        rally: selectedProject.analysis.rallyProbabilities,
        serve: selectedProject.analysis.serveProbabilities,
        deadState: selectedProject.analysis.deadStateProbabilities,
      },
      productionComponents: selectedProject.analysis.productionComponents,
      suppression: selectedProject.analysis.suppression,
    };
  }, [selectedProject, selectedVideoUrl]);

  if (selectedProject && productAnalysis) {
    return (
      <CutEditor
        key={productAnalysis.id}
        header={projectHeader}
        initialAnalysis={productAnalysis}
        sourceFile={selectedSourceFile}
        sourceError={error}
        onAttachSource={(selected) =>
          void attachSource(selectedProject, selected)
        }
        onRequestSuppression={() => queueAttachedProject(selectedProject)}
      />
    );
  }

  const selectedIsActive = selectedProject?.id === activeJobId;
  const displayedProgress = selectedIsActive
    ? activeProgress
    : candidateProgress;
  const percent = progressPercent(displayedProgress);
  const featurePerformance = displayedProgress?.performance ?? null;
  const featureRate =
    featurePerformance && featurePerformance.videoElapsedMs > 0
      ? featurePerformance.generatedVideoSeconds /
        (featurePerformance.videoElapsedMs / 1000)
      : null;
  const elapsedSeconds = selectedIsActive ? activeElapsedSeconds : 0;
  const audioProgressPercent =
    displayedProgress?.stage === "audio" && displayedProgress.total > 0
      ? Math.min(
          100,
          Math.max(
            0,
            (displayedProgress.completed / displayedProgress.total) * 100,
          ),
        )
      : null;
  const audioPerformance = displayedProgress?.audioPerformance ?? null;
  const audioElapsedSeconds = audioPerformance
    ? audioPerformance.elapsedMs / 1000
    : null;
  const audioDecodeRate =
    audioPerformance && audioPerformance.decodeElapsedMs > 0
      ? audioPerformance.decodedAudioSeconds /
        (audioPerformance.decodeElapsedMs / 1000)
      : null;
  const audioProcessingRate =
    audioElapsedSeconds &&
    displayedProgress?.stage === "audio" &&
    displayedProgress.completed > 0
      ? displayedProgress.completed / audioElapsedSeconds
      : null;
  const audioEtaSeconds =
    audioProcessingRate &&
    displayedProgress?.stage === "audio" &&
    displayedProgress.total > displayedProgress.completed
      ? (displayedProgress.total - displayedProgress.completed) /
        audioProcessingRate
      : audioProgressPercent !== null && audioProgressPercent >= 100
        ? 0
        : null;
  const analysisEtaSeconds =
    displayedProgress?.stage === "complete"
      ? 0
      : displayedProgress?.stage === "video" && featureRate && featureRate > 0
        ? Math.max(0, displayedProgress.total - displayedProgress.completed) /
          featureRate
        : elapsedSeconds >= 2 && percent > 2
          ? (elapsedSeconds * (100 - percent)) / percent
          : null;

  return (
    <main className={styles.page}>
      {projectHeader}

      {!selectedProject && (
        <>
          <section className={styles.hero}>
            <p>PRIVATE PROJECT WORKSPACE</p>
            <h1>
              Load. Detect. <em>Refine.</em>
            </h1>
            <p className={styles.lede}>
              Queue volleyball videos for local inference, switch between
              projects, and refine completed cuts while the next match generates
              features in the background. Selected videos are never uploaded.
            </p>
            <div
              className={styles.pipeline}
              role="group"
              aria-label="Local processing pipeline"
            >
              <span>
                <b>01</b> Local video
              </span>
              <i>→</i>
              <span>
                <b>02</b> Cached features
              </span>
              <i>→</i>
              <span>
                <b>03</b> Local inference
              </span>
              <i>→</i>
              <span>
                <b>04</b> Project editor
              </span>
            </div>
          </section>

          {safariUnsupported && (
            <p className={styles.notice} role="status">
              <strong>Safari is not supported.</strong> Feature extraction is
              unreliable in Safari on macOS and iOS. Open VolleyCut in the
              latest Google Chrome instead.
            </p>
          )}
          {!safariUnsupported && !secureContext && (
            <p className={styles.notice}>
              This page is not in a secure context. The interface is available,
              but local media analysis needs HTTPS or localhost.
            </p>
          )}
          {error && <p className={styles.error}>{error}</p>}

          <section className={styles.importCard} data-tour="source-picker">
            <div>
              <p>STEP 01 · NEW PROJECT SOURCE</p>
              <h2>{file?.name ?? "Choose a volleyball video"}</h2>
              <p>
                {file
                  ? `${compactBytes(file.size)} · a project is created only when you queue inference`
                  : "MP4, WebM, MOV, MKV, and other browser-decodable containers are supported."}
              </p>
            </div>
            <label
              className={styles.fileButton}
              data-disabled={busy || safariUnsupported || undefined}
            >
              {file ? "Choose another" : "Choose video"}
              <input
                type="file"
                accept="video/*,.mkv,.webm,.mov,.mp4,.m4v"
                disabled={busy || safariUnsupported}
                onChange={(event) =>
                  void chooseFile(event.currentTarget.files?.[0] ?? null)
                }
              />
            </label>
          </section>

          {info && previewUrl && (
            <section className={styles.workspace} data-tour="source-workspace">
              <div className={styles.viewer}>
                <div
                  className={styles.videoStage}
                  style={{ aspectRatio: `${info.width} / ${info.height}` }}
                >
                  <video
                    ref={candidateVideoRef}
                    src={previewUrl}
                    controls
                    preload="metadata"
                    playsInline
                    onPlay={() => {
                      resumePreviewAfterSeek.current = true;
                    }}
                    onPause={(event) => {
                      if (!event.currentTarget.seeking)
                        resumePreviewAfterSeek.current = false;
                    }}
                    onEnded={() => {
                      resumePreviewAfterSeek.current = false;
                    }}
                    onSeeked={(event) => {
                      if (resumePreviewAfterSeek.current) {
                        void event.currentTarget.play().catch(() => undefined);
                      }
                    }}
                  />
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
                </div>
              </div>

              <aside className={styles.inspector}>
                <p>STEP 02 · PROJECT FEATURES</p>
                <h2>Confirm the camera crop</h2>
                <dl>
                  <div>
                    <dt>Duration</dt>
                    <dd>{formatDuration(info.duration)}</dd>
                  </div>
                  <div>
                    <dt>Frame</dt>
                    <dd>
                      {info.width} × {info.height}
                    </dd>
                  </div>
                  <div>
                    <dt>Video</dt>
                    <dd>{info.videoCodecString ?? info.videoCodec}</dd>
                  </div>
                  <div>
                    <dt>Audio</dt>
                    <dd>
                      {info.hasAudio
                        ? (info.audioCodec ?? "Available")
                        : "No track"}
                    </dd>
                  </div>
                </dl>
                <p className={styles.cropHelp}>
                  Keep the court and players inside the box. Exclude static
                  borders, stands, or neighboring courts when practical.
                </p>
                <section
                  className={styles.gameWindow}
                  data-tour="source-game-window"
                  aria-labelledby="game-window-heading"
                >
                  <div className={styles.gameWindowHeading}>
                    <div>
                      <span>ANALYSIS WINDOW</span>
                      <strong id="game-window-heading">Mark game start &amp; end</strong>
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        setAnalysisWindow({ start: 0, end: info.duration })
                      }
                    >
                      Use full video
                    </button>
                  </div>
                  <p>
                    Seek the video, then mark each boundary. Only this range
                    generates features or appears on the editor overview.
                  </p>
                  <div className={styles.gameBoundaryGrid}>
                    <label>
                      <span>Game start</span>
                      <input
                        aria-label="Game start in seconds"
                        type="number"
                        min="0"
                        max={Math.max(0, analysisWindow.end - MIN_ANALYSIS_WINDOW_SECONDS)}
                        step="0.1"
                        value={analysisWindow.start}
                        onChange={(event) =>
                          setGameBoundary("start", event.currentTarget.valueAsNumber)
                        }
                      />
                      <output>{preciseTime(analysisWindow.start)}</output>
                      <button type="button" onClick={() => markGameBoundary("start")}>
                        Set to playhead
                      </button>
                    </label>
                    <label>
                      <span>Game end</span>
                      <input
                        aria-label="Game end in seconds"
                        type="number"
                        min={analysisWindow.start + MIN_ANALYSIS_WINDOW_SECONDS}
                        max={info.duration}
                        step="0.1"
                        value={analysisWindow.end}
                        onChange={(event) =>
                          setGameBoundary("end", event.currentTarget.valueAsNumber)
                        }
                      />
                      <output>{preciseTime(analysisWindow.end)}</output>
                      <button type="button" onClick={() => markGameBoundary("end")}>
                        Set to playhead
                      </button>
                    </label>
                  </div>
                  <strong className={styles.gameWindowSummary}>
                    {formatDuration(analysisWindow.end - analysisWindow.start)} analyzed
                    {analysisWindow.end - analysisWindow.start < info.duration - 0.05
                      ? ` · ${formatDuration(info.duration - (analysisWindow.end - analysisWindow.start))} skipped`
                      : " · full source"}
                  </strong>
                </section>
                <div data-tour="source-camera">
                  <div className={styles.presetButtons}>
                    <button
                      type="button"
                      onClick={() => setRoi(FULL_FRAME_ROI)}
                    >
                      Full frame
                    </button>
                  </div>
                  <div className={styles.roiGrid}>
                    {(["x", "y", "width", "height"] as const).map((field) => (
                      <label key={field}>
                        <span>
                          {field} <output>{Math.round(roi[field] * 100)}%</output>
                        </span>
                        <input
                          aria-label={`Feature crop ${field}`}
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={roi[field]}
                          onChange={(event) =>
                            setRoi(
                              clampRoi({
                                ...roi,
                                [field]: Number(event.currentTarget.value),
                              }),
                            )
                          }
                          disabled={busy}
                        />
                      </label>
                    ))}
                  </div>
                </div>
                <button
                  className={styles.analyzeButton}
                  data-tour="source-create"
                  type="button"
                  onClick={createAndQueueProject}
                  disabled={
                    busy ||
                    safariUnsupported ||
                    analysisWindow.end - analysisWindow.start <
                      MIN_ANALYSIS_WINDOW_SECONDS
                  }
                >
                  Create project &amp; queue inference
                </button>
                <small className={styles.runtimeNote}>
                  Both production models run on the same cached features.
                  Overlaps are merged; one-model detections are flagged for
                  validation. Results stay in IndexedDB.
                </small>
              </aside>
            </section>
          )}

          <GuidedTour
            stage="source"
            sourceReady={Boolean(info && previewUrl)}
          />
        </>
      )}

      {selectedProject && (
        <section className={styles.projectPanel}>
          <p>PROJECT · {selectedProject.status.toUpperCase()}</p>
          <h1>{selectedProject.source.name}</h1>
          <code>{selectedProject.id}</code>
          <p>
            {selectedProject.status === "analyzing"
              ? "This project is generating audiovisual features locally. You can select a ready project and edit it while this continues."
              : selectedProject.status === "queued"
                ? `Queued for local inference${queueIds.indexOf(selectedProject.id) > 0 ? ` · position ${queueIds.indexOf(selectedProject.id) + 1}` : ""}.`
                : (selectedProject.error ??
                  "Reconnect the exact local source file to continue this project.")}
          </p>
          {(selectedProject.status === "waiting" ||
            selectedProject.status === "error") &&
            (selectedSourceFile ? (
              <button
                className={styles.fileButton}
                type="button"
                onClick={() => queueAttachedProject(selectedProject)}
              >
                Queue inference again
              </button>
            ) : (
              <label className={styles.fileButton}>
                Reconnect source &amp; queue
                <input
                  type="file"
                  accept="video/*,.mkv,.webm,.mov,.mp4,.m4v"
                  onChange={(event) =>
                    void attachSource(
                      selectedProject,
                      event.currentTarget.files?.[0] ?? null,
                    )
                  }
                />
              </label>
            ))}
          <button
            className={styles.newProjectButton}
            type="button"
            onClick={() => selectProject(null)}
          >
            Start another project
          </button>
        </section>
      )}

      {displayedProgress && (
        <section className={styles.progressCard} aria-live="polite">
          <div>
            <p>LOCAL PROCESSING · {displayedProgress.stage.toUpperCase()}</p>
            <strong>{displayedProgress.detail}</strong>
          </div>
          <output>{Math.round(percent)}%</output>
          <div className={styles.progressTrack}>
            <i style={{ width: `${percent}%` }} />
          </div>
          {selectedIsActive && (
            <div className={styles.progressTiming}>
              <div>
                <span>Elapsed</span>
                <strong>{formatDuration(elapsedSeconds)}</strong>
              </div>
              <div>
                <span>Estimated remaining</span>
                <strong>
                  {analysisEtaSeconds === null
                    ? "Estimating…"
                    : analysisEtaSeconds <= 1
                      ? "Finishing…"
                      : `About ${formatDuration(analysisEtaSeconds)}`}
                </strong>
              </div>
              {audioProgressPercent !== null && (
                <>
                  <div>
                    <span>Audio processing</span>
                    <strong>{Math.round(audioProgressPercent)}%</strong>
                  </div>
                  <div>
                    <span>Audio elapsed</span>
                    <strong>
                      {audioElapsedSeconds === null
                        ? "Starting…"
                        : formatDuration(audioElapsedSeconds)}
                    </strong>
                  </div>
                  <div>
                    <span>Audio remaining</span>
                    <strong>
                      {audioEtaSeconds === null
                        ? "Estimating…"
                        : audioEtaSeconds <= 1
                          ? "Finishing…"
                          : `About ${formatDuration(audioEtaSeconds)}`}
                    </strong>
                  </div>
                </>
              )}
            </div>
          )}
          <p>
            {displayedProgress.stage === "video" && featureRate
              ? `${featureRate.toFixed(2)}× real-time feature generation`
              : displayedProgress.stage === "audio" && audioProcessingRate
                ? `${audioProcessingRate.toFixed(2)}× real-time audio processing${audioDecodeRate ? ` · ${audioDecodeRate.toFixed(2)}× decode` : ""}`
                : "Feature extraction, audio analysis, and both model passes run locally."}
            {displayedProgress.featureCache?.resumedRows
              ? ` · resumed ${displayedProgress.featureCache.resumedRows.toLocaleString()} saved frames`
              : ""}
            {wakeLockState === "active" ? " · screen wake lock active" : ""}
          </p>
        </section>
      )}

      <footer className={styles.footer}>
        <span>
          Project metadata, generated features, predictions, and edit drafts
          stay in this browser.
        </span>
        <span>
          Local video bytes are never uploaded or copied into project storage.
        </span>
      </footer>
    </main>
  );
}
