import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { MobileAppBanner } from "@/components/MobileAppBanner";
import { GuidedTour } from "@/components/GuidedTour";
import { InferenceProgressPanel } from "@/components/InferenceProgressPanel";
import { ProjectHeader } from "@/components/ProjectHeader";
import { SiteFooter } from "@/components/SiteFooter";
import { awaitWithAbort } from "@/lib/on-device/abortable";
import { missingScoreStages, scoreSpecialistReadiness, type ScoreSpecialistStage } from "@/lib/score-specialist-state";
import { projectDraftStorageKeys } from "@/lib/review-draft-persistence";
import {
  CORE_INFERENCE_STEP_IDS,
  createInferenceProgressSteps,
  finishInferenceStep,
  type InferenceProgressStep,
  overallInferenceProgress,
  SCORE_INFERENCE_STEP_IDS,
  updatePipelineInferenceSteps,
  updateSpecialistInferenceStep,
} from "@/lib/inference-progress";
import { importModelFeedbackProject } from "@/lib/model-feedback-import";
import {
  type AnalysisWindow,
  MIN_ANALYSIS_WINDOW_SECONDS,
  normalizeAnalysisWindow,
} from "@/lib/on-device/analysis-window";
import { isIosBrowser, isUnsupportedSafariBrowser } from "@/lib/on-device/browser-support";
import { DEFAULT_RALLY_MODEL, RALLY_MODEL_OPTIONS, isNeuralModelId, isRallyModelSelection, type RallyModelSelection } from "@/lib/on-device/rally-model";
import { deleteNeuralEmbeddingsForSource } from "@/lib/on-device/neural-cache";
import { deleteFeatureCachesForSource } from "@/lib/on-device/feature-cache";
import { type OpenedMedia, openLocalMedia } from "@/lib/on-device/media";
import {
  analyzeOpenedMedia,
  DEFAULT_FEATURE_REDUCTION_KERNEL,
  DEFAULT_VIDEO_DECODE_STRATEGY,
  VIDEO_DECODER_HARDWARE_ACCELERATION,
} from "@/lib/on-device/pipeline";
import { augmentStoredAnalysisWithSuppression, runProductionInferenceFromFeatures } from "@/lib/on-device/production-inference";
import {
  requestVideoExportTarget,
  type VideoExportTarget,
} from "@/lib/on-device/export-delivery";
import { DEFAULT_ON_DEVICE_RUNTIME_VARIANT } from "@/lib/on-device/runtime-variants";
import { prepareServiceWorkerStreamDownload } from "@/lib/on-device/stream-download";
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
import { RallyDesk } from "@/designs/taste";
import { editHistoryKey } from "@/designs/taste/edit-history";
import {
  type DesignExportJob,
  type DesignVideoExportRequest,
  type DesignWorkActivity,
  type DesignScoreRecovery,
  useDesignReview,
} from "@/designs/useDesignReview";
import {
  deleteProject,
  confirmProjectSource,
  getProject,
  listProjects,
  markProjectExported,
  projectAnalysisId,
  projectSource,
  ProjectSourceConflictError,
  putProject,
  resolveProjectSetup,
  sameProjectSource,
  SELECTED_PROJECT_STORAGE_KEY,
  sourceCanReconnectFile,
  sourceFileFingerprint,
  type ProjectSource,
  type StoredProject,
  type VolleySpliceProject,
} from "@/lib/project-store";

import styles from "./App.module.css";

type WorkState = "empty" | "opening" | "ready" | "error";

const FULL_FRAME_ROI: NormalizedRoi = { x: 0, y: 0, width: 1, height: 1 };

type AppExportJob = DesignExportJob & { projectInstanceId: string };

type ProjectWorkIdentity = Pick<StoredProject, "id" | "instanceId" | "source">;
type QueuedProjectWork = { id: string; instanceId: string; sourceFile: File };

type QueuedVideoExport = DesignVideoExportRequest & {
  projectId: string;
  projectName: string;
  projectInstanceId: string;
  sourceFile: File;
  duration: number;
  target: VideoExportTarget | null;
};

function ReadyProjectEditor({
  projectId: readyProjectId,
  projects,
  workActivity,
  exportJobs,
  sourceFile,
  videoUrl,
  productAnalysis,
  onSelectProject,
  onAttachSource,
  onQueueVideoExport,
  onDeleteProject,
  scoreRecovery,
  storageError,
  sourceError,
}: {
  projectId: string;
  projects: VolleySpliceProject[];
  workActivity: DesignWorkActivity[];
  exportJobs: DesignExportJob[];
  sourceFile: File | null;
  videoUrl: string | null;
  productAnalysis: ProductAnalysis;
  onSelectProject: (projectId: string | null) => void;
  onAttachSource: (file: File) => void | Promise<void>;
  onQueueVideoExport: (request: DesignVideoExportRequest) => void;
  onDeleteProject: () => void;
  scoreRecovery: DesignScoreRecovery;
  storageError: string | null;
  sourceError: string | null;
}) {
  const review = useDesignReview({
    projectId: readyProjectId,
    projects,
    workActivity,
    exportJobs,
    sourceFile,
    videoUrl,
    productAnalysis,
    onSelectProject,
    onAttachSource,
    onQueueVideoExport,
    scoreRecovery,
  });

  if (review.state === "ready") {
    return (
      <>
        {sourceError && <p className={styles.error} role="alert">{sourceError}</p>}
        {storageError && <p className={styles.error} role="alert">{storageError}</p>}
        <RallyDesk review={review} onDeleteProject={onDeleteProject} />
      </>
    );
  }

  return (
    <main className={styles.page} aria-busy={review.state === "loading"}>
      <section className={styles.notice}>
        {review.state === "loading"
          ? "Opening your saved review…"
          : review.message}
        {review.state !== "loading" && (
          <button type="button" onClick={() => onSelectProject(null)}>Open another project</button>
        )}
      </section>
    </main>
  );
}

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

function sortProjects(projects: VolleySpliceProject[]): VolleySpliceProject[] {
  return [...projects].sort((left, right) =>
    right.updatedAt.localeCompare(left.updatedAt),
  );
}

const LANDING_COPY: {
  kicker: string;
  title: React.ReactNode;
  description: string;
  steps: string[];
} = {
  kicker: "Bump. Set. Splice.",
  title: <>Choose the game.<br /><em>VolleySplice finds the rallies.</em></>,
  description: "Choose a game video from this device, confirm the part of the recording to analyze, then let VolleySplice prepare the review timeline.",
  steps: ["Choose video", "Set game window", "Analyze locally", "Review rallies"],
};

export function App() {
  if (isIosBrowser()) {
    return (
      <main className={styles.page}>
        <MobileAppBanner />
        <section className={styles.hero}>
          <p>VOLLEYSPLICE FOR iOS</p>
          <h1>Bump. Set. Splice.</h1>
          <p className={styles.lede}>
            To use VolleySplice on iPhone or iPad, download the app from the App
            Store using the link above. The web editor is unavailable on iOS.
          </p>
          <p className={styles.privacyNote}>
            You can also use the web editor in Chrome or Edge on a desktop computer.
          </p>
          <p><a href="./support.html">Support</a> · <a href="./privacy.html">Privacy</a> · <a href="./terms.html">Terms</a></p>
        </section>
      </main>
    );
  }
  return <WebApp />;
}

function WebApp() {
  const [projects, setProjects] = useState<VolleySpliceProject[]>([]);
  const [projectsLoaded, setProjectsLoaded] = useState(false);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(
    null,
  );
  const [queueIds, setQueueIds] = useState<string[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeProgress, setActiveProgress] = useState<AnalysisProgress | null>(
    null,
  );
  const [activeInferenceSteps, setActiveInferenceSteps] = useState<
    InferenceProgressStep[]
  >([]);
  const [wakeLockState, setWakeLockState] = useState<WakeLockState>("idle");
  const [, setFilesRevision] = useState(0);

  const [workState, setWorkState] = useState<WorkState>("empty");
  const [file, setFile] = useState<File | null>(null);
  const [info, setInfo] = useState<OnDeviceMediaInfo | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [roi, setRoi] = useState<NormalizedRoi>(FULL_FRAME_ROI);
  const [analysisWindow, setAnalysisWindow] = useState<AnalysisWindow>({
    start: 0,
    end: 0,
  });
  const [rallyModel, setRallyModel] = useState<RallyModelSelection>(DEFAULT_RALLY_MODEL);
  const [sideSwitchEnabled, setSideSwitchEnabled] = useState(false);
  const [candidateProgress, setCandidateProgress] =
    useState<AnalysisProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackImporting, setFeedbackImporting] = useState(false);
  const [checkingProjectSource, setCheckingProjectSource] = useState(false);
  const [exportJobs, setExportJobs] = useState<AppExportJob[]>([]);
  const [scoreJobs, setScoreJobs] = useState<Record<string, { status: "queued" | "running" | "cancelling"; detail: string }>>({});

  const projectsRef = useRef<VolleySpliceProject[]>([]);
  const filesRef = useRef(new Map<string, File>());
  const validatedSourcesRef = useRef(new Map<string, { source: ProjectSource; instanceId?: string }>());
  const videoUrlsRef = useRef(new Map<string, string>());
  const exportJobsRef = useRef(new Map<string, AppExportJob>());
  const exportQueueRef = useRef<QueuedVideoExport[]>([]);
  const exportWorkerRunningRef = useRef(false);
  const exportTargetRequestsRef = useRef(new Set<string>());
  const activeJobRef = useRef<string | null>(null);
  const activeProjectWorkRef = useRef<QueuedProjectWork | null>(null);
  const activeAbortRef = useRef<AbortController | null>(null);
  const activeMediaRef = useRef<{
    projectId: string;
    media: OpenedMedia;
  } | null>(null);
  const deletedProjectInstancesRef = useRef(new Set<string>());
  const previewUrlRef = useRef<string | null>(null);
  const mountedRef = useRef(true);
  const resumePreviewAfterSeek = useRef(false);
  const candidateVideoRef = useRef<HTMLVideoElement>(null);
  const suppressionAugmentingRef = useRef(new Map<string, { source: ProjectSource; promise: Promise<OnDeviceAnalysis | null> }>());
  const scoreRecoveryIdsRef = useRef(new Set<string>());
  const queuedProjectWorkRef = useRef(new Map<string, QueuedProjectWork>());
  const checkingProjectSourceRef = useRef(false);
  const candidateRevisionRef = useRef(0);

  const safariUnsupported = isUnsupportedSafariBrowser();
  const webCodecsReady =
    !safariUnsupported &&
    "VideoDecoder" in window &&
    "AudioDecoder" in window &&
    "VideoFrame" in window;
  const secureContext = window.isSecureContext;
  const busy = workState === "opening" || feedbackImporting || checkingProjectSource;
  const selectedProject =
    projects.find((project) => project.id === selectedProjectId) ?? null;
  const selectedSourceFile = selectedProjectId
    ? (filesRef.current.get(selectedProjectId) ?? null)
    : null;
  const selectedVideoUrl = selectedProjectId
    ? (videoUrlsRef.current.get(selectedProjectId) ?? null)
    : null;

  function replaceProjects(next: VolleySpliceProject[]) {
    const sorted = sortProjects(next);
    projectsRef.current = sorted;
    setProjects(sorted);
  }

  function wasDeleted(id: string, instanceId = projectsRef.current.find((project) => project.id === id)?.instanceId) {
    return !!instanceId && deletedProjectInstancesRef.current.has(instanceId);
  }

  function replaceProject(project: VolleySpliceProject) {
    replaceProjects(projectsRef.current.some((candidate) => candidate.id === project.id)
      ? projectsRef.current.map((candidate) => candidate.id === project.id ? project : candidate)
      : [...projectsRef.current, project]);
  }

  function unlinkSourceFile(id: string) {
    filesRef.current.delete(id);
    validatedSourcesRef.current.delete(id);
    const url = videoUrlsRef.current.get(id);
    if (url) URL.revokeObjectURL(url);
    videoUrlsRef.current.delete(id);
    setFilesRevision((current) => current + 1);
  }

  function reconcileProject(project: VolleySpliceProject) {
    if (!mountedRef.current || wasDeleted(project.id, project.instanceId)) return;
    const current = projectsRef.current.find((candidate) => candidate.id === project.id);
    if (current && (current.instanceId !== project.instanceId || !sameProjectSource(current.source, project.source))) unlinkSourceFile(project.id);
    if (current && current.instanceId !== project.instanceId) {
      exportJobsRef.current.delete(project.id);
      exportTargetRequestsRef.current.delete(project.id);
      exportQueueRef.current = exportQueueRef.current.filter((job) => job.projectId !== project.id || job.projectInstanceId === project.instanceId);
      setExportJobs([...exportJobsRef.current.values()]);
      if (activeMediaRef.current?.projectId === project.id) activeAbortRef.current?.abort();
    }
    replaceProject(project);
  }

  function sourceFailure(id: string, cause: unknown, expectedInstanceId?: string) {
    if (!mountedRef.current) return;
    if (expectedInstanceId && !projectsRef.current.some((project) =>
      project.id === id && project.instanceId === expectedInstanceId)) return;
    unlinkSourceFile(id);
    if (cause instanceof ProjectSourceConflictError) {
      if (cause.project) reconcileProject(cause.project);
      else replaceProjects(projectsRef.current.filter((project) => project.id !== id));
    }
    setError(cause instanceof Error ? cause.message : String(cause));
  }

  async function validateStoredSource(id: string, selected: File, expectedInstanceId: string | undefined, approved?: { source: ProjectSource; instanceId?: string }) {
    const latest = await getProject(id);
    if (!latest) throw new ProjectSourceConflictError(null);
    if (!expectedInstanceId || latest.instanceId !== expectedInstanceId) throw new ProjectSourceConflictError(latest, "This project instance changed. Reopen it before reconnecting.");
    if ((!approved || approved.instanceId !== latest.instanceId || !sameProjectSource(latest.source, approved.source)) &&
        !(await sourceCanReconnectFile(latest.source, selected))) {
      throw new ProjectSourceConflictError(latest,
        `Choose the original ${latest.source.name} file (${compactBytes(latest.source.size)}). The selected file does not match this project.`);
    }
    return latest;
  }

  async function commitSourceProject(project: VolleySpliceProject, expectedSource: ProjectSource | null) {
    if (wasDeleted(project.id, project.instanceId)) throw new ProjectSourceConflictError(null);
    const previous = projectsRef.current.find((candidate) => candidate.id === project.id);
    const persisted = await putProject(project, { expectedSource });
    if (wasDeleted(project.id, project.instanceId)) throw new ProjectSourceConflictError(null);
    if (mountedRef.current) {
      const current = projectsRef.current.find((candidate) => candidate.id === project.id);
      if (expectedSource !== null && previous && !current) throw new ProjectSourceConflictError(null);
      if (expectedSource !== null && current && (current.instanceId !== persisted.instanceId ||
          (!sameProjectSource(current.source, expectedSource) && !sameProjectSource(current.source, persisted.source)))) {
        throw new ProjectSourceConflictError(current);
      }
      replaceProject(persisted);
      if (filesRef.current.has(project.id)) validatedSourcesRef.current.set(project.id, { source: persisted.source, instanceId: persisted.instanceId });
    }
    return persisted;
  }

  function currentWorkProject(identity: ProjectWorkIdentity) {
    if (!mountedRef.current || wasDeleted(identity.id, identity.instanceId)) return null;
    const current = projectsRef.current.find((project) => project.id === identity.id);
    if (!current || current.instanceId !== identity.instanceId || !sameProjectSource(current.source, identity.source)) {
      throw new ProjectSourceConflictError(current ?? null);
    }
    return current;
  }

  async function commitAnalysisProject(identity: ProjectWorkIdentity, project: VolleySpliceProject) {
    if (!currentWorkProject(identity)) throw new DOMException("Analysis cancelled", "AbortError");
    // Never adopt a newer record's identity for evidence produced by this job.
    return commitSourceProject({ ...project, id: identity.id,
      instanceId: identity.instanceId, source: identity.source }, identity.source);
  }

  async function validateQueuedProject(work: QueuedProjectWork, signal?: AbortSignal) {
    signal?.throwIfAborted();
    const approved = filesRef.current.get(work.id) === work.sourceFile
      ? validatedSourcesRef.current.get(work.id) : undefined;
    const checked = await validateStoredSource(work.id, work.sourceFile, work.instanceId, approved);
    const project = await confirmProjectSource(work.id, checked.source, work.instanceId);
    signal?.throwIfAborted();
    if (!mountedRef.current || wasDeleted(work.id, work.instanceId)) throw new DOMException("Analysis cancelled", "AbortError");
    const current = projectsRef.current.find((candidate) => candidate.id === work.id);
    if (current?.instanceId !== work.instanceId) throw new ProjectSourceConflictError(current ?? null);
    reconcileProject(project);
    if (filesRef.current.get(work.id) !== work.sourceFile) linkSourceFile(work.id, work.sourceFile, project);
    else validatedSourcesRef.current.set(work.id, { source: project.source, instanceId: project.instanceId });
    return project;
  }

  function linkSourceFile(projectIdToLink: string, sourceFile: File, project: VolleySpliceProject) {
    const previousUrl = videoUrlsRef.current.get(projectIdToLink);
    if (previousUrl) URL.revokeObjectURL(previousUrl);
    filesRef.current.set(projectIdToLink, sourceFile);
    validatedSourcesRef.current.set(projectIdToLink, { source: project.source, instanceId: project.instanceId });
    videoUrlsRef.current.set(
      projectIdToLink,
      URL.createObjectURL(sourceFile),
    );
    setFilesRevision((current) => current + 1);
  }

  const updateExportJob = useCallback((job: AppExportJob) => {
    if (
      wasDeleted(job.projectId, job.projectInstanceId) ||
      !projectsRef.current.some((project) => project.id === job.projectId && project.instanceId === job.projectInstanceId)
    ) {
      return;
    }
    exportJobsRef.current.set(job.projectId, job);
    setExportJobs([...exportJobsRef.current.values()]);
    if (job.status === "saved") {
      const lastExportedAt = new Date().toISOString();
      replaceProjects(projectsRef.current.map((project) =>
        project.id === job.projectId ? { ...project, lastExportedAt } : project,
      ));
      void markProjectExported(job.projectId, lastExportedAt, job.projectInstanceId).catch((cause) => {
        if (cause instanceof ProjectSourceConflictError) { sourceFailure(job.projectId, cause); return; }
        if (mountedRef.current) setError("The video was exported, but its exported status could not be saved in browser storage.");
      });
    }
  }, []);

  const drainVideoExportQueue = useCallback(async () => {
    if (exportWorkerRunningRef.current) return;
    exportWorkerRunningRef.current = true;
    try {
      while (exportQueueRef.current.length > 0) {
        const request = exportQueueRef.current.shift();
        if (
          !request ||
          wasDeleted(request.projectId, request.projectInstanceId) ||
          !projectsRef.current.some(
            (project) => project.id === request.projectId && project.instanceId === request.projectInstanceId,
          )
        ) {
          continue;
        }
        const startedJob: AppExportJob = {
          projectId: request.projectId,
          projectName: request.projectName,
          projectInstanceId: request.projectInstanceId,
          status: "exporting",
          progress: 1,
          speed: null,
          etaSeconds: null,
          detail: "Preparing the final video from the original local source…",
        };
        updateExportJob(startedJob);
        try {
          if (!request.target) {
            const workerUrl = new URL(
              `${import.meta.env.BASE_URL}volleycut-export-sw.js`,
              window.location.href,
            ).href;
            const readiness = await prepareServiceWorkerStreamDownload(workerUrl);
            if (!readiness.ready) {
              throw new Error(readiness.reason ?? "Direct video download is unavailable. Retry the export.");
            }
            // The project can be deleted while worker activation is pending.
            if (wasDeleted(request.projectId, request.projectInstanceId) || !projectsRef.current.some((project) =>
              project.id === request.projectId && project.instanceId === request.projectInstanceId)) continue;
          }
          const { exportRawQualityReel } = await import(
            "@/lib/on-device/export"
          );
          // Readiness and module loading can outlive the source proof or project
          // generation captured when the export was queued. Keep its frozen cuts,
          // but confirm its immutable File against storage before encoding.
          await validateQueuedProject({ id: request.projectId,
            instanceId: request.projectInstanceId, sourceFile: request.sourceFile });
          await exportRawQualityReel(
            request.sourceFile,
            request.intervals,
            (progress) => {
              const percent =
                progress.totalSeconds > 0
                  ? Math.round(
                      (progress.completedSeconds / progress.totalSeconds) *
                        100,
                    )
                  : 0;
              const speed =
                progress.elapsedSeconds > 0 && progress.completedSeconds > 0
                  ? progress.completedSeconds / progress.elapsedSeconds
                  : null;
              const remainingSeconds = Math.max(
                0,
                progress.totalSeconds - progress.completedSeconds,
              );
              updateExportJob({
                ...startedJob,
                progress: Math.max(1, Math.min(99, percent)),
                speed,
                etaSeconds:
                  speed && speed > 0 && remainingSeconds > 0
                    ? remainingSeconds / speed
                    : null,
                detail: progress.detail,
              });
            },
            request.duration,
            () => undefined,
            request.target ? "compatible" : "stream-download",
            {
              scoreOverlay: request.scoreOverlay,
              target: request.target ?? undefined,
            },
          );
          updateExportJob({
            ...startedJob,
            status: "saved",
            progress: 100,
            speed:
              exportJobsRef.current.get(request.projectId)?.speed ?? null,
            etaSeconds: 0,
            detail: "Your final MP4 was saved.",
          });
        } catch (cause) {
          if (cause instanceof ProjectSourceConflictError) sourceFailure(request.projectId, cause, request.projectInstanceId);
          updateExportJob({
            ...startedJob,
            status: "error",
            progress: 0,
            detail:
              cause instanceof DOMException && cause.name === "AbortError"
                ? "Video export was canceled."
                : `Could not create the video: ${cause instanceof Error ? cause.message : String(cause)}`,
          });
        }
      }
    } finally {
      exportWorkerRunningRef.current = false;
    }
  }, [updateExportJob]);

  const queueProjectVideoExport = useCallback(
    (projectIdToExport: string, request: DesignVideoExportRequest) => {
      const current = exportJobsRef.current.get(projectIdToExport);
      if (
        current?.status === "queued" ||
        current?.status === "exporting" ||
        exportTargetRequestsRef.current.has(projectIdToExport)
      ) {
        return;
      }
      const project = projectsRef.current.find(
        (candidate) => candidate.id === projectIdToExport,
      );
      const sourceFile = filesRef.current.get(projectIdToExport);
      if (!project?.instanceId || !sourceFile) return;
      const enqueueWithTarget = (target: VideoExportTarget | null) => {
        exportTargetRequestsRef.current.delete(projectIdToExport);
        const latest = exportJobsRef.current.get(projectIdToExport);
        if (
          wasDeleted(projectIdToExport, project.instanceId) ||
          projectsRef.current.find((candidate) => candidate.id === project.id)?.instanceId !== project.instanceId ||
          latest?.status === "queued" ||
          latest?.status === "exporting"
        ) {
          return;
        }
        const waitsForAnotherExport =
          exportWorkerRunningRef.current || exportQueueRef.current.length > 0;
        exportQueueRef.current.push({
          ...request,
          projectId: project.id,
          projectName: project.source.name,
          projectInstanceId: project.instanceId!,
          sourceFile,
          duration: project.info.duration,
          target,
        });
        updateExportJob({
          projectId: project.id,
          projectName: project.source.name,
          projectInstanceId: project.instanceId!,
          status: "queued",
          progress: 0,
          speed: null,
          etaSeconds: null,
          detail: waitsForAnotherExport
            ? "Waiting for the current video export to finish…"
            : "MP4 export queued…",
        });
        void drainVideoExportQueue();
      };

      exportTargetRequestsRef.current.add(projectIdToExport);
      try {
        const targetRequest = requestVideoExportTarget(sourceFile.name);
        if (!targetRequest) {
          enqueueWithTarget(null);
          return;
        }
        void targetRequest
          .then((target) => enqueueWithTarget(target))
          .catch((cause) => {
            exportTargetRequestsRef.current.delete(projectIdToExport);
            if (cause instanceof DOMException && cause.name === "AbortError") {
              return;
            }
            updateExportJob({
              projectId: project.id,
              projectName: project.source.name,
          projectInstanceId: project.instanceId!,
              status: "error",
              progress: 0,
              speed: null,
              etaSeconds: null,
              detail: `Could not choose the MP4 destination: ${cause instanceof Error ? cause.message : String(cause)}`,
            });
          });
      } catch (cause) {
        exportTargetRequestsRef.current.delete(projectIdToExport);
        updateExportJob({
          projectId: project.id,
          projectName: project.source.name,
          projectInstanceId: project.instanceId!,
          status: "error",
          progress: 0,
          speed: null,
          etaSeconds: null,
          detail: `Could not choose the MP4 destination: ${cause instanceof Error ? cause.message : String(cause)}`,
        });
      }
    },
    [drainVideoExportQueue, updateExportJob],
  );

  function replacePreviewUrl(next: string | null) {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = next;
    setPreviewUrl(next);
  }

  function resetCandidate() {
    candidateRevisionRef.current += 1;
    replacePreviewUrl(null);
    setFile(null);
    setInfo(null);
    setRoi(FULL_FRAME_ROI);
    setAnalysisWindow({ start: 0, end: 0 });
    setSideSwitchEnabled(false);
    setCandidateProgress(null);
    setError(null);
    setWorkState("empty");
    resumePreviewAfterSeek.current = false;
  }

  function selectProject(projectIdToSelect: string | null) {
    candidateRevisionRef.current += 1;
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
        const searchParams = new URLSearchParams(window.location.search);
        const startNewProject = searchParams.get("new") === "1";
        if (startNewProject) {
          searchParams.delete("new");
          const nextSearch = searchParams.toString();
          window.history.replaceState(
            window.history.state,
            "",
            `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ""}${window.location.hash}`,
          );
        }
        let lastSelected: string | null = null;
        try {
          lastSelected = window.localStorage.getItem(
            SELECTED_PROJECT_STORAGE_KEY,
          );
        } catch {
          // Project selection still works when localStorage is restricted.
        }
        const initial = startNewProject
          ? null
          : storedProjects.find((project) => project.id === lastSelected) ??
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
      !project?.analysis || !project.instanceId ||
      project.status !== "ready" ||
      project.importedFeedback ||
      isNeuralModelId(project.analysis.modelId) ||
      (project.analysis.suppression &&
        project.analysis.productionServeOutputs &&
        project.analysis.productionStateOutputs) ||
      !project.analysis.featureNames ||
      !project.analysis.featureValues ||
      scoreRecoveryIdsRef.current.has(project.id) ||
      suppressionAugmentingRef.current.has(project.instanceId)
    ) {
      return;
    }
    let active = true;
    const augmentation = augmentStoredAnalysisWithSuppression(
      project.analysis,
      project.analysisWindow,
    );
    const identity: ProjectWorkIdentity = { id: project.id, instanceId: project.instanceId, source: project.source };
    suppressionAugmentingRef.current.set(identity.instanceId, { source: identity.source, promise: augmentation });
    void augmentation
      .then(async (analysis) => {
        if (!active || !analysis) return;
        const latest = currentWorkProject(identity);
        if (!latest?.analysis) return;
        await commitAnalysisProject(identity, {
          ...latest,
          analysis: { ...latest.analysis, suppression: analysis.suppression,
            productionComponents: analysis.productionComponents,
            productionServeOutputs: analysis.productionServeOutputs,
            productionStateOutputs: analysis.productionStateOutputs },
          updatedAt: new Date().toISOString(),
        });
      })
      .catch((cause) => {
        if (active) {
          if (cause instanceof ProjectSourceConflictError) { sourceFailure(project.id, cause, identity.instanceId); return; }
          setError(
            `VolleySplice could not refresh this saved project: ${cause instanceof Error ? cause.message : String(cause)}`,
          );
        }
      })
      .finally(() => {
        suppressionAugmentingRef.current.delete(identity.instanceId);
      });
    return () => {
      active = false;
    };
  }, [selectedProjectId, projects]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      activeAbortRef.current?.abort();
      activeMediaRef.current?.media.input.dispose();
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      for (const url of videoUrlsRef.current.values()) URL.revokeObjectURL(url);
      videoUrlsRef.current.clear();
    };
  }, []);

  async function chooseFile(selected: File | null) {
    if (!selected) return;
    candidateRevisionRef.current += 1;
    if (safariUnsupported) {
      setError(
        "Safari is not supported. Open VolleySplice in Google Chrome on your desktop computer instead.",
      );
      setWorkState("error");
      return;
    }
    if (!webCodecsReady) {
      setError(
        secureContext
          ? "This browser cannot open the video. Please use the latest version of Chrome or Edge."
          : "VolleySplice needs a secure connection before it can open your video.",
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
      detail: "Getting your video ready",
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

  async function importFeedback(selected: File | null) {
    if (!selected) return;
    setFeedbackImporting(true);
    setError(null);
    try {
      const { project } = importModelFeedbackProject(await selected.text(), {
        occupiedProjectIds: new Set(
          projectsRef.current.map((candidate) => candidate.id),
        ),
      });
      resetCandidate();
      const saved = await putProject(project, { expectedSource: null });
      reconcileProject(saved);
      setSelectedProjectId(saved.id);
    } catch (cause) {
      setError(
        `Could not open that saved project: ${cause instanceof Error ? cause.message : String(cause)}`,
      );
    } finally {
      setFeedbackImporting(false);
    }
  }

  function enqueueProject(project: VolleySpliceProject) {
    const sourceFile = filesRef.current.get(project.id);
    if (!project.instanceId || !sourceFile || queuedProjectWorkRef.current.has(project.id)) return;
    queuedProjectWorkRef.current.set(project.id, { id: project.id, instanceId: project.instanceId, sourceFile });
    setQueueIds((current) =>
      current.includes(project.id)
        ? current
        : [...current, project.id],
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

  async function createAndQueueProject() {
    if (!file || !info || checkingProjectSourceRef.current) return;
    const normalizedWindow = normalizeAnalysisWindow(
      analysisWindow,
      info.duration,
    );
    if (
      normalizedWindow.end - normalizedWindow.start <
      MIN_ANALYSIS_WINDOW_SECONDS
    ) {
      setError(
        `Choose at least ${MIN_ANALYSIS_WINDOW_SECONDS} seconds of game footage.`,
      );
      return;
    }
    const source = projectSource(file);
    const { id } = resolveProjectSetup(projectsRef.current, { source, info, analysisWindow: normalizedWindow, roi, rallyModel, sideSwitchEnabled });
    const revision = candidateRevisionRef.current;
    checkingProjectSourceRef.current = true;
    setCheckingProjectSource(true);
    try {
      let existing = await getProject(id);
      const captured = projectsRef.current.find((project) => project.id === id);
      if (existing && captured && existing.instanceId !== captured.instanceId) {
        throw new ProjectSourceConflictError(existing, "This project instance changed. Reopen it before reusing its saved analysis.");
      }
      if (existing) {
        if (!(await sourceCanReconnectFile(existing.source, file))) {
          throw new ProjectSourceConflictError(existing,
            `This video does not match ${existing.source.name}. Choose the original video to reuse its saved analysis.`);
        }
        existing = await confirmProjectSource(id, existing.source, existing.instanceId);
      }
      if (!mountedRef.current || revision !== candidateRevisionRef.current) return;
      if (existing && wasDeleted(id, existing.instanceId)) throw new ProjectSourceConflictError(null);
      if (existing?.status === "ready" && existing.analysis) {
        reconcileProject(existing);
        linkSourceFile(id, file, existing);
        setSelectedProjectId(id);
        resetCandidate();
        return;
      }
      const now = new Date().toISOString();
      const project: VolleySpliceProject = {
        schemaVersion: 1,
        id,
        instanceId: existing?.instanceId,
        source: existing?.source ?? source,
        info,
        analysisWindow: normalizedWindow,
        roi,
        servingSideEnabled: true,
        sideSwitchEnabled,
        rallyModel,
        status: "queued",
        analysis: null,
        error: null,
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
      };
      const queued = await commitSourceProject(project, existing?.source ?? null);
      if (!mountedRef.current || revision !== candidateRevisionRef.current) return;
      linkSourceFile(id, file, queued);
      setSelectedProjectId(id);
      enqueueProject(queued);
      resetCandidate();
    } catch (cause) {
      if (revision === candidateRevisionRef.current) sourceFailure(id, cause);
    } finally {
      checkingProjectSourceRef.current = false;
      if (mountedRef.current) setCheckingProjectSource(false);
    }
  }

  function updateScoreJob(projectId: string, status: "queued" | "running" | "cancelling", detail: string) {
    if (mountedRef.current) setScoreJobs((current) => ({ ...current, [projectId]: { status, detail } }));
  }

  function retryScoreSpecialists(projectId: string) {
    const project = projectsRef.current.find((candidate) => candidate.id === projectId);
    if (!project?.analysis || project.status !== "ready" || !filesRef.current.has(projectId) ||
        !project.instanceId || scoreRecoveryIdsRef.current.has(projectId) || wasDeleted(projectId, project.instanceId) ||
        missingScoreStages(project).length === 0) return;
    scoreRecoveryIdsRef.current.add(projectId);
    updateScoreJob(projectId, "queued", "Score markers queued. Your review stays available.");
    enqueueProject(project);
  }

  function cancelScoreSpecialists(projectId: string) {
    if (!scoreRecoveryIdsRef.current.has(projectId)) return;
    if (activeProjectWorkRef.current === queuedProjectWorkRef.current.get(projectId)) {
      updateScoreJob(projectId, "cancelling", "Stopping score markers…");
      activeAbortRef.current?.abort();
    } else {
      scoreRecoveryIdsRef.current.delete(projectId);
      queuedProjectWorkRef.current.delete(projectId);
      setQueueIds((current) => current.filter((id) => id !== projectId));
      setScoreJobs((current) => { const next = { ...current }; delete next[projectId]; return next; });
    }
  }

  async function runScoreStages(project: StoredProject, media: OpenedMedia, abort: AbortController) {
    const projectId = project.id;
    if (!project?.analysis) return;
    const stages = missingScoreStages(project);
    if (!stages.length) return;
    const stepId = (key: ScoreSpecialistStage) => key === "servingSide" ? "serving-side" as const : "side-switch" as const;
    const currentProject = () => currentWorkProject(project);
    const check = () => { abort.signal.throwIfAborted(); if (!currentProject()) throw new DOMException("Analysis cancelled", "AbortError"); };
    const saveFailure = async (message: string) => {
      const latest = currentProject();
      if (!latest) return;
      const missing = missingScoreStages(latest).filter((key) => stages.includes(key));
      const errors = { ...latest.scoreSpecialistErrors };
      for (const key of missing) errors[key] = message;
      await commitAnalysisProject(project, { ...latest, scoreSpecialistErrors: errors, updatedAt: new Date().toISOString() });
      for (const key of missing) setActiveInferenceSteps((current) => finishInferenceStep(current, stepId(key), message, performance.now(), "error"));
    };
    const progress = (key: ScoreSpecialistStage) => (value: { stage: "loading" | "frames" | "features" | "complete"; completed: number; total: number; detail: string }) => {
      check();
      setActiveProgress({ ...value, stage: "inference" });
      setActiveInferenceSteps((current) => updateSpecialistInferenceStep(current, stepId(key), value, performance.now()));
      if (scoreRecoveryIdsRef.current.has(projectId)) updateScoreJob(projectId, "running", value.detail);
    };
    if (scoreRecoveryIdsRef.current.has(projectId)) {
      setActiveInferenceSteps(createInferenceProgressSteps(stages.map(stepId)));
    }
    // Also serialize callbacks from callers that do not await onOutput, so a
    // partial stage is durable before failure handling or queue release.
    let outputWrites = Promise.resolve();
    try {
      check();
      let analysis = project.analysis;
      if ((stages.includes("servingSide") && !analysis.productionServeOutputs) ||
          (stages.includes("sideSwitch") && (!analysis.productionComponents || !analysis.productionStateOutputs))) {
        if (!analysis.featureNames || !analysis.featureValues) throw new Error("This saved analysis has no score evidence. Add markers manually.");
        const cached = suppressionAugmentingRef.current.get(project.instanceId);
        const cachedEvidence = await awaitWithAbort(Promise.resolve(cached && sameProjectSource(cached.source, project.source)
          ? cached.promise : undefined), abort.signal);
        const evidence = cachedEvidence ?? await awaitWithAbort(runProductionInferenceFromFeatures({ times: analysis.times, values: analysis.featureValues,
          rows: analysis.times.length, columns: analysis.featureNames.length, names: analysis.featureNames }, project.analysisWindow), abort.signal);
        check();
        analysis = { ...analysis, ...(cachedEvidence?.suppression ? { suppression: cachedEvidence.suppression } : {}),
          productionServeOutputs: evidence.productionServeOutputs,
          productionStateOutputs: evidence.productionStateOutputs, productionComponents: evidence.productionComponents };
        await commitAnalysisProject(project, { ...currentProject()!, analysis });
      }
      const { inferScoreSpecialists } = await awaitWithAbort(import("@/lib/on-device/score-specialists"), abort.signal);
      check();
      const acceptOutput = (output: { servingSide?: NonNullable<OnDeviceAnalysis["servingSide"]>; sideSwitch?: NonNullable<OnDeviceAnalysis["sideSwitch"]> }) => {
        const write = outputWrites.then(async () => {
          check();
          const latest = currentProject()!;
          const merged = { ...latest.analysis!, ...output };
          const readiness = scoreSpecialistReadiness({ ...latest, analysis: merged });
          const errors = { ...latest.scoreSpecialistErrors };
          for (const key of stages) {
            if (!output[key]) continue;
            if (readiness[key].status !== "complete") throw new Error("The score predictions did not match this analysis. Retry score markers.");
            delete errors[key];
          }
          await commitAnalysisProject(project, { ...latest, analysis: merged, scoreSpecialistErrors: errors, updatedAt: new Date().toISOString() });
        });
        outputWrites = write;
        void write.catch(() => undefined);
        return write;
      };
      const output = await inferScoreSpecialists(media, project.roi, {
        servingSide: stages.includes("servingSide") ? { analysis, onProgress: progress("servingSide") } : undefined,
        sideSwitch: stages.includes("sideSwitch") ? { analysis: { ...analysis,
          productionComponents: analysis.productionComponents!, productionStateOutputs: analysis.productionStateOutputs! }, onProgress: progress("sideSwitch") } : undefined,
      }, { signal: abort.signal, onOutput: acceptOutput });
      await acceptOutput(output);
      if (missingScoreStages(currentProject()!).some((key) => stages.includes(key))) {
        throw new Error("Score predictions were not returned. Retry score markers.");
      }
    } catch (cause) {
      try { await outputWrites; } catch (writeFailure) {
        if (writeFailure instanceof ProjectSourceConflictError) throw writeFailure;
      }
      if (cause instanceof ProjectSourceConflictError) throw cause;
      await saveFailure(abort.signal.aborted ? "Score markers cancelled. Retry when ready." :
        `Score markers failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
  }

  async function runScoreRecovery(work: QueuedProjectWork) {
    const projectId = work.id;
    const abort = new AbortController();
    activeAbortRef.current = abort;
    updateScoreJob(projectId, "running", "Opening the original video for score markers…");
    let media: OpenedMedia | null = null;
    let project: StoredProject | null = null;
    let releaseWakeLock = () => Promise.resolve();
    try {
      project = await validateQueuedProject(work, abort.signal);
      if (!project.analysis || project.status !== "ready") return;
      releaseWakeLock = await holdScreenWakeLock(setWakeLockState);
      abort.signal.throwIfAborted();
      media = await openLocalMedia(work.sourceFile);
      activeMediaRef.current = { projectId, media };
      abort.signal.throwIfAborted();
      await runScoreStages(project, media, abort);
    } catch (cause) {
      if (cause instanceof ProjectSourceConflictError || !project) {
        sourceFailure(projectId, cause, work.instanceId);
        return;
      }
      try {
        const latest = currentWorkProject(project);
        if (!latest) return;
        const errors = { ...latest.scoreSpecialistErrors };
        for (const key of missingScoreStages(latest)) errors[key] = abort.signal.aborted
          ? "Score markers cancelled. Retry when ready."
          : `Could not open score markers: ${cause instanceof Error ? cause.message : String(cause)}`;
        await commitAnalysisProject(project, { ...latest, scoreSpecialistErrors: errors });
      } catch (writeFailure) {
        sourceFailure(projectId, writeFailure, work.instanceId);
      }
    } finally {
      media?.input.dispose();
      activeMediaRef.current = null;
      activeAbortRef.current = null;
      await releaseWakeLock();
    }
  }

  async function runProject(work: QueuedProjectWork) {
    const projectIdToRun = work.id;
    const sourceFile = work.sourceFile;
    const abort = new AbortController();
    activeAbortRef.current = abort;
    let running: StoredProject | null = null;
    let opened: OpenedMedia | null = null;
    let releaseWakeLock = () => Promise.resolve();
    try {
      const project = await validateQueuedProject(work, abort.signal);
      const source = project.source.fingerprint !== undefined
        ? project.source
        : await sourceFileFingerprint(sourceFile)
            .then((fingerprint) => ({ ...project.source, fingerprint }))
            .catch(() => project.source);
      abort.signal.throwIfAborted();
      running = await commitSourceProject({
        ...project,
        source,
        status: "analyzing",
        error: null,
        updatedAt: new Date().toISOString(),
      }, project.source);
      const sideSwitchEnabled = running.sideSwitchEnabled !== false;
      setActiveInferenceSteps(
        createInferenceProgressSteps([
          ...CORE_INFERENCE_STEP_IDS,
          ...SCORE_INFERENCE_STEP_IDS.filter(
            (step) => step !== "side-switch" || sideSwitchEnabled,
          ),
        ]),
      );
      setActiveProgress({
        stage: "opening",
        completed: 0,
        total: 1,
        detail: "Getting your video ready",
      });
      releaseWakeLock = await holdScreenWakeLock(setWakeLockState);
      abort.signal.throwIfAborted();
      opened = await openLocalMedia(sourceFile);
      activeMediaRef.current = { projectId: projectIdToRun, media: opened };
      abort.signal.throwIfAborted();
      const result = await analyzeOpenedMedia(
        opened,
        running.roi,
        "local-source" satisfies OnDeviceAnalysis["featurePath"],
        DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
        (nextProgress) => {
          if (
            !wasDeleted(projectIdToRun, work.instanceId) &&
            mountedRef.current
          ) {
            setActiveProgress(nextProgress);
            setActiveInferenceSteps((current) =>
              updatePipelineInferenceSteps(
                current,
                nextProgress,
                performance.now(),
              ),
            );
          }
        },
        {
          name: sourceFile.name,
          size: sourceFile.size,
          lastModified: sourceFile.lastModified,
        },
        {
          detailedProfiling: false,
          rallyModel: running.rallyModel ?? "ensemble",
          signal: abort.signal,
          decodeStrategy: DEFAULT_VIDEO_DECODE_STRATEGY,
          decoderAcceleration: VIDEO_DECODER_HARDWARE_ACCELERATION,
          reductionKernel: DEFAULT_FEATURE_REDUCTION_KERNEL,
          analysisWindow: running.analysisWindow,
        },
      );
      if (!mountedRef.current || abort.signal.aborted || wasDeleted(projectIdToRun, work.instanceId)) return;
      // Save rally evidence before specialists run; their failure cannot discard it.
      const checkpoint = await commitAnalysisProject(running, { ...running, analysis: result, scoreSpecialistErrors: {} });
      await runScoreStages(checkpoint, opened, abort);
      const latest = currentWorkProject(running);
      if (!latest) return;
      await commitAnalysisProject(running, { ...latest, status: "ready", error: null, updatedAt: new Date().toISOString() });
    } catch (cause) {
      if (cause instanceof ProjectSourceConflictError || !running) { sourceFailure(projectIdToRun, cause, work.instanceId); return; }
      try {
        const latest = currentWorkProject(running);
        if (!latest) return;
        await commitAnalysisProject(running, { ...latest, status: "error",
          error: cause instanceof Error ? cause.message : String(cause), updatedAt: new Date().toISOString() });
      } catch (writeFailure) {
        sourceFailure(projectIdToRun, writeFailure, work.instanceId);
      }
    } finally {
      opened?.input.dispose();
      activeMediaRef.current = null;
      activeAbortRef.current = null;
      await releaseWakeLock();
      if (running && wasDeleted(projectIdToRun, work.instanceId)) {
        await deleteNeuralEmbeddingsForSource(running.source).catch(() => undefined);
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
    const work = queuedProjectWorkRef.current.get(next);
    if (!work) { setQueueIds((current) => current.filter((id) => id !== next)); return; }
    activeJobRef.current = next;
    activeProjectWorkRef.current = work;
    setActiveJobId(next);
    const recovery = scoreRecoveryIdsRef.current.has(next);
    void (recovery ? runScoreRecovery(work) : runProject(work)).finally(() => {
      // A same-ID project may have been recreated and queued while the old
      // lifetime was unwinding. Only remove the entry this invocation owned.
      const ownsEntry = queuedProjectWorkRef.current.get(next) === work;
      if (ownsEntry) {
        queuedProjectWorkRef.current.delete(next);
        scoreRecoveryIdsRef.current.delete(next);
      }
      activeJobRef.current = null;
      activeProjectWorkRef.current = null;
      if (mountedRef.current) {
        if (ownsEntry) setScoreJobs((current) => { const updated = { ...current }; delete updated[next]; return updated; });
        setQueueIds((current) => ownsEntry ? current.filter((id) => id !== next) : [...current]);
        setActiveJobId(null);
        setActiveProgress(null);
        setActiveInferenceSteps([]);
        setWakeLockState("idle");
      }
    });
  }, [queueIds]);

  function queueAttachedProject(project: VolleySpliceProject) {
    const selected = filesRef.current.get(project.id);
    if (selected) void attachSource(project, selected, true).catch(() => undefined);
  }

  async function attachSource(
    project: VolleySpliceProject,
    selected: File | null,
    forceQueue = false,
  ) {
    if (!selected) return;
    try {
      // File bytes are immutable; a retry can reuse its proof only while the
      // persisted source identity still matches the identity validated at link.
      const approved = forceQueue && filesRef.current.get(project.id) === selected
        ? validatedSourcesRef.current.get(project.id) : undefined;
      const checked = await validateStoredSource(project.id, selected, project.instanceId, approved);
      let attached = await confirmProjectSource(project.id, checked.source, checked.instanceId);
      const queue = forceQueue || attached.status !== "ready";
      if (queue) {
        attached = await commitSourceProject({ ...attached, status: "queued", error: null,
          updatedAt: new Date().toISOString() }, attached.source);
      }
      if (!mountedRef.current || wasDeleted(project.id, project.instanceId)) throw new ProjectSourceConflictError(null);
      reconcileProject(attached);
      linkSourceFile(project.id, selected, attached);
      setError(null);
      if (queue) enqueueProject(attached);
    } catch (cause) {
      sourceFailure(project.id, cause, project.instanceId);
      throw cause;
    }
  }

  async function removeSelectedProject() {
    if (!selectedProject) return;
    if (
      !window.confirm(
        `Delete ${selectedProject.source.name} and all of its saved edits from this device?`,
      )
    )
      return;

    try {
      await deleteProject(selectedProject.id, selectedProject.instanceId);
    } catch (cause) {
      sourceFailure(selectedProject.id, cause);
      return;
    }
    deletedProjectInstancesRef.current.add(selectedProject.instanceId ?? "");
    if (activeMediaRef.current?.projectId === selectedProject.id) {
      activeAbortRef.current?.abort();
      activeMediaRef.current.media.input.dispose();
    }
    setQueueIds((current) => current.filter((id) => id !== selectedProject.id));
    queuedProjectWorkRef.current.delete(selectedProject.id);
    scoreRecoveryIdsRef.current.delete(selectedProject.id);
    setScoreJobs((current) => { const next = { ...current }; delete next[selectedProject.id]; return next; });
    exportQueueRef.current = exportQueueRef.current.filter(
      (job) => job.projectId !== selectedProject.id,
    );
    exportTargetRequestsRef.current.delete(selectedProject.id);
    exportJobsRef.current.delete(selectedProject.id);
    setExportJobs([...exportJobsRef.current.values()]);
    filesRef.current.delete(selectedProject.id);
    validatedSourcesRef.current.delete(selectedProject.id);
    const linkedVideoUrl = videoUrlsRef.current.get(selectedProject.id);
    if (linkedVideoUrl) URL.revokeObjectURL(linkedVideoUrl);
    videoUrlsRef.current.delete(selectedProject.id);
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
        window.localStorage.removeItem(editHistoryKey(selectedProject.id, selectedProject.instanceId));
        if (selectedProject.legacyDraftStorage) window.localStorage.removeItem(editHistoryKey(selectedProject.id));
        for (const key of projectDraftStorageKeys(selectedProject, analysisId))
          window.localStorage.removeItem(key);
      } catch {
        // IndexedDB deletion remains useful if localStorage is restricted.
      }
    }
    await Promise.allSettled([
      deleteFeatureCachesForSource(selectedProject.source),
      deleteNeuralEmbeddingsForSource(selectedProject.source),
    ]);
  }

  const activePercent =
    activeInferenceSteps.length > 0
      ? overallInferenceProgress(activeInferenceSteps) * 100
      : progressPercent(activeProgress);
  const analysisQueueLabel = activeJobId
    ? `${projects.find((project) => project.id === activeJobId)?.source.name ?? "Project"} · ${Math.round(activePercent)}%${queueIds.length > 1 ? ` · ${queueIds.length - 1} queued` : ""}`
    : queueIds.length > 0
      ? `${queueIds.length} queued`
      : null;
  const activeVideoExport = exportJobs.find(
    (job) => job.status === "exporting",
  );
  const queuedVideoExportCount = exportJobs.filter(
    (job) => job.status === "queued",
  ).length;
  const exportQueueLabel = activeVideoExport
    ? `${activeVideoExport.projectName} · export ${activeVideoExport.progress}%${queuedVideoExportCount > 0 ? ` · ${queuedVideoExportCount} queued` : ""}`
    : queuedVideoExportCount > 0
      ? `${queuedVideoExportCount} export ${queuedVideoExportCount === 1 ? "queued" : "jobs queued"}`
      : null;
  const queueLabel = [analysisQueueLabel, exportQueueLabel]
    .filter(Boolean)
    .join(" · ") || null;
  const activeStep = activeInferenceSteps.find(
    (step) => step.status === "running",
  );
  const analysisActivity: DesignWorkActivity[] = projects
    .filter(
      (project): project is VolleySpliceProject & {
        status: "analyzing" | "queued";
      } => project.status === "analyzing" || project.status === "queued",
    )
    .map((project) => {
      const isActive = project.id === activeJobId;
      const queueIndex = queueIds.indexOf(project.id);
      return {
        projectId: project.id,
        name: project.source.name,
        kind: "analysis",
        status: project.status,
        progress: isActive ? Math.round(activePercent) : null,
        detail: isActive
          ? (activeProgress?.detail ?? activeStep?.detail ?? "Analyzing locally")
          : queueIndex <= 0
            ? "Next to analyze"
            : `${queueIndex} ${queueIndex === 1 ? "video" : "videos"} ahead`,
      };
    });
  const designExportJobs: DesignExportJob[] = exportJobs;
  const exportActivity: DesignWorkActivity[] = designExportJobs
    .filter(
      (job) =>
        job.status === "queued" ||
        job.status === "exporting",
    )
    .map((job) => ({
      projectId: job.projectId,
      name: job.projectName,
      kind: "export",
      status: job.status as "queued" | "exporting",
      progress: job.progress,
      detail: job.detail,
    }));
  const scoreActivity: DesignWorkActivity[] = Object.entries(scoreJobs).flatMap(([projectId, job]) => {
    const project = projects.find((candidate) => candidate.id === projectId);
    return project ? [{ projectId, name: project.source.name, kind: "analysis" as const,
      status: job.status === "queued" ? "queued" as const : "analyzing" as const,
      progress: projectId === activeJobId ? activePercent : null, detail: job.detail }] : [];
  });
  const workActivity = [...analysisActivity, ...scoreActivity, ...exportActivity];
  const projectHeader = (
    <>
      <MobileAppBanner />
      <ProjectHeader
        projects={projects}
        exportJobs={designExportJobs}
        selectedProjectId={selectedProjectId}
        queueLabel={queueLabel}
        onSelectProject={selectProject}
        onDeleteProject={() => void removeSelectedProject()}
      />
    </>
  );

  const productAnalysis = useMemo<ProductAnalysis | null>(() => {
    if (!selectedProject?.analysis || selectedProject.status !== "ready")
      return null;
    return {
      id: projectAnalysisId(selectedProject)!,
      recordingId: selectedProject.id,
      kind: "model",
      modelId: selectedProject.analysis.modelId,
      probabilityModelIds: selectedProject.analysis.probabilityModelIds,
      neuralScores: selectedProject.analysis.neuralScores,
      duration: selectedProject.info.duration,
      analysisWindow: selectedProject.analysisWindow,
      width: selectedProject.info.width,
      height: selectedProject.info.height,
      sourceFilename: selectedProject.source.name,
      source: selectedProject.source,
      mediaInfo: selectedProject.info,
      roi: selectedProject.roi,
      runtimeVariant:
        selectedProject.importedFeedback?.runtimeVariant ??
        DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
      videoUrl: selectedVideoUrl,
      rallies: selectedProject.analysis.intervals,
      ignoredIntervals: [
        ...(selectedProject.analysisWindow.start > 0
          ? [
              {
                start: 0,
                end: selectedProject.analysisWindow.start,
                reason: "outside-game-window",
              },
            ]
          : []),
        ...(selectedProject.analysisWindow.end < selectedProject.info.duration
          ? [
              {
                start: selectedProject.analysisWindow.end,
                end: selectedProject.info.duration,
                reason: "outside-game-window",
              },
            ]
          : []),
      ],
      features:
        selectedProject.analysis.featureNames &&
        selectedProject.analysis.featureValues
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
      productionServeOutputs: selectedProject.analysis.productionServeOutputs,
      productionStateOutputs: selectedProject.analysis.productionStateOutputs,
      servingSide: selectedProject.analysis.servingSide,
      sideSwitch: selectedProject.analysis.sideSwitch,
      sideSwitchEnabled: selectedProject.sideSwitchEnabled !== false,
      suppression: selectedProject.analysis.suppression,
    };
  }, [selectedProject, selectedVideoUrl]);

  if (selectedProject && productAnalysis) {
    return (
      <ReadyProjectEditor
        key={`${productAnalysis.id}:${selectedProject.instanceId}`}
        sourceError={error}
        projectId={selectedProject.id}
        storageError={error}
        projects={projects}
        workActivity={workActivity}
        exportJobs={designExportJobs}
        productAnalysis={productAnalysis}
        sourceFile={selectedSourceFile}
        videoUrl={selectedVideoUrl}
        onSelectProject={selectProject}
        onAttachSource={(selected) => attachSource(selectedProject, selected)}
        onQueueVideoExport={(request) =>
          queueProjectVideoExport(selectedProject.id, request)
        }
        onDeleteProject={() => void removeSelectedProject()}
        scoreRecovery={{
          stages: scoreSpecialistReadiness(selectedProject),
          status: scoreJobs[selectedProject.id]?.status ?? "idle",
          detail: scoreJobs[selectedProject.id]?.detail ?? "",
          steps: selectedProject.id === activeJobId && scoreRecoveryIdsRef.current.has(selectedProject.id) ? activeInferenceSteps : [],
          retry: () => retryScoreSpecialists(selectedProject.id),
          cancel: () => cancelScoreSpecialists(selectedProject.id),
        }}
      />
    );
  }

  const selectedIsActive = selectedProject?.id === activeJobId;
  const displayedProgress = selectedIsActive ? activeProgress : candidateProgress;
  const percent = selectedIsActive ? activePercent : progressPercent(displayedProgress);

  return (
    <main className={styles.page} data-design="classic">
      {projectHeader}

      {!selectedProject && (
        <>
          <div className={styles.sourceStart} data-layout="classic">
            <section className={styles.hero}>
            <p>{LANDING_COPY.kicker}</p>
            <h1>{LANDING_COPY.title}</h1>
            <p className={styles.lede}>
              {LANDING_COPY.description}
            </p>
            <div
              className={styles.pipeline}
              role="list"
              aria-label="How VolleySplice works"
              data-count={LANDING_COPY.steps.length}
            >
              {LANDING_COPY.steps.map((step, index) => (
                <span key={step} role="listitem"><b>{index + 1}</b><i>{step}</i></span>
              ))}
            </div>
            <p className={styles.privacyNote}>Your video stays on this device.</p>
            </section>

            {safariUnsupported ? (
              <p className={styles.notice} role="status">
                <strong>Safari is not supported.</strong> VolleySplice cannot open
                videos here yet. Please use Google Chrome on your desktop computer
                to continue.
              </p>
            ) : null}
            {!safariUnsupported && !secureContext && (
              <p className={styles.notice}>
                VolleySplice needs a secure connection before it can open your video.
              </p>
            )}
            {error && <p className={styles.error}>{error}</p>}

            <section className={styles.importCard} data-tour="source-picker">
            <div>
              <p>STEP 1 OF 3</p>
              <h2>{file?.name ?? "Choose your game video"}</h2>
              <p>
                {file
                  ? `${compactBytes(file.size)} · ready to set up`
                  : "Pick a video from this device. Nothing will be uploaded."}
              </p>
            </div>
            <div className={styles.importActions}>
              <label
                className={styles.fileButton}
                data-disabled={busy || safariUnsupported || undefined}
              >
                {file ? "Choose a different video" : "Choose video"}
                <input
                  type="file"
                  accept="video/*,.mkv,.webm,.mov,.mp4,.m4v"
                  disabled={busy || safariUnsupported}
                  onChange={(event) =>
                    void chooseFile(event.currentTarget.files?.[0] ?? null)
                  }
                />
              </label>
              <details className={styles.resumeDetails}>
                <summary>Open a saved project</summary>
                <label
                  className={`${styles.fileButton} ${styles.feedbackButton}`}
                  data-disabled={busy || undefined}
                >
                  {feedbackImporting ? "Opening…" : "Choose project file"}
                  <input
                    type="file"
                    accept="application/json,.json"
                    disabled={busy}
                    onChange={(event) => {
                      const input = event.currentTarget;
                      const feedbackFile = input.files?.[0] ?? null;
                      input.value = "";
                      void importFeedback(feedbackFile);
                    }}
                  />
                </label>
              </details>
            </div>
            </section>
          </div>

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
                    <span>AREA TO WATCH</span>
                  </div>
                </div>
              </div>

              <aside className={styles.inspector}>
                <p>VIDEO SETUP</p>
                <h2>Where is the game?</h2>
                <p className={styles.cropHelp}>
                  If the whole video is game footage, you can continue right
                  away. Otherwise, mark where the game starts and ends.
                </p>
                <section
                  className={styles.gameWindow}
                  data-tour="source-game-window"
                  aria-labelledby="game-window-heading"
                >
                  <div className={styles.gameWindowHeading}>
                    <div>
                      <span>PART TO USE</span>
                      <strong id="game-window-heading">
                        Mark game start &amp; end
                      </strong>
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
                    Move the video to the right moment, then use the buttons below.
                  </p>
                  <div className={styles.gameBoundaryGrid}>
                    <label>
                      <span>Game start</span>
                      <input
                        aria-label="Game start in seconds"
                        type="number"
                        min="0"
                        max={Math.max(
                          0,
                          analysisWindow.end - MIN_ANALYSIS_WINDOW_SECONDS,
                        )}
                        step="0.1"
                        value={analysisWindow.start}
                        onChange={(event) =>
                          setGameBoundary(
                            "start",
                            event.currentTarget.valueAsNumber,
                          )
                        }
                      />
                      <output>{preciseTime(analysisWindow.start)}</output>
                      <button
                        type="button"
                        onClick={() => markGameBoundary("start")}
                      >
                        Use current time
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
                          setGameBoundary(
                            "end",
                            event.currentTarget.valueAsNumber,
                          )
                        }
                      />
                      <output>{preciseTime(analysisWindow.end)}</output>
                      <button
                        type="button"
                        onClick={() => markGameBoundary("end")}
                      >
                        Use current time
                      </button>
                    </label>
                  </div>
                  <strong className={styles.gameWindowSummary}>
                    {formatDuration(analysisWindow.end - analysisWindow.start)} selected
                    {analysisWindow.end - analysisWindow.start <
                    info.duration - 0.05
                      ? ` · ${formatDuration(info.duration - (analysisWindow.end - analysisWindow.start))} left out`
                      : " · full video"}
                  </strong>
                </section>
                <details className={styles.optionalSetup} data-tour="source-camera">
                  <summary>Adjust the area to watch <span>Optional</span></summary>
                  <p>
                    Use this only when another court, the crowd, or a large border
                    is visible. Keep your court and players inside the yellow box.
                  </p>
                  <div className={styles.presetButtons}>
                    <button
                      type="button"
                      onClick={() => setRoi(FULL_FRAME_ROI)}
                    >
                      Reset to full video
                    </button>
                  </div>
                  <div className={styles.roiGrid}>
                    {(["x", "y", "width", "height"] as const).map((field) => (
                      <label key={field}>
                        <span>
                          {field}{" "}
                          <output>{Math.round(roi[field] * 100)}%</output>
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
                </details>
                <label className={styles.rallyModelChoice}>
                  <strong>Rally detection</strong>
                  <select aria-label="Rally detection model" value={rallyModel} disabled={busy}
                    onChange={event => { if (isRallyModelSelection(event.target.value)) setRallyModel(event.target.value); }}>
                    {RALLY_MODEL_OPTIONS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                  <small>{RALLY_MODEL_OPTIONS.find(option => option.value === rallyModel)?.description}</small>
                </label>
                <label
                  className={styles.sideSwitchToggle}
                  data-enabled={sideSwitchEnabled || undefined}
                >
                  <input
                    type="checkbox"
                    checked={sideSwitchEnabled}
                    onChange={(event) =>
                      setSideSwitchEnabled(event.currentTarget.checked)
                    }
                    disabled={busy}
                  />
                  <span>
                    <strong>Teams change court sides during this video</strong>
                    <small>
                      This is needed for accurate scorekeeping when teams switch
                      ends. VolleySplice will look for the switches; you can add any
                      it misses while reviewing the score.
                    </small>
                  </span>
                </label>
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
                  Find the rallies
                </button>
                <small className={styles.runtimeNote}>
                  This may take a while for a long video. Keep this page open and
                  come back when it is ready.
                </small>
              </aside>
            </section>
          )}

        </>
      )}

      {selectedProject && (
        <section className={styles.projectPanel}>
          <p>
            {selectedProject.status === "ready"
              ? "READY TO REVIEW"
              : "WORKING ON YOUR VIDEO"}
          </p>
          <h1>{selectedProject.source.name}</h1>
          <p>
            {selectedProject.status === "analyzing"
              ? "VolleySplice is finding the rallies. You can review another finished video while this continues."
              : selectedProject.status === "queued"
                ? `Waiting to start${queueIds.indexOf(selectedProject.id) > 0 ? ` · ${queueIds.indexOf(selectedProject.id)} video ahead` : ""}.`
                : (selectedProject.error ??
                  "Choose the original video again so you can continue.")}
          </p>
          {error && <p className={styles.error} role="alert">{error}</p>}
          {(selectedProject.status === "waiting" ||
            selectedProject.status === "error") &&
            (selectedSourceFile ? (
              <button
                className={styles.fileButton}
                type="button"
                onClick={() => queueAttachedProject(selectedProject)}
              >
                Try again
              </button>
            ) : (
              <label className={styles.fileButton}>
                Choose original video
                <input
                  type="file"
                  accept="video/*,.mkv,.webm,.mov,.mp4,.m4v"
                  onChange={(event) =>
                    void attachSource(
                      selectedProject,
                      event.currentTarget.files?.[0] ?? null,
                    ).catch((cause) => {
                      setError(
                        `Could not reconnect the video: ${cause instanceof Error ? cause.message : String(cause)}`,
                      );
                    })
                  }
                />
              </label>
            ))}
          <button
            className={styles.newProjectButton}
            type="button"
            onClick={() => selectProject(null)}
          >
            Start a new video
          </button>
        </section>
      )}

      {selectedIsActive && activeInferenceSteps.length > 0 && (
        <InferenceProgressPanel
          steps={activeInferenceSteps}
          wakeLockActive={wakeLockState === "active"}
        />
      )}

      {displayedProgress &&
        (!selectedIsActive || activeInferenceSteps.length === 0) && (
        <section className={styles.progressCard} aria-live="polite">
          <div>
            <p>
              {selectedIsActive ? "FINDING THE RALLIES" : "OPENING YOUR VIDEO"}
            </p>
            <strong>
              {selectedIsActive
                ? "You can leave this tab open and come back."
                : displayedProgress.detail}
            </strong>
          </div>
          <output>{Math.round(percent)}%</output>
          <div className={styles.progressTrack}>
            <i style={{ width: `${percent}%` }} />
          </div>
          <p>
            Your video stays on this device while VolleySplice works.
          </p>
        </section>
      )}

      {!selectedProject && (
        <GuidedTour stage="source" sourceReady={Boolean(info && previewUrl)} />
      )}
      <SiteFooter />
    </main>
  );
}
