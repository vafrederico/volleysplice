import { useEffect, useRef, useState } from "react";

import {
  createCutDraft,
  cutDraftStorageKey,
  cutDraftStorageKeys,
  alignRallyServeMarkers,
  parseCutDraft,
  type CutDraft,
  type CutDraftSeed,
} from "@/lib/cut-draft";
import {
  listProjects,
  projectAnalysisId,
  putProjectReviewDraft,
  SELECTED_PROJECT_STORAGE_KEY,
  sourceCanReconnectFile,
  type VolleySpliceProject,
} from "@/lib/project-store";
import type { OnDeviceSuppression } from "@/lib/on-device/types";
import type {
  ExportInterval,
  ScoreOverlayOptions,
} from "@/lib/on-device/export";
import type { ProductAnalysis } from "@/lib/product-analysis";
import {
  scoreTrackingWithServingSideOutput,
  scoreTrackingWithSideSwitchOutput,
} from "@/lib/score-tracking-inference";

export type DesignProjectOption = {
  id: string;
  name: string;
  status: VolleySpliceProject["status"];
  exportJob: DesignExportJob | null;
};

export type DesignExportJobStatus =
  | "queued"
  | "exporting"
  | "saved"
  | "error";

export type DesignExportJob = {
  projectId: string;
  projectName: string;
  status: DesignExportJobStatus;
  progress: number;
  speed: number | null;
  etaSeconds: number | null;
  detail: string;
};

export type DesignVideoExportRequest = {
  intervals: readonly ExportInterval[];
  scoreOverlay?: ScoreOverlayOptions;
};

export type DesignWorkActivity = {
  projectId: string;
  name: string;
  kind: "analysis" | "export";
  status: "analyzing" | "queued" | "exporting";
  progress: number | null;
  detail: string;
};

export type DesignCleanupSuggestion = {
  id: string;
  start: number;
  end: number;
  score: number;
  cutId: string | null;
  decision: "pending" | "keep" | "suppress";
};

export type ReadyDesignReview = {
  state: "ready";
  projectId: string;
  projectName: string;
  sourceName: string;
  sourceSize: number;
  sourceNeedsReconnect: boolean;
  sourceFile: File | null;
  videoUrl: string | null;
  productAnalysis: ProductAnalysis | null;
  duration: number;
  width: number;
  height: number;
  gameStart: number;
  gameEnd: number;
  cropCourt: boolean;
  sideSwitchEnabled: boolean;
  projectStatus: VolleySpliceProject["status"];
  draft: CutDraft;
  draftSeed: CutDraftSeed;
  suppression: OnDeviceSuppression | undefined;
  cleanupSuggestions: DesignCleanupSuggestion[];
  projects: DesignProjectOption[];
  workActivity: DesignWorkActivity[];
  exportJob: DesignExportJob | null;
  selectProject: (projectId: string | null) => void;
  attachSource: (file: File) => Promise<{
    ok: boolean;
    message: string;
    videoUrl: string | null;
  }>;
  queueVideoExport: (request: DesignVideoExportRequest) => void;
  saveDraft: (draft: CutDraft) => void;
};

export type DesignReviewState =
  | { state: "loading" }
  | {
      state: "empty" | "unavailable" | "error";
      message: string;
      projects: DesignProjectOption[];
      selectProject: (projectId: string | null) => void;
    }
  | ReadyDesignReview;

function reloadSelectedProject(projectId: string | null): void {
  if (projectId === null) {
    window.location.assign("/?new=1");
    return;
  }
  try {
    window.localStorage.setItem(SELECTED_PROJECT_STORAGE_KEY, projectId);
  } catch {
    // The reload still gives IndexedDB a chance to provide the newest project.
  }
  window.location.reload();
}

function projectOptions(
  projects: VolleySpliceProject[],
  exportJobs: readonly DesignExportJob[] = [],
): DesignProjectOption[] {
  const jobsByProject = new Map(
    exportJobs.map((job) => [job.projectId, job] as const),
  );
  return projects.map((project) => ({
    id: project.id,
    name: project.source.name,
    status: project.status,
    exportJob: jobsByProject.get(project.id) ?? null,
  }));
}

function selectedProject(
  projects: VolleySpliceProject[],
  preferredProjectId?: string,
): VolleySpliceProject | null {
  let selectedId: string | null = null;
  try {
    selectedId = window.localStorage.getItem(SELECTED_PROJECT_STORAGE_KEY);
  } catch {
    // Fall back to the newest project below.
  }
  return (
    projects.find((project) => project.id === preferredProjectId) ??
    projects.find((project) => project.id === selectedId) ??
    projects.find((project) => project.status === "ready") ??
    projects[0] ??
    null
  );
}

function reviewSeed(project: VolleySpliceProject): CutDraftSeed | null {
  const analysisId = projectAnalysisId(project);
  if (!project.analysis || !analysisId) return null;
  return {
    analysisId,
    recordingId: project.id,
    duration: project.info.duration,
    analysisStart: project.analysisWindow.start,
    analysisEnd: project.analysisWindow.end,
    scoreTrackingEnabled:
      project.servingSideEnabled !== false || Boolean(project.analysis.servingSide),
    rallies: project.analysis.intervals,
    ignoredIntervals: [
      ...(project.analysisWindow.start > 0
        ? [
            {
              start: 0,
              end: project.analysisWindow.start,
              reason: "outside-game-window",
            },
          ]
        : []),
      ...(project.analysisWindow.end < project.info.duration
        ? [
            {
              start: project.analysisWindow.end,
              end: project.info.duration,
              reason: "outside-game-window",
            },
          ]
        : []),
    ],
    suppressionContractVersion: project.analysis.suppression?.policyContractVersion,
  };
}

function restoredDraft(project: VolleySpliceProject, seed: CutDraftSeed): CutDraft {
  const withGeneratedScoreMarkers = (draft: CutDraft) => {
    let scoreTracking = draft.scoreTracking;
    if (project.analysis?.servingSide) {
      const hadServeMarkers = scoreTracking.serveMarkers.length > 0;
      const generated = scoreTrackingWithServingSideOutput(
        scoreTracking,
        project.analysis.servingSide,
      );
      scoreTracking =
        !hadServeMarkers && generated.serveMarkers.length > 0
          ? { ...generated, enabled: true }
          : generated;
    }
    if (project.analysis?.sideSwitch) {
      scoreTracking = scoreTrackingWithSideSwitchOutput(
        scoreTracking,
        project.analysis.sideSwitch,
      );
    }
    return alignRallyServeMarkers({ ...draft, scoreTracking });
  };
  const persistedDrafts: CutDraft[] = [];
  try {
    for (const key of cutDraftStorageKeys(seed.analysisId)) {
      const raw = window.localStorage.getItem(key);
      const parsed = raw ? parseCutDraft(raw, seed) : null;
      if (parsed) persistedDrafts.push(parsed);
    }
  } catch {
    // Imported or inferred state below remains usable when storage is restricted.
  }
  if (project.reviewDraft) {
    const indexedDbDraft = parseCutDraft(
      JSON.stringify(project.reviewDraft),
      seed,
    );
    if (indexedDbDraft) persistedDrafts.push(indexedDbDraft);
  }
  const latestPersistedDraft = persistedDrafts.sort(
    (left, right) =>
      (Date.parse(right.updatedAt) || 0) - (Date.parse(left.updatedAt) || 0),
  )[0];
  if (latestPersistedDraft) {
    return withGeneratedScoreMarkers(latestPersistedDraft);
  }
  if (project.importedFeedback?.initialDraft) {
    const imported = parseCutDraft(
      JSON.stringify(project.importedFeedback.initialDraft),
      seed,
    );
    if (imported) return withGeneratedScoreMarkers(imported);
  }
  return withGeneratedScoreMarkers(createCutDraft(seed));
}

function cleanupSuggestions(
  project: VolleySpliceProject,
  draft: CutDraft,
): DesignCleanupSuggestion[] {
  return (project.analysis?.suppression?.suggestions ?? []).map((suggestion) => {
    const override = draft.suppressionDecisionOverrides[suggestion.id];
    const enabledByPolicy =
      draft.selectedSuppressionPolicy !== "none" &&
      suggestion.eligiblePolicyIds.includes(draft.selectedSuppressionPolicy);
    const cut = draft.cuts.find(
      (candidate) =>
        candidate.coreStart < suggestion.end && candidate.coreEnd > suggestion.start,
    );
    return {
      id: suggestion.id,
      start: suggestion.start,
      end: suggestion.end,
      score: suggestion.score,
      cutId: cut?.id ?? null,
      decision:
        override === "keep"
          ? "keep"
          : override === "suppress"
            ? "suppress"
            : enabledByPolicy
              ? "pending"
              : "keep",
    };
  });
}

type DesignReviewOptions = {
  projectId?: string;
  projects?: readonly VolleySpliceProject[];
  workActivity?: DesignWorkActivity[];
  exportJobs?: DesignExportJob[];
  sourceFile?: File | null;
  videoUrl?: string | null;
  productAnalysis?: ProductAnalysis | null;
  onSelectProject?: (projectId: string | null) => void;
  onAttachSource?: (file: File) => void | Promise<void>;
  onQueueVideoExport?: (request: DesignVideoExportRequest) => void;
};

export function useDesignReview(
  options: DesignReviewOptions = {},
): DesignReviewState {
  const [review, setReview] = useState<DesignReviewState>({ state: "loading" });
  const videoUrlRef = useRef<string | null>(null);

  useEffect(() => {
    let canceled = false;
    const loadProjects = options.projects
      ? Promise.resolve([...options.projects])
      : listProjects();
    const navigateProject = options.onSelectProject ?? reloadSelectedProject;
    void loadProjects
      .then((projects) => {
        if (canceled) return;
        const projectChoices = projectOptions(projects, options.exportJobs);
        const project = selectedProject(projects, options.projectId);
        if (!project) {
          setReview({
            state: "empty",
            message: "No VolleySplice project is saved in this browser yet.",
            projects: projectChoices,
            selectProject: navigateProject,
          });
          return;
        }
        const seed = reviewSeed(project);
        if (project.status !== "ready" || !project.analysis || !seed) {
          setReview({
            state: "unavailable",
            message:
              project.status === "error"
                ? project.error ?? "The selected project could not be analyzed."
                : `“${project.source.name}” is ${project.status}. Open the live app to finish its analysis.`,
            projects: projectChoices,
            selectProject: navigateProject,
          });
          return;
        }
        const draft = restoredDraft(project, seed);
        const attachSource = async (file: File) => {
          if (!(await sourceCanReconnectFile(project.source, file))) {
            return {
              ok: false,
              message: `That file does not match ${project.source.name}. Choose the original video used by this project.`,
              videoUrl: null,
            };
          }
          await options.onAttachSource?.(file);
          const videoUrl = URL.createObjectURL(file);
          if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
          videoUrlRef.current = videoUrl;
          setReview((current) =>
            current.state === "ready" && current.projectId === project.id
              ? {
                  ...current,
                  sourceNeedsReconnect: false,
                  sourceFile: file,
                  videoUrl,
                }
              : current,
          );
          return {
            ok: true,
            message: `${file.name} is linked for playback and export.`,
            videoUrl,
          };
        };
        const saveDraft = (next: CutDraft) => {
          const persistedDraft = {
            ...next,
            updatedAt: new Date().toISOString(),
          };
          try {
            window.localStorage.setItem(
              cutDraftStorageKey(seed.analysisId),
              JSON.stringify(persistedDraft),
            );
          } catch {
            // The current editor remains usable in memory when browser storage is restricted.
          }
          void putProjectReviewDraft(project.id, persistedDraft).catch(() => {
            // localStorage remains the synchronous fallback when IndexedDB is unavailable.
          });
        };
        setReview({
          state: "ready",
          projectId: project.id,
          projectName: project.source.name.replace(/\.[^.]+$/, ""),
          sourceName: project.source.name,
          sourceSize: project.source.size,
          sourceNeedsReconnect: !options.sourceFile || !options.videoUrl,
          sourceFile: options.sourceFile ?? null,
          videoUrl: options.videoUrl ?? null,
          productAnalysis: options.productAnalysis ?? null,
          duration: project.info.duration,
          width: project.info.width,
          height: project.info.height,
          gameStart: project.analysisWindow.start,
          gameEnd: project.analysisWindow.end,
          cropCourt:
            project.roi.x > 0 ||
            project.roi.y > 0 ||
            project.roi.width < 0.999_999 ||
            project.roi.height < 0.999_999,
          sideSwitchEnabled: project.sideSwitchEnabled !== false,
          projectStatus: project.status,
          draft,
          draftSeed: seed,
          suppression: project.analysis.suppression,
          cleanupSuggestions: cleanupSuggestions(project, draft),
          projects: projectChoices,
          workActivity: options.workActivity ?? [],
          exportJob:
            options.exportJobs?.find((job) => job.projectId === project.id) ??
            null,
          selectProject: navigateProject,
          attachSource,
          queueVideoExport: (request) => options.onQueueVideoExport?.(request),
          saveDraft,
        });
      })
      .catch((cause: unknown) => {
        if (canceled) return;
        setReview({
          state: "error",
          message:
            cause instanceof Error
              ? `VolleySplice could not open the saved review: ${cause.message}`
              : "VolleySplice could not open the saved review.",
          projects: [],
          selectProject: navigateProject,
        });
      });
    return () => {
      canceled = true;
      if (videoUrlRef.current) {
        URL.revokeObjectURL(videoUrlRef.current);
        videoUrlRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!options.projects && !options.workActivity && !options.exportJobs) return;
    const navigateProject = options.onSelectProject ?? reloadSelectedProject;
    setReview((current) =>
      current.state === "loading"
        ? current
        : {
            ...current,
            projects: options.projects
              ? projectOptions([...options.projects], options.exportJobs)
              : current.projects,
            ...(current.state === "ready"
              ? {
                  workActivity: options.workActivity ?? [],
                  exportJob:
                    options.exportJobs?.find(
                      (job) => job.projectId === current.projectId,
                    ) ?? null,
                  queueVideoExport: (request: DesignVideoExportRequest) =>
                    options.onQueueVideoExport?.(request),
                }
              : {}),
            selectProject: navigateProject,
          },
    );
  }, [
    options.exportJobs,
    options.onQueueVideoExport,
    options.onSelectProject,
    options.projects,
    options.workActivity,
  ]);

  useEffect(() => {
    if (
      options.videoUrl &&
      videoUrlRef.current &&
      options.videoUrl !== videoUrlRef.current
    ) {
      URL.revokeObjectURL(videoUrlRef.current);
      videoUrlRef.current = null;
    }
    setReview((current) =>
      current.state === "ready" &&
      (!options.projectId || current.projectId === options.projectId)
        ? {
            ...current,
            sourceNeedsReconnect: !options.sourceFile || !options.videoUrl,
            sourceFile: options.sourceFile ?? null,
            videoUrl: options.videoUrl ?? null,
            productAnalysis: options.productAnalysis ?? current.productAnalysis,
          }
        : current,
    );
  }, [
    options.productAnalysis,
    options.projectId,
    options.sourceFile,
    options.videoUrl,
  ]);

  return review;
}
