// Lab adapter contract copied from production. No project-store hook or writes.
import type { CutDraft, CutDraftSeed } from "../lib/cut-draft";
import type { VolleySpliceProject } from "../lib/project-store";
import type { OnDeviceSuppression } from "../lib/on-device/types";
import type { ExportInterval, ScoreOverlayOptions } from "../lib/on-device/export";
import type { ProductAnalysis } from "../lib/product-analysis";

export type DesignProjectOption = {
  id: string;
  name: string;
  status: VolleySpliceProject["status"];
  lastExportedAt?: string;
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
  logicalId: string;
  eligiblePolicyIds: readonly string[];
  start: number;
  end: number;
  score: number;
  cutId: string | null;
  decision: "pending" | "keep" | "suppress";
};

export type ReadyDesignReview = {
  /** Recorded-model UX trial; no production inference or project persistence. */
  labMode?: boolean;
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
  modelDraft: CutDraft;
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
