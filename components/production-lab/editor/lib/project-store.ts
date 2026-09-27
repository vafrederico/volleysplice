// Type-only production shapes. The lab never opens the production project database.
import type { AnalysisWindow } from "./on-device/analysis-window";
import type { NormalizedRoi, OnDeviceAnalysis, OnDeviceMediaInfo } from "./on-device/types";
import type { CutDraft } from "./cut-draft";

export type ProjectSource = {
  name: string;
  size: number;
  lastModified: number;
  type: string;
  fingerprint?: string;
};

export type ProjectStatus =
  | "queued"
  | "analyzing"
  | "waiting"
  | "ready"
  | "error";

export type VolleySpliceProject = {
  schemaVersion: 1;
  id: string;
  source: ProjectSource;
  info: OnDeviceMediaInfo;
  analysisWindow: AnalysisWindow;
  roi: NormalizedRoi;
  /** Legacy score-tracking preference retained for stored-project compatibility. */
  servingSideEnabled?: boolean;
  /** Whether inference should generate team side-switch markers. */
  sideSwitchEnabled?: boolean;
  status: ProjectStatus;
  analysis: OnDeviceAnalysis | null;
  error: string | null;
  importedFeedback?: {
    schemaVersion: 1 | 2 | 3;
    generatedAt: string;
    importedAt: string;
    originalProjectId: string;
    originalAnalysisId: string;
    runtimeVariant: string;
    warnings: string[];
    initialDraft: CutDraft;
  };
  /** Latest editor state, mirrored from localStorage for durable project restore. */
  reviewDraft?: CutDraft;
  /** Most recent successful video export; retained across edits and reloads. */
  lastExportedAt?: string;
  createdAt: string;
  updatedAt: string;
};
