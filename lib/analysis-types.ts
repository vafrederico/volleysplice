import type { Rally } from "@/lib/edit-list";

export type AnalysisKind = "heuristic" | "model" | "sol" | "gold" | "unknown";

export type DatasetRole = "training" | "validation" | "evaluation" | "not-applicable";

export type TrainingCorpus = "original" | "without-beach" | "reference";

export type TrainingCorpusView = "original" | "without-beach" | "both";

export type CourtLine = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type ReviewAnalysis = {
  id: string;
  recordingId: string;
  title: string;
  variantLabel: string;
  variantDescription: string | null;
  kind: AnalysisKind;
  method: string;
  modelVersion: string | null;
  trainingCorpus: TrainingCorpus;
  trainingCorpusLabel: string;
  datasetRole: DatasetRole;
  datasetRoleLabel: string;
  duration: number;
  width: number;
  height: number;
  sourceFilename: string;
  videoUrl: string | null;
  courtPreviewUrl: string | null;
  courtConfidence: number;
  courtSource: string;
  courtLines: CourtLine[];
  cameraStability: number;
  warnings: string[];
  rallies: Rally[];
};

export type AnalysisOption = {
  id: string;
  recordingId: string;
  title: string;
  variantLabel: string;
  variantDescription: string | null;
  kind: AnalysisKind;
  modelVersion: string | null;
  trainingCorpus: TrainingCorpus;
  trainingCorpusLabel: string;
  datasetRole: DatasetRole;
  datasetRoleLabel: string;
  duration: number;
  rallyCount: number;
};

export type ReviewVideoOption = {
  id: string;
  title: string;
  environment: string;
  duration: number;
  analyses: AnalysisOption[];
};

export type ReviewCatalog = {
  analyses: ReviewAnalysis[];
  videos: ReviewVideoOption[];
};
