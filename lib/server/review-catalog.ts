import { readFile } from "node:fs/promises";
import path from "node:path";

import { loadAnalyses } from "@/lib/analysis";
import type {
  AnalysisOption,
  DatasetRole,
  ReviewAnalysis,
  ReviewCatalog,
  ReviewVideoOption,
} from "@/lib/analysis-types";
import { parseLabelDocument, type LabelDocument, type RallyLabel } from "@/lib/annotations";
import type { Rally } from "@/lib/edit-list";
import { getPreparedLabelingCatalog, type PreparedLabelingTask } from "@/lib/server/labeling-tasks";

const DEFAULT_LABELING_WORKSPACE = "/mnt/freenas/volleycut/labeling-v1-2026-08-09";

type ModelTrainingMetadata = {
  trainingRecordingIds: string[];
  validationRecordingIds: string[];
  trainingSourceGroups: string[];
  validationSourceGroups: string[];
};

function labelingWorkspace(): string {
  return path.resolve(
    /* turbopackIgnore: true */
    process.env.VOLLEYCUT_LABELING_WORKSPACE ?? DEFAULT_LABELING_WORKSPACE,
  );
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

async function readModelTrainingMetadata(
  modelVersion: string,
): Promise<ModelTrainingMetadata | null> {
  if (!/^[A-Za-z0-9_-]+$/.test(modelVersion)) return null;
  try {
    const value = JSON.parse(
      await readFile(
        path.join(labelingWorkspace(), "models", modelVersion, "model.json"),
        "utf8",
      ),
    ) as { training?: Record<string, unknown> };
    const training = value.training;
    if (!training || typeof training !== "object") return null;
    return {
      trainingRecordingIds: stringArray(training.trainingRecordingIds),
      validationRecordingIds: stringArray(training.validationRecordingIds),
      trainingSourceGroups: stringArray(training.trainingSourceGroups),
      validationSourceGroups: stringArray(training.validationSourceGroups),
    };
  } catch {
    return null;
  }
}

function modelDatasetRole(
  task: PreparedLabelingTask,
  metadata: ModelTrainingMetadata | null,
): { datasetRole: DatasetRole; datasetRoleLabel: string } {
  if (!metadata) {
    return { datasetRole: "not-applicable", datasetRoleLabel: "Lineage unavailable" };
  }
  if (metadata.trainingRecordingIds.includes(task.id)) {
    return { datasetRole: "training", datasetRoleLabel: "Training data" };
  }
  if (metadata.validationRecordingIds.includes(task.id)) {
    return { datasetRole: "validation", datasetRoleLabel: "Validation / tuning" };
  }
  if (metadata.trainingSourceGroups.includes(task.document.recording.sourceGroup)) {
    return { datasetRole: "training", datasetRoleLabel: "Training source (excerpt)" };
  }
  if (metadata.validationSourceGroups.includes(task.document.recording.sourceGroup)) {
    return { datasetRole: "validation", datasetRoleLabel: "Validation source (excerpt)" };
  }
  return { datasetRole: "evaluation", datasetRoleLabel: "Evaluation only" };
}

function confidenceFromTags(tags: string[]): number {
  const values = tags
    .filter((tag) => tag.includes("confidence:"))
    .map((tag) => tag.split(":").at(-1))
    .map((confidence) => (confidence === "high" ? 0.9 : confidence === "medium" ? 0.65 : 0.4));
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0.65;
}

function labelsToRallies(rows: RallyLabel[], verified: boolean): Rally[] {
  return rows.map((row, index) => ({
    id: `R${String(index + 1).padStart(3, "0")}`,
    start: row.start,
    end: row.end,
    confidence: verified ? 1 : confidenceFromTags(row.tags),
    included: true,
  }));
}

function labelAnalysis(
  task: PreparedLabelingTask,
  document: LabelDocument,
  kind: "gold" | "sol",
): ReviewAnalysis {
  const verified = kind === "gold";
  return {
    id: `${kind === "gold" ? "gold" : "sol-xhigh"}--${task.id}`,
    recordingId: task.id,
    title: task.originalFilename,
    variantLabel: verified ? "Human-verified labels" : "Blind Sol audiovisual prelabel",
    variantDescription: null,
    kind,
    method: verified
      ? "human-verified-serve-contact-to-dead-ball-v1"
      : document.prelabel?.analysisMethod ?? "blind-gpt-5.6-sol-xhigh-audiovisual",
    modelVersion: null,
    datasetRole: "not-applicable",
    datasetRoleLabel: verified ? "Reference labels" : "Not trained locally",
    duration: document.recording.durationSeconds,
    width: 16,
    height: 9,
    sourceFilename: task.originalFilename,
    videoUrl: `/api/labeling/tasks/${encodeURIComponent(task.id)}/video`,
    courtPreviewUrl: null,
    courtConfidence: document.recording.roi ? 1 : 0,
    courtSource: document.recording.roi ? "manual-labeling-roi" : "full-frame",
    courtLines: [],
    cameraStability: document.recording.capture.stationary === true ? 1 : 0,
    warnings: verified
      ? []
      : ["Blind Sol candidates were generated before continuous human verification."],
    rallies: labelsToRallies(document.rallies, verified),
  };
}

async function readLabelAnalysis(
  task: PreparedLabelingTask,
  filePath: string,
  kind: "gold" | "sol",
): Promise<ReviewAnalysis | null> {
  try {
    const document = parseLabelDocument(JSON.parse(await readFile(filePath, "utf8")) as unknown);
    return labelAnalysis(task, document, kind);
  } catch {
    return null;
  }
}

function matchesTask(analysis: ReviewAnalysis, task: PreparedLabelingTask): boolean {
  const baseId = task.id.replace(/-full$/, "");
  return (
    analysis.recordingId === task.id ||
    analysis.recordingId === baseId ||
    analysis.id.includes(baseId) ||
    analysis.sourceFilename.includes(baseId.split("-").at(-1) ?? baseId)
  );
}

function clampRunToTask(
  analysis: ReviewAnalysis,
  task: PreparedLabelingTask,
  datasetRole: DatasetRole,
  datasetRoleLabel: string,
): ReviewAnalysis {
  const duration = task.document.recording.durationSeconds;
  return {
    ...analysis,
    recordingId: task.id,
    title: task.originalFilename,
    duration,
    sourceFilename: task.originalFilename,
    videoUrl: `/api/labeling/tasks/${encodeURIComponent(task.id)}/video`,
    datasetRole,
    datasetRoleLabel,
    rallies: analysis.rallies
      .map((rally) => ({
        ...rally,
        start: Math.max(0, Math.min(duration, rally.start)),
        end: Math.max(0, Math.min(duration, rally.end)),
      }))
      .filter((rally) => rally.end > rally.start),
  };
}

function variantRank(analysis: ReviewAnalysis): number {
  if (analysis.kind === "gold") return 0;
  if (analysis.kind === "model") return 1;
  if (analysis.kind === "heuristic") return 2;
  if (analysis.kind === "sol") return 3;
  return 4;
}

function toOption(analysis: ReviewAnalysis): AnalysisOption {
  return {
    id: analysis.id,
    recordingId: analysis.recordingId,
    title: analysis.title,
    variantLabel: analysis.variantLabel,
    variantDescription: analysis.variantDescription,
    kind: analysis.kind,
    modelVersion: analysis.modelVersion,
    datasetRole: analysis.datasetRole,
    datasetRoleLabel: analysis.datasetRoleLabel,
    duration: analysis.duration,
    rallyCount: analysis.rallies.length,
  };
}

export async function loadReviewCatalog(): Promise<ReviewCatalog> {
  const [prepared, generated] = await Promise.all([
    getPreparedLabelingCatalog(),
    loadAnalyses(),
  ]);
  const tasks = prepared.tasks.filter((task) => task.batch === "full");
  const modelVersions = new Set(
    generated
      .map((analysis) => analysis.modelVersion)
      .filter((version): version is string => version !== null),
  );
  const modelMetadata = new Map(
    await Promise.all(
      [...modelVersions].map(async (version) => [
        version,
        await readModelTrainingMetadata(version),
      ] as const),
    ),
  );

  const analyses: ReviewAnalysis[] = [];
  const videos: ReviewVideoOption[] = [];
  for (const task of tasks) {
    const [gold, sol] = await Promise.all([
      readLabelAnalysis(
        task,
        path.join(labelingWorkspace(), "completed", "full-v1", `${task.id}.labels.json`),
        "gold",
      ),
      readLabelAnalysis(task, task.prelabelPath, "sol"),
    ]);
    const taskAnalyses = generated
      .filter((analysis) => matchesTask(analysis, task))
      .map((analysis) => {
        if (analysis.kind !== "model" || !analysis.modelVersion) {
          return clampRunToTask(analysis, task, "not-applicable", "Not applicable");
        }
        const role = modelDatasetRole(task, modelMetadata.get(analysis.modelVersion) ?? null);
        return clampRunToTask(
          analysis,
          task,
          role.datasetRole,
          role.datasetRoleLabel,
        );
      });
    if (gold) taskAnalyses.push(gold);
    if (sol) taskAnalyses.push(sol);
    taskAnalyses.sort((left, right) => {
      const rank = variantRank(left) - variantRank(right);
      if (rank !== 0) return rank;
      return right.variantLabel.localeCompare(left.variantLabel);
    });
    analyses.push(...taskAnalyses);
    videos.push({
      id: task.id,
      title: task.originalFilename,
      environment: task.document.recording.environment,
      duration: task.document.recording.durationSeconds,
      analyses: taskAnalyses.map(toOption),
    });
  }
  return { analyses, videos };
}
