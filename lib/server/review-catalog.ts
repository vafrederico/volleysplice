import { readFile } from "node:fs/promises";
import path from "node:path";

import { applyIgnoredIntervalRevision, loadAnalyses } from "@/lib/analysis";
import type {
  AnalysisOption,
  DatasetRole,
  ReviewAnalysis,
  ReviewCatalog,
  ReviewVideoOption,
  TrainingCorpus,
} from "@/lib/analysis-types";
import { parseLabelDocument, type LabelDocument, type RallyLabel } from "@/lib/annotations";
import type { Rally } from "@/lib/edit-list";
import {
  getPreparedLabelingCatalog,
  getSavedLabelingDocument,
  type PreparedLabelingTask,
} from "@/lib/server/labeling-tasks";

const DEFAULT_LABELING_WORKSPACE = "/mnt/freenas/volleycut/labeling-v1-2026-08-09";
const DEFAULT_NO_BEACH_LABELING_WORKSPACE =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12";
const DEFAULT_NO_BEACH_V0_WORKSPACE =
  "/mnt/freenas/volleycut/v0-2026-08-09-no-beach-2026-08-12";

type ModelTrainingMetadata = {
  trainingRecordingIds: string[];
  validationRecordingIds: string[];
  trainingSourceGroups: string[];
  validationSourceGroups: string[];
};

type BeachComparisonReport = {
  inferenceCoverage?: Array<{
    model?: unknown;
    rows?: Array<{ analysisPath?: unknown }>;
  }>;
};

function labelingWorkspace(): string {
  return path.resolve(
    /* turbopackIgnore: true */
    process.env.VOLLEYCUT_LABELING_WORKSPACE ?? DEFAULT_LABELING_WORKSPACE,
  );
}

function noBeachLabelingWorkspace(): string {
  return path.resolve(
    /* turbopackIgnore: true */
    process.env.VOLLEYCUT_NO_BEACH_LABELING_WORKSPACE ??
      DEFAULT_NO_BEACH_LABELING_WORKSPACE,
  );
}

function noBeachV0Workspace(): string {
  return path.resolve(
    /* turbopackIgnore: true */
    process.env.VOLLEYCUT_NO_BEACH_V0_WORKSPACE ?? DEFAULT_NO_BEACH_V0_WORKSPACE,
  );
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

async function noBeachModelVersions(): Promise<Map<string, string>> {
  try {
    const report = JSON.parse(
      await readFile(
        path.join(
          noBeachLabelingWorkspace(),
          "reports",
          "beach-exclusion-retraining-comparison.json",
        ),
        "utf8",
      ),
    ) as BeachComparisonReport;
    const versions = new Map<string, string>();
    for (const coverage of report.inferenceCoverage ?? []) {
      if (typeof coverage.model !== "string") continue;
      for (const row of coverage.rows ?? []) {
        if (typeof row.analysisPath !== "string") continue;
        const analysisId = path.basename(path.dirname(row.analysisPath));
        if (/^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$/.test(analysisId)) {
          versions.set(analysisId, coverage.model);
        }
      }
    }
    return versions;
  } catch {
    return new Map();
  }
}

function bindNoBeachModelVersion(
  analysis: ReviewAnalysis,
  versions: Map<string, string>,
): ReviewAnalysis {
  const exactVersion = versions.get(analysis.id);
  if (!exactVersion || exactVersion === analysis.modelVersion) return analysis;
  return {
    ...analysis,
    modelVersion: exactVersion,
    variantLabel: `Trained model · ${exactVersion}`,
    variantDescription: analysis.variantDescription
      ? `${analysis.variantDescription} Exact beach-exclusion lineage: ${exactVersion}.`
      : `Beach-exclusion comparison lineage ${exactVersion}.`,
  };
}

async function readModelTrainingMetadata(
  modelVersion: string,
  trainingCorpus: TrainingCorpus,
): Promise<ModelTrainingMetadata | null> {
  if (!/^[A-Za-z0-9_-]+$/.test(modelVersion)) return null;
  const workspace =
    trainingCorpus === "without-beach"
      ? modelVersion.startsWith("real-rally-v0")
        ? noBeachV0Workspace()
        : noBeachLabelingWorkspace()
      : labelingWorkspace();
  try {
    const value = JSON.parse(
      await readFile(
        path.join(workspace, "models", modelVersion, "model.json"),
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
    addedAt: null,
    trainingCorpus: "reference",
    trainingCorpusLabel: "Reference",
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
    ignoredIntervals: document.ignoredIntervals,
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
    ignoredIntervals: analysis.ignoredIntervals
      .map((interval) => ({
        ...interval,
        start: Math.max(0, Math.min(duration, interval.start)),
        end: Math.max(0, Math.min(duration, interval.end)),
      }))
      .filter((interval) => interval.end > interval.start),
  };
}

function variantRank(analysis: ReviewAnalysis): number {
  if (analysis.kind === "gold") return 0;
  if (analysis.kind === "model") return 1;
  if (analysis.kind === "heuristic") return 2;
  if (analysis.kind === "sol") return 3;
  return 4;
}

function addedAtTime(analysis: ReviewAnalysis): number {
  const value = analysis.addedAt ? Date.parse(analysis.addedAt) : Number.NaN;
  return Number.isFinite(value) ? value : 0;
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
    addedAt: analysis.addedAt,
    trainingCorpus: analysis.trainingCorpus,
    trainingCorpusLabel: analysis.trainingCorpusLabel,
    datasetRole: analysis.datasetRole,
    datasetRoleLabel: analysis.datasetRoleLabel,
    duration: analysis.duration,
    rallyCount: analysis.rallies.length,
  };
}

export async function loadReviewCatalog(): Promise<ReviewCatalog> {
  const [prepared, original, rawWithoutBeach, noBeachVersions] = await Promise.all([
    getPreparedLabelingCatalog(),
    loadAnalyses(),
    loadAnalyses({ trainingCorpus: "without-beach" }),
    noBeachModelVersions(),
  ]);
  const withoutBeach = rawWithoutBeach.map((analysis) =>
    bindNoBeachModelVersion(analysis, noBeachVersions),
  );
  const generated = [...original, ...withoutBeach];
  const tasks = prepared.tasks.filter((task) => task.batch === "full");
  const modelVersions = new Map(
    generated
      .filter((analysis) => analysis.modelVersion !== null)
      .map((analysis) => [
        `${analysis.trainingCorpus}:${analysis.modelVersion}`,
        analysis,
      ]),
  );
  const modelMetadata = new Map(
    await Promise.all(
      [...modelVersions].map(async ([key, analysis]) => [
        key,
        await readModelTrainingMetadata(
          analysis.modelVersion as string,
          analysis.trainingCorpus,
        ),
      ] as const),
    ),
  );

  const analyses: ReviewAnalysis[] = [];
  const videos: ReviewVideoOption[] = [];
  for (const task of tasks) {
    const [completedGold, sol, savedLabels] = await Promise.all([
      readLabelAnalysis(
        task,
        path.join(labelingWorkspace(), "completed", "full-v1", `${task.id}.labels.json`),
        "gold",
      ),
      readLabelAnalysis(task, task.prelabelPath, "sol"),
      getSavedLabelingDocument(task),
    ]);
    const gold = completedGold && savedLabels.source === "draft"
      ? applyIgnoredIntervalRevision(
          completedGold,
          savedLabels.document.ignoredIntervals,
        )
      : completedGold;
    const taskAnalyses = generated
      .filter((analysis) => matchesTask(analysis, task))
      .map((analysis) => {
        if (analysis.kind !== "model" || !analysis.modelVersion) {
          return clampRunToTask(analysis, task, "not-applicable", "Not applicable");
        }
        const role = modelDatasetRole(
          task,
          modelMetadata.get(
            `${analysis.trainingCorpus}:${analysis.modelVersion}`,
          ) ?? null,
        );
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
      if (left.kind === "model" && right.kind === "model") {
        const added = addedAtTime(right) - addedAtTime(left);
        if (added !== 0) return added;
      }
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
