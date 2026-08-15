import {
  type LabelDocument,
  parseLabelDocument,
  type RallyLabel,
  roundTime,
} from "./annotations.ts";

type JsonRecord = Record<string, unknown>;

export type ProductionLabelSeed = {
  document: LabelDocument;
  modelId: string;
  modelLabel: string;
};

function record(value: unknown): JsonRecord | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as JsonRecord)
    : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function modelRallies(value: unknown, duration: number): RallyLabel[] {
  if (!Array.isArray(value))
    throw new Error("production model rallies are missing");
  const rallies = value.flatMap((candidate, index) => {
    const row = record(candidate);
    if (!row || row.included === false) return [];
    const start = finiteNumber(row.start);
    const end = finiteNumber(row.end);
    const confidence = finiteNumber(row.confidence);
    if (
      start === null ||
      end === null ||
      confidence === null ||
      start < 0 ||
      end <= start ||
      end > duration ||
      confidence < 0 ||
      confidence > 1
    ) {
      throw new Error(`production model rally ${index + 1} is invalid`);
    }
    return [
      {
        start: roundTime(start),
        end: roundTime(end),
        tags: ["ai-prelabel", `model-confidence:${confidence.toFixed(3)}`],
      },
    ];
  });
  rallies.sort(
    (left, right) => left.start - right.start || left.end - right.end,
  );
  return rallies;
}

export function buildProductionLabelSeed(
  base: LabelDocument,
  value: unknown,
  modelId: string,
): ProductionLabelSeed {
  const root = record(value);
  const source = record(root?.source);
  const analysis = record(root?.analysis);
  const expectedAnalysisId = `${modelId}--${base.recording.id}`;
  if (
    root?.schemaVersion !== 1 ||
    root.id !== expectedAnalysisId ||
    root.recordingId !== base.recording.id ||
    source?.contentSha256 !== base.recording.contentSha256 ||
    source.filename !== base.recording.videoFilename
  ) {
    throw new Error(
      "production model analysis does not match the labeling task",
    );
  }
  const sourceDuration = finiteNumber(source.duration);
  if (
    sourceDuration === null ||
    Math.abs(sourceDuration - base.recording.durationSeconds) > 0.1
  ) {
    throw new Error(
      "production model analysis duration does not match the labeling task",
    );
  }
  if (!analysis || typeof analysis.method !== "string") {
    throw new Error("production model analysis metadata is missing");
  }
  const analyzedAt =
    typeof root.createdAt === "string" &&
    Number.isFinite(Date.parse(root.createdAt))
      ? new Date(root.createdAt).toISOString()
      : null;
  if (!analyzedAt)
    throw new Error("production model analysis timestamp is invalid");
  const modelLabel =
    typeof analysis.variantLabel === "string" && analysis.variantLabel.trim()
      ? analysis.variantLabel.trim()
      : modelId;
  const document = parseLabelDocument({
    ...base,
    annotation: {
      ...base.annotation,
      status: "in-progress",
      reviewedAt: null,
      notes:
        base.annotation.notes ||
        `Initialized from production model ${modelId}. Validate the full recording and every boundary.`,
    },
    prelabel: {
      analysisMethod: analysis.method,
      candidateFile: expectedAnalysisId,
      analyzedAt,
      ambiguities: [
        `Unvalidated starting labels from the production model ${modelId}.`,
        "Continuously review the full recording, correct both boundaries, add misses, and remove false positives.",
      ],
    },
    rallies: modelRallies(root.rallies, base.recording.durationSeconds),
  });
  return { document, modelId, modelLabel };
}
