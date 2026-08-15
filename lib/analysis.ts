import { promises as fs } from "node:fs";
import path from "node:path";
import type {
  CourtLine,
  ReviewAnalysis,
  TrainingCorpus,
} from "./analysis-types.ts";
import type { Rally } from "./edit-list.ts";
import type { IgnoredInterval } from "./annotations.ts";
import { getAnalysesRoot } from "./storage.ts";
import {
  PRODUCTION_MODEL_DESCRIPTION,
  PRODUCTION_MODEL_ID,
  PRODUCTION_MODEL_LABEL,
} from "./production-model.ts";

const ANALYSIS_ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$/;

export function applyIgnoredIntervalRevision(
  analysis: ReviewAnalysis,
  ignoredIntervals: IgnoredInterval[],
): ReviewAnalysis {
  return { ...analysis, ignoredIntervals };
}

const MODEL_VARIANT_DESCRIPTIONS: Readonly<Record<string, string>> = {
  "pilot-baseline-v0":
    "Pilot iteration v0: the original visual live/dead baseline, trained on 90-second excerpts without within-recording percentile normalization.",
  "pilot-percentile-v1":
    "Pilot iteration v1: adds within-recording percentile normalization to the visual live/dead baseline trained on 90-second excerpts.",
  "full-percentile-v1":
    "Full-corpus iteration v1: the visual live/dead baseline with within-recording percentile normalization, developed from the full-video corpus.",
  "full-audiovisual-v2-final":
    "Audiovisual iteration v2: adds camera-compensated motion, quality and formation proxies, plus legacy audio features to the full-corpus rally model.",
  "full-audiovisual-audio-normalized-v3":
    "Audiovisual iteration v3: adds causal noise-normalized frequency-band audio features to the v2 rally feature set.",
  "dual-serve-v4-v5-fusion-v1":
    "Fusion iteration v1: starts with the v4 audiovisual rally/serve pair and selectively refines boundaries with the v5 noise-normalized-audio pair; unmatched v5 rallies are not added.",
};

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function isoDate(value: unknown): string | null {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) return null;
  return new Date(value).toISOString();
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function analysisIdentity(
  root: Record<string, unknown>,
  analysis: Record<string, unknown>,
  sourceFilename: string,
) {
  const method = typeof analysis.method === "string" ? analysis.method : "unknown";
  const id = root.id as string;
  const modelId = /^model-(.+)--([A-Za-z0-9_-]+)$/.exec(id);
  const explicitVersion =
    typeof analysis.modelVersion === "string" && analysis.modelVersion.trim()
      ? analysis.modelVersion.trim()
      : null;
  const modelVersion = explicitVersion ?? modelId?.[1] ?? null;
  const kind = method.includes("heuristic")
    ? "heuristic"
    : method.includes("logistic") || modelVersion
      ? "model"
      : "unknown";
  const sourceStem = sourceFilename.replace(/\.[^.]+$/, "");
  const recordingId =
    typeof root.recordingId === "string" && ANALYSIS_ID.test(root.recordingId)
      ? root.recordingId
      : sourceStem.endsWith("-full") && ANALYSIS_ID.test(sourceStem)
        ? sourceStem
        : modelId?.[2] ?? id.replace(/-v\d+$/, "");
  const version = /-v(\d+)$/.exec(id)?.[1];
  const explicitVariantLabel =
    typeof analysis.variantLabel === "string" && analysis.variantLabel.trim()
      ? analysis.variantLabel.trim()
      : null;
  const variantLabel =
    id.startsWith(`${PRODUCTION_MODEL_ID}--`)
      ? PRODUCTION_MODEL_LABEL
      : explicitVariantLabel ??
    (kind === "model"
      ? `Trained model · ${modelVersion ?? "unknown version"}`
      : kind === "heuristic"
        ? `No-model heuristic · ${version ? `v${version}` : "v1"}`
        : method);
  const explicitVariantDescription =
    typeof analysis.variantDescription === "string" && analysis.variantDescription.trim()
      ? analysis.variantDescription.trim()
      : null;
  const variantDescription = id.startsWith(`${PRODUCTION_MODEL_ID}--`)
    ? PRODUCTION_MODEL_DESCRIPTION
    : explicitVariantDescription ?? (kind === "model"
      ? modelVersion && MODEL_VARIANT_DESCRIPTIONS[modelVersion]
        ? MODEL_VARIANT_DESCRIPTIONS[modelVersion]
        : `Model iteration ${modelVersion ?? "unknown"}. This analysis artifact does not include a detailed iteration description.`
      : null);
  return {
    kind,
    method,
    modelVersion,
    recordingId,
    variantLabel,
    variantDescription,
  } as const;
}

type AnalysisLoadOptions = {
  analysesRoot?: string;
  trainingCorpus?: Extract<TrainingCorpus, "original" | "without-beach">;
  assetSource?: "default" | "intake";
};

function trainingCorpusLabel(corpus: TrainingCorpus): string {
  if (corpus === "without-beach") return "Without beach training";
  if (corpus === "mixed") return "Mixed production training";
  if (corpus === "reference") return "Reference";
  return "Original training";
}

export function parseAnalysis(
  value: unknown,
  options: Pick<AnalysisLoadOptions, "trainingCorpus" | "assetSource"> = {},
): ReviewAnalysis | null {
  const root = record(value);
  if (!root || root.schemaVersion !== 1 || typeof root.id !== "string" || !ANALYSIS_ID.test(root.id)) return null;
  const proxy = record(root.proxy);
  const source = record(root.source);
  const analysis = record(root.analysis);
  const court = record(analysis?.court);
  const media = proxy ?? source;
  const duration = finiteNumber(media?.duration);
  if (!media || !duration || duration <= 0 || !source || !analysis || !court) return null;
  const sourceFilename = typeof source.filename === "string" ? source.filename : "Local video";
  const identity = analysisIdentity(root, analysis, sourceFilename);
  const trainingCorpus = options.trainingCorpus ?? "original";
  const mediaQuery = new URLSearchParams();
  if (trainingCorpus === "without-beach") mediaQuery.set("corpus", "without-beach");
  if (options.assetSource === "intake") mediaQuery.set("source", "intake");
  const mediaAssetQuery = mediaQuery.size > 0 ? `?${mediaQuery.toString()}` : "";
  const courtSource = typeof court.source === "string" ? court.source : "unknown";
  const courtConfidence =
    finiteNumber(court.confidence) ?? (courtSource === "manual-roi" ? 1 : 0);

  const seenIds = new Set<string>();
  const rallies: Rally[] = [];
  if (Array.isArray(root.rallies)) {
    for (const candidate of root.rallies) {
      const item = record(candidate);
      if (!item || typeof item.id !== "string" || seenIds.has(item.id)) continue;
      const rawStart = finiteNumber(item.start);
      const rawEnd = finiteNumber(item.end);
      const rawConfidence = finiteNumber(item.confidence);
      if (rawStart === null || rawEnd === null || rawConfidence === null) continue;
      const start = Math.max(0, Math.min(duration, rawStart));
      const end = Math.max(0, Math.min(duration, rawEnd));
      if (end <= start) continue;
      seenIds.add(item.id);
      rallies.push({
        id: item.id,
        start,
        end,
        confidence: Math.max(0, Math.min(1, rawConfidence)),
        included: item.included !== false,
      });
    }
  }
  rallies.sort((a, b) => a.start - b.start || a.end - b.end);

  const lines: CourtLine[] = [];
  if (Array.isArray(court.lines)) {
    for (const candidate of court.lines) {
      const line = record(candidate);
      const coordinates = [line?.x1, line?.y1, line?.x2, line?.y2].map(finiteNumber);
      if (coordinates.some((coordinate) => coordinate === null)) continue;
      const [x1, y1, x2, y2] = coordinates as number[];
      lines.push({
        x1: Math.max(0, Math.min(1, x1)),
        y1: Math.max(0, Math.min(1, y1)),
        x2: Math.max(0, Math.min(1, x2)),
        y2: Math.max(0, Math.min(1, y2)),
      });
    }
  }

  return {
    id: root.id,
    recordingId: identity.recordingId,
    title: typeof root.title === "string" && root.title.trim() ? root.title : root.id,
    variantLabel: identity.variantLabel,
    variantDescription: identity.variantDescription,
    kind: identity.kind,
    method: identity.method,
    modelVersion: identity.modelVersion,
    addedAt: isoDate(root.createdAt),
    trainingCorpus,
    trainingCorpusLabel: trainingCorpusLabel(trainingCorpus),
    datasetRole: "not-applicable",
    datasetRoleLabel: "Not applicable",
    duration,
    width: Math.max(1, finiteNumber(media.width) ?? 16),
    height: Math.max(1, finiteNumber(media.height) ?? 9),
    sourceFilename,
    videoUrl: `/api/media/${root.id}/proxy.mp4${mediaAssetQuery}`,
    courtPreviewUrl: `/api/media/${root.id}/court-preview.jpg${mediaAssetQuery}`,
    courtConfidence: Math.max(0, Math.min(1, courtConfidence)),
    courtSource,
    courtLines: lines,
    cameraStability: Math.max(0, Math.min(1, finiteNumber(analysis.cameraStability) ?? 0)),
    warnings: Array.isArray(analysis.warnings)
      ? analysis.warnings.filter((warning): warning is string => typeof warning === "string")
      : [],
    rallies,
    ignoredIntervals: [],
  };
}

export async function loadAnalyses(
  options: AnalysisLoadOptions = {},
): Promise<ReviewAnalysis[]> {
  const trainingCorpus = options.trainingCorpus ?? "original";
  const analysesRoot = options.analysesRoot ?? getAnalysesRoot(trainingCorpus);
  let entries;
  try {
    entries = await fs.readdir(analysesRoot, { withFileTypes: true });
  } catch {
    return [];
  }
  const candidates = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory() && ANALYSIS_ID.test(entry.name))
      .map(async (entry) => {
        const filename = path.join(analysesRoot, entry.name, "analysis.json");
        try {
          const stats = await fs.stat(filename);
          return { filename, modified: stats.mtimeMs };
        } catch {
          return null;
        }
      }),
  );
  candidates.sort((a, b) => (b?.modified ?? 0) - (a?.modified ?? 0));

  const analyses: ReviewAnalysis[] = [];
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      const parsed = parseAnalysis(
        JSON.parse(await fs.readFile(candidate.filename, "utf-8")),
        { trainingCorpus, assetSource: options.assetSource },
      );
      if (parsed) {
        analyses.push({
          ...parsed,
          addedAt: parsed.addedAt ?? new Date(candidate.modified).toISOString(),
        });
      }
    } catch {
      // Ignore incomplete or malformed local runs.
    }
  }
  return analyses;
}
