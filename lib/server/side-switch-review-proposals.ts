import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import {
  type SideSwitchModelProposal,
  type SideSwitchProposalBundle,
  type SideSwitchProposalLayer,
  type SideSwitchProposalModel,
} from "../side-switch-review-proposals.ts";

const DEFAULT_LABELING_WORKSPACE =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09";
const DEFAULT_REPORT_DIRECTORY = path.join(
  DEFAULT_LABELING_WORKSPACE,
  "reports",
  "side-switch",
);

export type ProposalArtifactSource = {
  modelId: SideSwitchProposalModel;
  label: string;
  detail: string;
  evaluationPath: string;
  featurePath: string;
};

type LoadedProposalLayer = {
  layer: SideSwitchProposalLayer;
  byEventId: Record<string, SideSwitchModelProposal>;
};

type JsonArtifact = {
  value: Record<string, unknown>;
  sha256: string;
};

export class SideSwitchProposalArtifactError extends Error {}

function configuredPath(value: string | undefined, fallback: string): string {
  return path.resolve(value?.trim() || fallback);
}

export function getSideSwitchProposalArtifactSources(): ProposalArtifactSource[] {
  return [
    {
      modelId: "v5",
      label: "V5 · motion players",
      detail:
        "Player-isolated motion proposals with the selected cadence decoder",
      evaluationPath: configuredPath(
        process.env.VOLLEYCUT_SIDE_SWITCH_V5_EVALUATION,
        path.join(
          DEFAULT_REPORT_DIRECTORY,
          "side-switch-specialist-v5-player-orientation-evaluation.json",
        ),
      ),
      featurePath: configuredPath(
        process.env.VOLLEYCUT_SIDE_SWITCH_V5_FEATURES,
        path.join(
          DEFAULT_REPORT_DIRECTORY,
          "side-switch-v5-player-orientation-features.json",
        ),
      ),
    },
    {
      modelId: "v6",
      label: "V6 · detected players",
      detail:
        "Quantized person detections and adaptive team features with the selected decoder",
      evaluationPath: configuredPath(
        process.env.VOLLEYCUT_SIDE_SWITCH_V6_EVALUATION,
        path.join(
          DEFAULT_REPORT_DIRECTORY,
          "side-switch-specialist-v6-detected-adaptive-evaluation.json",
        ),
      ),
      featurePath: configuredPath(
        process.env.VOLLEYCUT_SIDE_SWITCH_V6_FEATURES,
        path.join(
          DEFAULT_REPORT_DIRECTORY,
          "side-switch-v6-detected-adaptive-features.json",
        ),
      ),
    },
  ];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new SideSwitchProposalArtifactError(
      `${field} must be a finite number`,
    );
  }
  return value;
}

function nonNegativeInteger(value: unknown, field: string): number {
  const result = finiteNumber(value, field);
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new SideSwitchProposalArtifactError(
      `${field} must be a non-negative integer`,
    );
  }
  return result;
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new SideSwitchProposalArtifactError(`${field} must be a string`);
  }
  return value;
}

async function readJsonObject(
  artifactPath: string,
  description: string,
): Promise<JsonArtifact> {
  let value: unknown;
  let content: string;
  try {
    content = await readFile(artifactPath, "utf8");
    value = JSON.parse(content) as unknown;
  } catch (error) {
    throw new SideSwitchProposalArtifactError(
      `${description} could not be read: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (!isRecord(value)) {
    throw new SideSwitchProposalArtifactError(
      `${description} must contain a JSON object`,
    );
  }
  return {
    value,
    sha256: createHash("sha256").update(content).digest("hex"),
  };
}

async function loadProposalLayer(
  source: ProposalArtifactSource,
  reportEventIds: ReadonlySet<string>,
): Promise<LoadedProposalLayer> {
  const [evaluationArtifact, featureArtifact] = await Promise.all([
    readJsonObject(source.evaluationPath, `${source.modelId} evaluation`),
    readJsonObject(source.featurePath, `${source.modelId} features`),
  ]);
  const evaluation = evaluationArtifact.value;
  const features = featureArtifact.value;
  const expectedVersion = Number(source.modelId.slice(1));
  const expectedEvaluationKind = `volleycut-side-switch-specialist-evaluation-${source.modelId}`;
  const expectedFeatureKind = `volleycut-side-switch-features-${source.modelId}`;
  if (
    evaluation.schemaVersion !== expectedVersion ||
    features.schemaVersion !== expectedVersion ||
    evaluation.kind !== expectedEvaluationKind ||
    features.kind !== expectedFeatureKind ||
    !Array.isArray(evaluation.predictions) ||
    !Array.isArray(features.rows) ||
    !isRecord(evaluation.selectedDecoder)
  ) {
    throw new SideSwitchProposalArtifactError(
      `${source.modelId} artifacts have an incompatible schema`,
    );
  }
  const featureSource =
    isRecord(evaluation.sources) && isRecord(evaluation.sources.features)
      ? evaluation.sources.features
      : null;
  if (!featureSource || featureSource.sha256 !== featureArtifact.sha256) {
    throw new SideSwitchProposalArtifactError(
      `${source.modelId} evaluation does not bind the supplied feature SHA-256`,
    );
  }

  const sourceIdsByEvaluationEvent = new Map<string, string[]>();
  for (const rawRow of features.rows) {
    if (!isRecord(rawRow)) {
      throw new SideSwitchProposalArtifactError(
        `${source.modelId} features contain an invalid row`,
      );
    }
    const eventId = requiredString(rawRow.eventId, "feature row eventId");
    if (
      !Array.isArray(rawRow.sourceEventIds) ||
      rawRow.sourceEventIds.length === 0
    ) {
      throw new SideSwitchProposalArtifactError(
        `${source.modelId} feature row ${eventId} has no source event IDs`,
      );
    }
    const sourceEventIds = rawRow.sourceEventIds.map((value) =>
      requiredString(value, `source event ID for ${eventId}`),
    );
    if (sourceIdsByEvaluationEvent.has(eventId)) {
      throw new SideSwitchProposalArtifactError(
        `${source.modelId} features repeat event ${eventId}`,
      );
    }
    sourceIdsByEvaluationEvent.set(eventId, sourceEventIds);
  }

  const byEventId: Record<string, SideSwitchModelProposal> = {};
  const attachedEvaluationEvents = new Set<string>();
  const selectedAttachedEvaluationEvents = new Set<string>();
  let selectedEvents = 0;
  for (const rawPrediction of evaluation.predictions) {
    if (!isRecord(rawPrediction)) {
      throw new SideSwitchProposalArtifactError(
        `${source.modelId} evaluation contains an invalid prediction`,
      );
    }
    const evaluationEventId = requiredString(
      rawPrediction.eventId,
      "prediction eventId",
    );
    if (typeof rawPrediction.selectedPrediction !== "boolean") {
      throw new SideSwitchProposalArtifactError(
        `${source.modelId} prediction ${evaluationEventId} has no selectedPrediction flag`,
      );
    }
    const proposal: SideSwitchModelProposal = {
      modelId: source.modelId,
      evaluationEventId,
      gapOrder: nonNegativeInteger(
        rawPrediction.gapOrder,
        `gapOrder for ${evaluationEventId}`,
      ),
      score: finiteNumber(
        rawPrediction.score,
        `score for ${evaluationEventId}`,
      ),
      selected: rawPrediction.selectedPrediction,
    };
    if (proposal.selected) selectedEvents += 1;
    const sourceEventIds = sourceIdsByEvaluationEvent.get(evaluationEventId);
    if (!sourceEventIds) {
      throw new SideSwitchProposalArtifactError(
        `${source.modelId} prediction ${evaluationEventId} has no feature row`,
      );
    }
    for (const reportEventId of sourceEventIds) {
      if (!reportEventIds.has(reportEventId)) continue;
      if (byEventId[reportEventId]) {
        throw new SideSwitchProposalArtifactError(
          `${source.modelId} maps multiple predictions to ${reportEventId}`,
        );
      }
      byEventId[reportEventId] = proposal;
      attachedEvaluationEvents.add(evaluationEventId);
      if (proposal.selected) {
        selectedAttachedEvaluationEvents.add(evaluationEventId);
      }
    }
  }

  const decoder = evaluation.selectedDecoder;
  return {
    layer: {
      modelId: source.modelId,
      label: source.label,
      detail: source.detail,
      evaluationKind: requiredString(evaluation.kind, "evaluation kind"),
      evaluationCreatedAt: requiredString(
        evaluation.createdAt,
        "evaluation createdAt",
      ),
      evaluationFilename: path.basename(source.evaluationPath),
      evaluationSha256: evaluationArtifact.sha256,
      featureFilename: path.basename(source.featurePath),
      featureSha256: featureArtifact.sha256,
      threshold: finiteNumber(decoder.threshold, "selected decoder threshold"),
      candidateMargin: nonNegativeInteger(
        decoder.candidateMargin,
        "selected decoder candidate margin",
      ),
      evaluatedEvents: evaluation.predictions.length,
      selectedEvents,
      attachedEvents: attachedEvaluationEvents.size,
      selectedAttachedEvents: selectedAttachedEvaluationEvents.size,
    },
    byEventId,
  };
}

export async function loadSideSwitchReviewProposalBundle(
  reportEventIds: ReadonlySet<string>,
  sources = getSideSwitchProposalArtifactSources(),
): Promise<SideSwitchProposalBundle> {
  const settled = await Promise.allSettled(
    sources.map((source) => loadProposalLayer(source, reportEventIds)),
  );
  const bundle: SideSwitchProposalBundle = {
    layers: [],
    byEventId: {},
    errors: [],
  };
  settled.forEach((result, index) => {
    const source = sources[index];
    if (!source) return;
    if (result.status === "rejected") {
      bundle.errors.push({
        modelId: source.modelId,
        message:
          result.reason instanceof Error
            ? result.reason.message
            : String(result.reason),
      });
      return;
    }
    bundle.layers.push(result.value.layer);
    for (const [eventId, proposal] of Object.entries(result.value.byEventId)) {
      bundle.byEventId[eventId] = {
        ...bundle.byEventId[eventId],
        [source.modelId]: proposal,
      };
    }
  });
  return bundle;
}
