import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  buildFinalCutIntervals,
  createCutDraft,
  materializeFinalCutIntervals,
  parseCutDraft,
  suppressionSuggestionState,
} from "../../prod/src/lib/cut-draft.ts";
import {
  buildSuppressionSuggestions,
  nextSuppressionAfterTime,
  nextSuppressionSuggestion,
  suggestionsForPolicy,
  type ActiveSuppressionPolicyId,
  type SuppressionSuggestion,
} from "../../prod/src/lib/on-device/suppression-policy.ts";
import {
  HELD_SUPPRESSION_DECODER,
  loadSuppressionModelBundle,
  SUPPRESSION_ARTIFACT_SHA256,
  SUPPRESSION_BROWSER_ASSET_SHA256,
  SUPPRESSION_WEIGHTS_SHA256,
} from "../../prod/src/lib/on-device/suppression-model.ts";
import { createHash } from "node:crypto";

const fixture = JSON.parse(readFileSync(
  new URL("../fixtures/suppression-policy-golden.json", import.meta.url),
  "utf8",
));

function normalizedIntervals(
  suggestions: readonly SuppressionSuggestion[],
  policy: ActiveSuppressionPolicyId,
): number[][] {
  return suggestionsForPolicy(suggestions, policy).map(({ start, end }) => [start, end]);
}

test("suppression browser asset is pinned to the corrected head and held decoder", () => {
  const asset = readFileSync(new URL(
    "../../prod/public/runtime/suppression-39eddf581639.json",
    import.meta.url,
  ));
  assert.equal(createHash("sha256").update(asset).digest("hex"), SUPPRESSION_BROWSER_ASSET_SHA256);
  const bundle = loadSuppressionModelBundle(JSON.parse(asset.toString("utf8")));
  assert.equal(bundle.artifactSha256, SUPPRESSION_ARTIFACT_SHA256);
  assert.equal(bundle.weightsSha256, SUPPRESSION_WEIGHTS_SHA256);
  assert.deepEqual(bundle.head.decoder, HELD_SUPPRESSION_DECODER);
  assert.equal(bundle.featureNames.length, 520);
});

test("policy algebra protects full connected components and keeps exact thresholds open", () => {
  const result = buildSuppressionSuggestions(
    fixture.productionComponents,
    fixture.decodedSuppression,
    fixture.duration,
  );
  assert.equal(result.identicalPolicyResults, false);
  for (const policy of ["conservative", "balanced", "aggressive"] as const) {
    assert.deepEqual(
      normalizedIntervals(result.suggestions, policy),
      fixture.expectedPolicyIntervals[policy],
      policy,
    );
  }
  const sharedLogicalIds = new Set(result.suggestions.map(({ logicalId }) => logicalId));
  assert.equal(sharedLogicalIds.size, 1, "one decoded event keeps one decision across policies");
});

function materializerDraft() {
  const draft = createCutDraft({
    analysisId: "suppression-materializer",
    recordingId: "recording",
    duration: 30,
    rallies: [
      { id: "R001", start: 5, end: 15, confidence: 0.9, included: true },
    ],
    ignoredIntervals: [],
  });
  draft.beforePaddingSeconds = 1;
  draft.afterPaddingSeconds = 1;
  draft.joinGapSeconds = 0.5;
  draft.cuts[0].keepStart = 4;
  draft.cuts[0].keepEnd = 16;
  return draft;
}

const splitSuggestion: SuppressionSuggestion = {
  id: "suggestion-8-12",
  logicalId: "logical-suggestion",
  suppressionEventId: "S001-8000-12000",
  start: 8,
  end: 12,
  score: 0.9,
  sourceProductionIds: ["PP-001"],
  eligiblePolicyIds: ["conservative", "balanced", "aggressive"],
};

test("suppression navigation is source ordered, wraps, and never changes decisions", () => {
  const suggestions = [
    { ...splitSuggestion, id: "later", start: 12, end: 13 },
    { ...splitSuggestion, id: "earlier", start: 4, end: 5 },
  ];
  assert.equal(nextSuppressionSuggestion(suggestions, "", 1)?.id, "earlier");
  assert.equal(nextSuppressionSuggestion(suggestions, "", -1)?.id, "later");
  assert.equal(nextSuppressionSuggestion(suggestions, "later", 1)?.id, "earlier");
  assert.equal(nextSuppressionSuggestion(suggestions, "earlier", -1)?.id, "later");
  assert.equal(nextSuppressionAfterTime(suggestions, 3)?.id, "earlier");
  assert.equal(nextSuppressionAfterTime(suggestions, 4)?.id, "later");
  assert.equal(nextSuppressionAfterTime(suggestions, 13)?.id, "earlier");
});

test("No suppression exactly preserves the prior final interval list", () => {
  const draft = materializerDraft();
  assert.deepEqual(buildFinalCutIntervals(draft), [
    { start: 4, end: 16, cutIds: ["R001"] },
  ]);
  assert.deepEqual(
    buildFinalCutIntervals(draft, { suggestions: [splitSuggestion] }),
    [{ start: 4, end: 16, cutIds: ["R001"] }],
  );
});

test("suppression splits raw cores before padding and explicit decisions survive", () => {
  const draft = materializerDraft();
  draft.selectedSuppressionPolicy = "conservative";
  const suppression = { suggestions: [splitSuggestion] };
  assert.equal(suppressionSuggestionState(splitSuggestion, draft), "suppressed");
  assert.deepEqual(buildFinalCutIntervals(draft, suppression), [
    { start: 4, end: 9, cutIds: ["R001"] },
    { start: 11, end: 16, cutIds: ["R001"] },
  ]);

  draft.suppressionDecisionOverrides[splitSuggestion.logicalId] = "keep";
  assert.equal(suppressionSuggestionState(splitSuggestion, draft), "kept");
  assert.deepEqual(buildFinalCutIntervals(draft, suppression), [
    { start: 4, end: 16, cutIds: ["R001"] },
  ]);
});

test("touched rallies default to Keep, explicit Suppress wins, and manual time wins", () => {
  const draft = materializerDraft();
  draft.selectedSuppressionPolicy = "balanced";
  draft.userTouchedCutIds = ["R001"];
  assert.equal(suppressionSuggestionState(splitSuggestion, draft), "edited-kept");
  draft.suppressionDecisionOverrides[splitSuggestion.logicalId] = "suppress";
  assert.equal(suppressionSuggestionState(splitSuggestion, draft), "suppressed");
  draft.cuts.push({
    id: "M001",
    coreStart: 8,
    coreEnd: 12,
    keepStart: 8,
    keepEnd: 12,
    confidence: 1,
    included: true,
    origin: "manual",
  });
  assert.deepEqual(buildFinalCutIntervals(draft, { suggestions: [splitSuggestion] }), [
    { start: 4, end: 16, cutIds: ["R001", "M001"] },
  ]);
});

test("ignored intervals split output after joining and are never rejoined", () => {
  const draft = materializerDraft();
  draft.joinGapSeconds = 3;
  draft.ignoredIntervals = [{ id: "I001", start: 9, end: 11, reason: "camera-gap" }];
  const result = materializeFinalCutIntervals(draft);
  assert.deepEqual(result.intervals, [
    { start: 4, end: 9, cutIds: ["R001"] },
    { start: 11, end: 16, cutIds: ["R001"] },
  ]);
});

test("version eight drafts migrate suppression state and only material edits become touched", () => {
  const seed = {
    analysisId: "legacy-suppression",
    recordingId: "recording",
    duration: 30,
    rallies: [
      { id: "R001", start: 5, end: 10, confidence: 0.9, included: true },
      { id: "R002", start: 15, end: 20, confidence: 0.9, included: true },
    ],
    ignoredIntervals: [],
  };
  const legacy = createCutDraft(seed) as unknown as Record<string, unknown>;
  legacy.version = 8;
  delete legacy.analysisStart;
  delete legacy.analysisEnd;
  delete legacy.selectedSuppressionPolicy;
  delete legacy.suppressionDecisionOverrides;
  delete legacy.userTouchedCutIds;
  delete legacy.suppressionContractVersion;
  (legacy.cuts as Array<Record<string, unknown>>)[1].included = false;
  const restored = parseCutDraft(JSON.stringify(legacy), seed);
  assert.equal(restored?.selectedSuppressionPolicy, "none");
  assert.deepEqual(restored?.suppressionDecisionOverrides, {});
  assert.deepEqual(restored?.userTouchedCutIds, ["R002"]);
});
