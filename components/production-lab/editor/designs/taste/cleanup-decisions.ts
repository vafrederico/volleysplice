import { rallySuppressionDecisionKey, type CutDraft } from "../../lib/cut-draft.ts";

type SuggestionIdentity = { id: string; logicalId: string };
type ReviewDecision = "pending" | "kept" | "excluded";

export function cleanupDecisionsForRally(
  current: Record<string, ReviewDecision>,
  rallyId: string,
  decision: ReviewDecision,
): Record<string, ReviewDecision> {
  return { ...current, [rallyId]: decision };
}

// Older Rally Desk edits used the displayed fragment ID. Prefer that explicit
// decision during migration, then retain only the key read by the cut engine.
export function normalizeCleanupDecisions(draft: CutDraft, suggestions: readonly SuggestionIdentity[]): CutDraft {
  const overrides = { ...draft.suppressionDecisionOverrides };
  for (const suggestion of suggestions) {
    if (suggestion.id === suggestion.logicalId) continue;
    const legacy = draft.suppressionDecisionOverrides[suggestion.id];
    if (legacy) overrides[suggestion.logicalId] = legacy;
    delete overrides[suggestion.id];
  }
  return { ...draft, suppressionDecisionOverrides: overrides };
}

export function cleanupReviewDecisions(draft: CutDraft, suggestions: readonly (SuggestionIdentity & { start: number; end: number; eligiblePolicyIds?: readonly string[]; decision: "pending" | "keep" | "suppress" })[]): Record<string, ReviewDecision> {
  const normalized = normalizeCleanupDecisions(draft, suggestions);
  return Object.fromEntries(draft.cuts.flatMap((cut) => {
    const related = suggestions.filter((item) => cut.coreStart < item.end && item.start < cut.coreEnd);
    const own = normalized.suppressionDecisionOverrides[rallySuppressionDecisionKey(cut.id)];
    if (!own && related.length === 0) return [];
    const decision = own ?? related.map((item) => normalized.suppressionDecisionOverrides[item.logicalId]
      ?? (item.eligiblePolicyIds
        ? item.eligiblePolicyIds.includes(draft.selectedSuppressionPolicy) ? "pending" : "keep"
        : item.decision))[0];
    return [[cut.id, decision === "keep" ? "kept" : decision === "suppress" ? "excluded" : "pending"]];
  }));
}
