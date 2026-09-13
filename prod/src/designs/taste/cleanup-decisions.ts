import type { CutDraft } from "../../lib/cut-draft.ts";

type SuggestionIdentity = { id: string; logicalId: string };
type ReviewDecision = "pending" | "kept" | "excluded";

export function cleanupDecisionsForRally(
  current: Record<string, ReviewDecision>,
  suggestions: readonly (SuggestionIdentity & { cutId: string | null })[],
  rallyId: string,
  decision: ReviewDecision,
): Record<string, ReviewDecision> {
  const logicalIds = new Set(suggestions.filter((item) => item.cutId === rallyId).map((item) => item.logicalId));
  const next = { ...current, [rallyId]: decision };
  for (const item of suggestions) {
    if (item.cutId && logicalIds.has(item.logicalId)) next[item.cutId] = decision;
  }
  return next;
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
