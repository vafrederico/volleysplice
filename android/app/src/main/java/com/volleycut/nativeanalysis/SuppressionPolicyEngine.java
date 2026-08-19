package com.volleycut.nativeanalysis;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/** Language-neutral millisecond policy algebra from the suppression product contract. */
final class SuppressionPolicyEngine {
    enum Policy {
        NONE("none", "No suppression", "none", 0, 0),
        CONSERVATIVE("conservative", "Conservative", "zero-non-exempt-misses", 2_000, 500),
        BALANCED("balanced", "Balanced", "aggressive-intermediate", 1_500, 500),
        AGGRESSIVE("aggressive", "Aggressive", "raw-connected", 0, 0);

        final String wireName;
        final String label;
        final String recordedPolicy;
        final long agreementPaddingMs;
        final long agreementJoinMs;

        Policy(
                String wireName, String label, String recordedPolicy,
                long agreementPaddingMs, long agreementJoinMs
        ) {
            this.wireName = wireName;
            this.label = label;
            this.recordedPolicy = recordedPolicy;
            this.agreementPaddingMs = agreementPaddingMs;
            this.agreementJoinMs = agreementJoinMs;
        }

        static Policy fromWireName(String value) {
            for (Policy policy : values()) if (policy.wireName.equals(value)) return policy;
            return NONE;
        }
    }

    private enum Source { ALL_LABELS_V2, PREVIOUS_PRODUCTION }
    private record Tagged(
            String id, long startMs, long endMs, float confidence, Source source,
            long paddedStartMs, long paddedEndMs
    ) {}
    private record Span(long startMs, long endMs) {}
    private record PolicySpan(long startMs, long endMs, Policy policy) {}

    private SuppressionPolicyEngine() {}

    static AnalysisTypes.SuppressionAnalysis build(
            List<AnalysisTypes.Interval> allLabelsV2,
            List<AnalysisTypes.Interval> previousProduction,
            float[] probabilities,
            List<AnalysisTypes.Interval> decoded,
            double durationSeconds
    ) {
        long durationMs = milliseconds(durationSeconds);
        List<Tagged> raw = tagged(allLabelsV2, previousProduction, durationMs);
        ArrayList<PolicySpan> policySpans = new ArrayList<>();
        for (Policy policy : List.of(Policy.CONSERVATIVE, Policy.BALANCED, Policy.AGGRESSIVE)) {
            for (Span span : eligibleRawSpans(raw, policy, durationMs)) {
                policySpans.add(new PolicySpan(span.startMs, span.endMs, policy));
            }
        }

        ArrayList<AnalysisTypes.SuppressionSuggestion> suggestions = new ArrayList<>();
        for (int decodedIndex = 0; decodedIndex < decoded.size(); decodedIndex++) {
            AnalysisTypes.Interval event = decoded.get(decodedIndex);
            long eventStart = Math.max(0, milliseconds(event.start()));
            long eventEnd = Math.min(durationMs, milliseconds(event.end()));
            if (eventEnd <= eventStart) continue;
            List<Tagged> eventSources = raw.stream()
                    .filter(item -> overlaps(eventStart, eventEnd, item.startMs, item.endMs))
                    .toList();
            if (eventSources.isEmpty()) continue;
            String logicalId = logicalId(eventStart, eventEnd, eventSources);
            LinkedHashSet<Long> boundaries = new LinkedHashSet<>();
            boundaries.add(eventStart);
            boundaries.add(eventEnd);
            for (PolicySpan candidate : policySpans) {
                long start = Math.max(eventStart, candidate.startMs);
                long end = Math.min(eventEnd, candidate.endMs);
                if (end > start) {
                    boundaries.add(start);
                    boundaries.add(end);
                }
            }
            List<Long> ordered = boundaries.stream().sorted().toList();
            ArrayList<AnalysisTypes.SuppressionSuggestion> atoms = new ArrayList<>();
            for (int index = 0; index + 1 < ordered.size(); index++) {
                long start = ordered.get(index);
                long end = ordered.get(index + 1);
                if (end <= start) continue;
                long midpoint = start + (end - start) / 2;
                List<String> eligible = List.of(Policy.CONSERVATIVE, Policy.BALANCED, Policy.AGGRESSIVE)
                        .stream().filter(policy -> policySpans.stream().anyMatch(span ->
                                span.policy == policy && midpoint >= span.startMs && midpoint < span.endMs
                        )).map(policy -> policy.wireName).toList();
                if (eligible.isEmpty()) continue;
                List<String> sourceIds = eventSources.stream()
                        .filter(item -> overlaps(start, end, item.startMs, item.endMs))
                        .map(Tagged::id).sorted().toList();
                if (sourceIds.isEmpty()) continue;
                atoms.add(new AnalysisTypes.SuppressionSuggestion(
                        logicalId,
                        logicalId + ":" + start + ":" + end,
                        start,
                        end,
                        event.confidence(),
                        sourceIds,
                        eligible
                ));
            }
            for (AnalysisTypes.SuppressionSuggestion atom : atoms) {
                AnalysisTypes.SuppressionSuggestion previous = suggestions.isEmpty()
                        ? null : suggestions.get(suggestions.size() - 1);
                if (previous != null && previous.logicalId().equals(atom.logicalId())
                        && previous.endMs() == atom.startMs()
                        && previous.eligiblePolicyIds().equals(atom.eligiblePolicyIds())) {
                    ArrayList<String> ids = new ArrayList<>(previous.sourceProductionIds());
                    atom.sourceProductionIds().forEach(id -> { if (!ids.contains(id)) ids.add(id); });
                    ids.sort(String::compareTo);
                    suggestions.set(suggestions.size() - 1, new AnalysisTypes.SuppressionSuggestion(
                            previous.logicalId(),
                            previous.logicalId() + ":" + previous.startMs() + ":" + atom.endMs(),
                            previous.startMs(), atom.endMs(),
                            Math.max(previous.score(), atom.score()),
                            List.copyOf(ids), previous.eligiblePolicyIds()
                    ));
                } else suggestions.add(atom);
            }
        }
        return new AnalysisTypes.SuppressionAnalysis(
                FeatureSchema.SUPPRESSION_MODEL_ID,
                FeatureSchema.SUPPRESSION_ARTIFACT_SHA256,
                FeatureSchema.SUPPRESSION_WEIGHTS_SHA256,
                FeatureSchema.SUPPRESSION_DECODER_VERSION,
                probabilities.clone(),
                List.copyOf(decoded),
                List.copyOf(suggestions)
        );
    }

    static List<AnalysisTypes.SuppressionSuggestion> active(
            AnalysisTypes.SuppressionAnalysis analysis, Policy policy
    ) {
        if (analysis == null || policy == Policy.NONE) return List.of();
        return analysis.suggestions().stream()
                .filter(item -> item.eligiblePolicyIds().contains(policy.wireName))
                .sorted(Comparator.comparingLong(AnalysisTypes.SuppressionSuggestion::startMs)
                        .thenComparingLong(AnalysisTypes.SuppressionSuggestion::endMs))
                .toList();
    }

    private static List<Tagged> tagged(
            List<AnalysisTypes.Interval> allLabelsV2,
            List<AnalysisTypes.Interval> previousProduction,
            long durationMs
    ) {
        ArrayList<Tagged> result = new ArrayList<>();
        addTagged(result, allLabelsV2, Source.ALL_LABELS_V2, "all-labels-v2", durationMs);
        addTagged(result, previousProduction, Source.PREVIOUS_PRODUCTION,
                "previous-production", durationMs);
        return List.copyOf(result);
    }

    private static void addTagged(
            List<Tagged> target, List<AnalysisTypes.Interval> source, Source kind,
            String prefix, long durationMs
    ) {
        for (int index = 0; index < source.size(); index++) {
            AnalysisTypes.Interval interval = source.get(index);
            long start = Math.max(0, milliseconds(interval.start()));
            long end = Math.min(durationMs, milliseconds(interval.end()));
            if (end <= start) continue;
            target.add(new Tagged(
                    prefix + ":" + String.format("%04d", index + 1) + ":" + start + ":" + end,
                    start, end, interval.confidence(), kind, start, end
            ));
        }
    }

    private static List<Span> eligibleRawSpans(List<Tagged> raw, Policy policy, long durationMs) {
        List<Tagged> padded = raw.stream().map(item -> new Tagged(
                item.id, item.startMs, item.endMs, item.confidence, item.source,
                Math.max(0, item.startMs - policy.agreementPaddingMs),
                Math.min(durationMs, item.endMs + policy.agreementPaddingMs)
        )).sorted(Comparator.comparingLong(Tagged::paddedStartMs)
                .thenComparingLong(Tagged::paddedEndMs)).toList();
        ArrayList<List<Tagged>> components = new ArrayList<>();
        long componentEnd = Long.MIN_VALUE;
        for (Tagged item : padded) {
            long gap = item.paddedStartMs - componentEnd;
            if (components.isEmpty() || (gap > 0 && gap >= policy.agreementJoinMs)) {
                components.add(new ArrayList<>());
                componentEnd = item.paddedEndMs;
            } else componentEnd = Math.max(componentEnd, item.paddedEndMs);
            components.get(components.size() - 1).add(item);
        }
        ArrayList<Span> eligible = new ArrayList<>();
        for (List<Tagged> component : components) {
            Set<Source> sources = new HashSet<>();
            component.forEach(item -> sources.add(item.source));
            if (sources.size() != 1) continue;
            component.forEach(item -> eligible.add(new Span(item.startMs, item.endMs)));
        }
        return merge(eligible);
    }

    private static List<Span> merge(List<Span> source) {
        ArrayList<Span> merged = new ArrayList<>();
        source.stream().sorted(Comparator.comparingLong(Span::startMs)
                .thenComparingLong(Span::endMs)).forEach(item -> {
            Span previous = merged.isEmpty() ? null : merged.get(merged.size() - 1);
            if (previous == null || item.startMs > previous.endMs) merged.add(item);
            else merged.set(merged.size() - 1,
                    new Span(previous.startMs, Math.max(previous.endMs, item.endMs)));
        });
        return List.copyOf(merged);
    }

    private static String logicalId(long start, long end, List<Tagged> source) {
        String canonical = start + ":" + end + ":" + source.stream()
                .map(Tagged::id).sorted().reduce((left, right) -> left + "," + right).orElse("");
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(canonical.getBytes(StandardCharsets.UTF_8));
            StringBuilder value = new StringBuilder("S-");
            for (int index = 0; index < 10; index++) value.append(String.format("%02x", digest[index]));
            return value.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException(error);
        }
    }

    private static boolean overlaps(long leftStart, long leftEnd, long rightStart, long rightEnd) {
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    private static long milliseconds(double seconds) {
        return Math.round(seconds * 1_000.0);
    }
}
