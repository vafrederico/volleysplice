package com.volleycut.nativeanalysis;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Mirrors the production overlap-union-disagreement-v1 interval contract. */
final class ProductionEnsemble {
    static final String BOTH_MODELS = "both-models";
    static final String ALL_LABELS_V2_ONLY = "all-labels-v2-only";
    static final String PREVIOUS_PRODUCTION_ONLY = "previous-production-only";

    private static final float DISAGREEMENT_CONFIDENCE_CEILING = 0.49f;
    private static final float DISAGREEMENT_CONFIDENCE_SCALE = 0.6f;

    private enum Source { ALL_LABELS_V2, PREVIOUS_PRODUCTION }

    private record Candidate(AnalysisTypes.Interval interval, Source source) {}

    private ProductionEnsemble() {}

    static boolean isValidAgreement(String value) {
        return BOTH_MODELS.equals(value)
                || ALL_LABELS_V2_ONLY.equals(value)
                || PREVIOUS_PRODUCTION_ONLY.equals(value);
    }

    static boolean isDisagreement(String value) {
        return ALL_LABELS_V2_ONLY.equals(value) || PREVIOUS_PRODUCTION_ONLY.equals(value);
    }

    static List<AnalysisTypes.Interval> merge(
            List<AnalysisTypes.Interval> allLabelsV2,
            List<AnalysisTypes.Interval> previousProduction
    ) {
        ArrayList<Candidate> candidates = new ArrayList<>(
                allLabelsV2.size() + previousProduction.size()
        );
        addCandidates(candidates, allLabelsV2, Source.ALL_LABELS_V2);
        addCandidates(candidates, previousProduction, Source.PREVIOUS_PRODUCTION);
        candidates.sort(Comparator
                .comparingDouble((Candidate candidate) -> candidate.interval().start())
                .thenComparingDouble(candidate -> candidate.interval().end()));

        ArrayList<List<Candidate>> clusters = new ArrayList<>();
        double currentEnd = Double.NEGATIVE_INFINITY;
        for (Candidate candidate : candidates) {
            if (clusters.isEmpty() || candidate.interval().start() >= currentEnd) {
                ArrayList<Candidate> cluster = new ArrayList<>();
                cluster.add(candidate);
                clusters.add(cluster);
                currentEnd = candidate.interval().end();
            } else {
                clusters.get(clusters.size() - 1).add(candidate);
                currentEnd = Math.max(currentEnd, candidate.interval().end());
            }
        }

        ArrayList<AnalysisTypes.Interval> merged = new ArrayList<>(clusters.size());
        for (List<Candidate> cluster : clusters) {
            double start = Double.POSITIVE_INFINITY;
            double end = Double.NEGATIVE_INFINITY;
            float allLabelsConfidence = Float.NEGATIVE_INFINITY;
            float previousConfidence = Float.NEGATIVE_INFINITY;
            boolean hasAllLabels = false;
            boolean hasPrevious = false;
            for (Candidate candidate : cluster) {
                AnalysisTypes.Interval interval = candidate.interval();
                start = Math.min(start, interval.start());
                end = Math.max(end, interval.end());
                if (candidate.source() == Source.ALL_LABELS_V2) {
                    hasAllLabels = true;
                    allLabelsConfidence = Math.max(allLabelsConfidence, interval.confidence());
                } else {
                    hasPrevious = true;
                    previousConfidence = Math.max(previousConfidence, interval.confidence());
                }
            }

            String agreement;
            float confidence;
            if (hasAllLabels && hasPrevious) {
                agreement = BOTH_MODELS;
                confidence = (allLabelsConfidence + previousConfidence) / 2f;
            } else if (hasAllLabels) {
                agreement = ALL_LABELS_V2_ONLY;
                confidence = Math.min(
                        DISAGREEMENT_CONFIDENCE_CEILING,
                        allLabelsConfidence * DISAGREEMENT_CONFIDENCE_SCALE
                );
            } else {
                agreement = PREVIOUS_PRODUCTION_ONLY;
                confidence = Math.min(
                        DISAGREEMENT_CONFIDENCE_CEILING,
                        previousConfidence * DISAGREEMENT_CONFIDENCE_SCALE
                );
            }
            merged.add(new AnalysisTypes.Interval(start, end, confidence, agreement));
        }
        return List.copyOf(merged);
    }

    private static void addCandidates(
            List<Candidate> target,
            List<AnalysisTypes.Interval> intervals,
            Source source
    ) {
        for (AnalysisTypes.Interval interval : intervals) {
            if (Double.isFinite(interval.start())
                    && Double.isFinite(interval.end())
                    && interval.end() > interval.start()) {
                target.add(new Candidate(interval, source));
            }
        }
    }
}
