package com.volleycut.nativeanalysis;

/** Frozen bundles: encoder, temporal weights, normalizer and decoder move together. */
final class RallyModels {
    static final String RECALL = "distilled-large-recall-v1";
    static final String F1 = "distilled-large-f1-v1";
    static final String DEFAULT = F1;
    static final String[] OPTIONS = {F1, RECALL, FeatureSchema.MODEL_ID};
    private RallyModels() {}

    static boolean isNeural(String id) { return RECALL.equals(id) || F1.equals(id); }
    static boolean isValidAgreement(String value) { return "neural".equals(value) || ProductionEnsemble.isValidAgreement(value); }
    static boolean isSupported(String id) { return isNeural(id) || FeatureSchema.MODEL_ID.equals(id); }
    static String variant(String id) {
        if (RECALL.equals(id)) return "high-recall";
        if (F1.equals(id)) return "high-f1";
        throw new IllegalArgumentException("Unknown neural rally model");
    }
    static String shortLabel(String id) {
        if (RECALL.equals(id)) return "Maximum coverage · BETA";
        if (F1.equals(id)) return "Balanced · BETA";
        if (FeatureSchema.MODEL_ID.equals(id)) return "Legacy model";
        return "Unavailable model";
    }
    static String label(String id) {
        return shortLabel(id);
    }
    static String description(String id) {
        if (F1.equals(id)) return "Balances retained play and extra footage with tighter cuts.";
        if (RECALL.equals(id)) return "Keeps more possible play, with more extra footage to review.";
        return "Uses the previous production detector.";
    }
}
