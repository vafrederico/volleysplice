package com.volleycut.nativeanalysis;

/** Frozen bundles: encoder, temporal weights, normalizer and decoder move together. */
final class RallyModels {
    static final String RECALL = "distilled-large-recall-v1";
    static final String F1 = "distilled-large-f1-v1";
    static final String DEFAULT = RECALL;
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
        if (RECALL.equals(id)) return "Highest recall";
        if (F1.equals(id)) return "Highest F1";
        if (FeatureSchema.MODEL_ID.equals(id)) return "Ensemble";
        return "Unavailable model";
    }
    static String label(String id) {
        if (RECALL.equals(id)) return "Distilled Large · highest recall";
        if (F1.equals(id)) return "Distilled Large · highest F1";
        if (FeatureSchema.MODEL_ID.equals(id)) return "Production ensemble";
        return "Unavailable rally model";
    }
}
