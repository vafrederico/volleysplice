package com.volleycut.nativeanalysis;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.List;

public final class ProductionEnsembleTest {
    @Test
    public void unionsOverlapsAndFlagsEverySingleModelRange() {
        List<AnalysisTypes.Interval> merged = ProductionEnsemble.merge(
                List.of(
                        interval(10, 18, .9f),
                        interval(30, 35, .8f),
                        interval(50, 55, .7f)
                ),
                List.of(
                        interval(12, 20, .8f),
                        interval(35, 40, .95f),
                        interval(70, 75, .6f)
                )
        );

        assertEquals(5, merged.size());
        assertRange(merged.get(0), 10, 20, ProductionEnsemble.BOTH_MODELS);
        assertRange(merged.get(1), 30, 35, ProductionEnsemble.ALL_LABELS_V2_ONLY);
        assertRange(merged.get(2), 35, 40, ProductionEnsemble.PREVIOUS_PRODUCTION_ONLY);
        assertRange(merged.get(3), 50, 55, ProductionEnsemble.ALL_LABELS_V2_ONLY);
        assertRange(merged.get(4), 70, 75, ProductionEnsemble.PREVIOUS_PRODUCTION_ONLY);
        assertEquals(.85f, merged.get(0).confidence(), 1e-6f);
        for (AnalysisTypes.Interval interval : merged.subList(1, merged.size())) {
            assertTrue(interval.confidence() <= .49f);
        }
    }

    @Test
    public void transitivelyOverlappingDetectionsFormOneAgreementRange() {
        List<AnalysisTypes.Interval> merged = ProductionEnsemble.merge(
                List.of(interval(10, 15, .9f), interval(18, 22, .8f)),
                List.of(interval(14, 19, .7f))
        );

        assertEquals(1, merged.size());
        assertRange(merged.get(0), 10, 22, ProductionEnsemble.BOTH_MODELS);
    }

    @Test
    public void touchingDetectionsRemainSeparateDisagreements() {
        List<AnalysisTypes.Interval> merged = ProductionEnsemble.merge(
                List.of(interval(1, 3, .9f)),
                List.of(interval(3, 5, .8f))
        );

        assertEquals(2, merged.size());
        assertEquals(ProductionEnsemble.ALL_LABELS_V2_ONLY, merged.get(0).agreement());
        assertEquals(ProductionEnsemble.PREVIOUS_PRODUCTION_ONLY, merged.get(1).agreement());
    }

    private static AnalysisTypes.Interval interval(double start, double end, float confidence) {
        return new AnalysisTypes.Interval(start, end, confidence);
    }

    private static void assertRange(
            AnalysisTypes.Interval actual,
            double start,
            double end,
            String agreement
    ) {
        assertEquals(start, actual.start(), 0);
        assertEquals(end, actual.end(), 0);
        assertEquals(agreement, actual.agreement());
    }
}
