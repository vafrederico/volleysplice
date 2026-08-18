package com.volleycut.nativeanalysis;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class AnalysisStagesTest {
    @Test
    public void parsesSingleStagesAndCombinations() {
        assertEquals(
                new AnalysisTypes.AnalysisStages(false, true, false),
                AnalysisTypes.AnalysisStages.fromWireName("audio")
        );
        assertEquals(
                new AnalysisTypes.AnalysisStages(true, false, true),
                AnalysisTypes.AnalysisStages.fromWireName("inference,video")
        );
        assertEquals(
                "video,inference",
                new AnalysisTypes.AnalysisStages(true, false, true).wireName()
        );
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnknownStages() {
        AnalysisTypes.AnalysisStages.fromWireName("audio,export");
    }
}
