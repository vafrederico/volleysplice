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

    @Test
    public void parsesAudioDecoderModes() {
        assertEquals(
                AnalysisTypes.AudioDecoderMode.SINGLE_ACCESS_UNIT,
                AnalysisTypes.AudioDecoderMode.fromWireName(null)
        );
        assertEquals(
                AnalysisTypes.AudioDecoderMode.BATCHED_ACCESS_UNITS,
                AnalysisTypes.AudioDecoderMode.fromWireName("Batched")
        );
        assertEquals(
                AnalysisTypes.AudioDecoderMode.AUTO,
                AnalysisTypes.AudioDecoderMode.fromWireName("auto")
        );
        assertEquals("batched", AnalysisTypes.AudioDecoderMode.BATCHED_ACCESS_UNITS.wireName());
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnknownAudioDecoderMode() {
        AnalysisTypes.AudioDecoderMode.fromWireName("parallel");
    }
}
