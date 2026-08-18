package com.volleycut.nativeanalysis;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public final class AudioFeatureExtractorTest {
    @Test
    public void resamplesChunked48KhzPcmWithoutCrossingTheSourceBuffer() {
        AudioFeatureExtractor extractor = new AudioFeatureExtractor();
        int sampleRate = 48_000;
        int sourceFrames = sampleRate;
        int offset = 0;
        while (offset < sourceFrames) {
            int length = Math.min(1_024, sourceFrames - offset);
            float[] chunk = new float[length];
            for (int index = 0; index < length; index++) {
                chunk[index] = (float) Math.sin(2 * Math.PI * 440 * (offset + index) / sampleRate);
            }
            extractor.push(chunk, offset / (double) sampleRate, sampleRate);
            offset += length;
        }

        double[] analysisTimes = {0, 0.25, 0.5, 0.75};
        float[] features = extractor.finishAndPool(analysisTimes);
        assertEquals(analysisTimes.length * FeatureSchema.AUDIO.size(), features.length);
        assertTrue(extractor.resampledOutputSamples() >= 15_999);
        for (int row = 0; row < analysisTimes.length; row++) {
            assertEquals(1, features[row * FeatureSchema.AUDIO.size()], 0);
        }
    }

    @Test
    public void reportsPostDecodeAudioFeatureStepsInOrder() {
        AudioFeatureExtractor extractor = new AudioFeatureExtractor();
        extractor.push(new float[48_000], 0, 48_000);
        List<String> steps = new ArrayList<>();

        extractor.finishAndPool(
                new double[] {0, 0.25, 0.5, 0.75},
                (fraction, detail) -> steps.add(String.format("%.3f %s", fraction, detail))
        );

        assertEquals(List.of(
                "0.820 Flushing resampler + final FFT frames",
                "0.880 Computing audio ranks + rolling noise floors",
                "0.950 Pooling audio features to 4 Hz video timestamps",
                "0.990 Audio DSP complete; preparing feature cache"
        ), steps);
    }
}
