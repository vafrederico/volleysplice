package com.volleycut.nativeanalysis;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import java.nio.ByteBuffer;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

public final class NativeAudioDecoderTest {
    @Test
    public void keepsCompleteOutputAccessUnitBeforeWindowEnd() {
        assertEquals(
                1024,
                NativeAudioDecoder.outputFramesToKeep(1024, 1_000_000, 2_000_000, 48_000)
        );
    }

    @Test
    public void clipsPartialFinalOutputWithoutRequiringAnotherInputTimestamp() {
        assertEquals(
                1008,
                NativeAudioDecoder.outputFramesToKeep(1024, 0, 21_000, 48_000)
        );
    }

    @Test
    public void dropsOutputAtOrAfterWindowEnd() {
        assertEquals(
                0,
                NativeAudioDecoder.outputFramesToKeep(1024, 2_000_000, 2_000_000, 48_000)
        );
        assertEquals(
                0,
                NativeAudioDecoder.outputFramesToKeep(1024, 2_100_000, 2_000_000, 48_000)
        );
    }

    @Test
    public void carriesPartialAccessUnitsAcrossLargeOutputBuffers() {
        NativeAudioDecoder.DecodedAccessUnitAssembler assembler =
                new NativeAudioDecoder.DecodedAccessUnitAssembler();
        assembler.addInputTimes(new long[]{0, 1_000, 2_000, 3_000});
        assembler.setFramesPerUnit(4);
        List<Integer> lengths = new ArrayList<>();
        List<Long> timestamps = new ArrayList<>();
        NativeAudioDecoder.DecodedAccessUnitAssembler.Consumer consumer = (frames, timeUs) -> {
            lengths.add(frames.length);
            timestamps.add(timeUs);
        };

        assertEquals(1, assembler.accept(new float[6], 0, 4_000, consumer));
        assertEquals(1, assembler.accept(new float[5], 1_500, 4_000, consumer));
        assertEquals(1, assembler.flush(consumer));

        assertEquals(List.of(4, 4, 3), lengths);
        assertEquals(List.of(0L, 1_000L, 2_000L), timestamps);
        assertEquals(1, assembler.pendingInputTimeCount());
    }

    @Test
    public void rejectsPcmWithoutMatchingInputTimestampInsteadOfInventingPlacement() {
        NativeAudioDecoder.DecodedAccessUnitAssembler assembler =
                new NativeAudioDecoder.DecodedAccessUnitAssembler();
        assembler.setFramesPerUnit(4);
        List<Long> timestamps = new ArrayList<>();

        assertThrows(NativeAudioDecoder.BatchFramingException.class, () ->
                assembler.accept(
                        new float[4],
                        123_000,
                        48_000,
                        (frames, timeUs) -> timestamps.add(timeUs)
                )
        );

        assertEquals(List.of(), timestamps);
    }

    @Test
    public void irregularFirstTimestampDoesNotChangeDecodedFrameSizeOrLoseAudio() {
        int count = 2000;
        long[] timestamps = new long[count];
        for (int i = 1; i < count; i++) timestamps[i] = Math.round((82L + i * 1024L) * 1e6 / 48000);
        NativeAudioDecoder.DecodedAccessUnitAssembler assembler = new NativeAudioDecoder.DecodedAccessUnitAssembler();
        assembler.setFramesPerUnit(AacLcFrameLayout.parse(
                ByteBuffer.wrap(new byte[]{0x11, (byte) 0x90}), 48000, 2).framesPerUnit());
        assembler.addInputTimes(timestamps);
        AudioFeatureExtractor features = new AudioFeatureExtractor();
        long[] consumed = {0};
        NativeAudioDecoder.DecodedAccessUnitAssembler.Consumer consumer = (pcm, pts) -> {
            consumed[0] += pcm.length;
            features.push(pcm, pts / 1e6, 48000);
        };
        // Uneven codec buffers must not affect decoded-unit accounting.
        int samples = count * 1024;
        for (int start = 0; start < samples; start += 17001) {
            assembler.accept(new float[Math.min(17001, samples - start)], 0, 48000, consumer);
        }
        assembler.flush(consumer);
        assembler.verifyComplete();
        features.finishAndPool(new double[]{0});
        assertEquals(samples, consumed[0]);
        assertEquals(0, assembler.pendingInputTimeCount());
        assertEquals(0, features.overlapTrimmedSamples());
        assertEquals((samples + 82 + 2) / 3, features.resampledOutputSamples());
    }

    @Test
    public void rejectsUnconsumedTimelineAndChangingFraming() {
        NativeAudioDecoder.DecodedAccessUnitAssembler assembler = new NativeAudioDecoder.DecodedAccessUnitAssembler();
        assembler.setFramesPerUnit(1024);
        assembler.addInputTimes(new long[]{0, 21333, 42667});
        assertThrows(NativeAudioDecoder.BatchFramingException.class, assembler::verifyComplete);
        assertThrows(NativeAudioDecoder.BatchFramingException.class, () -> assembler.setFramesPerUnit(1106));
    }

    @Test
    public void appliesEncoderDelayAndPaddingToFirstDecodedOutput() {
        NativeAudioDecoder.DecodedAccessUnitAssembler assembler =
                new NativeAudioDecoder.DecodedAccessUnitAssembler();
        assembler.addInputTimes(new long[]{0, 21_333, 42_667});
        assembler.setFramesPerUnit(1024);
        assembler.setGaplessTrimming(1024, 16);
        List<Integer> lengths = new ArrayList<>();
        List<Long> timestamps = new ArrayList<>();

        assertEquals(
                2,
                assembler.accept(
                        new float[2032],
                        -20_480,
                        48_000,
                        (frames, timeUs) -> {
                            lengths.add(frames.length);
                            timestamps.add(timeUs);
                        }
                )
        );

        assertEquals(List.of(1008, 1024), lengths);
        assertEquals(List.of(21_333L, 42_667L), timestamps);
        assertEquals(0, assembler.pendingInputTimeCount());
    }
}
