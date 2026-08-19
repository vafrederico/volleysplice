package com.volleycut.nativeanalysis;

import static org.junit.Assert.assertEquals;

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
    public void usesDecoderTimestampIfOutputHasNoMatchingInputTimestamp() {
        NativeAudioDecoder.DecodedAccessUnitAssembler assembler =
                new NativeAudioDecoder.DecodedAccessUnitAssembler();
        assembler.setFramesPerUnit(4);
        List<Long> timestamps = new ArrayList<>();

        assertEquals(
                1,
                assembler.accept(
                        new float[4],
                        123_000,
                        48_000,
                        (frames, timeUs) -> timestamps.add(timeUs)
                )
        );

        assertEquals(List.of(123_000L), timestamps);
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
