package com.volleycut.video;

import static org.junit.Assert.*;
import org.junit.Test;
import java.util.Set;

public final class NearestFrameSelectionTest {
    @Test public void choosesActualNearestPtsAndEarlierOnTies() {
        assertArrayEquals(new long[]{0, 33_333, 33_333, 66_667},
                NearestFrameSelection.select(new long[]{0, 33_333, 66_667},
                        new double[]{0, .030, .050, .060}, .1));
    }

    @Test public void usesVariableRatePtsRatherThanDeclaredFrameRate() {
        assertArrayEquals(new long[]{0, 491_200, 1_003_050, 1_498_990},
                NearestFrameSelection.select(new long[]{0, 491_200, 525_400, 1_003_050, 1_498_990},
                        new double[]{0, .5, 1, 1.5}, 2));
    }

    @Test public void boundedWindowIncludesPriorFrameAndExcludesEndTarget() {
        assertArrayEquals(new long[]{970_000, 1_490_000},
                NearestFrameSelection.select(new long[]{970_000, 1_040_000, 1_490_000},
                        new double[]{1, 1.5, 2}, 2));
    }

    @Test public void eofRetainsFinalGridRowUsingLastFrame() {
        assertArrayEquals(new long[]{0, 470_000, 970_000},
                NearestFrameSelection.select(new long[]{0, 470_000, 970_000},
                        new double[]{0, .5, 1}, 1.01));
    }

    @Test public void sourceFrameLimitDoesNotInventFramesBeyondItsWindow() {
        assertArrayEquals(new long[]{0, 250_000},
                NearestFrameSelection.select(new long[]{0, 250_000},
                        new double[]{0, .25, .5, .75}, .5));
    }

    @Test public void duplicatePtsAndRequestsKeepOneSelectedImageForEachGridRow() {
        assertArrayEquals(new long[]{0, 0, 0, 600_000, 600_000},
                NearestFrameSelection.select(new long[]{0, 0, 600_000, 600_000},
                        new double[]{0, .25, .25, .5, .75}, 1));
    }

    @Test public void resumeRetainsWarmupImageEvenWhenAnotherRowSharesItsPts() {
        long[] selected = {0, 500_000, 500_000, 1_000_000};
        assertEquals(Set.of(500_000L, 1_000_000L),
                NearestFrameSelection.materializedTimestamps(selected, 1));
        assertEquals(Set.of(), NearestFrameSelection.materializedTimestamps(selected, 4));
    }

    @Test public void emptyInputProducesNoPlan() {
        assertArrayEquals(new long[0], NearestFrameSelection.select(new long[0], new double[]{0}, 1));
    }

    @Test(expected = IllegalArgumentException.class) public void rejectsUnsortedSourcePts() {
        NearestFrameSelection.select(new long[]{1, 0}, new double[]{0}, 1);
    }

    @Test(expected = IllegalArgumentException.class) public void rejectsUnsortedTargets() {
        NearestFrameSelection.select(new long[]{0}, new double[]{.5, 0}, 1);
    }
}
