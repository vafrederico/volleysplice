package com.volleycut.video;

import static org.junit.Assert.*;
import org.junit.Test;

public class EmbeddingFrameRequestsTest {
    @Test public void exclusiveEndDoesNotBorrowAnAvFrameBeyondTheWindow() {
        long[] pts={0,300_000,510_000};
        // AV may select 0.51 for a target at 0.5. The neural window ends at 0.505.
        assertArrayEquals(new long[]{0,510_000},NearestFrameSelection.select(pts,new double[]{0,.5},1));
        var plan=EmbeddingFrameRequests.create(pts,0,.505,2);
        assertArrayEquals(new long[]{0,300_000},plan.selectedPresentationUs());
    }
    @Test public void repeatedSparseFramesAndEarlierTiesKeepNominalGrid() {
        var plan=EmbeddingFrameRequests.create(new long[]{0,1_000_000},0,1.2,2);
        assertArrayEquals(new double[]{0,.5,1},plan.targetSeconds(),0);
        assertArrayEquals(new long[]{0,0,1_000_000},plan.selectedPresentationUs());
    }
    @Test public void irregularPtsMatchSubsampledAvRequestsWhenWindowBoundsAgree() {
        long[] pts={0,247_123,480_021,764_100,1_017_000,1_226_700,1_499_123};
        long[] av=NearestFrameSelection.select(pts,new double[]{0,.25,.5,.75,1,1.25,1.5},1.6);
        var neural=EmbeddingFrameRequests.create(pts,0,1.6,2);
        assertArrayEquals(new long[]{av[0],av[2],av[4],av[6]},neural.selectedPresentationUs());
    }
    @Test public void emptyInventoryProducesNoInventedFrames() {
        var plan=EmbeddingFrameRequests.create(new long[0],0,1,2);
        assertEquals(0,plan.selectedPresentationUs().length);
        assertEquals(0,plan.targetSeconds().length);
    }
    @Test(expected=IllegalArgumentException.class) public void rejectsUnsortedSource() {
        EmbeddingFrameRequests.create(new long[]{500_000,0},0,1,2);
    }
    @Test(expected=IllegalArgumentException.class) public void rejectsInvalidRate() {
        EmbeddingFrameRequests.create(new long[]{0},0,1,Double.NaN);
    }
}
