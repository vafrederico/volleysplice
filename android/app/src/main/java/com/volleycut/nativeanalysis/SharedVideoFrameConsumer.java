package com.volleycut.nativeanalysis;

import android.media.Image;
import com.volleycut.video.YuvColorConversion;
import java.io.IOException;
import java.util.Set;

/** Optional second consumer of original decoded images; the decoder owns each Image. */
interface SharedVideoFrameConsumer extends AutoCloseable {
    /** Return actual PTS to materialize, selected from the already scanned inventory. */
    Set<Long> plan(long[] sortedPresentationUs, String decoderName, boolean hardware) throws IOException;
    boolean wants(long presentationUs);
    /** Copy/prepare inputs before returning. Never retain or close the borrowed Image. */
    void accept(Image image, long presentationUs, YuvColorConversion.Profile color) throws IOException;
    /** Drain pending work and validate complete coverage before inference may proceed. */
    void finish() throws IOException;
    @Override void close() throws IOException;
}
