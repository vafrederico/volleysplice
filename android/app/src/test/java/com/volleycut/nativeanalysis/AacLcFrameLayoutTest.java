package com.volleycut.nativeanalysis;

import static org.junit.Assert.*;
import java.nio.ByteBuffer;
import org.junit.Test;

public final class AacLcFrameLayoutTest {
    @Test public void readsActualCodecFrameSizeInsteadOfPacketSpacing() {
        ByteBuffer config = ByteBuffer.wrap(new byte[]{0x11, (byte) 0x90});
        AacLcFrameLayout layout = AacLcFrameLayout.parse(config, 48000, 2);
        assertEquals(new AacLcFrameLayout(48000, 2, 1024), layout);
        assertEquals(0, config.position());
        assertEquals(960, AacLcFrameLayout.parse(
                ByteBuffer.wrap(new byte[]{0x11, (byte) 0x94}), 48000, 2).framesPerUnit());
    }

    @Test public void readsExplicitFrequency() {
        long bits = (2L << 35) | (15L << 31) | (48000L << 7) | (2L << 3);
        byte[] config = new byte[5];
        for (int i = 0; i < 5; i++) config[i] = (byte) (bits >> ((4 - i) * 8));
        assertEquals(new AacLcFrameLayout(48000, 2, 1024),
                AacLcFrameLayout.parse(ByteBuffer.wrap(config), 48000, 2));
    }

    @Test public void rejectsUnknownProfilesExtensionsAndInconsistentFormats() {
        for (byte[] config : new byte[][]{
                {}, {0x11}, {0x11, (byte) 0x90, 0},
                {0x29, (byte) 0x90}, // SBR: decoded frame count may double.
                {0x11, (byte) 0x92}, // dependsOnCoreCoder
                {0x11, (byte) 0x91}, // extensionFlag
                {0x11, (byte) 0x80}, // program-config element
        }) assertNull(AacLcFrameLayout.parse(ByteBuffer.wrap(config), 48000, 2));
        assertNull(AacLcFrameLayout.parse(null, 48000, 2));
        assertNull(AacLcFrameLayout.parse(ByteBuffer.wrap(new byte[]{0x11, (byte) 0x90}), 24000, 2));
        assertNull(AacLcFrameLayout.parse(ByteBuffer.wrap(new byte[]{0x11, (byte) 0x90}), 48000, 1));
    }
}
