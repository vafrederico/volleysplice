package com.volleycut.nativeanalysis;

import java.nio.ByteBuffer;

/** Conservative AudioSpecificConfig reader for batched, constant-frame AAC-LC only. */
record AacLcFrameLayout(int sampleRate, int channels, int framesPerUnit) {
    private static final int[] RATES = {
            96000, 88200, 64000, 48000, 44100, 32000, 24000,
            22050, 16000, 12000, 11025, 8000, 7350
    };

    static AacLcFrameLayout parse(ByteBuffer config, int trackRate, int trackChannels) {
        if (config == null) return null;
        ByteBuffer bytes = config.duplicate();
        // Basic AAC-LC is two bytes, or five with an explicit 24-bit rate.
        // Extensions (including SBR/PS), PCE, other profiles and unknown layouts
        // deliberately use synchronous decoding instead of guessing PCM framing.
        if (bytes.remaining() != 2 && bytes.remaining() != 5) return null;
        long bits = 0;
        int size = bytes.remaining();
        int remaining = size * 8;
        while (bytes.hasRemaining()) bits = (bits << 8) | (bytes.get() & 0xffL);
        if (((bits >> (remaining -= 5)) & 31) != 2) return null;
        int rateIndex = (int) ((bits >> (remaining -= 4)) & 15);
        if (size != (rateIndex == 15 ? 5 : 2)) return null;
        int rate;
        if (rateIndex == 15) {
            if (remaining < 24 + 7) return null;
            rate = (int) ((bits >> (remaining -= 24)) & 0xffffff);
        } else {
            if (rateIndex >= RATES.length) return null;
            rate = RATES[rateIndex];
        }
        int channels = (int) ((bits >> (remaining -= 4)) & 15);
        boolean shortFrame = ((bits >> (remaining -= 1)) & 1) != 0;
        // dependsOnCoreCoder, extensionFlag and any unused bits must be zero.
        if ((bits & ((1L << remaining) - 1)) != 0
                || rate != trackRate || channels != trackChannels
                || channels < 1 || channels > 2) return null;
        return new AacLcFrameLayout(rate, channels, shortFrame ? 960 : 1024);
    }
}
