package com.volleycut.video;

import java.nio.ByteBuffer;

/** Stateless JNI bridge; direct Image planes remain owned by the caller. */
final class NativeYuvArea {
    static final boolean AVAILABLE = load();

    private static boolean load() {
        try {
            System.loadLibrary("volleycut_yuv");
            return true;
        } catch (UnsatisfiedLinkError unavailable) {
            // Desktop JVM tests and unsupported runtimes retain the same Java math.
            return false;
        }
    }

    static native void resample(ByteBuffer y, int yRow, int yPixel, int yLimit,
            ByteBuffer u, int uRow, int uPixel, int uLimit,
            ByteBuffer v, int vRow, int vPixel, int vLimit,
            int[] geometry, int[] colorTables, byte[] output);

    private NativeYuvArea() {}
}
