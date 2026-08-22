package com.volleycut.nativeanalysis;

import android.content.Context;
import android.graphics.ImageFormat;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.BooleanSupplier;

/** Samples display-oriented 192x108 grayscale ROI frames at arbitrary source timestamps. */
final class ServingSideFrameDecoder {
    private static final long TIMEOUT_US = 10_000;
    private final Context context;

    ServingSideFrameDecoder(Context context) {
        this.context = context.getApplicationContext();
    }

    Map<Double, byte[]> decode(
            Uri uri,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            double[] requestedTimes,
            AnalysisTypes.ProgressListener progress,
            BooleanSupplier cancelled
    ) throws IOException {
        LinkedHashMap<Double, byte[]> result = new LinkedHashMap<>();
        if (requestedTimes.length == 0) return result;
        for (int i = 0; i < requestedTimes.length; i++) {
            if (!Double.isFinite(requestedTimes[i]) || requestedTimes[i] < 0
                    || (i > 0 && requestedTimes[i] <= requestedTimes[i - 1])) {
                throw new IllegalArgumentException("Serving-side timestamps must be finite, unique, and sorted");
            }
        }
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;
        try {
            extractor.setDataSource(context, uri, null);
            int track = NativeVideoDecoder.findTrack(extractor, "video/");
            if (track < 0) throw new IOException("The selected file has no video track");
            extractor.selectTrack(track);
            MediaFormat format = extractor.getTrackFormat(track);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime == null) throw new IOException("Video track has no MIME type");
            extractor.seekTo(Math.max(0, Math.round(requestedTimes[0] * 1_000_000)),
                    MediaExtractor.SEEK_TO_PREVIOUS_SYNC);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            format.setInteger(MediaFormat.KEY_OPERATING_RATE, 240);
            format.setInteger(MediaFormat.KEY_PRIORITY, 1);
            codec = MediaCodec.createDecoderByType(mime);
            codec.configure(format, null, null, 0);
            codec.start();
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            boolean inputEnded = false;
            boolean outputEnded = false;
            int target = 0;
            byte[] fallbackGray = null;
            while (!outputEnded && target < requestedTimes.length) {
                if (cancelled.getAsBoolean()) throw new IOException("Analysis cancelled");
                if (!inputEnded) {
                    int inputIndex = codec.dequeueInputBuffer(TIMEOUT_US);
                    if (inputIndex >= 0) {
                        ByteBuffer buffer = codec.getInputBuffer(inputIndex);
                        if (buffer == null) throw new IOException("Video decoder returned no input buffer");
                        int size = extractor.readSampleData(buffer, 0);
                        if (size < 0) {
                            codec.queueInputBuffer(inputIndex, 0, 0, 0,
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            inputEnded = true;
                        } else {
                            codec.queueInputBuffer(inputIndex, 0, size, extractor.getSampleTime(), 0);
                            extractor.advance();
                        }
                    }
                }
                int outputIndex = codec.dequeueOutputBuffer(info, TIMEOUT_US);
                if (outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER) continue;
                if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED
                        || outputIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED) continue;
                if (outputIndex < 0) continue;
                try {
                    boolean finalOutput = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                    // Match browser frame lookup at the end of a recording: use
                    // the last decoded image when no later timestamp exists.
                    long targetUs = Math.round(requestedTimes[target] * 1_000_000);
                    boolean atOrAfterTarget = info.presentationTimeUs + 1_000 >= targetUs;
                    boolean nearEndFallback = info.presentationTimeUs + 1_000_000 >= targetUs;
                    if (info.size > 0 && (finalOutput || atOrAfterTarget || nearEndFallback)) {
                        Image image = codec.getOutputImage(outputIndex);
                        if (image == null || image.getFormat() != ImageFormat.YUV_420_888) {
                            if (image != null) image.close();
                            throw new IOException("The device decoder did not expose YUV_420_888 output");
                        }
                        byte[] gray;
                        try (image) {
                            Mat rgba = NativeVideoDecoder.imageToAnalysisRgba(image, roi, media.rotation());
                            Mat grayMat = new Mat(
                                    ServingSideModelsKt.SERVING_SIDE_HEIGHT,
                                    ServingSideModelsKt.SERVING_SIDE_WIDTH,
                                    CvType.CV_8UC1
                            );
                            try {
                                Imgproc.cvtColor(rgba, grayMat, Imgproc.COLOR_RGBA2GRAY);
                                gray = new byte[ServingSideModelsKt.SERVING_SIDE_WIDTH
                                        * ServingSideModelsKt.SERVING_SIDE_HEIGHT];
                                grayMat.get(0, 0, gray);
                            } finally {
                                rgba.release();
                                grayMat.release();
                            }
                        }
                        if (finalOutput || atOrAfterTarget) {
                            while (target < requestedTimes.length && (finalOutput ||
                                    info.presentationTimeUs + 1_000 >=
                                            Math.round(requestedTimes[target] * 1_000_000))) {
                                result.put(requestedTimes[target], gray.clone());
                                target++;
                            }
                            fallbackGray = null;
                        } else {
                            fallbackGray = gray;
                        }
                        progress.onProgress(
                                "serving-side-frames",
                                target / (double) requestedTimes.length,
                                "Sampling serving-side windows · " + target + "/" + requestedTimes.length + " frames"
                        );
                    }
                } finally {
                    codec.releaseOutputBuffer(outputIndex, false);
                }
                outputEnded = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
            }
            if (target < requestedTimes.length && fallbackGray != null) {
                while (target < requestedTimes.length) {
                    result.put(requestedTimes[target], fallbackGray.clone());
                    target++;
                }
            }
            if (target != requestedTimes.length) {
                throw new IOException("Serving-side frame sampling ended early ("
                        + target + "/" + requestedTimes.length + ")");
            }
            return result;
        } finally {
            if (codec != null) {
                try { codec.stop(); } catch (RuntimeException ignored) {}
                codec.release();
            }
            extractor.release();
        }
    }
}
