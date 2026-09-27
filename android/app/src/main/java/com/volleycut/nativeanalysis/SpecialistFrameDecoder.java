package com.volleycut.nativeanalysis;

import android.content.Context;
import android.graphics.ImageFormat;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;

import com.volleycut.video.DecodedVideoColor;
import com.volleycut.video.YuvColorConversion;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.BooleanSupplier;

/**
 * One gap-aware decoder for the serving-side and team-switch specialists. Each dense segment is
 * decoded sequentially from its preceding sync frame; long dead gaps start a fresh codec session.
 */
final class SpecialistFrameDecoder {
    private static final long TIMEOUT_US = 10_000;
    static final double MAX_SEQUENTIAL_GAP_SECONDS = 5.0;

    record Stats(
            int segments,
            int requestedFrames,
            long decoderOutputFrames,
            long convertedOutputFrames,
            double setupMilliseconds,
            double conversionMilliseconds,
            double wallMilliseconds
    ) {}
    record Samples(
            Map<Double, byte[]> servingGray,
            Map<Double, byte[]> sideSwitchBgr,
            Stats stats
    ) {}
    private record Request(double time, boolean serving, boolean sideSwitch) {}
    private record Converted(byte[] servingGray, byte[] sideSwitchBgr) {}
    private record SegmentResult(
            int completed,
            long decoderOutputFrames,
            long convertedOutputFrames,
            long setupNanos,
            long conversionNanos
    ) {}

    private final Context context;
    private final double maximumSequentialGapSeconds;

    SpecialistFrameDecoder(Context context) {
        this(context, MAX_SEQUENTIAL_GAP_SECONDS);
    }

    SpecialistFrameDecoder(Context context, double maximumSequentialGapSeconds) {
        this.context = context.getApplicationContext();
        if (!(maximumSequentialGapSeconds > 0)) {
            throw new IllegalArgumentException("Maximum sequential gap must be positive");
        }
        this.maximumSequentialGapSeconds = maximumSequentialGapSeconds;
    }

    Samples decode(
            Uri uri,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            double[] servingTimes,
            double[] sideSwitchTimes,
            AnalysisTypes.ProgressListener progress,
            BooleanSupplier cancelled
    ) throws IOException {
        validate(servingTimes, "Serving-side");
        validate(sideSwitchTimes, "Team-switch");
        TreeMap<Double, boolean[]> merged = new TreeMap<>();
        for (double time : servingTimes) merged.computeIfAbsent(time, ignored -> new boolean[2])[0] = true;
        for (double time : sideSwitchTimes) merged.computeIfAbsent(time, ignored -> new boolean[2])[1] = true;
        if (merged.isEmpty()) {
            return new Samples(Map.of(), Map.of(), new Stats(0, 0, 0, 0, 0, 0, 0));
        }
        ArrayList<Request> requests = new ArrayList<>(merged.size());
        merged.forEach((time, flags) -> requests.add(new Request(time, flags[0], flags[1])));
        List<List<Request>> segments = segments(requests, maximumSequentialGapSeconds);
        LinkedHashMap<Double, byte[]> serving = new LinkedHashMap<>();
        LinkedHashMap<Double, byte[]> switches = new LinkedHashMap<>();
        FrameConverter converter = new FrameConverter(media, roi);
        int completed = 0;
        long decodedOutputs = 0;
        long convertedOutputs = 0;
        long setupNanos = 0;
        long conversionNanos = 0;
        long wallNanos = 0;
        for (int segmentIndex = 0; segmentIndex < segments.size(); segmentIndex++) {
            if (cancelled.getAsBoolean()) throw new IOException("Analysis cancelled");
            List<Request> segment = segments.get(segmentIndex);
            long segmentStarted = System.nanoTime();
            SegmentResult result = decodeSegment(
                    uri, media, roi, segment, serving, switches,
                    converter, completed, requests.size(), segmentIndex, segments.size(), progress, cancelled
            );
            wallNanos += System.nanoTime() - segmentStarted;
            completed += result.completed;
            decodedOutputs += result.decoderOutputFrames;
            convertedOutputs += result.convertedOutputFrames;
            setupNanos += result.setupNanos;
            conversionNanos += result.conversionNanos;
        }
        if (serving.size() != servingTimes.length || switches.size() != sideSwitchTimes.length) {
            throw new IOException("Specialist frame sampling ended early");
        }
        return new Samples(
                Map.copyOf(serving),
                Map.copyOf(switches),
                new Stats(
                        segments.size(), requests.size(), decodedOutputs, convertedOutputs,
                        setupNanos / 1_000_000.0,
                        conversionNanos / 1_000_000.0,
                        wallNanos / 1_000_000.0
                )
        );
    }

    private static List<List<Request>> segments(List<Request> requests, double maximumGapSeconds) {
        ArrayList<List<Request>> result = new ArrayList<>();
        ArrayList<Request> current = new ArrayList<>();
        for (Request request : requests) {
            if (!current.isEmpty()
                    && request.time - current.get(current.size() - 1).time > maximumGapSeconds) {
                result.add(List.copyOf(current));
                current.clear();
            }
            current.add(request);
        }
        if (!current.isEmpty()) result.add(List.copyOf(current));
        return List.copyOf(result);
    }

    static int plannedSegmentCount(double[] sortedTimes) {
        validate(sortedTimes, "Specialist");
        if (sortedTimes.length == 0) return 0;
        int count = 1;
        for (int index = 1; index < sortedTimes.length; index++) {
            if (sortedTimes[index] - sortedTimes[index - 1] > MAX_SEQUENTIAL_GAP_SECONDS) count++;
        }
        return count;
    }

    static boolean shouldConvertOutput(
            long presentationUs,
            long targetUs,
            long durationUs,
            boolean finalOutput
    ) {
        boolean atOrAfterTarget = presentationUs + 1_000 >= targetUs;
        boolean nearDeclaredEnd = targetUs + 1_000_000 >= durationUs;
        boolean nearEndFallback = nearDeclaredEnd && presentationUs + 1_000_000 >= targetUs;
        return finalOutput || atOrAfterTarget || nearEndFallback;
    }

    static int terminalConversionEnd(int satisfiedEnd, int requestCount,
            long presentationUs, long lastTargetUs, long durationUs) {
        // EOS can arrive in a separate empty output buffer. Preserve the last
        // actual image for a request just beyond its PTS, even when this image
        // also satisfies the preceding request. Include both consumers' formats.
        boolean terminal = lastTargetUs + 1_000_000 >= durationUs
                && presentationUs + 1_000_000 >= lastTargetUs;
        return satisfiedEnd < requestCount && terminal ? requestCount : satisfiedEnd;
    }

    private SegmentResult decodeSegment(
            Uri uri,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            List<Request> requests,
            Map<Double, byte[]> serving,
            Map<Double, byte[]> switches,
            FrameConverter converter,
            int alreadyCompleted,
            int total,
            int segmentIndex,
            int segmentCount,
            AnalysisTypes.ProgressListener progress,
            BooleanSupplier cancelled
    ) throws IOException {
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;
        long setupStarted = System.nanoTime();
        long setupNanos = 0;
        long conversionNanos = 0;
        long decodedOutputs = 0;
        long convertedOutputs = 0;
        try {
            extractor.setDataSource(context, uri, null);
            int track = NativeVideoDecoder.findTrack(extractor, "video/");
            if (track < 0) throw new IOException("The selected file has no video track");
            extractor.selectTrack(track);
            MediaFormat format = extractor.getTrackFormat(track);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime == null) throw new IOException("Video track has no MIME type");
            extractor.seekTo(
                    Math.max(0, Math.round(requests.get(0).time * 1_000_000)),
                    MediaExtractor.SEEK_TO_PREVIOUS_SYNC
            );
            format.setInteger(
                    MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
            );
            format.setInteger(MediaFormat.KEY_OPERATING_RATE, 240);
            format.setInteger(MediaFormat.KEY_PRIORITY, 1);
            codec = MediaCodec.createDecoderByType(mime);
            codec.configure(format, null, null, 0);
            codec.start();
            setupNanos = System.nanoTime() - setupStarted;
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            boolean inputEnded = false;
            boolean outputEnded = false;
            int target = 0;
            Converted fallback = null;
            while (!outputEnded && target < requests.size()) {
                if (cancelled.getAsBoolean()) throw new IOException("Analysis cancelled");
                if (!inputEnded) {
                    int inputIndex = codec.dequeueInputBuffer(TIMEOUT_US);
                    if (inputIndex >= 0) {
                        ByteBuffer buffer = codec.getInputBuffer(inputIndex);
                        if (buffer == null) throw new IOException("Video decoder returned no input buffer");
                        int size = extractor.readSampleData(buffer, 0);
                        if (size < 0) {
                            codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
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
                    if (info.size > 0) decodedOutputs++;
                    long targetUs = Math.round(requests.get(target).time * 1_000_000);
                    boolean atOrAfterTarget = info.presentationTimeUs + 1_000 >= targetUs;
                    long durationUs = Math.round(media.durationSeconds() * 1_000_000);
                    if (info.size > 0 && shouldConvertOutput(
                            info.presentationTimeUs, targetUs, durationUs, finalOutput
                    )) {
                        int satisfiedEnd = target;
                        if (finalOutput || atOrAfterTarget) {
                            while (satisfiedEnd < requests.size()
                                    && (finalOutput || info.presentationTimeUs + 1_000 >= Math.round(
                                            requests.get(satisfiedEnd).time * 1_000_000))) {
                                satisfiedEnd++;
                            }
                        } else {
                            // A terminal fallback may be reused for every remaining request.
                            satisfiedEnd = requests.size();
                        }
                        boolean produceServing = false;
                        boolean produceSideSwitch = false;
                        int conversionEnd = terminalConversionEnd(
                                satisfiedEnd, requests.size(), info.presentationTimeUs,
                                Math.round(requests.get(requests.size() - 1).time * 1_000_000), durationUs
                        );
                        for (int index = target; index < conversionEnd; index++) {
                            produceServing |= requests.get(index).serving;
                            produceSideSwitch |= requests.get(index).sideSwitch;
                        }
                        Image image = codec.getOutputImage(outputIndex);
                        if (image == null || image.getFormat() != ImageFormat.YUV_420_888) {
                            if (image != null) image.close();
                            throw new IOException("The device decoder did not expose YUV_420_888 output");
                        }
                        Converted converted;
                        try (image) {
                            long conversionStarted = System.nanoTime();
                            converted = converter.convert(
                                    image, produceServing, produceSideSwitch,
                                    DecodedVideoColor.resolve(codec.getOutputFormat(outputIndex), format).conversion()
                            );
                            conversionNanos += System.nanoTime() - conversionStarted;
                            convertedOutputs++;
                        }
                        if (finalOutput || atOrAfterTarget) {
                            while (target < satisfiedEnd) {
                                put(requests.get(target), converted, serving, switches);
                                target++;
                            }
                            fallback = conversionEnd > satisfiedEnd ? converted : null;
                        } else {
                            fallback = converted;
                        }
                        int completed = alreadyCompleted + target;
                        progress.onProgress(
                                "specialist-frames",
                                completed / (double) total,
                                "Shared score-feature decode · " + completed + "/" + total
                                        + " frames · segment " + (segmentIndex + 1) + "/" + segmentCount
                        );
                    }
                } finally {
                    codec.releaseOutputBuffer(outputIndex, false);
                }
                outputEnded = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
            }
            if (target < requests.size() && fallback != null) {
                while (target < requests.size()) put(requests.get(target++), fallback, serving, switches);
            }
            if (target != requests.size()) {
                throw new IOException("Specialist frame segment ended early (" + target + "/" + requests.size()
                        + "; next=" + requests.get(target).time + "; duration=" + media.durationSeconds() + ")");
            }
            return new SegmentResult(
                    target, decodedOutputs, convertedOutputs, setupNanos, conversionNanos
            );
        } finally {
            if (codec != null) {
                try { codec.stop(); } catch (RuntimeException ignored) {}
                codec.release();
            }
            extractor.release();
        }
    }

    private static final class FrameConverter {
        private final AnalysisTypes.MediaInfo media;
        private final AnalysisTypes.Roi roi;
        private NativeVideoDecoder.YuvCropSampler servingSampler;
        private NativeVideoDecoder.YuvCropSampler switchSampler;

        FrameConverter(AnalysisTypes.MediaInfo media, AnalysisTypes.Roi roi) {
            this.media = media;
            this.roi = roi;
        }

        Converted convert(Image image, boolean produceServing, boolean produceSideSwitch,
                YuvColorConversion color) {
            Mat servingRgba = null;
            Mat servingGray = null;
            Mat switchRgba = null;
            Mat switchBgr = null;
            try {
                byte[] gray = null;
                byte[] bgr = null;
                if (produceServing) {
                    if (servingSampler == null || !servingSampler.matches(image)) {
                        servingSampler = new NativeVideoDecoder.YuvCropSampler(
                                image, roi, media.rotation(),
                                ServingSideModelsKt.SERVING_SIDE_WIDTH,
                                ServingSideModelsKt.SERVING_SIDE_HEIGHT
                        );
                    }
                    servingRgba = servingSampler.convert(image, color);
                    servingGray = new Mat(
                            ServingSideModelsKt.SERVING_SIDE_HEIGHT,
                            ServingSideModelsKt.SERVING_SIDE_WIDTH,
                            CvType.CV_8UC1
                    );
                    Imgproc.cvtColor(servingRgba, servingGray, Imgproc.COLOR_RGBA2GRAY);
                    gray = new byte[ServingSideModelsKt.SERVING_SIDE_WIDTH
                            * ServingSideModelsKt.SERVING_SIDE_HEIGHT];
                    servingGray.get(0, 0, gray);
                }
                if (produceSideSwitch) {
                    if (switchSampler == null || !switchSampler.matches(image)) {
                        switchSampler = new NativeVideoDecoder.YuvCropSampler(
                                image, roi, media.rotation(),
                                SideSwitchModelsKt.SIDE_SWITCH_WIDTH,
                                SideSwitchModelsKt.SIDE_SWITCH_HEIGHT
                        );
                    }
                    switchRgba = switchSampler.convert(image, color);
                    switchBgr = new Mat(
                            SideSwitchModelsKt.SIDE_SWITCH_HEIGHT,
                            SideSwitchModelsKt.SIDE_SWITCH_WIDTH,
                            CvType.CV_8UC3
                    );
                    Imgproc.cvtColor(switchRgba, switchBgr, Imgproc.COLOR_RGBA2BGR);
                    bgr = new byte[SideSwitchModelsKt.SIDE_SWITCH_WIDTH
                            * SideSwitchModelsKt.SIDE_SWITCH_HEIGHT * 3];
                    switchBgr.get(0, 0, bgr);
                }
                return new Converted(gray, bgr);
            } finally {
                if (servingRgba != null) servingRgba.release();
                if (servingGray != null) servingGray.release();
                if (switchRgba != null) switchRgba.release();
                if (switchBgr != null) switchBgr.release();
            }
        }
    }

    private static void put(
            Request request,
            Converted converted,
            Map<Double, byte[]> serving,
            Map<Double, byte[]> switches
    ) {
        if (request.serving) serving.put(request.time, converted.servingGray.clone());
        if (request.sideSwitch) switches.put(request.time, converted.sideSwitchBgr.clone());
    }

    private static void validate(double[] times, String label) {
        for (int index = 0; index < times.length; index++) {
            if (!Double.isFinite(times[index]) || times[index] < 0
                    || (index > 0 && times[index] <= times[index - 1])) {
                throw new IllegalArgumentException(label + " timestamps must be finite, unique, and sorted");
            }
        }
    }
}
