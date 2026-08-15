package com.volleycut.nativeanalysis;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class AudioFeatureExtractor {
    static final int TARGET_SAMPLE_RATE = 16_000;
    private static final int FRAME_SAMPLES = 800;
    private static final double FRAME_SECONDS = 0.05;
    private static final int FFT_SIZE = 1024;
    private static final int[][] BAND_BOUNDS = {
            {80, 250}, {250, 500}, {500, 1000},
            {1000, 2000}, {2000, 4000}, {4000, 7800}
    };
    private static final String[] BAND_LABELS = {
            "80_250", "250_500", "500_1000", "1000_2000", "2000_4000", "4000_7800"
    };

    private final Radix2Fft fft = new Radix2Fft(FFT_SIZE);
    private final NanoProfiler profiler = new NanoProfiler();
    private final float[] window = new float[FRAME_SAMPLES];
    private final int[][] bandIndexes = new int[BAND_BOUNDS.length][];
    private final ArrayList<Float> rms = new ArrayList<>();
    private final ArrayList<Float> peak = new ArrayList<>();
    private final ArrayList<Float> spectralFlux = new ArrayList<>();
    private final List<ArrayList<Float>> bandPower = new ArrayList<>();
    private float[] previousSpectrum;
    private int sourceRate;
    private long nextSourceFrame;
    private float[] sourceBuffer = new float[0];
    private double sourcePosition;
    private final float[] outputFrame = new float[FRAME_SAMPLES];
    private int outputFrameLength;
    private long resampledOutputSamples;

    AudioFeatureExtractor() {
        for (int i = 0; i < FRAME_SAMPLES; i++) {
            window[i] = (float) (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FRAME_SAMPLES - 1)));
        }
        for (int band = 0; band < BAND_BOUNDS.length; band++) {
            ArrayList<Integer> indexes = new ArrayList<>();
            for (int bin = 0; bin <= FFT_SIZE / 2; bin++) {
                double frequency = bin * TARGET_SAMPLE_RATE / (double) FFT_SIZE;
                if (frequency >= BAND_BOUNDS[band][0] && frequency < BAND_BOUNDS[band][1]) indexes.add(bin);
            }
            bandIndexes[band] = indexes.stream().mapToInt(Integer::intValue).toArray();
            bandPower.add(new ArrayList<>());
        }
    }

    void push(float[] mono, double timestampSeconds, int sampleRate) {
        long started = System.nanoTime();
        try {
        if (sourceRate != 0 && sourceRate != sampleRate) {
            throw new IllegalArgumentException("Audio sample rate changed within the source track");
        }
        sourceRate = sampleRate;
        long startFrame = Math.round(timestampSeconds * sampleRate);
        int primingFrames = (int) Math.min(mono.length, Math.max(0, -startFrame));
        int presentedFrames = mono.length - primingFrames;
        if (presentedFrames == 0) return;
        long presentedStart = startFrame + primingFrames;
        long gapFrames = Math.max(0, presentedStart - nextSourceFrame);
        int overlapFrames = (int) Math.min(presentedFrames, Math.max(0, nextSourceFrame - presentedStart));
        int trimFrames = primingFrames + overlapFrames;
        if (gapFrames > 0) appendSilence(gapFrames);
        if (trimFrames < mono.length) appendSource(Arrays.copyOfRange(mono, trimFrames, mono.length));
        nextSourceFrame = Math.max(nextSourceFrame, presentedStart + presentedFrames);
        } finally {
            profiler.add("timeline_and_resample", System.nanoTime() - started);
        }
    }

    float[] finishAndPool(double[] analysisTimes) {
        long finishStarted = System.nanoTime();
        if (sourceBuffer.length > 0) {
            while (sourcePosition < sourceBuffer.length) {
                pushOutput(sourceBuffer[Math.min(sourceBuffer.length - 1, (int) Math.floor(sourcePosition))]);
                sourcePosition += sourceRate / (double) TARGET_SAMPLE_RATE;
            }
        }
        if (outputFrameLength > 0) {
            Arrays.fill(outputFrame, outputFrameLength, FRAME_SAMPLES, 0);
            processFrame(outputFrame);
            outputFrameLength = 0;
        }
        float[] output = new float[analysisTimes.length * FeatureSchema.AUDIO.size()];
        if (rms.isEmpty()) {
            profiler.add("finish_and_pool", System.nanoTime() - finishStarted);
            return output;
        }
        long operationStarted = System.nanoTime();
        Map<String, float[]> sources = buildFrameFeatureSources();
        profiler.add("whole_recording_audio_reductions", System.nanoTime() - operationStarted);
        double[] audioTimes = new double[rms.size()];
        for (int i = 0; i < audioTimes.length; i++) audioTimes[i] = (i + 0.5) * FRAME_SECONDS;
        double halfWidth = 0.5 / FeatureSchema.ANALYSIS_FPS;
        List<String> meanPooled = List.of(
                "audio_rms", "audio_noise_floor", "audio_snr", "audio_onset_cadence",
                "audio_cadence_collapse", "audio_seconds_since_transient"
        );
        operationStarted = System.nanoTime();
        for (int row = 0; row < analysisTimes.length; row++) {
            output[row * FeatureSchema.AUDIO.size()] = 1;
            int left = FeatureMath.lowerBound(audioTimes, analysisTimes[row] - halfWidth);
            int right = FeatureMath.upperBound(audioTimes, analysisTimes[row] + halfWidth);
            if (right <= left) {
                int nearest = Math.min(audioTimes.length - 1,
                        Math.max(0, (int) Math.round(analysisTimes[row] / FRAME_SECONDS - 0.5)));
                left = nearest;
                right = nearest + 1;
            }
            for (int column = 1; column < FeatureSchema.AUDIO.size(); column++) {
                String name = FeatureSchema.AUDIO.get(column);
                float[] source = sources.get(name);
                float pooled = meanPooled.contains(name)
                        ? (float) FeatureMath.mean(source, left, right)
                        : max(source, left, right);
                output[row * FeatureSchema.AUDIO.size() + column] = pooled;
            }
        }
        profiler.add("pool_to_analysis_timestamps", System.nanoTime() - operationStarted);
        profiler.add("finish_and_pool", System.nanoTime() - finishStarted);
        return output;
    }

    private void appendSource(float[] mono) {
        float[] combined = Arrays.copyOf(sourceBuffer, sourceBuffer.length + mono.length);
        System.arraycopy(mono, 0, combined, sourceBuffer.length, mono.length);
        sourceBuffer = combined;
        double step = sourceRate / (double) TARGET_SAMPLE_RATE;
        while (sourcePosition + 1 < sourceBuffer.length) {
            int lower = (int) Math.floor(sourcePosition);
            double fraction = sourcePosition - lower;
            double value = sourceBuffer[lower] * (1 - fraction) + sourceBuffer[lower + 1] * fraction;
            int quantized = (int) Math.round(value * 32768);
            pushOutput(Math.max(-32768, Math.min(32767, quantized)) / 32768f);
            sourcePosition += step;
        }
        int consumed = Math.min(sourceBuffer.length, (int) Math.floor(sourcePosition));
        if (consumed > 0) {
            sourceBuffer = Arrays.copyOfRange(sourceBuffer, consumed, sourceBuffer.length);
            sourcePosition -= consumed;
        }
    }

    private void appendSilence(long frameCount) {
        int blockSize = (int) Math.min(frameCount, Math.max(1, sourceRate));
        float[] silence = new float[blockSize];
        long remaining = frameCount;
        while (remaining > 0) {
            int length = (int) Math.min(remaining, blockSize);
            appendSource(length == blockSize ? silence : new float[length]);
            remaining -= length;
        }
    }

    private void pushOutput(float value) {
        resampledOutputSamples++;
        outputFrame[outputFrameLength++] = value;
        if (outputFrameLength == FRAME_SAMPLES) {
            processFrame(outputFrame);
            outputFrameLength = 0;
        }
    }

    private void processFrame(float[] frame) {
        long started = System.nanoTime();
        double squareTotal = 0;
        float maximum = 0;
        float[] fftInput = new float[FFT_SIZE];
        for (int i = 0; i < FRAME_SAMPLES; i++) {
            float value = frame[i];
            squareTotal += value * value;
            maximum = Math.max(maximum, Math.abs(value));
            fftInput[i] = value * window[i];
        }
        rms.add((float) Math.sqrt(squareTotal / FRAME_SAMPLES));
        peak.add(maximum);
        float[] spectrum = fft.magnitudes(fftInput);
        double spectrumTotal = 0;
        for (float value : spectrum) spectrumTotal += value;
        for (int band = 0; band < bandIndexes.length; band++) {
            double power = 0;
            for (int bin : bandIndexes[band]) power += spectrum[bin] * spectrum[bin];
            bandPower.get(band).add((float) power);
        }
        double divisor = Math.max(spectrumTotal, 1e-8);
        for (int bin = 0; bin < spectrum.length; bin++) spectrum[bin] /= divisor;
        double fluxSquares = 0;
        if (previousSpectrum != null) {
            for (int bin = 0; bin < spectrum.length; bin++) {
                double difference = Math.max(spectrum[bin] - previousSpectrum[bin], 0);
                fluxSquares += difference * difference;
            }
        }
        spectralFlux.add((float) Math.sqrt(fluxSquares));
        previousSpectrum = spectrum;
        profiler.add("fft_and_frame_features", System.nanoTime() - started);
    }

    Map<String, Double> performanceMilliseconds() {
        return profiler.milliseconds();
    }

    int audioFeatureFrameCount() {
        return rms.size();
    }

    long resampledOutputSamples() {
        return resampledOutputSamples;
    }

    private Map<String, float[]> buildFrameFeatureSources() {
        float[] rmsValues = toArray(rms);
        float[] peakValues = toArray(peak);
        float[] fluxValuesRaw = toArray(spectralFlux);
        int count = rmsValues.length;
        float[] rmsNovelty = new float[count];
        for (int i = 1; i < count; i++) rmsNovelty[i] = Math.max(rmsValues[i] - rmsValues[i - 1], 0);
        float[] peakRank = FeatureMath.percentileRanks(peakValues, count, 1);
        float[] noveltyRank = FeatureMath.percentileRanks(rmsNovelty, count, 1);
        float[] fluxRank = FeatureMath.percentileRanks(fluxValuesRaw, count, 1);
        float[] onsetStrength = new float[count];
        float[] contactLike = new float[count];
        for (int i = 0; i < count; i++) {
            onsetStrength[i] = 0.45f * fluxRank[i] + 0.35f * noveltyRank[i] + 0.2f * peakRank[i];
            contactLike[i] = onsetStrength[i] * (float) Math.sqrt(peakRank[i]);
        }
        float strongThreshold = Math.max(0.72f, FeatureMath.quantile(contactLike, 0.85));
        float[] cadence = FeatureMath.rollingMean(contactLike, (int) Math.round(2 / FRAME_SECONDS), false);
        float[] futureCadence = FeatureMath.rollingMean(contactLike, (int) Math.round(1 / FRAME_SECONDS), true);
        float[] cadenceCollapse = new float[count];
        float[] elapsed = new float[count];
        Arrays.fill(elapsed, 10);
        Double lastTime = null;
        for (int i = 0; i < count; i++) {
            cadenceCollapse[i] = Math.max(cadence[i] - futureCadence[i], 0);
            double timestamp = (i + 0.5) * FRAME_SECONDS;
            if (contactLike[i] >= strongThreshold && peakRank[i] >= 0.6) {
                lastTime = timestamp;
                elapsed[i] = 0;
            } else if (lastTime != null) elapsed[i] = (float) Math.min(10, timestamp - lastTime);
        }
        int noiseWindow = (int) Math.round(10 / FRAME_SECONDS);
        float[] noiseFloor = rollingPercentile(rmsValues, noiseWindow, 0.2);
        float[] snr = new float[count];
        float[] peakToRms = new float[count];
        for (int i = 0; i < count; i++) {
            snr[i] = (float) Math.log1p(Math.max(rmsValues[i] - noiseFloor[i], 0) / (noiseFloor[i] + 1e-4));
            peakToRms[i] = Math.min(30, Math.max(0, peakValues[i] / (rmsValues[i] + 1e-5f)));
        }
        Map<String, float[]> sources = new HashMap<>();
        sources.put("audio_rms", rmsValues);
        sources.put("audio_peak", peakValues);
        sources.put("audio_peak_to_rms", peakToRms);
        sources.put("audio_noise_floor", noiseFloor);
        sources.put("audio_snr", snr);
        sources.put("audio_spectral_flux", fluxValuesRaw);
        sources.put("audio_rms_novelty", rmsNovelty);
        sources.put("audio_onset_strength", onsetStrength);
        sources.put("audio_contact_like_transient", contactLike);
        sources.put("audio_onset_cadence", cadence);
        sources.put("audio_cadence_collapse", cadenceCollapse);
        sources.put("audio_seconds_since_transient", elapsed);

        float[] broadbandNumerator = new float[count];
        float[] broadbandDenominator = new float[count];
        float[] normalizedFluxSquares = new float[count];
        for (int band = 0; band < bandPower.size(); band++) {
            float[] power = toArray(bandPower.get(band));
            float[] floor = rollingPercentile(power, noiseWindow, 0.2);
            float[] snrValues = new float[count];
            float[] bandFlux = new float[count];
            for (int i = 0; i < count; i++) {
                float excess = Math.max(power[i] - floor[i], 0);
                snrValues[i] = (float) Math.log1p(excess / (floor[i] + 1e-8));
                bandFlux[i] = Math.max(snrValues[i] - snrValues[Math.max(0, i - 1)], 0);
                broadbandNumerator[i] += excess;
                broadbandDenominator[i] += floor[i];
                normalizedFluxSquares[i] += bandFlux[i] * bandFlux[i];
            }
            sources.put("audio_band_" + BAND_LABELS[band] + "_snr", snrValues);
            sources.put("audio_band_" + BAND_LABELS[band] + "_snr_flux", bandFlux);
        }
        float[] broadband = new float[count];
        float[] normalizedFlux = new float[count];
        for (int i = 0; i < count; i++) {
            broadband[i] = (float) Math.log1p(broadbandNumerator[i] / (broadbandDenominator[i] + 1e-8));
            normalizedFlux[i] = (float) Math.sqrt(normalizedFluxSquares[i]);
        }
        sources.put("audio_noise_removed_broadband", broadband);
        sources.put("audio_noise_normalized_flux", normalizedFlux);
        return sources;
    }

    private static float[] rollingPercentile(float[] values, int window, double percentile) {
        float[] output = new float[values.length];
        ArrayList<Float> sorted = new ArrayList<>();
        for (int i = 0; i < values.length; i++) {
            int insertion = Collections.binarySearch(sorted, values[i]);
            if (insertion < 0) insertion = -insertion - 1;
            sorted.add(insertion, values[i]);
            if (i >= window) {
                int removal = Collections.binarySearch(sorted, values[i - window]);
                sorted.remove(removal);
            }
            double position = (sorted.size() - 1) * percentile;
            int lower = (int) Math.floor(position);
            int upper = (int) Math.ceil(position);
            output[i] = (float) (sorted.get(lower) * (upper - position) + sorted.get(upper) * (position - lower));
        }
        return output;
    }

    private static float[] toArray(List<Float> values) {
        float[] output = new float[values.size()];
        for (int i = 0; i < values.size(); i++) output[i] = values.get(i);
        return output;
    }

    private static float max(float[] values, int start, int end) {
        float result = -Float.MAX_VALUE;
        for (int i = start; i < end; i++) result = Math.max(result, values[i]);
        return result;
    }
}
