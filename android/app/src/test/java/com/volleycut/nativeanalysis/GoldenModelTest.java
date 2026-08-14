package com.volleycut.nativeanalysis;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPInputStream;

public final class GoldenModelTest {
    @Test
    public void canonicalFeaturesReproduceAllThirtySevenRanges() throws Exception {
        Path repository = findRepositoryRoot();
        JSONObject metadata = new JSONObject(new String(Files.readAllBytes(
                repository.resolve("tests/fixtures/on-device-y9-golden.json")
        ), StandardCharsets.UTF_8));
        int rows = metadata.getInt("rows");
        int columns = metadata.getInt("columns");
        assertEquals(3474, rows);
        assertEquals(104, columns);
        assertEquals(FeatureSchema.BASE, jsonStrings(metadata.getJSONArray("names")));

        byte[] inflated;
        try (GZIPInputStream gzip = new GZIPInputStream(Files.newInputStream(
                repository.resolve("tests/fixtures/on-device-y9-base-features.bin.gz")
        )); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            gzip.transferTo(output);
            inflated = output.toByteArray();
        }
        ByteBuffer buffer = ByteBuffer.wrap(inflated).order(ByteOrder.LITTLE_ENDIAN);
        double[] times = new double[rows];
        for (int row = 0; row < rows; row++) times[row] = buffer.getDouble();
        float[] base = new float[rows * columns];
        for (int index = 0; index < base.length; index++) base[index] = buffer.getFloat();

        float[] contextual = FeatureMath.contextualize(times, base, FeatureSchema.BASE);
        JSONObject model = new JSONObject(new String(Files.readAllBytes(
                repository.resolve("public/on-device/model-9c92b8e9333f.json")
        ), StandardCharsets.UTF_8));
        List<AnalysisTypes.Interval> actual = new ModelRunner(model).run(
                times, contextual, metadata.getDouble("duration")
        );
        JSONArray expected = metadata.getJSONArray("rallies");
        assertEquals(37, actual.size());
        assertEquals(expected.length(), actual.size());
        for (int index = 0; index < actual.size(); index++) {
            JSONObject range = expected.getJSONObject(index);
            assertEquals(range.getDouble("start"), actual.get(index).start(), 0.00051);
            assertEquals(range.getDouble("end"), actual.get(index).end(), 0.00051);
            assertEquals(range.getDouble("confidence"), actual.get(index).confidence(), 2e-5);
        }
    }

    @Test
    public void fftImpulseHasFlatMagnitudeSpectrum() {
        float[] impulse = new float[1024];
        impulse[0] = 1;
        float[] magnitude = new Radix2Fft(1024).magnitudes(impulse);
        assertEquals(513, magnitude.length);
        for (float value : magnitude) assertEquals(1, value, 1e-6);
    }

    private static List<String> jsonStrings(JSONArray array) throws Exception {
        java.util.ArrayList<String> values = new java.util.ArrayList<>();
        for (int i = 0; i < array.length(); i++) values.add(array.getString(i));
        return values;
    }

    private static Path findRepositoryRoot() {
        Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        while (current != null) {
            if (Files.exists(current.resolve("public/on-device/model-9c92b8e9333f.json"))) return current;
            current = current.getParent();
        }
        throw new IllegalStateException("Could not locate the VolleyCut repository root");
    }
}
