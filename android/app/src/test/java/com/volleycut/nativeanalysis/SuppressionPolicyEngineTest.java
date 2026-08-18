package com.volleycut.nativeanalysis;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;

public final class SuppressionPolicyEngineTest {
    @Test
    public void sharedGoldenFixtureReproducesEveryAgreementPolicy() throws Exception {
        JSONObject fixture = json(repositoryRoot().resolve(
                "tests/fixtures/suppression-policy-golden.json"));
        JSONObject components = fixture.getJSONObject("productionComponents");
        List<AnalysisTypes.Interval> allLabels = intervals(
                components.getJSONArray("allLabelsV2"));
        List<AnalysisTypes.Interval> previous = intervals(
                components.getJSONArray("previousProduction"));
        List<AnalysisTypes.Interval> decoded = intervals(
                fixture.getJSONArray("decodedSuppression"));
        AnalysisTypes.SuppressionAnalysis analysis = SuppressionPolicyEngine.build(
                allLabels, previous, new float[0], decoded,
                fixture.getDouble("duration")
        );

        JSONObject expected = fixture.getJSONObject("expectedPolicyIntervals");
        for (SuppressionPolicyEngine.Policy policy : List.of(
                SuppressionPolicyEngine.Policy.CONSERVATIVE,
                SuppressionPolicyEngine.Policy.BALANCED,
                SuppressionPolicyEngine.Policy.AGGRESSIVE
        )) {
            List<AnalysisTypes.SuppressionSuggestion> actual =
                    SuppressionPolicyEngine.active(analysis, policy);
            JSONArray ranges = expected.getJSONArray(policy.wireName);
            assertEquals(policy.wireName, ranges.length(), actual.size());
            for (int index = 0; index < ranges.length(); index++) {
                assertEquals(milliseconds(ranges.getJSONArray(index).getDouble(0)),
                        actual.get(index).startMs());
                assertEquals(milliseconds(ranges.getJSONArray(index).getDouble(1)),
                        actual.get(index).endMs());
            }
        }
        assertTrue(SuppressionPolicyEngine.active(
                analysis, SuppressionPolicyEngine.Policy.NONE).isEmpty());
    }

    @Test
    public void frozenAssetHashProbabilityAndHeldDecoderAreStable() throws Exception {
        Path asset = repositoryRoot().resolve(
                "android/app/src/main/assets/suppression-overlap-exclusion-retrained.json");
        byte[] bytes = Files.readAllBytes(asset);
        assertEquals(
                "02274d0f17b89cd54ea24da7d1665a6475e48dc0f554d06092e9ef892332d4f1",
                HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))
        );
        JSONObject json = new JSONObject(new String(bytes, StandardCharsets.UTF_8));
        JSONObject head = json.getJSONObject("head");
        JSONArray mean = head.getJSONArray("mean");
        JSONArray scale = head.getJSONArray("scale");
        JSONArray weights = head.getJSONArray("weights");
        int rows = 8;
        float[] contextual = new float[rows * mean.length()];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < mean.length(); column++) {
                double direction = Math.signum(weights.getDouble(column));
                contextual[row * mean.length() + column] = (float) (
                        mean.getDouble(column) + scale.getDouble(column) * direction * 10
                );
            }
        }
        double[] times = {.125, .375, .625, .875, 1.125, 1.375, 1.625, 1.875};
        SuppressionModelRunner.Result result = new SuppressionModelRunner(json)
                .run(times, contextual, 2.0);

        for (float probability : result.probabilities()) assertEquals(1f, probability, 1e-6f);
        assertEquals(1, result.decodedIntervals().size());
        assertEquals(0, result.decodedIntervals().get(0).start(), 0);
        assertEquals(2, result.decodedIntervals().get(0).end(), 0);
    }

    private static List<AnalysisTypes.Interval> intervals(JSONArray values)
            throws Exception {
        ArrayList<AnalysisTypes.Interval> result = new ArrayList<>();
        for (int index = 0; index < values.length(); index++) {
            JSONObject value = values.getJSONObject(index);
            result.add(new AnalysisTypes.Interval(
                    value.getDouble("start"),
                    value.getDouble("end"),
                    (float) value.getDouble("confidence")
            ));
        }
        return result;
    }

    private static long milliseconds(double seconds) {
        return Math.round(seconds * 1_000);
    }

    private static JSONObject json(Path path) throws Exception {
        return new JSONObject(Files.readString(path, StandardCharsets.UTF_8));
    }

    private static Path repositoryRoot() {
        Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        while (current != null) {
            if (Files.exists(current.resolve("tests/fixtures/suppression-policy-golden.json"))) {
                return current;
            }
            current = current.getParent();
        }
        throw new IllegalStateException("Could not locate repository root");
    }
}
