package com.volleycut.neuralbenchmark;

import android.app.Activity;
import android.os.Bundle;
import android.os.Build;
import android.os.Debug;
import android.os.PowerManager;
import android.os.SystemClock;
import android.view.WindowManager;
import android.widget.TextView;
import ai.onnxruntime.*;
import ai.onnxruntime.providers.NNAPIFlags;
import org.json.*;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/** Tensor-stage benchmark. Decode/preprocessing timings must be measured separately. */
public final class BenchmarkActivity extends Activity {
    private TextView status;
    private File root;
    private PowerManager power;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setSustainedPerformanceMode(true);
        status = new TextView(this);
        status.setPadding(24, 48, 24, 24);
        setContentView(status);
        root = new File(getFilesDir(), "benchmark");
        root.mkdirs();
        power = getSystemService(PowerManager.class);
        String plan = getIntent().getStringExtra("plan");
        if (plan == null) { status.setText("Ready. Supply a private benchmark plan using ADB."); return; }
        new Thread(() -> execute(plan), "neural-benchmark").start();
    }

    private File local(String name) throws IOException {
        File file = new File(root, name).getCanonicalFile();
        if (!file.toPath().startsWith(root.getCanonicalFile().toPath()))
            throw new IOException("Benchmark path escapes private directory");
        return file;
    }
    private void save(JSONObject report) throws Exception {
        File temp = local("result.tmp");
        Files.writeString(temp.toPath(), report.toString(2));
        Files.move(temp.toPath(), local("result.json").toPath(), StandardCopyOption.REPLACE_EXISTING);
    }
    private void execute(String planName) {
        JSONObject report = new JSONObject();
        try {
            JSONObject plan = new JSONObject(Files.readString(local(planName).toPath()));
            report.put("runId", plan.getString("runId"));
            report.put("status", "running");
            report.put("device", Build.MODEL);
            report.put("soc", Build.SOC_MODEL);
            report.put("android", Build.VERSION.RELEASE);
            report.put("runtime", OrtEnvironment.getEnvironment().getVersion());
            report.put("availableProviders", OrtEnvironment.getAvailableProviders().toString());
            report.put("scope", "tensor inference only; excludes video decode, image preprocessing and AV generation");
            JSONArray rows = new JSONArray(); report.put("results", rows); save(report);
            JSONArray cases = plan.getJSONArray("cases");
            for (int i = 0; i < cases.length(); i++) {
                JSONObject spec = cases.getJSONObject(i);
                String id = spec.getString("id");
                runOnUiThread(() -> status.setText("Running " + id));
                JSONObject row;
                try { row = spec.optString("runtime").equals("litert") ? LiteRtBenchmark.measure(root, spec, power) : measure(spec); }
                catch (Exception | LinkageError error) {
                    row = new JSONObject().put("id", id).put("status", "failed").put("error", error.toString());
                }
                rows.put(row); save(report);
            }
            report.put("status", "complete"); save(report);
            runOnUiThread(() -> status.setText("Complete. Results saved in app-private benchmark/result.json."));
        } catch (Exception error) {
            try { report.put("status", "failed").put("error", error.toString()); save(report); } catch (Exception ignored) {}
            runOnUiThread(() -> status.setText("Failed: " + error));
        }
    }

    private JSONObject measure(JSONObject spec) throws Exception {
        JSONObject row = new JSONObject().put("id", spec.getString("id"));
        String provider = spec.optString("provider", "cpu");
        if (!Set.of("cpu", "nnapi", "nnapi-fp16", "webgpu").contains(provider)) throw new IllegalArgumentException("Unknown provider");
        row.put("requestedProvider", provider);
        row.put("thermalStart", power.getCurrentThermalStatus());
        row.put("modelBytes", local(spec.getString("model")).length());
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(spec.optInt("threads", 4));
            options.setInterOpNumThreads(1);
            if (provider.equals("webgpu")) options.addWebGPU(Collections.emptyMap());
            if (provider.startsWith("nnapi")) {
                EnumSet<NNAPIFlags> flags = EnumSet.of(NNAPIFlags.CPU_DISABLED);
                if (provider.equals("nnapi-fp16")) flags.add(NNAPIFlags.USE_FP16);
                options.addNnapi(flags);
            }
            // Profiling is an explicit separate pass, never part of the timing runs.
            if (spec.optBoolean("profile", false)) options.enableProfiling(local(spec.getString("id") + "-profile").toString());
            long start = SystemClock.elapsedRealtimeNanos();
            try (OrtSession session = env.createSession(local(spec.getString("model")).toString(), options)) {
                row.put("loadMs", elapsed(start));
                if (spec.has("video") || spec.has("videoUri")) {
                    JSONObject video = VideoEncoderBenchmark.run(this, session, env, local(spec.optString("video", "unused")),
                        root, spec);
                    row.put("video", video).put("status", "complete");
                    if (spec.optBoolean("profile", false)) row.put("profile", session.endProfiling());
                    row.put("thermalEnd", power.getCurrentThermalStatus());
                    return row;
                }
                JSONArray descriptions = spec.getJSONArray("inputs");
                start = SystemClock.elapsedRealtimeNanos();
                for (int i = 0; i < descriptions.length(); i++) {
                    JSONObject input = descriptions.getJSONObject(i);
                    JSONArray shape = input.getJSONArray("shape");
                    long[] dims = new long[shape.length()];
                    long count = 1;
                    for (int j = 0; j < dims.length; j++) { dims[j] = shape.getLong(j); count = Math.multiplyExact(count, dims[j]); }
                    byte[] bytes = Files.readAllBytes(local(input.getString("file")).toPath());
                    String type = input.optString("dtype", "float32");
                    OnnxJavaType dtype = switch (type) {
                        case "float32" -> OnnxJavaType.FLOAT;
                        case "float16" -> OnnxJavaType.FLOAT16;
                        case "int64" -> OnnxJavaType.INT64;
                        case "bool" -> OnnxJavaType.BOOL;
                        default -> throw new IllegalArgumentException("Unsupported input dtype: " + type);
                    };
                    int size = type.equals("float32") ? 4 : type.equals("float16") ? 2 : type.equals("int64") ? 8 : 1;
                    if (bytes.length != count * size) throw new IOException("Input size mismatch");
                    ByteBuffer buffer = ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.LITTLE_ENDIAN);
                    buffer.put(bytes).rewind();
                    inputs.put(input.getString("name"), OnnxTensor.createTensor(env, buffer, dims, dtype));
                }
                row.put("inputLoadMs", elapsed(start));
                int warmup = spec.optInt("warmup", 3), runs = spec.optInt("runs", 10);
                if (warmup < 0 || runs < 1 || runs > 1000) throw new IllegalArgumentException("Invalid run count");
                JSONArray times = new JSONArray();
                for (int i = -warmup; i < runs; i++) {
                    start = SystemClock.elapsedRealtimeNanos();
                    try (OrtSession.Result result = session.run(inputs)) {
                        // Force output materialization; asynchronous work must finish before timing ends.
                        for (Map.Entry<String, OnnxValue> output : result) output.getValue().getValue();
                        if (i >= 0) times.put(elapsed(start));
                        if (i == runs - 1) {
                            JSONArray outputs = new JSONArray();
                            for (Map.Entry<String, OnnxValue> output : result) {
                                if (output.getValue() instanceof OnnxTensor tensor && tensor.getInfo().type == OnnxJavaType.FLOAT) {
                                    FloatBuffer values = tensor.getFloatBuffer();
                                    ByteBuffer data = ByteBuffer.allocate(values.remaining() * 4).order(ByteOrder.LITTLE_ENDIAN);
                                    data.asFloatBuffer().put(values);
                                    String filename = spec.getString("id") + "-output-" + outputs.length() + ".f32";
                                    Files.write(local(filename).toPath(), data.array());
                                    outputs.put(new JSONObject().put("name", output.getKey()).put("file", filename));
                                }
                            }
                            row.put("outputs", outputs);
                        }
                    }
                }
                row.put("samplesMs", times);
                row.put("status", "complete");
                if (spec.optBoolean("profile", false)) row.put("profile", session.endProfiling());
            }
        } finally { for (OnnxTensor tensor : inputs.values()) tensor.close(); }
        row.put("thermalEnd", power.getCurrentThermalStatus());
        Debug.MemoryInfo memory = new Debug.MemoryInfo(); Debug.getMemoryInfo(memory);
        row.put("pssKbAfter", memory.getTotalPss());
        return row;
    }
    private static double elapsed(long start) { return (SystemClock.elapsedRealtimeNanos() - start) / 1e6; }
}
