package com.volleycut.nativeanalysis;

import android.app.Activity;
import android.os.Bundle;
import android.os.PowerManager;
import android.view.WindowManager;
import android.widget.TextView;
import org.json.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** Debug-only, separately installed, fresh-input complete-pipeline benchmark. */
public final class PipelineBenchmarkActivity extends Activity {
    private File root;
    private TextView status;
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setSustainedPerformanceMode(true);
        status=new TextView(this); setContentView(status);
        root=new File(getFilesDir(),"benchmark");root.mkdirs();
        new Thread(this::run,"full-pipeline-benchmark").start();
    }
    private void save(JSONObject report) throws Exception {
        File temp=new File(root,"pipeline-result.tmp");
        Files.writeString(temp.toPath(),report.toString(2));
        Files.move(temp.toPath(),new File(root,"pipeline-result.json").toPath(),StandardCopyOption.REPLACE_EXISTING);
    }
    private void run() {
        JSONObject report=new JSONObject();
        try {
            String filename=getIntent().getStringExtra("plan");
            if(filename==null || !new File(filename).getName().equals(filename)) throw new IOException("Invalid plan");
            JSONObject plan=new JSONObject(Files.readString(new File(root,filename).toPath()));
            report.put("runId",plan.getString("runId")).put("status","running")
                .put("device",android.os.Build.MODEL).put("cacheMode","bypass")
                .put("scope","fresh video through selected rallies and both score specialists; no export rendering");
            JSONArray rows=new JSONArray();report.put("results",rows);save(report);
            JSONArray cases=plan.getJSONArray("cases");
            for(int i=0;i<cases.length();i++) {
                JSONObject spec=cases.getJSONObject(i), row=new JSONObject().put("id",spec.getString("id"));
                rows.put(row);row.put("status","running");save(report);
                try {
                    if(spec.getString("family").endsWith("-terminal")) {
                        boolean mobile=spec.getString("family").startsWith("mobile");
                        AnalysisTypes.MediaInfo media=new AnalysisEngine(this).probe(getIntent().getData());
                        double end=media.durationSeconds(),start=Math.max(0,Math.floor(end)-1);
                        ai.onnxruntime.OrtEnvironment env=ai.onnxruntime.OrtEnvironment.getEnvironment();
                        JSONObject video=new JSONObject().put("id",spec.getString("id"))
                            .put("videoUri",getIntent().getData().toString()).put("startSeconds",start).put("seconds",end)
                            .put("imageSize",mobile?224:336).put("sampleFps",mobile?2:4)
                            .put("roi",new JSONArray(new double[]{.03,.12,.94,.86}));
                        if(mobile)video.put("poolWeights","mobile-encoder-pool_weights.f32");
                        try(ai.onnxruntime.OrtSession.SessionOptions options=new ai.onnxruntime.OrtSession.SessionOptions()) {
                            options.setIntraOpNumThreads(4);
                            try(ai.onnxruntime.OrtSession encoder=env.createSession(new File(root,(mobile?"mobile":"dino")+"-encoder-fp32.onnx").toString(),options)) {
                                JSONObject measured=com.volleycut.neuralbenchmark.VideoEncoderBenchmark.run(this,encoder,env,new File(root,"unused"),root,video);
                                if(measured.getInt("sampleCount")!=(int)Math.ceil((end-start)*(mobile?2:4)))throw new IOException("Terminal feature grid incomplete");
                                row.put("video",measured).put("status","complete");
                            }
                        }
                        save(report);continue;
                    }
                    if(spec.getString("family").equals("specialist-smoke")) {
                        AnalysisTypes.MediaInfo media=new AnalysisEngine(this).probe(getIntent().getData());
                        double end=media.durationSeconds();
                        double[] serve={end-.017},side={end-.01};
                        SpecialistFrameDecoder.Samples samples=new SpecialistFrameDecoder(this).decode(
                            getIntent().getData(),media,new AnalysisTypes.Roi(.03,.12,.94,.86,"test"),serve,side,
                            (stage,fraction,detail)->{},()->false);
                        if(samples.servingGray().get(serve[0])==null || samples.sideSwitchBgr().get(side[0])==null)
                            throw new IOException("Terminal specialist format missing");
                        row.put("status","complete").put("servingFrames",samples.servingGray().size())
                            .put("sideSwitchFrames",samples.sideSwitchBgr().size()).put("duration",end)
                            .put("requestedTimes",new JSONArray(new double[]{serve[0],side[0]}))
                            .put("decoderOutputs",samples.stats().decoderOutputFrames());
                        save(report);continue;
                    }
                    if(spec.getString("family").equals("tensor-io-smoke")) {
                        java.nio.FloatBuffer tokens=NeuralRallyPipeline.readFloats(new File(root,"dino-fp32-full-run0-tokens.f32"));
                        if(tokens.remaining()!=4245*3840)throw new IOException("Full DINO token capture required");
                        float[] fused=new float[4245*3944];
                        for(int tick=0;tick<4245;tick++)for(int c=0;c<3840;c++)fused[tick*3944+104+c]=tokens.get(tick*3840+c);
                        File scratch=new File(root,"tensor-io-smoke.f32");
                        NeuralRallyPipeline.writeFloats(scratch,fused);
                        java.nio.FloatBuffer replay=NeuralRallyPipeline.readFloats(scratch);
                        for(int valueIndex=0;valueIndex<fused.length;valueIndex++)if(Float.floatToIntBits(fused[valueIndex])!=Float.floatToIntBits(replay.get(valueIndex)))throw new IOException("Float tensor roundtrip differs");
                        row.put("status","complete").put("floatCount",fused.length).put("roundtripExact",true)
                            .put("scope","Full-size token mapping and fused feature write; no inference timing");
                        save(report);continue;
                    }
                    NeuralRallyPipeline neural=spec.getString("family").equals("production")?null:new NeuralRallyPipeline(this,root,spec);
                    AnalysisTypes.AnalysisResult result=new AnalysisEngine(this,neural).analyze(
                        getIntent().getData(),spec.optBoolean("fullFrame",false),Integer.MAX_VALUE,new AnalysisTypes.VideoDecoderOptions(240,1),
                        NativeFeatureCache.Mode.BYPASS,new AnalysisTypes.AnalysisWindow(0,spec.getDouble("seconds")),
                        AnalysisTypes.AnalysisStages.all(),AnalysisTypes.AudioDecoderMode.AUTO,new AtomicBoolean(),
                        new AnalysisTypes.ProgressListener() {
                            public void onProgress(String stage,double fraction,String detail) {runOnUiThread(()->status.setText(spec.optString("id")+"\n"+stage+"\n"+detail));}
                            public void onPerformance(AnalysisTypes.PerformanceStats stats) {}
                        },true,true);
                    JSONObject stages=new JSONObject(result.stageMilliseconds());
                    row.put("stagesMs",stages).put("profileMs",new JSONObject(result.profileMilliseconds()))
                        .put("fullFrame",spec.optBoolean("fullFrame",false))
                        .put("totalMs",result.totalMilliseconds()).put("rallies",intervals(result.ranges()))
                        .put("rallyCount",result.ranges().size()).put("sampleRows",result.sampleRows())
                        .put("thermalStart",result.thermalStatusStart()).put("thermalEnd",result.thermalStatusEnd())
                        .put("servingSideReady",result.servingSide()!=null).put("sideSwitchReady",result.sideSwitch()!=null)
                        .put("status","complete");
                    if(result.servingSide()!=null)row.put("servingSide",ServingSideJson.INSTANCE.encodeOutput(result.servingSide()));
                    if(result.sideSwitch()!=null)row.put("sideSwitch",SideSwitchJson.INSTANCE.encodeOutput(result.sideSwitch()));
                    if(neural!=null) row.put("neural",neural.report);
                    if(result.servingSideError()!=null || result.sideSwitchError()!=null)
                        throw new IOException("Incomplete specialists: "+result.servingSideError()+" / "+result.sideSwitchError());
                } catch(Exception | OutOfMemoryError error) {
                    android.util.Log.e("PipelineBenchmark","Case failed",error);
                    row.put("status","failed").put("error",error.toString());
                }
                save(report);
            }
            boolean complete=true;
            for(int i=0;i<rows.length();i++) complete &= rows.getJSONObject(i).optString("status").equals("complete");
            report.put("status",complete?"complete":"failed");save(report);
            runOnUiThread(()->status.setText("Pipeline benchmark finished"));
        } catch(Exception error) {
            try {report.put("status","failed").put("error",error.toString());save(report);}catch(Exception ignored){}
        }
    }
    static JSONArray intervals(List<AnalysisTypes.Interval> intervals) throws JSONException {
        JSONArray out=new JSONArray();
        for(AnalysisTypes.Interval r:intervals) out.put(new JSONObject().put("start",r.start()).put("end",r.end()).put("confidence",r.confidence()));
        return out;
    }
}
