package com.volleycut.nativeanalysis;

import ai.onnxruntime.*;
import android.content.Context;
import android.net.Uri;
import com.volleycut.neural.NeuralVideoEncoder;
import org.json.*;
import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;

/** Frozen FP32 neural input fusion, real-context chunks, and registered decoder. */
final class NeuralRallyPipeline implements AnalysisEngine.RallyOverride, Closeable {
    private final Context context;
    private final File root;
    private final File modelRoot;
    private final boolean diagnostics;
    private final java.util.function.BooleanSupplier cancelled;
    private final JSONObject spec;
    final JSONObject report=new JSONObject();
    private NeuralRallyScores scores;
    NeuralRallyScores scores() { return scores; }
    private SharedEmbeddingConsumer shared;
    NeuralRallyPipeline(Context context,File root,JSONObject spec) {
        this(context,root,spec,root,()->false,true);
    }
    NeuralRallyPipeline(Context context,File root,JSONObject spec,File modelRoot,
            java.util.function.BooleanSupplier cancelled,boolean diagnostics) {
        this.context=context;this.root=root;this.spec=spec;this.modelRoot=modelRoot;
        this.cancelled=cancelled;this.diagnostics=diagnostics;
    }
    private void checkCancelled() throws IOException {
        if(cancelled.getAsBoolean() || Thread.currentThread().isInterrupted()) throw new IOException("Analysis cancelled");
    }
    private File model(String suffix) throws IOException {
        if(diagnostics) return file(spec.optString("family")+suffix);
        String name=switch(suffix) {
            case "-encoder-fp32.onnx" -> "encoder.onnx";
            case "-tcn-dynamic-fp32.onnx" -> "temporal.onnx";
            case "-pipeline.json" -> "pipeline.json";
            default -> throw new IOException("Unknown neural asset");
        };
        return new File(modelRoot,name);
    }
    @Override public void close() throws IOException {
        if(shared!=null) shared.close();
        if(!diagnostics) {
            File[] temporary=root.listFiles();
            if(temporary!=null) for(File item:temporary) if(item.isFile()) Files.deleteIfExists(item.toPath());
            Files.deleteIfExists(root.toPath());
        }
    }
    @Override public boolean supportsEnsembleSuppression() { return diagnostics; }
    @Override public SharedVideoFrameConsumer prepareSharedVideo(AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi, AnalysisTypes.AnalysisWindow window, int sourceFrameLimit,
            java.util.function.BooleanSupplier cancelled) throws IOException {
        if(!spec.optBoolean("sharedDecoding",false)) return null;
        String family=spec.optString("family");
        if(!diagnostics && sourceFrameLimit!=Integer.MAX_VALUE) return null;
        if(diagnostics && (!(family.equals("mobile") || family.equals("mobile-large")) || media.rotation()!=0
                || window.start()!=0 || sourceFrameLimit!=Integer.MAX_VALUE)) {
            throw new IOException("Shared neural experiment requires MobileNet, unrotated input, zero start and full source frame limit");
        }
        double sampleStart=diagnostics?0:Math.floor(window.start()*2)/2;
        shared=new SharedEmbeddingConsumer(root,family,spec.optString("id"),sampleStart,window.end(),
                new double[]{roi.x(),roi.y(),roi.width(),roi.height()},media.rotation(),
                spec.optBoolean("capturePixelHashes",false),cancelled,
                model("-encoder-fp32.onnx"),file(family+"-encoder-pool_weights.f32"),!diagnostics);
        return shared;
    }
    private File file(String name) throws IOException {
        File result=new File(root,name).getCanonicalFile();
        if(!result.toPath().startsWith(root.getCanonicalFile().toPath()))throw new IOException("Path escape");
        return result;
    }
    private static double ms(long start) {return (System.nanoTime()-start)/1e6;}
    private static float scale(float value,JSONArray mean,JSONArray std,int c) throws JSONException {
        return (float)Math.max(-10,Math.min(10,(value-mean.getDouble(c))/std.getDouble(c)));
    }
    public List<AnalysisTypes.Interval> run(Uri uri,AnalysisTypes.Roi roi,double[] times,float[] contextual,double duration,Map<String,Double> profile) throws IOException {
        long start=System.nanoTime();
        try {
            checkCancelled();
            String family=spec.getString("family");
            boolean mobile=family.equals("mobile") || family.equals("mobile-large");
            String prefix=mobile?family:"dino";
            JSONObject config=new JSONObject(Files.readString(model("-pipeline.json").toPath()));
            JSONArray mean=config.getJSONArray("mean"),std=config.getJSONArray("scale");
            int tokenDim=family.equals("mobile-large")?3840:mobile?2304:3840;
            int dimension=104+tokenDim+(mobile?8:0),n=times.length;
            if(n==0) return List.of();
            if(mean.length()!=104+(mobile?8:0) || std.length()!=mean.length()) throw new IOException("Normalizer dimensions mismatch");
            for(int c=0;c<mean.length();c++) if(!Double.isFinite(mean.getDouble(c)) || !Double.isFinite(std.getDouble(c)) || std.getDouble(c)<=0) throw new IOException("Invalid neural normalization");
            if(FeatureSchema.BASE.size()!=104 || contextual.length!=n*520)throw new IOException("AV schema mismatch");
            OrtEnvironment env=OrtEnvironment.getEnvironment();
            try(OrtSession.SessionOptions options=new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(4);options.setInterOpNumThreads(1);
                JSONObject videoSpec=new JSONObject().put("id",spec.getString("id")).put("videoUri",uri.toString())
                    .put("imageSize",mobile?224:336).put("sampleFps",mobile?2:4).put("seconds",duration)
                    .put("roi",new JSONArray(new double[]{roi.x(),roi.y(),roi.width(),roi.height()})).put("quality",mobile)
                    .put("capturePixelHashes",spec.optBoolean("capturePixelHashes",false))
                    .put("dynamicPoolWeights",!diagnostics).put("rotation",spec.optInt("rotation",0)).put("startSeconds",diagnostics?0:Math.floor(times[0]*2)/2);
                if(mobile)videoSpec.put("poolWeights",prefix+"-encoder-pool_weights.f32");
                long stage=System.nanoTime();
                JSONObject video;
                if(shared!=null) {
                    video=shared.report();
                    profile.put("neural/shared_embedding_prepare",video.getDouble("prepareMs"));
                    profile.put("neural/shared_encoder_inference_readback",video.getDouble("encoderAndReadbackMs"));
                    profile.put("neural/shared_encoder_load",video.getDouble("encoderLoadMs"));
                    profile.put("neural/shared_queue_backpressure",video.getDouble("queueBackpressureMs"));
                    profile.put("neural/shared_video_span",video.getDouble("totalMs"));
                } else {
                    try(OrtSession encoder=env.createSession(model("-encoder-fp32.onnx").toString(),options)) {
                        profile.put("neural/encoder_load",ms(stage));
                        video=NeuralVideoEncoder.run(context,encoder,env,file("unused"),root,videoSpec,cancelled);
                    }
                    profile.put("neural/embedding_video_pass",video.getDouble("totalMs"));
                }
                report.put("video",video);
                report.put("sharedDecoding",shared!=null);
                stage=System.nanoTime();
                FloatBuffer tokens;
                // Map the saved embeddings rather than creating both a full
                // heap byte[] and Files.readAllBytes' full direct scratch buffer.
                // A full DINO recording exceeded the phone's 256MB Java heap.
                tokens=readFloats(file(video.getString("tokens")));
                int samples=video.getInt("sampleCount");
                if(tokens.remaining()!=samples*tokenDim)throw new IOException("Invalid token count");
                JSONArray quality=video.getJSONArray("quality");
                double sampleStart=video.optDouble("startSeconds",0);
                // Production fuses one bounded halo chunk at a time. Diagnostic runs
                // retain the full tensor for strict numerical qualification only.
                float[] values=diagnostics?fuse(times,contextual,tokens,quality,mean,std,
                        0,n,tokenDim,mobile,samples,sampleStart):null;
                if(diagnostics) writeFloats(file(spec.getString("id")+"-features.f32"),values);
                report.put("featureRows",n).put("featureDimension",dimension);
                profile.put("neural/fusion_normalization_and_save",ms(stage));
                stage=System.nanoTime();
                float[] probabilities=new float[n*4];
                try(OrtSession temporal=env.createSession(model("-tcn-dynamic-fp32.onnx").toString(),options)) {
                    profile.put("neural/temporal_load",ms(stage));stage=System.nanoTime();
                    for(int core=0;core<n;core+=128) {
                        int end=Math.min(n,core+128),left=Math.max(0,core-62),right=Math.min(n,end+62),count=right-left;
                        checkCancelled();
                        FloatBuffer chunk=FloatBuffer.wrap(diagnostics?Arrays.copyOfRange(values,left*dimension,right*dimension):
                                fuse(times,contextual,tokens,quality,mean,std,left,right,tokenDim,mobile,samples,sampleStart));
                        try(OnnxTensor input=OnnxTensor.createTensor(env,chunk,new long[]{1,count,dimension});
                            OrtSession.Result output=temporal.run(Map.of("features",input))) {
                            FloatBuffer logits=((OnnxTensor)output.get(0)).getFloatBuffer();
                            for(int row=core;row<end;row++)for(int c=0;c<4;c++) probabilities[row*4+c]=(float)(1/(1+Math.exp(-logits.get((row-left)*4+c))));
                        }
                    }
                }
                profile.put("neural/temporal_inference",ms(stage));stage=System.nanoTime();
                if(diagnostics) writeFloats(file(spec.getString("id")+"-probabilities.f32"),probabilities);
                List<AnalysisTypes.Interval> ranges=decode(times,probabilities,duration,config.getJSONObject("decoder"));
                profile.put("neural/decoder_and_save",ms(stage));
                profile.put("neural/total",ms(start));
                report.put("decoder",config.getJSONObject("decoder")).put("times",new JSONArray(times))
                    .put("precision","fp32").put("inputParity","Native pixel/PTS parity still requires qualification");
                checkCancelled();
                if(!diagnostics) scores=new NeuralRallyScores(spec.getString("id"),times.clone(),probabilities);
                return ranges;
            }
        }catch(Exception error){throw new IOException("Neural pipeline failed",error);}
    }
    static float[] fuse(double[] times,float[] contextual,FloatBuffer tokens,JSONArray quality,
            JSONArray mean,JSONArray std,int left,int right,int tokenDim,boolean mobile,int samples,
            double sampleStart) throws IOException,JSONException {
        int dimension=104+tokenDim+(mobile?8:0);
        float[] values=new float[(right-left)*dimension];
        for(int row=left;row<right;row++) {
            int dst=(row-left)*dimension;
            for(int c=0;c<104;c++) values[dst+c]=scale(contextual[row*520+208+c],mean,std,c);
            int sample=(int)Math.floor((times[row]-sampleStart)*(mobile?2:4)+1e-8);
            if(sample<0 || sample>=samples) throw new IOException("Uncovered AV timestamp");
            for(int c=0;c<tokenDim;c++) values[dst+104+c]=tokens.get(sample*tokenDim+c);
            if(mobile) {
                JSONArray q=quality.getJSONArray(sample);
                for(int c=0;c<8;c++) {
                    float v=c<6?(float)q.getDouble(c):c==6?(float)(times[row]-(sampleStart+sample/2.0)):1f;
                    values[dst+104+tokenDim+c]=scale(v,mean,std,104+c);
                }
            }
        }
        return values;
    }
    static FloatBuffer readFloats(File file)throws IOException {
        try(FileChannel channel=FileChannel.open(file.toPath(),StandardOpenOption.READ)) {
            if(channel.size()==0 || channel.size()%4!=0)throw new IOException("Invalid float tensor length");
            return channel.map(FileChannel.MapMode.READ_ONLY,0,channel.size()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        }
    }
    static void writeFloats(File file,float[] values)throws IOException {
        ByteBuffer bytes=ByteBuffer.allocate(65536).order(ByteOrder.LITTLE_ENDIAN);
        try(OutputStream output=new BufferedOutputStream(new FileOutputStream(file))) {
            for(int offset=0;offset<values.length;offset+=bytes.capacity()/4) {
                int count=Math.min(values.length-offset,bytes.capacity()/4);
                bytes.clear();bytes.asFloatBuffer().put(values,offset,count);
                output.write(bytes.array(),0,count*4);
            }
        }
    }
    static List<AnalysisTypes.Interval> decode(double[] times,float[] p,double duration,JSONObject config)throws JSONException {
        int n=times.length,w=Math.min(n,Math.max(1,(int)Math.rint(config.getDouble("smoothing")*4))),min=Math.max(1,(int)Math.rint(config.getDouble("minimum")*4));
        float[] smooth=new float[n];boolean[] mask=new boolean[n];boolean live=false;double enter=config.getDouble("enter");
        for(int i=0;i<n;i++) {
            for(int k=0;k<w;k++)smooth[i]+=p[Math.max(0,Math.min(n-1,i+k-w/2))*4]/w;
            if(!live&&smooth[i]>=enter)live=true;else if(live&&smooth[i]<enter-.1)live=false;mask[i]=live;
        }
        for(int i=0;i<n;) {int j=i+1;while(j<n&&mask[j]==mask[i])j++;if(!mask[i]&&i>0&&j<n&&j-i<=2)Arrays.fill(mask,i,j,true);i=j;}
        List<AnalysisTypes.Interval> ranges=new ArrayList<>();
        for(int i=0;i<n;) {
            if(!mask[i]){i++;continue;}int j=i+1;while(j<n&&mask[j])j++;
            float max=0,sum=0;for(int k=i;k<j;k++){max=Math.max(max,smooth[k]);sum+=smooth[k];}
            if(j-i>=min||max>=.9) {
                double a=Math.max(0,times[i]-.125),b=Math.min(duration,times[j-1]+.125);
                if(config.getBoolean("boundary")) {
                    double originalA=a,originalB=b;float bestA=-1,bestB=-1;
                    for(int k=0;k<n;k++) {
                        if(Math.abs(times[k]-originalA)<=.75&&p[k*4+1]>=.65&&p[k*4+1]>bestA){a=times[k];bestA=p[k*4+1];}
                        if(Math.abs(times[k]-originalB)<=.75&&p[k*4+2]>=.65&&p[k*4+2]>bestB){b=times[k];bestB=p[k*4+2];}
                    }
                }
                if(b>a)ranges.add(new AnalysisTypes.Interval(a,b,sum/(j-i),"neural"));
            }
            i=j;
        }
        return ranges;
    }
}
