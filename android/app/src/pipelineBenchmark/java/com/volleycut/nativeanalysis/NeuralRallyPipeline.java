package com.volleycut.nativeanalysis;

import ai.onnxruntime.*;
import android.content.Context;
import android.net.Uri;
import com.volleycut.neuralbenchmark.VideoEncoderBenchmark;
import org.json.*;
import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;

/** Frozen FP32 neural input fusion, real-context chunks, and registered decoder. */
final class NeuralRallyPipeline implements AnalysisEngine.RallyOverride {
    private final Context context;
    private final File root;
    private final JSONObject spec;
    final JSONObject report=new JSONObject();
    NeuralRallyPipeline(Context context,File root,JSONObject spec) {this.context=context;this.root=root;this.spec=spec;}
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
            String family=spec.getString("family");
            boolean mobile=family.equals("mobile") || family.equals("mobile-large");
            String prefix=mobile?family:"dino";
            JSONObject config=new JSONObject(Files.readString(file(prefix+"-pipeline.json").toPath()));
            JSONArray mean=config.getJSONArray("mean"),std=config.getJSONArray("scale");
            int tokenDim=family.equals("mobile-large")?3840:mobile?2304:3840;
            int dimension=104+tokenDim+(mobile?8:0),n=times.length;
            if(FeatureSchema.BASE.size()!=104 || contextual.length!=n*520)throw new IOException("AV schema mismatch");
            OrtEnvironment env=OrtEnvironment.getEnvironment();
            try(OrtSession.SessionOptions options=new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(4);options.setInterOpNumThreads(1);
                JSONObject videoSpec=new JSONObject().put("id",spec.getString("id")).put("videoUri",uri.toString())
                    .put("imageSize",mobile?224:336).put("sampleFps",mobile?2:4).put("seconds",duration)
                    .put("roi",new JSONArray(new double[]{roi.x(),roi.y(),roi.width(),roi.height()})).put("quality",mobile);
                if(mobile)videoSpec.put("poolWeights",prefix+"-encoder-pool_weights.f32");
                long stage=System.nanoTime();
                JSONObject video;
                try(OrtSession encoder=env.createSession(file(prefix+"-encoder-fp32.onnx").toString(),options)) {
                    profile.put("neural/encoder_load",ms(stage));
                    video=VideoEncoderBenchmark.run(context,encoder,env,file("unused"),root,videoSpec);
                }
                report.put("video",video);
                profile.put("neural/embedding_video_pass",video.getDouble("totalMs"));
                stage=System.nanoTime();
                FloatBuffer tokens;
                // Map the saved embeddings rather than creating both a full
                // heap byte[] and Files.readAllBytes' full direct scratch buffer.
                // A full DINO recording exceeded the phone's 256MB Java heap.
                tokens=readFloats(file(video.getString("tokens")));
                int samples=video.getInt("sampleCount");
                if(tokens.remaining()!=samples*tokenDim)throw new IOException("Invalid token count");
                float[] values=new float[n*dimension];
                JSONArray quality=video.getJSONArray("quality");
                for(int row=0;row<n;row++) {
                    int dst=row*dimension;
                    for(int c=0;c<104;c++)values[dst+c]=scale(contextual[row*520+208+c],mean,std,c);
                    int sample=(int)Math.floor(times[row]*(mobile?2:4)+1e-8);
                    if(sample<0 || sample>=samples)throw new IOException("Uncovered AV timestamp "+times[row]);
                    for(int c=0;c<tokenDim;c++)values[dst+104+c]=tokens.get(sample*tokenDim+c);
                    if(mobile) {
                        JSONArray q=quality.getJSONArray(sample);
                        for(int c=0;c<8;c++) {
                            float v=c<6?(float)q.getDouble(c):c==6?(float)(times[row]-sample/2.0):1f;
                            values[dst+104+tokenDim+c]=scale(v,mean,std,104+c);
                        }
                    }
                }
                writeFloats(file(spec.getString("id")+"-features.f32"),values);
                report.put("featureRows",n).put("featureDimension",dimension);
                profile.put("neural/fusion_normalization_and_save",ms(stage));
                stage=System.nanoTime();
                float[] probabilities=new float[n*4];
                try(OrtSession temporal=env.createSession(file(prefix+"-tcn-dynamic-fp32.onnx").toString(),options)) {
                    profile.put("neural/temporal_load",ms(stage));stage=System.nanoTime();
                    for(int core=0;core<n;core+=128) {
                        int end=Math.min(n,core+128),left=Math.max(0,core-62),right=Math.min(n,end+62),count=right-left;
                        FloatBuffer chunk=FloatBuffer.wrap(Arrays.copyOfRange(values,left*dimension,right*dimension));
                        try(OnnxTensor input=OnnxTensor.createTensor(env,chunk,new long[]{1,count,dimension});
                            OrtSession.Result output=temporal.run(Map.of("features",input))) {
                            FloatBuffer logits=((OnnxTensor)output.get(0)).getFloatBuffer();
                            for(int row=core;row<end;row++)for(int c=0;c<4;c++) probabilities[row*4+c]=(float)(1/(1+Math.exp(-logits.get((row-left)*4+c))));
                        }
                    }
                }
                profile.put("neural/temporal_inference",ms(stage));stage=System.nanoTime();
                writeFloats(file(spec.getString("id")+"-probabilities.f32"),probabilities);
                List<AnalysisTypes.Interval> ranges=decode(times,probabilities,duration,config.getJSONObject("decoder"));
                profile.put("neural/decoder_and_save",ms(stage));
                profile.put("neural/total",ms(start));
                report.put("decoder",config.getJSONObject("decoder")).put("times",new JSONArray(times))
                    .put("precision","fp32").put("inputParity","Native pixel/PTS parity still requires qualification");
                return ranges;
            }
        }catch(Exception error){throw new IOException("Neural pipeline failed",error);}
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
