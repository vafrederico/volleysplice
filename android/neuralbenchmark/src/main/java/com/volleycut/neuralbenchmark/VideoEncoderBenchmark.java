package com.volleycut.neuralbenchmark;

import ai.onnxruntime.*;
import android.content.Context;
import android.net.Uri;
import android.graphics.Rect;
import android.media.*;
import android.os.SystemClock;
import com.volleycut.video.DecodedVideoColor;
import com.volleycut.video.YuvColorConversion;
import org.json.*;
import java.io.*;
import java.nio.*;
import java.nio.file.Files;
import java.util.*;

/** Sequential MediaCodec-to-encoder prototype. Deliberately reports its own decode
 * cost; this is not the production asynchronous decode-only implementation. */
public final class VideoEncoderBenchmark {
    public static JSONObject run(Context context, OrtSession session, OrtEnvironment env, File video, File root,
                          JSONObject spec) throws Exception {
        long started = SystemClock.elapsedRealtimeNanos();
        int size = spec.getInt("imageSize");
        double fps = spec.getDouble("sampleFps");
        double seconds = spec.optDouble("seconds", 120);
        double startSeconds = spec.optDouble("startSeconds", 0);
        JSONArray roi = spec.getJSONArray("roi");
        double[] crop = {roi.getDouble(0),roi.getDouble(1),roi.getDouble(2),roi.getDouble(3)};
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;
        Map<String,OnnxTensor> inputs = new LinkedHashMap<>();
        JSONObject report = new JSONObject();
        JSONArray qualityRows = new JSONArray();
        long preparationNs = 0, inferenceNs = 0;
        int decoded = 0, sampled = 0;
        JSONArray timestamps = new JSONArray();
        JSONArray colorProfiles = new JSONArray();
        YuvColorConversion.Profile previousColor = null;
        ByteBuffer rgb = ByteBuffer.allocateDirect(3*size*size*4).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer pixels = rgb.asFloatBuffer();
        float[] preparedPixels = new float[3*size*size];
        try (FileOutputStream tokens = new FileOutputStream(new File(root,spec.getString("id")+"-tokens.f32"))) {
            inputs.put("image", OnnxTensor.createTensor(env,rgb,new long[]{1,3,size,size},OnnxJavaType.FLOAT));
            if (spec.has("poolWeights")) {
                String filename = spec.getString("poolWeights");
                if (!new File(filename).getName().equals(filename)) throw new IOException("Invalid pool file");
                byte[] raw = Files.readAllBytes(new File(root,filename).toPath());
                ByteBuffer weights = ByteBuffer.allocateDirect(raw.length).order(ByteOrder.LITTLE_ENDIAN);
                weights.put(raw).rewind();
                inputs.put("pool_weights",OnnxTensor.createTensor(env,weights,new long[]{1,4,7,7},OnnxJavaType.FLOAT));
            }
            if (spec.has("videoUri")) extractor.setDataSource(context, Uri.parse(spec.getString("videoUri")), null);
            else extractor.setDataSource(video.toString());
            MediaFormat format = null;
            for (int i=0;i<extractor.getTrackCount();i++) {
                MediaFormat candidate=extractor.getTrackFormat(i);
                if (candidate.getString(MediaFormat.KEY_MIME).startsWith("video/")) {
                    format=candidate; extractor.selectTrack(i); break;
                }
            }
            if (format==null) throw new IOException("No video track");
            if (startSeconds > 0) extractor.seekTo((long)(startSeconds*1e6),MediaExtractor.SEEK_TO_PREVIOUS_SYNC);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT,MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            format.setInteger(MediaFormat.KEY_OPERATING_RATE,240);
            format.setInteger(MediaFormat.KEY_PRIORITY,1);
            codec=MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME));
            report.put("decoder",codec.getName()).put("hardwareDecoder",codec.getCodecInfo().isHardwareAccelerated());
            report.put("pipeline","sequential MediaCodec; bilinear YUV-to-RGB ROI letterbox; encoder; token readback");
            codec.configure(format,null,null,0); codec.start();
            boolean inputDone=false, outputDone=false;
            boolean lastInputQueued=false;
            long maximumInputPts=Long.MIN_VALUE;
            MediaCodec.BufferInfo info=new MediaCodec.BufferInfo();
            long deadline=SystemClock.elapsedRealtime()+Math.max(30*60*1000L,(long)(seconds*10000));
            while (!outputDone) {
                if(SystemClock.elapsedRealtime()>deadline) throw new IOException("Video benchmark timed out");
                if(!inputDone) {
                    int index=codec.dequeueInputBuffer(0);
                    if(index>=0) {
                        ByteBuffer data=codec.getInputBuffer(index);
                        int count=extractor.readSampleData(data,0);
                        long pts=extractor.getSampleTime();
                        if(count<0) { codec.queueInputBuffer(index,0,0,0,MediaCodec.BUFFER_FLAG_END_OF_STREAM);inputDone=true; }
                        else {
                            codec.queueInputBuffer(index,0,count,pts,0);
                            maximumInputPts=Math.max(maximumInputPts,pts);
                            lastInputQueued=!extractor.advance();
                        }
                    }
                }
                int index=codec.dequeueOutputBuffer(info,1000);
                if(index<0) continue;
                try {
                    double timestamp=info.presentationTimeUs/1e6;
                    if(info.size>0) {
                        decoded++;
                        double target=startSeconds+sampled/fps;
                        boolean terminalFrame=lastInputQueued && info.presentationTimeUs==maximumInputPts;
                        if(shouldSample(timestamp,target,seconds,fps,terminalFrame)) {
                            long t=SystemClock.elapsedRealtimeNanos();
                            try(android.media.Image image=codec.getOutputImage(index)) {
                                if(image==null) throw new IOException("Decoder did not provide YUV image");
                                YuvColorConversion.Profile color = DecodedVideoColor.resolve(codec.getOutputFormat(index), format);
                                if (!color.equals(previousColor)) {
                                    colorProfiles.put(new JSONObject().put("firstSampleSeconds", timestamp)
                                            .put("standard", color.standard()).put("range", color.range())
                                            .put("standardSource", color.standardSource()).put("rangeSource", color.rangeSource()));
                                    previousColor = color;
                                }
                                prepare(image,preparedPixels,size,crop,color.conversion());
                                if (spec.optBoolean("quality", false)) {
                                    float[] q = quality(preparedPixels, size, image.getCropRect(), crop);
                                    q[5] = (float)(timestamp - target);
                                    JSONArray values = new JSONArray();
                                    for (float v : q) values.put(v);
                                    qualityRows.put(values);
                                }
                                pixels.rewind(); pixels.put(preparedPixels);
                            }
                            preparationNs+=SystemClock.elapsedRealtimeNanos()-t;
                            t=SystemClock.elapsedRealtimeNanos();
                            try(OrtSession.Result result=session.run(inputs)) {
                                OnnxTensor output=(OnnxTensor)result.get(0);
                                FloatBuffer values=output.getFloatBuffer();
                                ByteBuffer bytes=ByteBuffer.allocate(values.remaining()*4).order(ByteOrder.LITTLE_ENDIAN);
                                bytes.asFloatBuffer().put(values);
                                tokens.write(bytes.array());
                            }
                            inferenceNs+=SystemClock.elapsedRealtimeNanos()-t;
                            timestamps.put(timestamp);sampled++;
                        }
                        if(timestamp>=seconds) outputDone=true;
                    }
                    if((info.flags&MediaCodec.BUFFER_FLAG_END_OF_STREAM)!=0) outputDone=true;
                } finally {codec.releaseOutputBuffer(index,false);}
            }
        } finally {
            if(codec!=null) {try {codec.stop();} finally {codec.release();}}
            extractor.release();
            for(OnnxTensor input:inputs.values()) input.close();
        }
        double totalMs=(SystemClock.elapsedRealtimeNanos()-started)/1e6;
        return report.put("sampleCount",sampled).put("decodedFrames",decoded).put("sampleTimestamps",timestamps).put("quality", qualityRows)
            .put("decodedColorProfiles", colorProfiles)
            .put("seconds",seconds).put("startSeconds",startSeconds).put("sampleFps",fps).put("prepareMs",preparationNs/1e6)
            .put("encoderAndReadbackMs",inferenceNs/1e6).put("totalMs",totalMs)
            .put("decodeAndOtherMs",totalMs-(preparationNs+inferenceNs)/1e6)
            .put("tokens",spec.getString("id")+"-tokens.f32")
            .put("parityStatus","Native pixel conversion and frame selection require separate parity evaluation");
    }

    public static boolean shouldSample(double timestamp,double target,double end,double fps,boolean terminalFrame) {
        return timestamp<end && target<end && (timestamp+1e-7>=target
            || (terminalFrame && target-timestamp<=1/fps));
    }

    private static float[] quality(float[] chw, int size, Rect bounds, double[] roi) {
        int width=(int)Math.round(roi[2]*bounds.width()), height=(int)Math.round(roi[3]*bounds.height());
        double scale=Math.min((double)size/width,(double)size/height);
        int w=(int)Math.round(width*scale), h=(int)Math.round(height*scale);
        int left=(size-w)/2, top=(size-h)/2, n=w*h, plane=size*size;
        float[] gray=new float[n]; double sum=0, squares=0, clipped=0;
        for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
            int i=(y+top)*size+x+left;
            int r=Math.round((chw[i]*.229f+.485f)*255), g=Math.round((chw[plane+i]*.224f+.456f)*255), b=Math.round((chw[2*plane+i]*.225f+.406f)*255);
            float v=Math.round(.299*r+.587*g+.114*b)/255f;
            gray[y*w+x]=v;sum+=v;squares+=v*v;if(v<=2/255f||v>=253/255f)clipped++;
        }
        double lapSum=0,lapSquares=0;
        for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
            int xl=x==0?Math.min(1,w-1):x-1,xr=x==w-1?Math.max(0,w-2):x+1;
            int yu=y==0?Math.min(1,h-1):y-1,yd=y==h-1?Math.max(0,h-2):y+1;
            double v=gray[y*w+xl]+gray[y*w+xr]+gray[yu*w+x]+gray[yd*w+x]-4*gray[y*w+x];
            lapSum+=v;lapSquares+=v*v;
        }
        return new float[]{(float)n/plane,(float)(sum/n),(float)Math.sqrt(Math.max(0,squares/n-sum*sum/n/n)),
            (float)Math.max(0,lapSquares/n-lapSum*lapSum/n/n),(float)(clipped/n),0};
    }

    private static void prepare(Image image,float[] output,int size,double[] roi,YuvColorConversion color) {
        Rect bounds=image.getCropRect();
        int x0=bounds.left+(int)Math.round(roi[0]*bounds.width());
        int y0=bounds.top+(int)Math.round(roi[1]*bounds.height());
        int width=(int)Math.round(roi[2]*bounds.width()),height=(int)Math.round(roi[3]*bounds.height());
        double scale=Math.min((double)size/width,(double)size/height);
        int rw=(int)Math.round(width*scale),rh=(int)Math.round(height*scale);
        int left=(size-rw)/2,top=(size-rh)/2;
        Image.Plane[] sourcePlanes=image.getPlanes();
        PlaneData[] planes=new PlaneData[sourcePlanes.length];
        for(int i=0;i<planes.length;i++) {
            Image.Plane p=sourcePlanes[i];ByteBuffer buffer=p.getBuffer().duplicate();
            byte[] bytes=new byte[buffer.remaining()];buffer.get(bytes);
            planes[i]=new PlaneData(bytes,p.getRowStride(),p.getPixelStride());
        }
        float[] mean={.485f,.456f,.406f},std={.229f,.224f,.225f};
        for(int y=0;y<size;y++) for(int x=0;x<size;x++) {
            boolean inside=x>=left&&x<left+rw&&y>=top&&y<top+rh;
            int a=0,b=0,c=0,d=0; float dx=0,dy=0;
            if(inside) {
                double sx=Math.max(0,Math.min(width-1,(x-left+.5)*width/rw-.5));
                double sy=Math.max(0,Math.min(height-1,(y-top+.5)*height/rh-.5));
                int ix=(int)sx,iy=(int)sy;dx=(float)(sx-ix);dy=(float)(sy-iy);
                a=color(planes,x0+ix,y0+iy,color);b=color(planes,x0+Math.min(width-1,ix+1),y0+iy,color);
                c=color(planes,x0+ix,y0+Math.min(height-1,iy+1),color);d=color(planes,x0+Math.min(width-1,ix+1),y0+Math.min(height-1,iy+1),color);
            }
            for(int channel=0;channel<3;channel++) {
                int shift=16-channel*8;
                float value=inside?(((a>>shift)&255)*(1-dx)*(1-dy)+((b>>shift)&255)*dx*(1-dy)+((c>>shift)&255)*(1-dx)*dy+((d>>shift)&255)*dx*dy):0;
                output[channel*size*size+y*size+x]=(value/255-mean[channel])/std[channel];
            }
        }
    }
    private record PlaneData(byte[] values,int rowStride,int pixelStride) {}
    private static int color(PlaneData[] planes,int x,int y,YuvColorConversion color) {
        return color.rgb(sample(planes[0],x,y),sample(planes[1],x/2,y/2),sample(planes[2],x/2,y/2));
    }
    private static int sample(PlaneData p,int x,int y) {return p.values[y*p.rowStride+x*p.pixelStride]&255;}
}
