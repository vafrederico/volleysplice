package com.volleycut.nativeanalysis;

import ai.onnxruntime.*;
import android.media.Image;
import com.volleycut.neuralbenchmark.VideoEncoderBenchmark;
import com.volleycut.video.EmbeddingFrameRequests;
import com.volleycut.video.YuvColorConversion;
import java.io.*;
import java.nio.*;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;
import org.json.*;

/** Original YUV -> exact existing neural pixels -> bounded encoder worker. */
final class SharedEmbeddingConsumer implements SharedVideoFrameConsumer {
    private record Job(float[] pixels, float[] quality, long pts, int first, int end) {}
    private static final Job END = new Job(null, null, 0, 0, 0);
    private final File root;
    private final String prefix, id;
    private final double seconds;
    private final double[] roi;
    private final boolean captureHashes;
    private final BooleanSupplier cancelled;
    private final ArrayBlockingQueue<float[]> free = new ArrayBlockingQueue<>(4);
    private final ArrayBlockingQueue<Job> jobs = new ArrayBlockingQueue<>(4);
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final JSONArray timestamps = new JSONArray(), qualities = new JSONArray(), hashes = new JSONArray();
    private final JSONArray colorProfiles = new JSONArray();
    private final long started = System.nanoTime();
    private long prepareNs, inferenceNs, loadNs, waitNs, hashNs;
    private int submitted, sampled, uniqueImages;
    private EmbeddingFrameRequests.Plan sampling;
    private Set<Long> wanted = Set.of();
    private Thread worker;
    private volatile boolean closing;
    private boolean finished;
    private YuvColorConversion.Profile previousColor;
    private final JSONObject report = new JSONObject();

    SharedEmbeddingConsumer(File root, String prefix, String id, double seconds,
            double[] roi, boolean captureHashes, BooleanSupplier cancelled) {
        this.root=root; this.prefix=prefix; this.id=id; this.seconds=seconds;
        this.roi=roi.clone(); this.captureHashes=captureHashes; this.cancelled=cancelled;
        for(int i=0;i<4;i++) free.add(new float[3*224*224]);
    }
    @Override public Set<Long> plan(long[] source, String decoderName, boolean hardware) throws IOException {
        if(worker!=null) throw new IOException("Shared encoder already planned");
        sampling=EmbeddingFrameRequests.create(source, 0, seconds, 2);
        if(sampling.selectedPresentationUs().length==0) throw new IOException("No shared encoder frames");
        Set<Long> selected=new HashSet<>();
        for(long pts:sampling.selectedPresentationUs()) selected.add(pts);
        wanted=Set.copyOf(selected);
        try { report.put("decoder",decoderName).put("hardwareDecoder",hardware); }
        catch(JSONException error) { throw new IOException(error); }
        worker=new Thread(this::encode,"Shared-MobileNet-Encoder"); worker.start();
        return wanted;
    }
    @Override public boolean wants(long pts) {
        // A codec can emit duplicate PTS. Once all rows for that PTS were
        // submitted, match the standalone sampler and ignore later duplicates.
        return sampling!=null && submitted<sampling.selectedPresentationUs().length
                && sampling.selectedPresentationUs()[submitted]==pts;
    }
    private void check() throws IOException {
        if(failure.get()!=null) throw new IOException("Shared encoder failed",failure.get());
        if(closing || cancelled.getAsBoolean()) throw new IOException("Shared encoder cancelled");
    }
    @Override public void accept(Image image, long pts, YuvColorConversion.Profile color) throws IOException {
        check();
        if(submitted>=sampling.selectedPresentationUs().length
                || sampling.selectedPresentationUs()[submitted]!=pts) throw new IOException("Unexpected shared frame PTS");
        float[] pixels=null;
        try {
            long start=System.nanoTime();
            while(pixels==null) { check(); pixels=free.poll(20,TimeUnit.MILLISECONDS); }
            waitNs+=System.nanoTime()-start;
            start=System.nanoTime();
            VideoEncoderBenchmark.prepare(image,pixels,224,roi,color.conversion());
            float[] q=VideoEncoderBenchmark.quality(pixels,224,image.getCropRect(),roi);
            int end=submitted+1;
            while(end<sampling.selectedPresentationUs().length && sampling.selectedPresentationUs()[end]==pts) end++;
            if(!color.equals(previousColor)) {
                colorProfiles.put(new JSONObject().put("firstSampleSeconds",pts/1e6)
                    .put("standard",color.standard()).put("range",color.range())
                    .put("standardSource",color.standardSource()).put("rangeSource",color.rangeSource()));
                previousColor=color;
            }
            prepareNs+=System.nanoTime()-start;
            Job job=new Job(pixels,q,pts,submitted,end);
            while(!jobs.offer(job,20,TimeUnit.MILLISECONDS)) check();
            pixels=null; submitted=end; uniqueImages++;
        } catch(InterruptedException error) { Thread.currentThread().interrupt(); throw new IOException(error); }
        catch(JSONException error) { throw new IOException(error); }
        finally { if(pixels!=null) free.offer(pixels); }
    }
    private void encode() {
        Map<String,OnnxTensor> inputs=new LinkedHashMap<>();
        try(OrtSession.SessionOptions options=new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(4); options.setInterOpNumThreads(1);
            OrtEnvironment env=OrtEnvironment.getEnvironment();
            long start=System.nanoTime();
            try(OrtSession encoder=env.createSession(new File(root,prefix+"-encoder-fp32.onnx").toString(),options);
                FileOutputStream tokens=new FileOutputStream(new File(root,id+"-tokens.f32"))) {
                ByteBuffer rgb=ByteBuffer.allocateDirect(3*224*224*4).order(ByteOrder.LITTLE_ENDIAN);
                FloatBuffer imageFloats=rgb.asFloatBuffer();
                inputs.put("image",OnnxTensor.createTensor(env,rgb,new long[]{1,3,224,224},OnnxJavaType.FLOAT));
                byte[] raw=Files.readAllBytes(new File(root,prefix+"-encoder-pool_weights.f32").toPath());
                if(raw.length!=196*4) throw new IOException("Unexpected pooling shape");
                ByteBuffer weights=ByteBuffer.allocateDirect(raw.length).order(ByteOrder.LITTLE_ENDIAN);
                weights.put(raw).rewind();
                inputs.put("pool_weights",OnnxTensor.createTensor(env,weights,new long[]{1,4,7,7},OnnxJavaType.FLOAT));
                loadNs=System.nanoTime()-start;
                while(true) {
                    check(); Job job=jobs.poll(20,TimeUnit.MILLISECONDS);
                    if(job==null) continue;
                    if(job==END) break;
                    try {
                        imageFloats.rewind(); imageFloats.put(job.pixels());
                        String hash=null;
                        if(captureHashes) { start=System.nanoTime(); hash=VideoEncoderBenchmark.pixelHash(rgb); hashNs+=System.nanoTime()-start; }
                        start=System.nanoTime();
                        try(OrtSession.Result result=encoder.run(inputs)) {
                            FloatBuffer values=((OnnxTensor)result.get(0)).getFloatBuffer();
                            ByteBuffer bytes=ByteBuffer.allocate(values.remaining()*4).order(ByteOrder.LITTLE_ENDIAN);
                            bytes.asFloatBuffer().put(values);
                            for(int i=job.first();i<job.end();i++) {
                                if(sampled!=i) throw new IOException("Shared tokens out of order");
                                tokens.write(bytes.array());
                                float[] q=job.quality().clone(); q[5]=(float)(job.pts()/1e6-sampling.targetSeconds()[i]);
                                JSONArray quality=new JSONArray(); for(float v:q) quality.put(v);
                                qualities.put(quality); timestamps.put(job.pts()/1e6);
                                if(hash!=null) hashes.put(hash);
                                sampled++;
                            }
                        }
                        inferenceNs+=System.nanoTime()-start;
                    } finally { free.put(job.pixels()); }
                }
            }
        } catch(Throwable error) { failure.compareAndSet(null,error); }
        finally { for(OnnxTensor tensor:inputs.values()) tensor.close(); }
    }
    @Override public void finish() throws IOException {
        if(finished) return;
        if(worker==null) throw new IOException("Shared encoder not planned");
        try {
            while(!jobs.offer(END,20,TimeUnit.MILLISECONDS)) check();
            while(worker.isAlive()) { check(); worker.join(20); }
            check();
            if(sampled!=sampling.selectedPresentationUs().length || submitted!=sampled) throw new IOException("Incomplete shared embeddings");
            report.put("sampleCount",sampled).put("sampleTimestamps",timestamps).put("quality",qualities)
                .put("pixelHashes",hashes).put("decodedColorProfiles",colorProfiles)
                .put("seconds",seconds).put("startSeconds",0).put("sampleFps",2)
                .put("prepareMs",prepareNs/1e6).put("encoderAndReadbackMs",inferenceNs/1e6)
                .put("encoderLoadMs",loadNs/1e6).put("queueBackpressureMs",waitNs/1e6)
                .put("pixelHashMs",hashNs/1e6).put("samplePlanMs",0)
                .put("totalMs",(System.nanoTime()-started)/1e6)
                .put("uniqueSelectedImages",uniqueImages).put("bufferCount",4)
                .put("preparedBufferBytes",4*3*224*224*4)
                .put("tokens",id+"-tokens.f32").put("frameSelection","nearest-media-pts-earlier-tie-v1")
                .put("sharedDecoding",true).put("timingScope","Nested concurrent work inside AV video stage; do not add totalMs again")
                .put("pipeline","shared AV timestamp inventory and async decode-only; original YUV -> existing neural letterbox; bounded encoder worker");
            finished=true;
        } catch(InterruptedException error) { Thread.currentThread().interrupt(); throw new IOException(error); }
        catch(JSONException error) { throw new IOException(error); }
    }
    JSONObject report() throws IOException {
        if(!finished || failure.get()!=null) throw new IOException("Shared embeddings not complete",failure.get());
        return report;
    }
    @Override public void close() throws IOException {
        closing=true;
        if(worker!=null && worker.isAlive()) {
            worker.interrupt();
            boolean interrupted=false;
            while(worker.isAlive()) {
                try { worker.join(100); } catch(InterruptedException error) { interrupted=true; }
            }
            if(interrupted) Thread.currentThread().interrupt();
        }
        jobs.clear(); free.clear();
    }
}
