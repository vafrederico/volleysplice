package com.volleycut.nativeanalysis;

import android.content.Context;
import android.net.Uri;
import java.io.Closeable;
import java.io.IOException;
import java.util.*;
import java.util.function.BooleanSupplier;
import org.json.JSONObject;

/** Opt-in paired analysis: one fresh video pass and one prepared image for both encoders. */
final class PairedNeuralPipeline implements AnalysisEngine.RallyOverride, Closeable {
    final String primaryId, alternateId;
    final NeuralRallyPipeline primary, alternate;
    final Map<String,Double> alternateProfile = new LinkedHashMap<>();
    List<AnalysisTypes.Interval> alternateRanges = List.of();
    private final BooleanSupplier cancelled;
    private SharedEmbeddingConsumer consumer;
    private double start;
    long alternateHeadMilliseconds;

    PairedNeuralPipeline(Context context,String primaryId,AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,BooleanSupplier cancelled) throws IOException {
        if(!RallyModels.isNeural(primaryId)) throw new IOException("Paired analysis requires a distilled model");
        this.primaryId=primaryId;
        this.alternateId=primaryId.equals(RallyModels.RECALL)?RallyModels.F1:RallyModels.RECALL;
        this.cancelled=cancelled;
        primary=DistilledRallyModels.open(context,primaryId,media,roi,cancelled,true,true);
        try { alternate=DistilledRallyModels.open(context,alternateId,media,roi,cancelled,true,false); }
        catch(IOException error) { primary.close(); throw error; }
    }
    @Override public boolean supportsEnsembleSuppression() { return false; }
    @Override public SharedVideoFrameConsumer prepareSharedVideo(AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,AnalysisTypes.AnalysisWindow window,int sourceFrameLimit,
            BooleanSupplier cancelled) throws IOException {
        if(sourceFrameLimit!=Integer.MAX_VALUE) throw new IOException("Paired analysis requires an unrestricted source-frame limit");
        start=window.start();
        consumer=new SharedEmbeddingConsumer(List.of(primary.encoderTarget(),alternate.encoderTarget()),
                "mobile-large",Math.floor(start*2)/2,window.end(),
                new double[]{roi.x(),roi.y(),roi.width(),roi.height()},media.rotation(),false,cancelled,true);
        primary.useSharedEmbeddings(consumer); alternate.useSharedEmbeddings(consumer);
        return consumer;
    }
    @Override public List<AnalysisTypes.Interval> run(Uri uri,AnalysisTypes.Roi roi,double[] times,
            float[] contextual,double duration,Map<String,Double> profile) throws IOException {
        List<AnalysisTypes.Interval> selected=primary.run(uri,roi,times,contextual,duration,profile);
        if(cancelled.getAsBoolean()) throw new IOException("Analysis cancelled");
        long began=System.nanoTime();
        alternateRanges=alternate.run(uri,roi,times,contextual,duration,alternateProfile);
        alternateHeadMilliseconds=(System.nanoTime()-began)/1_000_000;
        profile.put("paired/alternate_rally_wall",(System.nanoTime()-began)/1e6);
        return selected;
    }
    JSONObject report() throws Exception {
        return new JSONObject().put("primaryModel",primaryId).put("alternateModel",alternateId)
                .put("primary",primary.report).put("alternate",alternate.report)
                .put("alternateProfileMilliseconds",new JSONObject(alternateProfile))
                .put("sharedImagePreparation",consumer!=null);
    }
    @Override public void close() throws IOException {
        try { primary.close(); } finally { alternate.close(); }
    }
}
