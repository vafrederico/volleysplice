package com.volleycut.nativeanalysis;

import com.volleycut.neural.RegionalPoolWeights;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import java.nio.FloatBuffer;
import java.util.List;
import static org.junit.Assert.*;

public class NeuralRallyPipelineTest {
    @Test public void boundedFusionMatchesFullTensorAndUsesAbsoluteEmbeddingGrid() throws Exception {
        double[] times={5.25,5.5,5.75,6,6.25};
        float[] contextual=new float[times.length*520];
        for(int i=0;i<contextual.length;i++) contextual[i]=(i%37)/37f;
        JSONArray mean=new JSONArray(),scale=new JSONArray(),quality=new JSONArray();
        for(int c=0;c<112;c++) { mean.put(.1); scale.put(.2); }
        float[] tokens={11,12,21,22,31,32};
        for(int i=0;i<3;i++) quality.put(new JSONArray(new float[]{.1f,.2f,.3f,.4f,.5f,i*.01f}));
        float[] whole=NeuralRallyPipeline.fuse(times,contextual,FloatBuffer.wrap(tokens),quality,
                mean,scale,0,5,2,true,3,5);
        float[] part=NeuralRallyPipeline.fuse(times,contextual,FloatBuffer.wrap(tokens),quality,
                mean,scale,1,4,2,true,3,5);
        assertArrayEquals(java.util.Arrays.copyOfRange(whole,114,456),part,0);
        assertEquals(11,whole[104],0);
        assertEquals(21,whole[114+104],0);
        assertEquals(.75,whole[112],1e-6); // (.25s age - .1 mean)/.2
        assertEquals(-.5,whole[114+112],1e-6);
    }

    @Test public void regionalPoolingTracksRotationAndNormalizedContent() {
        double[] roi={0,0,1,1};
        float[] landscape=RegionalPoolWeights.forGeometry(1920,1080,0,roi);
        float[] rotated=RegionalPoolWeights.forGeometry(1920,1080,90,roi);
        assertArrayEquals(rotated,RegionalPoolWeights.forGeometry(1080,1920,0,roi),0);
        assertFalse(java.util.Arrays.equals(landscape,rotated));
        for(int region=0;region<4;region++) {
            double total=0;
            for(int i=0;i<49;i++) { assertTrue(landscape[region*49+i]>=0); total+=landscape[region*49+i]; }
            assertEquals(1,total,1e-6);
        }
        // Letterbox padding must have zero contribution to every regional token.
        assertEquals(0,landscape[0],0);
        assertEquals(0,landscape[48],0);
        assertEquals(0,rotated[0],0);
    }

    @Test public void frozenDecoderUsesChosenConfigurationAndHandlesEmptyInput() throws Exception {
        JSONObject recall=new JSONObject().put("smoothing",.5).put("enter",.2).put("minimum",1).put("boundary",true);
        JSONObject f1=new JSONObject().put("smoothing",1).put("enter",.9).put("minimum",.25).put("boundary",false);
        double[] times={0,.25,.5,.75,1,1.25};
        float[] probabilities=new float[24];
        for(int i=0;i<6;i++) probabilities[i*4]=.6f;
        List<AnalysisTypes.Interval> kept=NeuralRallyPipeline.decode(times,probabilities,1.5,recall);
        assertEquals(1,kept.size());
        assertEquals(0,kept.get(0).start(),0);
        assertEquals(1.375,kept.get(0).end(),0);
        assertTrue(NeuralRallyPipeline.decode(times,probabilities,1.5,f1).isEmpty());
        assertTrue(NeuralRallyPipeline.decode(new double[0],new float[0],1.5,recall).isEmpty());
    }
}
