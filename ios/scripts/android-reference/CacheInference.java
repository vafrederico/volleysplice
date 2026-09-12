package com.volleycut.nativeanalysis;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import org.json.*;

/** Read-only offline replay of Android's own compiled math on its retained cache. */
public class CacheInference {
  static ByteBuffer read(Path path) throws Exception { return ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.BIG_ENDIAN); }
  static JSONArray ranges(List<AnalysisTypes.Interval> values) {
    JSONArray out=new JSONArray();
    for(var x:values) out.put(new JSONObject().put("start",x.start()).put("end",x.end()).put("confidence",x.confidence()).put("agreement",x.agreement()));
    return out;
  }
  static JSONObject result(ModelRunner.RunResult r) {
    return new JSONObject().put("intervals",ranges(r.intervals())).put("rallyProbabilities",r.rallyProbabilities()).put("serveProbabilities",r.serveProbabilities()).put("deadStateProbabilities",r.deadStateProbabilities());
  }
  public static void main(String[] args) throws Exception {
    Path repo=Path.of(args[0]), cache=Path.of(args[1]); double duration=Double.parseDouble(args[2]);
    ByteBuffer v=read(cache.resolve("visual-complete.bin"));
    if(v.getInt()!=0x56434643 || v.getInt()!=2) throw new IllegalArgumentException("Unexpected visual schema");
    int rows=v.getInt(), cols=v.getInt(); if(cols!=73) throw new IllegalArgumentException("Visual columns");
    double[] times=new double[rows]; for(int i=0;i<rows;i++)times[i]=v.getDouble();
    float[] visual=new float[rows*cols]; v.asFloatBuffer().get(visual);
    ByteBuffer a=read(cache.resolve("audio.bin"));
    if(a.getInt()!=0x56434131 || a.getInt()!=1 || a.getInt()!=rows || a.getInt()!=27) throw new IllegalArgumentException("Audio header");
    float[] audio=new float[rows*27]; a.asFloatBuffer().get(audio);
    ByteBuffer c=read(cache.resolve("context.bin"));
    if(c.getInt()!=0x56434331 || c.getInt()!=1 || c.getInt()!=rows || c.getInt()!=520) throw new IllegalArgumentException("Context header");
    float[] context=new float[rows*520]; c.asFloatBuffer().get(context);
    float[] temporal=FeatureMath.temporalVisualFeatures(visual,rows),base=new float[rows*104];
    for(int i=0;i<rows;i++) { System.arraycopy(visual,i*73,base,i*104,73);System.arraycopy(temporal,i*4,base,i*104+73,4);System.arraycopy(audio,i*27,base,i*104+77,27); }
    float[] recreated=FeatureMath.contextualize(times,base,FeatureSchema.BASE);int changed=0;
    for(int i=0;i<context.length;i++)if(Float.floatToIntBits(context[i])!=Float.floatToIntBits(recreated[i]))changed++;
    var all=new ModelRunner(new JSONObject(Files.readString(repo.resolve("android/app/src/main/assets/model-1ca43e38eefc.json")))).runProfiled(times,context,duration);
    var previous=new ModelRunner(new JSONObject(Files.readString(repo.resolve("android/app/src/main/assets/model-9c92b8e9333f.json")))).runProfiled(times,context,duration);
    var ensemble=ProductionEnsemble.merge(all.intervals(),previous.intervals());
    JSONObject out=new JSONObject().put("provenance","Offline Android JVM inference from retained native cache; not a new device decode").put("cacheKey",cache.getFileName().toString()).put("times",times).put("base",base).put("contextual",context).put("featureNames",FeatureSchema.BASE).put("end",duration).put("start",0).put("allLabels",result(all)).put("previous",result(previous)).put("intervals",ranges(ensemble)).put("recreatedContextDifferentElements",changed);
    Files.writeString(cache.resolve("android-analysis.json"),out.toString());
    Files.writeString(cache.resolve("identity-schema.txt"),"frame="+String.join(",",FeatureSchema.FRAME)+"\naudio="+String.join(",",FeatureSchema.AUDIO));
    System.out.println(cache.getFileName()+" rows="+rows+" all="+all.intervals().size()+" previous="+previous.intervals().size()+" ensemble="+ensemble.size()+" regeneratedContextDifferences="+changed);
  }
}
