package com.volleycut.nativeanalysis;
import java.nio.file.*;
import java.util.*;
import org.json.*;

/** Counterfactual feature-block audit only; does not select or modify a model. */
public class AudioAttribution {
  static float[] floats(JSONArray a) {float[] out=new float[a.length()];for(int j=0;j<out.length;j++)out[j]=(float)a.getDouble(j);return out;}
  static JSONObject evaluate(Path repo,double[] times,float[] context,double duration)throws Exception {
    var all=new ModelRunner(new JSONObject(Files.readString(repo.resolve("android/app/src/main/assets/model-1ca43e38eefc.json")))).runProfiled(times,context,duration);
    var previous=new ModelRunner(new JSONObject(Files.readString(repo.resolve("android/app/src/main/assets/model-9c92b8e9333f.json")))).runProfiled(times,context,duration);
    return new JSONObject().put("allLabels",CacheInference.result(all)).put("previous",CacheInference.result(previous)).put("intervals",CacheInference.ranges(ProductionEnsemble.merge(all.intervals(),previous.intervals())));
  }
  public static void main(String[] args)throws Exception {
    Path repo=Path.of(args[0]),out=Path.of(args[3]);
    JSONObject android=new JSONObject(Files.readString(Path.of(args[1]))),ipad=new JSONObject(Files.readString(Path.of(args[2])));
    JSONArray t=android.getJSONArray("times");double[] times=new double[t.length()];for(int j=0;j<times.length;j++)times[j]=t.getDouble(j);
    float[] a=floats(android.getJSONArray("base")),i=floats(ipad.getJSONArray("base")),mixed=a.clone();
    for(int row=0;row<times.length;row++)System.arraycopy(i,row*104+77,mixed,row*104+77,27);
    JSONObject results=new JSONObject().put("purpose","Frozen feature-block attribution; no tuning or model selection");
    results.put("androidVisualIPadAudio",evaluate(repo,times,FeatureMath.contextualize(times,mixed,FeatureSchema.BASE),android.getDouble("end")));
    mixed=i.clone();for(int row=0;row<times.length;row++)System.arraycopy(a,row*104+77,mixed,row*104+77,27);
    results.put("iPadVisualAndroidAudio",evaluate(repo,times,FeatureMath.contextualize(times,mixed,FeatureSchema.BASE),android.getDouble("end")));
    results.put("jvmOnIPadContext",evaluate(repo,times,floats(ipad.getJSONArray("contextual")),android.getDouble("end")));
    Files.writeString(out,results.toString());
    for(String key:results.keySet())if(results.optJSONObject(key)!=null)System.out.println(key+": "+results.getJSONObject(key).getJSONArray("intervals").length()+" ensemble");
  }
}
