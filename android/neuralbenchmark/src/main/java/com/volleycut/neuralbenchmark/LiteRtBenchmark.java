package com.volleycut.neuralbenchmark;

import android.os.PowerManager;
import android.os.SystemClock;
import org.json.*;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import java.io.*;
import java.nio.*;
import java.nio.file.Files;
import java.util.*;

final class LiteRtBenchmark {
    private static File local(File root,String name) throws IOException {
        File file=new File(root,name).getCanonicalFile();
        if(!file.toPath().startsWith(root.getCanonicalFile().toPath()))throw new IOException("Invalid benchmark path");
        return file;
    }
    static JSONObject measure(File root,JSONObject spec,PowerManager power) throws Exception {
        String provider=spec.getString("provider");
        if(!Set.of("cpu","gpu").contains(provider))throw new IllegalArgumentException("Invalid LiteRT provider");
        JSONObject row=new JSONObject().put("id",spec.getString("id")).put("runtime","LiteRT 1.4.2")
            .put("requestedProvider",provider).put("thermalStart",power.getCurrentThermalStatus());
        GpuDelegate delegate=null;
        Interpreter.Options options=new Interpreter.Options().setNumThreads(4);
        long started=SystemClock.elapsedRealtimeNanos();
        try {
            if(provider.equals("gpu")) {
                delegate=new GpuDelegate(new GpuDelegate.Options().setPrecisionLossAllowed(spec.optBoolean("allowFp16",false)));
                options.addDelegate(delegate);
            }
            try(Interpreter interpreter=new Interpreter(local(root,spec.getString("model")),options)) {
                interpreter.allocateTensors();
                row.put("loadMs",elapsed(started));
                JSONArray description=spec.getJSONArray("inputs");
                if(description.length()!=interpreter.getInputTensorCount())throw new IOException("Input count mismatch");
                Object[] inputs=new Object[description.length()];
                JSONArray inputInfo=new JSONArray();
                for(int i=0;i<inputs.length;i++) {
                    byte[] data=Files.readAllBytes(local(root,description.getJSONObject(i).getString("file")).toPath());
                    if(data.length!=interpreter.getInputTensor(i).numBytes())throw new IOException("Input byte count mismatch");
                    ByteBuffer buffer=ByteBuffer.allocateDirect(data.length).order(ByteOrder.LITTLE_ENDIAN);
                    buffer.put(data).rewind();inputs[i]=buffer;
                    inputInfo.put(new JSONObject().put("name",interpreter.getInputTensor(i).name())
                        .put("shape",new JSONArray(interpreter.getInputTensor(i).shape())));
                }
                row.put("inputInfo",inputInfo);
                Map<Integer,Object> outputs=new HashMap<>();
                for(int i=0;i<interpreter.getOutputTensorCount();i++)
                    outputs.put(i,ByteBuffer.allocateDirect(interpreter.getOutputTensor(i).numBytes()).order(ByteOrder.LITTLE_ENDIAN));
                JSONArray times=new JSONArray();
                int warmup=spec.optInt("warmup",3),runs=spec.optInt("runs",10);
                for(int i=-warmup;i<runs;i++) {
                    for(Object buffer:inputs)((ByteBuffer)buffer).rewind();
                    for(Object buffer:outputs.values())((ByteBuffer)buffer).rewind();
                    started=SystemClock.elapsedRealtimeNanos();
                    interpreter.runForMultipleInputsOutputs(inputs,outputs);
                    if(i>=0)times.put(elapsed(started));
                }
                JSONArray saved=new JSONArray();
                for(int i=0;i<outputs.size();i++) {
                    ByteBuffer buffer=(ByteBuffer)outputs.get(i);buffer.rewind();byte[] data=new byte[buffer.remaining()];buffer.get(data);
                    String name=spec.getString("id")+"-output-"+i+".f32";
                    Files.write(local(root,name).toPath(),data);
                    saved.put(new JSONObject().put("file",name).put("name",interpreter.getOutputTensor(i).name()));
                }
                row.put("outputs",saved).put("samplesMs",times).put("status","complete")
                    .put("allowFp16",spec.optBoolean("allowFp16",false))
                    .put("placementNote","GPU delegate initialization is explicit; inspect logcat partition/delegate messages for partial CPU execution");
            }
        } finally {if(delegate!=null)delegate.close();}
        return row.put("thermalEnd",power.getCurrentThermalStatus());
    }
    private static double elapsed(long start){return (SystemClock.elapsedRealtimeNanos()-start)/1e6;}
}
