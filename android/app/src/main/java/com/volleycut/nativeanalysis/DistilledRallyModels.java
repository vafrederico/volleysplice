package com.volleycut.nativeanalysis;

import com.volleycut.neural.RegionalPoolWeights;

import android.content.Context;
import org.json.JSONObject;
import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.function.BooleanSupplier;

/** Verified bundled assets and an isolated, temporary workspace for one analysis. */
final class DistilledRallyModels {
    private DistilledRallyModels() {}

    static NeuralRallyPipeline open(Context context, String modelId, AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi, BooleanSupplier cancelled) throws IOException {
        return open(context,modelId,media,roi,cancelled,true);
    }
    static NeuralRallyPipeline open(Context context, String modelId, AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi, BooleanSupplier cancelled, boolean sharedDecoding) throws IOException {
        if (FeatureSchema.MODEL_ID.equals(modelId)) return null;
        if (!RallyModels.isNeural(modelId)) throw new IOException("The selected rally model is unavailable");
        File work=null;
        try {
            JSONObject manifest;
            try(InputStream input=context.getAssets().open("rally-models/manifest.json")) {
                manifest=new JSONObject(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8));
            }
            if(manifest.getInt("schemaVersion")!=1) throw new IOException("Unsupported rally model bundle");
            JSONObject variant=manifest.getJSONObject("variants").getJSONObject(RallyModels.variant(modelId));
            if(!modelId.equals(variant.getString("id"))) throw new IOException("Rally model identity mismatch");
            String directory=safeName(variant.getString("directory"));
            File models=new File(context.getFilesDir(),"rally-models/"+modelId);
            if(!models.isDirectory() && !models.mkdirs()) throw new IOException("Cannot create model directory");
            JSONObject files=variant.getJSONObject("files");
            for(String key:new String[]{"encoder","temporal","pipeline"}) {
                if(cancelled.getAsBoolean()) throw new IOException("Analysis cancelled");
                JSONObject metadata=files.getJSONObject(key);
                String name=safeName(metadata.getString("name"));
                String expected=metadata.getString("sha256");
                long size=metadata.getLong("sizeBytes");
                File target=new File(models,name);
                if(target.isFile() && target.length()==size && expected.equals(hash(target))) continue;
                File temporary=new File(models,name+".tmp");
                try {
                    try(InputStream input=context.getAssets().open("rally-models/"+directory+"/"+name);
                        OutputStream output=new BufferedOutputStream(new FileOutputStream(temporary))) {
                        byte[] buffer=new byte[65536]; int read;
                        while((read=input.read(buffer))!=-1) {
                            if(cancelled.getAsBoolean()) throw new IOException("Analysis cancelled");
                            output.write(buffer,0,read);
                        }
                    }
                    if(temporary.length()!=size || !expected.equals(hash(temporary))) throw new IOException("Rally model asset integrity check failed");
                    Files.move(temporary.toPath(),target.toPath(),StandardCopyOption.REPLACE_EXISTING);
                } finally { Files.deleteIfExists(temporary.toPath()); }
            }
            File parent=new File(context.getCacheDir(),"neural-analysis");
            if(!parent.isDirectory() && !parent.mkdirs()) throw new IOException("Cannot create neural workspace");
            // The foreground service serializes analyses. A killed process can leave
            // temporary tokens; remove those before allocating another recording.
            File[] stale=parent.listFiles();
            if(stale!=null) for(File old:stale) removeWorkspace(old);
            work=Files.createTempDirectory(parent.toPath(),"run-").toFile();
            NeuralRallyPipeline.writeFloats(new File(work,"mobile-large-encoder-pool_weights.f32"),
                    RegionalPoolWeights.forGeometry(media.width(),media.height(),media.rotation(),
                            new double[]{roi.x(),roi.y(),roi.width(),roi.height()}));
            JSONObject spec=new JSONObject().put("family","mobile-large").put("id",modelId)
                    .put("rotation",media.rotation()).put("sharedDecoding",sharedDecoding);
            return new NeuralRallyPipeline(context,work,spec,models,cancelled,false);
        } catch(Exception error) {
            if(work!=null) removeWorkspace(work);
            if(error instanceof IOException io) throw io;
            throw new IOException("Cannot load the selected rally model",error);
        }
    }
    private static String safeName(String value) throws IOException {
        if(!value.matches("[A-Za-z0-9._-]+") || value.equals(".") || value.equals("..")) throw new IOException("Invalid asset name");
        return value;
    }
    private static String hash(File file) throws Exception {
        MessageDigest digest=MessageDigest.getInstance("SHA-256");
        try(InputStream input=new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer=new byte[65536]; int read;
            while((read=input.read(buffer))!=-1) digest.update(buffer,0,read);
        }
        return HexFormat.of().formatHex(digest.digest());
    }
    private static void removeWorkspace(File directory) throws IOException {
        if(!directory.getName().startsWith("run-")) return;
        File[] files=directory.listFiles();
        if(files!=null) for(File file:files) if(file.isFile()) Files.deleteIfExists(file.toPath());
        Files.deleteIfExists(directory.toPath());
    }
}
