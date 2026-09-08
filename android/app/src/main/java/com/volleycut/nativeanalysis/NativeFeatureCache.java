package com.volleycut.nativeanalysis;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Locale;
import java.util.Properties;

/** Restart-safe cache for the expensive raw visual and audio feature matrices. */
final class NativeFeatureCache {
    enum Mode {
        USE("use"), BYPASS("bypass"), REFRESH("refresh");

        private final String wireName;

        Mode(String wireName) {
            this.wireName = wireName;
        }

        String wireName() {
            return wireName;
        }

        static Mode fromWireName(String value) {
            if (value != null) {
                for (Mode mode : values()) {
                    if (mode.wireName.equalsIgnoreCase(value)) return mode;
                }
            }
            return USE;
        }
    }

    record LoadedVisual(
            float[] values,
            double[] times,
            int rows,
            boolean complete,
            double analyzedDurationSeconds,
            boolean sourceFrameLimitReached,
            int decodedSourceFrames,
            int decodeOnlySourceFrames,
            int decoderOutputFrames,
            String decoderName,
            boolean hardwareDecoder
    ) {
        static LoadedVisual empty() {
            return new LoadedVisual(
                    new float[0], new double[0], 0, false, 0, false,
                    0, 0, 0, "", false
            );
        }
    }

    record CacheStats(
            String mode,
            String key,
            boolean visualHit,
            boolean audioHit,
            boolean contextHit,
            int resumedVisualRows,
            int savedVisualRows,
            boolean complete,
            long bytes,
            String failure
    ) {}

    private static final String CACHE_VERSION = "native-features-v1";
    private static final int CHUNK_MAGIC = 0x56434631;
    private static final int COMPLETE_VISUAL_MAGIC = 0x56434643;
    private static final int AUDIO_MAGIC = 0x56434131;
    private static final int CONTEXT_MAGIC = 0x56434331;
    private static final int FILE_SCHEMA = 1;
    private static final int COMPLETE_VISUAL_SCHEMA = 2;
    private static final int CHUNK_ROWS = 16;
    private static final String MANIFEST_NAME = "manifest.properties";
    private static final String COMPLETE_VISUAL_NAME = "visual-complete.bin";
    private static final String AUDIO_NAME = "audio.bin";
    private static final String CONTEXT_NAME = "context.bin";

    private final Mode mode;
    private final String key;
    private final File cacheRoot;
    private final File entryDirectory;
    private final int maximumRows;
    private final Properties manifest = new Properties();
    private int savedVisualRows;
    private int chunkCount;
    private boolean visualComplete;
    private boolean audioComplete;
    private boolean contextComplete;
    private boolean visualHit;
    private boolean audioHit;
    private boolean contextHit;
    private int resumedVisualRows;
    private String failure = "";

    private NativeFeatureCache(
            Context context,
            Uri uri,
            String displayName,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            int sourceFrameLimit,
            int maximumRows,
            AnalysisTypes.AnalysisWindow analysisWindow,
            Mode mode,
            Long knownSourceSize,
            Long knownSourceModified
    ) {
        this.mode = mode;
        this.maximumRows = maximumRows;
        cacheRoot = new File(context.getFilesDir(), CACHE_VERSION);
        key = buildKey(
                context, uri, displayName, media, roi, sourceFrameLimit, analysisWindow,
                knownSourceSize, knownSourceModified
        );
        entryDirectory = new File(cacheRoot, key);
        if (mode == Mode.BYPASS) return;
        if (mode == Mode.REFRESH) deleteRecursively(entryDirectory);
        loadManifest();
    }

    static NativeFeatureCache open(
            Context context,
            Uri uri,
            String displayName,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            int sourceFrameLimit,
            int maximumRows,
            AnalysisTypes.AnalysisWindow analysisWindow,
            Mode mode
    ) {
        return new NativeFeatureCache(
                context.getApplicationContext(), uri, displayName, media, roi,
                sourceFrameLimit, maximumRows, analysisWindow, mode, null, null
        );
    }

    /** Opens a retained cache without requiring the original document URI to remain readable. */
    static NativeFeatureCache openWithSourceMetadata(
            Context context,
            Uri uri,
            String displayName,
            long sourceSize,
            long sourceModified,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            int sourceFrameLimit,
            int maximumRows,
            AnalysisTypes.AnalysisWindow analysisWindow
    ) {
        return new NativeFeatureCache(
                context.getApplicationContext(), uri, displayName, media, roi,
                sourceFrameLimit, maximumRows, analysisWindow, Mode.USE,
                sourceSize, sourceModified
        );
    }

    synchronized LoadedVisual loadVisual() {
        if (mode == Mode.BYPASS || savedVisualRows <= 0) return LoadedVisual.empty();
        int columns = FeatureSchema.FRAME.size();
        float[] values = new float[savedVisualRows * columns];
        double[] times = new double[savedVisualRows];
        int expectedStart = 0;
        try {
            File completeFile = new File(entryDirectory, COMPLETE_VISUAL_NAME);
            if (visualComplete && completeFile.isFile()) {
                int completeSchema = readCompleteVisual(
                        completeFile, values, times, savedVisualRows, columns
                );
                if (completeSchema < COMPLETE_VISUAL_SCHEMA) {
                    compactCompletedVisual(values, times);
                }
                visualHit = true;
                resumedVisualRows = savedVisualRows;
                return loadedVisual(values, times);
            }
            for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
                File file = chunkFile(chunkIndex);
                try (DataInputStream input = new DataInputStream(new BufferedInputStream(
                        new FileInputStream(file)))) {
                    if (input.readInt() != CHUNK_MAGIC || input.readInt() != FILE_SCHEMA) {
                        throw new IOException("Invalid visual cache chunk header");
                    }
                    int start = input.readInt();
                    int rows = input.readInt();
                    int storedColumns = input.readInt();
                    if (start != expectedStart || rows <= 0 || storedColumns != columns
                            || start + rows > savedVisualRows) {
                        throw new IOException("Invalid visual cache chunk dimensions");
                    }
                    for (int row = 0; row < rows; row++) {
                        int targetRow = start + row;
                        times[targetRow] = input.readDouble();
                        int target = targetRow * columns;
                        for (int column = 0; column < columns; column++) {
                            values[target + column] = input.readFloat();
                        }
                    }
                    if (input.read() != -1) throw new IOException("Trailing visual cache data");
                    expectedStart += rows;
                }
            }
            if (expectedStart != savedVisualRows) {
                throw new IOException("Incomplete visual cache chunk sequence");
            }
            visualHit = visualComplete;
            resumedVisualRows = savedVisualRows;
            if (visualComplete) compactCompletedVisual(values, times);
            return loadedVisual(values, times);
        } catch (IOException | RuntimeException error) {
            failure = "visual read: " + error.getMessage();
            invalidateEntry();
            return LoadedVisual.empty();
        }
    }

    synchronized boolean timesMatch(LoadedVisual loaded, double[] expectedTimes) {
        if (loaded.rows() == 0) return true;
        if (loaded.rows() > expectedTimes.length) {
            invalidateEntry();
            return false;
        }
        for (int row = 0; row < loaded.rows(); row++) {
            if (Math.abs(loaded.times()[row] - expectedTimes[row]) > 1e-9) {
                invalidateEntry();
                return false;
            }
        }
        return true;
    }

    VisualWriter newVisualWriter(LoadedVisual loaded) {
        return new VisualWriter(this, loaded.rows());
    }

    synchronized float[] loadAudio(int expectedRows) {
        if (mode == Mode.BYPASS || !audioComplete || !visualComplete) return null;
        int columns = FeatureSchema.AUDIO.size();
        File file = new File(entryDirectory, AUDIO_NAME);
        try {
            byte[] bytes = Files.readAllBytes(file.toPath());
            int expectedBytes = Integer.BYTES * 4
                    + expectedRows * columns * Float.BYTES;
            if (bytes.length != expectedBytes) {
                throw new IOException("Invalid audio cache size");
            }
            ByteBuffer input = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
            if (input.getInt() != AUDIO_MAGIC || input.getInt() != FILE_SCHEMA
                    || input.getInt() != expectedRows || input.getInt() != columns) {
                throw new IOException("Invalid audio cache dimensions");
            }
            float[] values = new float[expectedRows * columns];
            input.asFloatBuffer().get(values);
            audioHit = true;
            return values;
        } catch (IOException | RuntimeException error) {
            audioComplete = false;
            manifest.setProperty("audioComplete", "false");
            failure = "audio read: " + error.getMessage();
            tryWriteManifest();
            return null;
        }
    }

    synchronized void storeAudio(float[] values, int rows) {
        if (mode == Mode.BYPASS || failure.length() > 0) return;
        int columns = FeatureSchema.AUDIO.size();
        if (values.length != rows * columns) {
            failure = "audio write: matrix dimensions do not match";
            return;
        }
        try {
            ensureEntryDirectory();
            File target = new File(entryDirectory, AUDIO_NAME);
            File temporary = new File(entryDirectory, AUDIO_NAME + ".tmp");
            try (FileOutputStream stream = new FileOutputStream(temporary);
                 DataOutputStream output = new DataOutputStream(new BufferedOutputStream(stream))) {
                output.writeInt(AUDIO_MAGIC);
                output.writeInt(FILE_SCHEMA);
                output.writeInt(rows);
                output.writeInt(columns);
                for (float value : values) output.writeFloat(value);
                output.flush();
                stream.getFD().sync();
            }
            atomicReplace(temporary, target);
            audioComplete = true;
            contextComplete = false;
            manifest.setProperty("audioComplete", "true");
            manifest.setProperty("contextComplete", "false");
            writeManifest();
        } catch (IOException error) {
            failure = "audio write: " + error.getMessage();
        }
    }

    synchronized float[] loadContext(int expectedRows) {
        if (mode == Mode.BYPASS || !visualComplete || !audioComplete || !contextComplete) {
            return null;
        }
        int columns = FeatureSchema.BASE.size() * FeatureSchema.CONTEXT_OFFSETS_SECONDS.length;
        File file = new File(entryDirectory, CONTEXT_NAME);
        try {
            byte[] bytes = Files.readAllBytes(file.toPath());
            int expectedBytes = Integer.BYTES * 4
                    + expectedRows * columns * Float.BYTES;
            if (bytes.length != expectedBytes) {
                throw new IOException("Invalid context cache size");
            }
            ByteBuffer input = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
            if (input.getInt() != CONTEXT_MAGIC || input.getInt() != FILE_SCHEMA
                    || input.getInt() != expectedRows || input.getInt() != columns) {
                throw new IOException("Invalid context cache dimensions");
            }
            float[] values = new float[expectedRows * columns];
            input.asFloatBuffer().get(values);
            contextHit = true;
            return values;
        } catch (IOException | RuntimeException error) {
            contextComplete = false;
            manifest.setProperty("contextComplete", "false");
            failure = "context read: " + error.getMessage();
            tryWriteManifest();
            return null;
        }
    }

    synchronized void storeContext(float[] values, int rows) {
        if (mode == Mode.BYPASS || failure.length() > 0) return;
        int columns = FeatureSchema.BASE.size() * FeatureSchema.CONTEXT_OFFSETS_SECONDS.length;
        if (values.length != rows * columns) {
            failure = "context write: matrix dimensions do not match";
            return;
        }
        try {
            ensureEntryDirectory();
            ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES * 4 + values.length * Float.BYTES)
                    .order(ByteOrder.BIG_ENDIAN);
            buffer.putInt(CONTEXT_MAGIC);
            buffer.putInt(FILE_SCHEMA);
            buffer.putInt(rows);
            buffer.putInt(columns);
            buffer.asFloatBuffer().put(values);
            File target = new File(entryDirectory, CONTEXT_NAME);
            File temporary = new File(entryDirectory, CONTEXT_NAME + ".tmp");
            try (FileOutputStream stream = new FileOutputStream(temporary)) {
                stream.write(buffer.array());
                stream.getFD().sync();
            }
            atomicReplace(temporary, target);
            contextComplete = true;
            manifest.setProperty("contextComplete", "true");
            writeManifest();
        } catch (IOException error) {
            failure = "context write: " + error.getMessage();
        }
    }

    synchronized CacheStats stats() {
        return new CacheStats(
                mode.wireName(), key.substring(0, Math.min(12, key.length())),
                visualHit, audioHit, contextHit, resumedVisualRows, savedVisualRows,
                visualComplete && audioComplete && contextComplete,
                mode == Mode.BYPASS ? 0 : directoryBytes(entryDirectory), failure
        );
    }

    static void clearAll(Context context) {
        deleteRecursively(new File(context.getFilesDir(), CACHE_VERSION));
    }

    static void clearEntry(
            Context context,
            Uri uri,
            String displayName,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            int sourceFrameLimit,
            AnalysisTypes.AnalysisWindow analysisWindow
    ) {
        String key = buildKey(
                context.getApplicationContext(), uri, displayName, media, roi, sourceFrameLimit,
                analysisWindow
        );
        deleteRecursively(new File(new File(context.getFilesDir(), CACHE_VERSION), key));
    }

    static void clearEntryWithSourceMetadata(
            Context context,
            Uri uri,
            String displayName,
            long sourceSize,
            long sourceModified,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            int sourceFrameLimit,
            AnalysisTypes.AnalysisWindow analysisWindow
    ) {
        String key = buildKey(
                context.getApplicationContext(), uri, displayName, media, roi, sourceFrameLimit,
                analysisWindow, sourceSize, sourceModified
        );
        deleteRecursively(new File(new File(context.getFilesDir(), CACHE_VERSION), key));
    }

    static long totalBytes(Context context) {
        return directoryBytes(new File(context.getFilesDir(), CACHE_VERSION));
    }

    private synchronized void appendVisualChunk(
            int startRow,
            double[] times,
            float[] values,
            int rows
    ) throws IOException {
        if (startRow != savedVisualRows || rows <= 0) {
            throw new IOException("Visual cache append is not contiguous");
        }
        int columns = FeatureSchema.FRAME.size();
        ensureEntryDirectory();
        File target = chunkFile(chunkCount);
        File temporary = new File(entryDirectory, target.getName() + ".tmp");
        try (FileOutputStream stream = new FileOutputStream(temporary);
             DataOutputStream output = new DataOutputStream(new BufferedOutputStream(stream))) {
            output.writeInt(CHUNK_MAGIC);
            output.writeInt(FILE_SCHEMA);
            output.writeInt(startRow);
            output.writeInt(rows);
            output.writeInt(columns);
            for (int row = 0; row < rows; row++) {
                output.writeDouble(times[row]);
                int source = row * columns;
                for (int column = 0; column < columns; column++) {
                    output.writeFloat(values[source + column]);
                }
            }
            output.flush();
            stream.getFD().sync();
        }
        atomicReplace(temporary, target);
        savedVisualRows += rows;
        chunkCount++;
        visualComplete = false;
        audioComplete = false;
        contextComplete = false;
        manifest.setProperty("schema", Integer.toString(FILE_SCHEMA));
        manifest.setProperty("key", key);
        manifest.setProperty("columns", Integer.toString(columns));
        manifest.setProperty("rowCount", Integer.toString(savedVisualRows));
        manifest.setProperty("chunkCount", Integer.toString(chunkCount));
        manifest.setProperty("visualComplete", "false");
        manifest.setProperty("audioComplete", "false");
        manifest.setProperty("contextComplete", "false");
        writeManifest();
    }

    private synchronized void markVisualComplete(AnalysisTypes.VideoFeatures video)
            throws IOException {
        if (savedVisualRows != video.analysisTimes().length) {
            throw new IOException("Cannot complete a partial visual feature cache");
        }
        writeCompleteVisual(video.values(), video.analysisTimes());
        visualComplete = true;
        manifest.setProperty("visualComplete", "true");
        manifest.setProperty("analyzedDurationSeconds",
                Double.toString(video.analyzedDurationSeconds()));
        manifest.setProperty("sourceFrameLimitReached",
                Boolean.toString(video.sourceFrameLimitReached()));
        manifest.setProperty("decodedSourceFrames", Integer.toString(video.decodedSourceFrames()));
        manifest.setProperty("decodeOnlySourceFrames",
                Integer.toString(video.decodeOnlySourceFrames()));
        manifest.setProperty("decoderOutputFrames", Integer.toString(video.decoderOutputFrames()));
        manifest.setProperty("decoderName", video.decoderName());
        manifest.setProperty("hardwareDecoder", Boolean.toString(video.hardwareDecoder()));
        writeManifest();
        deleteVisualChunks();
    }

    private LoadedVisual loadedVisual(float[] values, double[] times) {
        return new LoadedVisual(
                values,
                times,
                savedVisualRows,
                visualComplete,
                doubleProperty("analyzedDurationSeconds", 0),
                booleanProperty("sourceFrameLimitReached", false),
                intProperty("decodedSourceFrames", 0),
                intProperty("decodeOnlySourceFrames", 0),
                intProperty("decoderOutputFrames", 0),
                manifest.getProperty("decoderName", ""),
                booleanProperty("hardwareDecoder", false)
        );
    }

    private void compactCompletedVisual(float[] values, double[] times) {
        try {
            writeCompleteVisual(values, times);
            deleteVisualChunks();
        } catch (IOException ignored) {
            // The validated chunks remain the authoritative fallback.
        }
    }

    private void writeCompleteVisual(float[] values, double[] times) throws IOException {
        int columns = FeatureSchema.FRAME.size();
        if (times.length != savedVisualRows || values.length != savedVisualRows * columns) {
            throw new IOException("Completed visual cache dimensions do not match");
        }
        int bytesPerRow = Double.BYTES + columns * Float.BYTES;
        ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES * 4 + savedVisualRows * bytesPerRow)
                .order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(COMPLETE_VISUAL_MAGIC);
        buffer.putInt(COMPLETE_VISUAL_SCHEMA);
        buffer.putInt(savedVisualRows);
        buffer.putInt(columns);
        for (int row = 0; row < savedVisualRows; row++) {
            buffer.putDouble(times[row]);
        }
        buffer.asFloatBuffer().put(values);
        File target = new File(entryDirectory, COMPLETE_VISUAL_NAME);
        File temporary = new File(entryDirectory, COMPLETE_VISUAL_NAME + ".tmp");
        try (FileOutputStream stream = new FileOutputStream(temporary)) {
            stream.write(buffer.array());
            stream.getFD().sync();
        }
        atomicReplace(temporary, target);
    }

    private static int readCompleteVisual(
            File file,
            float[] values,
            double[] times,
            int rows,
            int columns
    ) throws IOException {
        byte[] bytes = Files.readAllBytes(file.toPath());
        int expectedBytes = Integer.BYTES * 4
                + rows * (Double.BYTES + columns * Float.BYTES);
        if (bytes.length != expectedBytes) {
            throw new IOException("Completed visual cache size does not match");
        }
        ByteBuffer input = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
        if (input.getInt() != COMPLETE_VISUAL_MAGIC) {
            throw new IOException("Invalid completed visual cache header");
        }
        int schema = input.getInt();
        if ((schema != FILE_SCHEMA && schema != COMPLETE_VISUAL_SCHEMA)
                || input.getInt() != rows || input.getInt() != columns) {
            throw new IOException("Invalid completed visual cache dimensions");
        }
        if (schema == FILE_SCHEMA) {
            for (int row = 0; row < rows; row++) {
                times[row] = input.getDouble();
                int target = row * columns;
                for (int column = 0; column < columns; column++) {
                    values[target + column] = input.getFloat();
                }
            }
        } else {
            for (int row = 0; row < rows; row++) times[row] = input.getDouble();
            input.asFloatBuffer().get(values);
        }
        return schema;
    }

    private void deleteVisualChunks() {
        for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
            chunkFile(chunkIndex).delete();
        }
    }

    private void loadManifest() {
        File file = new File(entryDirectory, MANIFEST_NAME);
        if (!file.isFile()) return;
        try (FileInputStream input = new FileInputStream(file)) {
            manifest.load(input);
            if (intProperty("schema", -1) != FILE_SCHEMA
                    || !key.equals(manifest.getProperty("key"))
                    || intProperty("columns", -1) != FeatureSchema.FRAME.size()) {
                throw new IOException("Manifest identity does not match");
            }
            savedVisualRows = intProperty("rowCount", 0);
            chunkCount = intProperty("chunkCount", 0);
            if (savedVisualRows < 0 || savedVisualRows > maximumRows || chunkCount < 0) {
                throw new IOException("Manifest dimensions are out of range");
            }
            visualComplete = booleanProperty("visualComplete", false);
            audioComplete = booleanProperty("audioComplete", false);
            contextComplete = booleanProperty("contextComplete", false);
        } catch (IOException | RuntimeException error) {
            failure = "manifest read: " + error.getMessage();
            invalidateEntry();
        }
    }

    private void invalidateEntry() {
        deleteRecursively(entryDirectory);
        manifest.clear();
        savedVisualRows = 0;
        chunkCount = 0;
        visualComplete = false;
        audioComplete = false;
        contextComplete = false;
        visualHit = false;
        audioHit = false;
        contextHit = false;
        resumedVisualRows = 0;
    }

    private void ensureEntryDirectory() throws IOException {
        if (!entryDirectory.isDirectory() && !entryDirectory.mkdirs()) {
            throw new IOException("Could not create feature cache directory");
        }
    }

    private void writeManifest() throws IOException {
        ensureEntryDirectory();
        File target = new File(entryDirectory, MANIFEST_NAME);
        File temporary = new File(entryDirectory, MANIFEST_NAME + ".tmp");
        try (FileOutputStream stream = new FileOutputStream(temporary)) {
            manifest.store(stream, "VolleySplice native feature cache");
            stream.getFD().sync();
        }
        atomicReplace(temporary, target);
    }

    private void tryWriteManifest() {
        try {
            writeManifest();
        } catch (IOException ignored) {}
    }

    private File chunkFile(int chunkIndex) {
        return new File(entryDirectory, String.format(Locale.US, "visual-%06d.bin", chunkIndex));
    }

    private int intProperty(String name, int fallback) {
        try {
            return Integer.parseInt(manifest.getProperty(name, Integer.toString(fallback)));
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private double doubleProperty(String name, double fallback) {
        try {
            return Double.parseDouble(manifest.getProperty(name, Double.toString(fallback)));
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private boolean booleanProperty(String name, boolean fallback) {
        return Boolean.parseBoolean(manifest.getProperty(name, Boolean.toString(fallback)));
    }

    private static String buildKey(
            Context context,
            Uri uri,
            String displayName,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            int sourceFrameLimit,
            AnalysisTypes.AnalysisWindow requestedWindow
    ) {
        return buildKey(
                context, uri, displayName, media, roi, sourceFrameLimit, requestedWindow,
                null, null
        );
    }

    private static String buildKey(
            Context context,
            Uri uri,
            String displayName,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            int sourceFrameLimit,
            AnalysisTypes.AnalysisWindow requestedWindow,
            Long knownSourceSize,
            Long knownSourceModified
    ) {
        long sourceSize = knownSourceSize == null ? -1 : knownSourceSize;
        long sourceModified = knownSourceModified == null ? -1 : knownSourceModified;
        if (knownSourceSize == null || knownSourceModified == null) {
            try (Cursor cursor = context.getContentResolver().query(uri, null, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    int sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE);
                    if (knownSourceSize == null && sizeIndex >= 0 && !cursor.isNull(sizeIndex)) {
                        sourceSize = cursor.getLong(sizeIndex);
                    }
                    int modifiedIndex = cursor.getColumnIndex("last_modified");
                    if (modifiedIndex < 0) modifiedIndex = cursor.getColumnIndex("date_modified");
                    if (knownSourceModified == null && modifiedIndex >= 0 && !cursor.isNull(modifiedIndex)) {
                        sourceModified = cursor.getLong(modifiedIndex);
                    }
                }
            } catch (RuntimeException ignored) {}
        }
        AnalysisTypes.AnalysisWindow analysisWindow = AnalysisTypes.AnalysisWindow.normalize(
                requestedWindow, media.durationSeconds()
        );
        String identity = String.join("\n",
                CACHE_VERSION,
                "visual-extractor=opencv-v1-yuv-lut",
                "audio-extractor=native-dsp-v1",
                "frame=" + String.join(",", FeatureSchema.FRAME),
                "audio=" + String.join(",", FeatureSchema.AUDIO),
                "fps=" + FeatureSchema.ANALYSIS_FPS,
                "sourceUri=" + uri,
                "sourceName=" + displayName,
                "sourceSize=" + sourceSize,
                "sourceModified=" + sourceModified,
                "duration=" + media.durationSeconds(),
                "width=" + media.width(),
                "height=" + media.height(),
                "rotation=" + media.rotation(),
                "videoMime=" + media.videoMime(),
                "audioMime=" + media.audioMime(),
                "roi=" + roi.x() + "," + roi.y() + "," + roi.width() + "," + roi.height(),
                "sourceFrameLimit=" + sourceFrameLimit
        ) + (analysisWindow.isFull(media.durationSeconds()) ? "" :
                "\nanalysisWindow=" + analysisWindow.start() + "," + analysisWindow.end());
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(identity.getBytes(StandardCharsets.UTF_8));
            StringBuilder output = new StringBuilder(digest.length * 2);
            for (byte value : digest) output.append(String.format(Locale.US, "%02x", value & 0xff));
            return output.toString();
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private static void atomicReplace(File source, File target) throws IOException {
        try {
            Files.move(
                    source.toPath(), target.toPath(),
                    StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE
            );
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static long directoryBytes(File file) {
        if (!file.exists()) return 0;
        if (file.isFile()) return file.length();
        long total = 0;
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) total += directoryBytes(child);
        }
        return total;
    }

    private static void deleteRecursively(File file) {
        if (!file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) deleteRecursively(child);
            }
        }
        // Internal app storage only; a failed deletion simply makes the next write fail closed.
        file.delete();
    }

    static final class VisualWriter {
        private final NativeFeatureCache cache;
        private final int columns = FeatureSchema.FRAME.size();
        private final double[] times = new double[CHUNK_ROWS];
        private final float[] values = new float[CHUNK_ROWS * FeatureSchema.FRAME.size()];
        private int startRow;
        private int bufferedRows;
        private boolean enabled;
        private long writeNanos;

        VisualWriter(NativeFeatureCache cache, int startRow) {
            this.cache = cache;
            this.startRow = startRow;
            enabled = cache.mode != Mode.BYPASS && cache.failure.length() == 0;
        }

        synchronized void append(int rowIndex, double time, float[] row) {
            if (!enabled) return;
            if (rowIndex < startRow) return;
            if (rowIndex != startRow + bufferedRows || row.length != columns) {
                fail("visual write: rows are not contiguous");
                return;
            }
            times[bufferedRows] = time;
            System.arraycopy(row, 0, values, bufferedRows * columns, columns);
            bufferedRows++;
            if (bufferedRows == CHUNK_ROWS) flush();
        }

        synchronized void checkpoint() {
            if (enabled && bufferedRows > 0) flush();
        }

        synchronized void finish(AnalysisTypes.VideoFeatures video) {
            if (!enabled) return;
            checkpoint();
            if (!enabled) return;
            long started = System.nanoTime();
            try {
                cache.markVisualComplete(video);
            } catch (IOException error) {
                fail("visual complete: " + error.getMessage());
            } finally {
                writeNanos += System.nanoTime() - started;
            }
        }

        synchronized double writeMilliseconds() {
            return writeNanos / 1_000_000.0;
        }

        private void flush() {
            int rows = bufferedRows;
            long started = System.nanoTime();
            try {
                cache.appendVisualChunk(
                        startRow,
                        Arrays.copyOf(times, rows),
                        Arrays.copyOf(values, rows * columns),
                        rows
                );
                startRow += rows;
                bufferedRows = 0;
            } catch (IOException error) {
                fail("visual write: " + error.getMessage());
            } finally {
                writeNanos += System.nanoTime() - started;
            }
        }

        private void fail(String message) {
            enabled = false;
            cache.failure = message;
        }
    }
}
