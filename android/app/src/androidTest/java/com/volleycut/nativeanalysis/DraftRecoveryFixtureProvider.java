package com.volleycut.nativeanalysis;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Test APK only: deterministic provider failures and a slow read, without app dependencies. */
public class DraftRecoveryFixtureProvider extends ContentProvider {
    private volatile CountDownLatch readGate = new CountDownLatch(1);

    @Override public boolean onCreate() { return true; }
    @Override public String getType(Uri uri) { return "application/json"; }
    @Override public Cursor query(Uri uri, String[] projection, String selection,
            String[] selectionArgs, String sortOrder) {
        MatrixCursor cursor = new MatrixCursor(new String[] {OpenableColumns.DISPLAY_NAME});
        cursor.addRow(new Object[] {uri.getLastPathSegment() + ".json"});
        return cursor;
    }

    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        if ("unreadable".equals(uri.getLastPathSegment())) {
            throw new FileNotFoundException("Synthetic provider read failure");
        }
        String content = "unsupported".equals(uri.getLastPathSegment())
            ? "{\"schema\":\"unsupported-backup\",\"schemaVersion\":999}" : "{\"schema\":";
        try {
            ParcelFileDescriptor[] pipe = ParcelFileDescriptor.createPipe();
            new Thread(() -> {
                try (var stream = new ParcelFileDescriptor.AutoCloseOutputStream(pipe[1])) {
                    if ("slow".equals(uri.getLastPathSegment())) readGate.await(15, TimeUnit.SECONDS);
                    stream.write(content.getBytes(StandardCharsets.UTF_8));
                } catch (IOException ignored) {
                    // The client may cancel and close its read side.
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                }
            }).start();
            return pipe[0];
        } catch (IOException error) {
            throw new FileNotFoundException(error.getMessage());
        }
    }

    @Override public Bundle call(String method, String arg, Bundle extras) {
        if ("reset".equals(method)) readGate = new CountDownLatch(1);
        if ("release".equals(method)) readGate.countDown();
        return Bundle.EMPTY;
    }
    @Override public Uri insert(Uri uri, ContentValues values) { return null; }
    @Override public int delete(Uri uri, String selection, String[] selectionArgs) { return 0; }
    @Override public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) { return 0; }
}
