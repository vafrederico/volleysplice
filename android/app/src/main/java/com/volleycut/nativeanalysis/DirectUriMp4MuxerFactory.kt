package com.volleycut.nativeanalysis

import android.content.Context
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.system.Os
import android.system.OsConstants
import android.util.Log
import androidx.annotation.OptIn
import androidx.media3.common.Format
import androidx.media3.common.Metadata
import androidx.media3.common.MimeTypes
import androidx.media3.common.util.UnstableApi
import androidx.media3.container.Mp4OrientationData
import androidx.media3.muxer.BufferInfo
import androidx.media3.muxer.Mp4Muxer
import androidx.media3.muxer.Muxer
import androidx.media3.muxer.MuxerException
import androidx.media3.muxer.MuxerUtil
import androidx.media3.muxer.SeekableMuxerOutput
import androidx.media3.transformer.InAppMp4Muxer
import java.io.FileOutputStream
import java.nio.ByteBuffer

/** Writes Transformer's MP4 output directly into a seekable Storage Access Framework document. */
@OptIn(markerClass = [UnstableApi::class])
internal class DirectUriMp4MuxerFactory(
    context: Context,
    private val destination: Uri,
) : Muxer.Factory {
    private val resolver = context.applicationContext.contentResolver
    private val capabilities = InAppMp4Muxer.Factory()

    /** Probes the document provider before Transformer starts so streaming-only providers can fall back. */
    fun isSeekable(): Boolean {
        val descriptor = runCatching { openTruncatedDescriptor() }.getOrNull() ?: return false
        return try {
            Os.lseek(descriptor.fileDescriptor, 0, OsConstants.SEEK_SET)
            Os.ftruncate(descriptor.fileDescriptor, 0)
            true
        } catch (_: Exception) {
            false
        } finally {
            runCatching { descriptor.close() }
        }
    }

    override fun create(path: String): Muxer {
        val descriptor = try {
            openTruncatedDescriptor()
        } catch (error: Exception) {
            throw MuxerException("Could not open selected export document", error)
        }
        try {
            Os.lseek(descriptor.fileDescriptor, 0, OsConstants.SEEK_SET)
            Os.ftruncate(descriptor.fileDescriptor, 0)
            val stream: FileOutputStream = ParcelFileDescriptor.AutoCloseOutputStream(descriptor)
            val mp4Muxer = Mp4Muxer.Builder(SeekableMuxerOutput.of(stream)).build()
            return FilteringMp4Muxer(mp4Muxer)
        } catch (error: Exception) {
            runCatching { descriptor.close() }
            throw MuxerException("Selected export document is not seekable", error)
        }
    }

    override fun getSupportedSampleMimeTypes(trackType: Int) =
        capabilities.getSupportedSampleMimeTypes(trackType)

    override fun supportsWritingNegativeTimestampsInEditList(): Boolean =
        capabilities.supportsWritingNegativeTimestampsInEditList()

    private fun openTruncatedDescriptor(): ParcelFileDescriptor {
        var failure: Throwable? = null
        for (mode in listOf("rwt", "rw", "wt", "w")) {
            try {
                return resolver.openFileDescriptor(destination, mode)
                    ?: throw IllegalStateException("Document provider returned no file descriptor")
            } catch (error: Exception) {
                if (failure == null) failure = error
            }
        }
        throw IllegalStateException("Document provider cannot open the destination for writing", failure)
    }

    /** Mirrors Media3's InAppMp4Muxer behavior for a caller-owned output stream. */
    private class FilteringMp4Muxer(private val delegate: Mp4Muxer) : Muxer {
        private val metadataEntries = linkedSetOf<Metadata.Entry>()

        override fun addTrack(format: Format): Int {
            val trackId = delegate.addTrack(format)
            if (MimeTypes.isVideo(format.sampleMimeType)) {
                delegate.addMetadataEntry(Mp4OrientationData(format.rotationDegrees))
            }
            return trackId
        }

        override fun writeSampleData(trackId: Int, byteBuffer: ByteBuffer, bufferInfo: BufferInfo) {
            delegate.writeSampleData(trackId, byteBuffer, bufferInfo)
        }

        override fun addMetadataEntry(metadataEntry: Metadata.Entry) {
            if (MuxerUtil.isMetadataSupported(metadataEntry)) {
                metadataEntries += metadataEntry
            } else {
                Log.d(TAG, "Skipping unsupported MP4 metadata ${metadataEntry.javaClass.name}")
            }
        }

        override fun close() {
            metadataEntries.forEach(delegate::addMetadataEntry)
            delegate.close()
        }
    }

    private companion object {
        const val TAG = "VolleySpliceMuxer"
    }
}
