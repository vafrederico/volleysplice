package com.volleycut.nativeanalysis

import android.content.Context
import androidx.annotation.OptIn
import androidx.media3.common.util.Clock
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.mediacodec.MediaCodecSelector
import androidx.media3.transformer.DefaultAssetLoaderFactory
import androidx.media3.transformer.DefaultDecoderFactory
import androidx.media3.transformer.Transformer
import androidx.test.platform.app.InstrumentationRegistry

/** Optional test-only workaround for emulator host-decoder failures; production uses device defaults. */
@OptIn(UnstableApi::class)
internal fun Transformer.Builder.withInstrumentedDecoder(context: Context): Transformer.Builder {
    if (InstrumentationRegistry.getArguments().getString("softwareDecoder") != "true") return this
    val decoder = DefaultDecoderFactory.Builder(context)
        .setMediaCodecSelector { mime, secure, tunneling ->
            MediaCodecSelector.DEFAULT.getDecoderInfos(mime, secure, tunneling).filter { it.softwareOnly }
        }.build()
    return setAssetLoaderFactory(DefaultAssetLoaderFactory(context, decoder, Clock.DEFAULT, null))
}
