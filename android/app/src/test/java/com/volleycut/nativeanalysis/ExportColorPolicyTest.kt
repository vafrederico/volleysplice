package com.volleycut.nativeanalysis

import androidx.media3.common.C
import androidx.media3.common.ColorInfo
import androidx.media3.common.MimeTypes
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ExportColorPolicyTest {
    @Test fun hdrRequiresHevcAndRejectsChangedTransferPrimariesRangeOrEightBitOutput() {
        for (transfer in listOf(C.COLOR_TRANSFER_HLG, C.COLOR_TRANSFER_ST2084)) {
            val color = ColorInfo.Builder().setColorSpace(C.COLOR_SPACE_BT2020)
                .setColorTransfer(transfer).setColorRange(C.COLOR_RANGE_LIMITED)
                .setLumaBitdepth(10).setChromaBitdepth(10).build()
            val policy = ExportColorPolicy(color)
            assertTrue(policy.isHdr)
            assertEquals(MimeTypes.VIDEO_H265, policy.outputMimeType)
            assertTrue(policy.matches(color))
            assertFalse(policy.matches(null))
            assertFalse(policy.matches(ColorInfo.SDR_BT709_LIMITED))
            assertFalse(policy.matches(color.buildUpon().setColorSpace(C.COLOR_SPACE_BT709).build()))
            assertFalse(policy.matches(color.buildUpon().setColorRange(C.COLOR_RANGE_FULL).build()))
            assertFalse(policy.matches(color.buildUpon().setLumaBitdepth(8).build()))
            assertFalse(policy.matches(color.buildUpon().setColorTransfer(
                if (transfer == C.COLOR_TRANSFER_HLG) C.COLOR_TRANSFER_ST2084 else C.COLOR_TRANSFER_HLG).build()))
        }
    }

    @Test fun sdrUsesH264WithoutChangingColorMetadata() {
        val policy = ExportColorPolicy(ColorInfo.SDR_BT709_LIMITED)
        assertFalse(policy.isHdr)
        assertEquals(MimeTypes.VIDEO_H264, policy.outputMimeType)
        assertTrue(policy.matches(ColorInfo.SDR_BT709_LIMITED))
    }
}
