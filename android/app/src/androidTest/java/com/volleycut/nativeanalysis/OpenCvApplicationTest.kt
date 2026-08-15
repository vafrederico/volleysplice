package com.volleycut.nativeanalysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat

@RunWith(AndroidJUnit4::class)
class OpenCvApplicationTest {
    @Test
    fun applicationLoadsNativeOpenCvRuntime() {
        val matrix = Mat(2, 2, CvType.CV_8UC1)
        try {
            assertFalse(matrix.empty())
        } finally {
            matrix.release()
        }
    }
}
