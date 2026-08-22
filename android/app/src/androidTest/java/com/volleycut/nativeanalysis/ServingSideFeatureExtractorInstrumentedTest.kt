package com.volleycut.nativeanalysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader

@RunWith(AndroidJUnit4::class)
class ServingSideFeatureExtractorInstrumentedTest {
    @Test
    fun staticGrayFixtureProducesFrozenCourtAndFlightBanks() {
        val frame = ByteArray(SERVING_SIDE_WIDTH * SERVING_SIDE_HEIGHT) { 73 }
        val court = ServingSideFeatureExtractor.extractCourtFlowRawFeatures(
            Array(COURT_FLOW_OFFSETS_SECONDS.size) { frame.clone() },
        )
        val flight = ServingSideFeatureExtractor.extractFlightRawFeatures(
            Array(FLIGHT_OFFSETS_SECONDS.size) { frame.clone() },
        )
        assertArrayEquals(DoubleArray(82), court, 0.0)
        assertEquals(155, flight.size)
        val expectedFlight = DoubleArray(155)
        repeat(3) { phase ->
            val phaseStart = phase * 43
            expectedFlight[phaseStart + 30] = .5 // centroidX
            expectedFlight[phaseStart + 31] = .5 // centroidY
        }
        assertArrayEquals(expectedFlight, flight, 1e-12)
        assertArrayEquals(
            expectedFlight,
            ServingSideFeatureExtractor.extract(
                Array(COURT_FLOW_OFFSETS_SECONDS.size) { frame.clone() },
                Array(FLIGHT_OFFSETS_SECONDS.size) { frame.clone() },
            ).copyOfRange(82, 237),
            1e-12,
        )
    }

    companion object {
        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            check(OpenCVLoader.initLocal()) { "OpenCV could not initialize" }
        }
    }
}
