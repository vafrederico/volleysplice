package com.volleycut.nativeanalysis

/** One feature budget for fixed-flight features followed by optional neural context. */
internal class NeuralServingProgress(private val downstream: AnalysisTypes.ProgressListener) {
    val specialist: AnalysisTypes.ProgressListener = object : AnalysisTypes.ProgressListener {
        override fun onProgress(stage: String, fraction: Double, detail: String) {
            when (stage) {
                "serving-side-features" -> downstream.onProgress(stage, fraction.coerceIn(0.0, 1.0) * 0.5, detail)
                // These preliminary verdicts are inputs to the neural head, not completion.
                "serving-side" -> downstream.onProgress("serving-side-features", 0.5, detail)
                else -> downstream.onProgress(stage, fraction, detail)
            }
        }

        override fun onPerformance(stats: AnalysisTypes.PerformanceStats) = downstream.onPerformance(stats)
    }

    val neural: AnalysisTypes.ProgressListener = object : AnalysisTypes.ProgressListener {
        override fun onProgress(stage: String, fraction: Double, detail: String) {
            downstream.onProgress(stage, if (stage == "serving-side-features") 0.5 + fraction.coerceIn(0.0, 1.0) * 0.5 else fraction, detail)
        }

        override fun onPerformance(stats: AnalysisTypes.PerformanceStats) = downstream.onPerformance(stats)
    }
}
