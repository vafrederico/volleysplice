package com.volleycut.nativeanalysis

import java.security.MessageDigest

internal const val SERVING_SIDE_DECODE_VARIANT = "native-mediacodec-yuv-rgb-gray-v1"

internal data class ServingSideCacheIdentity(
    val sourceUri: String,
    val sourceName: String,
    val sourceSize: Long,
    val sourceLastModified: Long,
    val sourceSampledFingerprint: String?,
    val durationSeconds: Double,
    val width: Int,
    val height: Int,
    val rotation: Int,
    val roiX: Double,
    val roiY: Double,
    val roiWidth: Double,
    val roiHeight: Double,
    val analysisStart: Double,
    val analysisEnd: Double,
    val decodeVariant: String,
    val featureSignature: String,
    val candidateSignature: String,
)

internal object ServingSideCache {
    fun identity(project: NativeProject): ServingSideCacheIdentity = ServingSideCacheIdentity(
        project.source.uri,
        project.source.name,
        project.source.size,
        project.source.lastModified,
        project.source.sampledFingerprint,
        project.media.durationSeconds(),
        project.media.width(), project.media.height(), project.media.rotation(),
        project.roi.x(), project.roi.y(), project.roi.width(), project.roi.height(),
        project.analysisWindow.start(), project.analysisWindow.end(),
        SERVING_SIDE_DECODE_VARIANT,
        signature(ServingSideFeatureNames.all()),
        signature(ServingSideModelRunner.candidates(project.ranges).map {
            "${it.id}|${it.start}|${it.end}|${it.agreement.orEmpty()}"
        }),
    )

    fun isReusable(
        cachedIdentity: ServingSideCacheIdentity,
        currentProject: NativeProject,
        output: ServingSideOutput,
    ): Boolean = cachedIdentity == identity(currentProject) && output.isReusableFor(currentProject.ranges)

    private fun signature(values: List<String>): String = MessageDigest.getInstance("SHA-256")
        .digest(values.joinToString("\u0000").toByteArray())
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
