package com.volleycut.nativeanalysis

import android.content.Context
import android.opengl.GLES20
import androidx.annotation.OptIn
import androidx.media3.common.C
import androidx.media3.common.VideoFrameProcessingException
import androidx.media3.common.util.GlProgram
import androidx.media3.common.util.GlUtil
import androidx.media3.common.util.Size
import androidx.media3.common.util.UnstableApi
import androidx.media3.effect.BaseGlShaderProgram
import androidx.media3.effect.GlEffect
import androidx.media3.effect.GlShaderProgram

/** The Canvas is sRGB artwork; HDR video stays in Media3's high precision BT.2020 pipeline. */
@OptIn(UnstableApi::class)
internal class ScoreOverlayEffect(
    private val overlay: ScoreCanvasOverlay,
    private val sourceTransfer: Int,
) : GlEffect {
    override fun toGlShaderProgram(context: Context, useHdr: Boolean): GlShaderProgram {
        if (!useHdr) {
            check(sourceTransfer != C.COLOR_TRANSFER_HLG && sourceTransfer != C.COLOR_TRANSFER_ST2084) {
                "This device cannot export this video in its original HDR color space."
            }
            return ScoreShaderProgram(overlay, isHlg = false, useHdr = false)
        }
        check(sourceTransfer == C.COLOR_TRANSFER_HLG || sourceTransfer == C.COLOR_TRANSFER_ST2084)
        return ScoreShaderProgram(overlay, sourceTransfer == C.COLOR_TRANSFER_HLG, useHdr = true)
    }
}

@OptIn(UnstableApi::class)
internal class ScoreShaderProgram(
    private val overlay: ScoreCanvasOverlay,
    private val isHlg: Boolean,
    private val useHdr: Boolean,
) : BaseGlShaderProgram(useHdr, 1) {
    private val program = GlProgram(VERTEX_SHADER, FRAGMENT_SHADER).apply {
        setBufferAttribute("aFramePosition", GlUtil.getNormalizedCoordinateBounds(), 4)
    }

    override fun configure(inputWidth: Int, inputHeight: Int): Size =
        Size(inputWidth, inputHeight).also(overlay::configure)

    override fun drawFrame(inputTexId: Int, presentationTimeUs: Long) {
        try {
            program.use()
            program.setSamplerTexIdUniform("uVideo", inputTexId, 0)
            program.setSamplerTexIdUniform("uScore", overlay.getTextureId(presentationTimeUs), 1)
            program.setIntUniform("uHlg", if (isHlg) 1 else 0)
            program.setIntUniform("uHdr", if (useHdr) 1 else 0)
            program.bindAttributesAndUniforms()
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
            GlUtil.checkGlError()
        } catch (error: GlUtil.GlException) {
            throw VideoFrameProcessingException(error, presentationTimeUs)
        }
    }

    override fun release() {
        try {
            super.release()
            overlay.release()
        } finally {
            program.delete()
        }
    }

    private companion object {
        const val VERTEX_SHADER = """
            attribute vec4 aFramePosition;
            varying vec2 vUv;
            void main() {
                gl_Position = aFramePosition;
                vUv = aFramePosition.xy * 0.5 + 0.5;
            }
        """
        // Media3 1.10.1's PQ intermediate is display light in units of 1000 nits;
        // its HLG intermediate is scene light, before the BT.2100 reference OOTF.
        // Composite at 203-nit graphics white in display light, then undo the HLG
        // OOTF so Media3 can apply the original transfer function on output.
        const val FRAGMENT_SHADER = """
            precision highp float;
            uniform sampler2D uVideo;
            uniform sampler2D uScore;
            uniform int uHlg;
            uniform int uHdr;
            varying vec2 vUv;
            const vec3 Y2020 = vec3(0.2627, 0.6780, 0.0593);
            float srgbLinear(float c) {
                return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
            }
            void main() {
                vec4 video = texture2D(uVideo, vUv);
                vec4 score = texture2D(uScore, vec2(vUv.x, 1.0 - vUv.y));
                if (score.a <= 0.0) {
                    gl_FragColor = video;
                    return;
                }
                // Media3's default SDR working space retains electrical video
                // values. Canvas already premultiplied RGB: apply alpha once.
                if (uHdr == 0) {
                    gl_FragColor = vec4(score.rgb + video.rgb * (1.0 - score.a),
                        score.a + video.a * (1.0 - score.a));
                    return;
                }
                // Android Canvas textures contain premultiplied electrical sRGB.
                vec3 rgb = clamp(score.rgb / score.a, 0.0, 1.0);
                rgb = vec3(srgbLinear(rgb.r), srgbLinear(rgb.g), srgbLinear(rgb.b));
                mat3 bt709To2020 = mat3(
                    0.6274040, 0.0690970, 0.0163916,
                    0.3292820, 0.9195400, 0.0880132,
                    0.0433136, 0.0113612, 0.8955950);
                vec3 graphics = bt709To2020 * rgb * 0.203;
                vec3 display = video.rgb;
                if (uHlg == 1) {
                    float luminance = max(dot(display, Y2020), 0.0);
                    display *= pow(luminance, 0.2);
                }
                vec3 result = mix(display, graphics, score.a);
                if (uHlg == 1) {
                    float luminance = max(dot(result, Y2020), 0.0);
                    result = luminance > 0.0 ? result * pow(luminance, -1.0 / 6.0) : vec3(0.0);
                }
                gl_FragColor = vec4(result, score.a + video.a * (1.0 - score.a));
            }
        """
    }
}
