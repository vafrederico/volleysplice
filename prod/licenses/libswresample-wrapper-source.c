#include <errno.h>
#include <stdint.h>

#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>

static SwrContext *context = NULL;

int vc_swr_init(int input_sample_rate, int input_channels) {
    AVChannelLayout input_layout;
    AVChannelLayout output_layout;
    int result;

    if (input_sample_rate <= 0 || input_channels <= 0 || input_channels > 2) {
        return AVERROR(EINVAL);
    }
    swr_free(&context);
    av_channel_layout_default(&input_layout, input_channels);
    av_channel_layout_default(&output_layout, 1);
    result = swr_alloc_set_opts2(
        &context,
        &output_layout,
        AV_SAMPLE_FMT_S16,
        16000,
        &input_layout,
        AV_SAMPLE_FMT_FLTP,
        input_sample_rate,
        0,
        NULL
    );
    av_channel_layout_uninit(&input_layout);
    av_channel_layout_uninit(&output_layout);
    if (result < 0) {
        swr_free(&context);
        return result;
    }
    result = swr_init(context);
    if (result < 0) {
        swr_free(&context);
    }
    return result;
}

int vc_swr_get_out_samples(int input_frames) {
    if (context == NULL || input_frames < 0) {
        return AVERROR(EINVAL);
    }
    return swr_get_out_samples(context, input_frames);
}

int vc_swr_convert(
    const float *input_plane_0,
    const float *input_plane_1,
    int input_frames,
    int16_t *output,
    int output_capacity
) {
    const uint8_t *input_planes[2];
    uint8_t *output_planes[1];

    if (context == NULL || input_frames < 0 || output_capacity < 0) {
        return AVERROR(EINVAL);
    }
    input_planes[0] = (const uint8_t *)input_plane_0;
    input_planes[1] = (const uint8_t *)input_plane_1;
    output_planes[0] = (uint8_t *)output;
    return swr_convert(
        context,
        output_planes,
        output_capacity,
        input_plane_0 == NULL ? NULL : input_planes,
        input_frames
    );
}

void vc_swr_close(void) {
    swr_free(&context);
}
