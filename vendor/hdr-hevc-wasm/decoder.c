/* MIT-licensed wrapper; linked FFmpeg libraries are LGPL-2.1-or-later. */
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <math.h>
#include <string.h>

static AVCodecContext *decoder;
static AVFrame *frame;
static AVPacket *packet;

void vc_hdr_close(void) {
    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_free_context(&decoder);
}

int vc_hdr_open(const unsigned char *extra, int length) {
    vc_hdr_close();
    if (length <= 0 || length > 16 * 1024 * 1024) return AVERROR(EINVAL);
    decoder = avcodec_alloc_context3(avcodec_find_decoder(AV_CODEC_ID_HEVC));
    frame = av_frame_alloc();
    packet = av_packet_alloc();
    if (!decoder || !frame || !packet) return AVERROR(ENOMEM);
    decoder->extradata = av_mallocz(length + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!decoder->extradata) return AVERROR(ENOMEM);
    memcpy(decoder->extradata, extra, length);
    decoder->extradata_size = length;
    decoder->pkt_timebase = (AVRational){1, 1000000};
    decoder->thread_count = 1;
    decoder->max_pixels = 8192LL * 4320;
    decoder->err_recognition = AV_EF_CAREFUL | AV_EF_EXPLODE;
    return avcodec_open2(decoder, NULL, NULL);
}

void vc_hdr_reset(void) {
    if (decoder) avcodec_flush_buffers(decoder);
    if (frame) av_frame_unref(frame);
}

int vc_hdr_send(const unsigned char *data, int length, double pts, double duration) {
    if (!decoder || length < 0) return AVERROR(EINVAL);
    if (!data) return avcodec_send_packet(decoder, NULL);
    av_packet_unref(packet);
    int result = av_new_packet(packet, length);
    if (result < 0) return result;
    memcpy(packet->data, data, length);
    packet->pts = (int64_t)llround(pts);
    packet->dts = AV_NOPTS_VALUE;
    packet->duration = (int64_t)llround(duration);
    result = avcodec_send_packet(decoder, packet);
    av_packet_unref(packet);
    return result;
}

/* One frame per receive keeps memory bounded even while draining a reorder queue. */
int vc_hdr_receive(void) {
    if (!decoder) return AVERROR(EINVAL);
    av_frame_unref(frame);
    int result = avcodec_receive_frame(decoder, frame);
    if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) return 0;
    if (result < 0) return result;
    if (frame->format != AV_PIX_FMT_YUV420P10LE) return AVERROR(ENOSYS);
    return 1;
}

int vc_hdr_width(void) { return frame->width; }
int vc_hdr_height(void) { return frame->height; }
double vc_hdr_pts(void) { return (double)frame->pts; }
double vc_hdr_duration(void) { return (double)frame->duration; }
int vc_hdr_primaries(void) { return frame->color_primaries; }
int vc_hdr_transfer(void) { return frame->color_trc; }
int vc_hdr_matrix(void) { return frame->colorspace; }
int vc_hdr_range(void) { return frame->color_range; }
int vc_hdr_size(void) {
    return av_image_get_buffer_size(frame->format, frame->width, frame->height, 1);
}
int vc_hdr_copy(unsigned char *destination, int capacity) {
    return av_image_copy_to_buffer(destination, capacity,
        (const uint8_t * const *)frame->data, frame->linesize,
        frame->format, frame->width, frame->height, 1);
}
