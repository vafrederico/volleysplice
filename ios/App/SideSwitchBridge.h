#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Fixed 256x144 BGR inputs. Hough candidate returns 0 absent, 1 found, -1 error.
int vc_switch_net_line(const uint8_t *bgr, double *position_score);
// Seven packed BGR frames -> seven packed grayscale + HSV frames and [max shift,min response].
int vc_switch_prepare(const uint8_t *bgr_frames, double net_y, uint8_t *gray_frames, uint8_t *hsv_frames, double *quality);
#ifdef __cplusplus
}
#endif
