#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// 192x108 grayscale conversion, Farneback and morphology only; reductions remain Swift.
int vc_serving_gray(const uint8_t *rgba, uint8_t *gray);
int vc_serving_flow(const uint8_t *before, const uint8_t *after, int flight, float *flow);
int vc_serving_open(const uint8_t *active, uint8_t *opened);
#ifdef __cplusplus
}
#endif
