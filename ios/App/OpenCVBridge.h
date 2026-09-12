#include <stdint.h>
#include "ServingSideBridge.h"
#include "SideSwitchBridge.h"
#ifdef __cplusplus
extern "C" {
#endif
// Only OpenCV primitives cross this boundary. Feature reductions live in Swift.
int vc_filters(const uint8_t *rgba, const uint8_t *previous, int width, int height,
               uint8_t *gray, uint8_t *hsv, uint8_t *edges,
               float *laplacian, float *gradientX, float *gradientY, float *flow,
               double *shift);
#ifdef __cplusplus
}
#endif
