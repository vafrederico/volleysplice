#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Direct NV12 conversion: no full-resolution RGB allocation. Geometry is in
// displayed coordinates; source clean aperture is applied before rotation.
int vc_nv12_area(const uint8_t *y, int yStride, const uint8_t *uv, int uvStride,
                 int sourceLeft, int sourceTop, int sourceWidth, int sourceHeight,
                 int rotation, int matrix, int fullRange, const double *roi,
                 int outputWidth, int outputHeight, uint8_t *rgba);
int vc_nv12_letterbox(const uint8_t *y, int yStride, const uint8_t *uv, int uvStride,
                      int sourceLeft, int sourceTop, int sourceWidth, int sourceHeight,
                      int rotation, int matrix, int fullRange, const int *geometry,
                      int size, float *rgb);
#ifdef __cplusplus
}
#endif
