#include "OpenCVBridge.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

int vc_filters(const uint8_t *rgba, const uint8_t *previous, int width, int height,
               uint8_t *gray, uint8_t *hsv, uint8_t *edges,
               float *laplacian, float *gradientX, float *gradientY, float *flow,
               double *shift) {
    try {
        cv::Mat input(height, width, CV_8UC4, const_cast<uint8_t *>(rgba));
        cv::Mat g(height, width, CV_8U, gray), rgb, h(height, width, CV_8UC3, hsv);
        cv::Mat e(height, width, CV_8U, edges), l(height, width, CV_32F, laplacian);
        cv::Mat gx(height, width, CV_32F, gradientX), gy(height, width, CV_32F, gradientY);
        cv::cvtColor(input, g, cv::COLOR_RGBA2GRAY);
        cv::cvtColor(input, rgb, cv::COLOR_RGBA2RGB);
        cv::cvtColor(rgb, h, cv::COLOR_RGB2HSV);
        cv::Canny(g, e, 60, 140);
        cv::Laplacian(g, l, CV_32F);
        cv::Sobel(g, gx, CV_32F, 1, 0, 3);
        cv::Sobel(g, gy, CV_32F, 0, 1, 3);
        if (previous) {
            cv::Mat p(height, width, CV_8U, const_cast<uint8_t *>(previous));
            cv::Mat first, second;
            p.convertTo(first, CV_32F); g.convertTo(second, CV_32F);
            try {
                cv::Point2d point = cv::phaseCorrelate(first, second, cv::noArray(), shift + 2);
                shift[0] = point.x; shift[1] = point.y;
            } catch (const cv::Exception &) { shift[0] = shift[1] = shift[2] = 0; }
            cv::Mat f(height, width, CV_32FC2, flow);
            cv::calcOpticalFlowFarneback(p, g, f, 0.5, 2, 13, 2, 5, 1.1, 0);
        }
        return 0;
    } catch (const cv::Exception &) { return -1; }
}
