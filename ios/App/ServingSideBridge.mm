#include "ServingSideBridge.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

int vc_serving_gray(const uint8_t *rgba, uint8_t *gray) {
    try {
        cv::Mat source(108, 192, CV_8UC4, const_cast<uint8_t *>(rgba));
        cv::Mat output(108, 192, CV_8UC1, gray);
        cv::cvtColor(source, output, cv::COLOR_RGBA2GRAY);
        return 0;
    } catch (const cv::Exception &) { return -1; }
}
int vc_serving_flow(const uint8_t *before, const uint8_t *after, int flight, float *flow) {
    try {
        cv::Mat a(108, 192, CV_8UC1, const_cast<uint8_t *>(before));
        cv::Mat b(108, 192, CV_8UC1, const_cast<uint8_t *>(after));
        cv::Mat output(108, 192, CV_32FC2, flow);
        cv::calcOpticalFlowFarneback(a, b, output, 0.5, flight ? 3 : 2,
                                    flight ? 15 : 13, flight ? 3 : 2, 5, 1.1, 0);
        return 0;
    } catch (const cv::Exception &) { return -1; }
}
int vc_serving_open(const uint8_t *active, uint8_t *opened) {
    try {
        cv::Mat source(108, 192, CV_8UC1, const_cast<uint8_t *>(active)), binary;
        cv::compare(source, 0, binary, cv::CMP_NE);
        cv::Mat output(108, 192, CV_8UC1, opened);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(binary, output, cv::MORPH_OPEN, kernel);
        for (int i = 0; i < 192 * 108; ++i) opened[i] = opened[i] ? 1 : 0;
        return 0;
    } catch (const cv::Exception &) { return -1; }
}
