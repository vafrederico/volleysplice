LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := volleycut_yuv
LOCAL_SRC_FILES := yuv_area.cpp
# Keep debug benchmarks optimized too. No fast-math/FMA: fractional weights must
# retain the Java reference's operation order and rounding.
LOCAL_CPPFLAGS := -std=c++17 -O3 -ffp-contract=off -Wall -Wextra -Werror
include $(BUILD_SHARED_LIBRARY)
