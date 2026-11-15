#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${VOLLEYCUT_MACOS_OPENCV_ROOT:?Set the native OpenCV 4.12.0 installation prefix}"
test -f "$VOLLEYCUT_MACOS_OPENCV_ROOT/lib/libopencv_core.a"
grep -q '#define CV_VERSION_MAJOR    4' "$VOLLEYCUT_MACOS_OPENCV_ROOT/include/opencv4/opencv2/core/version.hpp"
grep -q '#define CV_VERSION_MINOR    12' "$VOLLEYCUT_MACOS_OPENCV_ROOT/include/opencv4/opencv2/core/version.hpp"
grep -q '#define CV_VERSION_REVISION 0' "$VOLLEYCUT_MACOS_OPENCV_ROOT/include/opencv4/opencv2/core/version.hpp"
python3 scripts/generate-project.py
xcodebuild -project VolleySplice.xcodeproj -scheme VolleySpliceBenchmark \
  -configuration Debug -destination "platform=macOS,arch=$(uname -m)" \
  -derivedDataPath build/macos-benchmark -jobs 4 \
  CODE_SIGNING_ALLOWED=NO ONLY_ACTIVE_ARCH=YES \
  "VOLLEYCUT_MACOS_OPENCV_ROOT=$VOLLEYCUT_MACOS_OPENCV_ROOT" \
  "$@" build-for-testing
