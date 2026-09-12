#!/bin/bash
set -euo pipefail
project="$(cd "$(dirname "$0")/.." && pwd)"
device="${1:?Pass a simulator UDID from xcrun simctl list devices}"
mode="${2:-build}"
mkdir -p "$project/artifacts"
# The lab OpenCV framework has an x86_64 simulator slice; arm64 is its device slice.
case "$mode" in
  build) action=build ;;
  tests) action=build-for-testing ;;
  *) echo 'Expected build or tests' >&2; exit 2 ;;
esac
xcodebuild -project "$project/VolleySplice.xcodeproj" -scheme VolleySplice \
  -configuration Debug -sdk iphonesimulator -destination "id=$device" \
  -derivedDataPath "$project/build/simulator" -jobs 4 ARCHS=x86_64 ONLY_ACTIVE_ARCH=YES \
  CODE_SIGNING_ALLOWED=NO "$action" > "$project/artifacts/simulator-build.log" 2>&1 || {
    grep -A5 'error:' "$project/artifacts/simulator-build.log" || tail -30 "$project/artifacts/simulator-build.log"
    exit 1
  }
tail -6 "$project/artifacts/simulator-build.log"
