#!/bin/bash
set -euo pipefail
: "${VOLLEYCUT_IOS_REMOTE_PROJECT_ROOT:?Set VOLLEYCUT_IOS_REMOTE_PROJECT_ROOT to the remote project directory}"
project="$VOLLEYCUT_IOS_REMOTE_PROJECT_ROOT"
derived="$project/build/device"
mkdir -p "$project/artifacts"
: "${VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT:?Set VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT to the OpenCV framework directory}"
if [[ "${1:-}" == "--unsigned" ]]; then
  xcodebuild -project "$project/VolleySplice.xcodeproj" -scheme VolleySplice \
    -configuration Debug -sdk iphoneos -destination 'generic/platform=iOS' \
    -derivedDataPath "$derived" -jobs 4 CODE_SIGNING_ALLOWED=NO \
    VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT="$VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT" \
    build > "$project/artifacts/unsigned-build.log" 2>&1 || { grep -A 5 'error:' "$project/artifacts/unsigned-build.log"; exit 1; }
  tail -5 "$project/artifacts/unsigned-build.log"
  exit 0
fi
: "${VOLLEYCUT_IOS_DEVICE_UDID:?Set VOLLEYCUT_IOS_DEVICE_UDID to the target device UDID}"
: "${VOLLEYCUT_IOS_DEVELOPMENT_TEAM:?Set VOLLEYCUT_IOS_DEVELOPMENT_TEAM to the Apple development team ID}"
xcodebuild -project "$project/VolleySplice.xcodeproj" -scheme VolleySplice \
  -configuration Debug -sdk iphoneos -destination 'generic/platform=iOS' \
  -derivedDataPath "$derived" -jobs 4 -allowProvisioningUpdates \
  DEVELOPMENT_TEAM="$VOLLEYCUT_IOS_DEVELOPMENT_TEAM" \
  VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT="$VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT" \
  build > "$project/artifacts/build.log" 2>&1 || { grep -A 5 'error:' "$project/artifacts/build.log" || tail -20 "$project/artifacts/build.log"; exit 1; }
app="$derived/Build/Products/Debug-iphoneos/VolleySplice.app"
codesign --verify --deep --strict "$app"
test -f "$app/embedded.mobileprovision"
security cms -D -i "$app/embedded.mobileprovision" > "$project/artifacts/profile.plist"
/usr/bin/python3 - "$project/artifacts/profile.plist" "$app/Info.plist" \
  "$VOLLEYCUT_IOS_DEVICE_UDID" "$VOLLEYCUT_IOS_DEVELOPMENT_TEAM" <<'PY'
import datetime, plistlib, sys
with open(sys.argv[1], 'rb') as handle:
    profile = plistlib.load(handle)
with open(sys.argv[2], 'rb') as handle:
    info = plistlib.load(handle)
assert info['CFBundleIdentifier'] == 'com.volleysplice.VolleySplice', 'Unexpected bundle identity'
assert profile['Entitlements']['application-identifier'] == f'{sys.argv[4]}.com.volleysplice.VolleySplice', 'Wrong provisioning identity'
assert sys.argv[3] in profile.get('ProvisionedDevices', []), 'Profile excludes the requested iOS device'
assert profile['ExpirationDate'] > datetime.datetime.utcnow(), 'Provisioning profile expired'
assert info.get('UIFileSharingEnabled') and info.get('LSSupportsOpeningDocumentsInPlace'), 'Files access is missing'
print('Provisioning identity, requested device, expiry and Files access verified')
PY
/usr/libexec/PlistBuddy -c 'Print :Entitlements:application-identifier' "$project/artifacts/profile.plist"
/usr/libexec/PlistBuddy -c 'Print :ExpirationDate' "$project/artifacts/profile.plist"
stage=$(mktemp -d "$project/artifacts/package.XXXXXX")
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/Payload"
ditto "$app" "$stage/Payload/VolleySplice.app"
(cd "$stage" && /usr/bin/zip -qry "$stage/VolleySplice.ipa" Payload)
mv "$stage/VolleySplice.ipa" "$project/artifacts/VolleySplice.ipa"
shasum -a 256 "$project/artifacts/VolleySplice.ipa"
tail -8 "$project/artifacts/build.log"
