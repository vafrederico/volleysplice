#!/bin/bash
set -euo pipefail
project="$HOME/wslmac-projects/volleysplice"
derived="$project/build/device"
mkdir -p "$project/artifacts"
if [[ "${1:-}" == "--unsigned" ]]; then
  xcodebuild -project "$project/VolleySplice.xcodeproj" -scheme VolleySplice \
    -configuration Debug -sdk iphoneos -destination 'generic/platform=iOS' \
    -derivedDataPath "$derived" -jobs 4 CODE_SIGNING_ALLOWED=NO \
    build > "$project/artifacts/unsigned-build.log" 2>&1 || { grep -A 5 'error:' "$project/artifacts/unsigned-build.log"; exit 1; }
  tail -5 "$project/artifacts/unsigned-build.log"
  exit 0
fi
xcodebuild -project "$project/VolleySplice.xcodeproj" -scheme VolleySplice \
  -configuration Debug -sdk iphoneos -destination 'generic/platform=iOS' \
  -derivedDataPath "$derived" -jobs 4 -allowProvisioningUpdates \
  build > "$project/artifacts/build.log" 2>&1 || { grep -A 5 'error:' "$project/artifacts/build.log" || tail -20 "$project/artifacts/build.log"; exit 1; }
app="$derived/Build/Products/Debug-iphoneos/VolleySplice.app"
codesign --verify --deep --strict "$app"
test -f "$app/embedded.mobileprovision"
security cms -D -i "$app/embedded.mobileprovision" > "$project/artifacts/profile.plist"
/usr/bin/python3 - "$project/artifacts/profile.plist" "$app/Info.plist" <<'PY'
import datetime, plistlib, sys
with open(sys.argv[1], 'rb') as handle:
    profile = plistlib.load(handle)
with open(sys.argv[2], 'rb') as handle:
    info = plistlib.load(handle)
assert info['CFBundleIdentifier'] == 'com.vafrederico.VolleySplice', 'Unexpected bundle identity'
assert profile['Entitlements']['application-identifier'] == '<apple-development-team-id>.com.vafrederico.VolleySplice', 'Wrong provisioning identity'
assert '<ios-device-udid>' in profile.get('ProvisionedDevices', []), 'Profile excludes the lab iPad'
assert profile['ExpirationDate'] > datetime.datetime.utcnow(), 'Provisioning profile expired'
assert info.get('UIFileSharingEnabled') and info.get('LSSupportsOpeningDocumentsInPlace'), 'Files access is missing'
print('Provisioning identity, iPad UDID, expiry and Files access verified')
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
