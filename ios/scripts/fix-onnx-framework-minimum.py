"""Repair Xcode's embedded static-framework stub metadata, before app signing."""
import argparse
import os
from pathlib import Path
import plistlib
import subprocess

from macho_versions import check_bundle_version, deployment_versions, version, version_text


def repair(framework, deployment_target, platform=2, signing_identity=None):
    if framework.name != 'onnxruntime.framework':
        raise ValueError('Only the embedded ONNX Runtime framework may be repaired')
    plist = framework / 'Info.plist'
    original = plist.read_bytes()
    info = plistlib.loads(original)
    if info.get('CFBundleIdentifier') != 'com.microsoft.onnxruntime' or info.get('CFBundleExecutable') != 'onnxruntime':
        raise ValueError('Unexpected ONNX Runtime framework identity')
    executable = (framework / 'onnxruntime').read_bytes()
    slices = deployment_versions(executable)
    if any(item['platform'] != platform for item in slices):
        raise ValueError('ONNX Runtime executable targets the wrong Apple platform')
    current = version(info.get('MinimumOSVersion'))
    required = max(current, *(item['minimum'] for item in slices))
    if required > version(deployment_target):
        raise ValueError('ONNX Runtime requires a newer OS than the app deployment target')
    changed = required != current
    if changed:
        if (framework / '_CodeSignature').exists() and not signing_identity:
            raise ValueError('Changing a signed framework requires its Xcode signing identity')
        info['MinimumOSVersion'] = version_text(required)
        fmt = plistlib.FMT_BINARY if original.startswith(b'bplist00') else plistlib.FMT_XML
        plist.write_bytes(plistlib.dumps(info, fmt=fmt, sort_keys=False))
    # An always-running phase must also recover a prior interrupted signing attempt.
    if signing_identity:
        subprocess.run(['/usr/bin/codesign', '--force', '--sign', signing_identity,
                        '--preserve-metadata=identifier,entitlements,flags',
                        '--generate-entitlement-der', str(framework)], check=True)
    report = check_bundle_version(info, executable, version(deployment_target), platform)
    return {'changed': changed, 'previous': version_text(current), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', type=Path, required=True)
    parser.add_argument('--deployment-target', required=True)
    parser.add_argument('--platform', choices=['iphoneos', 'iphonesimulator'], required=True)
    args = parser.parse_args()
    identity = None
    if os.environ.get('CODE_SIGNING_ALLOWED') != 'NO':
        identity = os.environ.get('EXPANDED_CODE_SIGN_IDENTITY')
        if not identity:
            raise ValueError('Missing Xcode signing identity; refusing to invalidate the framework signature')
    result = repair(args.framework, args.deployment_target,
                    2 if args.platform == 'iphoneos' else 7, identity)
    print('ONNX Runtime MinimumOSVersion: ' + result['previous'] + ' -> ' + result['declared'] +
          '; Mach-O requires ' + result['binary_minimum'])


if __name__ == '__main__':
    main()
