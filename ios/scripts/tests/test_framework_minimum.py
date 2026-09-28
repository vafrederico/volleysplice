import importlib.util
from pathlib import Path
import plistlib
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macho_versions import check_bundle_version, deployment_versions, version

spec = importlib.util.spec_from_file_location('framework_fix', Path(__file__).parents[1] / 'fix-onnx-framework-minimum.py')
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)


def thin(minimum=17, platform=2, legacy=False, endian='<', cpu=0x100000c):
    command = struct.pack(endian + '4I', 0x25, 16, minimum << 16, 26 << 16) if legacy else \
        struct.pack(endian + '6I', 0x32, 24, platform, minimum << 16, 26 << 16, 0)
    return struct.pack(endian + '8I', 0xfeedfacf, cpu, 0, 6, 1, len(command), 0, 0) + command


def fat(slices, wide=False, endian='>'):
    width = 32 if wide else 20
    offset = 8 + len(slices) * width
    result = struct.pack(endian + 'II', 0xcafebabf if wide else 0xcafebabe, len(slices))
    for data in slices:
        result += struct.pack(endian + ('IIQQII' if wide else 'IIIII'),
                              0x100000c, 0, offset, len(data), 0, *([0] if wide else []))
        offset += len(data)
    return result + b''.join(slices)


class FrameworkMinimumTests(unittest.TestCase):
    def test_reads_every_fat_slice_and_legacy_commands(self):
        self.assertEqual(deployment_versions(thin(15, legacy=True))[0]['minimum'], (15, 0, 0))
        self.assertEqual(deployment_versions(thin(17, endian='>'))[0]['minimum'], (17, 0, 0))
        for wide in [False, True]:
            for endian in ['<', '>']:
                data = fat([thin(15), thin(17)], wide, endian)
                self.assertEqual(len(deployment_versions(data)), 2)
                with self.assertRaisesRegex(ValueError, 'below Mach-O'):
                    check_bundle_version({'MinimumOSVersion': '15.1'}, data)

    def test_rejects_wrong_platform_and_malformed_commands(self):
        with self.assertRaisesRegex(ValueError, 'wrong Apple platform'):
            check_bundle_version({'MinimumOSVersion': '17.0'}, thin(platform=7))
        for data in [b'bad', thin()[:20], thin()[:-1], fat([thin()])[:-1]]:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                deployment_versions(data)
        damaged = bytearray(thin());struct.pack_into('<I', damaged, 36, 0)
        with self.assertRaises(ValueError):
            deployment_versions(bytes(damaged))

    def test_versions_are_numeric_not_lexicographic(self):
        self.assertLess(version('17.9'), version('17.10'))
        for value in [None, '', '17.beta', '0', 17]:
            with self.subTest(value=value), self.assertRaises(ValueError):version(value)

    def fixture(self, directory, minimum=17):
        framework = Path(directory) / 'onnxruntime.framework';framework.mkdir()
        (framework / 'onnxruntime').write_bytes(fat([thin(minimum)]))
        (framework / 'Info.plist').write_bytes(plistlib.dumps({
            'CFBundleIdentifier': 'com.microsoft.onnxruntime', 'CFBundleExecutable': 'onnxruntime',
            'MinimumOSVersion': '15.1'}, fmt=plistlib.FMT_BINARY))
        return framework

    def test_repairs_only_metadata_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            framework = self.fixture(directory)
            binary = (framework / 'onnxruntime').read_bytes()
            self.assertTrue(fix.repair(framework, '17.0')['changed'])
            self.assertEqual((framework / 'onnxruntime').read_bytes(), binary)
            self.assertEqual(plistlib.loads((framework / 'Info.plist').read_bytes())['MinimumOSVersion'], '17.0')
            self.assertFalse(fix.repair(framework, '17.0')['changed'])

    def test_never_papers_over_a_higher_binary_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            framework = self.fixture(directory, minimum=18)
            before = (framework / 'Info.plist').read_bytes()
            with self.assertRaisesRegex(ValueError, 'newer OS'):
                fix.repair(framework, '17.0')
            self.assertEqual((framework / 'Info.plist').read_bytes(), before)

    def test_resigns_modified_framework_and_recovers_interrupted_signing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fix.subprocess, 'run') as signing:
            framework = self.fixture(directory)
            (framework / '_CodeSignature').mkdir()
            with self.assertRaisesRegex(ValueError, 'signing identity'):
                fix.repair(framework, '17.0')
            fix.repair(framework, '17.0', signing_identity='synthetic-identity')
            self.assertIn('synthetic-identity', signing.call_args.args[0])
            self.assertIn('--preserve-metadata=identifier,entitlements,flags', signing.call_args.args[0])
            fix.repair(framework, '17.0', signing_identity='synthetic-identity')
            self.assertEqual(signing.call_count, 2)


if __name__ == '__main__':
    unittest.main()
