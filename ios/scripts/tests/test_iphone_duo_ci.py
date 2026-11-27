"""Unit tests for iPhone Duo CI helpers and arm64 simulator OpenCV build args."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import zlib


SCRIPTS = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CI_DUO = load_module("ci_iphone_duo", "ci-iphone-duo.py")
OPENCV_SIM = load_module("build_opencv_sim", "build-opencv-simulator-framework.py")
PREPARE_ORT = load_module("prepare_ort", "prepare-onnxruntime.py")


def make_png(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    row = b"\x00" + bytes(rgba) * width
    idat = zlib.compress(row * height)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


class IPhoneDuoCITests(unittest.TestCase):
    def test_select_duo_runtime_prefers_duo_specific_27_2_over_standard_27_2_and_27_1(self):
        devicetypes = {
            "devicetypes": [
                {"name": "iPhone 18 Pro", "identifier": "com.apple.CoreSimulator.SimDeviceType.iPhone-18-Pro"},
                {"name": "iPhone Duo", "identifier": "com.apple.CoreSimulator.SimDeviceType.iPhone-Duo"},
            ]
        }
        runtimes = {
            "runtimes": [
                {
                    "name": "iOS 27.1",
                    "version": "27.1",
                    "buildversion": "24A100",
                    "isAvailable": True,
                    "platform": "iOS",
                    "identifier": "com.apple.CoreSimulator.SimRuntime.iOS-27-1",
                    "supportedDeviceTypes": [
                        {"identifier": "com.apple.CoreSimulator.SimDeviceType.iPhone-Duo"}
                    ],
                },
                {
                    "name": "iOS 27.2",
                    "version": "27.2",
                    "buildversion": "24B8117",
                    "isAvailable": True,
                    "platform": "iOS",
                    "identifier": "com.apple.CoreSimulator.SimRuntime.iOS-27-2",
                    "supportedDeviceTypes": [
                        {"identifier": "com.apple.CoreSimulator.SimDeviceType.iPhone-18-Pro"}
                    ],
                },
                {
                    "name": "iOS 27.2",
                    "version": "27.2",
                    "buildversion": "24B91016s",
                    "isAvailable": True,
                    "platform": "iOS",
                    "identifier": "com.apple.CoreSimulator.SimRuntime.iOS-27-2-iPhone19-4",
                    "supportedDeviceTypes": [
                        {"identifier": "com.apple.CoreSimulator.SimDeviceType.iPhone-Duo"}
                    ],
                },
            ]
        }
        device_id, runtime_id, runtime_name = CI_DUO.select_duo_runtime(runtimes, devicetypes)
        self.assertEqual(device_id, "com.apple.CoreSimulator.SimDeviceType.iPhone-Duo")
        self.assertEqual(runtime_id, "com.apple.CoreSimulator.SimRuntime.iOS-27-2-iPhone19-4")
        self.assertEqual(runtime_name, "iOS 27.2")

    def test_select_duo_runtime_rejects_missing_runtime(self):
        devicetypes = {"devicetypes": [{"name": "iPhone Duo", "identifier": "duo.type"}]}
        runtimes = {"runtimes": [{"name": "iOS 27.2", "version": "27.2", "isAvailable": True, "supportedDeviceTypes": []}]}
        with self.assertRaises(ValueError):
            CI_DUO.select_duo_runtime(runtimes, devicetypes)

    def test_select_xcode_prefers_27_2_then_27_1_and_honors_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            apps = Path(tmp)
            (apps / "Xcode_27.1.app/Contents/Developer").mkdir(parents=True)
            self.assertEqual(CI_DUO.select_xcode("auto", apps), apps / "Xcode_27.1.app/Contents/Developer")
            (apps / "Xcode_27.2.app/Contents/Developer").mkdir(parents=True)
            self.assertEqual(CI_DUO.select_xcode("auto", apps), apps / "Xcode_27.2.app/Contents/Developer")
            self.assertEqual(
                CI_DUO.select_xcode(str(apps / "Xcode_27.1.app"), apps),
                apps / "Xcode_27.1.app/Contents/Developer",
            )

    def test_validate_png_accepts_visible_frame_and_rejects_black_or_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "visible.png"
            good.write_bytes(make_png(4, 3, (248, 247, 238, 255)))
            info = CI_DUO.validate_png(good)
            self.assertEqual((info["width"], info["height"], info["non_black"]), (4, 3, True))

            black = root / "black.png"
            black.write_bytes(make_png(4, 3, (0, 0, 0, 255)))
            with self.assertRaises(ValueError):
                CI_DUO.validate_png(black)

            corrupt = root / "corrupt.png"
            corrupt.write_bytes(b"not-a-png")
            with self.assertRaises(ValueError):
                CI_DUO.validate_png(corrupt)

    def test_gallery_html_escapes_metadata_and_lists_screenshots(self):
        html_doc = CI_DUO.build_gallery_html({
            "xcode_version": "Xcode 27.2",
            "runtime_name": "iOS 27.2",
            "runtime_id": "com.apple.CoreSimulator.SimRuntime.iOS-27-2-iPhone19-4",
            "phases": [{
                "title": "Unfolded <iPad>",
                "test_status": "passed",
                "duoctl_state": {"hinge": 180, "screen": "inner"},
                "screenshots": [{"file": "unfolded-landscape-editor.png", "label": "editor", "width": 2853, "height": 2007}],
            }],
            "live_transition": {"screenshots": [{"file": "live-02-unfolded-editor.png", "label": "live", "width": 2853, "height": 2007}]},
        })
        self.assertIn("Unfolded &lt;iPad&gt;", html_doc)
        self.assertIn("unfolded-landscape-editor.png", html_doc)
        self.assertIn("live-02-unfolded-editor.png", html_doc)

    def test_opencv_simulator_cmake_args_target_arm64_and_cmake_4_policy(self):
        args = OPENCV_SIM.cmake_configure_args(
            Path("/tmp/src"), Path("/tmp/build"), Path("/tmp/install"), "/sdk/iPhoneSimulator.sdk", "arm64",
        )
        self.assertIn("-DCMAKE_POLICY_VERSION_MINIMUM=3.5", args)
        self.assertIn("-DCMAKE_OSX_ARCHITECTURES=arm64", args)
        self.assertIn("-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY", args)
        self.assertIn("-DBUILD_LIST=core,imgproc,video", args)
        self.assertIn("-DBUILD_ZLIB=ON", args)
        with self.assertRaises(ValueError):
            OPENCV_SIM.cmake_configure_args(Path("/a"), Path("/b"), Path("/c"), "/sdk", "armv7")

    def test_vendored_onnxruntime_bindings_match_pinned_manifest_and_zip_locator(self):
        bindings_root = SCRIPTS.parent / "Dependencies/ONNXRuntime/Bindings"
        PREPARE_ORT.verify_bindings_tree(bindings_root, bindings_root / "manifest.json")
        # Build a tiny ZIP in memory and confirm locate_zip_entry finds the exact entry.
        import io
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("other.txt", b"hello")
            zf.writestr(PREPARE_ORT.NUGET_ENTRY, b"xcframework-bytes")
        raw = buf.getvalue()
        eocd = raw.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", raw[eocd + 12:eocd + 20])
        method, comp_size, local_off = PREPARE_ORT.locate_zip_entry(raw[cd_off:cd_off + cd_size], PREPARE_ORT.NUGET_ENTRY)
        self.assertEqual(method, 8)
        self.assertGreater(comp_size, 0)
        self.assertEqual(raw[local_off:local_off + 4], b"PK\x03\x04")


if __name__ == "__main__":
    unittest.main()
