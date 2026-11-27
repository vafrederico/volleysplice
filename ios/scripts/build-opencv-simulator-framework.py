"""Build a minimal static opencv2.framework for iOS Simulator (arm64 or x86_64).

Official OpenCV 4.12.0 iOS framework releases ship arm64 for iphoneos and x86_64
for iphonesimulator, without an arm64 simulator slice. Xcode 27 / iOS 27
simulators on Apple Silicon are arm64-only, so this script builds the exact same
pinned OpenCV 4.12.0 commit (core, imgproc, video) as a drop-in opencv2.framework.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tarfile
import tempfile


OPENCV_VERSION = "4.12.0"
OPENCV_COMMIT = "49486f61fb25722cbcf586b7f4320921d46fb38e"
OPENCV_URL = f"https://github.com/opencv/opencv/archive/{OPENCV_COMMIT}.tar.gz"
OPENCV_SHA256 = "8f00b42869ab2836be36090f6631ae4c38ba59171e22a69a8e0f92f8ef1771d4"
DEPLOYMENT_TARGET = "17.0"


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual} != {expected}")


def cmake_configure_args(
    source_dir: Path,
    build_dir: Path,
    install_dir: Path,
    sdk_path: str,
    arch: str,
    deployment_target: str = DEPLOYMENT_TARGET,
) -> list[str]:
    if arch not in ("arm64", "x86_64"):
        raise ValueError(f"Unsupported simulator arch: {arch}")
    return [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        "-G", "Ninja",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_SYSTEM_NAME=iOS",
        f"-DCMAKE_SYSTEM_PROCESSOR={arch}",
        f"-DCMAKE_OSX_SYSROOT={sdk_path}",
        f"-DCMAKE_OSX_ARCHITECTURES={arch}",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}",
        "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DBUILD_LIST=core,imgproc,video",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DBUILD_ZLIB=ON",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_DOCS=OFF",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_opencv_dnn=OFF",
        "-DBUILD_opencv_objc=OFF",
        "-DBUILD_opencv_python3=OFF",
        "-DBUILD_JAVA=OFF",
        "-DWITH_IPP=OFF",
        "-DWITH_ITT=OFF",
        "-DWITH_OPENCL=OFF",
        "-DWITH_LAPACK=OFF",
        "-DWITH_EIGEN=OFF",
        "-DWITH_PROTOBUF=OFF",
        "-DWITH_ADE=OFF",
        "-DWITH_PNG=OFF",
        "-DWITH_JPEG=OFF",
        "-DWITH_TIFF=OFF",
        "-DWITH_WEBP=OFF",
        "-DWITH_OPENJPEG=OFF",
        "-DWITH_OPENEXR=OFF",
        "-DWITH_IMGCODEC_HDR=OFF",
        "-DWITH_IMGCODEC_SUNRASTER=OFF",
        "-DWITH_IMGCODEC_PXM=OFF",
        "-DWITH_IMGCODEC_PFM=OFF",
    ]


def collect_static_libs(install_dir: Path) -> list[Path]:
    libs = sorted((install_dir / "lib").glob("libopencv_*.a"))
    third_party = install_dir / "lib/opencv4/3rdparty"
    if third_party.is_dir():
        libs += sorted(third_party.glob("*.a"))
    if not libs:
        raise ValueError(f"No OpenCV static archives found under {install_dir / 'lib'}")
    return libs


def package_framework(install_dir: Path, output_root: Path, arch: str, deployment_target: str = DEPLOYMENT_TARGET) -> Path:
    headers_src = install_dir / "include/opencv4/opencv2"
    if not (headers_src / "core.hpp").is_file():
        raise ValueError(f"Missing installed OpenCV headers at {headers_src}")
    libs = collect_static_libs(install_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / "opencv2.framework.staging"
    final = output_root / "opencv2.framework"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.copytree(headers_src, staging / "Headers")
    subprocess.run(
        ["libtool", "-static", "-o", str(staging / "opencv2"), *[str(p) for p in libs]],
        check=True,
    )
    info = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleExecutable": "opencv2",
        "CFBundleIdentifier": "org.opencv.opencv2",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": "opencv2",
        "CFBundlePackageType": "FMWK",
        "CFBundleShortVersionString": OPENCV_VERSION,
        "CFBundleVersion": OPENCV_VERSION,
        "MinimumOSVersion": deployment_target,
    }
    (staging / "Info.plist").write_bytes(plistlib.dumps(info))
    lipo = subprocess.run(["lipo", "-info", str(staging / "opencv2")], check=True, capture_output=True, text=True).stdout
    if arch not in lipo:
        raise ValueError(f"Expected {arch} in built opencv2.framework, got: {lipo.strip()}")
    shutil.rmtree(final, ignore_errors=True)
    staging.rename(final)
    return final


def build(output_root: Path, arch: str, source_archive: Path | None = None, jobs: int | None = None) -> Path:
    existing = output_root / "opencv2.framework"
    if (existing / "opencv2").is_file() and (existing / "Headers/core.hpp").is_file():
        lipo = subprocess.run(["lipo", "-info", str(existing / "opencv2")], check=True, capture_output=True, text=True).stdout
        if arch in lipo:
            print(f"Reusing cached opencv2.framework ({arch}) at {existing}")
            return existing
    sdk_path = subprocess.run(
        ["xcrun", "--sdk", "iphonesimulator", "--show-sdk-path"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not sdk_path:
        raise ValueError("Unable to resolve iphonesimulator SDK path via xcrun")
    parallel = jobs or max(2, os.cpu_count() or 4)
    with tempfile.TemporaryDirectory(prefix="volleycut-opencv-sim-") as tmp:
        work = Path(tmp)
        archive = source_archive
        if archive is None:
            archive = work / f"opencv-{OPENCV_VERSION}.tar.gz"
            subprocess.run(
                ["curl", "--fail", "--location", "--retry", "3", "--silent", "--show-error", OPENCV_URL, "-o", str(archive)],
                check=True,
            )
        verify_sha256(archive, OPENCV_SHA256)
        extract_root = work / "src"
        extract_root.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_root)
        source_dir = extract_root / f"opencv-{OPENCV_COMMIT}"
        if not source_dir.is_dir():
            candidates = [p for p in extract_root.iterdir() if p.is_dir()]
            if len(candidates) != 1:
                raise ValueError("Unexpected OpenCV archive layout")
            source_dir = candidates[0]
        build_dir = work / "build"
        install_dir = work / "install"
        subprocess.run(cmake_configure_args(source_dir, build_dir, install_dir, sdk_path, arch), check=True)
        subprocess.run(["cmake", "--build", str(build_dir), "-j", str(parallel)], check=True)
        subprocess.run(["cmake", "--install", str(build_dir)], check=True)
        framework = package_framework(install_dir, output_root, arch)
        print(f"Built minimal OpenCV {OPENCV_VERSION} simulator framework ({arch}) at {framework}")
        return framework


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory that will contain opencv2.framework")
    parser.add_argument("--arch", choices=("arm64", "x86_64"), default="arm64")
    parser.add_argument("--source-archive", type=Path, help="Optional pre-downloaded OpenCV 4.12.0 tar.gz")
    parser.add_argument("--jobs", type=int, help="Parallel Ninja jobs")
    args = parser.parse_args()
    build(args.output, args.arch, args.source_archive, args.jobs)


if __name__ == "__main__":
    main()
