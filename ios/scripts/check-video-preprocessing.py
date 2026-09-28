"""CPU-only smoke/golden checks for the exact app NV12 kernel; no Apple GPU needed."""
import argparse
import ctypes as c
import json
import platform
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "App/VideoPreprocessingBridge.mm"
    library = args.build_dir / ("video-preprocessing.dylib" if platform.system() == "Darwin" else "video-preprocessing.so")
    subprocess.run([args.cxx, "-x", "c++", "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    native = c.CDLL(str(library))
    u8p = c.POINTER(c.c_uint8)
    prefix = [u8p, c.c_int, u8p, c.c_int] + [c.c_int] * 7
    native.vc_nv12_area.argtypes = prefix + [c.POINTER(c.c_double), c.c_int, c.c_int, u8p]
    native.vc_nv12_letterbox.argtypes = prefix + [c.POINTER(c.c_int), c.c_int, c.POINTER(c.c_float)]
    roi = (c.c_double * 4)(0, 0, 1, 1)
    y = (c.c_uint8 * 24)(*((i // 6 * 67 + i % 6 * 29) % 256 for i in range(24)))
    uv = (c.c_uint8 * 12)(*(v for i in range(6) for v in (40 + i * 29, 205 - i * 23)))
    # Generated with checked-in Android YuvAreaResampler + YuvColorConversion,
    # independently of the iOS native bridge. Includes nonintegral rotate+resize.
    expected = [
        [175,27,0,255,201,88,10,255,221,162,109,255,206,189,191,255,98,117,158,255,56,106,191,255],
        [164,156,173,255,195,121,93,255,159,26,0,255,39,86,168,255,196,176,162,255,203,124,63,255],
        [56,106,191,255,98,117,158,255,206,189,191,255,221,162,109,255,201,88,10,255,175,27,0,255],
        [203,124,63,255,196,176,162,255,39,86,168,255,159,26,0,255,195,121,93,255,164,156,173,255],
    ]
    for rotation, golden in zip((0,90,180,270), expected):
        result = (c.c_uint8 * 24)()
        assert native.vc_nv12_area(y,6,uv,6,0,0,6,4,rotation,1,0,roi,3,2,result) == 0
        assert list(result) == golden, (rotation, list(result), golden)

    # BT.601 vs BT.709 matrix, full/video range, and metadata validation.
    pixel_y, pixel_uv = (c.c_uint8 * 4)(81,81,81,81), (c.c_uint8 * 2)(90,240)
    for matrix, full, golden in [(0,0,[254,0,0]), (1,0,[255,24,0]), (0,1,[238,14,14]), (1,1,[255,36,10])]:
        result = (c.c_uint8 * 4)()
        assert native.vc_nv12_area(pixel_y,2,pixel_uv,2,0,0,2,2,0,matrix,full,roi,1,1,result) == 0
        assert list(result)[:3] == golden, (matrix, full, list(result))

    # Bilinear interpolation must retain fractional pixels before normalization.
    # Full-range neutral gray maps identically to RGB; a 2x2 -> 3x3 sample has
    # arithmetic midpoint 111.75, not a rounded byte. Black padding stays zero.
    yy, uu = (c.c_uint8 * 4)(0,64,128,255), (c.c_uint8 * 2)(128,128)
    geometry = (c.c_int * 8)(0,0,2,2,3,3,1,1)
    rgb = (c.c_float * 75)()
    assert native.vc_nv12_letterbox(yy,2,uu,2,0,0,2,2,0,1,1,geometry,5,rgb) == 0
    assert list(rgb)[36:39] == [111.75]*3
    assert list(rgb)[:15] == [0]*15
    for rotation, corner in [(0,0),(90,128),(180,255),(270,64)]:
        assert native.vc_nv12_letterbox(yy,2,uu,2,0,0,2,2,rotation,1,1,geometry,5,rgb) == 0
        assert list(rgb)[18:21] == [corner]*3

    # Saturated white at this nonintegral scale previously yielded 255.000015
    # in 10,368 channels, causing the app's strict normalizedCHW check to fail.
    white = (c.c_uint8 * (1920*1080))(*([255]*(1920*1080)))
    neutral = (c.c_uint8 * (1920*1080//2))(*([128]*(1920*1080//2)))
    white_geometry = (c.c_int * 8)(0,0,1920,1080,224,126,0,49)
    white_rgb = (c.c_float * (224*224*3))()
    assert native.vc_nv12_letterbox(white,1920,neutral,1920,0,0,1920,1080,0,1,1,white_geometry,224,white_rgb) == 0
    assert min(white_rgb) >= 0 and max(white_rgb) <= 255
    assert max(white_rgb) == 255

    # A small synthetic CPU timing, deliberately not an iPhone performance claim.
    yy = (c.c_uint8 * (1920*1080))(*([81]*(1920*1080)))
    uu = (c.c_uint8 * (1920*1080//2))(*([128]*(1920*1080//2)))
    rgba = (c.c_uint8 * (192*108*4))()
    rgb = (c.c_float * (224*224*3))()
    geometry = (c.c_int * 8)(0,0,1920,1080,224,126,0,49)
    started = time.perf_counter()
    for _ in range(8):
        assert native.vc_nv12_area(yy,1920,uu,1920,0,0,1920,1080,0,1,0,roi,192,108,rgba) == 0
    area_ms = (time.perf_counter()-started)*1000/8
    started = time.perf_counter()
    for _ in range(8):
        assert native.vc_nv12_letterbox(yy,1920,uu,1920,0,0,1920,1080,0,1,0,geometry,224,rgb) == 0
    print(json.dumps({"goldenChecks": "passed", "syntheticSource": "1920x1080 NV12", "samplesPerTiming": 8,
                      "areaMillisecondsPerFrame": round(area_ms,3),
                      "letterboxMillisecondsPerFrame": round((time.perf_counter()-started)*1000/8,3),
                      "timingScope": "Host CPU kernel only; excludes video decoding, OpenCV features and inference"}))


if __name__ == "__main__":
    main()
