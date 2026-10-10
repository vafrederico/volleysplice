#!/usr/bin/env bash
set -euo pipefail

# Activate the pinned SDK (emsdk install/activate 3.1.69) before running this script.
emcc --version | head -n 1 | grep -q '3.1.69' || { echo 'Emscripten 3.1.69 is required.' >&2; exit 1; }
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${VOLLEYCUT_HDR_BUILD_DIR:-$(mktemp -d /tmp/volleycut-hdr-wasm-XXXXXX)}"
mkdir -p "$build_root"
ffmpeg_version=7.1.5
archive="$build_root/ffmpeg-$ffmpeg_version.tar.xz"
archive_sha256=de668509caf9e35e3cd162473441fdb29538c6d96ed080292b3cf9e6fc5d558f
if [ ! -f "$archive" ]; then
  curl --fail --location "https://ffmpeg.org/releases/ffmpeg-$ffmpeg_version.tar.xz" --output "$archive"
fi
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum --check --status
if [ ! -d "$build_root/ffmpeg-$ffmpeg_version" ]; then tar -xf "$archive" -C "$build_root"; fi
cd "$build_root/ffmpeg-$ffmpeg_version"
if [ ! -f ffbuild/config.mak ]; then
  emconfigure ./configure --cc=emcc --cxx=em++ --ar=emar --ranlib=emranlib --nm=emnm \
    --arch=wasm32 --target-os=none --disable-programs --disable-doc --disable-debug \
    --disable-network --disable-autodetect --disable-asm --disable-inline-asm \
    --disable-pthreads --disable-w32threads --disable-os2threads --disable-everything \
    --disable-avformat --disable-avdevice --disable-avfilter --disable-swscale \
    --disable-swresample --disable-iconv --enable-avutil --enable-avcodec \
    --enable-decoder=hevc --enable-parser=hevc
fi
emmake make -j"${VOLLEYCUT_BUILD_JOBS:-4}" libavutil/libavutil.a libavcodec/libavcodec.a
destination="$repository_root/vendor/hdr-hevc-wasm"
mkdir -p "$destination/dist" "$destination/sources"
emcc "$destination/decoder.c" -I. -Llibavcodec -Llibavutil -lavcodec -lavutil \
  -O3 --no-entry -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORT_NAME=createHdrHevcModule \
  -s ENVIRONMENT=web,worker,node -s FILESYSTEM=0 -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=33554432 -s MAXIMUM_MEMORY=1073741824 -s STACK_SIZE=1048576 \
  -s ASSERTIONS=0 -s INVOKE_RUN=0 \
  -s 'EXPORTED_FUNCTIONS=["_vc_hdr_open","_vc_hdr_close","_vc_hdr_reset","_vc_hdr_send","_vc_hdr_receive","_vc_hdr_width","_vc_hdr_height","_vc_hdr_pts","_vc_hdr_duration","_vc_hdr_primaries","_vc_hdr_transfer","_vc_hdr_matrix","_vc_hdr_range","_vc_hdr_size","_vc_hdr_copy","_malloc","_free"]' \
  -o "$destination/dist/hdr-hevc.mjs"
install -m 0644 "$archive" "$destination/sources/ffmpeg-$ffmpeg_version.tar.xz"
install -m 0644 COPYING.LGPLv2.1 "$destination/COPYING.LGPLv2.1"
sha256sum "$destination/dist/hdr-hevc.mjs" "$destination/dist/hdr-hevc.wasm"
