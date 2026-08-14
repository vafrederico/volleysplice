#!/usr/bin/env bash
set -euo pipefail

ffmpeg_version="7.1.5"
ffmpeg_archive_sha256="de668509caf9e35e3cd162473441fdb29538c6d96ed080292b3cf9e6fc5d558f"
emscripten_image="emscripten/emsdk:3.1.69"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="$(mktemp -d /tmp/volleycut-libswresample-wasm-XXXXXX)"

cleanup() {
  case "$build_root" in
    /tmp/volleycut-libswresample-wasm-*) rm -rf -- "$build_root" ;;
    *) echo "Refusing to remove unexpected build directory: $build_root" >&2 ;;
  esac
}
trap cleanup EXIT

archive="$build_root/ffmpeg-$ffmpeg_version.tar.xz"
curl --fail --location \
  "https://ffmpeg.org/releases/ffmpeg-$ffmpeg_version.tar.xz" \
  --output "$archive"
printf '%s  %s\n' "$ffmpeg_archive_sha256" "$archive" | sha256sum --check --status
tar -xf "$archive" -C "$build_root"
mkdir -p "$build_root/output"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$build_root:/work" \
  --workdir "/work/ffmpeg-$ffmpeg_version" \
  "$emscripten_image" \
  emconfigure ./configure \
    --prefix=/work/install \
    --cc=emcc \
    --cxx=em++ \
    --ar=emar \
    --ranlib=emranlib \
    --nm=emnm \
    --arch=wasm32 \
    --target-os=none \
    --disable-programs \
    --disable-doc \
    --disable-debug \
    --disable-network \
    --disable-autodetect \
    --disable-asm \
    --disable-inline-asm \
    --disable-pthreads \
    --disable-w32threads \
    --disable-os2threads \
    --disable-everything \
    --disable-avcodec \
    --disable-avformat \
    --disable-avdevice \
    --disable-avfilter \
    --disable-swscale \
    --disable-iconv \
    --enable-avutil \
    --enable-swresample

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$build_root:/work" \
  --workdir "/work/ffmpeg-$ffmpeg_version" \
  "$emscripten_image" \
  emmake make -j"$(nproc)" libavutil/libavutil.a libswresample/libswresample.a

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$build_root:/work" \
  --volume "$repository_root:/repo:ro" \
  --workdir "/work/ffmpeg-$ffmpeg_version" \
  "$emscripten_image" \
  emcc \
    /repo/vendor/libswresample-wasm/resampler.c \
    -I"/work/ffmpeg-$ffmpeg_version" \
    -L"/work/ffmpeg-$ffmpeg_version/libswresample" \
    -L"/work/ffmpeg-$ffmpeg_version/libavutil" \
    -lswresample \
    -lavutil \
    -O3 \
    --no-entry \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s EXPORT_NAME=createLibswresampleModule \
    -s ENVIRONMENT=web \
    -s FILESYSTEM=0 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=16777216 \
    -s STACK_SIZE=1048576 \
    -s ASSERTIONS=0 \
    -s INVOKE_RUN=0 \
    -s 'EXPORTED_FUNCTIONS=["_vc_swr_init","_vc_swr_get_out_samples","_vc_swr_convert","_vc_swr_close","_malloc","_free"]' \
    -o /work/output/libswresample.mjs

destination="$repository_root/vendor/libswresample-wasm/dist"
mkdir -p "$destination"
install -m 0644 "$build_root/output/libswresample.mjs" "$destination/libswresample.mjs"
install -m 0644 "$build_root/output/libswresample.wasm" "$destination/libswresample.wasm"
sha256sum "$destination/libswresample.mjs" "$destination/libswresample.wasm"
