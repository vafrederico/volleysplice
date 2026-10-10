# HDR HEVC decoder

FFmpeg 7.1.5, built with Emscripten 3.1.69, supplies a bounded, single-threaded
HEVC Main10 decoder for browsers whose native decoder returns opaque HDR GPU
surfaces. It emits planar 10-bit YUV without color conversion or tone mapping.
The export worker is terminated after each interval to release decoder memory.

Run `scripts/build-hdr-hevc-wasm.sh` from an activated Emscripten 3.1.69 shell.
The script verifies the upstream archive SHA-256 and prints output hashes.
`VOLLEYCUT_HDR_BUILD_DIR` optionally selects a persistent build directory;
`VOLLEYCUT_BUILD_JOBS` defaults to four. Update production integrity hashes only
after validating rebuilt artifacts. All build flags and exported functions are
recorded in the script. Modify the wrapper or upstream sources and rerun the
link command to produce a replacement module.

The wrapper is MIT; FFmpeg is LGPL-2.1-or-later. The exact unmodified upstream
source archive and license are included here and copied to the site's licenses
directory with the wrapper and build script. GPL, nonfree, networking, demuxing,
encoding, rendering, and audio components are disabled. Dolby Vision dynamic
metadata is not reproduced: compatible Profile 8 HDR base layers become normal
HLG or PQ output. Incompatible Dolby Vision profiles must not be interpreted as
ordinary BT.2020 video.
