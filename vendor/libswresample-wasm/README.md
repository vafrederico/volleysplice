# libswresample WASM experiment

This directory contains a separately replaceable, single-threaded WebAssembly build of
FFmpeg 7.1.5's `libswresample` and `libavutil`. VolleyCut uses it only for the explicit
`libswresample-wasm-v1` browser inference experiment. The normal browser path remains
`linear-v1`.

The wrapper accepts mono or stereo planar Float32 PCM at the decoded source rate and
streams mono signed-16-bit PCM at 16 kHz. It deliberately lets libswresample perform
stereo rematrixing, filtering, rate conversion, and S16 quantization together, matching
the offline FFmpeg feature command as closely as the browser decoder permits.

## Rebuild and verify

Run from the repository root:

```sh
scripts/build-libswresample-wasm.sh
node scripts/validate-libswresample-wasm.mjs
```

The build script downloads the official FFmpeg 7.1.5 source archive (SHA-256
`de668509caf9e35e3cd162473441fdb29538c6d96ed080292b3cf9e6fc5d558f`) and uses the
pinned `emscripten/emsdk:3.1.69` container. The exact configure and link flags are in
that script. The checked-in module hashes are:

- `libswresample.mjs`: `022782d1e08e483d8c67de30f68177eff5904998c410df028f26e342c793ae48`
- `libswresample.wasm`: `c7ed95ed8b6f5e11ea9e86262214b978bd145a6ec1f648dd56616af298449f90`

The deterministic native-FFmpeg parity probe is streaming-chunk invariant. Against
the host's FFmpeg 7.1.5 SIMD build it produces the same 16,046 samples; 16,042 are
bit-identical and the remaining four differ by one signed-16-bit least-significant bit.

## License

FFmpeg's libraries are built here without GPL or nonfree components. They are licensed
under the GNU Lesser General Public License version 2.1 or later; see
`COPYING.LGPLv2.1`. The wrapper source is provided in `resampler.c`, and the exact
corresponding FFmpeg source can be reconstructed by the pinned build script. This is an
engineering experiment, not a legal opinion; production distribution should receive a
license-compliance review.
