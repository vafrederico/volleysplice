# Third-party notices

VolleySplice source code is licensed under the MIT License. The components and
assets below remain under their own licenses or usage terms; the project MIT
license does not replace those terms.

Release builds must generate their final notice inventory from the exact web,
Android, iOS, and Python dependency graphs. This file identifies the direct and
vendored components known to be distributed by the current project.

## Web applications

| Component | Version | License | Source |
| --- | --- | --- | --- |
| Mediabunny | 1.53.1 | MPL-2.0 | <https://github.com/Vanilagy/mediabunny> |
| OpenCV.js (`@techstark/opencv-js`) | 4.12.0-release.1 | Apache-2.0 | <https://github.com/TechStark/opencv-js> |
| React | 19.2.8 | MIT | <https://github.com/facebook/react> |
| ReactDOM | 19.2.8 | MIT | <https://github.com/facebook/react> |
| fft.js | 4.0.4 | MIT | <https://github.com/indutny/fft.js> |
| Next.js (lab application) | 16.3.5 | MIT | <https://github.com/vercel/next.js> |

The production build copies the available license texts for bundled browser
dependencies to its public `/licenses/` directory. The pinned package lockfiles
identify the exact corresponding source versions. In particular, recipients of
the minified Mediabunny code can obtain its corresponding source at the pinned
upstream version above.

## FFmpeg libswresample WebAssembly

The separately replaceable WebAssembly module under
`vendor/libswresample-wasm/` contains FFmpeg 7.1.5 `libswresample` and
`libavutil`, built without GPL or nonfree components. It is licensed under
LGPL-2.1-or-later, not MIT.

The repository includes the LGPL text, project wrapper source, exact build
script, upstream archive SHA-256, and output hashes. See
`vendor/libswresample-wasm/README.md` and
`vendor/libswresample-wasm/COPYING.LGPLv2.1`. The production build also exposes
the applicable material under `/licenses/`.

## Android application

The Android application compiles OpenCV 4.12.0, AndroidX, Jetpack Compose,
AndroidX Media3, and Kotlin runtime components, primarily under Apache-2.0.
The app's Open-source notices view identifies these components and links to the
license text. Exact transitive versions are determined by the pinned Gradle
configuration and the release-runtime dependency report.

The settings vector is derived from the Google Material settings icon and is
used under Apache-2.0.

## iOS application

The iOS application links an OpenCV 4.12.0 framework prepared by
`ios/scripts/prepare-opencv.ps1`. OpenCV 4.5 and later are licensed under
Apache-2.0. The app's Open-source notices view identifies OpenCV and links to
the license text. Apple system frameworks and codecs are supplied under their
platform terms.

## Python analysis environment

The source distribution pins NumPy 2.2.4 (BSD-3-Clause) and
opencv-python-headless 4.12.0.88 (OpenCV Apache-2.0 plus notices for components
bundled in the wheel). The repository does not redistribute those wheels.
Anyone redistributing a packaged Python environment must include the license
material supplied with the exact wheels they distribute.

## Google Play badge

`prod/public/google-play-badge.png` is an official Google Play brand asset. It
is not MIT-licensed project artwork. Its use is governed by Google's current
Google Play badge and brand guidelines:
<https://play.google.com/console/about/brand-and-marketing/>.
