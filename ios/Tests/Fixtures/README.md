# Synthetic simulator media

These are UI/encoder regression inputs, not model predictions or canonical inference parity evidence.

Both base64 files are intentional checked-in test inputs (about 28 KB each),
derived from Android's synthetic instrumentation clip. They contain no personal
recordings or saved user projects. Keeping them fixed avoids requiring FFmpeg
or regenerating media before every build.

- ios-editor-fixture.mp4.b64: twenty stream-copied repeats of Android androidTest/assets/overlay-fixture.mp4.b64 (one second), 320x240, 20 seconds.
- ios-export-fixture.mp4.b64: same media with explicit limited-range Rec.709 H.264 VUI/container tags added using ffmpeg h264_metadata. The original Android fixture is untagged and deliberately fails the production tagged-SDR compositor guard. No frame payload was re-encoded; color interpretation is explicitly a test convention.

Source preparation decodes these small checked-in base64 assets into the app bundle. SimulatorParityFixture is enabled only in DEBUG Simulator with --parity-editor. DeviceExportCheck exports source intervals [1,6), [6.5,11), [14,18), retaining a joined gap and a separate ignored gap, with score changes/point timeline and native chapters.

The decoded copies belong to the generated `ios/Fixtures/` staging directory,
which is ignored. Build archives, result bundles, screenshots, reference-project
exports and real match recordings also stay outside Git. Do not remove these
source fixtures without updating `scripts/prepare-source.py`, the UI fixture
loader and the rendering integration tests that consume them.
