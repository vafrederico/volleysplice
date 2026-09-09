# VolleySplice Play Store submission pack

## Current brand graphics

- `app-icon-512.png` — 512×512 PNG, existing launcher artwork
- `feature-graphic-1024x500.png` — 1024×500 RGB PNG

The feature graphic now uses **VolleySplice** and **Bump. Set. Splice.** See
[`../../docs/brand.md`](../../docs/brand.md) for the selected artwork and exact
built-in image generation prompts. The icon contains no wordmark and is retained.

## Current screenshots — refreshed 2026-09-08

The current phone and tablet sets are genuine captures of VolleySplice 0.10.12
from the minified, non-debuggable `screenshot` build on the Pixel 10 Pro Android
17 emulator. The isolated package is `com.volleycut.nativeanalysis.screenshot`.
No production install, release signing, or Play Console upload is involved.

- `screenshots/phone-upload-featured/` — eight ordered 1080×2160 RGB PNGs with tutorial-led Play Store copy
- `screenshots/phone-release-sources-v2/` — eight unretouched 1080×2160 source captures at 400 dpi
- `screenshots/phone-upload-final/` — historical captures of the old brand; do not upload
- `screenshots/tablet-release-sources-v1/` — eight ordered, Play-ready 2560×1600 landscape PNGs showing the desktop-mode tablet layout

The featured upload set turns real app captures into the same ordered story as the
in-app tutorial: choose a game video, fine-tune optional settings, understand the
final-video card, check score markers, start with review queues, inspect the whole-game
timeline, fix one clip, and add anything VolleySplice missed. Large branded headers make
each section readable in the Play Store thumbnail while the underlying product UI
demonstrates the exact feature.

The tablet set was captured with the emulator display set to 2560×1600 at 240 dpi
using the release-equivalent `screenshot` build. It demonstrates the
large-screen three-column review workspace, resizable player, two-row game timeline,
score and rally controls, Settings, Export, saved projects, and new-project setup.

## Text and declarations

- `store-listing.md` — title, descriptions, category, contact fields, release notes, and alt text
- `play-console-answers.md` — Data safety, content, audience, app access, and foreground-service answers
- `release-audit.md` — prioritized release-readiness findings and final checklist

## Legal pages

- `../../prod/public/privacy.html`
- `../../prod/public/terms.html`

After the production site is built and deployed, the expected URLs are:

- `https://volleycut.vafrederico.com/privacy.html`
- `https://volleycut.vafrederico.com/terms.html`

These are shared policies for the Android app and browser-based web app.

The Android app includes Privacy and Open source links.

## Original artwork provenance (before the rebrand)

The feature graphic was generated with the built-in image generation tool using the existing `logo.png` as a brand reference. Final prompt:

> Create a polished wide Google Play feature graphic for VolleyCut, an offline Android app that automatically finds volleyball rallies in a user-selected recording and helps trim them into a highlight video. Preserve the recognizable volleyball, video timeline, scissors motif, navy and electric-blue palette, and exact VolleyCut spelling. Use a clean deep-navy-to-blue abstract court-line background with subtle motion trails and a video-editing timeline. Keep the design crisp, modern, restrained, and uncluttered. No device mockup, rating badge, pricing, Google Play mark, claim, extra copy, or watermark.

The generated source is `feature-graphic-source.png`; the upload file was mechanically normalized to Play's exact 1024×500 RGB requirement.

## Featured screenshot provenance

The eight files in `screenshots/phone-upload-featured/` use exact release-equivalent
app captures from `screenshots/phone-release-sources-v2/`. The shared backdrop was
generated with the built-in image generation tool; the official logo, exact copy,
and unmodified app captures were then composed deterministically by
`scripts/compose_featured_screenshots.py`. The exact headline/supporting-copy pairs
follow the tutorial sections and are:

1. `Bump. Set. Splice.` / `Choose your game video. Find the rallies. Keep the moments that matter.`
2. `Fine-tune only when you need to` / `Automatic cleanup, clip padding, short breaks, and review sensitivity live in Settings.`
3. `Save the finished video` / `Save the final video when the review looks right—or create YouTube chapters.`
4. `Check the score markers` / `Correct which side serves, add missed serves, and record team side switches.`
5. `Start with what needs attention` / `Review automatic cleanup, clips, and serves; ignored footage stays out of the queues.`
6. `Review the suggested clips` / `Each timeline block is a clip planned for the final video—select one to check it.`
7. `Fix one clip` / `Keep or remove a rally, then adjust padding, rally length, or split at the playhead.`
8. `Add anything VolleySplice missed` / `Mark a missed rally, or leave out camera gaps, breaks, and other unusable footage.`

The generated background prompt was:

> Create a polished, restrained portrait background for VolleyCut using a deep-navy indoor volleyball court, subtle electric-blue court lines, faint net geometry, and a small coral-orange accent glow. Preserve generous clean negative space for store copy and a darker lower region for a real app screenshot. Background only: no phone mockup, app interface, logo, text, icons, badges, people, or watermark.

The screenshot build inherits the minified, non-debuggable release configuration
and uses an isolated `.screenshot` application ID with the standard debug
certificate because Android cannot install an unsigned APK. The production
`app-release-unsigned.apk` remains unsigned and untouched.

The setup and review captures were generated from the complete `Z:\1080p60.mp4` match. Because
the Pixel 10 Pro AVD cannot decode that source's H.264 stream, it was transcoded at
full duration to an emulator-compatible MPEG-4 copy before import. VolleyCut then
decoded 33,175 video frames, generated 4,424 feature rows, found 61 rally clips and
50 serve markers, and calculated the displayed 11–13 score at 9:25.9. The
screenshot build replaces only the decoded video layer with black; the scoreboard,
point history, playhead, rally ranges, cleanup suggestions, and score are real app
output from that analysis.

For the September refresh, the existing `1080p60.model-feedback.json` analysis
was imported through the app and reconnected to the matching 371,854,171-byte
`Movies/1080p60.mp4` MPEG-4 source on the emulator. Its duration is 1105.833 seconds.
The phone score view is at 1:51.1 (1–2); the main tablet workspace is at 9:26.8
(11–13). Counts and review decisions come from the imported project, not fixtures
invented for the screenshots. Setup is captured before choosing another source.

Regenerate the eight upload images with Python and Pillow:

```powershell
python android/play-store/scripts/compose_featured_screenshots.py
```

The compositor removes only Android's system bars from the phone captures and
fits the complete remaining viewport inside the frame. It does not retouch UI
text, scores, or controls. The tablet PNGs are direct full-display captures.
`screenshots/capture-manifest.json` records build identity and image hashes.
Keep the historical `phone-upload-final` set out of the upload selection.
