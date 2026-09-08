# VolleySplice Play Store submission pack

## Current brand graphics

- `app-icon-512.png` — 512×512 PNG, existing launcher artwork
- `feature-graphic-1024x500.png` — 1024×500 RGB PNG

The feature graphic now uses **VolleySplice** and **Bump. Set. Splice.** See
[`../../docs/brand.md`](../../docs/brand.md) for the selected artwork and exact
built-in image generation prompts. The icon contains no wordmark and is retained.

## Historical screenshot sets — refresh before the rebrand release

These are genuine captures of earlier VolleyCut releases. They retain the old
in-app name and must be recaptured from the rebranded screenshot build before a new
Play submission. The composition script is updated to use the new logo and name.

- `screenshots/phone-upload-featured/` — eight ordered 1080×2160 RGB PNGs with tutorial-led Play Store copy
- `screenshots/phone-release-sources-v2/` — eight native-resolution source captures from the release-equivalent screenshot build
- `screenshots/phone-upload-final/` — eight legacy 1280×2560 source/reference captures
- `screenshots/tablet-release-sources-v1/` — eight ordered, Play-ready 2560×1600 landscape PNGs showing the desktop-mode tablet layout

The featured upload set turns real app captures into the same ordered story as the
in-app tutorial: choose the game window, fine-tune optional settings, understand the
final-video card, check score markers, start with review queues, inspect the whole-game
timeline, fix one clip, and add anything VolleyCut missed. Large branded headers make
each section readable in the Play Store thumbnail while the underlying product UI
demonstrates the exact feature.

The tablet set was captured directly from the Pixel 9 Pro XL's 2560×1600 overlay
display using the release-equivalent `screenshot` build. It demonstrates the
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

The Android app still needs an in-app privacy link or privacy text before submission.

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

1. `Choose the part with the game` / `Use the full video, or mark exactly where the game starts and ends.`
2. `Fine-tune only when you need to` / `Automatic cleanup, clip padding, short breaks, and review sensitivity live in Settings.`
3. `Save the finished video` / `Save the final video when the review looks right—or create YouTube chapters.`
4. `Check the score markers` / `Correct which side serves, add missed serves, and record team side switches.`
5. `Start with what needs attention` / `Review automatic cleanup, clips, and serves; ignored footage stays out of the queues.`
6. `Review the suggested clips` / `Each timeline block is a clip planned for the final video—select one to check it.`
7. `Fix one clip` / `Keep or remove a rally, then adjust padding, rally length, or split at the playhead.`
8. `Add anything VolleyCut missed` / `Mark a missed rally, or leave out camera gaps, breaks, and other unusable footage.`

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
