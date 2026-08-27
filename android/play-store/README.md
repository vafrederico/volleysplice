# Play Store submission pack

## Ready-to-upload graphics

- `app-icon-512.png` — 512×512 PNG, existing launcher artwork
- `feature-graphic-1024x500.png` — 1024×500 RGB PNG
- `screenshots/phone-upload-featured/` — five ordered 1080×2160 RGB PNGs with feature-focused Play Store copy
- `screenshots/phone-release-sources-v2/` — five 1344×2992 source captures from the release-equivalent screenshot build
- `screenshots/phone-upload-final/` — eight legacy 1280×2560 source/reference captures

The featured upload set turns real app captures into a consistent five-step
store-listing sequence: select the game window, confirm Strong automatic cleanup,
review suggested rallies, fix or add rallies, and export the finished MP4. Large
branded headers make each value proposition readable in the Play Store thumbnail
while the underlying product UI demonstrates the exact feature. The fresh project
was created normally on a Pixel 10 Pro AVD with a synthetic, non-personal fixture.

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

## Generated artwork provenance

The feature graphic was generated with the built-in image generation tool using the existing `logo.png` as a brand reference. Final prompt:

> Create a polished wide Google Play feature graphic for VolleyCut, an offline Android app that automatically finds volleyball rallies in a user-selected recording and helps trim them into a highlight video. Preserve the recognizable volleyball, video timeline, scissors motif, navy and electric-blue palette, and exact VolleyCut spelling. Use a clean deep-navy-to-blue abstract court-line background with subtle motion trails and a video-editing timeline. Keep the design crisp, modern, restrained, and uncluttered. No device mockup, rating badge, pricing, Google Play mark, claim, extra copy, or watermark.

The generated source is `feature-graphic-source.png`; the upload file was mechanically normalized to Play's exact 1024×500 RGB requirement.

## Featured screenshot provenance

The five files in `screenshots/phone-upload-featured/` use exact release-equivalent
app captures from `screenshots/phone-release-sources-v2/`. The shared backdrop was
generated with the built-in image generation tool; the official logo, exact copy,
and unmodified app captures were then composed deterministically by
`scripts/compose_featured_screenshots.py`. The exact headline/supporting-copy pairs
are:

1. `Start with any game video` / `Select the game window in seconds — your video stays on your phone.`
2. `Strong cleanup, ready by default` / `Adjust extra time, short breaks, and how many clips need a check.`
3. `Review every suggested rally` / `See the final timeline and quickly confirm the clips that need attention.`
4. `Fix cuts or add a missed rally` / `Trim clip edges, split a rally, or mark missing action yourself.`
5. `Export one finished highlight video` / `Save a private, offline MP4 that is ready to share.`

The generated background prompt was:

> Create a polished, restrained portrait background for VolleyCut using a deep-navy indoor volleyball court, subtle electric-blue court lines, faint net geometry, and a small coral-orange accent glow. Preserve generous clean negative space for store copy and a darker lower region for a real app screenshot. Background only: no phone mockup, app interface, logo, text, icons, badges, people, or watermark.

The screenshot build inherits the minified, non-debuggable release configuration
and uses an isolated `.screenshot` application ID with the standard debug
certificate because Android cannot install an unsigned APK. The production
`app-release-unsigned.apk` remains unsigned and untouched.
