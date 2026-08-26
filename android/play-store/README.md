# Play Store submission pack

## Ready-to-upload graphics

- `app-icon-512.png` — 512×512 PNG, existing launcher artwork
- `feature-graphic-1024x500.png` — 1024×500 RGB PNG
- `screenshots/phone-upload-featured/` — four ordered 887×1774 RGB PNGs with feature-focused Play Store copy
- `screenshots/phone-upload-final/` — eight 1280×2560 RGB source/reference captures from the signed release build

The featured upload set turns real app captures into a consistent store-listing
sequence: local video selection, automatic rally detection, precise rally editing,
and offline MP4 export. Large branded headers make each value proposition readable in
the Play Store thumbnail while the underlying product UI demonstrates the feature.
The original source file is unavailable, so no personal video frame or thumbnail
can appear. The filename was confirmed non-sensitive by the owner.

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

The four files in `screenshots/phone-upload-featured/` were generated with the
built-in image generation tool from real app captures, including the signed-build
captures in `screenshots/phone-upload-final/`. The shared direction was a deep-navy and
electric-blue volleyball-court backdrop, the official VolleyCut logo, bold
white-and-orange feature copy, and a large recognizable view of the source app UI.
The exact headline/supporting-copy pairs are:

1. `Choose your match` / `Your video stays on your device`
2. `Find rallies automatically` / `On-device video + audio analysis`
3. `Fine-tune every cut` / `Trim, split, pad, or add missed rallies`
4. `Export your highlight reel` / `Private, offline MP4 creation`

The source UI, labels, values, and feature claims were explicitly constrained to
remain faithful to the supplied captures. The originals remain checked in for
audit and future regeneration.
