# Play Store submission pack

## Ready-to-upload graphics

- `app-icon-512.png` — 512×512 PNG, existing launcher artwork
- `feature-graphic-1024x500.png` — 1024×500 RGB PNG
- `screenshots/phone-upload-final/` — eight ordered 1280×2560 RGB PNGs from the signed release build

The screenshots use the in-app tour to explain the real saved inference, settings, timeline, editing tools, cut list, and export. The original source file is unavailable, so no personal video frame or thumbnail can appear. The filename was confirmed non-sensitive by the owner.

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
