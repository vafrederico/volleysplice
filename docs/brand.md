# VolleySplice brand

The product is **VolleySplice** (formerly VolleyCut). Use `volleysplice` in the
lowercase wordmark and new download filenames. The campaign line is
**Bump. Set. Splice.**

## Where the campaign line belongs

Use it in the first-run/new-project introduction on web and Android, the website
meta description, the README, the Play listing's full description, and the Play
feature graphic. Keep editing controls, progress messages, notifications, error
messages, and policy text focused on their task.

## Assets

- `logo.png`: generated horizontal master, 2159 × 728, opaque white background.
- `public/volleysplice-logo.png`: internal lab wordmark, normalized to 920 × 310.
- `prod/public/runtime/volleysplice-logo.png`: production wordmark, 920 × 310.
- `android/app/src/main/res/drawable-xxxhdpi/volleysplice_logo.png`: native wordmark and splash artwork, 920 × 310.
- `prod/public/runtime/volleysplice-icon-transparent.png` and Android's
  `volleysplice_icon.png`: retained volleyball/video/scissors symbol, without text.
- Existing favicon and launcher images also contain only that symbol and remain valid.
- `android/play-store/feature-graphic-source.png`: generated campaign artwork.
- `android/play-store/feature-graphic-1024x500.png`: the campaign artwork normalized
  to Play's exact 1024 × 500 RGB dimensions.

Web and Android headers use the retained transparent symbol with a native text
wordmark: navy `volley`, blue `splice`, bold italic lowercase lettering. Android
uses just the accessible symbol in the condensed desktop review header to leave
room for editing controls. The white-background horizontal artwork remains on
the white Android splash screen.

## Compatibility and historical names

The following uses of the former name are intentional:

- Android application ID/package `com.volleycut.nativeanalysis`, its debug suffix,
  Play listing URL, intent actions, preference files and notification-channel IDs.
  Retaining the package lets the renamed app update the existing installation.
- Browser database, local-storage and OPFS keys; JSON schema/model kind strings;
  export service-worker URL/protocol; internal API headers. These preserve existing
  projects, imported feedback and in-flight/browser-cached communication contracts.
- Existing production hosts, support email, deployment/service names, SDK override
  variables, signing-key filename/alias and machine/NAS paths. The rebrand does not
  provision a new domain, mailbox, certificate, or deployment.
- Frozen model artifacts, datasets, research reports, external memory-note titles,
  and genuine screenshots of earlier releases. These record the original state.

Privacy-policy and terms links use `https://www.volleysplice.com/privacy.html`
and `https://www.volleysplice.com/terms.html`, as requested for the new domain.
This link update does not provision DNS, hosting, or a new support mailbox.
A future web-origin
move also needs a project transfer plan: unchanged IndexedDB keys do not make
browser storage accessible across origins. Users can export project JSON and
reconnect source media on the new origin.

The current featured phone screenshots, their source captures, and the tablet
screenshots were recaptured from the rebranded Android screenshot build on
2026-09-08. The first featured phone image uses the campaign line. See
`android/play-store/README.md` and the capture manifest for provenance. Only the
legacy `phone-upload-final` directory retains historical UI; do not upload it.

This change prepares the rebrand in source. It does not publish the website, update
Play Console, change the Android version, or sign a release.

## Generated artwork provenance

Both assets were edited with the built-in image generation tool. The wordmark edit
keeps a white background, matching the original assets; an initial attempt that
painted a checkerboard was rejected. Generated originals were retained outside the
repository, and the selected outputs were copied into the paths above.

Wordmark replacement prompt:

> Use case: text-localization. Edit target: existing horizontal VolleyCut production logo. Replace the wordmark "volleycut" with exact lowercase text "volleysplice". Keep "volley" deep navy and "splice" electric blue, and preserve the bold italic sports typography and the left-hand volleyball/video/scissors/timeline symbol. Expand width for the new word so the full logo is visible with a small even margin. Background MUST be plain solid pure white #FFFFFF, completely flat. No checkerboard, no transparency pattern, no texture, no watermark, no campaign line, no extra text. Preserve original symbol design and colors as closely as possible. Production logo, clean sharp lettering.

Final wordmark framing correction:

> Use case: precise-object-edit. Edit target: the attached VolleySplice horizontal production logo. Correct ONLY the framing and the slightly clipped right edge of the final "e". Keep the exact lowercase wordmark "volleysplice", with "volley" navy and "splice" blue, existing bold italic font and volleyball/video/scissors artwork unchanged. Fit the ENTIRE logo inside the canvas with a solid white margin of at least 6 percent of canvas width on BOTH the left and the right, plus a small white top and bottom margin. Scale the whole logo down uniformly to create that space. No ink may touch any edge. The final e must be fully visible and complete. Background is plain solid white #FFFFFF, no checkerboard or texture. Do not add text, icons, campaign lines or watermarks. Keep wide horizontal approximately 3:1 composition.

Final feature-graphic prompt:

> Use case: text-localization. Asset type: Google Play feature graphic, final canvas exactly 1024 by 500 pixels. Edit target: attached existing VolleyCut feature graphic. Rebrand it to "volleysplice": spell v o l l e y s p l i c e exactly, all lowercase, preserving bold italic typography with "volley" white and "splice" electric blue. Keep the established volleyball/video/scissors logo symbol on the left, the dark navy and electric-blue court background and subtle video timeline. Adjust logo scale so the complete longer wordmark fits comfortably within the frame with safe margins. Add the campaign line "Bump. Set. Splice." beneath the wordmark in clean readable white sans-serif, smaller than the brand, but readable at thumbnail size. Preserve polished restrained composition. No other text, no phone mockups, no badges, no watermark. Solid opaque background.

The runtime logos and feature graphic were resized mechanically with Sharp after
generation; their artwork and text were produced by the built-in tool. The master
logo retains its generated resolution. The final wordmark has a clear white margin
so the longer name is not clipped.
