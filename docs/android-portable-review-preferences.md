# Android portable review preferences

Android adopts the [shared UI v1 contract](portable-review-preferences.md) without
changing feedback schema v3 or cleanup/export geometry. The contract and shared
synthetic fixture are copied unchanged from the reviewed web implementation;
integration should retain identical files.

`corrections.ui.preferences.renderScoreOnFinalVideo` controls Android's encoded
score output. The independent web preview preference is retained unchanged:
Android currently shares its preview/output switch. Score tracking remains a
separate gate; disabling output retains timeline and fade choices.

Imported speeds in `[0.25, 8]` are stored as doubles and added to both speed menus.
Media3 receives the requested multiplier through its float playback API. There is
no selection of a nearest menu rate. The exact JSON multiplier survives local
drafts and export. The confidence slider retains the original wire probability
until changed. Reset restores Android's existing new-edit defaults while retaining
unknown extension members and the web-only preview preference.

Local drafts advance to version 9 and retain immutable UI JSON. Saved project fade
preferences take precedence over device fallback on reopening. Unsupported UI
versions remain opaque through persistence, edits, reset, and export; the editor
discloses that they are not applied. Changes made locally do not modify that
future envelope.

`PortableReviewPreferencesTest` covers production feedback parse/export/reimport,
the shared fragments, strict validation and bounds, alias precedence, local file
persistence, legacy migration, source revision changes, reset, and future versions.
`PortableReviewPreferencesInstrumentedTest` covers Android file persistence,
fractional speed menu interaction, and the Media3 speed parameter using synthetic
state. These checks do not exercise physical-device playback, browser exchange,
or encoded video output; those remain manual release checks. No inference or
private media is needed.

Integration with the separately developed retained editor-session work must keep
`PortableReviewPreferences.onOpen` in the initial draft path so device fade cannot
override a project preference. Draft copies retain the envelope and fade priority.
