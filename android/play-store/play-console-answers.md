# Google Play Console answers

Prepared for package `com.volleycut.nativeanalysis` on 2026-08-19.

These answers describe the audited source and dependency set. Recheck them if permissions, SDKs, networking, accounts, cloud features, or monetization are added.

## Data safety

Google Play defines collection as transmitting user data off the device and excludes access or processing that remains only on the device. VolleyCut does not request `INTERNET`, includes no analytics or advertising SDK, and performs its media processing locally.

- Does the app collect or share any required user data types? **No**
- Is any user data shared with other companies or organizations? **No**
- Data types selected: **None**
- Encryption in transit: **Not applicable because the app does not transmit user data**
- Account creation: **No accounts**
- Account deletion URL: **Not applicable**
- Data deletion: Users can delete individual projects and feature caches in the app, clear app storage in Android settings, or uninstall the app. The developer has no server-side copy to delete.
- Privacy policy URL: `https://volleycut.vafrederico.com/privacy.html`

User-initiated exports to a destination chosen through Android's system file picker are not developer collection. If model-feedback files are later uploaded automatically or a cloud feature is added, this answer must change.

## Ads and monetization

- Contains ads: **No**
- In-app purchases: **No**
- Subscriptions: **No**
- Paid app: **No**

## App access

- Are all features available without special access? **Yes**
- Login credentials or reviewer instructions: **Not required**
- Suggested review note: "VolleyCut has no account or network dependency. Choose a local volleyball recording through Android's system document picker. Analysis and export can be time-consuming because all processing happens on-device."

Do not attach a personal recording to the Play listing. For review, prepare a short, rights-cleared synthetic or public-domain volleyball clip that demonstrates analysis and export.

## Content rating

Select the app/utility questionnaire rather than a game questionnaire.

- Violence: No app-provided violent content
- Sexuality or nudity: No
- Language: No
- Controlled substances: No
- Gambling: No
- User interaction or communication: No
- User-generated content sharing: No
- Location sharing: No
- Digital purchases: No

The app opens media selected by the user but does not publish or share it. Answer any console wording about unrestricted user-selected local content literally if the questionnaire changes.

## Target audience

- Recommended age groups: **13–15, 16–17, 18+**
- Designed specifically for children: **No**
- Appeal to children: The listing should use a general sports-video-editor presentation and should not use child-directed characters, language, or imagery.

## Foreground service declaration

Declared type: `mediaProcessing`

**Feature description**

"VolleyCut uses a media-processing foreground service for user-started, on-device analysis of a selected volleyball recording and for user-started MP4 highlight export. The service keeps long video decoding, audiovisual feature extraction, rally inference, and video transcoding running while the user views another screen. A persistent notification shows progress. Export can be cancelled from the notification or the app, and deleting an active project cancels its analysis."

**Impact if deferred**

"The user explicitly starts analysis or export and expects processing to begin immediately. Deferral would leave the selected project or export appearing stalled and would prevent the user from reviewing or sharing the requested highlight video."

**Impact if interrupted**

"Interruption delays the requested result. Analysis checkpoints local feature work and can reuse it on a later run. Export cancellation removes the incomplete destination so the user is not left with a corrupt MP4."

**Required demonstration video**

Play requires a public or unlisted video link for each foreground-service use. Record a short, rights-cleared demonstration that shows:

1. The user choosing a non-personal demo recording.
2. The user tapping **Create & queue** and the analysis notification appearing.
3. Progress continuing after leaving the app, followed by returning to the project.
4. The user queueing an MP4 export, the export notification appearing, and the cancel control.

No demonstration video was created during this audit because the requested screenshot workflow must not open or relink any video.

## App content declarations

- Privacy policy: Required; use the URL above after deployment
- News app: No
- COVID-19 contact tracing or status app: No
- Health app declaration: No health features
- Government app: No
- Financial features declaration: None
- Permissions declaration: No SMS, Call Log, All files access, broad photo/video, location, camera, or microphone permission
- Foreground service declaration: Required for `mediaProcessing`

## Testing-track requirement

If the Play developer account is a personal account created after 2023-11-13, plan for a closed test with at least 12 opted-in testers continuously for 14 days before applying for production access. New accounts may also need real-device verification through the Play Console mobile app.
