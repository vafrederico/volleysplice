# Volleyball Video Analysis Web App — Feasibility Report

## Verdict

Yes—this is very feasible if the first release is designed as an AI-assisted editor, not a fully autonomous volleyball scorekeeper.

With a stationary, landscape phone showing the complete court from behind the service line:

- Uploading, reviewing, trimming, and exporting are routine engineering.
- Automatically finding most rallies is achievable.
- A short review/correction step should remain.
- Player touches and action labels are possible later with custom training.
- Official-quality error attribution is much harder and should remain human-confirmed.

Commercial products already validate the category. Balltime recommends essentially this exact camera position and automatically removes dead time, but also provides correction tools for misidentified rallies and actions. AutoCut Volleyball similarly detects rally boundaries and lets users adjust them. These are useful proof-of-market, although their published claims are not independent accuracy benchmarks. [Balltime recording and breakdown guidance](https://academy.balltime.com/getting-started/faqs), [AutoCut Volleyball](https://autocut.smokinserver.com/volleyball).

| Capability | Feasibility | Product promise |
|---|---:|---|
| Upload, review, trim, export | Very high | Fully reliable |
| Rally splitting on supported footage | High | Automatic suggestions with review |
| Arbitrary handheld/panning footage | Medium-low | Best effort or reject |
| Player/team tracking within a rally | Medium-high | Add after MVP |
| Touch and skill attribution | Medium | Suggestions with confidence |
| Rally winner and obvious ace/kill/error | Medium | Human confirmation |
| Official-grade errors, line/net calls | Low fully automatic | Manual or assisted |

## Recommended recording contract

The capture requirements will influence accuracy more than sophisticated model changes:

- Stationary tripod, centered behind the end line.
- Landscape, with the complete court and service areas visible.
- Elevated if possible; roughly 8–10 feet is a proven recommendation.
- 1080p at 30 or 60 fps. Prefer 60 fps if player-touch analysis is planned.
- No digital zoom, fisheye, camera panning, or interruptions during a set.
- Keep audio.
- Record each set separately to reduce file size and upload failures.

Balltime publishes very similar specifications, including stationary end-line footage, 1080p, 30/60 fps, and avoiding 4K because of upload size. [Recording optimization guide](https://academy.balltime.com/record-and-upload-your-videos/optimize-your-recordings).

## How the MVP should work

```text
Phone video
    ↓ direct resumable upload
Original storage
    ↓
Proxy generation + rally analysis
    ↓
Suggested rally boundaries and confidence
    ↓
Timeline review: drag / split / merge / delete
    ↓
Combined rally-only video or individual point clips
```

The detector should first solve a simple temporal classification problem:

```text
DEAD → SERVE → LIVE → DEAD
          └────────→ DEAD   immediate service fault
```

For every short video window, combine:

- A learned visual `live/dead/serve` classifier.
- Motion inside the court after compensating for camera shake.
- Player count, movement, formation, and posture.
- Audio cues such as contact, whistle, or celebration.
- Optional ball visibility and scoreboard OCR.

No individual signal is reliable enough alone. Walking and celebrations defeat motion-only rules; nearby courts defeat audio-only rules; and the volleyball is frequently blurred or hidden.

### Customizable dead-time behavior

The model should detect the objective core rally:

```text
[start of serve contact, end of live play]
```

Then apply user settings separately:

```text
kept interval = [start − pre_roll, end + post_roll]
```

For example:

- 3 seconds before each serve.
- 2 seconds after each point.
- Merge intervals that overlap.
- If exposing one “maximum seconds between points” setting, divide that allowance between the previous rally’s post-roll and the next rally’s pre-roll.

Keep these settings in an edit decision list rather than rerunning analysis or rendering after every adjustment.

## What existing research suggests

The strongest directly relevant result found is a 2024 thesis on automated beach-volleyball editing. It evaluated more than 300 hours of video and trained its best short-frame-sequence classifier on 221 recordings. It reported:

- 93% frame classification accuracy.
- 77.2% rally-interval IoU.
- 99.6% rally precision.
- Correct rally count for 64% of complete matches.
- Average boundary errors of no more than two seconds.

That strongly validates padded automatic editing, while the 64% exact-match figure demonstrates why a review interface is necessary. It is beach volleyball, so the numbers cannot be assumed to transfer directly to indoor phone footage. [Automated cutting of beach volleyball match videos](https://dspace.cvut.cz/handle/10467/114627).

For advanced events, the 2025 VNL-STES project provides code and 6,137 annotations across 1,028 rallies for serves, receives, sets, spikes, blocks, and scores. Its serve spotting was substantially stronger than score-event spotting; score events achieved only 43.05 mTAP and were affected by occlusion and confusion with failed receives. [VNL-STES project](https://hoangqnguyen.github.io/stes/), [code](https://github.com/hoangqnguyen/spot).

Camera quality matters considerably. PathFinderPlus, a single-camera volleyball system, reported 71.24% set-tactic accuracy with a level camera and simple background, falling to 51.52% with an angled camera and noisy background. [PathFinderPlus paper](https://arxiv.org/abs/2309.14753).

## Training-data plan

Public volleyball datasets are overwhelmingly professional broadcast footage or already-trimmed rallies. They are useful for initializing models, but they do not adequately represent a parent’s phone in different gyms.

Recommended approach:

1. Collect 20–30 representative full matches for the feasibility study.
2. Annotate serve contact and end-of-play timestamps.
3. Include warmups, timeouts, celebrations, neighboring courts, camera movement, and service errors as hard negatives.
4. Split testing by complete match, gym, team, and device—not random frames.
5. Expand toward 50–100+ diverse matches for a robust beta.

Every user boundary correction can become valuable training data, but only with explicit permission.

The most important product metrics are:

- Rallies missed or falsely added per set.
- Percentage of matches with the correct rally count.
- Start/end boundary error.
- Live-play seconds accidentally removed.
- Percentage of boundaries needing correction.
- Human review time compared with manual editing.

The system should favor keeping an extra second over deleting live play.

## Advanced analytics roadmap

1. **Court and player tracking:** Calibrate the court, detect players, assign team/libero, and maintain anonymous track IDs within each rally.

2. **Roster identification:** Aggregate jersey-number readings over a whole track rather than one frame. Constrain choices using the roster and rotation, with one-click user correction. Avoid face recognition.

3. **Touch spotting:** Detect serve, receive, set, spike, block, and terminal events, then associate each event location with the most plausible player track.

4. **Ball tracking:** Use a temporal heatmap model on high-resolution court regions, with explicit visible/occluded/unknown states. Use it as supporting evidence rather than the sole source of truth.

5. **Outcomes:** Automate obvious service errors, aces, kills, and attacked balls out. Suggest ambiguous reception, attack, or block errors for confirmation.

6. **Official statistics:** Keep handling errors, net touches, line calls, rotation faults, ambiguous block touches, and responsibility judgments human-reviewed. Some are not visible from the baseline camera or depend on the referee’s call.

This is consistent with Balltime’s own workflow: some end-rally attack results still use a grading wizard instead of relying entirely on automation. [Balltime practice-grading workflow](https://academy.balltime.com/record-and-upload-your-videos/setting-up-your-videos/uploading-practice-videos).

## Practical web architecture

A sensible first stack is:

- React/Next.js timeline editor.
- Direct multipart upload to private object storage.
- Postgres for matches, jobs, timestamps, revisions, and metadata.
- Queue plus containerized Python/FFmpeg workers.
- Low-resolution H.264 review proxy.
- Original-resolution source retained for export and future analytics.
- Optional GPU workers only for later player/ball models.

A 90-minute phone recording can easily be 5–17 GB, so video data should never pass through the ordinary application API. S3 supports direct, independently retryable multipart uploads. [AWS multipart uploads](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html).

Use `ffprobe` to normalize rotation, variable frame rate, HEVC/HDR, and audio metadata. For previews, play the proxy according to the edit list. Render only when the user requests an export. Exact arbitrary cuts generally require one final decode/re-encode; stream-copy cuts are limited by keyframes. [FFmpeg documentation](https://ffmpeg.org/ffmpeg.html).

For youth footage, make everything private by default, use short-lived access links, provide deletion and retention controls, strip location metadata, and require separate consent before using uploads for model training. FTC guidance treats video or audio containing a child’s image or voice as personal information when COPPA applies. [FTC COPPA guidance](https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions).

## Estimated effort and cost

Planning estimates for one experienced full-stack/video engineer with ML support:

- 2–3 weeks: data collection and feasibility prototype.
- 6–10 weeks: usable upload, review, automatic suggestions, and export MVP.
- Another 2–4 months: robustness across gyms, phones, and teams.
- Another 3–6+ months: meaningful player/action attribution and analytics.

For a 90-minute, roughly 8 GB match, an AWS-style CPU pipeline with one proxy, one condensed export, and one month of storage is approximately $1.50–$2.50 per processed game before heavy viewing, GPU analytics, and fixed application costs. Treat that as a benchmarking envelope, not a quote. [S3 pricing](https://aws.amazon.com/s3/pricing/), [Fargate pricing](https://aws.amazon.com/fargate/pricing/), [MediaConvert pricing](https://aws.amazon.com/mediaconvert/pricing/). Development, annotation, quality control, and customer support will cost much more than the underlying trimming compute.

## Recommendation

Green-light the project with this scope:

- Supported stationary end-line footage only.
- Automatic, conservative rally suggestions.
- Mandatory fast review/editing interface.
- Configurable pre-roll and post-roll.
- Individual rally and combined-video export.
- No player statistics in the initial critical path.

Before investing heavily, benchmark 10–20 representative recordings against Balltime and AutoCut. If this is only for internal team use, an existing product may be cheaper. If this is intended as a new product, the strongest differentiators would be customizable editing, transparent confidence/corrections, private or on-device processing, open CSV/API exports, and a simpler pay-per-match workflow.
