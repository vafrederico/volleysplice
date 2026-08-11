# Blind GPT-5.6 Sol audiovisual prelabels — 2026-08-10

## Decision

Use a blind GPT-5.6 Sol xhigh pass to accelerate annotation, but treat every interval as an unvalidated suggestion. Keep candidates and editor-ready prelabels separate from human drafts, and never use the model pass itself as evaluation truth.

This increment answers a narrower question than the learned live/dead classifier: can the general model inspect each complete proxy with visual and audio evidence and produce a useful first set of serve-contact-to-dead-ball intervals for a person to correct?

## Relationship to the shared ChatGPT run

The reference conversation, [Volleyball Serve Timestamps](internal-reference-removed), did not send a native video stream directly through a video-capable model endpoint. Its sandbox used OpenCV/FFmpeg-style preprocessing, audio novelty/transient measurements, sampled frames and contact sheets, followed by model visual reasoning. The run initially proposed 42 starts for one 22:08 grass video, then corrected the set to 41 by removing the final 1309.2-second candidate.

That distinction matters for reproducibility. The official [GPT-5.6 Sol model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol) lists image input but not audio or video input. Our blind agents therefore reproduced the useful behavior of the ChatGPT sandbox workflow: inspect the complete MP4, derive audio and visual evidence locally, and reason over sampled imagery. The candidate metadata describes this as audiovisual analysis, not native video-token input.

## Blind-run isolation

Three GPT-5.6 Sol xhigh agents received disjoint sets of three full MP4 proxy paths. Their prompt supplied only the operational event definition:

- enumerate every rally, including aces and service faults;
- start at serve-ball contact;
- end at the first instant live play is over;
- use visual and audio evidence across the complete file;
- attach independent high/medium/low confidence and ambiguity notes.

They were explicitly forbidden to inspect the VolleyCut worktree, research, manifests, code, existing human labels, trained models, or prior predictions. This prevents accidental answer leakage, although the nine outputs still share the same model family and are not statistically independent annotations.

## Artifact and safety design

Raw blind outputs live under:

```text
/mnt/freenas/volleycut/direct-sol-prelabels-2026-08-10/results/
```

The importer verifies candidate schema, exact task/video identity, duration, finite and in-range boundaries, confidence vocabulary, ordering, and absence of overlap. It refuses to overwrite a changed prelabel. Validated editor documents live under:

```text
/mnt/freenas/volleycut/labeling-v1-2026-08-09/prelabels/sol-xhigh/
```

The editor loads an AI prelabel only if `labels/full/<recording-id>.labels.json` does not exist. The first direct save creates that human draft; later loads always prefer it. AI-origin rows remain tagged with their boundary confidences and source provenance.

## Validation protocol

For each recording, the reviewer must play the complete video and verify both boundaries for every suggested rally. They must also add missed rallies, remove non-serve false positives, resolve ambiguities, and record ignored spans where an exact decision is impossible. Only a continuously reviewed, human-completed document can enter a training manifest.

When human labels are complete, assess the prelabels without tuning them against those labels:

- serve-start precision/recall and absolute timestamp error at 0.25, 0.5, and 1.0 seconds;
- interval match precision/recall and IoU;
- missed short service faults and aces;
- errors stratified by environment and AI confidence;
- human correction time compared with blank-slate labeling.

The speed comparison is necessary: even accurate-looking prelabels are not useful if exhaustive verification and deletion take as long as independent annotation.

## Results

All nine candidates passed the importer and were materialized as isolated prelabels. The pass contains **388 proposed rally intervals across 2.52 hours of video**.

| Recording | Rallies | Serve H/M/L | End H/M/L | File ambiguities |
| --- | ---: | ---: | ---: | ---: |
| `beach-source-02` | 42 | 25/15/2 | 14/27/1 | 3 |
| `beach-source-01` | 38 | 25/13/0 | 23/14/1 | 3 |
| `grass-source-04` | 40 | 14/18/8 | 39/1/0 | 4 |
| `grass-source-01` | 52 | 0/42/10 | 21/27/4 | 5 |
| `grass-source-09` | 47 | 26/20/1 | 28/17/2 | 4 |
| `grass-source-10` | 44 | 12/29/3 | 35/9/0 | 4 |
| `indoor-source-01` | 47 | 13/30/4 | 39/8/0 | 3 |
| `indoor-source-07` | 37 | 20/14/3 | 29/8/0 | 3 |
| `indoor-source-05` | 41 | 0/23/18 | 0/38/3 | 6 |

The two explicitly write-first files (`GYU` and `tds`) correctly advertise no high-confidence starts. That is desirable provenance, but it also means they deserve priority review.

### Repeatability warning

The shared ChatGPT conversation's corrected `GYU` list has 41 serve starts; the blind repository run proposed 52. A chronological one-to-one comparison matched only 6 starts within 0.5 seconds, 9 within 1 second, 13 within 2 seconds, and 16 within 3 seconds. This comparison has no human ground truth, so it cannot identify which list is more accurate. It does establish that direct model passes are not repeatable enough to use as truth or unattended training labels.

### Execution lessons

The initial orchestration was inefficient: three videos were assigned to each agent, exact tenths were refined before any file was written, and interrupted turns lost unflushed reasoning. The successful recovery used these rules:

- one video per active reviewer;
- derive reusable 1 fps frames, sparse contact sheets, mono audio, and transient features once;
- write a full-coverage 0.5–1 second candidate JSON before dense refinement;
- use confidence and ambiguity notes instead of withholding questionable short points;
- refine only short windows around uncertain boundaries;
- validate and import each file immediately;
- resume from disk rather than repeating evidence extraction.

These rules are codified in the repository skill at `.agents/skills/analyze-volleyball-video/`.

The reusable evidence pack probes and pins the absolute source path, size, modification time, duration, dimensions, frame rate, and audio presence. It derives 1 fps width-limited frames, sparse 3-second contact sheets, mono 16 kHz PCM audio, and 50 ms audio rows containing RMS, peak amplitude, zero-crossing rate, positive RMS novelty, and a bounded robust transient score. Optional 4 fps windows cover only ambiguous start/end neighborhoods. Completion markers make each stage resumable without modifying the MP4.

Frame-index times are sampling-grid approximations and must be checked against the video clock before claiming precise boundaries. Audio novelty uses absolute RMS-change noise statistics and a bounded score so near-constant tracks do not turn floating-point/codec variation into repeated high-confidence impacts.

### Reviewer cues and feature hypotheses

Across the blind reports, serve evidence combined toss/arm acceleration, first outbound ball flight, a receiving formation breaking into coordinated tracking, and an aligned audio transient. Distant/offscreen contacts were back-timed from receiver response. Preparation bounces and unrelated gym/court impacts were rejected when no organized receiving response followed.

The strongest rally-end cue was an abrupt state change from court-wide tracking/defensive motion to collective stand-down, upright walking, retrieval, huddle, or formation reset. Audio cadence collapse and reactions helped bracket the moment, but reverberation made visual stop/reset evidence decisive indoors. Several seconds of context prevented brief low-motion exchanges from splitting long rallies and preserved one-contact faults/aces.

The most promising reusable features are therefore court-aware temporal ones: service-zone occupancy and ball proximity; toss/striking-arm optical flow; receiver reaction synchrony; court-ROI motion energy; ball/net trajectories; audio onset novelty and cadence; collective pose relaxation; centroid contraction into a huddle; retrieval/walking behavior; and blur/occlusion/camera-motion quality gates. A hysteretic state model should fuse these signals and explicitly support immediate-result points. See the skill's `references/evidence-signals.md` for the full cue and heuristic backlog.

## Artifacts

- Raw blind candidates and reports: `/mnt/freenas/volleycut/direct-sol-prelabels-2026-08-10/results/`
- Editor-ready prelabels: `/mnt/freenas/volleycut/labeling-v1-2026-08-09/prelabels/sol-xhigh/`
- Reusable workflow: `.agents/skills/analyze-volleyball-video/`
