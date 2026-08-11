# Audiovisual evidence and feature candidates

## Serve-contact evidence

Use several synchronized cues:

- visible toss, approach, arm swing, and ball contact;
- first coherent outbound ball flight from the serving baseline;
- receiver's first coordinated reaction or the receiving formation breaking;
- a sharp audio transient aligned with flight/reaction;
- a setup lull before contact and continuous play immediately afterward.

For offscreen or distant servers, back-time from the first ball flight and receiver response, anchored by audio. Lower confidence when those cues disagree.

Reject preparation bounces, ball handling, tosses, nearby-court contacts, warmups, and camera motion when they lack the immediate organized receiving response. Repeated pre-serve bounces often create stronger audio peaks than the serve itself.

## Rally-end evidence

- visible ball down/out or terminal contact;
- a dive or defensive action with no recovery into continued play;
- contact cadence stopping after the terminal impact;
- multiple players relaxing or standing down together;
- transition from play to retrieval, celebration, or regrouping;
- formation reset, server change, or side switch;
- whistle as supporting evidence, never the sole signal in noisy venues.

End at the earliest dead-ball evidence, before celebration and retrieval. Lower confidence when blur, occlusion, camera disruption, or the recording boundary hides the terminal moment.

## Temporal reasoning

- Require a serve candidate to be followed by a coherent live-play cluster.
- Treat a sustained sequence of ball contacts, player reactions, and court-directed motion as one rally even when sparse survey frames miss individual touches.
- Keep short one-contact sequences: they may be service faults or aces.
- Use dead-time gaps and formation resets to separate adjacent rallies.
- Review a small window around the proposed start/end at 2–5 fps; do not densely decode the complete video.

## Signals already useful in manual review

### Audio

- short-window RMS and peak amplitude;
- positive RMS novelty and spectral flux;
- transient density/cadence during play;
- time since the previous strong transient;
- post-contact silence or cadence collapse;
- whistle-like narrowband energy;
- crowd/noise-floor estimate to down-weight unreliable peaks.

### Visual

- global and court-ROI optical-flow energy;
- flow direction from serving baseline toward receivers;
- player pose/centroid velocity and synchronized reaction onset;
- receiving-formation dispersion after contact;
- ball presence, velocity, and trajectory when detectable;
- count and cadence of abrupt local motions near players;
- collective stand-down/pose relaxation probability;
- transition from court-directed play to walking/retrieval;
- camera motion, blur, occlusion, and focus-quality gates;
- court/net geometry and near/far baseline ROIs.

### Context

- setup-lull duration before a candidate serve;
- elapsed live-state duration;
- target-score/side-switch context when available;
- previous and next candidate spacing;
- environment and camera-distance calibration.

## Practical heuristic baseline

1. Generate audio-onset candidates and visual reaction candidates independently.
2. Propose a serve when an onset and coordinated receiver/ball motion align within a short window and follow a setup lull.
3. Reject bounce/handling sequences without organized receiving motion.
4. Enter a live-state with hysteresis; sustain it through intermittent missed contacts using motion and transient cadence.
5. Exit on terminal impact plus collective stand-down/cadence collapse. Permit a short immediate exit for faults and aces.
6. Apply camera-quality gates and emit confidence rather than deleting uncertain candidates.

Tune thresholds only on training/validation recordings grouped by source video. Keep the test/challenge groups untouched. Compare each signal family with ablations; cues that help a reviewer are hypotheses, not proven model features.
