---
name: analyze-volleyball-video
description: Blind audiovisual analysis of volleyball MP4 footage to create validation-ready serve-contact-to-dead-ball candidate JSON, evidence packs, confidence notes, and heuristic feature proposals. Use when Codex must inspect volleyball videos without existing labels or predictions, reproduce a direct-Sol/ChatGPT-style video review, generate VolleyCut AI prelabels, or analyze frame/audio cues for serve and rally detection.
---

# Analyze Volleyball Video

Create coverage-first rally candidates from only the assigned video and its derived audiovisual evidence. Treat every result as an unvalidated prelabel.

## Preserve blindness

- Inspect only the assigned MP4 and evidence derived from it.
- Do not inspect existing labels, manifests, predictions, trained models, evaluation results, or project research while producing candidates.
- When delegating, assign one video per agent and use a context-free fork. Give the event contract and output schema, not project history.
- State that GPT-5.6 Sol reasons over extracted images and locally derived audio evidence. Do not describe it as native raw-video/audio model input.

## Execute the workflow

1. Derive the recording ID from the complete MP4 basename without `.mp4`, retaining suffixes such as `-full`.
2. Probe the source and create a resumable evidence pack:

   ```bash
   python3 .agents/skills/analyze-volleyball-video/scripts/prepare_evidence.py \
     /absolute/video.mp4 /tmp/volley-evidence/<recording-id>
   ```

3. Review low-density contact sheets and the full audio-feature timeline first. Enumerate every plausible serve-to-dead-ball sequence, including aces and service faults.
4. Write a complete coarse `*.candidates.json` checkpoint before extracting dense windows or tightening tenths. Use medium/low confidence rather than withholding uncertain events.
5. Refine only ambiguous boundaries. Reuse the evidence directory and request short dense windows:

   ```bash
   python3 .agents/skills/analyze-volleyball-video/scripts/prepare_evidence.py \
     /absolute/video.mp4 /tmp/volley-evidence/<recording-id> \
     --window 120.0:128.0 --window 441.0:448.0
   ```

6. Update the same candidate file atomically after each meaningful group of corrections. Never retain several completed videos only in model context.
7. Validate and materialize candidates with the project importer. Read [references/artifact-contract.md](references/artifact-contract.md) for the exact schema and commands.
8. Record file-level ambiguities and actual decision cues in a batch report. Read [references/evidence-signals.md](references/evidence-signals.md) when analyzing cues or proposing deterministic/model features.

## Boundary contract

- Start at serve-ball contact, not the toss, approach, or preparation bounce.
- End at the first instant live play is over, not celebration, retrieval, or repositioning.
- Include very short faults and aces.
- Keep events chronological and non-overlapping.
- Use visual motion/formation changes and audio transients together; do not let crowd noise or preparation bounces define a boundary alone.

## Optimize for useful v0 output

- Prefer full coverage at roughly 0.5–1 second precision over exhaustive sub-frame refinement.
- Limit dense extraction to a few seconds around low-confidence starts/ends.
- Checkpoint one video at a time. Do not batch three videos before the first write.
- If interrupted, resume from the candidate JSON and ambiguity list rather than repeating extraction.
- Keep boundary confidence separate for serve contact and rally end.
- Treat extracted frame-index times as sampling-grid approximations. Verify final candidate boundaries against the video clock; do not claim frame-exact source PTS from the index.

## Validate the result

Structural validation proves only that a candidate is safe to load. It does not measure accuracy or recall. Require a person to review the entire continuous video, correct both boundaries, add missed rallies, remove false positives, and resolve ignored spans before using the document as training or evaluation truth.

An empty `events` array is valid when a complete blind pass finds no plausible rally. Explain that result in `ambiguities`; it remains an in-progress candidate requiring human review.
