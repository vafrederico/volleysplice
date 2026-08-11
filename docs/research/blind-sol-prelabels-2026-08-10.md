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

The batch was still running when this decision record was opened. Final per-recording counts, confidence distributions, ambiguity totals, and repeatability against the shared grass-video run are appended after validated artifacts are materialized.
