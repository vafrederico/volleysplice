# Player-motion engineering qualification — 2026-09-22

**Decision: reject the current player-feature arm before full extraction or neural training.**
The CPU extractors produce aligned, finite data, but the available pretrained person
detectors do not yet provide sufficiently reliable anonymous player observations.
This does not establish that valid player or court-motion features cannot improve
rally recall. The mobile CNN and temporal-model experiments are independent.

Qualification used the fixed first 30 seconds of `grass-source-03` and
`indoor-source-01`, without consulting rally labels. These are engineering
examples, not a person-detection benchmark. No detector precision/recall or rally
accuracy is claimed. All partial caches have `trainingEligible=false`.

| Frozen pilot | Grass / indoor CPU wall seconds per 30 seconds | Result |
| --- | --- | --- |
| Pinned MediaPipe person adapter, four tiles | 10.45 / 10.65 | Extensive duplicate candidates and obvious non-person locations. Rejected. |
| OpenCV Zoo NanoDet 416, four tiles | 23.56 / 21.49 | Better person locations, but duplicate and partial tile-border boxes. Rejected. |
| Same NanoDet, one full ROI | 7.31 / 7.16 | Nested duplicate boxes, indoor wall-photo/net-post false positives, and distant-player misses. Rejected. |

The final pilot preserves the pretrained confidence threshold 0.35 and NMS IoU
threshold 0.6, COCO person class 0, a global cap of 24, 2 Hz detections, and 4 Hz
flow/aligned output. It has no per-side cap. Both sources have 120 aligned ticks
and 49 finite features; 36 grass and 45 indoor features vary. No detector cap is
saturated. A proxy far-half track is available on only 1.67% of grass ticks and
98.33% of indoor ticks. These are availability diagnostics, not person recall;
the fallback image midline is not a measured court net.

The duplicate issue is not a demonstrated coordinate-decoding bug. The pinned
official implementation converts corner boxes to width/height boxes correctly
before OpenCV NMS; the adapter uses its official preprocessing, decoding,
letterboxing and inverse mapping. Nested boxes can have high containment but
IoU below the frozen 0.6 suppression threshold. For example, grass at 10.5 seconds
contains a visually duplicate pair with IoU 0.487 and 96.6% containment; indoor at
20.5 seconds contains pairs with IoU 0.296 and 0.582. They survive NMS by design.
No score, NMS or containment threshold sweep was performed.

The originally proposed candidate was **NanoDet-Plus-m 320**. The actual fallback
is the exact OpenCV Zoo `object_detection_nanodet_2022nov.onnx` 416 graph, pinned
at upstream commit `47534e27c9851bb1128ccc0102f1145e27f23f98`. Its ONNX SHA-256 is
`4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186`.
Although the upstream README names a Plus variant, the graph has 941,863
initializer elements and three output scales, inconsistent with the advertised
2.44M architecture. We identify the tested graph by bytes, and do not present
these results as a verified NanoDet-Plus-m 320 evaluation. Model, source wrappers
and Apache 2.0 license are pinned in the NAS asset receipt. See the
[official model directory](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_nanodet).

The next specific qualification should verify an actual official NanoDet-Plus-m
320 checkpoint/config/export and its preprocessing/postprocessing before a
similarly bounded pilot. Alternatively, first create a small, diverse person/court
observability set covering near/far players, occlusion, duplicate boxes, spectators
and posters. Freeze the detection policy before evaluating it. Reliable occupancy,
formation and synchrony features need this evidence before another rally fit.
Neither step has been executed. No broad detector sweep is planned here.

Reproducibility assets are retained under
`private-reference-0174`:

- `engineering/player-person-qualification-rejection-v2.json` is the immutable
  consolidated decision, including `rejected=true`, all three pilot indexes,
  six cache receipts, source/config/model/runtime hashes and overlay identities.
  SHA-256: `194016a58974d5f89d2e791b1d67d256c37efc0e4066d91399dc64a819a69cfc`.
- `engineering/player-overlay-audit-v1/`,
  `engineering/player-overlay-nanodet-v1/` and
  `engineering/player-overlay-nanodet-single-v1/` contain the rejected overlays.
- The single-ROI folder includes `grass-source-03-maximum.jpg` at 10.5s and
  `indoor-source-01-maximum.jpg` at 20.5s, plus the fixed first frame of each.
- `player-pilot/`, `player-pilot-nanodet-v1/` and
  `player-pilot-nanodet-single-v1/` contain the unmodified partial caches.

The extractor, NanoDet adapter and 24 passing unit tests remain research plumbing.
They are not enabled in production or used for a person-feature training run.
