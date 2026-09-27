# YOLOX and SSDLite person-detector qualification

**YOLOX-Nano produced substantially more useful person observations than the
previous detectors, but neither new configuration passed the frozen gate for
player-count and formation features.** YOLOX usually located the main grass
players and many indoor players. It still missed visible distant indoor players
and sometimes produced extra boxes around overlapping people. SSDLite missed
many distant people in both environments. These are visual engineering findings,
not measured detector precision/recall or evidence that person features cannot
improve rally recall.

No person-feature rally model was trained, no thresholds or tiles were swept,
and no production behavior changed. The study used CPU inference only. The
[combined experiment protocol](neural-recall-distillation-protocol-2026-09-23.md)
was frozen before outcomes; its 99% retained-play selection requirement applies
to the separate rally-model experiments, not to an unmeasured detector recall.

## What was tested

| Candidate | Parameters | Official checkpoint bytes | Fixed inference policy |
| --- | ---: | ---: | --- |
| YOLOX-Nano | 912,159 | 7,694,953 | 416×416, one existing ROI, person confidence ≥0.25, class-aware NMS IoU 0.45 |
| SSDLite320 MobileNetV3-Large | 3,440,060 | 14,069,355 | 320×320, one existing ROI, person confidence ≥0.25, official NMS IoU 0.55 |

The checkpoint sizes are the downloaded training/checkpoint files, not measured
mobile graph sizes. Both detectors have a final cap of 24 person boxes, applied
after class filtering. No frame reached that cap. There is no cap by court side,
tiling, tuned containment suppression, or new court geometry.

YOLOX uses the [official Nano configuration](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/exps/default/yolox_nano.py)
at commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`, with the official
`0.1.1rc0/yolox_nano.pth` release. Its source is Apache 2.0. Preprocessing calls
the pinned `ValTransform(legacy=False)`: BGR values in 0–255, aspect-preserving
resize, top-left letterbox filled with 114. Official postprocessing multiplies
objectness by the best-class probability, performs class-aware NMS, then this
adapter retains COCO person class 0, inversely scales, and clips coordinates.

SSDLite uses Torchvision 0.26.0's official COCO_V1 checkpoint and the unmodified
SSD transform, anchors, decoder, and postprocessing. RGB values start in 0–1;
the model applies its own normalization and fixed-size resize. Its person class
is 1 because the taxonomy includes background class 0. Torchvision source is
BSD-3-Clause. See the [official implementation explanation](https://pytorch.org/blog/torchvision-ssdlite-implementation/)
and [pinned implementation](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/detection/ssdlite.py).
No phone/browser export or legal deployment assessment was performed here.

## Scope and review

Four existing non-beach development recordings cover the four exact-label source
groups: `grass-source-03` (source-group-007), `grass-source-01` (source-group-005),
`indoor-source-01` (indoor source), and `indoor-source-08` (source-group-012). Rally labels
were not read or used to choose frames. Protected test sources were excluded.

Each recording contributes eight fixed times: 0, 5, 10.5, 20.5, and 29.5 seconds,
then 25%, 50%, and 75% of duration. The two prior qualification clips, Bauos and
9lcXIf, also contribute 60 frames each at 2 Hz over the first 30 seconds. There
are 32 fixed review frames and 142 distinct frames per detector after overlaps.
Each frame is selected by nearest actual presentation timestamp; seek results,
source content hashes, saved pixels, scores, boxes, and overlay hashes are retained.

All eight fixed-frame contact sheets and all four sequence sheets were visually
reviewed, with individual close-ups for ambiguous overlaps and distant omissions.
The fixed sheets support these observations:

| Source | YOLOX-Nano | SSDLite |
| --- | --- | --- |
| source-group-007 | Usually detects the main visible players, including the farther pair; occasionally includes a person on the adjacent court. Occlusion can create an oversized extra box. | Both clearly visible people are missed at 0s. Distant observations remain sparse; a nested box appears at 632.7s. |
| source-group-005 | Main visible players are generally localized; overlapping/occluded players remain ambiguous. | No detections at 0s and only the foreground person at 5s, 10.5s, and 20.5s despite other visible players. |
| indoor source | More far-player detections, but still omissions; 5.5s has an extra narrow box alongside the blue-shirt player and 20.5s an oversized overlapping box at the foreground walker. | Repeated far-player omissions and occasional nested/partial boxes. |
| source-group-012 | Distant-player omissions recur at 10.5s, 29.5s, 344.67s, and 689.37s; the 29.5s close-up shows the far-right pair without boxes. | Most far players are missing repeatedly; some foreground detections also split into nested boxes. |

There were no obvious poster/net-post hallucinations in these fixed sheets,
unlike the previous NanoDet examples. This is a limited visual observation, not
a false-positive-rate claim. Adjacent-court people are genuine people but are
still unwanted input for main-court player counts.

Sequence diagnostics further illustrate why count changes must not be treated
as reliable player motion. YOLOX's counts range from 2–6 on grass and 4–9 at
indoor source; SSDLite's corresponding ranges are 0–4 and 2–6. The YOLOX count changes
between 22/59 adjacent grass frame pairs and 40/59 indoor source pairs. These include real
movement, entering/leaving the view, occlusion and detector instability; they
are **not labeled error or identity-switch rates**. High box containment is
recorded only as an overlap warning, because two real overlapping people can
also trigger it. No tracks or serving-side predictions were evaluated.

## CPU timing and checks

Two PyTorch/OpenCV CPU threads, batch size one, one synthetic warmup, no GPU:

| Candidate | Forward median / p95 | Preprocess + forward + postprocess median / p95 |
| --- | ---: | ---: |
| YOLOX-Nano | 47.25 / 60.34 ms | 49.36 / 63.81 ms |
| SSDLite | 37.82 / 46.17 ms | 51.91 / 66.14 ms |

These are desktop measurements under the concurrent experiment workload. They
exclude loading, video decoding, ROI cropping and writing overlays. Full study
wall time was 59.61 seconds, including asset verification, model loading/warmup,
source hashing, seeking/decoding, both detectors, and review artifacts. It is not
a sequential whole-video throughput benchmark or a mobile latency estimate.

Five focused tests passed for coordinate clipping, deterministic ordering,
score filtering, caps, invalid outputs and overlap diagnostics. The runtime
audit passed all artifact/source/frame associations and compared the SSDLite
timing adapter against the untouched public `model([image])` path on one real
frame from every source. Boxes and scores matched exactly, including the saved
run outputs. No training, protected-test evaluation or detector accuracy metric
was needed for this qualification.

## Decision and next useful step

The frozen gate rejects repeated visible distant-player omissions or obvious
duplicate/non-person boxes before the proposed count/formation feature arm.
Both configurations trigger it. That gate concerns this particular feature
design; it does not require every useful visual feature to detect every player.

**YOLOX remains a credible candidate for a separate, explicitly noisy-observation
experiment.** Its boxes could locate regions in which to measure optical flow
or summarize confidence-weighted occupancy, with availability/uncertainty inputs
and no assertion that a missing box means an absent player. Before treating
counts, formation, near/far synchrony or serving-side geometry as dependable,
use a small independently annotated observability set containing distant
players, overlaps, adjacent courts, spectators and posters, plus actual court
geometry. A new feature hypothesis should be frozen and evaluated against the
same 99% recall requirement; detector visual quality alone cannot establish a
rally-recall improvement.

## Reproducibility

All downloads, dependencies, caches, frames and results are on the direct NAS
mount under `private-reference-0193`
(`private-reference-0194`).
No package was installed into Windows or the WSL root environment. The runner
uses NAS temporary/cache locations and disables Python bytecode writes.

| Artifact | SHA-256 |
| --- | --- |
| `protocol.json` | `9598e0d446ccbded6521ea139408ebd5f3a5b3e2eac8cec845138116497f4bc7` |
| `asset-receipt.json` | `c4db5f0cdc0b5d5708be7a1c1d04f49852daa1268d0cf855d1f65f6d8e3b45a0` |
| `audit.json` | `bed52899d20ed78dc50afb7a9791b1a21f852ce0ecfe6757429ef579cc18e079` |
| `assets/yolox_nano.pth` | `cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b` |
| `assets/ssdlite320_mobilenet_v3_large_coco-a79551df.pth` | `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2` |

The asset receipt pins all upstream source and weight bytes, source licenses,
runtime implementation snapshots, and the three small NAS-only Python
dependencies. The protocol binds the combined experiment protocol and copies
of the local adapter/runner. `results/report.json` links per-record results;
`results/*-sheet.jpg` and `results/*-sequence-sheet.jpg` contain the visual review
evidence. Original source PNGs and individual scored overlays are retained under
each recording directory.
