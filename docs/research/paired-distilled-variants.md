# Preparing both Distilled Large variants

Android now has an opt-in **Prepare both versions (experimental)** setting for
new neural analyses. It produces separate highest-recall and highest-F1 projects
with their own rally boundaries and score results. Choosing a saved project
loads that result instead of starting another analysis. Existing completed
companion projects and reviewed edits are preserved.

This is an execution experiment with the already frozen model bundles, not new
training, calibration, selection, or a changed rally decoder. Web integration is
unchanged. The physical Pixel is unavailable; measurements use the API 37 x86_64
emulator with four CPU cores, 4 GB RAM, and 24 GB of virtual storage. Video files
and emulator storage remain on the configured external storage.

## What is shared

For fresh input, both variants share the source timestamp inventory, MediaCodec
pass, letterbox/image normalization, quality measurements, and four-image queue.
The worker runs the two encoders sequentially on each prepared image, keeping
their embeddings separate. AV features, contextual features, and production
serve/state evidence are generated once. Each frozen TCN/normalizer/decoder then
produces its own rallies. Serving-side and side-switch inference currently run
separately for those two proposal sets.

If AV features are already cached, the paired path currently uses independent
embedding passes. It still prepares both results but does not share those
decodes. Reopening an already completed result requires no embedding generation.
Temporary embeddings are deleted after analysis; switching uses saved results.

## Validation and protocol

Inputs resolve through recording index `recording-044` and external artifact
index `private-reference-0223`. The short excerpt is 120 seconds. The complete
recording is 1,061.016489 seconds. Exact source bytes and the experiment APK are
verified before timing.

The short protocol compares a fresh recall analysis, a subsequent F1 analysis
reusing its AV cache, and a fresh paired analysis. Jobs run sequentially in fresh
app processes. There is one observation per configuration; filesystem/model
warmth and host scheduling are not controlled. Source verification and transfer
are outside the measured readiness interval. There are no discarded pipeline
warmups.

The paired short outputs must exactly match each individual model's embeddings,
quality rows, selected timestamps, audio features, boundaries, confidence, and
both specialist feature matrices and decisions before the full run is allowed.
Synthetic instrumentation also checks rotated fractional windows, persistence,
preservation of an existing companion edit, cancellation, and temporary cleanup.

Readiness includes all feature generation and both score specialists. Encoder,
preparation, and decoder-wait counters overlap within the shared video stage;
they must not be added to its wall time. Saved-result loading measures project
read/validation and editor-seed construction, not UI tap-to-first-paint.

The [runner](../../scripts/benchmark-paired-neural-variants.py) and
[qualification/report script](../../scripts/report-paired-neural-variants.py)
keep raw receipts outside Git and publish only indexed summaries.

## Two-minute results

| Execution | Ready, including score tracking | Additional wait when switching |
| --- | ---: | ---: |
| Recall only, fresh features | 177.258 s | F1 still needs analysis |
| F1 after recall, reusing AV cache | 157.973 s additional | 335.231 s cumulative for both |
| Both prepared together, fresh features | 226.180 s | Saved-result load: median 5.753 ms, maximum 12.386 ms |

Preparing both adds **48.923 s (27.6%)** to the recall-only observation, while
saving **109.050 s (32.5%)** against the sequential recall-then-F1 observations.
Recall had six proposals and F1 seven in this excerpt. Both variants exactly
matched their independent outputs in every short qualification check, including
serving-side and side-switch features and decisions.

| Stage | Recall only | Both prepared |
| --- | ---: | ---: |
| Shared video decoding, AV features and embeddings | 115.397 s | 117.850 s |
| Audio decoding and features | 8.176 s | 7.826 s |
| Context features | 0.059 s | 0.062 s |
| Rally inference stage | 0.411 s | 0.571 s |
| Recall score specialists | 53.082 s | 53.286 s |
| F1 score specialists | — | 46.421 s |

Stage rows exclude small opening/bookkeeping costs. The extra F1 score pass
accounts for 46.421 s of the observed 48.923 s increase; its shared score-frame
decode alone took 42.429 s. The first shared video stage grew by 2.453 s. These
are observations from single runs, not an isolated causal estimate.

Within the paired video stage, image preparation took 4.695 s. Recall encoder
execution took 2.799 s and F1 4.834 s; output readback and writing added 0.884 s
across both. That is approximately **11.7 ms and 20.1 ms of encoder execution per
sampled image**, respectively, over 240 images. These operations overlap with
the video/AV pipeline; adding them to 117.850 s would count them twice. The two
temporal heads took 0.195 s and 0.150 s, respectively.

By contrast, switching to F1 after a completed recall-only run reused all three
AV caches but needed a separate 110.611 s embedding video pass: 5.007 s image
preparation, 2.657 s encoder/readback, and 102.947 s of remaining pipeline work.
That residual includes timestamp inventory and decoding, and is not a pure
decoder measurement. Its score analysis then took 46.569 s. Computing a second
encoder is much cheaper here than reopening and traversing the source video.

## Full-video results

The 1,061.016489-second recording completed both variants and their score results
in **1,777.940 s (29 min 37.940 s)**. Recall's result was ready at 1,487.336 s;
F1's subsequent score pass added 290.604 s. There is no same-session full-video
single-variant control, so the short-run relative speedup must not be assumed to
hold for this longer recording.

| Full paired stage | Wall time |
| --- | ---: |
| Shared video decoding, AV features and both embeddings | 1,072.663 s |
| Audio decoding and features | 54.597 s |
| Context features | 0.257 s |
| Rally inference stage, both variants | 3.643 s |
| Recall score specialists | 355.900 s |
| F1 score specialists | 290.604 s |
| All results ready, including small opening/bookkeeping costs | 1,777.940 s |

The score-frame decodes took **333.009 s + 270.915 s = 603.925 s**, about 34%
of total readiness time. Their remaining serving-side/team-switch evaluations
took 22.735 s and 19.664 s, respectively. This reinforces the short result:
duplicated score decoding is the first target for making the second result
cheaper.

| Nested embedding work | Full video | Per sampled image |
| --- | ---: | ---: |
| Shared image preparation | 38.707 s | 18.23 ms |
| Recall encoder execution | 28.582 s | 13.46 ms |
| F1 encoder execution | 44.021 s | 20.74 ms |
| Both encoders' output readback and token writing | 9.123 s | 4.30 ms |
| Recall temporal head | 1.648 s | — |
| F1 temporal head | 1.536 s | — |

There were 2,123 prepared images, with only **0.202 s** of queue backpressure
and 0.007 s waiting for the encoder worker at the end. These encoder/preparation
costs overlap within the shared video stage; the table is not an additive
end-to-end decomposition. The initial timestamp inventory took 78.170 s and
OpenCV AV-feature calls accumulated 317.901 s in that stage. Codec input queue
calls accumulated 532.337 s, but include waits and overlap other workers; that
counter is not pure decoder compute time.

Saving both projects took 104.613 ms. Six subsequent project reads and editor
seed constructions had median **43.154 ms**, maximum **48.732 ms**. These are
local result-loading measurements, excluding UI rendering, video opening and
playback seeking.

Full-video recall has **37** proposals and F1 **34**. Both boundary lists and
selected embedding timestamps exactly match the earlier frozen full-video
outputs. Recall confidence also matches exactly. F1's maximum confidence
difference is 0.000060 against its earlier physical-device reference, without
any changed boundaries. This full check is a regression check, not a new
accuracy ranking or a same-device full embedding comparison. The stricter
short comparison uses the same APK/device and is exact for all tested tensors,
features, boundaries, confidences and specialist decisions.

The pair's full-video embedding files total **65,218,560 bytes**, versus
32,609,280 bytes for one. Shared AV104 and contextual-520 float payloads are
1,765,920 and 8,829,600 bytes, respectively; these are calculated tensor payloads,
not total cache-directory sizes. Temporary embeddings are removed after use.

## Optimization priorities

1. Plan the union of both variants' score-frame requests, decode those frames
   once, and evaluate each variant's specialists using its original boundaries
   and frame order. Keep candidate-specific normalization/context separate;
   reusing pixels must not merge the two models' candidate populations.
   The duplicated score decode is the largest measured added
   cost; confirm exact feature and decision parity before adopting this.
2. Give paired analyses with an existing AV cache one shared embedding-only
   decode. The present fallback traverses the video once per encoder.
3. Profile timestamp scanning, decoding and frame conversion independently.
   The independent embedding pass spends very little of its time executing
   the neural network in this emulator setup.
4. Investigate encoder scheduling only after those larger costs. F1 encoder
   execution was slower in the paired run than in its independent pass, but
   host scheduling, memory/cache effects and session contention have not been
   isolated. Queue backpressure was only 23 ms in the short paired run.

Both model bundles were already packaged, so this experiment adds no model
weight download. Four prepared-image buffers still use 2,408,448 bytes total;
the pair does not retain the whole video's images. Each variant writes its own
temporary FP32 token file: 3,686,400 bytes for the short excerpt. Saved projects
retain predictions and score evidence, not embeddings. Existing companion edits
are preserved rather than replaced by another paired analysis.

The primary project's saved timing covers both results becoming ready. The
companion timing is explicitly labeled as its rally/score work with shared
preparation excluded; it is not a standalone end-to-end processing measurement.
Memory snapshots in the JSON are taken at primary completion. They are not
peak-memory measurements or per-encoder allocations. Physical-device latency,
thermal behavior, battery cost, and tap-to-render switching remain unmeasured
for this paired experiment.

## Reproducibility and validation

The [JSON report](paired-distilled-variants.json) contains the measured APK hash,
indexed input scope, exact counters, and qualification checks. That APK and its
source snapshot are archived under the external artifact index. After timing,
only progress/timing presentation was corrected: the primary saved total now
includes companion scoring, companion costs have an explicit label, and the
progress step covers both score passes. The measured inference, image
preparation, model weights, and decoder calculations were not changed.

Final debug and instrumentation APK builds passed, together with **208 JVM
tests**. The final APK passed both synthetic paired integration tests; the
external-video test was intentionally skipped in that synthetic-only invocation.
All four measured indexed-video runs passed their instrumentation checks. The
publication audit found zero working-tree and historical-blob findings. No web
runtime, trained artifact, release signing, deployment, or physical-device run
was part of this experiment.
